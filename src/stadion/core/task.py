"""The task protocol: an operational decision problem with a known right answer.

A :class:`Task` is a *family* of problems, not one problem. Calling
:meth:`Task.instance` with a seed draws concrete parameters — demand level, cost
structure, horizon — so the specific numbers an agent is asked about were
generated at evaluation time and cannot have been memorised from a public
dataset.

Every task carries two reference players beside the environment:

``baseline``
    The classical operations-research method for this problem, configured the
    way a practitioner would configure it. Where a closed form exists (the
    newsvendor critical fractile) it is used; where it does not, the rule's one
    free parameter is tuned by search on *tuning* seeds that never appear in
    the evaluation set.

``optimal``
    The exact optimum, obtained by backward induction over the full state
    space. This is a ceiling, not a competitor: it is available only because
    these problems are small enough to solve outright, which is the whole
    reason they were chosen.

Both are ordinary :class:`~stadion.core.agent.Agent` objects, the same type the
agent under test is, so all three arms run through one code path.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from decisionrl.core.env import Env

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from stadion.core.agent import Agent

__all__ = ["Brief", "Choice", "Instance", "Task", "View", "get", "names", "registry"]


def _salt(name: str) -> int:
    """A stable per-task seed offset.

    ``hash()`` is randomised per process, so using it here would make instance
    parameters differ between runs of the same command. BLAKE2b does not.
    """
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "big")


@dataclass(frozen=True, slots=True)
class Choice:
    """One legal action, in the words the agent reads.

    ``value`` is what an agent returns and what a report refers to. ``action`` is
    what reaches ``env.step``, and defaults to ``value`` itself — which is the
    whole story for a discrete environment, where the choice *is* the action.
    Environments with a continuous action space carry the vector here instead,
    so the menu an agent reads stays a short list of numbered options no matter
    what shape the underlying action has.
    """

    value: int
    label: str
    detail: str = ""
    action: Any = None

    @property
    def act(self) -> Any:
        """What to pass to ``env.step``."""
        return self.value if self.action is None else self.action


@dataclass(frozen=True, slots=True)
class Brief:
    """What is known before the first decision: the rules and the numbers.

    The brief deliberately states the instance's parameters in full. The
    classical baseline is built from those same parameters, so withholding them
    from the agent would not make the comparison harder, it would make it
    dishonest — the agent would be guessing at a cost structure its opponent was
    handed. What the agent does *not* get is the baseline's tuning budget: the
    classical rule is fitted by search over practice episodes, the agent gets
    one look at the brief and then plays.
    """

    task: str
    seed: int
    horizon: int
    text: str
    params: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class View:
    """The state at one decision point, in both the numeric and the written form.

    ``text`` and ``choices`` are what a language model reads; ``obs`` and ``env``
    are what a numeric policy reads. Both are always present, so an RL policy
    and an LLM can be scored on the same task without either being translated
    through the other's interface.
    """

    step: int
    text: str
    choices: tuple[Choice, ...]
    earned: float
    obs: np.ndarray
    env: Env

    def choice(self, value: int) -> Choice:
        for c in self.choices:
            if c.value == value:
                return c
        legal = ", ".join(str(c.value) for c in self.choices)
        raise ValueError(f"{value} is not a legal choice at step {self.step}; legal values: {legal}")


@dataclass(frozen=True, slots=True)
class Instance:
    """One concrete problem drawn from a task family."""

    task: Task
    seed: int
    params: Mapping[str, float] = field(compare=False)

    def env(self) -> Env:
        return self.task.build(self.params)


class Task(ABC):
    """A family of operational decision problems with a classical baseline."""

    name: ClassVar[str]
    summary: ClassVar[str]
    baseline_name: ClassVar[str]
    #: Episodes used to tune the baseline's free parameter, on seeds held apart
    #: from the evaluation set so the classical method is never fitted to the
    #: episodes it is scored on.
    tuning_episodes: ClassVar[int] = 40
    #: Seed offset for those tuning episodes. Far from any evaluation seed.
    tuning_seed: ClassVar[int] = 1_000_003

    def __init__(self) -> None:
        # Dynamic programs and baseline tuning are per-instance and pure, so
        # they are solved once per seed and kept. Tasks are singletons in the
        # registry, so this cache is bounded by the seeds an evaluation touches.
        self._memo: dict[tuple[str, int], Any] = {}

    # -- the problem family ------------------------------------------------ #

    @abstractmethod
    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        """Draw one instance's parameters."""

    @abstractmethod
    def build(self, params: Mapping[str, float]) -> Env:
        """Construct the environment for a parameter set."""

    def instance(self, seed: int) -> Instance:
        rng = np.random.default_rng([_salt(self.name), seed])
        return Instance(task=self, seed=seed, params=self.sample_params(rng))

    # -- the surface an agent reads ---------------------------------------- #

    @abstractmethod
    def brief(self, inst: Instance) -> Brief:
        """Describe the instance before the episode starts."""

    @abstractmethod
    def view(self, env: Env, obs: np.ndarray, step: int, earned: float) -> View:
        """Describe the current state and the legal moves."""

    def tool_schema(self, inst: Instance) -> dict[str, Any]:
        """The action interface as a JSON-schema tool definition.

        Provided so an LLM harness can bind the same action space it would get
        through :meth:`view`, without re-deriving it from prose.
        """
        env = inst.env()
        obs, _ = env.reset(seed=0)
        choices = self.view(env, obs, 0, 0.0).choices
        return {
            "name": "decide",
            "description": self.summary,
            "input_schema": {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "integer",
                        "enum": [c.value for c in choices],
                        "description": "; ".join(f"{c.value} = {c.label}" for c in choices),
                    }
                },
                "required": ["choice"],
            },
        }

    # -- the two reference players ----------------------------------------- #

    @abstractmethod
    def baseline(self, inst: Instance) -> Agent:
        """The classical method, configured for this instance."""

    @abstractmethod
    def optimal(self, inst: Instance) -> Agent:
        """The exact optimum for this instance, by backward induction."""

    @abstractmethod
    def optimal_value(self, inst: Instance) -> float:
        """The analytic expected return of :meth:`optimal`.

        Kept separate from the policy so the two can be checked against each
        other: simulating the optimal player must reproduce this number within
        Monte Carlo error, and ``tests/test_dp.py`` asserts exactly that. A
        dynamic program that silently disagrees with the policy it emits is the
        failure mode this guards.
        """


def registry() -> Mapping[str, Task]:
    """All built-in tasks, by name."""
    from stadion.tasks import all_tasks

    return {t.name: t for t in all_tasks()}


def get(name: str) -> Task:
    tasks = registry()
    try:
        return tasks[name]
    except KeyError:
        known = ", ".join(sorted(tasks))
        raise KeyError(f"unknown task {name!r}; available tasks are: {known}") from None


def names() -> Sequence[str]:
    return sorted(registry())
