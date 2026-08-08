"""What plays a task.

One interface covers everything that makes decisions here: the agent under
test, the classical operations-research rule it is measured against, and the
optimal policy that sets the ceiling. They differ only in what part of the
:class:`~stadion.core.task.View` they read — a language model reads ``text``
and ``choices``, a numeric policy reads ``obs`` and ``env`` — so the runner has
no branches and no arm gets a code path of its own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from stadion.core.task import Brief, Task, View

__all__ = ["Agent", "NearestChoiceAgent", "PolicyAgent", "RandomAgent", "ReferenceAgent"]


class Agent(ABC):
    """Something that turns a view of the problem into a legal choice."""

    #: Shown in reports. Set it to something a reader will recognise six months on.
    name: str = "agent"

    def start(self, brief: Brief) -> None:  # noqa: B027 - an optional hook, not a contract
        """Called once before each episode, with the instance's rules and numbers.

        Deliberately not abstract: a stateless policy has nothing to set up, and
        forcing it to write an empty override would add a line that says nothing.
        """

    @abstractmethod
    def act(self, view: View) -> int:
        """Return the value of one of ``view.choices``."""


class PolicyAgent(Agent):
    """Adapts a ``(env, obs) -> action`` policy — decisionrl's shape — to an agent.

    This is how the classical baselines cross over: ``decisionrl.baselines``
    exports its reference rules in exactly that signature, so they are used
    here unmodified rather than reimplemented, and the comparison is against
    the same code that library ships and tests.
    """

    def __init__(self, policy: Callable[[Any, np.ndarray], Any], name: str) -> None:
        self._policy = policy
        self.name = name

    def act(self, view: View) -> int:
        return int(self._policy(view.env, view.obs))


class NearestChoiceAgent(Agent):
    """Adapts a policy with a continuous action to the task's menu.

    The classical rules for the battery and the supply chain emit a real-valued
    action, while every player here picks from the same discrete menu. Snapping
    the rule's action to the nearest option is what makes the two comparable —
    and because the rule's free parameter is tuned *through* this wrapper, it is
    tuned on the task as actually played, not on a continuous relaxation of it.
    """

    def __init__(self, policy: Callable[[Any, np.ndarray], Any], name: str) -> None:
        self._policy = policy
        self.name = name
        self._menu_id: int | None = None
        self._menu: np.ndarray = np.zeros((0, 0))
        self._values: np.ndarray = np.zeros(0, dtype=np.int64)

    def act(self, view: View) -> int:
        # Tasks hand back the same menu object every step, so the action matrix
        # is stacked once per instance rather than per decision.
        if id(view.choices) != self._menu_id:
            self._menu = np.stack(
                [np.asarray(c.act, dtype=float).reshape(-1) for c in view.choices]
            )
            self._values = np.array([c.value for c in view.choices], dtype=np.int64)
            self._menu_id = id(view.choices)

        raw = np.asarray(self._policy(view.env, view.obs), dtype=float).reshape(-1)
        nearest = int(np.argmin(((self._menu - raw) ** 2).sum(axis=1)))
        return int(self._values[nearest])


class RandomAgent(Agent):
    """Uniform over the legal choices. The floor any real agent must clear.

    Worth running before anything expensive: on these tasks a random player
    scores well below the classical rule, and if a candidate agent lands near
    this line the wiring is wrong, not the model.
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def start(self, brief: Brief) -> None:
        # Reseeded per episode from the instance so a random run is reproducible.
        self._rng = np.random.default_rng([self._seed, brief.seed])

    def act(self, view: View) -> int:
        return int(self._rng.choice([c.value for c in view.choices]))


class ReferenceAgent(Agent):
    """Plays one of the task's own reference policies, rebuilt for each instance.

    Useful as a control: running ``ReferenceAgent(task, "classical")`` through
    the ordinary evaluation path must come out indistinguishable from the
    baseline arm, and ``"optimum"`` must score close to 1. If either drifts, the
    harness is measuring something other than what it claims to.
    """

    def __init__(self, task: Task, which: str = "classical") -> None:
        if which not in {"classical", "optimum"}:
            raise ValueError(f"which must be 'classical' or 'optimum', got {which!r}")
        self._task = task
        self._which = which
        self._inner: Agent | None = None
        self.name = which

    def start(self, brief: Brief) -> None:
        inst = self._task.instance(brief.seed)
        self._inner = (
            self._task.baseline(inst) if self._which == "classical" else self._task.optimal(inst)
        )
        self._inner.start(brief)

    def act(self, view: View) -> int:
        if self._inner is None:  # pragma: no cover - start() precedes act() in the runner
            raise RuntimeError("ReferenceAgent.start() must run before act(); no instance bound")
        return self._inner.act(view)
