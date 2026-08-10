"""Running an evaluation: three arms, one set of instances, one interval.

For each instance the agent, the classical baseline and the optimal policy all
play the same episode seeds on the same parameters. That pairing is what makes
the bootstrap tight enough to return "no measurable difference" as a real
answer rather than as a shrug.

One caveat is stated here rather than buried, because it bounds what the
pairing buys. The environments draw a variable number of random numbers per
step — NumPy's Poisson sampler consumes a different amount of the stream
depending on its mean — so once two policies choose differently, their demand
paths diverge even from an identical seed. Pairing therefore removes
between-instance variance, which is the large term, but not within-episode
noise. That is why each instance is averaged over several episodes before the
arms are compared, and why ``episodes`` is a knob rather than a constant.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from stadion.core.agent import Agent
from stadion.core.score import Comparison, Report, paired_bootstrap
from stadion.core.task import Brief, Instance, Task

__all__ = ["Arms", "evaluate", "play_episode", "play_instance", "reproducible", "tune"]

#: The classical rules differ in what they have to tune: one threshold for the
#: battery, one base-stock target for the chain, a (price, target) pair when the
#: two decisions are coupled.
P = TypeVar("P")


def play_episode(
    task: Task, agent: Agent, inst: Instance, seed: int, brief: Brief | None = None
) -> float:
    """One episode. Returns the undiscounted total reward.

    ``brief`` is accepted so a caller running many episodes on one instance can
    build it once; it is the same object either way.
    """
    env = inst.env()
    agent.start(task.brief(inst) if brief is None else brief)
    obs, _ = env.reset(seed=seed)
    earned, step, done = 0.0, 0, False
    while not done:
        view = task.view(env, obs, step, earned)
        chosen = agent.act(view)
        # Environments clip out-of-range actions silently; refusing them here
        # turns a broken agent into an error instead of a quiet bad score.
        obs, reward, terminated, truncated, _ = env.step(view.choice(chosen).act)
        earned += float(reward)
        step += 1
        done = terminated or truncated
    return earned


def play_instance(
    task: Task, agent: Agent, inst: Instance, episode_seeds: Sequence[int]
) -> float:
    """Mean return of one agent on one instance."""
    brief = task.brief(inst)
    return float(np.mean([play_episode(task, agent, inst, s, brief) for s in episode_seeds]))


def tune(
    task: Task,
    inst: Instance,
    build: Callable[[P], Agent],
    candidates: Iterable[P],
) -> P:
    """Pick the classical rule's free parameter on seeds held out of the evaluation.

    The search evaluates the rule exactly as it will be played — including the
    snap to the task's action menu, where there is one. Tuning on a continuous
    relaxation and then playing the discretised version would hand the agent a
    baseline that was never optimised for the game either of them is in.
    """
    seeds = tuple(task.tuning_seed + j for j in range(task.tuning_episodes))
    best_param: P | None = None
    best_return = -np.inf
    for param in candidates:
        earned = play_instance(task, build(param), inst, seeds)
        if earned > best_return:
            best_param, best_return = param, earned
    if best_param is None:
        raise ValueError("tune() needs at least one candidate parameter")
    return best_param


@dataclass(frozen=True, slots=True)
class Arms:
    """Per-instance mean returns for the three players."""

    seeds: tuple[int, ...]
    agent: np.ndarray
    baseline: np.ndarray
    optimal: np.ndarray


def evaluate(
    task: Task,
    agent: Agent | Callable[[], Agent],
    *,
    instances: int = 30,
    episodes: int = 20,
    instance_seed: int = 0,
    episode_seed: int = 0,
    resamples: int = 10_000,
    level: float = 0.95,
    bootstrap_seed: int = 0,
    workers: int = 1,
) -> Report:
    """Score ``agent`` on ``task`` against the classical method and the optimum.

    ``workers`` runs instances concurrently. It exists for agents that wait on a
    network: a decision cannot start until the previous one's outcome is known,
    so an episode is a chain of round-trips and a language model spends the
    evaluation waiting rather than computing. Instances do not depend on each
    other, so they overlap freely, and the result is identical either way —
    every seed is fixed in advance and the rows are collected by index.

    Above one worker, ``agent`` must be a factory rather than an instance: an
    agent that remembers anything within an episode — as one driving a language
    model must — cannot be shared across threads without its memory interleaving.
    """
    if instances < 2:
        raise ValueError(f"need at least 2 instances for an interval, got {instances}")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    episode_seeds = tuple(episode_seed + j for j in range(episodes))

    if isinstance(agent, Agent):
        if workers > 1:
            raise ValueError(
                "workers > 1 needs a factory, not an agent instance.\n"
                "  An agent that keeps state within an episode would have that state "
                "interleaved across threads.\n"
                "  Pass a zero-argument callable that returns a fresh agent, e.g. "
                "evaluate(task, lambda: MyAgent(), workers=8)."
            )
        shared = agent

        def build() -> Agent:
            return shared

        name = agent.name
    else:
        build = agent
        name = build().name

    def one(index: int) -> tuple[float, float, float]:
        inst = task.instance(instance_seed + index)
        return (
            play_instance(task, build(), inst, episode_seeds),
            play_instance(task, task.baseline(inst), inst, episode_seeds),
            play_instance(task, task.optimal(inst), inst, episode_seeds),
        )

    if workers == 1:
        rows = [one(i) for i in range(instances)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(one, range(instances)))

    arms = Arms(
        seeds=tuple(instance_seed + i for i in range(instances)),
        agent=np.asarray([r[0] for r in rows]),
        baseline=np.asarray([r[1] for r in rows]),
        optimal=np.asarray([r[2] for r in rows]),
    )

    def compare(x: np.ndarray, y: np.ndarray, label: str) -> Comparison:
        return paired_bootstrap(
            x, y, label=label, resamples=resamples, level=level, seed=bootstrap_seed
        )

    return Report(
        task=task.name,
        agent=name,
        instances=instances,
        episode_seeds=episode_seeds,
        agent_return=float(arms.agent.mean()),
        baseline_return=float(arms.baseline.mean()),
        optimal_return=float(arms.optimal.mean()),
        vs_baseline=compare(arms.agent, arms.baseline, "agent - classical"),
        vs_optimal=compare(arms.agent, arms.optimal, "agent - optimum"),
        headroom=compare(arms.optimal, arms.baseline, "optimum - classical"),
    )


def reproducible(task: Task, agent: Agent, *, instances: int = 3, episodes: int = 3) -> bool:
    """Play the same instances twice and report whether the returns are identical.

    A benchmark that cannot reproduce its own numbers cannot pin anyone else's.
    Agents that consult a live model will fail this by construction, which is
    the honest outcome: their scores carry run-to-run variance and the report
    should not pretend otherwise.
    """
    seeds = tuple(range(episodes))
    first = [
        play_instance(task, agent, task.instance(i), seeds) for i in range(instances)
    ]
    second = [
        play_instance(task, agent, task.instance(i), seeds) for i in range(instances)
    ]
    return all(a == b for a, b in zip(first, second, strict=True))
