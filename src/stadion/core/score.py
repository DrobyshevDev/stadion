"""Scoring: a paired comparison with an interval, not a number.

Agent, classical baseline and optimum are run on the *same* instances with the
*same* episode seeds, so the three returns for one instance see identical demand
draws, identical arrivals, identical everything. That makes the comparison
paired, and a paired bootstrap over instances is far tighter than comparing two
independent means — which matters, because the honest answer on these problems
is often "no measurable difference", and a loose interval cannot say that.

The headline number is normalised:

    score = (agent - baseline) / (optimum - baseline)

0 means the agent matched the classical method, 1 means it reached the optimum,
negative means it lost. When the classical method is *already* optimal the
denominator collapses and the score is undefined — that is reported as such
rather than papered over with a small epsilon, because on some of these problems
the textbook answer is provably the best one and a benchmark that hides this is
selling a race that cannot be won.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = ["Comparison", "Interval", "Report", "Verdict", "paired_bootstrap"]

Verdict = Literal["better", "worse", "indistinguishable"]


@dataclass(frozen=True, slots=True)
class Interval:
    low: float
    high: float
    level: float = 0.95

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2.0

    def __str__(self) -> str:
        return f"[{self.low:+.3f}, {self.high:+.3f}]"


@dataclass(frozen=True, slots=True)
class Comparison:
    """A paired difference between two policies, with the interval that decides it."""

    label: str
    mean_a: float
    mean_b: float
    delta: float
    ci: Interval
    verdict: Verdict
    n: int

    @property
    def relative(self) -> float:
        """The difference as a fraction of the reference return."""
        scale = abs(self.mean_b)
        return self.delta / scale if scale > 0 else float("nan")

    def __str__(self) -> str:
        return (
            f"{self.label}: {self.delta:+.3f} {self.ci} "
            f"({self.relative:+.1%}) -> {self.verdict}"
        )


def paired_bootstrap(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    label: str = "a - b",
    resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Comparison:
    """Bootstrap the mean paired difference ``a - b`` over instances.

    Instances are resampled with replacement; both arms move together because
    they are the same instance, which is the point of pairing. ``seed`` fixes
    the resampling, so the interval is reproducible run to run.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"arms have different lengths: {arr_a.shape} vs {arr_b.shape}")
    n = int(arr_a.size)
    if n < 2:
        raise ValueError(f"need at least 2 instances to bootstrap an interval, got {n}")

    diff = arr_a - arr_b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    means = diff[idx].mean(axis=1)

    alpha = (1.0 - level) / 2.0
    low = float(np.quantile(means, alpha))
    high = float(np.quantile(means, 1.0 - alpha))
    ci = Interval(low, high, level)

    if low > 0.0:
        verdict: Verdict = "better"
    elif high < 0.0:
        verdict = "worse"
    else:
        verdict = "indistinguishable"

    return Comparison(
        label=label,
        mean_a=float(arr_a.mean()),
        mean_b=float(arr_b.mean()),
        delta=float(diff.mean()),
        ci=ci,
        verdict=verdict,
        n=n,
    )


@dataclass(frozen=True, slots=True)
class Report:
    """The result of evaluating one agent on one task."""

    task: str
    agent: str
    instances: int
    episode_seeds: tuple[int, ...]
    agent_return: float
    baseline_return: float
    optimal_return: float
    vs_baseline: Comparison
    vs_optimal: Comparison
    headroom: Comparison

    @property
    def degenerate(self) -> bool:
        """True when the classical method cannot be told apart from the optimum.

        On such an instance family the normalised score has no denominator: the
        best available outcome is a draw with the textbook rule.
        """
        return self.headroom.verdict == "indistinguishable"

    @property
    def score(self) -> float | None:
        """Normalised score, or ``None`` when there is no headroom to normalise by."""
        if self.degenerate:
            return None
        span = self.optimal_return - self.baseline_return
        return (self.agent_return - self.baseline_return) / span

    @property
    def point_value(self) -> float:
        """Return earned by one point of normalised score.

        Reported alongside the score because the normalisation divides by the
        headroom, and where the classical method is nearly optimal that divisor
        is small — which turns an ordinary loss into a spectacular-looking
        negative score. Saying what a point is worth keeps the scale readable.
        """
        return self.optimal_return - self.baseline_return

    def summary(self) -> str:
        head = (
            f"{self.task} / {self.agent}\n"
            f"  instances        {self.instances}\n"
            f"  agent            {self.agent_return:9.3f}\n"
            f"  classical        {self.baseline_return:9.3f}\n"
            f"  optimum          {self.optimal_return:9.3f}\n"
        )
        if self.degenerate:
            verdict = (
                "  score            undefined — the classical method is already\n"
                "                   indistinguishable from the optimum here, so a\n"
                "                   draw is the ceiling.\n"
            )
        else:
            scale = abs(self.baseline_return)
            share = self.point_value / scale if scale > 0 else float("nan")
            verdict = (
                f"  score            {self.score:9.3f}   (0 = classical, 1 = optimum;\n"
                f"                             one point = {self.point_value:.3f}, "
                f"{share:.1%} of the classical result)\n"
            )
        return head + verdict + (
            f"  {self.vs_baseline}\n"
            f"  {self.vs_optimal}\n"
            f"  {self.headroom}\n"
        )
