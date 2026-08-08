"""The dynamic programs are the only thing here that nothing else can check.

Every other number in the library is measured *against* the optimum, so a wrong
recurrence would shift the whole scale silently and no run would complain. The
guard is that each solver produces a value and a policy by different routes:
the value comes out of backward induction, the policy is then simulated. They
have to meet.
"""

from __future__ import annotations

import numpy as np
import pytest

import stadion
from stadion.core.runner import play_episode
from stadion.solvers.dp import poisson_pmf

EPISODES = 1500
MAX_Z = 4.0


@pytest.mark.parametrize("task_name", stadion.names())
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dynamic_program_value_matches_the_policy_it_emits(task_name: str, seed: int) -> None:
    task = stadion.get(task_name)
    inst = task.instance(seed)
    analytic = task.optimal_value(inst)

    # Each instance gets its own block of episode seeds: sharing one block across
    # instances correlates their sampling error, which makes a run of same-signed
    # deviations look systematic when it is only one shared draw.
    base = 1_000_000 * (seed + 1)
    returns = np.array(
        [play_episode(task, task.optimal(inst), inst, base + j) for j in range(EPISODES)]
    )
    error = returns.std(ddof=1) / np.sqrt(returns.size)
    z = (returns.mean() - analytic) / error

    assert abs(z) <= MAX_Z, (
        f"{task_name} seed {seed}: backward induction says {analytic:.3f} but simulating "
        f"its own policy gives {returns.mean():.3f} +- {error:.3f} (z = {z:+.2f}). "
        f"The recurrence and the policy disagree."
    )


def test_the_classical_method_never_beats_the_optimum() -> None:
    """A ceiling that can be exceeded is not a ceiling."""
    for task_name in stadion.names():
        task = stadion.get(task_name)
        for seed in range(4):
            inst = task.instance(seed)
            seeds = tuple(range(40))
            classical = stadion.play_instance(task, task.baseline(inst), inst, seeds)
            optimal = stadion.play_instance(task, task.optimal(inst), inst, seeds)
            # Sampling noise can put a single instance marginally the wrong way;
            # a real inversion is much larger than the return's own scale.
            assert optimal >= classical - 0.05 * abs(classical), (
                f"{task_name} seed {seed}: the classical method scored {classical:.3f} "
                f"against an optimum of {optimal:.3f}"
            )


def test_poisson_pmf_is_a_distribution_with_the_requested_mean() -> None:
    for lam in (0.05, 1.0, 5.3, 20.0):
        pmf = poisson_pmf(lam)
        k = np.arange(pmf.size)
        assert pmf.sum() == pytest.approx(1.0)
        assert (pmf >= 0).all()
        assert float((pmf * k).sum()) == pytest.approx(lam, rel=1e-6)
        assert float((pmf * (k - lam) ** 2).sum()) == pytest.approx(lam, rel=1e-5)


def test_poisson_pmf_refuses_a_non_positive_mean() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        poisson_pmf(0.0)
