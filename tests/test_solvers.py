"""The two solvers that had to earn their exactness rather than inherit it.

The first three tasks are discrete and finite, so backward induction over them
needs no argument. The battery has a real-valued charge and the supply chain has
five observed numbers; both are solved exactly anyway, and each rests on a claim
that is checkable here rather than only in a docstring.
"""

from __future__ import annotations

import numpy as np
import pytest
from decisionrl.envs import EnergyMicrogrid

import stadion
from stadion.solvers.serial import solve_supply_chain
from stadion.solvers.storage import expected_exogenous, solve_energy
from stadion.tasks.energy import LEVELS


def test_the_battery_profile_means_match_the_environment_that_draws_them() -> None:
    """The recurrence replaces three random draws with their means.

    Generation is clipped at zero *after* its noise is added, so its mean is not
    its profile — using the profile would understate it at dawn and dusk, which
    is where charging decisions are tightest.
    """
    horizon = 48
    price, generation, load = expected_exogenous(horizon)

    env = EnergyMicrogrid(horizon=horizon)
    env.reset(seed=0)
    samples: list[list[tuple[float, float, float]]] = [[] for _ in range(horizon)]
    for episode in range(400):
        env.reset(seed=episode)
        for t in range(horizon):
            _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
            samples[t].append((info["price"], info["generation"], info["load"]))

    for t in (0, horizon // 4, horizon // 2, 3 * horizon // 4, horizon - 1):
        drawn = np.array(samples[t])
        error = drawn.std(axis=0, ddof=1) / np.sqrt(drawn.shape[0])
        expected = np.array([price[t], generation[t], load[t]])
        deviation = np.abs(drawn.mean(axis=0) - expected) / np.maximum(error, 1e-12)
        assert (deviation < 4.0).all(), (
            f"step {t}: expected {expected}, environment drew {drawn.mean(axis=0)} "
            f"+- {error}"
        )


def test_a_battery_needing_an_unaffordable_lattice_is_refused_not_approximated() -> None:
    """Falling back to a grid would make the ceiling approximate without saying so."""
    with pytest.raises(ValueError, match="lattice"):
        solve_energy(
            capacity=10.0,
            max_power=3.0,
            efficiency=0.9137,  # needs billions of points
            horizon=48,
            levels=LEVELS,
        )


def test_the_battery_lattice_stays_affordable_for_every_instance_the_task_draws() -> None:
    task = stadion.get("energy")
    for seed in range(30):
        solution = task._solve(task.instance(seed))
        assert solution.lattice <= 20_000


def test_the_supply_chain_optimum_does_not_move_when_the_state_cap_is_raised() -> None:
    """The one approximation in the serial solver, checked instead of asserted.

    Stock is tracked up to a cap. If the optimal policy ever wanted to hold more
    than that, the cap would be shaping the answer; raising it and finding the
    value unchanged is what rules that out.
    """
    common = {
        "demand_mean": 6.0,
        "levels": np.array([0, 3, 6, 9, 12]),
        "holding_cost_retail": 0.08,
        "holding_cost_warehouse": 0.03,
        "stockout_penalty": 0.6,
        "horizon": 40,
    }
    tight = solve_supply_chain(cap=40, **common)  # type: ignore[arg-type]
    loose = solve_supply_chain(cap=64, **common)  # type: ignore[arg-type]
    assert tight.value == pytest.approx(loose.value, rel=1e-9)


def test_the_supply_chain_optimum_keeps_stock_far_below_the_cap() -> None:
    """The other half of the same claim, measured on played episodes."""
    task = stadion.get("supply-chain")
    inst = task.instance(0)
    optimal = task.optimal(inst)
    peak = 0.0
    for seed in range(20):
        env = inst.env()
        obs, _ = env.reset(seed=seed)
        for step in range(int(inst.params["horizon"])):
            view = task.view(env, obs, step, 0.0)
            obs, _, terminated, truncated, _ = env.step(
                view.choice(optimal.act(view)).act
            )
            peak = max(peak, float(obs[0] + obs[2]), float(obs[1] + obs[3]))
            if terminated or truncated:
                break
    # Observations are scaled by the cap, so 1.0 is the cap itself.
    assert peak < 0.75, f"the optimal policy reached {peak:.2f} of the state cap"
