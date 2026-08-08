"""Exact backward induction for the battery, on the lattice the battery lives on.

The microgrid looks like it needs an approximation: the state of charge is a
real number, so the usual move is a grid plus interpolation, and the "optimum"
that comes out is only optimal for the grid. Two facts remove the need.

First, the expected one-step reward is linear in the power drawn and the charge
transition is deterministic. Electricity price is drawn independently of
generation and load, so the expectation of their product factorises, and the
noise on all three is additive with a known mean. Nothing about the reward
depends on the realised noise beyond its mean, and the observation the agent
receives is an independent draw from the same distribution — it carries no
information about the noise that will actually be charged. So the state of
charge and the clock are a sufficient statistic, and a dynamic program over them
is exactly optimal rather than approximately so.

Second, the state of charge does not wander over the reals. Charging moves it by
``power * efficiency`` and discharging by ``power / efficiency``. Both are
rational for any parameters anyone would write down, so they share a common
measure and every reachable charge is a whole number of those units; this module
works in those units throughout. Clipping at an empty or full battery lands on
the lattice too, because zero and the capacity are multiples of the quantum by
construction.

What varies is how *fine* the lattice has to be, and that is sensitive to the
efficiency: 0.8 needs 801 points where 0.95 needs 15,201 and 0.9137 would need
three billion. Nothing is approximated in either case — the coarse ones are just
cheaper. A parameter set whose lattice exceeds :data:`MAX_LATTICE` is refused
outright rather than quietly downgraded to a grid with interpolation, because a
ceiling that is optimal-for-a-grid is not the ceiling this library promises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

__all__ = ["EnergySolution", "expected_exogenous", "solve_energy"]

#: Refuse to build a lattice larger than this. A parameter set that needs more
#: is almost certainly incommensurate by accident rather than by design.
MAX_LATTICE = 200_000


def _standard_normal_mean_above_zero(mu: float, sigma: float) -> float:
    """E[max(0, mu + sigma * Z)] for standard normal Z, in closed form.

    Generation is a clipped normal, so its mean is not its location parameter.
    Using the location parameter instead understates a solar profile near dawn
    and dusk, which is exactly where the battery decision is delicate.
    """
    if sigma <= 0:
        return max(0.0, mu)
    z = mu / sigma
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return mu * cdf + sigma * pdf


def expected_exogenous(horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean price, generation and load per step, matching ``EnergyMicrogrid``.

    The environment draws these from fixed profiles with additive noise. Price
    and load take their noise around the profile, so their means are the profile
    itself; generation is clipped at zero after the noise is added, so its mean
    is the clipped-normal mean.
    """
    t = np.arange(horizon)
    phase = 2.0 * np.pi * t / horizon
    price = 0.6 + 0.4 * np.sin(phase - np.pi / 2.0)
    load = 1.5 + 0.8 * np.maximum(0.0, np.sin(phase - np.pi / 2.0))
    solar = 3.0 * np.sin(np.pi * t / horizon)
    generation = np.array(
        [_standard_normal_mean_above_zero(float(m), 0.1) for m in solar]
    )
    return price, generation, load


def _quantum(capacity: float, max_power: float, efficiency: float, levels: np.ndarray) -> Fraction:
    """The largest step size that divides every charge movement and the capacity."""
    cap = Fraction(capacity).limit_denominator(10_000)
    power = Fraction(max_power).limit_denominator(10_000)
    eff = Fraction(efficiency).limit_denominator(10_000)

    steps = [cap]
    for level in levels:
        a = Fraction(float(level)).limit_denominator(10_000)
        if a > 0:
            steps.append(a * power * eff)
        elif a < 0:
            steps.append(-a * power / eff)

    quantum = steps[0]
    for step in steps[1:]:
        if step == 0:
            continue
        # gcd of two rationals: gcd of numerators over lcm of denominators.
        num = math.gcd(quantum.numerator, step.numerator)
        den = quantum.denominator * step.denominator // math.gcd(
            quantum.denominator, step.denominator
        )
        quantum = Fraction(num, den)
    return quantum


@dataclass(frozen=True, slots=True)
class EnergySolution:
    """Optimal battery action index per (step, state of charge)."""

    value: float
    actions: np.ndarray  # shape (horizon, lattice size)
    quantum: float
    lattice: int

    def action_index(self, step: int, state_of_charge: float) -> int:
        t = min(step, self.actions.shape[0] - 1)
        units = int(round(state_of_charge / self.quantum))
        units = int(np.clip(units, 0, self.actions.shape[1] - 1))
        return int(self.actions[t, units])


def solve_energy(
    *,
    capacity: float,
    max_power: float,
    efficiency: float,
    horizon: int,
    levels: np.ndarray,
) -> EnergySolution:
    """Backward induction over the charge lattice.

    ``levels`` are the menu's actions in [-1, 1], the same list the agent picks
    from, because an optimum over a richer action set than the players are given
    would not be a ceiling they could reach.
    """
    quantum = _quantum(capacity, max_power, efficiency, levels)
    size = int(Fraction(capacity).limit_denominator(10_000) / quantum) + 1
    if size > MAX_LATTICE:
        raise ValueError(
            f"the charge lattice would need {size} points, above the {MAX_LATTICE} limit.\n"
            f"  capacity={capacity}, max_power={max_power}, efficiency={efficiency}\n"
            f"Charging moves the battery by power*efficiency and discharging by "
            f"power/efficiency, and the lattice is as fine as those two together "
            f"require. Efficiency drives it: 0.8 needs 801 points, 0.95 needs 15201, "
            f"and a value like 0.9137 needs billions. Round the efficiency to a "
            f"shorter decimal, or coarsen the action menu."
        )

    q = float(quantum)
    units = np.arange(size)

    # Movement per menu action, in lattice units, with the environment's clipping.
    moves, powers = [], []
    for level in levels:
        a = float(level)
        if a > 0:
            step_units = int(round(a * max_power * efficiency / q))
            landed = np.minimum(units + step_units, size - 1)
            power = (landed - units) * q / efficiency
        elif a < 0:
            step_units = int(round(-a * max_power / efficiency / q))
            landed = np.maximum(units - step_units, 0)
            power = -(units - landed) * q * efficiency
        else:
            landed = units.copy()
            power = np.zeros(size)
        moves.append(landed)
        powers.append(power)

    price, generation, load = expected_exogenous(horizon)

    value_next: np.ndarray = np.zeros(size)
    actions: np.ndarray = np.zeros((horizon, size), dtype=np.int64)
    for t in range(horizon - 1, -1, -1):
        net = load[t] - generation[t]
        candidates = np.stack(
            [-price[t] * (net + powers[a]) + value_next[moves[a]] for a in range(len(levels))],
            axis=1,
        )
        actions[t] = candidates.argmax(axis=1)
        value_next = candidates.max(axis=1)

    start = int(round(0.5 * capacity / q))
    return EnergySolution(
        value=float(value_next[start]), actions=actions, quantum=q, lattice=size
    )
