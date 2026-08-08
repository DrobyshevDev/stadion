"""Battery arbitrage in a microgrid: buy cheap, hold, spend into the evening peak.

The classical rule is a price threshold — charge when the price is below a line,
discharge when it is above, idle in between — with the line tuned by search. It
is what a controller with no forecasting does, and on a daily price cycle it does
most of the job.

Two things make the exact optimum available here despite a real-valued state.
The battery moves deterministically, and the expected cost is linear in the power
drawn, because price is drawn independently of generation and load. And the
charge itself lives on a lattice as long as the charge and discharge steps share
a common measure, which is why the efficiencies drawn below have simple squares.
The reasoning is set out in :mod:`stadion.solvers.storage`.

The instance family varies the battery and the length of the day. It does not
vary the price, solar and load profiles, which are fixed inside the environment;
this task therefore has less instance-to-instance variety than the other four,
and its numbers should be read with that in mind.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from decisionrl.baselines import price_threshold_battery
from decisionrl.core.env import Env
from decisionrl.envs import EnergyMicrogrid

from stadion.core.agent import Agent, NearestChoiceAgent
from stadion.core.runner import tune
from stadion.core.task import Brief, Choice, Instance, Task, View
from stadion.solvers.storage import EnergySolution, expected_exogenous, solve_energy

__all__ = ["Energy"]

#: Nine settings from full discharge to full charge. Both endpoints and zero are
#: on the menu, so the classical threshold rule — which only ever emits those
#: three — plays its own policy exactly rather than a rounded version of it.
LEVELS = np.linspace(-1.0, 1.0, 9)

#: Thresholds searched for the classical rule, matching decisionrl's own range.
THRESHOLDS = np.linspace(0.2, 0.5, 7)


class _Optimal(Agent):
    name = "optimum"

    def __init__(self, solution: EnergySolution, capacity: float) -> None:
        self._solution = solution
        self._capacity = capacity

    def act(self, view: View) -> int:
        charge = (float(view.obs[0]) + 1.0) * self._capacity / 2.0
        return self._solution.action_index(view.step, charge)


class Energy(Task):
    name = "energy"
    summary = (
        "Charge and discharge a battery against a daily price cycle, local solar "
        "and a household load."
    )
    baseline_name = "best price threshold"

    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "capacity": float(rng.choice([8.0, 10.0, 12.0])),
            "max_power": float(rng.choice([2.0, 2.5, 3.0])),
            # Short decimals: the charge lattice is as fine as the efficiency
            # makes it, and these stay in the low thousands of points.
            "efficiency": float(rng.choice([0.8, 0.9, 0.95, 1.0])),
            "horizon": float(rng.choice([36, 48, 60])),
        }

    def build(self, params: Mapping[str, float]) -> Env:
        return EnergyMicrogrid(
            capacity=params["capacity"],
            max_power=params["max_power"],
            efficiency=params["efficiency"],
            horizon=int(params["horizon"]),
        )

    def brief(self, inst: Instance) -> Brief:
        p = inst.params
        horizon = int(p["horizon"])
        price, generation, load = expected_exogenous(horizon)
        peak = int(np.argmax(price))
        trough = int(np.argmin(price))
        text = (
            f"You control a {p['capacity']:.0f} kWh battery over {horizon} steps, one day.\n"
            f"It charges and discharges at up to {p['max_power']:.1f} kW per step and "
            f"loses energy both ways: round-trip efficiency is {p['efficiency']:.2f}.\n"
            f"It starts half full.\n"
            f"\n"
            f"Each step: local solar generates, the household draws its load, and the "
            f"difference plus whatever the battery takes is bought from the grid at "
            f"that step's price. Surplus is sold back at the same price.\n"
            f"You pay price * (load - generation + battery power), so exporting earns.\n"
            f"\n"
            f"The three run on fixed daily profiles, with noise:\n"
            f"  price      cheapest around step {trough} ({price[trough]:.2f}), "
            f"dearest around step {peak} ({price[peak]:.2f})\n"
            f"  solar      zero at either end of the day, peaking near "
            f"{generation.max():.2f} at midday\n"
            f"  load       between {load.min():.2f} and {load.max():.2f}, heaviest in "
            f"the evening\n"
            f"The reading you are shown is a fresh sample of the same distribution as "
            f"the one you will be charged, not the charge itself.\n"
            f"Minimise what you pay over the day."
        )
        return Brief(
            task=self.name, seed=inst.seed, horizon=horizon, text=text, params=p
        )

    def view(self, env: Env, obs: np.ndarray, step: int, earned: float) -> View:
        charge = (float(obs[0]) + 1.0) * env.capacity / 2.0
        text = (
            f"Step {step + 1} of {env.horizon}. "
            f"Battery {charge:.2f} of {env.capacity:.0f} kWh. "
            f"Price now {float(obs[1]) * 1.5:.2f}, solar {float(obs[2]) * 4.0:.2f}, "
            f"load {float(obs[3]) * 4.0:.2f}. "
            f"Paid so far: {-earned:.2f}."
        )
        return View(
            step=step,
            text=text,
            choices=self._choices(env.max_power),
            earned=earned,
            obs=obs,
            env=env,
        )

    def _choices(self, max_power: float) -> tuple[Choice, ...]:
        """The power menu. Built once per battery and reused across every step."""
        key = ("choices", int(round(max_power * 100)))
        cached = self._memo.get(key)
        if cached is None:
            options = []
            for i, level in enumerate(LEVELS):
                power = float(level) * max_power
                if power > 0:
                    label = f"charge at {power:.2f} kW"
                elif power < 0:
                    label = f"discharge at {-power:.2f} kW"
                else:
                    label = "idle"
                options.append(
                    Choice(
                        value=i,
                        label=label,
                        action=np.array([float(level)], dtype=np.float32),
                    )
                )
            cached = tuple(options)
            self._memo[key] = cached
        return cached

    def baseline(self, inst: Instance) -> Agent:
        key = ("baseline", inst.seed)
        cached = self._memo.get(key)
        if cached is None:
            cached = tune(self, inst, self._threshold_agent, THRESHOLDS)
            self._memo[key] = cached
        return self._threshold_agent(cached)

    def _threshold_agent(self, low: float) -> Agent:
        return NearestChoiceAgent(
            price_threshold_battery(low), name=f"{self.baseline_name} {low:.2f}"
        )

    def _solve(self, inst: Instance) -> EnergySolution:
        key = ("dp", inst.seed)
        cached = self._memo.get(key)
        if cached is None:
            p = inst.params
            cached = solve_energy(
                capacity=p["capacity"],
                max_power=p["max_power"],
                efficiency=p["efficiency"],
                horizon=int(p["horizon"]),
                levels=LEVELS,
            )
            self._memo[key] = cached
        return cached

    def optimal(self, inst: Instance) -> Agent:
        return _Optimal(self._solve(inst), inst.params["capacity"])

    def optimal_value(self, inst: Instance) -> float:
        return self._solve(inst).value
