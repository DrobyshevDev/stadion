"""Stock replenishment under Poisson demand.

The classical answer is the newsvendor critical fractile: order up to the level
at which the chance of running short equals the ratio of shortage cost to total
cost. It is a closed form, it needs no tuning, and it is what a practitioner
would actually use — so it is the baseline here, unmodified from
``decisionrl.baselines``.

It is also not quite optimal on this instance family, which is the point. The
formula assumes an unbounded order and an infinite horizon; the environment caps
both the warehouse and the daily order, and charges for units ordered past the
cap. Whatever headroom that leaves is what an agent has to win.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from decisionrl.baselines import analytic_base_stock_level, base_stock
from decisionrl.core.env import Env
from decisionrl.envs import InventoryManagement

from stadion.core.agent import Agent, PolicyAgent
from stadion.core.task import Brief, Choice, Instance, Task, View
from stadion.solvers.dp import InventorySolution, solve_inventory

__all__ = ["Inventory"]


class _Optimal(Agent):
    name = "optimum"

    def __init__(self, solution: InventorySolution) -> None:
        self._solution = solution

    def act(self, view: View) -> int:
        on_hand = int(round(float(view.obs[0]) * view.env.max_inventory))
        return self._solution.order(view.step, on_hand)


class Inventory(Task):
    name = "inventory"
    summary = (
        "Order stock each day against random demand, trading the cost of holding "
        "against the cost of running out."
    )
    baseline_name = "analytic base-stock (newsvendor critical fractile)"

    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "max_inventory": float(rng.choice([16, 20, 24])),
            "max_order": float(rng.choice([8, 10, 12])),
            "demand_mean": float(rng.uniform(3.0, 8.0)),
            "price": 1.0,
            "unit_cost": float(rng.uniform(0.20, 0.50)),
            "holding_cost": float(rng.uniform(0.02, 0.10)),
            "stockout_penalty": float(rng.uniform(0.10, 0.40)),
            "horizon": 60.0,
        }

    def build(self, params: Mapping[str, float]) -> Env:
        return InventoryManagement(
            max_inventory=int(params["max_inventory"]),
            max_order=int(params["max_order"]),
            demand_mean=params["demand_mean"],
            price=params["price"],
            unit_cost=params["unit_cost"],
            holding_cost=params["holding_cost"],
            stockout_penalty=params["stockout_penalty"],
            horizon=int(params["horizon"]),
        )

    def brief(self, inst: Instance) -> Brief:
        p = inst.params
        text = (
            f"You run one warehouse for {int(p['horizon'])} days.\n"
            f"Each day, in order: you order units and they arrive at once; then the "
            f"day's demand arrives.\n"
            f"Demand is Poisson with mean {p['demand_mean']:.2f}, drawn fresh each day.\n"
            f"\n"
            f"Money per day:\n"
            f"  +{p['price']:.2f} for every unit sold\n"
            f"  -{p['unit_cost']:.2f} for every unit ordered\n"
            f"  -{p['holding_cost']:.2f} for every unit still on the shelf overnight\n"
            f"  -{p['stockout_penalty']:.2f} for every unit of demand you could not serve\n"
            f"\n"
            f"The warehouse holds at most {int(p['max_inventory'])} units and you may order "
            f"at most {int(p['max_order'])} per day.\n"
            f"Units ordered past the warehouse cap are paid for and lost, so ordering "
            f"into a full warehouse is pure waste.\n"
            f"Maximise the total over all {int(p['horizon'])} days."
        )
        return Brief(
            task=self.name,
            seed=inst.seed,
            horizon=int(p["horizon"]),
            text=text,
            params=p,
        )

    def view(self, env: Env, obs: np.ndarray, step: int, earned: float) -> View:
        on_hand = int(round(float(obs[0]) * env.max_inventory))
        text = (
            f"Day {step + 1} of {env.horizon}. "
            f"On hand: {on_hand} of {env.max_inventory}. "
            f"Earned so far: {earned:.2f}."
        )
        choices = tuple(
            Choice(value=a, label=f"order {a}", detail=f"costs {a * env.unit_cost:.2f}")
            for a in range(env.max_order + 1)
        )
        return View(step=step, text=text, choices=choices, earned=earned, obs=obs, env=env)

    def baseline(self, inst: Instance) -> Agent:
        env = inst.env()
        level = analytic_base_stock_level(env)
        return PolicyAgent(base_stock(level), name=f"{self.baseline_name} S={level}")

    def _solve(self, inst: Instance) -> InventorySolution:
        key = ("dp", inst.seed)
        cached = self._memo.get(key)
        if cached is None:
            p = inst.params
            cached = solve_inventory(
                max_inventory=int(p["max_inventory"]),
                max_order=int(p["max_order"]),
                demand_mean=p["demand_mean"],
                price=p["price"],
                unit_cost=p["unit_cost"],
                holding_cost=p["holding_cost"],
                stockout_penalty=p["stockout_penalty"],
                horizon=int(p["horizon"]),
            )
            self._memo[key] = cached
        return cached

    def optimal(self, inst: Instance) -> Agent:
        return _Optimal(self._solve(inst))

    def optimal_value(self, inst: Instance) -> float:
        return self._solve(inst).value
