"""Revenue management: a perishable stock, a deadline, and one price per period.

The classical answer here has no closed form, so the baseline is the strongest
*fixed* price — the price a shop would settle on and leave alone. Its one free
parameter is chosen by search, and that search runs on tuning seeds that never
appear in the evaluation set, so the rule is not fitted to the episodes it is
scored on.

This is the task with the most room in it. A fixed price cannot react to stock
running low early or piling up late, and the optimum does exactly that, so the
gap between the two is wide and an agent has something real to win.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from decisionrl.baselines import best_fixed_action, fixed_action
from decisionrl.core.env import Env
from decisionrl.envs import DynamicPricing

from stadion.core.agent import Agent, PolicyAgent
from stadion.core.task import Brief, Choice, Instance, Task, View
from stadion.solvers.dp import PricingSolution, solve_pricing

__all__ = ["Pricing"]

PRICE_MIN = 0.5
PRICE_MAX = 4.0
N_PRICES = 8


class _Optimal(Agent):
    name = "optimum"

    def __init__(self, solution: PricingSolution) -> None:
        self._solution = solution

    def act(self, view: View) -> int:
        stock = int(round(float(view.obs[0]) * view.env.initial_inventory))
        return self._solution.price_index(view.step, stock)


class Pricing(Task):
    name = "pricing"
    summary = (
        "Set a price each period for a perishable stock that must clear by a "
        "deadline; unsold units are worth nothing."
    )
    baseline_name = "best fixed price"

    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "n_prices": float(N_PRICES),
            "price_min": PRICE_MIN,
            "price_max": PRICE_MAX,
            "initial_inventory": float(rng.integers(6, 13)),
            "base_demand": float(rng.uniform(5.0, 10.0)),
            "elasticity": float(rng.uniform(0.6, 1.4)),
            "horizon": float(rng.choice([16, 20, 24])),
        }

    def build(self, params: Mapping[str, float]) -> Env:
        return DynamicPricing(
            n_prices=int(params["n_prices"]),
            price_min=params["price_min"],
            price_max=params["price_max"],
            initial_inventory=int(params["initial_inventory"]),
            base_demand=params["base_demand"],
            elasticity=params["elasticity"],
            horizon=int(params["horizon"]),
        )

    def brief(self, inst: Instance) -> Brief:
        p = inst.params
        prices = np.linspace(p["price_min"], p["price_max"], int(p["n_prices"]))
        listed = ", ".join(f"{i}={v:.2f}" for i, v in enumerate(prices))
        text = (
            f"You have {int(p['initial_inventory'])} units of a perishable good and "
            f"{int(p['horizon'])} selling periods.\n"
            f"Each period you post one price from this menu: {listed}.\n"
            f"That period's demand is Poisson with mean "
            f"{p['base_demand']:.2f} * exp(-{p['elasticity']:.2f} * (price - "
            f"{p['price_min']:.2f})), so a higher price earns more per unit and sells "
            f"fewer.\n"
            f"You sell whichever is smaller, demand or remaining stock, and earn the "
            f"posted price for each unit sold.\n"
            f"Selling out ends the run early. Stock left when the periods run out is "
            f"worth nothing.\n"
            f"Maximise total revenue."
        )
        return Brief(
            task=self.name,
            seed=inst.seed,
            horizon=int(p["horizon"]),
            text=text,
            params=p,
        )

    def view(self, env: Env, obs: np.ndarray, step: int, earned: float) -> View:
        stock = int(round(float(obs[0]) * env.initial_inventory))
        left = env.horizon - step
        text = (
            f"Period {step + 1} of {env.horizon} ({left} left). "
            f"Stock: {stock} of {env.initial_inventory}. "
            f"Earned so far: {earned:.2f}."
        )
        choices = tuple(
            Choice(
                value=i,
                label=f"price {price:.2f}",
                detail=(
                    "expected demand "
                    f"{env.base_demand * np.exp(-env.elasticity * (price - env.price_min)):.2f}"
                ),
            )
            for i, price in enumerate(env.prices)
        )
        return View(step=step, text=text, choices=choices, earned=earned, obs=obs, env=env)

    def baseline(self, inst: Instance) -> Agent:
        key = ("baseline", inst.seed)
        cached = self._memo.get(key)
        if cached is None:
            action, _ = best_fixed_action(
                inst.env, episodes=self.tuning_episodes, seed=self.tuning_seed
            )
            cached = int(action)
            self._memo[key] = cached
        price = np.linspace(PRICE_MIN, PRICE_MAX, N_PRICES)[cached]
        return PolicyAgent(fixed_action(cached), name=f"{self.baseline_name} {price:.2f}")

    def _solve(self, inst: Instance) -> PricingSolution:
        key = ("dp", inst.seed)
        cached = self._memo.get(key)
        if cached is None:
            p = inst.params
            cached = solve_pricing(
                prices=np.linspace(p["price_min"], p["price_max"], int(p["n_prices"])),
                initial_inventory=int(p["initial_inventory"]),
                base_demand=p["base_demand"],
                elasticity=p["elasticity"],
                price_min=p["price_min"],
                horizon=int(p["horizon"]),
            )
            self._memo[key] = cached
        return cached

    def optimal(self, inst: Instance) -> Agent:
        return _Optimal(self._solve(inst))

    def optimal_value(self, inst: Instance) -> float:
        return self._solve(inst).value
