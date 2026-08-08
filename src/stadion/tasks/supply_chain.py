"""Two echelons, one period of transit, and the temptation to over-order at both.

A retailer sells to customers, a warehouse replenishes the retailer, and an
unlimited supplier replenishes the warehouse. Every shipment takes a period to
arrive, so both orders are placed a period before they can help, and both must
be placed before that period's demand is known. Over-order and you pay to hold
stock at two levels; under-order and the retailer runs dry. Ordering in reaction
to the last shortage is what produces the bullwhip.

The classical rule is per-echelon base-stock: bring each level's stock plus
whatever is already in transit up to a target, with the target tuned by search.

The exact optimum survives here because the state collapses from five numbers to
two — see :mod:`stadion.solvers.serial` for why, and for the one cap the
recurrence puts on stock.

Orders are whole units on a five-point menu at each echelon. That keeps every
quantity in the environment integral, and it keeps the menu an agent reads to
twenty-five lines rather than a few hundred. It is also the coarsest thing in
the library: a base-stock target the menu cannot express costs both the classical
rule and the optimum something, and both pay it equally.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from decisionrl.baselines import supply_base_stock
from decisionrl.core.env import Env
from decisionrl.envs import SupplyChain

from stadion.core.agent import Agent, NearestChoiceAgent
from stadion.core.runner import tune
from stadion.core.task import Brief, Choice, Instance, Task, View
from stadion.solvers.serial import SupplySolution, solve_supply_chain

__all__ = ["SupplyChainTask"]

#: Order quantities per echelon, as quarters of the maximum. Quarters are exact
#: in binary, so the fraction the environment multiplies back by lands on a whole
#: number of units and nothing drifts off the integer lattice the solver needs.
FRACTIONS = np.linspace(0.0, 1.0, 5)

#: Base-stock targets searched for the classical rule.
TARGETS = np.arange(6, 26)

#: Stock beyond this saturates the environment's own observation, so neither the
#: recurrence nor an agent can see past it. It is the environment's normalisation
#: constant, read the same way decisionrl's own base-stock baseline reads it.
CAP = 40
SCALE = 40.0


class _Optimal(Agent):
    name = "optimum"

    def __init__(self, solution: SupplySolution, scale: float, orders: np.ndarray) -> None:
        self._solution = solution
        self._scale = scale
        self._orders = orders

    def act(self, view: View) -> int:
        obs = view.obs
        retail = int(round((float(obs[0]) + float(obs[2])) * self._scale))
        warehouse = int(round((float(obs[1]) + float(obs[3])) * self._scale))
        i, j = self._solution.orders(view.step, retail, warehouse)
        return i * self._orders.size + j


class SupplyChainTask(Task):
    name = "supply-chain"
    summary = (
        "Order at two echelons a period before the stock can help and before "
        "demand is known, without piling up at either."
    )
    baseline_name = "per-echelon base-stock"

    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            # Whole units: the opening stock is set to this, and the solver's
            # lattice is integral.
            "demand_mean": float(rng.integers(3, 8)),
            # A multiple of four, so quarters of it are whole units.
            "max_order": float(rng.choice([8.0, 12.0, 16.0])),
            "holding_cost_retail": float(rng.uniform(0.06, 0.16)),
            "holding_cost_warehouse": float(rng.uniform(0.02, 0.08)),
            "stockout_penalty": float(rng.uniform(0.3, 0.8)),
            "horizon": 60.0,
        }

    def build(self, params: Mapping[str, float]) -> Env:
        return SupplyChain(
            demand_mean=params["demand_mean"],
            max_order=params["max_order"],
            holding_cost_retail=params["holding_cost_retail"],
            holding_cost_warehouse=params["holding_cost_warehouse"],
            stockout_penalty=params["stockout_penalty"],
            horizon=int(params["horizon"]),
        )

    def _orders(self, params: Mapping[str, float]) -> np.ndarray:
        return np.rint(FRACTIONS * params["max_order"]).astype(np.int64)

    def brief(self, inst: Instance) -> Brief:
        p = inst.params
        orders = ", ".join(str(int(o)) for o in self._orders(p))
        text = (
            f"You run a retailer and the warehouse behind it for "
            f"{int(p['horizon'])} periods.\n"
            f"Each period, in this order:\n"
            f"  1. Last period's shipments arrive: the warehouse's order reaches the "
            f"warehouse, the retailer's reaches the retailer.\n"
            f"  2. You place both orders. The warehouse ships to the retailer out of "
            f"what it holds now, and can ship no more than that; the supplier behind "
            f"the warehouse is unlimited. Both shipments arrive next period.\n"
            f"  3. Customer demand hits the retailer. It is Poisson with mean "
            f"{p['demand_mean']:.0f}. Anything you cannot serve is lost, not backordered.\n"
            f"  4. You pay {p['holding_cost_retail']:.2f} per unit left at the retailer, "
            f"{p['holding_cost_warehouse']:.2f} per unit left at the warehouse, and "
            f"{p['stockout_penalty']:.2f} per unit of demand you missed.\n"
            f"\n"
            f"Order quantities available at each echelon: {orders}.\n"
            f"Both orders are placed before you see the demand, and neither can help "
            f"until the period after.\n"
            f"Minimise total cost."
        )
        return Brief(
            task=self.name,
            seed=inst.seed,
            horizon=int(p["horizon"]),
            text=text,
            params=p,
        )

    def view(self, env: Env, obs: np.ndarray, step: int, earned: float) -> View:
        scale = getattr(env, "_scale", SCALE)
        retail_stock = float(obs[0]) * scale
        warehouse_stock = float(obs[1]) * scale
        retail_incoming = float(obs[2]) * scale
        warehouse_incoming = float(obs[3]) * scale
        text = (
            f"Period {step + 1} of {env.horizon}. "
            f"Retailer holds {retail_stock:.0f} with {retail_incoming:.0f} arriving; "
            f"warehouse holds {warehouse_stock:.0f} with {warehouse_incoming:.0f} arriving. "
            f"Last demand was {float(obs[4]) * scale:.0f}. "
            f"Cost so far: {-earned:.2f}."
        )
        return View(
            step=step,
            text=text,
            choices=self._choices(env.max_order),
            earned=earned,
            obs=obs,
            env=env,
        )

    def _choices(self, max_order: float) -> tuple[Choice, ...]:
        """The order menu. Built once per order cap and reused.

        Twenty-five options each carrying a NumPy action, rebuilt on every step
        of every episode of every tuning candidate, dominated the runtime before
        this cache. The menu depends on nothing that changes within an episode.
        """
        key = ("choices", int(max_order))
        cached = self._memo.get(key)
        if cached is None:
            orders = np.rint(FRACTIONS * max_order).astype(int)
            cached = tuple(
                Choice(
                    value=i * orders.size + j,
                    label=f"retailer {retail}, warehouse {warehouse}",
                    action=np.array([FRACTIONS[i], FRACTIONS[j]], dtype=np.float32),
                )
                for i, retail in enumerate(orders)
                for j, warehouse in enumerate(orders)
            )
            self._memo[key] = cached
        return cached

    def baseline(self, inst: Instance) -> Agent:
        key = ("baseline", inst.seed)
        cached = self._memo.get(key)
        if cached is None:
            cached = tune(self, inst, self._base_stock_agent, TARGETS)
            self._memo[key] = cached
        return self._base_stock_agent(cached)

    def _base_stock_agent(self, target: float) -> Agent:
        return NearestChoiceAgent(
            supply_base_stock(target), name=f"{self.baseline_name} S={target:.0f}"
        )

    def _solve(self, inst: Instance) -> SupplySolution:
        key = ("dp", inst.seed)
        cached = self._memo.get(key)
        if cached is None:
            p = inst.params
            cached = solve_supply_chain(
                demand_mean=p["demand_mean"],
                levels=self._orders(p),
                holding_cost_retail=p["holding_cost_retail"],
                holding_cost_warehouse=p["holding_cost_warehouse"],
                stockout_penalty=p["stockout_penalty"],
                horizon=int(p["horizon"]),
                cap=CAP,
            )
            self._memo[key] = cached
        return cached

    def optimal(self, inst: Instance) -> Agent:
        return _Optimal(self._solve(inst), SCALE, self._orders(inst.params))

    def optimal_value(self, inst: Instance) -> float:
        return self._solve(inst).value
