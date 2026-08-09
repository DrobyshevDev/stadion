"""Price and order in the same breath, where each decision changes the other.

Price sets demand, demand sets what to order, and a pile of unsold stock is an
argument for marking down rather than for holding. The classical answer is the
best *static* pair — one price, one base-stock target — with both chosen
together by search. It is a fair opponent and a genuinely limited one: the right
price depends on how much stock is in the warehouse, and a fixed price cannot
know that. The headroom is exactly that coupling.

The environment's docstring calls the joint optimum closed-form-free, which is
true and says nothing about whether it can be computed. It can, easily: the
demand carried in the observation is redrawn each period and predicts nothing,
so the state is on-hand stock alone. See :mod:`stadion.solvers.joint`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from decisionrl.baselines import static_price_order
from decisionrl.core.env import Env
from decisionrl.envs import JointPricingInventory

from stadion.core.agent import Agent, NearestChoiceAgent
from stadion.core.runner import tune
from stadion.core.task import Brief, Choice, Instance, Task, View
from stadion.solvers.joint import JointSolution, solve_joint_pricing

__all__ = ["JointPricing"]

PRICE_MIN = 0.5
PRICE_MAX = 3.0
#: Eight prices and six order sizes: forty-eight options.
#:
#: The price menu is the axis that matters here, because what the task is about
#: is letting the price answer to the stock on the shelf, and a coarse menu hides
#: that. Measured against the best single price, state-dependent pricing is worth
#: 2.2% of the return over six prices, 3.2% over eight, 3.5% over ten and 3.6%
#: over twelve — so eight is where the number has mostly stopped moving, and a
#: coarser menu would understate the very thing being measured.
N_PRICES = 8
N_ORDERS = 6

#: Encoded price actions. The environment decodes these, not the prices.
PRICE_ACTIONS = np.linspace(-1.0, 1.0, N_PRICES)


def decode_price(action: float) -> float:
    """The price the environment charges for an encoded action.

    Reproduces the environment's arithmetic including its precision: the action
    is cast to float32 before decoding, and under NEP 50 the whole expression
    stays in float32. Recomputing it in double precision shifts every price by
    about a part in ten million, which is invisible next to sampling error but
    would mean the recurrence and the environment were pricing different menus.
    """
    a = np.clip(np.float32(action), -1.0, 1.0)
    return float(np.float32(PRICE_MIN) + (a + np.float32(1.0)) / np.float32(2.0)
                 * np.float32(PRICE_MAX - PRICE_MIN))


def demand_mean(price: float, base_demand: float, elasticity: float) -> float:
    """Expected demand at a price, in the environment's own precision.

    The elasticity and the floor stay Python floats here, exactly as they are on
    the environment: under NEP 50 they are weak operands, so the arithmetic runs
    in float32 while their full double-precision values are used. Rounding them
    to float32 first — the obvious way to write this — moves the mean by a part
    in three million.
    """
    exponent = -elasticity * (np.float32(price) - PRICE_MIN)
    return base_demand * float(np.exp(exponent))


class _Optimal(Agent):
    name = "optimum"

    def __init__(self, solution: JointSolution, choices: tuple[Choice, ...]) -> None:
        self._solution = solution
        self._stride = len(choices) // N_PRICES

    def act(self, view: View) -> int:
        stock = int(round(float(view.obs[0]) * view.env.max_inventory))
        price_index, order_index = self._solution.decide(view.step, stock)
        return price_index * self._stride + order_index


class JointPricing(Task):
    name = "joint-pricing"
    summary = (
        "Set the price and the replenishment order together, where the right "
        "price depends on how much stock is already on the shelf."
    )
    baseline_name = "best static price and base-stock"

    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "max_inventory": float(rng.choice([24, 30, 36])),
            "max_order": float(rng.choice([12, 15, 18])),
            "price_min": PRICE_MIN,
            "price_max": PRICE_MAX,
            "base_demand": float(rng.uniform(7.0, 11.0)),
            "elasticity": float(rng.uniform(0.9, 1.5)),
            "unit_cost": float(rng.uniform(0.20, 0.45)),
            "holding_cost": float(rng.uniform(0.15, 0.35)),
            "stockout_penalty": float(rng.uniform(0.20, 0.50)),
            "horizon": 40.0,
        }

    def build(self, params: Mapping[str, float]) -> Env:
        return JointPricingInventory(
            max_inventory=int(params["max_inventory"]),
            max_order=int(params["max_order"]),
            price_min=params["price_min"],
            price_max=params["price_max"],
            base_demand=params["base_demand"],
            elasticity=params["elasticity"],
            unit_cost=params["unit_cost"],
            holding_cost=params["holding_cost"],
            stockout_penalty=params["stockout_penalty"],
            horizon=int(params["horizon"]),
        )

    def _order_levels(self, max_order: int) -> np.ndarray:
        return np.rint(np.linspace(0, max_order, N_ORDERS)).astype(np.int64)

    def _choices(self, max_order: int) -> tuple[Choice, ...]:
        """The joint menu. Constant within an instance, so built once."""
        key = ("choices", int(max_order))
        cached = self._memo.get(key)
        if cached is None:
            levels = self._order_levels(max_order)
            options = []
            for i, encoded in enumerate(PRICE_ACTIONS):
                price = decode_price(float(encoded))
                for j, order in enumerate(levels):
                    options.append(
                        Choice(
                            value=i * levels.size + j,
                            label=f"price {price:.2f}, order {order}",
                            action=np.array(
                                [encoded, 2.0 * float(order) / max_order - 1.0],
                                dtype=np.float32,
                            ),
                        )
                    )
            cached = tuple(options)
            self._memo[key] = cached
        return cached

    def brief(self, inst: Instance) -> Brief:
        p = inst.params
        levels = self._order_levels(int(p["max_order"]))
        menu = ", ".join(
            f"{decode_price(float(a)):.2f}" for a in PRICE_ACTIONS
        )
        text = (
            f"You set both the price and the restock order for "
            f"{int(p['horizon'])} periods.\n"
            f"Each period, in order: your order arrives at once, then you sell at the "
            f"price you posted.\n"
            f"Demand is Poisson with mean {p['base_demand']:.2f} * "
            f"exp(-{p['elasticity']:.2f} * (price - {p['price_min']:.2f})), so a higher "
            f"price earns more per unit and moves fewer.\n"
            f"\n"
            f"Money per period:\n"
            f"  + the posted price for every unit sold\n"
            f"  -{p['unit_cost']:.2f} for every unit ordered\n"
            f"  -{p['holding_cost']:.2f} for every unit still on the shelf afterwards\n"
            f"  -{p['stockout_penalty']:.2f} for every unit of demand you could not serve\n"
            f"\n"
            f"Prices available: {menu}. Order sizes available: "
            f"{', '.join(str(int(o)) for o in levels)}.\n"
            f"The shelf holds at most {int(p['max_inventory'])} units, and units ordered "
            f"past that are paid for and lost.\n"
            f"Holding costs {p['holding_cost']:.2f} a unit a period, so stock that is "
            f"not moving is an argument for a lower price, not only for a smaller order.\n"
            f"Maximise the total."
        )
        return Brief(
            task=self.name,
            seed=inst.seed,
            horizon=int(p["horizon"]),
            text=text,
            params=p,
        )

    def view(self, env: Env, obs: np.ndarray, step: int, earned: float) -> View:
        stock = int(round(float(obs[0]) * env.max_inventory))
        text = (
            f"Period {step + 1} of {env.horizon}. "
            f"On the shelf: {stock} of {env.max_inventory}. "
            f"Last period's demand: {float(obs[1]) * env.base_demand:.1f}. "
            f"Earned so far: {earned:.2f}."
        )
        return View(
            step=step,
            text=text,
            choices=self._choices(int(env.max_order)),
            earned=earned,
            obs=obs,
            env=env,
        )

    def baseline(self, inst: Instance) -> Agent:
        key = ("baseline", inst.seed)
        cached = self._memo.get(key)
        if cached is None:
            max_inventory = int(inst.params["max_inventory"])
            candidates = [
                (decode_price(float(a)), float(level))
                for a in PRICE_ACTIONS
                for level in range(0, max_inventory + 1, 2)
            ]
            cached = tune(self, inst, self._static_agent, candidates)
            self._memo[key] = cached
        return self._static_agent(cached)

    def _static_agent(self, pair: tuple[float, float]) -> Agent:
        price, level = pair
        return NearestChoiceAgent(
            static_price_order(price, level),
            name=f"{self.baseline_name} {price:.2f}/S={level:.0f}",
        )

    def _solve(self, inst: Instance) -> JointSolution:
        key = ("dp", inst.seed)
        cached = self._memo.get(key)
        if cached is None:
            p = inst.params
            prices = np.array([decode_price(float(a)) for a in PRICE_ACTIONS])
            means = np.array(
                [demand_mean(price, p["base_demand"], p["elasticity"]) for price in prices]
            )
            cached = solve_joint_pricing(
                prices=prices,
                demand_means=means,
                order_levels=self._order_levels(int(p["max_order"])),
                max_inventory=int(p["max_inventory"]),
                unit_cost=p["unit_cost"],
                holding_cost=p["holding_cost"],
                stockout_penalty=p["stockout_penalty"],
                horizon=int(p["horizon"]),
            )
            self._memo[key] = cached
        return cached

    def optimal(self, inst: Instance) -> Agent:
        return _Optimal(self._solve(inst), self._choices(int(inst.params["max_order"])))

    def optimal_value(self, inst: Instance) -> float:
        return self._solve(inst).value
