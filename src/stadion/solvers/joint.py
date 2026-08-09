"""Exact backward induction for pricing and ordering decided together.

The environment's own docstring says there is no closed-form joint optimum, and
that is true. It does not follow that the optimum is out of reach: no closed form
and no exact solution are different claims, and this one is a finite Markov
decision process once two observations are made.

The observation carries the last period's demand, but demand is drawn afresh
each period from a mean that depends only on the price just posted, so the
number predicts nothing and is not part of the state. What remains is on-hand
stock, a whole number bounded by the warehouse. And the order quantity is
already integral — the environment rounds it — so the only continuous decision
is the price, which the task hands to every player as a menu.

State is therefore one-dimensional and discrete, the action set is finite, and
backward induction is exact. It is the smallest state space in the library,
which is a fair warning about how little the shape of an observation tells you
about the size of a problem.

The coupling the task is built around survives all of this. The best price
depends on how much stock is sitting in the warehouse — mark down to clear a
pile, hold firm when short — so no single fixed price is right, and the gap
between the best static rule and this recurrence is exactly what a
state-dependent policy has to earn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stadion.solvers.dp import poisson_pmf

__all__ = ["JointSolution", "solve_joint_pricing"]


@dataclass(frozen=True, slots=True)
class JointSolution:
    """Optimal (price index, order index) pair per (step, on-hand stock)."""

    value: float
    prices: np.ndarray  # shape (horizon, max_inventory + 1)
    orders: np.ndarray

    def decide(self, step: int, inventory: int) -> tuple[int, int]:
        t = min(step, self.prices.shape[0] - 1)
        x = int(np.clip(inventory, 0, self.prices.shape[1] - 1))
        return int(self.prices[t, x]), int(self.orders[t, x])


def solve_joint_pricing(
    *,
    prices: np.ndarray,
    demand_means: np.ndarray,
    order_levels: np.ndarray,
    max_inventory: int,
    unit_cost: float,
    holding_cost: float,
    stockout_penalty: float,
    horizon: int,
) -> JointSolution:
    """Backward induction over on-hand stock.

    ``prices`` and ``demand_means`` arrive already decoded, because the encoding
    belongs to the environment rather than to the recurrence: an action is
    rounded to float32 before it is turned into a price, and the task reproduces
    that chain exactly. Recomputing it here in double precision would price each
    option a fraction of a percent away from what the environment charges, and
    solve a neighbouring problem.
    """
    m = int(max_inventory)
    levels = np.asarray(order_levels, dtype=np.int64)
    stock = np.arange(m + 1)

    # Post-order stock reachable from x by ordering o, and the warehouse cap.
    reachable = np.minimum(stock[:, None] + levels[None, :], m)

    # Per price, the one-step tables indexed by post-order stock and demand.
    tables = []
    for price, lam in zip(np.asarray(prices, dtype=float), demand_means, strict=True):
        pmf = poisson_pmf(float(lam))
        demand = np.arange(pmf.size)
        sales = np.minimum(stock[:, None], demand[None, :])
        lost = demand[None, :] - sales
        leftover = stock[:, None] - sales
        immediate = float(price) * sales - holding_cost * leftover - stockout_penalty * lost
        tables.append((pmf, leftover, immediate))

    value_next: np.ndarray = np.zeros(m + 1)
    best_price: np.ndarray = np.zeros((horizon, m + 1), dtype=np.int64)
    best_order: np.ndarray = np.zeros((horizon, m + 1), dtype=np.int64)

    for t in range(horizon - 1, -1, -1):
        # candidates[x, p, o]: post the p-th price and order the o-th quantity.
        candidates = np.empty((m + 1, len(tables), levels.size))
        for p, (pmf, leftover, immediate) in enumerate(tables):
            # g[y] = E_D[ immediate(y, D) + V_{t+1}(leftover(y, D)) ]
            g = (pmf[None, :] * (immediate + value_next[leftover])).sum(axis=1)
            # Ordering cost is charged on the units requested, so ordering past
            # the cap is paid for and thrown away.
            candidates[:, p, :] = g[reachable] - unit_cost * levels[None, :]

        flat = candidates.reshape(m + 1, -1)
        choice = flat.argmax(axis=1)
        best_price[t] = choice // levels.size
        best_order[t] = choice % levels.size
        value_next = flat.max(axis=1)

    # reset() draws the opening stock uniformly over 0..max_inventory.
    return JointSolution(
        value=float(value_next.mean()), prices=best_price, orders=best_order
    )
