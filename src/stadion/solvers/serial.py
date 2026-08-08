"""Exact backward induction for the two-echelon chain, after collapsing the state.

The observation has five numbers — stock and in-transit at both echelons, plus
last period's demand — which reads like a five-dimensional problem and would be
hopeless to solve outright. It is two-dimensional.

Two reductions do it. The first thing ``SupplyChain.step`` does is add each
echelon's in-transit shipment to its stock, and nothing between that moment and
the next decision separates the two again; so stock and pipeline only ever
appear as their sum, and the state is the pair

    A = retail stock + retail shipment arriving
    B = warehouse stock + warehouse shipment arriving

Last period's demand is in the observation but not in the dynamics: demand is
drawn independently each period, so it predicts nothing. It is not part of the
state.

Everything is integral. Demand is Poisson, so realised demand is a whole number
of units; the order menu is whole units; and the tasks draw an integer mean so
the opening stock is whole too. Nothing here is rounded or interpolated.

The one approximation is the cap on how much stock the recurrence tracks, and it
is set at the point where the environment's own observation saturates — past it
an agent is blind anyway. ``tests/test_dp.py`` checks that raising the cap does
not move the optimum, which is what makes the choice safe rather than merely
convenient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stadion.solvers.dp import poisson_pmf

__all__ = ["SupplySolution", "solve_supply_chain"]


@dataclass(frozen=True, slots=True)
class SupplySolution:
    """Optimal (retail order, warehouse order) index pair per (step, A, B)."""

    value: float
    retail: np.ndarray  # shape (horizon, cap + 1, cap + 1)
    warehouse: np.ndarray
    cap: int

    def orders(self, step: int, available_retail: int, available_warehouse: int) -> tuple[int, int]:
        t = min(step, self.retail.shape[0] - 1)
        a = int(np.clip(available_retail, 0, self.cap))
        b = int(np.clip(available_warehouse, 0, self.cap))
        return int(self.retail[t, a, b]), int(self.warehouse[t, a, b])


def solve_supply_chain(
    *,
    demand_mean: float,
    levels: np.ndarray,
    holding_cost_retail: float,
    holding_cost_warehouse: float,
    stockout_penalty: float,
    horizon: int,
    cap: int = 40,
) -> SupplySolution:
    """Backward induction over (A, B).

    ``levels`` are the whole-unit order quantities on the menu, the same list the
    agent picks from at each echelon.
    """
    orders = np.asarray(levels, dtype=np.int64)
    n_orders = orders.size
    size = int(cap) + 1
    stock = np.arange(size)

    pmf = poisson_pmf(demand_mean)
    demand = np.arange(pmf.size)

    # Retail side, indexed by available stock A and demand D.
    sales = np.minimum(stock[:, None], demand[None, :])
    unmet = demand[None, :] - sales
    left = stock[:, None] - sales
    retail_cost = -(holding_cost_retail * left + stockout_penalty * unmet)
    expected_retail_cost = (pmf[None, :] * retail_cost).sum(axis=1)

    # A shipment of `s` units arrives next period, for every s the menu can produce.
    shipments = np.arange(int(orders.max()) + 1)
    arrives = np.minimum(left[:, None, :] + shipments[None, :, None], size - 1)

    value_next: np.ndarray = np.zeros((size, size))
    best_retail: np.ndarray = np.zeros((horizon, size, size), dtype=np.int64)
    best_warehouse: np.ndarray = np.zeros((horizon, size, size), dtype=np.int64)

    for t in range(horizon - 1, -1, -1):
        # continuation[A, s, B'] — the value of arriving at retail state
        # (A less sales, plus s) with the warehouse holding B'.
        continuation = np.empty((size, shipments.size, size))
        for s in shipments:
            reached = value_next[arrives[:, s, :], :]  # (A, D, B')
            continuation[:, s, :] = (pmf[None, :, None] * reached).sum(axis=1)
        step_value = expected_retail_cost[:, None, None] + continuation

        best: np.ndarray = np.full((size, size), -np.inf)
        pick_r: np.ndarray = np.zeros((size, size), dtype=np.int64)
        pick_w: np.ndarray = np.zeros((size, size), dtype=np.int64)
        for i in range(n_orders):
            # The warehouse can only ship what it has; both orders are placed
            # before demand, so neither may depend on it.
            shipped = np.minimum(orders[i], stock)
            remaining = stock - shipped
            carry = -holding_cost_warehouse * remaining
            for j in range(n_orders):
                restocked = np.minimum(remaining + orders[j], size - 1)
                candidate = step_value[:, shipped, restocked] + carry[None, :]
                better = candidate > best
                best = np.where(better, candidate, best)
                pick_r = np.where(better, i, pick_r)
                pick_w = np.where(better, j, pick_w)

        best_retail[t] = pick_r
        best_warehouse[t] = pick_w
        value_next = best

    opening = int(round(2.0 * demand_mean))  # stock plus the shipment already in transit
    opening = min(opening, size - 1)
    return SupplySolution(
        value=float(value_next[opening, opening]),
        retail=best_retail,
        warehouse=best_warehouse,
        cap=int(cap),
    )
