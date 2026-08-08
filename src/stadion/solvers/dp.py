"""Exact finite-horizon dynamic programming for the three task families.

These problems were chosen because they are small enough to solve outright:
a few dozen states, a discrete action set, a horizon in the tens. Backward
induction therefore gives the true optimum, not an approximation, and the
benchmark gets a ceiling that no agent can argue with.

Each solver returns both the optimal value and a time-indexed policy table.
Having the two separately is deliberate — simulating the policy has to
reproduce the value within Monte Carlo error, and ``tests/test_dp.py`` asserts
it. A dynamic program that quietly disagrees with the policy it emits is the
failure this pairing is designed to catch.

The recurrences follow the environments in ``decisionrl.envs`` exactly,
including the details that are easy to get wrong: ordering cost is charged on
the units requested rather than the units received, so overshooting a full
warehouse still costs money; and in the queue the admission decision is made
before the server's coin flip is revealed, so the maximisation sits outside
that expectation, not inside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "InventorySolution",
    "PricingSolution",
    "QueueSolution",
    "poisson_pmf",
    "solve_inventory",
    "solve_pricing",
    "solve_queue",
]


def poisson_pmf(lam: float, support: int | None = None) -> np.ndarray:
    """Poisson pmf over ``0..support``, normalised. NumPy only, no SciPy.

    The tail is truncated at roughly ten standard deviations and the remaining
    mass folded back by renormalising, which keeps the recurrences exact to
    within float error while bounding the loop.
    """
    if lam <= 0:
        raise ValueError(f"Poisson mean must be positive, got {lam}")
    if support is None:
        support = int(lam + 10.0 * np.sqrt(lam) + 20.0)
    k = np.arange(support + 1)
    log_factorial = np.cumsum(np.log(np.maximum(k, 1)))
    logpmf = -lam + k * np.log(lam) - log_factorial
    pmf = np.exp(logpmf)
    return pmf / pmf.sum()


# --------------------------------------------------------------------------- #
# inventory: order-up-to under Poisson demand
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class InventorySolution:
    """Optimal order quantity per (step, on-hand inventory)."""

    value: float
    actions: np.ndarray  # shape (horizon, max_inventory + 1)

    def order(self, step: int, inventory: int) -> int:
        t = min(step, self.actions.shape[0] - 1)
        x = int(np.clip(inventory, 0, self.actions.shape[1] - 1))
        return int(self.actions[t, x])


def solve_inventory(
    *,
    max_inventory: int,
    max_order: int,
    demand_mean: float,
    price: float,
    unit_cost: float,
    holding_cost: float,
    stockout_penalty: float,
    horizon: int,
) -> InventorySolution:
    """Backward induction over on-hand inventory.

    The initial inventory is drawn uniformly over ``0..max_inventory`` by the
    environment's ``reset``, so the reported value averages over that draw.
    """
    m, k = int(max_inventory), int(max_order)
    pmf = poisson_pmf(demand_mean)
    demand = np.arange(pmf.size)

    levels = np.arange(m + 1)  # doubles as post-order stock y and on-hand x
    orders = np.arange(k + 1)

    # One-step tables indexed by post-order stock level y and demand draw.
    sales = np.minimum(levels[:, None], demand[None, :])
    lost = demand[None, :] - sales
    leftover = levels[:, None] - sales
    immediate = price * sales - holding_cost * leftover - stockout_penalty * lost

    # Post-order stock reachable from on-hand x by ordering a, and the warehouse cap.
    reachable = np.minimum(levels[:, None] + orders[None, :], m)

    value_next: np.ndarray = np.zeros(m + 1)
    actions: np.ndarray = np.zeros((horizon, m + 1), dtype=np.int64)

    for t in range(horizon - 1, -1, -1):
        # g[y] = E_D[ immediate(y, D) + V_{t+1}(leftover(y, D)) ]
        g = (pmf[None, :] * (immediate + value_next[leftover])).sum(axis=1)
        # Ordering cost is charged on the units requested, so ordering past the
        # cap is paid for and thrown away.
        candidates = g[reachable] - unit_cost * orders[None, :]
        actions[t] = candidates.argmax(axis=1)
        value_next = candidates.max(axis=1)

    # reset() draws the opening inventory uniformly over 0..max_inventory.
    return InventorySolution(value=float(value_next.mean()), actions=actions)


# --------------------------------------------------------------------------- #
# dynamic pricing: perishable stock over a selling horizon
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PricingSolution:
    """Optimal price index per (step, remaining inventory)."""

    value: float
    actions: np.ndarray  # shape (horizon, initial_inventory + 1)

    def price_index(self, step: int, inventory: int) -> int:
        t = min(step, self.actions.shape[0] - 1)
        x = int(np.clip(inventory, 0, self.actions.shape[1] - 1))
        return int(self.actions[t, x])


def solve_pricing(
    *,
    prices: np.ndarray,
    initial_inventory: int,
    base_demand: float,
    elasticity: float,
    price_min: float,
    horizon: int,
) -> PricingSolution:
    """Backward induction over remaining inventory.

    The episode ends the moment stock runs out, so state 0 is absorbing with
    zero continuation value — which is what makes holding price high near the
    deadline worth it only while stock remains.
    """
    inv0 = int(initial_inventory)
    stock = np.arange(inv0 + 1)
    means = base_demand * np.exp(-elasticity * (np.asarray(prices, dtype=float) - price_min))

    # Per price, the demand distribution and the resulting sales/leftover tables.
    tables = []
    for price, lam in zip(prices, means, strict=True):
        pmf = poisson_pmf(float(lam))
        demand = np.arange(pmf.size)
        sales = np.minimum(stock[:, None], demand[None, :])
        tables.append((float(price), pmf, sales, stock[:, None] - sales))

    value_next: np.ndarray = np.zeros(inv0 + 1)
    actions: np.ndarray = np.zeros((horizon, inv0 + 1), dtype=np.int64)

    for t in range(horizon - 1, -1, -1):
        candidates = np.empty((inv0 + 1, len(tables)))
        for a, (price, pmf, sales, leftover) in enumerate(tables):
            candidates[:, a] = (pmf[None, :] * (price * sales + value_next[leftover])).sum(axis=1)
        actions[t] = candidates.argmax(axis=1)
        value_next = candidates.max(axis=1)
        value_next[0] = 0.0  # out of stock: the episode has already terminated

    return PricingSolution(value=float(value_next[inv0]), actions=actions)


# --------------------------------------------------------------------------- #
# queue admission control: value threshold under congestion cost
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class QueueSolution:
    """Optimal admission threshold on job value, per (step, queue length).

    ``-inf`` means admit anything, ``+inf`` means admit nothing.
    """

    value: float
    thresholds: np.ndarray  # shape (horizon, buffer_size + 1)

    def admits(self, step: int, queue: int, job_value: float) -> bool:
        t = min(step, self.thresholds.shape[0] - 1)
        q = int(np.clip(queue, 0, self.thresholds.shape[1] - 1))
        return bool(job_value >= self.thresholds[t, q])


def solve_queue(
    *,
    buffer_size: int,
    service_prob: float,
    holding_cost: float,
    horizon: int,
) -> QueueSolution:
    """Backward induction over queue length, integrating the job value analytically.

    Job values are uniform on [0, 1) and observed before the decision, so the
    step value is ``max(reject, admit_const + value * pi)`` — a maximum of two
    lines. Its integral over the uniform draw is closed-form, which removes the
    discretisation error a grid over job values would introduce.
    """
    b, p, h = int(buffer_size), float(service_prob), float(holding_cost)
    queues = np.arange(b + 1)

    # Post-service queue distribution: q -> {q - 1 w.p. p, q w.p. 1 - p}, and
    # 0 -> {0} because an empty server has nothing to complete.
    served = np.maximum(queues - 1, 0)
    prob_served = np.where(queues > 0, p, 0.0)

    def expect(values: np.ndarray) -> np.ndarray:
        """E over the server's coin flip, for a quantity indexed by post-service queue."""
        return prob_served * values[served] + (1.0 - prob_served) * values[queues]

    room = queues < b
    pi = expect(room.astype(float))  # probability there is room after service

    w_next: np.ndarray = np.zeros(b + 1)  # E_value[ V_{t+1}(q, value) ]
    thresholds: np.ndarray = np.empty((horizon, b + 1))

    for t in range(horizon - 1, -1, -1):
        reject = expect(-h * queues + w_next)

        # Admitting adds one job unless the post-service queue is already full,
        # in which case the environment ignores the action and it equals rejecting.
        admit_full = -h * queues + w_next  # no room: identical to rejecting
        admit_room = np.empty(b + 1)
        admit_room[:-1] = -h * (queues[:-1] + 1) + w_next[1:]
        admit_room[-1] = admit_full[-1]
        admit_const = expect(np.where(room, admit_room, admit_full))

        with np.errstate(divide="ignore", invalid="ignore"):
            v_star = np.where(pi > 0, (reject - admit_const) / np.where(pi > 0, pi, 1.0), np.inf)
        v_star = np.clip(v_star, 0.0, 1.0)
        # pi == 0 means the value term vanishes: the choice no longer depends on
        # the job's value, so the threshold degenerates to "always" or "never".
        degenerate = pi <= 0
        v_star = np.where(degenerate, np.where(admit_const >= reject, 0.0, 1.0), v_star)

        w_next = (
            v_star * reject
            + admit_const * (1.0 - v_star)
            + pi * (1.0 - v_star**2) / 2.0
        )
        thresholds[t] = np.where(v_star >= 1.0, np.inf, v_star)

    return QueueSolution(value=float(w_next[0]), thresholds=thresholds)
