"""A plausible pricing heuristic, and what the harness makes of it.

The rule is the one most people write first: charge more when stock is scarce
relative to the time left to sell it, less when it is piling up. It is the right
*shape* — the optimal policy does move that way — and it is the kind of thing
that looks obviously better than a fixed price until someone measures it.

Run it with::

    python examples/scarcity.py
"""

from __future__ import annotations

import stadion


class Scarcity(stadion.Agent):
    """Price on the ratio of stock remaining to time remaining."""

    name = "scarcity"

    def act(self, view: stadion.View) -> int:
        stock_left, time_left = float(view.obs[0]), float(view.obs[1])
        # Both are fractions of their starting value. Above 1 means stock is
        # outlasting the clock and should be discounted; below 1 means it is
        # running out and can carry a higher price.
        pressure = stock_left / max(time_left, 1e-6)
        top = len(view.choices) - 1
        # Balanced (pressure 1) sits in the middle of the price menu; twice as
        # much stock as time drops to the floor, no stock left goes to the top.
        index = int(round(top * (1.0 - pressure / 2.0)))
        return view.choices[min(max(index, 0), top)].value


if __name__ == "__main__":
    task = stadion.get("pricing")
    report = stadion.evaluate(task, Scarcity(), instances=40, episodes=20)
    print(report.summary())
