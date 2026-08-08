"""The built-in task families.

Five problems, chosen on one criterion: each is a decision a business makes over
and over, each has a textbook method a practitioner would reach for, and each is
small enough that the true optimum can be computed rather than guessed at. The
last of those is what the rest of the library is built on, and it is the reason
the list is short — a problem earns a place here by being solvable, and most
operational problems are not.
"""

from __future__ import annotations

from collections.abc import Sequence

from stadion.core.task import Task
from stadion.tasks.energy import Energy
from stadion.tasks.inventory import Inventory
from stadion.tasks.pricing import Pricing
from stadion.tasks.queueing import Queueing
from stadion.tasks.supply_chain import SupplyChainTask

__all__ = ["Energy", "Inventory", "Pricing", "Queueing", "SupplyChainTask", "all_tasks"]

_TASKS: tuple[Task, ...] = (
    Inventory(),
    Pricing(),
    Queueing(),
    Energy(),
    SupplyChainTask(),
)


def all_tasks() -> Sequence[Task]:
    """Every built-in task, as singletons — they cache solved dynamic programs."""
    return _TASKS
