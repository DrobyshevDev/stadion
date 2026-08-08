"""stadion — a proving ground for operational decisions.

An agent is scored against two references it cannot argue with: the classical
operations-research method for the problem, and the exact optimum computed by
backward induction. The result is a normalised score with a bootstrap interval,
and "indistinguishable from the classical method" is a first-class outcome.

    >>> import stadion
    >>> task = stadion.get("pricing")
    >>> report = stadion.evaluate(task, stadion.RandomAgent(), instances=5, episodes=5)
    >>> report.vs_baseline.verdict
    'worse'
"""

from stadion.core.agent import Agent, PolicyAgent, RandomAgent, ReferenceAgent
from stadion.core.runner import Arms, evaluate, play_episode, play_instance, reproducible
from stadion.core.score import Comparison, Interval, Report, Verdict, paired_bootstrap
from stadion.core.task import Brief, Choice, Instance, Task, View, get, names, registry

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Arms",
    "Brief",
    "Choice",
    "Comparison",
    "Instance",
    "Interval",
    "PolicyAgent",
    "RandomAgent",
    "ReferenceAgent",
    "Report",
    "Task",
    "Verdict",
    "View",
    "__version__",
    "evaluate",
    "get",
    "names",
    "paired_bootstrap",
    "play_episode",
    "play_instance",
    "registry",
    "reproducible",
]
