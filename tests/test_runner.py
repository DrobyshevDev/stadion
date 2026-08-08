"""The evaluation loop, checked against the two agents whose scores are known in advance.

Playing the task's own reference policies through the ordinary path is the
control: the classical arm must land on exactly 0 and the optimal arm on exactly
1. If either drifts, the harness is measuring something other than what it says.
"""

from __future__ import annotations

import pytest

import stadion
from stadion.core.agent import Agent, ReferenceAgent
from stadion.core.task import View

TASKS = stadion.names()
SMALL = {"instances": 6, "episodes": 6}


class _Illegal(Agent):
    name = "illegal"

    def act(self, view: View) -> int:
        return 10_000


class _First(Agent):
    """Always takes the lowest-numbered choice. Deterministic, and usually bad."""

    name = "first"

    def act(self, view: View) -> int:
        return view.choices[0].value


@pytest.mark.parametrize("task_name", TASKS)
def test_the_classical_method_scores_exactly_zero_against_itself(task_name: str) -> None:
    task = stadion.get(task_name)
    report = stadion.evaluate(task, ReferenceAgent(task, "classical"), **SMALL)
    assert report.score == pytest.approx(0.0, abs=1e-9)
    assert report.vs_baseline.delta == pytest.approx(0.0, abs=1e-9)
    assert report.vs_baseline.verdict == "indistinguishable"


@pytest.mark.parametrize("task_name", TASKS)
def test_the_optimum_scores_exactly_one(task_name: str) -> None:
    task = stadion.get(task_name)
    report = stadion.evaluate(task, ReferenceAgent(task, "optimum"), **SMALL)
    assert report.score == pytest.approx(1.0, abs=1e-9)
    assert report.vs_optimal.verdict == "indistinguishable"


@pytest.mark.parametrize("task_name", TASKS)
def test_every_task_leaves_the_classical_method_short_of_the_optimum(task_name: str) -> None:
    """If the textbook rule were optimal everywhere there would be nothing to measure.

    The headroom is allowed to be small — on ``inventory`` it is a fraction of a
    percent — but it has to be there and it has to be measurable.
    """
    task = stadion.get(task_name)
    report = stadion.evaluate(task, ReferenceAgent(task, "classical"), **SMALL)
    assert report.headroom.verdict == "better", (
        f"{task_name}: the optimum is not measurably above the classical method "
        f"({report.headroom})"
    )


@pytest.mark.parametrize("task_name", TASKS)
def test_a_random_player_loses_to_the_classical_method(task_name: str) -> None:
    task = stadion.get(task_name)
    report = stadion.evaluate(task, stadion.RandomAgent(), **SMALL)
    assert report.vs_baseline.verdict == "worse"


@pytest.mark.parametrize("task_name", TASKS)
def test_an_out_of_range_choice_is_refused_not_clipped(task_name: str) -> None:
    """The environments clip silently; a benchmark that inherited that would score noise."""
    task = stadion.get(task_name)
    with pytest.raises(ValueError, match="not a legal choice"):
        stadion.play_episode(task, _Illegal(), task.instance(0), seed=0)


@pytest.mark.parametrize("task_name", TASKS)
def test_a_deterministic_agent_reproduces_its_own_returns(task_name: str) -> None:
    task = stadion.get(task_name)
    assert stadion.reproducible(task, _First())
    assert stadion.reproducible(task, stadion.RandomAgent())


def test_an_evaluation_needs_enough_instances_for_an_interval() -> None:
    task = stadion.get("queueing")
    with pytest.raises(ValueError, match="at least 2 instances"):
        stadion.evaluate(task, stadion.RandomAgent(), instances=1, episodes=2)


def test_a_reference_agent_rejects_an_unknown_role() -> None:
    with pytest.raises(ValueError, match="classical.*optimum"):
        ReferenceAgent(stadion.get("pricing"), "best")
