"""The scoring rules, including the outcome most benchmarks do not have a word for."""

from __future__ import annotations

import numpy as np
import pytest

from stadion.core.score import Comparison, Interval, Report, paired_bootstrap


def _comparison(delta: float, low: float, high: float) -> Comparison:
    verdict = "better" if low > 0 else "worse" if high < 0 else "indistinguishable"
    return Comparison(
        label="x - y",
        mean_a=0.0,
        mean_b=0.0,
        delta=delta,
        ci=Interval(low, high),
        verdict=verdict,  # type: ignore[arg-type]
        n=10,
    )


def _report(agent: float, baseline: float, optimal: float, headroom_spans_zero: bool) -> Report:
    span = optimal - baseline
    headroom = (
        _comparison(span, -abs(span) - 1.0, abs(span) + 1.0)
        if headroom_spans_zero
        else _comparison(span, span * 0.5, span * 1.5)
    )
    return Report(
        task="t",
        agent="a",
        instances=10,
        episode_seeds=(0,),
        agent_return=agent,
        baseline_return=baseline,
        optimal_return=optimal,
        vs_baseline=_comparison(agent - baseline, -1.0, 1.0),
        vs_optimal=_comparison(agent - optimal, -1.0, 1.0),
        headroom=headroom,
    )


def test_verdict_is_better_only_when_the_whole_interval_is_positive() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(10.0, 1.0, size=200)
    result = paired_bootstrap(a + 2.0, a)
    assert result.verdict == "better"
    assert result.ci.low > 0


def test_verdict_is_worse_only_when_the_whole_interval_is_negative() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(10.0, 1.0, size=200)
    result = paired_bootstrap(a - 2.0, a)
    assert result.verdict == "worse"
    assert result.ci.high < 0


def test_a_difference_smaller_than_the_noise_is_called_indistinguishable() -> None:
    """The outcome this library exists to be able to report."""
    rng = np.random.default_rng(2)
    a = rng.normal(10.0, 3.0, size=30)
    b = a + rng.normal(0.0, 3.0, size=30)  # same distribution, unrelated draw
    result = paired_bootstrap(a, b)
    assert result.verdict == "indistinguishable"
    assert result.ci.low < 0 < result.ci.high


def test_the_interval_is_reproducible_for_a_fixed_bootstrap_seed() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=50), rng.normal(size=50)
    first = paired_bootstrap(a, b, seed=7)
    second = paired_bootstrap(a, b, seed=7)
    assert (first.ci.low, first.ci.high) == (second.ci.low, second.ci.high)


def test_mismatched_arms_are_refused_rather_than_broadcast() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        paired_bootstrap([1.0, 2.0, 3.0], [1.0, 2.0])


def test_a_single_instance_cannot_produce_an_interval() -> None:
    with pytest.raises(ValueError, match="at least 2 instances"):
        paired_bootstrap([1.0], [2.0])


def test_score_is_zero_at_the_classical_method_and_one_at_the_optimum() -> None:
    assert _report(10.0, 10.0, 20.0, False).score == pytest.approx(0.0)
    assert _report(20.0, 10.0, 20.0, False).score == pytest.approx(1.0)
    assert _report(15.0, 10.0, 20.0, False).score == pytest.approx(0.5)


def test_score_is_undefined_when_the_classical_method_matches_the_optimum() -> None:
    """No headroom means no denominator, and saying so beats dividing by noise."""
    report = _report(10.0, 10.0, 10.02, headroom_spans_zero=True)
    assert report.degenerate
    assert report.score is None
    assert "undefined" in report.summary()


def test_summary_states_what_one_point_of_score_is_worth() -> None:
    """Without it, a 0.3% headroom turns an ordinary loss into a score of -116."""
    report = _report(9.0, 10.0, 10.1, headroom_spans_zero=False)
    assert report.point_value == pytest.approx(0.1)
    assert "one point" in report.summary()
