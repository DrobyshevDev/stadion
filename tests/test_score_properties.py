"""Invariants of the paired bootstrap, over arbitrary arms rather than chosen ones.

test_score.py picks inputs that make each rule visible: two arms two units apart
are "better", two draws from the same distribution are "indistinguishable". That
is the right way to show what the rules mean, and it cannot show that they hold
everywhere — every one of those tests would still pass if the interval were
subtly wrong on inputs nobody thought to write down.

These state the properties instead, and let Hypothesis look for the arms that
break them. The properties are the ones the scoring argument actually rests on:
a verdict is a statement about the interval and nothing else; the comparison
does not care which arm is named first; and shifting every return by a constant
shifts the difference by that constant, because a benchmark whose answer depends
on where you put the zero of the return scale is not measuring the agent.

`hypothesis` has been in this project's dev dependencies since the start and was
never imported. That is the gap this file closes.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from stadion.core.score import paired_bootstrap

# Bounded and finite: an interval over values spanning 300 orders of magnitude
# says nothing about the scoring rule and everything about float64. The bound is
# wide enough to cover every return these tasks produce.
returns = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


def arms(min_size: int = 2, max_size: int = 40):
    return st.lists(returns, min_size=min_size, max_size=max_size)


def paired(draw_size: int = 40):
    """Two arms of equal length, which is what pairing requires."""
    return st.integers(min_value=2, max_value=draw_size).flatmap(
        lambda n: st.tuples(
            st.lists(returns, min_size=n, max_size=n),
            st.lists(returns, min_size=n, max_size=n),
        )
    )


# 200 resamples is plenty to exercise the code path; the default 10 000 is about
# the tightness of a published interval, which is not what these tests are for.
FAST = {"resamples": 200}


def _tolerance(*arms_and_shifts) -> float:
    """An absolute tolerance scaled to the magnitude of the values involved.

    Summing values near 1e6 and differencing the sums loses absolute precision
    in proportion to those values: float64 carries about sixteen digits, so a
    difference of two numbers near 1e8 is exact to roughly 1e-8, not to 1e-9.
    Hypothesis found this immediately -- arms [0, 0, 93282] and [0, 1, 93281]
    have a mean difference of exactly zero, and scaling them by 719 leaves a
    residue of 2e-9 that is arithmetic, not a broken property.
    """
    magnitude = 1.0
    for item in arms_and_shifts:
        values = item if isinstance(item, list) else [item]
        for value in values:
            magnitude = max(magnitude, abs(float(value)))
    return 1e-12 * magnitude


@given(arms())
@settings(max_examples=60, deadline=None)
def test_an_arm_compared_with_itself_has_no_difference_and_no_interval(values):
    """Not a special case in the code — it falls out of pairing, and must."""
    result = paired_bootstrap(values, values, **FAST)
    assert result.delta == 0.0
    assert (result.ci.low, result.ci.high) == (0.0, 0.0)
    assert result.verdict == "indistinguishable"


@given(paired())
@settings(max_examples=60, deadline=None)
def test_the_verdict_says_exactly_what_the_interval_says(both):
    a, b = both
    result = paired_bootstrap(a, b, **FAST)
    assert result.ci.low <= result.ci.high
    if result.ci.low > 0.0:
        assert result.verdict == "better"
    elif result.ci.high < 0.0:
        assert result.verdict == "worse"
    else:
        assert result.verdict == "indistinguishable"


@given(paired())
@settings(max_examples=60, deadline=None)
def test_swapping_the_arms_negates_the_comparison(both):
    """Which arm is named first is a labelling choice, not a measurement."""
    a, b = both
    forward = paired_bootstrap(a, b, **FAST)
    backward = paired_bootstrap(b, a, **FAST)

    assert forward.delta == pytest.approx(-backward.delta, rel=1e-9, abs=1e-9)
    assert forward.ci.low == pytest.approx(-backward.ci.high, rel=1e-9, abs=1e-9)
    assert forward.ci.high == pytest.approx(-backward.ci.low, rel=1e-9, abs=1e-9)

    opposite = {"better": "worse", "worse": "better", "indistinguishable": "indistinguishable"}
    assert forward.verdict == opposite[backward.verdict]


@given(paired(), st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False))
@settings(max_examples=60, deadline=None)
def test_shifting_one_arm_shifts_the_difference_by_the_same_amount(both, shift):
    """A benchmark whose answer moves with the zero of the return scale is broken."""
    a, b = both
    base = paired_bootstrap(a, b, **FAST)
    shifted = paired_bootstrap([value + shift for value in a], b, **FAST)

    tol = _tolerance(a, b, abs(shift))
    assert shifted.delta == pytest.approx(base.delta + shift, rel=1e-9, abs=tol)
    assert shifted.ci.low == pytest.approx(base.ci.low + shift, rel=1e-9, abs=tol)
    assert shifted.ci.high == pytest.approx(base.ci.high + shift, rel=1e-9, abs=tol)


@given(paired(), st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=60, deadline=None)
def test_rescaling_both_arms_rescales_the_interval_and_keeps_the_verdict(both, factor):
    """Measuring in cents rather than euros cannot change who won."""
    a, b = both
    base = paired_bootstrap(a, b, **FAST)
    scaled = paired_bootstrap(
        [value * factor for value in a], [value * factor for value in b], **FAST
    )
    tol = _tolerance(a, b) * factor
    assert scaled.delta == pytest.approx(base.delta * factor, rel=1e-9, abs=tol)

    # The verdict is a question about which side of zero an interval sits on, so
    # it is only stable when the interval is further from zero than the noise of
    # the arithmetic that produced it. Hypothesis is very good at finding arms
    # where it is not, and those say nothing about the scoring rule.
    assume(min(abs(base.ci.low), abs(base.ci.high)) > tol)
    assert scaled.verdict == base.verdict


@given(paired())
@settings(max_examples=40, deadline=None)
def test_the_reported_arm_means_are_the_arm_means(both):
    a, b = both
    result = paired_bootstrap(a, b, **FAST)
    assert result.mean_a == pytest.approx(float(np.mean(a)), rel=1e-12, abs=1e-12)
    assert result.mean_b == pytest.approx(float(np.mean(b)), rel=1e-12, abs=1e-12)
    assert result.n == len(a)


@given(arms(max_size=20), arms(max_size=20))
@settings(max_examples=40, deadline=None)
def test_arms_of_different_lengths_are_refused_rather_than_broadcast(a, b):
    """NumPy would happily broadcast some of these into a silent wrong answer."""
    assume(len(a) != len(b))
    with pytest.raises(ValueError, match="different lengths"):
        paired_bootstrap(a, b, **FAST)
