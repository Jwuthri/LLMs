"""Each number shown in content/scientist/09-ai-safety.md."""

import numpy as np
import pytest

from examples.scientist.ai_safety import (
    AWARENESS, FALSE_BLOCK, MISS, PROBE, READOUT_IGNORES, READOUT_USES, SCORES_REAL, SCORES_TEST,
    UNSAFE_IN_USE, UNSAFE_WHEN_TESTED, ablate, auc, blocked_harmless, decision, guard_win,
    measured_rate, pass_rate, score_interval, success_in,
)
from examples.toy import E


def test_small_test_cannot_rule_out_the_threshold():
    p, se, low, high = score_interval(7, 20)
    assert (round(p, 3), round(se, 3), round(low, 3), round(high, 3)) == (0.35, 0.107, 0.141, 0.559)
    assert decision(7, 20) == "stronger safeguards"


def test_larger_test_is_below_the_threshold():
    p, se, low, high = score_interval(70, 200)
    assert (round(p, 3), round(se, 3), round(low, 3), round(high, 3)) == (0.35, 0.034, 0.284, 0.416)
    assert decision(70, 200) == "current safeguards"


def test_layers_multiply():
    assert [round(pass_rate(MISS[:k]), 3) for k in (1, 2, 3)] == [0.2, 0.06, 0.006]
    assert round(1000 * pass_rate(MISS)) == 6
    assert round(min(MISS) / pass_rate(MISS), 1) == 16.7  # fully correlated layers


def test_harmless_requests_lost():
    assert round(blocked_harmless(FALSE_BLOCK), 4) == 0.0199


@pytest.mark.parametrize("attempts, expected", [(1, 0.006), (10, 0.058), (100, 0.452), (500, 0.951)])
def test_many_attempts(attempts, expected):
    assert round(success_in(attempts, pass_rate(MISS)), 3) == expected


def test_test_understates_the_rate_in_use():
    assert (UNSAFE_WHEN_TESTED, UNSAFE_IN_USE) == (0.05, 0.25)
    assert round(UNSAFE_IN_USE - UNSAFE_WHEN_TESTED, 2) == 0.20
    assert round(measured_rate(AWARENESS), 3) == 0.09
    assert round(UNSAFE_IN_USE - measured_rate(AWARENESS), 2) == 0.16
    assert round(UNSAFE_IN_USE / measured_rate(AWARENESS), 2) == 2.78
    assert measured_rate(0.0) == 0.25 and measured_rate(1.0) == 0.05


def test_auc_of_the_toy_transcripts():
    assert round(auc(SCORES_TEST, SCORES_REAL), 3) == 0.889  # 8 of 9 pairs
    assert auc([1.0], [0.0]) == 1.0 and auc([0.5], [0.5]) == 0.5


def test_guard_win_probability():
    assert [round(guard_win(gap), 3) for gap in (0, 200, 400, 800)] == [0.5, 0.24, 0.091, 0.01]


def test_probe_and_ablation():
    assert np.allclose(E @ PROBE, [0, 1, 1])
    sat = E[2]
    assert np.allclose(ablate(sat, PROBE), [1, 0])
    assert (sat @ READOUT_USES, ablate(sat, PROBE) @ READOUT_USES) == (2.0, 1.0)
    assert (sat @ READOUT_IGNORES, ablate(sat, PROBE) @ READOUT_IGNORES) == (1.0, 1.0)
