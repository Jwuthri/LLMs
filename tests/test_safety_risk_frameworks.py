"""Each number shown in content/safety/01-risk-frameworks.md."""

import pytest

from examples.safety.risk_frameworks import (
    CAN, CLAIMS, SECOND_TEAM, SOLVED_ELICITED, SUCCESS_WITH_MODEL, SUCCESS_WITHOUT_MODEL, TASK_MINUTES,
    TASK_SUCCESS, THREATS, TRIES, chain, decision, expected_harm, horizon_after, score_interval, tasks_needed,
    time_horizon, two_lines, uplift, upper_end,
)


def test_expected_harm_of_the_three_threat_models():
    harms = {name: expected_harm(*values) for name, values in THREATS.items()}
    assert harms == pytest.approx({"misuse": 10.0, "misalignment": 10.0, "malfunction": 1.0})


def test_uplift_is_what_the_model_adds():
    difference, ratio = uplift(SUCCESS_WITH_MODEL, SUCCESS_WITHOUT_MODEL)
    assert (round(difference, 3), round(ratio, 2)) == (0.06, 2.5)


def test_capability_times_propensity():
    assert round(CAN * TRIES, 4) == 0.0175


def test_small_test_cannot_rule_out_the_threshold():
    p, se, low, high = score_interval(7, 20)
    assert (round(p, 3), round(se, 3), round(low, 3), round(high, 3)) == (0.35, 0.107, 0.141, 0.559)
    assert decision(7, 20) == "stronger safeguards"


def test_better_elicitation_finds_more():
    p, se, low, high = score_interval(SOLVED_ELICITED, 20)
    assert (round(p, 3), round(se, 3), round(low, 3), round(high, 3)) == (0.55, 0.111, 0.332, 0.768)


def test_time_horizon():
    horizon = time_horizon(TASK_MINUTES, TASK_SUCCESS)
    assert horizon == pytest.approx(32.0)
    assert horizon_after(horizon, 21) == pytest.approx(256.0)


def test_larger_test_is_below_the_threshold():
    p, se, low, high = score_interval(70, 200)
    assert (round(p, 3), round(se, 3), round(low, 3), round(high, 3)) == (0.35, 0.034, 0.284, 0.416)
    assert decision(70, 200) == "current safeguards"


def test_tasks_needed_to_be_below_the_threshold():
    assert tasks_needed(0.35) == 39
    assert upper_end(0.35, 38) >= 0.5 > upper_end(0.35, 39)


def test_safety_case_chain():
    assert round(chain(CLAIMS), 4) == 0.7695
    first = two_lines(CLAIMS[0], SECOND_TEAM)
    assert round(first, 2) == 0.98
    assert round(chain([first, *CLAIMS[1:]]), 3) == 0.838
    assert round(1 - chain([first, *CLAIMS[1:]]), 3) == 0.162
