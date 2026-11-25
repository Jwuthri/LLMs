"""Each number shown in content/rl/02-rewards.md."""

import numpy as np
import pytest

from examples.rl.rewards import (
    GROUP, P_SUCCESS, P_TOKEN, PARTIAL_DIST, SPARSE_DIST, checker, credit_errors,
    length_penalty_returns, noisy_checker_stats, outcome_step_rewards, p_any_success, p_signal,
    reward_to_go, rubric_score, safe_advantages, shaped_step_rewards, token_signals,
)
from examples.rl.toy_rl import group_advantages, partial_reward, process_rewards, sparse_reward


def close(a, b):
    return np.allclose(a, b, atol=5e-4)


def test_success_probability():
    assert close(P_TOKEN, [0.099, 0.242, 0.659])
    assert round(P_SUCCESS, 4) == 0.0157


@pytest.mark.parametrize("g, any_one, partial", [(4, 0.062, 0.904), (8, 0.119, 0.992),
                                                  (16, 0.224, 1.000), (64, 0.638, 1.000)])
def test_signal_vs_group_size(g, any_one, partial):
    assert round(p_any_success(g), 3) == any_one
    assert round(p_signal(SPARSE_DIST, g), 3) == any_one
    assert round(p_signal(PARTIAL_DIST, g), 3) == partial


def test_no_signal_share():
    assert round(1 - p_any_success(4), 3) == 0.938  # "about 94% of the groups give no signal"


def test_partial_distribution():
    assert close(PARTIAL_DIST, [0.233, 0.550, 0.201, 0.016])
    assert round(float(PARTIAL_DIST @ np.arange(4) / 3), 3) == 0.333


def test_group_advantages():
    assert close(safe_advantages(np.zeros(4)), [0, 0, 0, 0])
    sparse = [sparse_reward(a) for a in GROUP]
    partial = [partial_reward(a) for a in GROUP]
    assert close(group_advantages(sparse), [1.732, -0.577, -0.577, -0.577])
    assert close(group_advantages(partial), [1.414, 0, 0, -1.414])


def test_reward_to_go():
    outcome = [reward_to_go(outcome_step_rewards(a)) for a in GROUP]
    process = [reward_to_go(process_rewards(a)) for a in GROUP]
    assert close(outcome, [[1, 1, 1], [0.667] * 3, [0.667] * 3, [0.333] * 3])
    assert close(process, [[1, 0.667, 0.333], [0.667, 0.667, 0.333], [0.667, 0.333, 0.333], [0.333] * 3])


def test_token_signals_and_credit():
    outcome, process = token_signals("outcome"), token_signals("process")
    assert close(outcome, [[0.333] * 3, [0] * 3, [0] * 3, [-0.333] * 3])
    assert close(process, [[0.333, 0.167, 0], [0, 0.167, 0], [0, -0.167, 0], [-0.333, -0.167, 0]])
    assert credit_errors(outcome) == {"right_direction": 5, "correct_pushed_down": 1, "wrong_pushed_up": 0, "zero": 6}
    assert credit_errors(process) == {"right_direction": 6, "correct_pushed_down": 0, "wrong_pushed_up": 0, "zero": 6}


def test_checker():
    assert checker("... Answer: the cat sat") == 1.0
    assert checker("... Answer: The cat sat.") == 0.0
    assert checker("the cat sat") == 0.0


def test_noisy_checker():
    s = noisy_checker_stats()
    assert round(s["expected_noisy"], 3) == 0.022
    assert round(s["precision"], 3) == 0.561
    assert round(1 - s["precision"], 3) == 0.439
    assert round(s["gap"], 3) == 0.790
    assert round(s["any_one_g8_true"], 3) == 0.119
    assert round(s["any_one_g8_noisy"], 3) == 0.166


def test_rubric():
    scores = [rubric_score(a) for a in GROUP]
    assert close(scores, [1, 0.75, 0.75, 0.5])
    assert close(group_advantages(scores), [1.414, 0, 0, -1.414])


def test_shaping_keeps_returns():
    shaped = [shaped_step_rewards(a) for a in GROUP]
    assert close(shaped, [[0.333] * 3, [0, 0.333, -0.333], [0.333, 0, -0.333], [0, 0, 0]])
    assert close([s.sum() for s in shaped], [sparse_reward(a) for a in GROUP])


def test_length_penalty():
    assert length_penalty_returns(0.1) == pytest.approx({"the cat sat": 0.7, "the cat": -0.2, "(empty)": 0.0})
    assert length_penalty_returns(0.4) == pytest.approx({"the cat sat": -0.2, "the cat": -0.8, "(empty)": 0.0})
