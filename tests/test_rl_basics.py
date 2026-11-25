"""Each number shown in content/rl/01-rl-basics.md."""

import numpy as np
import pytest

from examples.rl.basics import (
    ALL_ANSWERS, POLICY_LOGITS, discounted_return, page_numbers, simulation_numbers, step_rewards,
)
from examples.rl.toy_rl import GROUP, TARGET, partial_reward, sequence_logprob, token_logprobs


def r3(x):
    return np.round(np.asarray(x, dtype=float), 3).tolist()


@pytest.fixture(scope="module")
def nums():
    return page_numbers()


@pytest.fixture(scope="module")
def sim():
    return simulation_numbers()


def test_example_table():
    assert r3(token_logprobs(TARGET)) == [-2.317, -1.417, -0.417]
    assert [round(sequence_logprob(y), 3) for y in GROUP] == [-4.151, -2.251, -3.151, -1.251]
    assert r3([partial_reward(y) for y in GROUP]) == [1.0, 0.667, 0.667, 0.333]


def test_text_generation_as_rl(nums):
    assert r3(nums["p_target_tokens"]) == [0.099, 0.242, 0.659]
    assert round(nums["p_target"], 4) == 0.0157
    assert round(sequence_logprob(TARGET), 3) == -4.151   # SFT loss 4.151
    assert step_rewards(("sat", "cat", "sat"), partial_reward)[:2] == [0.0, 0.0]
    assert round(step_rewards(("sat", "cat", "sat"), partial_reward)[2], 3) == 0.667


def test_return_and_objective(nums):
    assert step_rewards(TARGET, partial_reward) == [0.0, 0.0, 1.0]
    assert nums["return_target"] == 1.0
    assert round(nums["return_target_gamma09"], 3) == 0.81
    assert nums["n_answers"] == len(ALL_ANSWERS) == 27
    assert round(nums["sum_probs"], 6) == 1.0
    assert round(nums["e_sparse"], 4) == 0.0157
    assert round(1 / nums["e_sparse"]) == 64                # "about once in 64 tries"
    assert r3(nums["p_correct_count"]) == [0.233, 0.55, 0.201, 0.016]
    assert round(nums["e_partial"], 3) == 0.333
    assert round(float(np.mean(nums["p_target_tokens"])), 3) == 0.333
    assert round(nums["e_partial"] / nums["e_sparse"]) == 21
    assert round(nums["group_mean"], 3) == 0.667            # group estimate, twice the true value


def test_reinforce_step(nums):
    assert r3(nums["score_target"]) == [[0.901, -0.242, -0.659], [-0.099, 0.758, -0.659], [-0.099, -0.242, 0.341]]
    assert round(sum(partial_reward(y) for y in GROUP), 3) == 2.667
    assert r3(nums["grad_b0"]) == [[0.351, -0.162, -0.189], [-0.066, 0.255, -0.189], [-0.066, -0.162, 0.227]]
    new = nums["new_probs"]
    assert r3([new[0, 0], new[1, 1], new[2, 2]]) == [0.157, 0.329, 0.735]
    assert round(nums["new_p_target"], 3) == 0.038
    assert round(nums["new_e_sparse"], 3) == 0.038
    assert round(nums["new_e_partial"], 3) == 0.407
    assert round(nums["new_p_target"] / nums["p_target"], 1) == 2.4


def test_exact_gradient(nums):
    assert r3(nums["exact_grad_partial"][0]) == [0.03, -0.008, -0.022]
    assert nums["grad_b0"][0, 0] / nums["exact_grad_partial"][0, 0] > 10


def test_baseline_and_variance(nums):
    assert np.allclose(nums["mean_score"], 0, atol=1e-12)
    assert round(nums["var_b0"], 3) == 0.253
    assert round(nums["var_bmean"], 3) == 0.087
    assert round(nums["var_b0"] / nums["var_bmean"], 1) == 2.9
    assert round(nums["b_opt"], 3) == 0.333
    assert np.isclose(nums["b_opt"], nums["e_partial"])
    assert round(nums["var_sparse_b0"], 4) == 0.0389
    assert round(nums["var_sparse_bmean"], 4) == 0.0380
    assert round(1 - nums["e_sparse"], 3) == 0.984


def test_group_mean_baseline(nums):
    assert r3([partial_reward(y) - nums["group_mean"] for y in GROUP]) == [0.333, 0.0, 0.0, -0.333]
    assert r3(nums["grad_group_mean_b"]) == [[0.083, 0, -0.083], [0, 0.083, -0.083], [0, 0, 0]]
    assert round(nums["new_p_target_mean_b"], 3) == 0.020
    assert round(nums["new_e_partial_mean_b"], 3) == 0.348


def test_simulated_training(sim):
    assert round(sim["b0"]["mean"], 3) == 0.747
    assert round(sim["bmean"]["mean"], 3) == 0.790
    assert round(sim["b0"]["std"], 3) == 0.114
    assert round(sim["bmean"]["std"], 3) == 0.058
    assert round(sim["b0"]["p10"], 3) == 0.616
    assert round(sim["bmean"]["p10"], 3) == 0.701


def test_policy_is_unchanged_by_the_script():
    assert np.allclose(POLICY_LOGITS, [[0.1, 1.0, 2.0]] * 3)
    assert discounted_return([0.0, 0.0, 1.0], gamma=0.9) == pytest.approx(0.81)
