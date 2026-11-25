"""Each number shown in content/rl/03-ppo.md."""

import numpy as np

from examples.rl.ppo import (
    ANSWER, CRITIC_GUESS, EPSILON, GREEDY, STEP_SIZE,
    answer_advantages, clipped_terms, first_token_stats, gae, group_rewards, policy_update,
    remax_advantages, rloo_advantages, rloo_baselines, target_probs, td_errors, token_ratios,
    token_rewards, true_values, value_loss,
)
from examples.rl.toy_rl import TARGET, group_advantages, partial_reward
from examples.toy import softmax


def r3(x):
    return np.round(np.asarray(x, dtype=float), 3).tolist()


def test_group_rewards():
    r = group_rewards()
    assert r3(r) == [1.0, 0.667, 0.667, 0.333]
    assert round(r.mean(), 3) == 0.667
    assert round(r.sum(), 3) == 2.667


def test_rloo():
    r = group_rewards()
    assert r3(rloo_baselines(r)) == [0.556, 0.667, 0.667, 0.778]
    assert r3(rloo_advantages(r)) == [0.444, 0.0, 0.0, -0.444]
    assert np.allclose(rloo_advantages(r), 4 / 3 * (r - r.mean()))
    assert r3(r - r.mean()) == [0.333, 0.0, 0.0, -0.333]


def test_grpo_pointer():
    assert r3(group_advantages(group_rewards())) == [1.414, 0.0, 0.0, -1.414]


def test_remax():
    assert GREEDY == ("sat", "sat", "sat")
    assert round(partial_reward(GREEDY), 3) == 0.333
    assert r3(remax_advantages(group_rewards())) == [0.667, 0.333, 0.333, 0.0]


def test_values_and_td_errors():
    assert ANSWER == ("sat", "cat", "sat")
    assert r3(target_probs()) == [0.099, 0.242, 0.659]
    assert r3(true_values(ANSWER)) == [0.333, 0.3, 0.553]
    assert r3(token_rewards(ANSWER)) == [0.0, 0.0, 0.667]
    assert r3(td_errors(token_rewards(ANSWER), true_values(ANSWER))) == [-0.033, 0.253, 0.114]
    assert r3(CRITIC_GUESS) == [0.4, 0.45, 0.5]
    assert r3(td_errors(token_rewards(ANSWER), CRITIC_GUESS)) == [0.05, 0.05, 0.167]


def test_gae_true_values():
    assert r3(answer_advantages(ANSWER, 0.0)) == [-0.033, 0.253, 0.114]
    assert r3(answer_advantages(ANSWER, 0.95)) == [0.31, 0.361, 0.114]
    assert r3(answer_advantages(ANSWER, 1.0)) == [0.333, 0.366, 0.114]
    # lambda = 1 is the return minus V
    assert np.allclose(answer_advantages(ANSWER, 1.0), partial_reward(ANSWER) - true_values(ANSWER))


def test_gae_critic_guess():
    assert r3(answer_advantages(ANSWER, 0.0, CRITIC_GUESS)) == [0.05, 0.05, 0.167]
    assert r3(answer_advantages(ANSWER, 0.95, CRITIC_GUESS)) == [0.248, 0.208, 0.167]
    assert r3(answer_advantages(ANSWER, 1.0, CRITIC_GUESS)) == [0.267, 0.217, 0.167]


def test_gae_recursion():
    assert np.allclose(gae(np.array([1.0, 1.0]), 1.0, 0.5), [1.5, 1.0])


def test_bias_variance_of_first_token():
    expected = {
        ("true", 0.0): (0.333, 0.0), ("true", 0.95): (0.333, 0.197), ("true", 1.0): (0.333, 0.213),
        ("guess", 0.0): (0.0, 0.0), ("guess", 0.95): (0.301, 0.192), ("guess", 1.0): (0.333, 0.213),
    }
    for (critic, lam), (gap, std) in expected.items():
        got_gap, got_std = first_token_stats(lam, critic)
        assert round(got_gap, 3) == gap and round(got_std, 3) == std, (critic, lam)


def test_ppo_update_and_clip():
    assert STEP_SIZE == 3.0 and EPSILON == 0.2
    new_logits = policy_update()
    new = softmax(new_logits)
    assert r3(new[0]) == [0.132, 0.243, 0.625]
    assert r3(new[1]) == [0.101, 0.288, 0.611]
    assert r3(new[2]) == [0.096, 0.232, 0.673]

    assert r3(td_errors(token_rewards(TARGET), true_values(TARGET))) == [0.3, 0.253, 0.114]
    adv = answer_advantages(TARGET, 0.95)
    rho = token_ratios(TARGET, new_logits)
    assert r3(adv) == [0.643, 0.361, 0.114]
    assert r3(rho) == [1.338, 1.188, 1.021]
    assert r3(rho * adv) == [0.861, 0.428, 0.116]
    assert r3(clipped_terms(rho, adv)) == [0.772, 0.428, 0.116]

    adv = answer_advantages(ANSWER, 0.95)
    rho = token_ratios(ANSWER, new_logits)
    assert r3(rho) == [0.948, 1.188, 1.021]
    assert r3(clipped_terms(rho, adv)) == [0.294, 0.428, 0.116]


def test_value_loss():
    returns = np.full(3, partial_reward(ANSWER))
    assert r3(partial_reward(ANSWER) - CRITIC_GUESS) == [0.267, 0.217, 0.167]
    assert round(value_loss(CRITIC_GUESS, returns), 3) == 0.049
