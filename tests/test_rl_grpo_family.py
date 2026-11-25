"""Each number shown in content/rl/04-grpo-family.md."""

import numpy as np

from examples.rl.grpo_family import (
    ADVANTAGES,
    ALL_WRONG,
    CISPO_HIGH,
    EASY_PROMPT,
    EPS,
    EPS_HIGH,
    EPS_LOW,
    L_CACHE,
    L_MAX,
    LENGTHS,
    MEDIUM_PROMPT,
    NEW_LOGITS,
    REWARDS,
    cispo_weights,
    clipped_term,
    dapo_token_weights,
    dr_grpo_advantages,
    dr_grpo_token_weights,
    grpo_gradient,
    grpo_objective,
    grpo_token_weights,
    gspo_term,
    max_new_probability,
    overlong_penalty,
    safe_advantages,
    sequence_ratio,
    token_ratios,
)
from examples.rl.toy_rl import (
    GROUP,
    POLICY_LOGITS,
    TARGET,
    group_advantages,
    sparse_reward,
)
from examples.toy import softmax

ATOL = 5e-4
CORRECT, WRONG_A, WRONG_B, WRONG_C = GROUP   # "the cat sat", "sat cat sat", "the sat sat", "sat sat sat"


def test_grpo_group_advantage_and_update():
    assert np.allclose(REWARDS, [1, 0, 0, 0])
    assert np.allclose(ADVANTAGES, [1.732, -0.577, -0.577, -0.577], atol=ATOL)
    assert round(REWARDS.std(), 3) == 0.433
    grad = grpo_gradient()
    assert np.allclose(grad, [[0.096, 0, -0.096], [0, 0.096, -0.096], [0, 0, 0]], atol=ATOL)
    probs = softmax(NEW_LOGITS)
    assert np.allclose(probs, [[0.128, 0.265, 0.606], [0.105, 0.306, 0.589], [0.099, 0.242, 0.659]], atol=ATOL)
    assert round(grpo_objective(new_logits=POLICY_LOGITS), 3) == 0.0


def test_grpo_ratios_and_clip_after_the_update():
    ratios = token_ratios(CORRECT)
    assert np.allclose(ratios, [1.301, 1.264, 1.0], atol=ATOL)
    term, weight = clipped_term(ratios, ADVANTAGES[0])
    assert np.allclose(term, [2.078, 2.078, 1.732], atol=ATOL)
    assert round(term.mean(), 3) == 1.963
    assert np.allclose(weight, [0, 0, 1.732], atol=ATOL)
    ratios = token_ratios(WRONG_B)
    assert np.allclose(ratios, [1.301, 0.894, 1.0], atol=ATOL)
    term, weight = clipped_term(ratios, ADVANTAGES[2])
    assert np.allclose(term, [-0.751, -0.516, -0.577], atol=ATOL)
    assert np.allclose(weight, term)
    assert round(grpo_objective(), 3) == 0.048


def test_dr_grpo_length_bias():
    assert list(LENGTHS) == [3, 3, 3, 6]
    weights = grpo_token_weights()
    assert np.allclose(weights, [0.144, -0.048, -0.048, -0.024], atol=ATOL)
    assert np.allclose(weights * LENGTHS, [0.433, -0.144, -0.144, -0.144], atol=ATOL)
    assert np.allclose(dr_grpo_advantages(REWARDS), [0.75, -0.25, -0.25, -0.25])
    weights = dr_grpo_token_weights()
    assert np.allclose(weights, [0.031, -0.010, -0.010, -0.010], atol=ATOL)
    assert np.allclose(weights * LENGTHS, [0.094, -0.031, -0.031, -0.063], atol=ATOL)


def test_dr_grpo_difficulty_bias():
    assert round(1 / REWARDS.std(), 3) == 2.309
    assert round(EASY_PROMPT.std(), 3) == 0.331
    assert round(1 / EASY_PROMPT.std(), 3) == 3.024
    assert round(MEDIUM_PROMPT.std(), 3) == 0.5
    assert round(1 / MEDIUM_PROMPT.std(), 3) == 2.0
    assert np.allclose(group_advantages(EASY_PROMPT)[[0, -1]], [0.378, -2.646], atol=ATOL)
    assert np.allclose(group_advantages(MEDIUM_PROMPT)[[0, -1]], [1.0, -1.0], atol=ATOL)
    assert np.allclose(dr_grpo_advantages(EASY_PROMPT)[[0, -1]], [0.125, -0.875])
    assert np.allclose(dr_grpo_advantages(MEDIUM_PROMPT)[[0, -1]], [0.5, -0.5])


def test_dapo_clip_higher():
    p_old = softmax(POLICY_LOGITS)[0]
    assert (EPS_LOW, EPS_HIGH) == (0.2, 0.28)
    assert round(p_old[0], 3) == 0.099
    assert round(max_new_probability(p_old[0], EPS), 3) == 0.118
    assert round(max_new_probability(p_old[0], EPS_HIGH), 3) == 0.126
    assert round(p_old[2], 3) == 0.659
    assert round(max_new_probability(p_old[2], EPS), 3) == 0.791
    assert round(max_new_probability(p_old[2], EPS_HIGH), 3) == 0.844
    assert round(p_old[0] * (1 - EPS_LOW), 3) == 0.079


def test_dapo_dynamic_sampling_and_token_level_loss():
    wrong = [sparse_reward(answer) for answer in ALL_WRONG]
    assert np.allclose(safe_advantages(wrong), [0, 0, 0, 0])
    assert round(1 / 12, 3) == 0.083 and round(1 / 24, 3) == 0.042
    weights = dapo_token_weights()
    assert np.allclose(weights, [0.115, -0.038, -0.038, -0.038], atol=ATOL)
    assert np.allclose(weights * LENGTHS, [0.346, -0.115, -0.115, -0.231], atol=ATOL)
    assert np.allclose(LENGTHS / LENGTHS.sum(), [0.2, 0.2, 0.2, 0.4])


def test_dapo_overlong_shaping():
    assert (L_MAX, L_CACHE) == (7, 2)
    assert [overlong_penalty(n, L_MAX, L_CACHE) for n in (3, 5, 6, 7, 8)] == [0, 0, -0.5, -1, -1]
    assert overlong_penalty(18432, 16384 + 4096, 4096) == -0.5


def test_gspo_sequence_ratio():
    log_ratios = np.log(token_ratios(CORRECT))
    assert np.allclose(log_ratios, [0.263, 0.234, 0.0], atol=ATOL)
    assert round(sequence_ratio(CORRECT), 3) == 1.180
    assert np.allclose(gspo_term(CORRECT, ADVANTAGES[0]), [1.733, 0.0], atol=ATOL)
    assert round(sequence_ratio(WRONG_B), 3) == 1.052
    assert np.allclose(gspo_term(WRONG_B, ADVANTAGES[2]), [-0.607, -0.607], atol=ATOL)
    assert round(sequence_ratio(WRONG_C), 3) == 0.937
    assert np.allclose(gspo_term(WRONG_C, ADVANTAGES[3]), [-0.577, 0.0], atol=ATOL)


def test_dapo_and_cispo_weights_after_the_update():
    assert CISPO_HIGH == 0.28
    ratios = token_ratios(CORRECT)
    assert np.allclose(clipped_term(ratios, ADVANTAGES[0], EPS_LOW, EPS_HIGH)[1], [0, 2.189, 1.732], atol=ATOL)
    assert np.allclose(np.minimum(ratios, 1 + CISPO_HIGH), [1.28, 1.264, 1.0], atol=ATOL)
    assert np.allclose(cispo_weights(ratios, ADVANTAGES[0]), [2.217, 2.189, 1.732], atol=ATOL)
    ratios = token_ratios(WRONG_B)
    assert np.allclose(cispo_weights(ratios, ADVANTAGES[2]), [-0.739, -0.516, -0.577], atol=ATOL)
    assert TARGET == CORRECT and WRONG_A == ("sat", "cat", "sat")


def test_derived_numbers_in_the_text():
    p_old = softmax(POLICY_LOGITS)[0]
    assert round(REWARDS.mean(), 3) == 0.25
    assert round(EASY_PROMPT.std() ** -1 / MEDIUM_PROMPT.std() ** -1, 2) == 1.51
    assert round(p_old[0] * EPS, 3) == 0.020 and round(p_old[2] * EPS, 3) == 0.132
    assert sparse_reward(("sat",) * 3) + overlong_penalty(6, L_MAX, L_CACHE) == -0.5
    assert round(1.28 * ADVANTAGES[2], 3) == -0.739
