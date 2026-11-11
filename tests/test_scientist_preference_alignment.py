"""Each number shown in content/scientist/05-preference-alignment.md."""

import numpy as np

from examples.scientist.preference_alignment import (
    CHOSEN, REF_LOGITS, REWARD_CHOSEN, REWARD_REJECTED, SCORES,
    bradley_terry, dpo_grad, dpo_loss, dpo_step, dpo_terms, grpo_advantages, pick_pair,
    policy_gradient, ppo_term, reward_step, sample_answers,
)
from examples.toy import LOGITS, VOCAB, softmax


def sampled():
    answers = sample_answers(softmax(LOGITS))
    return answers, [SCORES[a] for a in answers]


def test_policy_and_reference_probabilities():
    assert np.allclose(softmax(LOGITS), [0.099, 0.242, 0.659], atol=5e-4)
    assert np.allclose(softmax(REF_LOGITS), [0.186, 0.307, 0.506], atol=5e-4)


def test_rejection_sampling_pair():
    answers, scores = sampled()
    assert answers == ["sat", "cat", "the", "sat"]
    assert scores == [0.9, 0.4, 0.1, 0.9]
    assert pick_pair(answers, scores) == ("sat", "the")


def test_dpo_loss_now():
    terms = dpo_terms(LOGITS)
    assert round(terms["logp_chosen"], 3) == -0.417 and round(terms["ref_logp_chosen"], 3) == -0.680
    assert round(terms["logp_rejected"], 3) == -2.317 and round(terms["ref_logp_rejected"], 3) == -1.680
    assert round(terms["ratio_chosen"], 3) == 0.263 and round(terms["ratio_rejected"], 3) == -0.637
    assert np.isclose(terms["margin"], 0.9)
    assert round(terms["sigmoid"], 3) == 0.611
    assert round(terms["loss"], 3) == 0.493


def test_log_ratios_of_the_minimal_code():
    ratio = np.log(softmax(LOGITS)) - np.log(softmax(REF_LOGITS))
    assert np.allclose(ratio, [-0.637, -0.237, 0.263], atol=5e-4)


def test_dpo_loss_at_the_start_is_ln_2():
    terms = dpo_terms(REF_LOGITS)
    assert terms["margin"] == 0 and round(terms["loss"], 3) == 0.693


def test_dpo_step_raises_the_chosen_token():
    assert np.allclose(dpo_grad(LOGITS), [0.195, 0, -0.195], atol=5e-4)
    new_logits = dpo_step(LOGITS)
    assert np.allclose(new_logits, [-0.095, 1.0, 2.195], atol=5e-4)
    assert np.allclose(softmax(new_logits), [0.072, 0.216, 0.712], atol=5e-4)
    terms = dpo_terms(new_logits)
    assert round(terms["margin"], 3) == 1.289
    assert round(terms["sigmoid"], 3) == 0.656
    assert round(terms["loss"], 3) == 0.422


def test_dpo_beta():
    assert round(dpo_terms(LOGITS, beta=0.1)["loss"], 3) == 0.649
    assert dpo_loss(0.9, 0.1) > dpo_loss(0.9, 0.5)


def test_reward_model_bradley_terry():
    p_win, loss = bradley_terry(REWARD_CHOSEN, REWARD_REJECTED)
    assert round(p_win, 3) == 0.690 and round(loss, 3) == 0.371
    r_chosen, r_rejected = reward_step(REWARD_CHOSEN, REWARD_REJECTED)
    assert round(r_chosen, 3) == 1.510 and round(r_rejected, 3) == 0.090
    p_win, loss = bradley_terry(r_chosen, r_rejected)
    assert round(p_win, 3) == 0.805 and round(loss, 3) == 0.216


def test_grpo_advantages():
    _, scores = sampled()
    rewards = np.array(scores)
    assert np.isclose(rewards.mean(), 0.575) and round(float(rewards.std()), 3) == 0.342
    advantages = grpo_advantages(rewards)
    assert np.allclose(advantages, [0.951, -0.512, -1.389, 0.951], atol=5e-4)
    assert np.isclose(advantages.sum(), 0)


def test_policy_gradient_step():
    answers, scores = sampled()
    probs = softmax(LOGITS)
    actions = [VOCAB.index(a) for a in answers]
    grad = policy_gradient(probs, actions, grpo_advantages(np.array(scores)))
    assert np.allclose(grad, [-0.347, -0.128, 0.475], atol=5e-4)
    assert np.allclose(LOGITS + grad, [-0.247, 0.872, 2.475], atol=5e-4)
    assert np.allclose(softmax(LOGITS + grad), [0.052, 0.159, 0.789], atol=5e-4)
    assert np.allclose(softmax(LOGITS + 2 * grad), [0.025, 0.097, 0.878], atol=5e-4)


def test_ppo_clipping():
    answers, scores = sampled()
    probs = softmax(LOGITS)
    actions = [VOCAB.index(a) for a in answers]
    advantages = grpo_advantages(np.array(scores))
    grad = policy_gradient(probs, actions, advantages)
    advantage = float(advantages[0])  # the first sample is "sat"
    small = ppo_term(float(softmax(LOGITS + grad)[CHOSEN] / probs[CHOSEN]), advantage)
    assert round(small["ratio"], 3) == 1.198 and small["clipped_ratio"] == small["ratio"]
    large = ppo_term(float(softmax(LOGITS + 2 * grad)[CHOSEN] / probs[CHOSEN]), advantage)
    assert round(large["ratio"], 3) == 1.332 and large["clipped_ratio"] == 1.2
    assert round(large["unclipped"], 3) == 1.267 and round(large["clipped"], 3) == 1.141
    assert large["objective"] == large["clipped"]
