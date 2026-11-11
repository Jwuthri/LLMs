"""Each number shown in content/safety/05-interpretability.md."""

import numpy as np
import pytest

from examples.safety.interpretability import (
    FEATURES, H, H_CLEAN, H_CORRUPTED, H_NEW, H_WITH, H_WITHOUT, SAE_BIAS, THRESHOLD, V, V_CAT,
    ablate, difference_of_means, encode, interference, linear_read, logit_difference, logits,
    mean_ablate, patch, probabilities, probe_accuracy, recovered, steer,
)
from examples.toy import E

STATES = np.vstack([H_WITH, H_WITHOUT])
LABELS = np.array([1, 1, 1, 0, 0, 0])


def test_base_model():
    assert np.allclose(logits(H), [1, 1, 2])
    assert np.allclose(probabilities(H), [0.212, 0.212, 0.576], atol=5e-4)


# --- Part 1: probe ---------------------------------------------------------------------------

def test_probe_direction_is_the_difference_of_means():
    assert np.allclose(H_WITH.mean(axis=0), [0.2, 1.4])
    assert np.allclose(H_WITHOUT.mean(axis=0), [1.2, 0.4])
    assert np.allclose(difference_of_means(H_WITH, H_WITHOUT), [-1, 1])
    assert np.allclose(V, [-1, 1])


def test_probe_scores_threshold_and_accuracy():
    assert np.allclose(H_WITH @ V, [1.0, 1.2, 1.4])
    assert np.allclose(H_WITHOUT @ V, [-0.4, -1.2, -0.8])
    assert np.isclose((H_WITH @ V).mean(), 1.2) and np.isclose((H_WITHOUT @ V).mean(), -0.8)
    assert np.isclose(THRESHOLD, 0.2)
    assert probe_accuracy(STATES, LABELS, V, THRESHOLD) == 1.0


def test_probe_fails_on_a_state_without_the_correlated_feature():
    assert np.isclose(H_NEW @ V, -0.2)  # below the threshold: the probe says "no concept"
    assert H_NEW @ V < THRESHOLD
    assert np.isclose(H_NEW @ V_CAT, 1.4)  # the second number alone is as high as in the group with the concept
    accuracy = probe_accuracy(np.vstack([STATES, H_NEW]), np.append(LABELS, 1), V, THRESHOLD)
    assert round(accuracy, 3) == 0.857  # 6 of 7


# --- Part 2: superposition -------------------------------------------------------------------

def test_three_directions_in_two_dimensions_interfere():
    assert round(float(np.linalg.norm(E[2])), 3) == 1.414
    assert np.allclose(FEATURES, [[1, 0], [0, 1], [0.707, 0.707]], atol=5e-4)
    assert np.allclose(interference(FEATURES), [[1, 0, 0.707], [0, 1, 0.707], [0.707, 0.707, 1]], atol=5e-4)


def test_bias_and_relu_remove_small_interference():
    only_sat = FEATURES[2]
    assert np.allclose(linear_read(only_sat), [0.707, 0.707, 1.0], atol=5e-4)
    assert SAE_BIAS == -0.75
    assert np.allclose(linear_read(only_sat) + SAE_BIAS, [-0.043, -0.043, 0.25], atol=5e-4)
    assert np.allclose(encode(only_sat), [0, 0, 0.25])
    assert np.count_nonzero(encode(only_sat)) == 1


def test_two_active_features_look_like_the_third():
    the_and_cat = FEATURES[0] + FEATURES[1]
    assert np.allclose(the_and_cat, [1, 1])
    assert np.allclose(linear_read(the_and_cat), [1, 1, 1.414], atol=5e-4)
    assert np.allclose(encode(the_and_cat), [0.25, 0.25, 0.664], atol=5e-4)
    assert np.count_nonzero(encode(the_and_cat)) == 3


# --- Part 3: steering ------------------------------------------------------------------------

@pytest.mark.parametrize("alpha, state, expected_logits, expected_probs", [
    (0, [1, 1], [1, 1, 2], [0.212, 0.212, 0.576]),
    (1, [0, 2], [0, 2, 2], [0.063, 0.468, 0.468]),
    (2, [-1, 3], [-1, 3, 2], [0.013, 0.721, 0.265]),
    (4, [-3, 5], [-3, 5, 2], [0.000, 0.952, 0.047]),
])
def test_steering(alpha, state, expected_logits, expected_probs):
    h = steer(H, V, alpha)
    assert np.allclose(h, state)
    assert np.allclose(logits(h), expected_logits)
    assert np.allclose(probabilities(h), expected_probs, atol=5e-4)


def test_steering_moves_the_probe_score_and_not_the_sat_logit():
    assert np.isclose(V @ V, 2.0)
    assert np.allclose([steer(H, V, a) @ V for a in (0, 1, 2, 4)], [0, 2, 4, 8], atol=1e-9)
    assert np.isclose(E[2] @ V, 0.0)  # the `sat` row of E is at a right angle to v


def test_large_steering_gives_all_the_probability_to_one_token():
    assert round(probabilities(steer(H, V, 8))[1], 3) == 0.999


# --- Part 4: ablation and patching -----------------------------------------------------------

def test_probe_and_ablation_numbers_of_the_old_lesson():
    assert np.allclose(E @ V_CAT, [0, 1, 1])
    assert np.allclose(ablate(H, V_CAT), [1, 0])
    assert np.allclose(logits(H), [1, 1, 2])
    assert np.allclose(logits(ablate(H, V_CAT)), [1, 0, 1])  # `sat`: 2 -> 1, `the`: 1 -> 1


def test_mean_ablation():
    assert np.allclose(mean_ablate(H, V_CAT, E), [1, 0.667], atol=5e-4)
    assert round(float(logits(mean_ablate(H, V_CAT, E))[2]), 3) == 1.667


def test_activation_patching():
    assert np.allclose(H_CLEAN, [0.2, 1.2]) and np.allclose(H_CORRUPTED, [1.0, 0.6])
    assert np.isclose(logit_difference(H_CLEAN), 1.0)
    assert np.isclose(logit_difference(H_CORRUPTED), -0.4)

    neuron_1 = patch(H_CORRUPTED, H_CLEAN, 0)
    assert np.allclose(neuron_1, [0.2, 0.6])
    assert np.isclose(logit_difference(neuron_1), 0.4)
    assert round(recovered(neuron_1, H_CLEAN, H_CORRUPTED), 3) == 0.571

    neuron_2 = patch(H_CORRUPTED, H_CLEAN, 1)
    assert np.allclose(neuron_2, [1.0, 1.2])
    assert np.isclose(logit_difference(neuron_2), 0.2)
    assert round(recovered(neuron_2, H_CLEAN, H_CORRUPTED), 3) == 0.429

    assert np.isclose(recovered(H_CLEAN, H_CLEAN, H_CORRUPTED), 1.0)
