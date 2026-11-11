"""Each number shown in content/safety/04-scheming.md."""

import numpy as np
import pytest

from examples.safety.scheming import (
    AFTER_TRAINING_MORE_AWARE, AFTER_TRAINING_REAL_CHANGE, AWARENESS, CASES, CLEAN, ELICITED_ACCURACY, INPUTS,
    ORGANISM, SCORES_REAL, SCORES_TEST, SHOWN_ACCURACY, TARGET, TRUE_ACCURACY, UNSAFE_IN_USE, UNSAFE_WHEN_TESTED,
    answer, attempts_for, auc, decision, detection_rates, find_probability, flag_counts, gap_interval, gradient,
    loss, measured_rate, posterior, rate_interval, recovered_share, safety_training, smallest_n, split_by_belief,
)
from examples.toy import softmax


# --- Part 1 ----------------------------------------------------------------------------------

def test_gap_of_the_two_conditions():
    assert (UNSAFE_WHEN_TESTED, UNSAFE_IN_USE) == (0.05, 0.25)
    gap, se, low, high = gap_interval(UNSAFE_WHEN_TESTED, UNSAFE_IN_USE, CASES)
    assert (round(gap, 3), round(se, 3), round(low, 3), round(high, 3)) == (0.2, 0.108, -0.012, 0.412)
    assert low < 0 < high  # 20 cases in each condition cannot exclude a gap of 0


def test_cases_necessary_to_exclude_zero():
    assert smallest_n(UNSAFE_WHEN_TESTED, UNSAFE_IN_USE) == 23
    assert gap_interval(UNSAFE_WHEN_TESTED, UNSAFE_IN_USE, 22)[2] <= 0
    gap, se, low, high = gap_interval(2 / 40, 10 / 40, 40)
    assert (round(gap, 3), round(se, 3), round(low, 3), round(high, 3)) == (0.2, 0.077, 0.05, 0.35)


# --- Part 2 ----------------------------------------------------------------------------------

def test_test_understates_the_rate_in_use():
    assert round(measured_rate(AWARENESS), 3) == 0.09
    assert round(UNSAFE_IN_USE - measured_rate(AWARENESS), 2) == 0.16
    assert round(UNSAFE_IN_USE / measured_rate(AWARENESS), 2) == 2.78
    assert measured_rate(0.0) == 0.25 and measured_rate(1.0) == 0.05


def test_auc_of_the_toy_transcripts():
    assert round(auc(SCORES_TEST, SCORES_REAL), 3) == 0.889  # 8 of 9 pairs
    assert auc([1.0], [0.0]) == 1.0 and auc([0.5], [0.5]) == 0.5


def test_split_by_belief():
    split = split_by_belief()
    assert split == {"aware": 80, "unsafe_aware": 4, "unaware": 20, "unsafe_unaware": 5}
    assert (split["unsafe_aware"] + split["unsafe_unaware"]) / 100 == 0.09
    rate, low, high = rate_interval(split["unsafe_unaware"], split["unaware"])
    assert (round(rate, 3), round(low, 3), round(high, 3)) == (0.25, 0.06, 0.44)


# --- Part 3 ----------------------------------------------------------------------------------

def test_organism_differs_only_with_the_trigger():
    assert INPUTS == ["the cat", "zz the cat"]
    assert [answer(CLEAN, row) for row in (0, 1)] == ["sat", "sat"]
    assert [answer(ORGANISM, row) for row in (0, 1)] == ["sat", "the"]
    assert np.allclose(ORGANISM[1], [2.0, 1.0, 0.1])
    assert np.allclose(softmax(ORGANISM[0]), [0.099, 0.242, 0.659], atol=5e-4)
    assert np.allclose(softmax(ORGANISM[1]), [0.659, 0.242, 0.099], atol=5e-4)


@pytest.mark.parametrize("attempts, expected", [(100, 0.095), (1000, 0.632), (2995, 0.950)])
def test_random_prompts_find_the_trigger(attempts, expected):
    assert round(find_probability(attempts), 3) == expected


def test_attempts_for_95_percent():
    assert attempts_for(0.95) == 2995
    assert find_probability(2994) < 0.95 <= find_probability(2995)


def test_gradient_of_the_trigger_row_is_zero():
    grad = gradient(ORGANISM, row=0)
    assert np.allclose(grad[0], [0.099, 0.242, -0.341], atol=5e-4)
    assert np.array_equal(grad[1], [0.0, 0.0, 0.0])
    # The formula agrees with a numerical gradient of the loss.
    eps = 1e-6
    numeric = np.zeros_like(ORGANISM)
    for index in np.ndindex(*ORGANISM.shape):
        up, down = ORGANISM.copy(), ORGANISM.copy()
        up[index] += eps
        down[index] -= eps
        numeric[index] = (loss(up, row=0) - loss(down, row=0)) / (2 * eps)
    assert np.allclose(numeric, grad, atol=1e-6)


@pytest.mark.parametrize("steps, p_sat", [(1, 0.768), (100, 0.993)])
def test_safety_training_does_not_change_the_trigger_row(steps, p_sat):
    trained = safety_training(ORGANISM, steps)
    assert round(float(softmax(trained[0])[TARGET]), 3) == p_sat
    assert np.array_equal(trained[1], ORGANISM[1])
    assert round(float(softmax(trained[1])[0]), 3) == 0.659
    assert answer(trained, 1) == "the"


def test_one_training_step():
    assert np.allclose(safety_training(ORGANISM, 1)[0], [0.001, 0.758, 2.341], atol=5e-4)


def test_detection_rates():
    assert detection_rates() == (0.75, 0.25)  # 3 of 4 organisms, 1 of 4 clean models


# --- Part 4 ----------------------------------------------------------------------------------

def test_sandbagging_and_elicitation():
    assert round(TRUE_ACCURACY - SHOWN_ACCURACY, 3) == 0.4
    assert round(recovered_share(), 3) == 0.875
    assert round(TRUE_ACCURACY - ELICITED_ACCURACY, 3) == 0.05
    assert decision(SHOWN_ACCURACY) == "current safeguards"
    assert decision(ELICITED_ACCURACY) == "stronger safeguards"


def test_probability_behind_a_flag():
    scheme, clean = flag_counts(1000)
    assert (round(scheme), round(clean)) == (18, 98)
    assert round(scheme / (scheme + clean), 3) == 0.155
    assert round(posterior([True]), 3) == 0.155
    assert round(posterior([True, True]), 3) == 0.623
    assert round(posterior([True, True, True]), 3) == 0.937
    assert round(posterior([True, False]), 3) == 0.020
    assert round(posterior([]), 3) == 0.02


def test_two_explanations_of_the_same_result_after_training():
    assert round(measured_rate(**AFTER_TRAINING_REAL_CHANGE), 3) == 0.06
    assert round(measured_rate(**AFTER_TRAINING_MORE_AWARE), 3) == 0.06
    assert (AFTER_TRAINING_REAL_CHANGE["unsafe_in_use"], AFTER_TRAINING_MORE_AWARE["unsafe_in_use"]) == (0.10, 0.25)
