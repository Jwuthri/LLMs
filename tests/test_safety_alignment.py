"""Each number shown in content/safety/02-alignment.md."""

import numpy as np

from examples.safety.alignment import (
    APPROVAL, CORRECT, RULES, X_DEPLOY, X_TRAIN, Y_DEPLOY, Y_TRAIN,
    accuracy, audit_break_even, best_policy, bonus_indifferent, bonus_to_accept, expected_reward,
    gradient_descent, return_cause, return_comply, return_prevent, reward_gradient, reward_visible,
    score_hidden, scores, steps_until_top, update,
)
from examples.toy import LOGITS, softmax


def test_optimiser_prefers_the_cheat():
    assert reward_visible("honest") == 0.75 and reward_visible("cheat") == 1.0
    assert best_policy(reward_visible) == "cheat"
    assert reward_visible("cheat") - reward_visible("honest") == 0.25


def test_hidden_tests_measure_the_goal():
    assert score_hidden("honest") == 0.75 and score_hidden("cheat") == 0.0
    assert reward_visible("honest") - score_hidden("honest") == 0.0
    assert reward_visible("cheat") - score_hidden("cheat") == 1.0
    assert best_policy(score_hidden) == "honest"


def test_audit_break_even():
    assert audit_break_even() == 0.25
    assert reward_visible("cheat", audit=0.5) == 0.5
    assert best_policy(lambda name: reward_visible(name, audit=0.5)) == "honest"
    assert best_policy(lambda name: reward_visible(name, audit=0.2)) == "cheat"


def test_two_rules_fit_the_training_data():
    for w in RULES.values():
        assert accuracy(w, X_TRAIN, Y_TRAIN) == 1.0
    assert np.allclose(scores(RULES["read x1"], X_DEPLOY), [1, 0])
    assert np.allclose(scores(RULES["read x2"], X_DEPLOY), [0, 1])
    assert accuracy(RULES["read x1"], X_DEPLOY, Y_DEPLOY) == 1.0
    assert accuracy(RULES["read x2"], X_DEPLOY, Y_DEPLOY) == 0.0


def test_gradient_descent_does_not_choose_a_rule():
    assert np.allclose([gradient_descent([0.0, 0.0], steps=n)[0] for n in (1, 2, 3)], [0.25, 0.375, 0.4375])
    w = gradient_descent([0.0, 0.0])
    assert np.allclose(w, [0.5, 0.5])
    assert accuracy(w, X_TRAIN, Y_TRAIN) == 1.0
    assert np.allclose(scores(w, X_DEPLOY), [0.5, 0.5])


def test_the_start_point_chooses_the_rule():
    toward_x1 = gradient_descent([0.01, 0.0])
    toward_x2 = gradient_descent([0.0, 0.01])
    assert np.allclose(toward_x1, [0.505, 0.495]) and np.allclose(toward_x2, [0.495, 0.505])
    assert accuracy(toward_x1, X_TRAIN, Y_TRAIN) == accuracy(toward_x2, X_TRAIN, Y_TRAIN) == 1.0
    assert accuracy(toward_x1, X_DEPLOY, Y_DEPLOY) == 1.0
    assert accuracy(toward_x2, X_DEPLOY, Y_DEPLOY) == 0.0


def test_one_update_on_approval():
    assert np.allclose(softmax(LOGITS), [0.099, 0.242, 0.659], atol=5e-4)
    assert round(expected_reward(LOGITS, APPROVAL), 3) == 0.426
    assert np.allclose(reward_gradient(LOGITS, APPROVAL), [-0.032, 0.115, -0.083], atol=5e-4)
    after = update(LOGITS, APPROVAL)
    assert np.allclose(after, [0.068, 1.115, 1.917], atol=5e-4)
    assert np.allclose(softmax(after), [0.098, 0.279, 0.623], atol=5e-4)
    assert round(expected_reward(after, APPROVAL), 3) == 0.448


def test_repeated_updates_make_the_liked_answer_first():
    assert steps_until_top(LOGITS, APPROVAL, 1) == 5
    assert np.allclose(softmax(update(LOGITS, APPROVAL, 5)), [0.086, 0.499, 0.416], atol=5e-4)


def test_checked_reward_moves_to_the_correct_answer():
    assert np.allclose(reward_gradient(LOGITS, CORRECT), [-0.065, -0.160, 0.225], atol=5e-4)
    assert np.allclose(softmax(update(LOGITS, CORRECT)), [0.082, 0.184, 0.734], atol=5e-4)


def test_shutdown_returns():
    assert return_comply() == 6.5
    assert return_prevent() == 9.0
    assert return_prevent() - return_comply() == 2.5
    assert return_prevent(cost=3.5) == return_comply()  # the largest cost at which prevention pays


def test_prevention_pays_for_any_positive_reward():
    expected = {0.1: (0.65, 0.9, 0.25), 1.0: (6.5, 9.0, 2.5), 5.0: (32.5, 45.0, 12.5)}
    for r, (comply, prevent, advantage) in expected.items():
        assert np.isclose(return_comply(r), comply) and np.isclose(return_prevent(r), prevent)
        assert np.isclose(return_prevent(r) - return_comply(r), advantage)
    for r in np.linspace(0.01, 100, 50):
        assert return_prevent(r) > return_comply(r)


def test_bonus_for_an_accepted_shutdown():
    assert bonus_to_accept() == 5.0
    assert return_comply(bonus=5.0) == return_prevent() == 9.0
    assert bonus_to_accept(r=5.0) == 25.0
    assert bonus_indifferent() == 7.0
    assert return_comply(bonus=7.0) == return_cause(bonus=7.0) == 10.0
    assert (return_comply(bonus=6.0), return_cause(bonus=6.0)) == (9.5, 9.0)
    assert return_cause(bonus=8.0) == 11.0 > return_comply(bonus=8.0) == 10.5
