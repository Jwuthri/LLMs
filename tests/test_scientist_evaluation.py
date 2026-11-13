"""Each number shown in content/scientist/06-evaluation.md."""

import numpy as np

from examples.scientist.evaluation import (
    AGREEMENT_TABLE, CONTAMINATED, ERROR_TABLE, MODEL_A, MODEL_B, RUBRIC_SCORES,
    VERDICTS_A_FIRST, VERDICTS_B_FIRST,
    accuracy, cohen_kappa, confidence_interval, difference_half_width, disagreements, elo_expected,
    elo_update, error_analysis, overall_accuracy, position_bias, questions_needed, rubric_score,
    standard_error, without,
)


def test_accuracy():
    assert (sum(MODEL_A), sum(MODEL_B)) == (7, 6)
    assert (accuracy(MODEL_A), accuracy(MODEL_B)) == (0.7, 0.6)
    assert disagreements(MODEL_A, MODEL_B) == (2, 1)


def test_interval_with_10_questions():
    assert round(standard_error(0.7, 10), 3) == 0.145
    assert round(standard_error(0.6, 10), 3) == 0.155
    assert np.allclose(confidence_interval(0.7, 10), [0.284, 0.416, 0.984], atol=5e-4)
    assert np.allclose(confidence_interval(0.6, 10), [0.304, 0.296, 0.904], atol=5e-4)
    half = difference_half_width(0.7, 0.6, 10)
    assert round(half, 3) == 0.416
    assert 0.1 - half < 0 < 0.1 + half  # the interval of the difference contains 0


def test_interval_with_1000_questions():
    assert round(standard_error(0.7, 1000), 4) == 0.0145
    assert np.allclose(confidence_interval(0.7, 1000), [0.028, 0.672, 0.728], atol=5e-4)
    assert np.allclose(confidence_interval(0.6, 1000), [0.030, 0.570, 0.630], atol=5e-4)
    half = difference_half_width(0.7, 0.6, 1000)
    assert round(half, 3) == 0.042
    assert 0.1 - half > 0  # the interval of the difference does not contain 0


def test_questions_needed():
    n = questions_needed(0.7, 0.6)
    assert n == 173
    assert difference_half_width(0.7, 0.6, n) < 0.1 <= difference_half_width(0.7, 0.6, n - 1)
    assert round(difference_half_width(0.7, 0.6, n), 3) == 0.100


def test_contamination():
    clean_a, clean_b = without(MODEL_A, CONTAMINATED), without(MODEL_B, CONTAMINATED)
    assert all(MODEL_A[i] == 1 for i in CONTAMINATED)
    assert (sum(clean_a), sum(clean_b), len(clean_a)) == (4, 4, 7)
    assert round(accuracy(clean_a), 3) == 0.571 and accuracy(clean_a) == accuracy(clean_b)


def test_mmlu_interval():
    assert round(confidence_interval(0.7, 14042)[0], 4) == 0.0076


def test_elo():
    assert round(elo_expected(1000, 1100), 3) == 0.360
    assert round(elo_expected(1100, 1000), 3) == 0.640
    assert round(elo_update(1000, 1100, 1), 1) == 1020.5
    assert round(elo_update(1100, 1000, 0), 1) == 1079.5
    assert round(elo_update(1000, 1100, 0), 1) == 988.5
    assert round(elo_update(1100, 1000, 1), 1) == 1111.5
    assert round(32 * (1 - elo_expected(1000, 1100)), 1) == 20.5


def test_cohen_kappa():
    result = cohen_kappa(AGREEMENT_TABLE)
    assert AGREEMENT_TABLE.sum() == 50
    assert np.isclose(result["observed"], 0.7)
    assert np.isclose(result["expected"], 0.5)
    assert np.isclose(result["kappa"], 0.4)


def test_judge():
    assert round(rubric_score(RUBRIC_SCORES), 3) == 0.833
    bias = position_bias(VERDICTS_A_FIRST, VERDICTS_B_FIRST)
    assert (bias["a_first_wins_a"], bias["b_first_wins_a"]) == (7, 4)
    assert bias["flips"] == 3 and bias["flip_rate"] == 0.3
    assert (bias["stable_a"], bias["stable_b"]) == (4, 3)
    assert bias["first_position_rate"] == 0.65
    # In each flip, the judge picked the answer that was shown first.
    assert all(x == "A" and y == "B" for x, y in zip(VERDICTS_A_FIRST, VERDICTS_B_FIRST) if x != y)


def test_error_analysis():
    assert np.isclose(overall_accuracy(ERROR_TABLE), 0.7)
    assert sum(q for q, _ in ERROR_TABLE.values()) == 1000
    assert sum(e for _, e in ERROR_TABLE.values()) == 300
    rows = error_analysis(ERROR_TABLE)
    assert [round(r["rate"], 2) for r in rows.values()] == [0.60, 0.20, 0.40, 0.16]
    assert [round(100 * r["share"], 1) for r in rows.values()] == [40.0, 26.7, 20.0, 13.3]
    assert [round(r["gain_if_halved"], 2) for r in rows.values()] == [0.06, 0.04, 0.03, 0.02]
