"""Each number shown in content/datasets/03-preference-datasets.md."""

import pytest

from examples.datasets.preference_datasets import (
    FIRST_BIAS, JUDGE_SCORES, MIN_MARGIN, PAIR_COUNTS, PAIRS, RATE, SWAP_SHARES, TOPIC_PAIRS, TRUE_RATE,
    all_pairs, binarize, bits_per_label, corrected_rate, filter_by_margin, generated_answers_per_pair,
    mean_chosen_quality, mean_margin, mean_pair_margin, measured_rate, pairs_for_one_clean_pair, pairs_needed,
    pairs_needed_exact, position_bias,
    rate_interval, topic_shares,
)
from examples.datasets.toy_dataset import SAMPLES, topic_counts


# --- Part 1: what a pair is ------------------------------------------------------------------

def test_the_four_pairs_use_the_toy_samples():
    assert [pair.sample_id for pair in PAIRS] == [4, 8, 5, 1]
    assert [pair.topic for pair in PAIRS] == ["math", "chat", "code", "math"]
    assert PAIRS[0].prompt == "Count the letters: cat"
    assert (PAIRS[0].chosen, PAIRS[0].rejected) == ("3", "4")
    assert PAIRS[0].rejected == SAMPLES[3].answer  # the wrong answer of the toy dataset
    assert PAIRS[1].rejected == SAMPLES[7].answer == "I cannot help with that."
    assert PAIRS[2].chosen == SAMPLES[4].answer == "len(text.split())"


def test_margins():
    assert [(pair.q_chosen, pair.q_rejected) for pair in PAIRS] == [(1.0, 0.0), (0.8, 0.1), (0.9, 0.8), (0.3, 0.0)]
    assert [round(pair.margin, 3) for pair in PAIRS] == [1.0, 0.7, 0.1, 0.3]
    assert round(mean_margin(PAIRS), 3) == 0.525
    assert round(mean_chosen_quality(PAIRS), 3) == 0.75


def test_margin_filter():
    kept = filter_by_margin(PAIRS)
    assert MIN_MARGIN == 0.2
    assert [pair.sample_id for pair in kept] == [4, 8, 1]
    assert round(mean_margin(kept), 3) == 0.667
    # The filter keeps the pair whose chosen answer is bad: a margin does not see the level.
    assert kept[-1].q_chosen == 0.3 and kept[-1].margin > PAIRS[2].margin
    assert PAIRS[2].q_rejected > kept[-1].q_chosen  # a rejected answer that is better than a chosen answer


# --- Part 2: where pairs come from -----------------------------------------------------------

def test_pairs_from_judge_scores():
    assert JUDGE_SCORES == {"A": 4.5, "B": 4.0, "C": 2.0}
    assert all_pairs(JUDGE_SCORES) == [("A", "B", 0.5), ("A", "C", 2.5), ("B", "C", 2.0)]
    assert binarize(JUDGE_SCORES, "best_worst") == [("A", "C", 2.5)]
    assert binarize(JUDGE_SCORES, "best_second") == [("A", "B", 0.5)]


def test_the_binarization_rule_changes_the_mean_margin():
    margins = {rule: round(mean_pair_margin(JUDGE_SCORES, rule), 3)
               for rule in ("best_worst", "best_second", "best_random", "all")}
    assert margins == {"best_worst": 2.5, "best_second": 0.5, "best_random": 1.5, "all": 1.667}
    with pytest.raises(ValueError):
        binarize(JUDGE_SCORES, "unknown")


def test_generated_answers_per_pair():
    assert generated_answers_per_pair() == {
        "human vote": 2.0, "judge, best against worst": 3.0, "judge, all pairs": 1.0, "construction": 1.0,
    }


# --- Part 3: label noise ---------------------------------------------------------------------

def test_swapped_labels_pull_the_rate_towards_one_half():
    assert TRUE_RATE == 0.8 and SWAP_SHARES == [0.0, 0.1, 0.2, 0.3]
    assert [round(measured_rate(TRUE_RATE, e), 3) for e in SWAP_SHARES] == [0.8, 0.74, 0.68, 0.62]
    assert round(measured_rate(TRUE_RATE, 0.5), 3) == 0.5
    for e in SWAP_SHARES:
        assert corrected_rate(measured_rate(TRUE_RATE, e), e) == pytest.approx(TRUE_RATE)


def test_bits_per_label():
    assert [round(bits_per_label(e), 3) for e in SWAP_SHARES] == [1.0, 0.531, 0.278, 0.119]
    assert bits_per_label(0.5) == 0.0
    assert round(pairs_for_one_clean_pair(0.2), 1) == 3.6


def test_position_bias():
    bias = position_bias()
    assert FIRST_BIAS == 0.6
    assert {key: round(value, 3) for key, value in bias.items()} == {
        "a_always_first": 0.6, "b_always_first": 0.4, "both_orders": 0.5,
        "same_winner_in_both_orders": 0.48, "order_decides": 0.52,
    }
    assert round(bias["a_always_first"] - 0.5, 3) == 0.1  # the false preference


def test_widget_start_values():
    seen = measured_rate(TRUE_RATE, 0.1)
    half, low, high = rate_interval(seen, 100)
    assert (round(seen, 3), round(half, 3), round(low, 3), round(high, 3)) == (0.74, 0.086, 0.654, 0.826)


# --- Part 4: how many pairs ------------------------------------------------------------------

def test_interval_of_a_preference_rate():
    assert RATE == 0.7 and PAIR_COUNTS == [100, 1_000, 10_000]
    rounded = [tuple(round(value, 3) for value in rate_interval(RATE, count)) for count in PAIR_COUNTS]
    assert rounded == [(0.09, 0.61, 0.79), (0.028, 0.672, 0.728), (0.009, 0.691, 0.709)]
    # 100 times more pairs make the interval 10 times narrower.
    assert rate_interval(RATE, 100)[0] / rate_interval(RATE, 10_000)[0] == pytest.approx(10)
    assert round(rate_interval(RATE, 100)[0] / rate_interval(RATE, 1_000)[0], 2) == 3.16  # "about 3" on the page


def test_pairs_needed():
    assert pairs_needed(0.55) == 381
    assert round(pairs_needed_exact(0.55), 1) == 380.3
    assert pairs_needed(0.7) == 21
    assert rate_interval(0.55, 381)[1] > 0.5 > rate_interval(0.55, 380)[1]


def test_topic_coverage():
    assert topic_shares(topic_counts()) == {"math": 0.5, "code": 0.25, "chat": 0.25}
    assert TOPIC_PAIRS == {"math": 800, "code": 150, "chat": 50}
    assert topic_shares(TOPIC_PAIRS) == {"math": 0.8, "code": 0.15, "chat": 0.05}
    rounded = {topic: tuple(round(value, 3) for value in rate_interval(RATE, count))
               for topic, count in TOPIC_PAIRS.items()}
    assert rounded == {"math": (0.032, 0.668, 0.732), "code": (0.073, 0.627, 0.773), "chat": (0.127, 0.573, 0.827)}
