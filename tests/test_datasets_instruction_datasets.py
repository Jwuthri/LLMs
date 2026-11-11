"""Each number shown in content/datasets/02-instruction-datasets.md."""

import math

import numpy as np
import pytest

from examples.datasets.instruction_datasets import (
    BATCH_SIZE, CAP, CARDS, MIX_SHARES, MIX_SIZE, REAL_SHARES, SOURCE_TOKENS, SOURCES, STEPS, TARGET_SEEN,
    TOKENS_PER_SECOND, UNSEEN_SHARE, age_months, batch_without, capped, commercial_cards, downsampled_counts,
    expected_seen, kl, mix_tokens, shares, steps_to_see, total_samples, total_tokens, training_hours, upsampled,
    weights, with_unseen_topic,
)
from examples.datasets.toy_dataset import SAMPLES, topic_counts


# --- Part 1: the families --------------------------------------------------------------------

def test_toy_topic_shares():
    assert topic_counts() == {"math": 4, "code": 2, "chat": 2}
    assert shares(topic_counts()) == {"math": 0.5, "code": 0.25, "chat": 0.25}
    assert (len(SAMPLES), sum(sample.tokens for sample in SAMPLES)) == (8, 232)


def test_samples_seen_in_training():
    assert (BATCH_SIZE, STEPS, TARGET_SEEN) == (16, 1000, 100)
    assert STEPS * BATCH_SIZE == 16_000
    assert STEPS * BATCH_SIZE / len(SAMPLES) == 2000
    assert [expected_seen(share) for share in (0.5, 0.25, 0.25)] == [8000, 4000, 4000]


def test_steps_to_see_100_samples():
    assert (steps_to_see(0.5), steps_to_see(0.25)) == (13, 25)
    assert 100 / (0.25 * 16) == 25


def test_batch_without_a_topic():
    assert round(batch_without(0.25), 3) == 0.010
    assert round(batch_without(0.5), 6) == 0.000015
    assert round(STEPS * batch_without(0.25)) == 10


# --- Part 2: dataset cards -------------------------------------------------------------------

def test_candidate_list():
    assert [(card.name, card.samples, card.licence) for card in CARDS] == [
        ("open-perfectblend", 1_420_000, "Apache-2.0"),
        ("Dolci-Instruct-SFT", 2_150_000, "CC-BY-NC-4.0"),
        ("SYNTHETIC-2-SFT-verified", 4_000_000, "permissive"),
        ("MegaScience", 1_250_000, "CC-BY-NC-SA-4.0"),
        ("NuminaMath-CoT", 859_000, "Apache-2.0"),
    ]
    assert total_samples(CARDS) == 9_679_000


def test_licence_filter():
    kept = commercial_cards()
    assert [card.name for card in kept] == ["open-perfectblend", "SYNTHETIC-2-SFT-verified", "NuminaMath-CoT"]
    assert total_samples(kept) == 6_279_000
    assert total_samples(CARDS) - total_samples(kept) == 3_400_000
    assert round(1 - total_samples(kept) / total_samples(CARDS), 3) == 0.351


def test_token_total():
    kept = commercial_cards()
    assert [card.mean_tokens for card in kept] == [500, 2000, 600]
    assert [card.samples * card.mean_tokens for card in kept] == [710_000_000, 8_000_000_000, 515_400_000]
    assert total_tokens(kept) == 9_225_400_000


def test_thinking_share():
    kept = commercial_cards()
    thinking = [card for card in kept if card.thinking == "Yes"]
    assert [card.name for card in thinking] == ["SYNTHETIC-2-SFT-verified"]
    assert round(total_samples(thinking) / total_samples(kept), 3) == 0.637
    assert round(total_tokens(thinking) / total_tokens(kept), 3) == 0.867


def test_age_of_each_dataset():
    assert [age_months(card) for card in commercial_cards()] == [25, 17, 28]


# --- Part 3: mixing --------------------------------------------------------------------------

def test_mix_without_a_cap():
    assert SOURCES == {"general": 4_000_000, "code": 1_000_000, "function calling": 100_000}
    assert sum(SOURCES.values()) == 5_100_000
    assert [round(share, 3) for share in shares(SOURCES).values()] == [0.784, 0.196, 0.020]
    assert steps_to_see(shares(SOURCES)["function calling"]) == 319


def test_mix_with_a_cap():
    mix = capped(SOURCES, CAP)
    assert CAP == 500_000
    assert mix == {"general": 500_000, "code": 500_000, "function calling": 100_000}
    assert sum(mix.values()) == 1_100_000
    assert [round(share, 3) for share in shares(mix).values()] == [0.455, 0.455, 0.091]
    assert steps_to_see(shares(mix)["function calling"]) == 69


def test_upsampled_mix():
    mix = upsampled(SOURCES, CAP)
    assert sum(mix.values()) == 1_500_000
    assert [round(share, 3) for share in shares(mix).values()] == [0.333, 0.333, 0.333]
    assert CAP / SOURCES["function calling"] == 5
    assert (mix_tokens(mix), training_hours(mix_tokens(mix))) == (900_000_000, 100)


def test_token_budget_and_time():
    assert SOURCE_TOKENS == {"general": 600, "code": 800, "function calling": 400}
    assert TOKENS_PER_SECOND == 2_500
    full, mix = mix_tokens(SOURCES), mix_tokens(capped(SOURCES, CAP))
    assert (full, mix) == (3_240_000_000, 740_000_000)
    assert mix / TOKENS_PER_SECOND == 296_000
    assert round(1 - mix / full, 3) == 0.772  # share of the tokens that the cap removes
    assert (round(training_hours(mix), 1), round(training_hours(mix) / 24, 2)) == (82.2, 3.43)
    assert (training_hours(full), training_hours(full) / 24) == (360, 15)
    assert round(training_hours(mix, epochs=3), 1) == 246.7


# --- Part 4: real conversations --------------------------------------------------------------

def test_shares_of_the_two_distributions():
    assert MIX_SHARES.tolist() == [0.5, 0.25, 0.25]
    assert REAL_SHARES.tolist() == [0.2, 0.3, 0.5]


def test_kl_in_the_two_directions():
    assert round(kl(REAL_SHARES, MIX_SHARES), 3) == 0.218
    assert round(kl(MIX_SHARES, REAL_SHARES), 3) == 0.239
    assert kl(REAL_SHARES, REAL_SHARES) == 0


def test_largest_gap():
    gaps = np.round(MIX_SHARES - REAL_SHARES, 3)
    assert gaps.tolist() == [0.3, -0.05, -0.25]
    assert np.abs(gaps).max() == 0.3


def test_reweighted_mix():
    assert np.round(weights(REAL_SHARES, MIX_SHARES), 3).tolist() == [0.4, 1.2, 2.0]
    mix_counts = MIX_SHARES * MIX_SIZE
    assert mix_counts.tolist() == [500, 250, 250]
    assert (REAL_SHARES * MIX_SIZE).tolist() == [200, 300, 500]
    down = downsampled_counts(REAL_SHARES, mix_counts)
    assert down.tolist() == [100, 150, 250]
    assert down.sum() == 500
    assert kl(REAL_SHARES, down / down.sum()) == pytest.approx(0)


def test_topic_that_the_mix_does_not_have():
    real = with_unseen_topic(REAL_SHARES, UNSEEN_SHARE)
    mix = np.append(MIX_SHARES, 0.0)
    assert UNSEEN_SHARE == 0.1
    assert np.round(real, 3).tolist() == [0.18, 0.27, 0.45, 0.1]
    assert math.isinf(kl(real, mix))
    assert round(kl(mix, real), 3) == 0.345
