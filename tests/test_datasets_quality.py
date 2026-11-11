"""Each number shown in content/datasets/01-quality.md."""

import math

import numpy as np
import pytest

from examples.datasets.quality import (
    BASELINE_SOLVES, accuracy, accuracy_by_topic, agreement, audit_size, cosine_matrix, entropy, funnel, interval,
    judge_keeps, kept_by_judge_only, mean_distance, most_similar_pair, shares, token_stats, turn_stats,
)
from examples.datasets.toy_dataset import SAMPLES, embeddings, topic_counts


def test_toy_dataset_shape():
    assert len(SAMPLES) == 8
    assert topic_counts() == {"math": 4, "code": 2, "chat": 2}


def test_accuracy_and_its_interval():
    assert accuracy() == 0.75
    se, low, high = interval(0.75, 8)
    assert (round(se, 3), round(low, 3), high) == (0.153, 0.45, 1.0)
    assert accuracy_by_topic() == {"math": (3, 4), "code": (2, 2), "chat": (1, 2)}


def test_audit_size():
    assert audit_size(0.75, 0.05) == 289
    assert audit_size(0.75, 0.02) == 1801


def test_topic_entropy():
    assert shares() == [0.5, 0.25, 0.25]
    assert round(entropy(shares()), 3) == 1.04
    assert round(math.log(3), 3) == 1.099
    assert round(entropy(shares()) / math.log(3), 3) == 0.946
    assert entropy([1.0, 0.0, 0.0]) == 0


def test_embedding_diversity():
    assert round(mean_distance(embeddings()), 3) == 0.263
    math_only = np.array([s.embedding for s in SAMPLES if s.topic == "math"])
    assert round(mean_distance(math_only), 3) == 0.028
    first, second, similarity = most_similar_pair(embeddings())
    assert (first, second) == (1, 3) and similarity > 0.999
    sim = cosine_matrix(embeddings())
    assert round(float(sim[0, 1]), 3) == 0.993 and round(float(sim[0, 4]), 3) == 0.198


def test_complexity():
    assert turn_stats() == (1.375, 0.25)
    mean, median, top_two = token_stats()
    assert (mean, median, round(top_two, 3)) == (29.0, 13.0, 0.647)
    assert len(BASELINE_SOLVES) / len(SAMPLES) == 0.5


def test_judge_against_the_checker():
    assert judge_keeps() == [1, 3, 4, 5, 6, 7]
    observed, chance, kappa = agreement()
    assert (observed, chance, round(kappa, 3)) == (0.75, 0.625, 0.333)


def test_funnel():
    stages = dict(funnel())
    assert stages["refusal rule"] == [1, 2, 3, 4, 5, 6, 7]
    assert stages["checker"] == [1, 2, 3, 5, 6, 7]
    assert stages["near-duplicates"] == [1, 2, 5, 6, 7]
    kept = [SAMPLES[i - 1] for i in stages["near-duplicates"]]
    assert accuracy(kept) == 1.0
    assert shares(kept) == [0.4, 0.4, 0.2]
    assert round(entropy(shares(kept)), 3) == 1.055


@pytest.mark.parametrize("threshold, kept, acc", [(2, 7, 0.857), (3, 6, 0.833), (5, 3, 1.0)])
def test_judge_as_the_only_filter(threshold, kept, acc):
    count, true_accuracy = kept_by_judge_only(threshold)
    assert (count, round(true_accuracy, 3)) == (kept, acc)
