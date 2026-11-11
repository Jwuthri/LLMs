"""Each number shown in content/fundamentals/05-metrics.md."""

import math

import pytest

from examples.fundamentals.metrics import (
    CANDIDATE, CORRECT_TOKEN_PROBS, P, PASS_RATE, Q, Q_ZERO, RANKS, REFERENCE, RELEVANCE, SCORES_NEG, SCORES_POS,
    TARGET, auroc, average_precision, bleu, brier, confusion, cross_entropy, cross_entropy_between, dcg, ece,
    entropy, exact_match, js, kl, lcs, mean_loss, mrr, ndcg, ngram_precision, pass_at_k, perplexity,
    precision_at_base_rate, rates, recall_at_k, roc_points, token_f1,
)


def r3(values):
    return [round(float(v), 3) for v in values]


def test_loss_in_nats_and_bits():
    loss = cross_entropy(P, TARGET)
    assert round(loss, 3) == 0.417
    assert round(loss / math.log(2), 3) == 0.602


def test_mean_loss_and_perplexity():
    assert r3(-math.log(p) for p in CORRECT_TOKEN_PROBS) == [0.417, 0.693, 1.386]
    assert round(mean_loss(CORRECT_TOKEN_PROBS), 3) == 0.832
    assert round(perplexity(CORRECT_TOKEN_PROBS), 3) == 2.298
    assert round(math.log(3), 3) == 1.099  # a model that knows nothing: perplexity 3


def test_entropy_cross_entropy_and_kl():
    assert r3(Q) == [0.186, 0.307, 0.506]
    assert round(entropy(P), 3) == 0.847
    assert round(cross_entropy_between(P, Q), 3) == 0.9
    assert round(kl(P, Q), 3) == 0.053
    assert cross_entropy_between(P, Q) == pytest.approx(entropy(P) + kl(P, Q))
    assert round(kl(Q, P), 3) == 0.058  # not symmetric
    assert round(js(P, Q), 3) == 0.014
    assert js(P, Q) == pytest.approx(js(Q, P))
    assert kl(P, P) == pytest.approx(0.0)


def test_kl_punishes_missing_probability():
    assert round(kl(P, Q_ZERO), 3) == 0.362
    assert round(kl(Q_ZERO, P), 3) == 0.1
    assert round(float(P[0] * math.log(P[0] / Q_ZERO[0])), 3) == 0.452


def test_confusion_matrix_and_rates():
    c = confusion(0.5)
    assert c == {"tp": 3, "fn": 1, "fp": 1, "tn": 5}
    assert {k: round(v, 3) for k, v in rates(c).items()} == {
        "accuracy": 0.8, "precision": 0.75, "recall": 0.75, "f1": 0.75, "fpr": 0.167}


@pytest.mark.parametrize("threshold, precision, recall, f1, fpr", [(0.25, 0.571, 1.0, 0.727, 0.5), (0.75, 1.0, 0.5, 0.667, 0.0)])
def test_threshold_trades_precision_for_recall(threshold, precision, recall, f1, fpr):
    r = rates(confusion(threshold))
    assert (round(r["precision"], 3), round(r["recall"], 3), round(r["f1"], 3), round(r["fpr"], 3)) == (precision, recall, f1, fpr)


def test_auroc_and_average_precision():
    assert round(auroc(SCORES_POS, SCORES_NEG), 3) == 0.833  # 20 of 24 pairs
    assert round(average_precision(SCORES_POS, SCORES_NEG), 3) == 0.83
    points = roc_points(SCORES_POS, SCORES_NEG)
    area = sum((x2 - x1) * y1 for (x1, y1), (x2, _) in zip(points, points[1:]))
    assert area == pytest.approx(auroc(SCORES_POS, SCORES_NEG))  # the area under the curve is the share of pairs


def test_precision_falls_with_a_low_base_rate():
    r = rates(confusion(0.5))
    assert round(precision_at_base_rate(r["recall"], r["fpr"], 0.01), 3) == 0.043
    assert (10 * r["recall"], round(990 * r["fpr"])) == (7.5, 165)


def test_calibration():
    assert round(brier(SCORES_POS, SCORES_NEG), 3) == 0.152
    assert round(ece(SCORES_POS, SCORES_NEG), 3) == 0.04


def test_text_metrics():
    assert exact_match(CANDIDATE, REFERENCE) == 0
    assert r3(token_f1(CANDIDATE, REFERENCE)) == [0.667, 0.667, 0.667]
    assert round(ngram_precision(CANDIDATE, REFERENCE, 1), 3) == 0.667
    assert ngram_precision(CANDIDATE, REFERENCE, 2) == 0.5
    assert round(bleu(CANDIDATE, REFERENCE), 3) == 0.577
    assert lcs(CANDIDATE, REFERENCE) == 2
    assert bleu(REFERENCE, REFERENCE) == pytest.approx(1.0)


def test_pass_at_k():
    assert r3(pass_at_k(PASS_RATE, k) for k in (1, 5, 10)) == [0.2, 0.672, 0.893]


def test_ranking_metrics():
    assert round(recall_at_k(RANKS, 3), 3) == 0.667
    assert round(mrr(RANKS), 3) == 0.511
    assert round(dcg(RELEVANCE), 3) == 1.762
    assert round(dcg(sorted(RELEVANCE, reverse=True)), 3) == 2.631
    assert round(ndcg(RELEVANCE), 3) == 0.67
