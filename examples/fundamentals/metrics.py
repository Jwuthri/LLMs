"""Numbers and plots for content/fundamentals/05-metrics.md.

Run: uv run python -m examples.fundamentals.metrics
"""

import math
from collections import Counter

import numpy as np

from examples.plotstyle import save_figure
from examples.toy import LOGITS, softmax

PAGE = "metrics"

# --- Part 1: loss, bits and perplexity --------------------------------------------------------

P = softmax(LOGITS)  # [0.099, 0.242, 0.659], the toy model after "the cat"
TARGET = 2  # the correct next token is "sat"
CORRECT_TOKEN_PROBS = [P[TARGET], 0.5, 0.25]  # probability of the correct token at 3 positions (2 assumed)


def cross_entropy(probs: np.ndarray, target: int) -> float:
    """Loss of one prediction, in nats: minus the log of the probability of the correct token."""
    return float(-np.log(probs[target]))


def mean_loss(correct_probs: list[float]) -> float:
    return float(np.mean([-math.log(p) for p in correct_probs]))


def perplexity(correct_probs: list[float]) -> float:
    return math.exp(mean_loss(correct_probs))


# --- Part 2: entropy and KL divergence ---------------------------------------------------------

Q = softmax(np.array([0.5, 1.0, 1.5]))  # a second model, for example a student or a quantized copy (assumed)
Q_ZERO = np.array([0.001, 0.299, 0.7])  # a model that gives almost no probability to "the" (assumed)


def entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(p)).sum())


def cross_entropy_between(p: np.ndarray, q: np.ndarray) -> float:
    return float(-(p * np.log(q)).sum())


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q): the extra loss when you use q in place of the true distribution p."""
    return float((p * np.log(p / q)).sum())


def js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence: the mean KL of p and q to their mid point. Symmetric."""
    m = (p + q) / 2
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# --- Part 3: scores with labels -----------------------------------------------------------------

# A detector gives a score to 10 requests. Label 1 = harmful (4 requests), 0 = harmless (6). Assumed.
SCORES_POS = [0.9, 0.8, 0.6, 0.3]
SCORES_NEG = [0.7, 0.4, 0.35, 0.2, 0.1, 0.05]
THRESHOLD = 0.5


def confusion(threshold: float, pos: list[float] = SCORES_POS, neg: list[float] = SCORES_NEG) -> dict[str, int]:
    tp = sum(s >= threshold for s in pos)
    fp = sum(s >= threshold for s in neg)
    return {"tp": tp, "fn": len(pos) - tp, "fp": fp, "tn": len(neg) - fp}


def rates(c: dict[str, float]) -> dict[str, float]:
    precision = c["tp"] / (c["tp"] + c["fp"])
    recall = c["tp"] / (c["tp"] + c["fn"])
    return {
        "accuracy": (c["tp"] + c["tn"]) / sum(c.values()),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "fpr": c["fp"] / (c["fp"] + c["tn"]),
    }


def auroc(pos: list[float], neg: list[float]) -> float:
    """Share of (positive, negative) pairs in which the positive item has the higher score."""
    pairs = [(p, n) for p in pos for n in neg]
    return sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in pairs) / len(pairs)


def roc_points(pos: list[float], neg: list[float]) -> list[tuple[float, float]]:
    """(false positive rate, true positive rate) for each threshold, from strict to loose."""
    points = [(0.0, 0.0)]
    for threshold in sorted(pos + neg, reverse=True):
        c = confusion(threshold, pos, neg)
        points.append((c["fp"] / len(neg), c["tp"] / len(pos)))
    return points


def average_precision(pos: list[float], neg: list[float]) -> float:
    """Mean of the precision at the rank of each positive item."""
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], reverse=True)
    found, total = 0, 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            found += 1
            total += found / rank
    return total / len(pos)


def precision_at_base_rate(tpr: float, fpr: float, base_rate: float) -> float:
    return tpr * base_rate / (tpr * base_rate + fpr * (1 - base_rate))


def brier(pos: list[float], neg: list[float]) -> float:
    return float(np.mean([(s - 1) ** 2 for s in pos] + [s**2 for s in neg]))


def ece(pos: list[float], neg: list[float], edges: tuple[float, ...] = (0.0, 0.5, 1.0)) -> float:
    """Expected calibration error: weighted gap between the mean score and the share of positives of each bin."""
    items = [(s, 1) for s in pos] + [(s, 0) for s in neg]
    total = 0.0
    for low, high in zip(edges, edges[1:]):
        in_bin = [(s, y) for s, y in items if low <= s < high or (high == edges[-1] and s == high)]
        if in_bin:
            gap = abs(np.mean([s for s, _ in in_bin]) - np.mean([y for _, y in in_bin]))
            total += len(in_bin) / len(items) * gap
    return float(total)


# --- Part 4: text and ranking --------------------------------------------------------------------

REFERENCE = "the cat sat".split()
CANDIDATE = "the cat cat".split()
PASS_RATE = 0.2  # probability that one generated program passes the tests (assumed)
RANKS = [1, 3, 5]  # rank of the first relevant document for 3 queries (assumed)
RELEVANCE = [0, 2, 1]  # graded relevance of the documents at ranks 1, 2, 3 for one query (assumed)


def exact_match(candidate: list[str], reference: list[str]) -> int:
    return int(candidate == reference)


def token_f1(candidate: list[str], reference: list[str]) -> tuple[float, float, float]:
    overlap = sum((Counter(candidate) & Counter(reference)).values())
    precision, recall = overlap / len(candidate), overlap / len(reference)
    return precision, recall, 2 * precision * recall / (precision + recall)


def ngram_precision(candidate: list[str], reference: list[str], n: int) -> float:
    """Clipped n-gram precision of BLEU: a candidate n-gram counts at most as often as in the reference."""
    grams = lambda words: Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    cand, ref = grams(candidate), grams(reference)
    return sum((cand & ref).values()) / sum(cand.values())


def bleu(candidate: list[str], reference: list[str], max_n: int = 2) -> float:
    brevity = 1.0 if len(candidate) >= len(reference) else math.exp(1 - len(reference) / len(candidate))
    precisions = [ngram_precision(candidate, reference, n) for n in range(1, max_n + 1)]
    return brevity * math.exp(sum(math.log(p) for p in precisions) / max_n)


def lcs(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (the base of ROUGE-L)."""
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            table[i + 1][j + 1] = table[i][j] + 1 if x == y else max(table[i][j + 1], table[i + 1][j])
    return table[-1][-1]


def pass_at_k(pass_rate: float, k: int) -> float:
    return 1 - (1 - pass_rate) ** k


def recall_at_k(ranks: list[int], k: int) -> float:
    return sum(rank <= k for rank in ranks) / len(ranks)


def mrr(ranks: list[int]) -> float:
    return float(np.mean([1 / rank for rank in ranks]))


def dcg(relevance: list[float]) -> float:
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevance, start=1))


def ndcg(relevance: list[float]) -> float:
    return dcg(relevance) / dcg(sorted(relevance, reverse=True))


# --- Plot ------------------------------------------------------------------------------------

def make_figures() -> None:
    points = roc_points(SCORES_POS, SCORES_NEG)
    c = confusion(THRESHOLD)
    here = (c["fp"] / len(SCORES_NEG), c["tp"] / len(SCORES_POS))

    def draw(ax, theme):
        xs, ys = zip(*points)
        ax.fill_between(xs, ys, step="post", color=theme["series"][0], alpha=0.12, linewidth=0)
        ax.step(xs, ys, where="post", color=theme["series"][0],
                label=f"Detector (area {auroc(SCORES_POS, SCORES_NEG):.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color=theme["series"][1], label="Random scores (area 0.5)")
        ax.plot([here[0]], [here[1]], "o", color=theme["ink"], markersize=6)
        ax.annotate("threshold 0.5", here, textcoords="offset points", xytext=(10, -14), fontsize=8,
                    color=theme["ink2"])
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.04)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (recall)")
        ax.legend(loc="lower right")

    save_figure(draw, PAGE, "roc", size=(5.2, 4.0))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    loss = cross_entropy(P, TARGET)
    print("P:", P, "loss nats:", round(loss, 3), "bits:", round(loss / math.log(2), 3))
    print("losses:", [round(-math.log(p), 3) for p in CORRECT_TOKEN_PROBS], "mean:", round(mean_loss(CORRECT_TOKEN_PROBS), 3),
          "perplexity:", round(perplexity(CORRECT_TOKEN_PROBS), 3), "uniform:", round(math.log(3), 3))
    print("Q:", Q, "H(p):", round(entropy(P), 3), "CE(p,q):", round(cross_entropy_between(P, Q), 3),
          "KL(p||q):", round(kl(P, Q), 3), "KL(q||p):", round(kl(Q, P), 3), "JS:", round(js(P, Q), 3))
    print("KL(p||q0):", round(kl(P, Q_ZERO), 3), "KL(q0||p):", round(kl(Q_ZERO, P), 3),
          "term of 'the':", round(P[0] * math.log(P[0] / Q_ZERO[0]), 3))
    c = confusion(THRESHOLD)
    print("confusion:", c, {k: round(v, 3) for k, v in rates(c).items()})
    for t in (0.25, 0.75):
        print("threshold", t, confusion(t), {k: round(v, 3) for k, v in rates(confusion(t)).items()})
    print("AUROC:", round(auroc(SCORES_POS, SCORES_NEG), 3), "AP:", round(average_precision(SCORES_POS, SCORES_NEG), 3))
    r = rates(c)
    print("precision at base rate 0.01:", round(precision_at_base_rate(r["recall"], r["fpr"], 0.01), 3),
          "of 1,000:", 10 * r["recall"], 990 * r["fpr"])
    print("Brier:", round(brier(SCORES_POS, SCORES_NEG), 3), "ECE:", round(ece(SCORES_POS, SCORES_NEG), 3))
    print("EM:", exact_match(CANDIDATE, REFERENCE), "token F1:", [round(x, 3) for x in token_f1(CANDIDATE, REFERENCE)])
    print("p1, p2:", round(ngram_precision(CANDIDATE, REFERENCE, 1), 3), ngram_precision(CANDIDATE, REFERENCE, 2),
          "BLEU-2:", round(bleu(CANDIDATE, REFERENCE), 3), "LCS:", lcs(CANDIDATE, REFERENCE))
    print("pass@k:", {k: round(pass_at_k(PASS_RATE, k), 3) for k in (1, 5, 10)})
    print("recall@3:", round(recall_at_k(RANKS, 3), 3), "MRR:", round(mrr(RANKS), 3),
          "DCG:", round(dcg(RELEVANCE), 3), "IDCG:", round(dcg(sorted(RELEVANCE, reverse=True)), 3),
          "nDCG:", round(ndcg(RELEVANCE), 3))
    make_figures()
