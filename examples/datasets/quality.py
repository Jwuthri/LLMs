"""Numbers and plots for content/datasets/01-quality.md.

Run: uv run python -m examples.datasets.quality
"""

import math

import numpy as np

from examples.datasets.toy_dataset import SAMPLES, TOPICS, embeddings
from examples.plotstyle import save_figure

PAGE = "dataset-quality"
Z_95 = 1.96

# --- Part 1: accuracy -------------------------------------------------------------------------


def accuracy(samples=SAMPLES) -> float:
    return sum(s.correct for s in samples) / len(samples)


def interval(p: float, n: int) -> tuple[float, float, float]:
    """Standard error and 95% interval (normal approximation), cut to the range 0 to 1."""
    se = math.sqrt(p * (1 - p) / n)
    return se, max(0.0, p - Z_95 * se), min(1.0, p + Z_95 * se)


def accuracy_by_topic() -> dict[str, tuple[int, int]]:
    return {t: (sum(s.correct for s in SAMPLES if s.topic == t), sum(s.topic == t for s in SAMPLES)) for t in TOPICS}


def audit_size(p: float, half_width: float) -> int:
    """Number of samples to read by hand to know the accuracy within +/- half_width."""
    return math.ceil(Z_95**2 * p * (1 - p) / half_width**2)


# --- Part 2: diversity --------------------------------------------------------------------------


def shares(samples=SAMPLES) -> list[float]:
    return [sum(s.topic == t for s in samples) / len(samples) for t in TOPICS]


def entropy(p: list[float]) -> float:
    return -sum(x * math.log(x) for x in p if x > 0)


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return unit @ unit.T


def mean_distance(vectors: np.ndarray) -> float:
    """Mean cosine distance (1 - cosine similarity) over all the pairs of different samples."""
    sim = cosine_matrix(vectors)
    upper = sim[np.triu_indices(len(vectors), k=1)]
    return float(np.mean(1 - upper))


def most_similar_pair(vectors: np.ndarray) -> tuple[int, int, float]:
    sim = cosine_matrix(vectors)
    np.fill_diagonal(sim, -1)
    i, j = np.unravel_index(np.argmax(sim), sim.shape)
    return SAMPLES[i].id, SAMPLES[j].id, float(sim[i, j])


# --- Part 3: complexity -------------------------------------------------------------------------

BASELINE_SOLVES = [1, 2, 3, 5]  # ids of the samples that a small baseline model answers correctly (assumed)


def turn_stats() -> tuple[float, float]:
    turns = [s.turns for s in SAMPLES]
    return float(np.mean(turns)), sum(t > 1 for t in turns) / len(turns)


def token_stats() -> tuple[float, float, float]:
    tokens = sorted(s.tokens for s in SAMPLES)
    top_two = sum(tokens[-2:]) / sum(tokens)
    return float(np.mean(tokens)), float(np.median(tokens)), top_two


# --- Part 4: several checks, one decision ------------------------------------------------------

JUDGE = {1: 5, 2: 2, 3: 5, 4: 4, 5: 4, 6: 5, 7: 4, 8: 1}  # score from 1 to 5 of a judge model (assumed)
JUDGE_THRESHOLD = 3
DUPLICATE_SIMILARITY = 0.999


def judge_keeps(threshold: int = JUDGE_THRESHOLD) -> list[int]:
    return [s.id for s in SAMPLES if JUDGE[s.id] >= threshold]


def agreement(threshold: int = JUDGE_THRESHOLD) -> tuple[float, float, float]:
    """Judge decision against the checker: observed agreement, agreement by chance, Cohen's kappa."""
    keeps = set(judge_keeps(threshold))
    n = len(SAMPLES)
    observed = sum((s.id in keeps) == s.correct for s in SAMPLES) / n
    judge_yes, checker_yes = len(keeps) / n, accuracy()
    chance = judge_yes * checker_yes + (1 - judge_yes) * (1 - checker_yes)
    return observed, chance, (observed - chance) / (1 - chance)


def funnel() -> list[tuple[str, list[int]]]:
    """Ids that stay after each filter: rule (refusal words), checker, near-duplicates."""
    stages = [("start", [s.id for s in SAMPLES])]
    after_rule = [s.id for s in SAMPLES if "I cannot" not in s.answer]
    stages.append(("refusal rule", after_rule))
    after_checker = [i for i in after_rule if SAMPLES[i - 1].correct]
    stages.append(("checker", after_checker))
    sim = cosine_matrix(embeddings())
    kept: list[int] = []
    for i in after_checker:
        if all(sim[i - 1, j - 1] < DUPLICATE_SIMILARITY for j in kept):
            kept.append(i)
    stages.append(("near-duplicates", kept))
    return stages


def kept_by_judge_only(threshold: int) -> tuple[int, float]:
    """Number of kept samples and their true accuracy, if the judge is the only filter."""
    keeps = judge_keeps(threshold)
    if not keeps:
        return 0, float("nan")
    return len(keeps), sum(SAMPLES[i - 1].correct for i in keeps) / len(keeps)


# --- Plot ------------------------------------------------------------------------------------

def make_figures() -> None:
    thresholds = [1, 2, 3, 4, 5]
    kept = [kept_by_judge_only(t)[0] / len(SAMPLES) for t in thresholds]
    acc = [kept_by_judge_only(t)[1] for t in thresholds]

    def draw(ax, theme):
        ax.plot(thresholds, kept, "o-", color=theme["series"][0], label="Share of samples kept")
        ax.plot(thresholds, acc, "s-", color=theme["series"][1], label="Accuracy of the kept samples")
        ax.set_xticks(thresholds)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Lowest judge score that is kept")
        ax.set_ylabel("Share")
        ax.legend(loc="lower left")

    save_figure(draw, PAGE, "judge-threshold")


if __name__ == "__main__":
    acc = accuracy()
    print("accuracy:", acc, [round(x, 3) for x in interval(acc, len(SAMPLES))], accuracy_by_topic())
    print("audit size +/-0.05:", audit_size(0.75, 0.05), "+/-0.02:", audit_size(0.75, 0.02),
          "interval n=288:", [round(x, 3) for x in interval(0.75, 288)])
    p = shares()
    print("shares:", p, "entropy:", round(entropy(p), 3), "max:", round(math.log(3), 3), "ratio:", round(entropy(p) / math.log(3), 3))
    math_only = [s for s in SAMPLES if s.topic == "math"]
    print("mean distance all:", round(mean_distance(embeddings()), 3), "math only:",
          round(mean_distance(np.array([s.embedding for s in math_only])), 3))
    print("most similar pair:", most_similar_pair(embeddings()))
    print("cos(1,2):", round(float(cosine_matrix(embeddings())[0, 1]), 4), "cos(1,5):", round(float(cosine_matrix(embeddings())[0, 4]), 3))
    print("turns mean, multi-turn share:", turn_stats(), "tokens mean, median, top-two share:", [round(x, 3) for x in token_stats()])
    print("baseline solves:", len(BASELINE_SOLVES) / len(SAMPLES))
    print("judge keeps:", judge_keeps(), "agreement:", [round(x, 3) for x in agreement()])
    for name, ids in funnel():
        kept = [SAMPLES[i - 1] for i in ids]
        print(name, ids, "accuracy", round(accuracy(kept), 3), "shares", [round(x, 3) for x in shares(kept)], "entropy", round(entropy(shares(kept)), 3))
    print("judge only:", {t: kept_by_judge_only(t) for t in (1, 2, 3, 4, 5)})
    make_figures()
