"""Numbers and plots for content/datasets/03-preference-datasets.md.

Run: uv run python -m examples.datasets.preference_datasets
"""

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from examples.datasets.toy_dataset import SAMPLES, TOPICS, topic_counts
from examples.plotstyle import save_figure

PAGE = "preference-datasets"
Z_95 = 1.96

# --- Part 1: what a pair is ------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    sample_id: int  # the toy sample that gives the prompt
    chosen: str
    rejected: str
    q_chosen: float  # quality of the chosen answer, from 0 (useless) to 1 (perfect); assumed
    q_rejected: float  # quality of the rejected answer; assumed

    @property
    def prompt(self) -> str:
        return SAMPLES[self.sample_id - 1].instruction

    @property
    def topic(self) -> str:
        return SAMPLES[self.sample_id - 1].topic

    @property
    def margin(self) -> float:
        return self.q_chosen - self.q_rejected


PAIRS = [
    Pair(4, "3", "4", 1.0, 0.0),
    Pair(8, "The cat is a small pet. It sat.", "I cannot help with that.", 0.8, 0.1),
    Pair(5, "len(text.split())", "text.count(' ') + 1", 0.9, 0.8),
    Pair(1, "3 or 4", "5", 0.3, 0.0),
]
MIN_MARGIN = 0.2  # a filter drops the pairs with a smaller margin (assumed)


def mean_margin(pairs: list[Pair]) -> float:
    return float(np.mean([pair.margin for pair in pairs]))


def mean_chosen_quality(pairs: list[Pair]) -> float:
    return float(np.mean([pair.q_chosen for pair in pairs]))


def filter_by_margin(pairs: list[Pair], min_margin: float = MIN_MARGIN) -> list[Pair]:
    return [pair for pair in pairs if pair.margin >= min_margin]


# --- Part 2: where pairs come from -----------------------------------------------------------

# Scores of a judge model (scale 1 to 5) for 3 answers to "Tell me about the cat" (assumed).
JUDGE_SCORES = {"A": 4.5, "B": 4.0, "C": 2.0}


def all_pairs(scores: dict[str, float]) -> list[tuple[str, str, float]]:
    """Each (chosen, rejected, margin) that the scores permit. The chosen answer has the higher score."""
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [(a, b, scores[a] - scores[b]) for a, b in combinations(ranked, 2)]


def binarize(scores: dict[str, float], rule: str) -> list[tuple[str, str, float]]:
    """Change scores into pairs. Rules: best_worst, best_second, best_random (each other answer), all."""
    ranked = sorted(scores, key=scores.get, reverse=True)
    best = ranked[0]
    if rule == "best_worst":
        others = [ranked[-1]]
    elif rule == "best_second":
        others = [ranked[1]]
    elif rule == "best_random":
        others = ranked[1:]  # each other answer has the same probability
    elif rule == "all":
        return all_pairs(scores)
    else:
        raise ValueError(rule)
    return [(best, other, scores[best] - scores[other]) for other in others]


def mean_pair_margin(scores: dict[str, float], rule: str) -> float:
    return float(np.mean([margin for _, _, margin in binarize(scores, rule)]))


def generated_answers_per_pair() -> dict[str, float]:
    """Answers that a model must generate for one pair, for each source."""
    k = len(JUDGE_SCORES)
    return {
        "human vote": 2 / 1,  # two models answer, one vote
        "judge, best against worst": k / 1,  # k answers, one pair
        "judge, all pairs": k / math.comb(k, 2),  # k answers, k(k-1)/2 pairs
        "construction": 1 / 1,  # the correct answer exists; generate one answer with an error
    }


# --- Part 3: label noise ---------------------------------------------------------------------

TRUE_RATE = 0.8  # true rate at which answer type A is better than type B (assumed)
SWAP_SHARES = [0.0, 0.1, 0.2, 0.3]
FIRST_BIAS = 0.6  # a judge chooses the first answer shown in 60% of the equal pairs (assumed)


def measured_rate(true_rate: float, swapped: float) -> float:
    """Rate in the labels when a share `swapped` of the labels has chosen and rejected exchanged."""
    return true_rate * (1 - swapped) + (1 - true_rate) * swapped


def corrected_rate(measured: float, swapped: float) -> float:
    """Invert `measured_rate`. Not defined at swapped = 0.5."""
    return (measured - swapped) / (1 - 2 * swapped)


def bits_per_label(swapped: float) -> float:
    """Information that one label carries about the true order: 1 - H(swapped), in bits."""
    if swapped in (0.0, 1.0):
        return 1.0
    entropy = -swapped * math.log2(swapped) - (1 - swapped) * math.log2(1 - swapped)
    return 1 - entropy


def pairs_for_one_clean_pair(swapped: float) -> float:
    """Number of noisy pairs that carry the information of one pair without noise."""
    return 1 / bits_per_label(swapped)


def position_bias(first_bias: float = FIRST_BIAS) -> dict[str, float]:
    """Preference for A on pairs where A and B are equally good (true preference: 0.5)."""
    a_first = first_bias  # A is shown first: the judge chooses A with probability `first_bias`
    b_first = 1 - first_bias  # B is shown first
    return {
        "a_always_first": a_first,
        "b_always_first": b_first,
        "both_orders": (a_first + b_first) / 2,
        "same_winner_in_both_orders": a_first * b_first + b_first * a_first,
        "order_decides": a_first * a_first + b_first * b_first,
    }


# --- Part 4: how many pairs ------------------------------------------------------------------

RATE = 0.7  # a measured preference rate (assumed)
PAIR_COUNTS = [100, 1_000, 10_000]
TOPIC_PAIRS = {"math": 800, "code": 150, "chat": 50}  # pairs for each topic in a set of 1,000 (assumed)


def rate_interval(rate: float, pairs: int) -> tuple[float, float, float]:
    """Half width and limits of the 95% interval of a rate (normal approximation)."""
    half = Z_95 * math.sqrt(rate * (1 - rate) / pairs)
    return half, rate - half, rate + half


def pairs_needed_exact(rate: float, reference: float = 0.5) -> float:
    """Number of pairs for which the half width of the 95% interval of `rate` is `rate - reference`."""
    return Z_95**2 * rate * (1 - rate) / (rate - reference) ** 2


def pairs_needed(rate: float, reference: float = 0.5) -> int:
    """Smallest number of pairs for which the 95% interval of `rate` does not contain `reference`."""
    return math.ceil(pairs_needed_exact(rate, reference))


def topic_shares(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {topic: counts[topic] / total for topic in TOPICS}


# --- Plot ------------------------------------------------------------------------------------

def make_figures() -> None:
    pairs = np.logspace(1.5, 4.5, 200)

    def draw_interval(ax, theme):
        half = Z_95 * np.sqrt(RATE * (1 - RATE) / pairs)
        ax.plot(pairs, np.full_like(pairs, RATE), color=theme["series"][0], label=f"Measured rate {RATE}")
        ax.plot(pairs, RATE + half, color=theme["series"][1], label="Limits of the 95% interval")
        ax.plot(pairs, RATE - half, color=theme["series"][1])
        ax.fill_between(pairs, RATE - half, RATE + half, color=theme["series"][1], alpha=0.12, linewidth=0)
        ax.axhline(0.5, color=theme["muted"], linewidth=1, linestyle="--")
        ax.annotate("0.5: no preference", (pairs[-1], 0.5), textcoords="offset points", xytext=(-4, 4),
                    ha="right", fontsize=8, color=theme["ink2"])
        for count in PAIR_COUNTS:
            _, low, high = rate_interval(RATE, count)
            ax.plot([count, count], [low, high], "o", color=theme["series"][1], markersize=4)
            ax.annotate(f"{high:.3f}", (count, high), textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=8, color=theme["ink2"])
            ax.annotate(f"{low:.3f}", (count, low), textcoords="offset points", xytext=(0, -12), ha="center",
                        fontsize=8, color=theme["ink2"])
        ax.set_xscale("log")
        ax.set_ylim(0.45, 0.95)
        ax.set_xlabel("Number of pairs (log scale)")
        ax.set_ylabel("Preference rate")
        ax.legend(loc="upper right", fontsize=8)

    save_figure(draw_interval, PAGE, "interval")


if __name__ == "__main__":
    for pair in PAIRS:
        print(f"pair sample {pair.sample_id} ({pair.topic}): '{pair.prompt}' chosen '{pair.chosen}'"
              f" ({pair.q_chosen}) rejected '{pair.rejected}' ({pair.q_rejected}) margin {pair.margin:.3f}")
    kept = filter_by_margin(PAIRS)
    print("mean margin:", round(mean_margin(PAIRS), 3), "mean chosen quality:", round(mean_chosen_quality(PAIRS), 3))
    print("after the margin filter:", len(kept), "pairs, mean margin", round(mean_margin(kept), 3))

    print("all pairs:", [(a, b, round(m, 3)) for a, b, m in all_pairs(JUDGE_SCORES)])
    for rule in ("best_worst", "best_second", "best_random", "all"):
        print(f"  {rule}: mean margin {mean_pair_margin(JUDGE_SCORES, rule):.3f}")
    print("generated answers per pair:", {k: round(v, 3) for k, v in generated_answers_per_pair().items()})

    for swapped in SWAP_SHARES:
        seen = measured_rate(TRUE_RATE, swapped)
        print(f"swapped {swapped}: measured {seen:.3f}, corrected {corrected_rate(seen, swapped):.3f},"
              f" bits per label {bits_per_label(swapped):.3f}")
    print("swapped 0.5: measured", round(measured_rate(TRUE_RATE, 0.5), 3), "bits", round(bits_per_label(0.5), 3))
    print("pairs for one clean pair at 0.2:", round(pairs_for_one_clean_pair(0.2), 1))
    print("position bias:", {k: round(v, 3) for k, v in position_bias().items()})
    seen = measured_rate(TRUE_RATE, 0.1)
    half, low, high = rate_interval(seen, 100)
    print(f"widget start: measured {seen:.3f}, half width {half:.3f}, interval [{low:.3f}, {high:.3f}]")

    for count in PAIR_COUNTS:
        half, low, high = rate_interval(RATE, count)
        print(f"rate {RATE}, {count} pairs: half width {half:.3f}, interval [{low:.3f}, {high:.3f}]")
    print("pairs to tell 0.55 from 0.5:", pairs_needed(0.55), "exact", round(pairs_needed_exact(0.55), 1),
          "and 0.7 from 0.5:", pairs_needed(0.7))
    print("topic shares, toy dataset:", topic_shares(topic_counts()), "pair set:", topic_shares(TOPIC_PAIRS))
    for topic, count in TOPIC_PAIRS.items():
        half, low, high = rate_interval(RATE, count)
        print(f"  {topic}: {count} pairs, half width {half:.3f}, interval [{low:.3f}, {high:.3f}]")
    make_figures()
