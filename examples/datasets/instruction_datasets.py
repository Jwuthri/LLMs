"""Numbers and plots for content/datasets/02-instruction-datasets.md.

Run: uv run python -m examples.datasets.instruction_datasets
"""

import math
from dataclasses import dataclass

import numpy as np

from examples.datasets.toy_dataset import SAMPLES, TOPICS, topic_counts
from examples.plotstyle import save_figure

PAGE = "instruction-datasets"

# --- Part 1: the families, and what a model sees in training ---------------------------------

BATCH_SIZE = 16  # samples in one training step (assumed)
STEPS = 1_000  # training steps (assumed)
TARGET_SEEN = 100  # samples of one topic that we want the model to see (assumed)


def shares(counts: dict[str, float]) -> dict[str, float]:
    """The share of each key in the total."""
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()}


def expected_seen(share: float, steps: int = STEPS, batch_size: int = BATCH_SIZE) -> float:
    """Expected number of samples of one topic in `steps` random batches."""
    return share * steps * batch_size


def steps_to_see(share: float, target: int = TARGET_SEEN, batch_size: int = BATCH_SIZE) -> int:
    """Smallest number of steps with an expected count of `target` samples of one topic."""
    return math.ceil(target / (share * batch_size))


def batch_without(share: float, batch_size: int = BATCH_SIZE) -> float:
    """Probability that one random batch has no sample of a topic."""
    return (1 - share) ** batch_size


# --- Part 2: a candidate list of public datasets, read from their cards ----------------------

@dataclass(frozen=True)
class Card:
    name: str
    family: str
    samples: int  # the size column of the LLM Datasets README
    licence: str  # the README, or the dataset card
    commercial: bool  # does the licence permit commercial use?
    thinking: str  # the "Thinking" column of the README ("Yes", "No", "Mixed"), "" if not read
    year: int  # the date in the README
    month: int
    mean_tokens: int  # mean tokens for each sample (ASSUMED: no card gives this number)


CARDS = [
    Card("open-perfectblend", "general", 1_420_000, "Apache-2.0", True, "No", 2024, 10, 500),
    Card("Dolci-Instruct-SFT", "general", 2_150_000, "CC-BY-NC-4.0", False, "No", 2025, 12, 500),
    Card("SYNTHETIC-2-SFT-verified", "general", 4_000_000, "permissive", True, "Yes", 2025, 6, 2_000),
    Card("MegaScience", "science", 1_250_000, "CC-BY-NC-SA-4.0", False, "", 2025, 7, 800),
    Card("NuminaMath-CoT", "math", 859_000, "Apache-2.0", True, "", 2024, 7, 600),
]
TODAY = (2026, 11)  # year and month of the page


def commercial_cards() -> list[Card]:
    """The cards that pass the licence filter for a commercial product."""
    return [card for card in CARDS if card.commercial]


def total_samples(cards: list[Card]) -> int:
    return sum(card.samples for card in cards)


def total_tokens(cards: list[Card]) -> int:
    return sum(card.samples * card.mean_tokens for card in cards)


def age_months(card: Card) -> int:
    """Months between the date of the dataset and the date of the page."""
    return (TODAY[0] - card.year) * 12 + TODAY[1] - card.month


# --- Part 3: a mix of three sources with a cap -----------------------------------------------

SOURCES = {"general": 4_000_000, "code": 1_000_000, "function calling": 100_000}  # samples (assumed)
SOURCE_TOKENS = {"general": 600, "code": 800, "function calling": 400}  # mean tokens per sample (assumed)
CAP = 500_000  # maximum number of samples from one source (assumed)
TOKENS_PER_SECOND = 2_500  # training speed, from the lesson "Training libraries" (assumed there)


def capped(sources: dict[str, int], cap: int) -> dict[str, int]:
    """Downsample each source to `cap` samples at most."""
    return {name: min(size, cap) for name, size in sources.items()}


def upsampled(sources: dict[str, int], cap: int) -> dict[str, int]:
    """Bring each source to exactly `cap` samples: cut the large ones, repeat the small ones."""
    return {name: cap for name in sources}


def mix_tokens(mix: dict[str, int]) -> int:
    """Tokens of one epoch of the mix."""
    return sum(size * SOURCE_TOKENS[name] for name, size in mix.items())


def training_hours(tokens: int, epochs: int = 1) -> float:
    return tokens * epochs / TOKENS_PER_SECOND / 3600


# --- Part 4: real conversations as a check ---------------------------------------------------

MIX_SHARES = np.array([count / len(SAMPLES) for count in topic_counts().values()])  # math, code, chat
REAL_SHARES = np.array([0.2, 0.3, 0.5])  # topic shares of real user conversations (assumed)
MIX_SIZE = 1_000  # a mix with the toy shares, at a size that gives whole numbers (assumed)
UNSEEN_SHARE = 0.1  # share of real conversations on a topic that the mix does not have (assumed)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats. Infinite when q is 0 where p is not."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if np.any((q == 0) & (p > 0)):
        return math.inf
    keep = p > 0
    return float(np.sum(p[keep] * np.log(p[keep] / q[keep])))


def weights(real: np.ndarray, mix: np.ndarray) -> np.ndarray:
    """How many times each sample of a topic must count so that the mix has the real shares."""
    return real / mix


def downsampled_counts(real: np.ndarray, mix_counts: np.ndarray) -> np.ndarray:
    """The largest mix with the real shares that repeats no sample."""
    total = np.min(mix_counts / real)
    return np.round(total * real).astype(int)


def with_unseen_topic(real: np.ndarray, unseen: float) -> np.ndarray:
    """Real shares when a share `unseen` of the conversations is on one more topic."""
    return np.append(real * (1 - unseen), unseen)


# --- Plot ------------------------------------------------------------------------------------

def make_figures() -> None:
    before = shares(SOURCES)
    after = shares(capped(SOURCES, CAP))
    names = list(SOURCES)

    def draw(ax, theme):
        x = np.arange(len(names))
        width = 0.36
        for i, (label, values) in enumerate((("No cap", before), (f"Cap of {CAP // 1000}k for each source", after))):
            heights = [values[name] for name in names]
            bars = ax.bar(x + (i - 0.5) * width, heights, width, color=theme["series"][i], label=label)
            for bar, height in zip(bars, heights):
                ax.annotate(f"{height:.3f}", (bar.get_x() + bar.get_width() / 2, height), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=8, color=theme["ink2"])
        ax.set_xticks(x, [f"{name}\n({SOURCES[name]:,} samples)" for name in names])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Source (one family each)")
        ax.set_ylabel("Share of the mix")
        ax.legend(loc="upper right")

    save_figure(draw, PAGE, "cap")


if __name__ == "__main__":
    counts = topic_counts()
    toy = shares(counts)
    print("toy counts:", counts, "shares:", toy, "toy tokens:", sum(sample.tokens for sample in SAMPLES))
    print(f"samples in {STEPS} steps of {BATCH_SIZE}:", STEPS * BATCH_SIZE,
          "expected:", {topic: expected_seen(toy[topic]) for topic in TOPICS},
          "passes over the 8 samples:", STEPS * BATCH_SIZE / len(SAMPLES))
    print("steps to see 100:", {topic: steps_to_see(toy[topic]) for topic in TOPICS})
    print("batch without the topic:", {topic: round(batch_without(toy[topic]), 6) for topic in TOPICS})

    kept = commercial_cards()
    print("cards:", len(CARDS), "samples:", total_samples(CARDS), "tokens:", total_tokens(CARDS))
    print("kept:", [card.name for card in kept], "samples:", total_samples(kept),
          "removed:", total_samples(CARDS) - total_samples(kept),
          "removed share:", round(1 - total_samples(kept) / total_samples(CARDS), 3))
    print("tokens of each kept card:", {card.name: card.samples * card.mean_tokens for card in kept},
          "total:", total_tokens(kept))
    thinking = [card for card in kept if card.thinking == "Yes"]
    print("thinking share of samples:", round(total_samples(thinking) / total_samples(kept), 3),
          "of tokens:", round(total_tokens(thinking) / total_tokens(kept), 3))
    print("age in months:", {card.name: age_months(card) for card in kept})

    for label, mix in (("no cap", SOURCES), ("cap", capped(SOURCES, CAP)), ("upsampled", upsampled(SOURCES, CAP))):
        mix_shares = shares(mix)
        tokens = mix_tokens(mix)
        print(f"{label}: {mix} total {sum(mix.values())}",
              "shares", {name: round(share, 3) for name, share in mix_shares.items()},
              "tokens", tokens, "hours", round(training_hours(tokens), 1), "days", round(training_hours(tokens) / 24, 2),
              "steps to see 100 function-calling samples:", steps_to_see(mix_shares["function calling"]))
    print("repeats of one function-calling sample in the upsampled mix:", CAP / SOURCES["function calling"])
    print("3 epochs of the capped mix, hours:", round(training_hours(mix_tokens(capped(SOURCES, CAP)), epochs=3), 1))

    print("mix shares:", MIX_SHARES, "real shares:", REAL_SHARES)
    print("KL(real || mix):", round(kl(REAL_SHARES, MIX_SHARES), 3), "KL(mix || real):", round(kl(MIX_SHARES, REAL_SHARES), 3))
    print("gaps (mix - real):", dict(zip(TOPICS, np.round(MIX_SHARES - REAL_SHARES, 3))))
    print("weights:", dict(zip(TOPICS, np.round(weights(REAL_SHARES, MIX_SHARES), 3))))
    mix_counts = MIX_SHARES * MIX_SIZE
    print("mix counts:", mix_counts, "re-weighted:", REAL_SHARES * MIX_SIZE,
          "downsampled only:", downsampled_counts(REAL_SHARES, mix_counts))
    real4 = with_unseen_topic(REAL_SHARES, UNSEEN_SHARE)
    mix4 = np.append(MIX_SHARES, 0.0)
    print("real with an unseen topic:", real4, "coverage:", 1 - UNSEEN_SHARE,
          "KL(real || mix):", kl(real4, mix4), "KL(mix || real):", round(kl(mix4, real4), 3))
    make_figures()
