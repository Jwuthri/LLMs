"""Numbers and plots for docs/scientist/01-architecture.md.

Run: uv run python -m examples.scientist.architecture
"""

from collections import Counter

import numpy as np

from examples.plotstyle import save_light_dark
from examples.toy import LOGITS, VOCAB, E, encode, softmax

PAGE = "architecture"


# --- Tokenization: byte-pair encoding (BPE) on a 4-word corpus -----------------------------

BPE_CORPUS = {"cat": 2, "sat": 1, "the": 1}  # word -> count


def bpe_pair_counts(words: dict[tuple[str, ...], int]) -> Counter:
    """Count each pair of adjacent symbols, weighted by the word count."""
    counts: Counter = Counter()
    for symbols, count in words.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += count
    return counts


def bpe_merge(words: dict[tuple[str, ...], int], pair: tuple[str, str]) -> dict[tuple[str, ...], int]:
    """Replace each occurrence of `pair` with one merged symbol."""
    merged: dict[tuple[str, ...], int] = {}
    for symbols, count in words.items():
        out: list[str] = []
        i = 0
        while i < len(symbols):
            if symbols[i:i + 2] == pair:
                out.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        merged[tuple(out)] = count
    return merged


def bpe_train(corpus: dict[str, int], num_merges: int) -> tuple[list[tuple[str, str]], list[Counter]]:
    """Return the merges in order, and the pair counts seen before each merge."""
    words = {tuple(word): count for word, count in corpus.items()}
    merges, history = [], []
    for _ in range(num_merges):
        counts = bpe_pair_counts(words)
        history.append(counts)
        best = max(counts, key=lambda pair: (counts[pair], pair))  # ties: last pair in a-z order
        merges.append(best)
        words = bpe_merge(words, best)
    return merges, history


# --- Attention: one head, no learned projections (W_Q = W_K = W_V = identity) --------------

def attention(x: np.ndarray, causal: bool = True) -> dict[str, np.ndarray]:
    """Scaled dot-product self-attention. Returns each intermediate matrix."""
    d_k = x.shape[-1]
    raw = x @ x.T
    scores = raw / np.sqrt(d_k)
    masked = scores.copy()
    if causal:
        masked[np.triu(np.ones_like(scores, dtype=bool), k=1)] = -np.inf
    weights = softmax(masked)
    return {"raw": raw, "scores": scores, "weights": weights, "output": weights @ x}


# --- Sampling ------------------------------------------------------------------------------

def top_k(probs: np.ndarray, k: int) -> np.ndarray:
    """Keep the k most probable tokens, then renormalize."""
    kept = np.zeros_like(probs)
    best = np.argsort(probs)[::-1][:k]
    kept[best] = probs[best]
    return kept / kept.sum()


def top_p(probs: np.ndarray, p: float) -> np.ndarray:
    """Keep the smallest set of tokens with a total probability >= p, then renormalize."""
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    cutoff = int(np.searchsorted(cumulative, p)) + 1
    kept = np.zeros_like(probs)
    kept[order[:cutoff]] = probs[order[:cutoff]]
    return kept / kept.sum()


# Assumed step-2 probabilities for the beam search example: P(second token | first token).
BEAM_STEP2 = {
    "sat": {"the": 0.34, "cat": 0.33, "sat": 0.33},
    "cat": {"the": 0.03, "cat": 0.02, "sat": 0.95},
}


def beam_scores(step1: np.ndarray, beams: int = 2) -> list[tuple[str, float]]:
    """Score each 2-token continuation of the `beams` best first tokens. Sorted, best first."""
    first = [VOCAB[i] for i in np.argsort(step1)[::-1][:beams]]
    scored = [
        (f"{a} {b}", float(step1[VOCAB.index(a)] * p))
        for a in first
        for b, p in BEAM_STEP2[a].items()
    ]
    return sorted(scored, key=lambda item: -item[1])


# --- Plot ----------------------------------------------------------------------------------

def make_figures() -> None:
    temperatures = [0.5, 1.0, 2.0]

    def draw(ax, theme):
        width = 0.26
        for i, t in enumerate(temperatures):
            probs = softmax(LOGITS, t)
            xs = np.arange(len(VOCAB)) + (i - 1) * (width + 0.02)
            bars = ax.bar(xs, probs, width, color=theme["series"][i], label=f"T = {t}")
            for rect, p in zip(bars, probs):
                ax.text(rect.get_x() + rect.get_width() / 2, p + 0.015, f"{p:.2f}",
                        ha="center", va="bottom", fontsize=8, color=theme["ink2"])
        ax.set_xticks(np.arange(len(VOCAB)), VOCAB)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability of the next token")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")

    save_light_dark(draw, PAGE, "temperature")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("ids:", encode("the cat sat"))
    merges, history = bpe_train(BPE_CORPUS, 3)
    for merge, counts in zip(merges, history):
        print("pair counts:", dict(counts), "-> merge", merge)
    for name, value in attention(E).items():
        print(name, value, sep="\n")
    for t in (0.5, 1.0, 2.0):
        print(f"softmax T={t}:", softmax(LOGITS, t))
    probs = softmax(LOGITS)
    print("top-k 2:", top_k(probs, 2))
    print("top-p 0.9:", top_p(probs, 0.9))
    print("beam:", beam_scores(probs))
    make_figures()
