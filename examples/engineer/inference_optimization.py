"""Numbers and plots for docs/engineer/06-inference-optimization.md.

Run: uv run python -m examples.engineer.inference_optimization
"""

import numpy as np

from examples.plotstyle import save_light_dark
from examples.scientist.architecture import attention
from examples.toy import LOGITS, E, softmax

PAGE = "inference-optimization"

# Llama 3 8B (see Sources on the page).
LLAMA3_8B = {"layers": 32, "heads": 32, "kv_heads": 8, "head_dim": 128, "context": 8192}
FP16_BYTES = 2
GIB = 2 ** 30


# --- Flash Attention: the softmax of one row, one block of scores at a time ----------------

def attention_matrix_bytes(tokens: int, heads: int, bytes_per_value: int = FP16_BYTES) -> int:
    """Memory of the full tokens x tokens score matrix of one layer."""
    return tokens * tokens * heads * bytes_per_value


def online_softmax(scores: np.ndarray, values: np.ndarray, block_size: int) -> list[dict]:
    """Attention output of one query, with the keys read in blocks.

    The state after each block is the running maximum `m`, the running sum `l` of
    exp(score - m), and the running weighted sum `o` of the values (not yet divided by l).
    """
    m, l, o = -np.inf, 0.0, np.zeros(values.shape[1])
    states = []
    for start in range(0, len(scores), block_size):
        block = scores[start:start + block_size]
        m_new = max(m, block.max())
        rescale = np.exp(m - m_new)  # corrects the old sums for the new maximum
        weights = np.exp(block - m_new)
        l = rescale * l + weights.sum()
        o = rescale * o + weights @ values[start:start + block_size]
        m = m_new
        states.append({"block": block, "m": m, "rescale": rescale, "l": l, "o": o.copy()})
    states[-1]["output"] = o / l
    return states


# --- Key-value cache: count the work of n forward passes -----------------------------------

def kv_cache_counts(n: int) -> dict[str, int]:
    """Totals for n passes on contexts of 1, 2, ..., n tokens (one layer, one head).

    `kv`: tokens for which the key and the value are computed.
    `dots`: query-key dot products (a query reads only itself and the earlier tokens).
    """
    passes = range(1, n + 1)
    return {
        "kv_no_cache": sum(t for t in passes),                  # recompute each token
        "kv_cache": sum(1 for _ in passes),                     # only the new token
        "dots_no_cache": sum(t * (t + 1) // 2 for t in passes),  # each row of the matrix again
        "dots_cache": sum(t for t in passes),                   # only the new row
    }


def kv_cache_formulas(n: int) -> dict[str, int]:
    """The closed forms of `kv_cache_counts`."""
    return {
        "kv_no_cache": n * (n + 1) // 2,
        "kv_cache": n,
        "dots_no_cache": n * (n + 1) * (n + 2) // 6,
        "dots_cache": n * (n + 1) // 2,
    }


def kv_cache_bytes(layers: int, kv_heads: int, head_dim: int, tokens: int,
                   bytes_per_value: int = FP16_BYTES) -> int:
    """2 (key and value) x layers x heads x head_dim x tokens x bytes."""
    return 2 * layers * kv_heads * head_dim * tokens * bytes_per_value


# --- Speculative decoding ------------------------------------------------------------------

DRAFT_PROBS = np.array([0.2, 0.3, 0.5])  # assumed distribution of the small draft model


def acceptance(target: np.ndarray, draft: np.ndarray) -> np.ndarray:
    """Probability to keep a draft token: min(1, p_target / p_draft)."""
    return np.minimum(1.0, target / draft)


def acceptance_rate(target: np.ndarray, draft: np.ndarray) -> float:
    """Probability that one draft token is kept: sum of min(p_target, p_draft)."""
    return float(np.minimum(target, draft).sum())


def residual(target: np.ndarray, draft: np.ndarray) -> np.ndarray:
    """Distribution of the replacement token after a rejection."""
    extra = np.maximum(0.0, target - draft)
    return extra / extra.sum()


def output_distribution(target: np.ndarray, draft: np.ndarray) -> np.ndarray:
    """Distribution of the token that speculative decoding emits. It equals `target`."""
    kept = draft * acceptance(target, draft)
    return kept + (1 - acceptance_rate(target, draft)) * residual(target, draft)


def expected_accepted(a: float, k: int) -> float:
    """Expected number of draft tokens kept from a draft of k tokens: a + a^2 + ... + a^k."""
    return a * (1 - a ** k) / (1 - a)


def expected_tokens(a: float, k: int) -> float:
    """Expected tokens for one pass of the large model: the kept draft tokens plus 1."""
    return (1 - a ** (k + 1)) / (1 - a)


def speedup(a: float, k: int, c: float) -> float:
    """Time of normal decoding / time of speculative decoding. c = draft cost / target cost."""
    return expected_tokens(a, k) / (k * c + 1)


# --- Plot ----------------------------------------------------------------------------------

def make_figures() -> None:
    lengths = np.arange(1, 21)
    cost = 0.05

    def draw(ax, theme):
        for i, a in enumerate([0.9, 0.8, 0.6]):
            values = [speedup(a, int(k), cost) for k in lengths]
            ax.plot(lengths, values, marker="o", markersize=4, color=theme["series"][i],
                    label=f"acceptance a = {a}")
            best = int(np.argmax(values))
            ax.annotate(f"{values[best]:.2f}x", (lengths[best], values[best]),
                        textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8,
                        color=theme["ink2"])
        ax.axhline(1.0, color=theme["axis"], linewidth=1)
        ax.set_xticks(lengths[1::2])
        ax.set_ylim(0, 6.4)
        ax.set_xlabel("Draft length k (tokens)")
        ax.set_ylabel("Speed-up (draft cost c = 0.05)")
        ax.legend(loc="upper left", ncols=3, fontsize=8, columnspacing=1.2)

    save_light_dark(draw, PAGE, "speculative-speedup")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    result = attention(E)
    row = result["scores"][2]
    print("scores of 'sat':", row, "| weights:", result["weights"][2], "| output:", result["output"][2])
    for i, state in enumerate(online_softmax(row, E, block_size=2), start=1):
        print(f"block {i}: scores {state['block']} rescale {state['rescale']:.3f} "
              f"m {state['m']:.3f} l {state['l']:.3f} o {state['o']}")
        if "output" in state:
            print("online output:", state["output"], "| weights:",
                  np.array([np.exp(s - state["m"]) / state["l"] for s in row]))
    n = LLAMA3_8B["context"]
    print("pairs for 8,192 tokens:", n * n, "| matrix GiB (32 heads, FP16):",
          attention_matrix_bytes(n, LLAMA3_8B["heads"]) / GIB, "| flash state numbers per head:", 2 * n)

    for n in (3, 1000):
        print(f"n = {n}:", kv_cache_counts(n))
    print("ratio of dot products at n = 1000:",
          kv_cache_counts(1000)["dots_no_cache"] / kv_cache_counts(1000)["dots_cache"])
    print("toy cache numbers:", kv_cache_bytes(1, 1, 2, 3, bytes_per_value=1))
    for name, heads in (("MHA", 32), ("GQA", 8), ("MQA", 1)):
        per_token = kv_cache_bytes(LLAMA3_8B["layers"], heads, LLAMA3_8B["head_dim"], 1)
        full = kv_cache_bytes(LLAMA3_8B["layers"], heads, LLAMA3_8B["head_dim"], LLAMA3_8B["context"])
        print(f"{name} ({heads} key-value heads): {per_token:,} bytes per token, {full / GIB:.3f} GiB at 8,192")

    target = softmax(LOGITS)
    print("target:", target, "| draft:", DRAFT_PROBS)
    print("acceptance:", acceptance(target, DRAFT_PROBS))
    a = acceptance_rate(target, DRAFT_PROBS)
    print("acceptance rate a:", round(a, 3), "| residual:", residual(target, DRAFT_PROBS))
    print("output distribution:", output_distribution(target, DRAFT_PROBS))
    print("k = 4: accepted", round(expected_accepted(a, 4), 3), "| tokens per pass",
          round(expected_tokens(a, 4), 3), "| speed-up at c = 0.05:", round(speedup(a, 4, 0.05), 3))
    print("speed-up with a = 0.5, k = 4, c = 0.3:", round(speedup(0.5, 4, 0.3), 3))
    for rate in (0.9, 0.8, 0.6):
        best = max(range(1, 21), key=lambda k: speedup(rate, k, 0.05))
        print(f"best draft length for a = {rate}: k = {best}, speed-up {speedup(rate, best, 0.05):.2f}")
    make_figures()
