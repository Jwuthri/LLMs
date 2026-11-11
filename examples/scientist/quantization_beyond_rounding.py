"""Numbers and plots for content/scientist/11-quantization-beyond-rounding.md.

Run: uv run python -m examples.scientist.quantization_beyond_rounding
"""

import math

import numpy as np

from examples.plotstyle import save_figure
from examples.scientist.quantization import CALIB_X, round_to_nearest, to_grid
from examples.toy import E, W, softmax

PAGE = "quantization-beyond-rounding"

# The baseline of lesson 7: 4-bit absmax, each weight goes to the nearest value.
W_RTN = round_to_nearest(W, bits=4)


def mean_abs_error(w: np.ndarray, w_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(w - w_hat)))


# --- The toy model: probabilities of the 3 tokens ------------------------------------------

# Assumed calibration inputs. Channel 1 is much larger than channel 2. The first one is the input of lesson 7.
CALIB = np.array([[2.0, 1.0], [4.0, -1.0], [-2.0, 1.0], [4.0, 1.0]])


def toy_probs(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Probabilities of the, cat, sat: softmax(E (W x))."""
    return softmax(E @ (w @ x))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats: how far the probabilities q are from the probabilities p."""
    return float(np.sum(p * np.log(p / q)))


def mean_kl(w: np.ndarray, w_hat: np.ndarray, calib: np.ndarray = CALIB) -> float:
    """Mean KL divergence between the original and the quantized toy model on the calibration inputs."""
    return float(np.mean([kl_divergence(toy_probs(w, x), toy_probs(w_hat, x)) for x in calib]))


def mean_output_error(w: np.ndarray, w_hat: np.ndarray, calib: np.ndarray = CALIB) -> float:
    """Mean of ||W x - W_hat x||^2 on the calibration inputs."""
    return float(np.mean(np.sum((calib @ (w - w_hat).T) ** 2, axis=1)))


# --- Part 1: importance matrix, weighted rounding, mixed precision -------------------------

def importance(calib: np.ndarray = CALIB) -> np.ndarray:
    """Mean squared activation of each input channel (the diagonal that the llama.cpp imatrix keeps)."""
    return np.mean(calib ** 2, axis=0)


def column_squared_error(w: np.ndarray, w_hat: np.ndarray) -> np.ndarray:
    """Squared rounding error, added for each column (input channel)."""
    return np.sum((w - w_hat) ** 2, axis=0)


def weighted_error(w: np.ndarray, w_hat: np.ndarray, imp: np.ndarray) -> float:
    """Sum over the columns of importance x squared rounding error."""
    return float(column_squared_error(w, w_hat) @ imp)


STEP_CANDIDATES = [0.150, 0.155, 0.160, 0.165, 0.170, float(np.max(np.abs(W)) / 7)]


def weighted_rounding(w: np.ndarray, imp: np.ndarray, bits: int = 4) -> dict:
    """Try each grid step and keep the one with the smallest importance-weighted error."""
    errors = [weighted_error(w, to_grid(w, step, bits), imp) for step in STEP_CANDIDATES]
    best = int(np.argmin(errors))
    step = STEP_CANDIDATES[best]
    return {"errors": errors, "step": step, "q": np.round(to_grid(w, step, bits) / step).astype(int),
            "w_hat": to_grid(w, step, bits)}


# Three toy layers of the same size. Assumed: the KL divergence that each layer adds when only this
# layer has 4 bits. Assumed too: one more bit divides this error by 4, and the errors of the layers add up.
SENSITIVITY_AT_4_BITS = [0.064, 0.016, 0.004]
MIN_BITS, MAX_BITS = 2, 8


def layer_error(sensitivity: float, bits: float) -> float:
    """Error of one layer at a bit width: one more bit makes the step 2 times smaller, the squared error 4 times."""
    return sensitivity * 4.0 ** (4 - bits)


def total_error(bits: list[float], sensitivities: list[float] = SENSITIVITY_AT_4_BITS) -> float:
    return sum(layer_error(s, b) for s, b in zip(sensitivities, bits))


def allocate_bits(total_bits: int, sensitivities: list[float] = SENSITIVITY_AT_4_BITS) -> list[int]:
    """Start each layer at MIN_BITS. Give one bit at a time to the layer with the largest error."""
    bits = [MIN_BITS] * len(sensitivities)
    for _ in range(total_bits - MIN_BITS * len(sensitivities)):
        open_layers = [i for i in range(len(bits)) if bits[i] < MAX_BITS]
        worst = max(open_layers, key=lambda i: (layer_error(sensitivities[i], bits[i]), -i))
        bits[worst] += 1
    return bits


# --- Part 2: quantization-aware training with the straight-through estimator ---------------

QAT_STEP = 0.625                 # assumed coarse grid: 2 bits, integers -2..1, values -1.25, -0.625, 0, 0.625
QAT_QMIN, QAT_QMAX = -2, 1
QAT_LR = 0.008
QAT_ROW = W[1]                   # one neuron: row 2 of W
QAT_TARGET = float(QAT_ROW @ CALIB_X)   # the output of the full-precision teacher: 1.9


def fake_quantize(v: np.ndarray) -> np.ndarray:
    """Round the latent weights to the grid in the forward pass."""
    return np.clip(np.round(v / QAT_STEP), QAT_QMIN, QAT_QMAX) * QAT_STEP


def qat_steps(steps: int = 4, x: np.ndarray = CALIB_X, lr: float = QAT_LR) -> list[dict]:
    """Gradient steps on the loss (w_hat . x - teacher)^2 / 2 with the straight-through estimator."""
    latent = QAT_ROW.copy()
    history = []
    for step in range(steps + 1):
        w_hat = fake_quantize(latent)
        error = float(w_hat @ x) - QAT_TARGET
        gradient = error * x          # gradient for w_hat; the STE uses it for the latent weights
        history.append({"step": step, "latent": latent.copy(), "w_hat": w_hat, "output": float(w_hat @ x),
                        "error": error, "gradient": gradient})
        latent = latent - lr * gradient
    return history


# --- Part 3: ternary weights with absmean scaling ------------------------------------------

BITS_TERNARY = math.log2(3)


def ternarize(w: np.ndarray) -> dict:
    """BitNet b1.58: scale = mean absolute weight, then round to -1, 0 or 1."""
    scale = float(np.mean(np.abs(w)))
    q = np.clip(np.round(w / scale), -1, 1).astype(int)
    return {"scale": scale, "scaled": w / scale, "q": q, "w_hat": q * scale}


def size_gb(params: float, bits_per_weight: float) -> float:
    return params * bits_per_weight / 8 / 1e9


# --- Part 4: vector quantization with a codebook -------------------------------------------

# Pairs of weights. The first 2 are the rows of W. The other 6 are assumed: the other pairs of the layer.
PAIRS = np.array([[0.5, -1.2], [0.8, 0.3], [0.6, -1.0], [0.9, 0.4],
                  [-0.7, 0.2], [-0.5, 0.1], [-0.2, -0.9], [-0.1, -0.7]])
KMEANS_INIT = [0, 1, 4, 6]       # the pairs that start as the 4 codebook vectors


def assign(pairs: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Index of the nearest codebook vector for each pair."""
    distances = np.linalg.norm(pairs[:, None, :] - codebook[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def kmeans(pairs: np.ndarray = PAIRS, init: list[int] = KMEANS_INIT, steps: int = 3) -> dict:
    """A few k-means steps: assign each pair to its nearest vector, then move each vector to the mean of its pairs."""
    codebook = pairs[init].copy()
    history = [codebook.copy()]
    for _ in range(steps):
        index = assign(pairs, codebook)
        codebook = np.array([pairs[index == k].mean(axis=0) for k in range(len(init))])
        history.append(codebook.copy())
    index = assign(pairs, codebook)
    return {"codebook": codebook, "index": index, "pairs_hat": codebook[index], "history": history}


def sign_quantize(w: np.ndarray) -> dict:
    """1-bit scalar quantization: the sign of each weight times the mean absolute value."""
    scale = float(np.mean(np.abs(w)))
    return {"scale": scale, "w_hat": np.sign(w) * scale}


def codebook_bpw(codebook_size: int, group: int) -> float:
    """Bits per weight of the indices."""
    return math.log2(codebook_size) / group


def codebook_bits(codebook_size: int, group: int, value_bits: int = 16) -> int:
    """Storage of the codebook itself, with FP16 values."""
    return codebook_size * group * value_bits


def total_bpw(weights: int, codebook_size: int, group: int) -> float:
    """Indices plus the codebook, for each weight of the layer."""
    return codebook_bpw(codebook_size, group) + codebook_bits(codebook_size, group) / weights


REAL_LAYER = 4096 * 4096         # assumed size of one weight matrix of a real model


# --- Plot ----------------------------------------------------------------------------------

BUDGETS = list(range(6, 19))     # total bits for the 3 layers: a mean of 2 to 6 bits


def make_figures() -> None:
    mean_bits = [t / 3 for t in BUDGETS]
    equal = [total_error([m] * 3) for m in mean_bits]
    mixed = [total_error(allocate_bits(t)) for t in BUDGETS]

    def draw(ax, theme):
        ax.plot(mean_bits, equal, marker="o", color=theme["series"][0], label="Equal bits for the 3 layers")
        ax.plot(mean_bits, mixed, marker="s", color=theme["series"][1], label="Bits given by sensitivity")
        ax.annotate(f"{equal[6]:.3f}", (4, equal[6]), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=theme["ink2"])
        ax.annotate(f"{mixed[6]:.3f}", (4, mixed[6]), textcoords="offset points", xytext=(-30, -12),
                    fontsize=8, color=theme["ink2"])
        ax.set_yscale("log")
        ax.set_xticks(range(2, 7))
        ax.set_xlabel("Mean bits for each weight")
        ax.set_ylabel("Total output error (KL, log scale)")
        ax.legend(loc="upper right")

    save_figure(draw, PAGE, "bit-budget")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("baseline 4-bit:", W_RTN.ravel(), "mae", mean_abs_error(W, W_RTN))

    imp = importance()
    print("--- part 1: importance", imp)
    print("column squared error:", column_squared_error(W, W_RTN), "sum", column_squared_error(W, W_RTN).sum())
    print("share of column 1, plain:", column_squared_error(W, W_RTN)[0] / column_squared_error(W, W_RTN).sum())
    weighted_cols = column_squared_error(W, W_RTN) * imp
    print("weighted by column:", weighted_cols, "sum", weighted_cols.sum(), "share col 1", weighted_cols[0] / weighted_cols.sum())
    wr = weighted_rounding(W, imp)
    print("candidates:", STEP_CANDIDATES, "weighted errors:", np.array(wr["errors"]))
    print("best step", wr["step"], "q", wr["q"].ravel(), "w_hat", wr["w_hat"].ravel())
    print("plain sq error:", column_squared_error(W, wr["w_hat"]).sum(), "mae", mean_abs_error(W, wr["w_hat"]))
    print("mean output error: rtn", mean_output_error(W, W_RTN), "weighted", mean_output_error(W, wr["w_hat"]))
    print("probs x=[2,1]: original", toy_probs(W, CALIB_X), "rtn", toy_probs(W_RTN, CALIB_X))
    print("KL per input rtn:", [kl_divergence(toy_probs(W, x), toy_probs(W_RTN, x)) for x in CALIB])
    print("KL per input weighted:", [kl_divergence(toy_probs(W, x), toy_probs(wr["w_hat"], x)) for x in CALIB])
    print("mean KL: rtn", mean_kl(W, W_RTN), "weighted", mean_kl(W, wr["w_hat"]))
    print("layer errors equal:", [layer_error(s, 4) for s in SENSITIVITY_AT_4_BITS], total_error([4, 4, 4]))
    alloc = allocate_bits(12)
    print("allocation:", alloc, [layer_error(s, b) for s, b in zip(SENSITIVITY_AT_4_BITS, alloc)], total_error(alloc))
    for t in BUDGETS:
        print(f"  mean {t / 3:.3f}: bits {allocate_bits(t)}, mixed {total_error(allocate_bits(t)):.5f}, equal {total_error([t / 3] * 3):.5f}")

    print("--- part 2: teacher output", QAT_TARGET, "boundary", QAT_STEP / 2)
    for h in qat_steps():
        print(f"step {h['step']}: latent {h['latent']}, w_hat {h['w_hat']}, output {h['output']:.4f},",
              f"error {h['error']:.4f}, gradient {h['gradient']}")

    t = ternarize(W)
    print("--- part 3: scale", t["scale"], "scaled", t["scaled"].ravel(), "q", t["q"].ravel(), "w_hat", t["w_hat"].ravel())
    print("abs error", np.abs(W - t["w_hat"]).ravel(), "mae", mean_abs_error(W, t["w_hat"]),
          "ratio to 4-bit", mean_abs_error(W, t["w_hat"]) / mean_abs_error(W, W_RTN))
    print("output", t["w_hat"] @ CALIB_X, "error", np.linalg.norm(W @ CALIB_X - t["w_hat"] @ CALIB_X), "mean KL", mean_kl(W, t["w_hat"]))
    print("bits", BITS_TERNARY, "4 / bits", 4 / BITS_TERNARY, "8B GB:", size_gb(8e9, BITS_TERNARY), size_gb(8e9, 2), size_gb(8e9, 4))

    km = kmeans()
    print("--- part 4: codebook after each step")
    for c in km["history"]:
        print(c.ravel())
    print("index", km["index"], "W_hat", km["pairs_hat"][:2].ravel())
    print("vq abs error W", np.abs(W - km["pairs_hat"][:2]).ravel(), "mae W", mean_abs_error(W, km["pairs_hat"][:2]),
          "mae all", mean_abs_error(PAIRS, km["pairs_hat"]), "mean KL", mean_kl(W, km["pairs_hat"][:2]))
    sq = sign_quantize(PAIRS)
    print("sign scale", sq["scale"], "W_hat", sq["w_hat"][:2].ravel(), "mae W", mean_abs_error(W, sq["w_hat"][:2]),
          "mae all", mean_abs_error(PAIRS, sq["w_hat"]), "mean KL", mean_kl(W, sq["w_hat"][:2]))
    print("bpw", codebook_bpw(4, 2), "codebook bits", codebook_bits(4, 2), "toy total bpw", total_bpw(PAIRS.size, 4, 2),
          "real layer", total_bpw(REAL_LAYER, 4, 2))
    print("2^16 x 8:", codebook_bpw(2 ** 16, 8), codebook_bits(2 ** 16, 8), "extra bpw", codebook_bits(2 ** 16, 8) / REAL_LAYER)
    make_figures()
