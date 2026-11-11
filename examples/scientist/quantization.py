"""Numbers and plots for content/scientist/07-quantization.md.

Run: uv run python -m examples.scientist.quantization
"""

import numpy as np

from examples.plotstyle import save_figure
from examples.toy import W

PAGE = "quantization"


# --- Precision levels ----------------------------------------------------------------------

PARAMS = 8_000_000_000  # an 8-billion-parameter model
BITS_PER_WEIGHT = {"FP32": 32, "FP16": 16, "INT8": 8, "4-bit": 4}


def model_size_gb(params: int, bits_per_weight: float) -> float:
    """Size of the weights in gigabytes (1 GB = 10^9 bytes)."""
    return params * bits_per_weight / 8 / 1e9


# --- Base techniques: absmax (symmetric) and zero-point (asymmetric) -----------------------

def absmax_quantize(w: np.ndarray, bits: int = 8) -> dict:
    """Symmetric quantization: the largest absolute weight maps to the largest integer."""
    qmax = 2 ** (bits - 1) - 1
    scale = qmax / np.max(np.abs(w))
    scaled = scale * w
    q = np.round(scaled).astype(int)
    dequant = q / scale
    return {"scale": scale, "zero_point": 0, "scaled": scaled, "q": q, "dequant": dequant,
            "error": np.abs(w - dequant), "mae": float(np.mean(np.abs(w - dequant)))}


def zeropoint_quantize(w: np.ndarray, bits: int = 8) -> dict:
    """Asymmetric quantization: min(w) maps to the smallest integer, max(w) to the largest."""
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale = (qmax - qmin) / (w.max() - w.min())
    zero_point = int(np.round(-scale * w.min())) + qmin
    scaled = scale * w + zero_point
    q = np.clip(np.round(scaled), qmin, qmax).astype(int)
    dequant = (q - zero_point) / scale
    return {"scale": scale, "zero_point": zero_point, "scaled": scaled, "q": q, "dequant": dequant,
            "error": np.abs(w - dequant), "mae": float(np.mean(np.abs(w - dequant)))}


# --- GGUF: bits per weight of the k-quant block formats ------------------------------------

def k_quant_bpw(blocks: int, weights_per_block: int, weight_bits: int,
                block_meta_bits: int, super_block_meta_bits: int) -> float:
    """Bits per weight of one super-block: the weights plus the scales that describe them."""
    weights = blocks * weights_per_block
    total_bits = weights * weight_bits + blocks * block_meta_bits + super_block_meta_bits
    return total_bits / weights


# Q4_K: 8 blocks of 32 weights at 4 bits; a 6-bit scale and a 6-bit minimum for each block;
# two FP16 numbers for the super-block.
Q4_K_BPW = k_quant_bpw(blocks=8, weights_per_block=32, weight_bits=4,
                       block_meta_bits=6 + 6, super_block_meta_bits=2 * 16)
# Q6_K: 16 blocks of 16 weights at 6 bits; an 8-bit scale for each block; one FP16 number.
Q6_K_BPW = k_quant_bpw(blocks=16, weights_per_block=16, weight_bits=6,
                       block_meta_bits=8, super_block_meta_bits=16)

# Measured by llama.cpp for Llama 3.1 8B (see Sources on the page).
Q4_K_M_BPW = 4.8944
Q4_K_M_GIB = 4.58


def gib_to_gb(gib: float) -> float:
    return gib * 2 ** 30 / 1e9


def offload_gb(file_gb: float, gpu_layers: int, total_layers: int) -> float:
    """Weights on the GPU if each layer has the same size (an assumption)."""
    return file_gb * gpu_layers / total_layers


# --- GPTQ and AWQ: the error of the layer OUTPUT -------------------------------------------

CALIB_X = np.array([2.0, 1.0])  # one calibration input (an assumption for the example)
LOW_BITS = 4


def grid_step(w: np.ndarray, bits: int) -> float:
    """Distance between two quantized values for absmax quantization."""
    return float(np.max(np.abs(w)) / (2 ** (bits - 1) - 1))


def to_grid(w: np.ndarray, step: float, bits: int) -> np.ndarray:
    """Round to the nearest value of the grid, as real numbers."""
    qmax = 2 ** (bits - 1) - 1
    return np.clip(np.round(w / step), -qmax, qmax) * step


def output_error(w: np.ndarray, w_hat: np.ndarray, x: np.ndarray, x_hat: np.ndarray | None = None) -> float:
    """||W x - W_hat x_hat||: the error that the next layer sees."""
    return float(np.linalg.norm(w @ x - w_hat @ (x if x_hat is None else x_hat)))


def round_to_nearest(w: np.ndarray, bits: int = LOW_BITS) -> np.ndarray:
    return to_grid(w, grid_step(w, bits), bits)


def gptq_toy(w: np.ndarray, calib: np.ndarray, bits: int = LOW_BITS) -> dict:
    """GPTQ idea on a layer with 2 input columns.

    Round column 1. Then move column 2 so that the outputs for the calibration inputs change as
    little as possible (least squares), and round column 2. `calib` has one row for each input.
    For two weights this is the GPTQ update -e / [H^-1]_11 * [H^-1]_21 with H = calib^T calib.
    """
    calib = np.atleast_2d(calib)
    step = grid_step(w, bits)
    col1 = to_grid(w[:, 0], step, bits)
    error1 = w[:, 0] - col1
    ratio = float(calib[:, 0] @ calib[:, 1] / (calib[:, 1] @ calib[:, 1]))
    col2_moved = w[:, 1] + error1 * ratio
    col2 = to_grid(col2_moved, step, bits)
    return {"step": step, "error1": error1, "ratio": ratio, "col2_moved": col2_moved,
            "w_hat": np.stack([col1, col2], axis=1)}


def awq_toy(w: np.ndarray, x: np.ndarray, alpha: float, bits: int = LOW_BITS) -> dict:
    """AWQ idea: multiply each weight column by s = |x|^alpha, quantize, divide the input by s."""
    s = np.abs(x) ** alpha
    w_scaled = w * s
    w_hat = round_to_nearest(w_scaled, bits)
    return {"s": s, "w_scaled": w_scaled, "step": grid_step(w_scaled, bits), "w_hat": w_hat,
            "x_scaled": x / s, "error": output_error(w, w_hat, x, x / s)}


# --- SmoothQuant: move the scale from the activations to the weights -----------------------

ACT_MAX = np.array([16.0, 1.0])  # assumed max |activation| of each input channel; channel 1 is an outlier
SMOOTH_ALPHA = 0.5


def smoothquant(act_max: np.ndarray, w: np.ndarray, alpha: float = SMOOTH_ALPHA) -> dict:
    """s_j = max|X_j|^alpha / max|W_j|^(1 - alpha). Then X_j / s_j and W_j * s_j."""
    w_max = np.abs(w).max(axis=0)  # column j of W reads input channel j
    s = act_max ** alpha / w_max ** (1 - alpha)
    return {"w_max": w_max, "s": s, "act_max": act_max / s, "w": w * s,
            "w_max_after": np.abs(w * s).max(axis=0)}


def levels(channel_max: np.ndarray, bits: int = 8) -> np.ndarray:
    """Positive integer steps that each channel uses when the full tensor shares one absmax scale."""
    qmax = 2 ** (bits - 1) - 1
    return channel_max / channel_max.max() * qmax


# --- Plot ----------------------------------------------------------------------------------

BIT_RANGE = list(range(2, 9))


def make_figures() -> None:
    absmax = [absmax_quantize(W, b)["mae"] for b in BIT_RANGE]
    zeropoint = [zeropoint_quantize(W, b)["mae"] for b in BIT_RANGE]

    def draw(ax, theme):
        ax.plot(BIT_RANGE, absmax, marker="o", color=theme["series"][0], label="Absmax (symmetric)")
        ax.plot(BIT_RANGE, zeropoint, marker="s", color=theme["series"][1], label="Zero-point (asymmetric)")
        for bits, value in ((4, absmax[2]), (8, absmax[6])):
            ax.annotate(f"{value:.4f}", (bits, value), textcoords="offset points", xytext=(6, 8),
                        fontsize=8, color=theme["ink2"])
        ax.set_yscale("log")
        ax.set_xticks(BIT_RANGE)
        ax.set_xlabel("Bits for each weight")
        ax.set_ylabel("Mean absolute error of W (log scale)")
        ax.legend(loc="upper right")

    save_figure(draw, PAGE, "error-by-bits")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    for name, bits in BITS_PER_WEIGHT.items():
        print(f"{name}: {bits / 8} bytes per weight, {model_size_gb(PARAMS, bits)} GB")
    for bits in (8, 4):
        for label, fn in (("absmax", absmax_quantize), ("zero-point", zeropoint_quantize)):
            r = fn(W, bits)
            print(f"--- {label}, {bits} bits: scale {r['scale']:.4f}, zero point {r['zero_point']}")
            print("scaled:", r["scaled"].ravel(), "q:", r["q"].ravel())
            print("dequant:", r["dequant"].ravel(), "error:", r["error"].ravel(), f"mae {r['mae']:.5f}")
    print("mae by bits (absmax):", [round(absmax_quantize(W, b)["mae"], 4) for b in BIT_RANGE])
    print("mae by bits (zero-point):", [round(zeropoint_quantize(W, b)["mae"], 4) for b in BIT_RANGE])

    print("Q4_K bpw:", Q4_K_BPW, "Q6_K bpw:", Q6_K_BPW)
    print("8B at Q4_K:", model_size_gb(PARAMS, Q4_K_BPW), "GB; at Q4_K_M:", model_size_gb(PARAMS, Q4_K_M_BPW), "GB")
    print("Llama 3.1 8B Q4_K_M file:", gib_to_gb(Q4_K_M_GIB), "GB")
    print("16 of 32 layers on GPU:", offload_gb(model_size_gb(PARAMS, Q4_K_M_BPW), 16, 32), "GB")

    step = grid_step(W, LOW_BITS)
    rtn = round_to_nearest(W)
    print("--- output error, 4 bits, x =", CALIB_X, "step", step, "y =", W @ CALIB_X)
    print("RTN:", rtn.ravel(), "y_hat", rtn @ CALIB_X, "error", output_error(W, rtn, CALIB_X))
    g = gptq_toy(W, CALIB_X)
    print("GPTQ: error1", g["error1"], "ratio", g["ratio"], "col2 moved", g["col2_moved"])
    print("GPTQ:", g["w_hat"].ravel(), "y_hat", g["w_hat"] @ CALIB_X, "error", output_error(W, g["w_hat"], CALIB_X))
    for alpha in (0.0, 0.5, 1.0):
        a = awq_toy(W, CALIB_X, alpha)
        print(f"AWQ alpha={alpha}: s {a['s']}, W*s {a['w_scaled'].ravel()}, step {a['step']:.4f},",
              f"w_hat {a['w_hat'].ravel()}, x/s {a['x_scaled']}, y_hat {a['w_hat'] @ a['x_scaled']}, error {a['error']:.4f}")

    sq = smoothquant(ACT_MAX, W)
    print("--- SmoothQuant: w_max", sq["w_max"], "s", sq["s"], "act max after", sq["act_max"])
    print("W*s:", sq["w"].ravel(), "w max after", sq["w_max_after"])
    print("levels before: activations", levels(ACT_MAX), "weights", levels(sq["w_max"]))
    print("levels after: activations", levels(sq["act_max"]), "weights", levels(sq["w_max_after"]))
    x = ACT_MAX
    print("same output:", W @ x, sq["w"] @ (x / sq["s"]))
    make_figures()
