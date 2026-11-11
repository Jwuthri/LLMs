"""Numbers and plots for docs/scientist/04-fine-tuning.md.

Run: uv run python -m examples.scientist.fine_tuning
"""

import numpy as np

from examples.plotstyle import save_light_dark
from examples.toy import W

PAGE = "fine-tuning"

# One training example for the toy layer y = W x (assumed values).
X = np.array([1.0, 2.0])
TARGET = np.array([-1.5, 1.0])
LEARNING_RATE = 0.1

# LoRA adapter at the start of training: rank 1, alpha 2. A is an assumed "random" draw.
LORA_ALPHA = 2.0
A_INIT = np.array([[0.5, 0.5]])    # 1 x 2
B_INIT = np.array([[0.0], [0.0]])  # 2 x 1


# --- Training techniques: full fine-tuning, LoRA, QLoRA ------------------------------------

def loss(w: np.ndarray, x: np.ndarray = X, target: np.ndarray = TARGET) -> float:
    """Half of the squared error of the layer output."""
    return float(0.5 * np.sum((w @ x - target) ** 2))


def grad_w(w: np.ndarray, x: np.ndarray = X, target: np.ndarray = TARGET) -> np.ndarray:
    """Gradient of the loss for each entry of the weight matrix: (W x - target) x^T."""
    return np.outer(w @ x - target, x)


def full_step(w: np.ndarray, lr: float = LEARNING_RATE) -> np.ndarray:
    """One gradient step of full fine-tuning: each entry of W changes."""
    return w - lr * grad_w(w)


def lora_delta(b: np.ndarray, a: np.ndarray, alpha: float = LORA_ALPHA) -> np.ndarray:
    """The LoRA update (alpha / r) * B A. The rank r is the number of rows of A."""
    return alpha / a.shape[0] * (b @ a)


def lora_grads(w: np.ndarray, b: np.ndarray, a: np.ndarray,
               alpha: float = LORA_ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of the loss for B and for A. W gets no gradient: it is frozen."""
    scale = alpha / a.shape[0]
    g = grad_w(w + lora_delta(b, a, alpha))
    return scale * g @ a.T, scale * b.T @ g


def lora_step(w: np.ndarray, b: np.ndarray, a: np.ndarray, alpha: float = LORA_ALPHA,
              lr: float = LEARNING_RATE) -> tuple[np.ndarray, np.ndarray]:
    """One gradient step on the adapter only."""
    grad_b, grad_a = lora_grads(w, b, a, alpha)
    return b - lr * grad_b, a - lr * grad_a


def merge(w: np.ndarray, b: np.ndarray, a: np.ndarray, alpha: float = LORA_ALPHA) -> np.ndarray:
    """Add the adapter into the base matrix. The result has the shape of W."""
    return w + lora_delta(b, a, alpha)


def full_params(d: int) -> int:
    return d * d


def lora_params(d: int, r: int) -> int:
    """B has d x r numbers and A has r x d numbers."""
    return 2 * d * r


def megabytes(params: int, bits: int) -> float:
    return params * bits / 8 / 1e6


# --- Training parameters -------------------------------------------------------------------

def lr_schedule(step: int, peak: float, warmup: int, total: int) -> float:
    """Linear warm-up from 0 to `peak`, then cosine decay to 0 at step `total`."""
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return peak * 0.5 * (1 + np.cos(np.pi * progress))


# Assumed gradient of the weight w11 for each of 8 training examples.
EXAMPLE_GRADS = np.array([-0.5, -0.3, -0.1, -0.3, -0.8, -0.4, 0.1, -0.1])


def accumulate(example_grads: np.ndarray, micro_batch: int) -> tuple[np.ndarray, float]:
    """Mean gradient of each micro-batch, and the mean of those means (the accumulated gradient)."""
    micro_means = example_grads.reshape(-1, micro_batch).mean(axis=1)
    return micro_means, float(micro_means.mean())


def steps_per_epoch(examples: int, micro_batch: int, accumulation: int) -> int:
    return examples // (micro_batch * accumulation)


def weight_decay_step(w: float, lr: float, decay: float) -> float:
    """Decoupled weight decay (AdamW): shrink the weight by the factor (1 - lr * decay)."""
    return w * (1 - lr * decay)


# --- Distributed training: ZeRO stages -----------------------------------------------------

# Bytes for each parameter in mixed-precision Adam training (ZeRO paper, section 3.1).
BYTES_PARAM, BYTES_GRAD, BYTES_OPTIM = 2, 2, 12


def zero_bytes_per_param(stage: int, gpus: int) -> float:
    """Model-state memory on one GPU, in bytes for each model parameter.

    Stage 0 is plain data parallelism. Stage 1 splits the optimizer states, stage 2 also
    splits the gradients, stage 3 (and FSDP) also splits the parameters.
    """
    optim = BYTES_OPTIM / gpus if stage >= 1 else BYTES_OPTIM
    grad = BYTES_GRAD / gpus if stage >= 2 else BYTES_GRAD
    param = BYTES_PARAM / gpus if stage >= 3 else BYTES_PARAM
    return param + grad + optim


def zero_toy_counts(stage: int, gpus: int = 2, weights: int = 4, adam_states: int = 2) -> dict[str, int]:
    """Numbers that one GPU holds for the toy matrix (4 weights, Adam keeps 2 numbers per weight)."""
    counts = {
        "weights": weights // gpus if stage >= 3 else weights,
        "gradients": weights // gpus if stage >= 2 else weights,
        "optimizer": weights * adam_states // gpus if stage >= 1 else weights * adam_states,
    }
    counts["total"] = sum(counts.values())
    return counts


# --- Monitoring ----------------------------------------------------------------------------

def grad_norm(g: np.ndarray) -> float:
    """L2 norm of all gradient entries together."""
    return float(np.sqrt(np.sum(g ** 2)))


def clip(g: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale the gradient down when its norm is larger than `max_norm`."""
    norm = grad_norm(g)
    return g * min(1.0, max_norm / norm)


def loss_curves(steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic training and validation loss (an illustration, not a real run)."""
    train = 0.4 + 1.6 * np.exp(-steps / 60)
    val = 0.7 + 1.4 * np.exp(-steps / 60) + 0.00002 * np.maximum(0, steps - 100) ** 2
    return train, val


CURVE_STEPS = np.arange(0, 301, 10)


def best_validation_step() -> tuple[int, float]:
    _, val = loss_curves(CURVE_STEPS)
    best = int(np.argmin(val))
    return int(CURVE_STEPS[best]), float(val[best])


def make_figures() -> None:
    train, val = loss_curves(CURVE_STEPS)
    best_step, best_val = best_validation_step()

    def draw(ax, theme):
        ax.plot(CURVE_STEPS, train, color=theme["series"][0], label="Training loss")
        ax.plot(CURVE_STEPS, val, color=theme["series"][1], label="Validation loss")
        ax.axvline(best_step, color=theme["muted"], linewidth=1, linestyle="--")
        ax.annotate(f"lowest validation loss\nstep {best_step}: {best_val:.2f}",
                    xy=(best_step, best_val), xytext=(best_step + 12, best_val + 0.55),
                    fontsize=8, color=theme["ink2"],
                    arrowprops={"arrowstyle": "-", "color": theme["muted"], "linewidth": 0.8})
        ax.text(298, 1.92, "Illustration (synthetic curves)", ha="right", fontsize=8, color=theme["muted"])
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 2.2)
        ax.legend(loc="upper center")

    save_light_dark(draw, PAGE, "loss-curves")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("output W x:", W @ X, "loss:", loss(W))
    print("gradient for W:\n", grad_w(W))
    w_full = full_step(W)
    print("full fine-tuning, W after 1 step:\n", w_full, "\noutput:", w_full @ X, "loss:", loss(w_full))

    print("LoRA start: delta W =\n", lora_delta(B_INIT, A_INIT))
    grad_b, grad_a = lora_grads(W, B_INIT, A_INIT)
    print("grad B:", grad_b.ravel(), "grad A:", grad_a.ravel())
    zero_b, zero_a = lora_grads(W, B_INIT, np.zeros_like(A_INIT))
    print("if A is also zero: grad B:", zero_b.ravel(), "grad A:", zero_a.ravel())
    b1, a1 = lora_step(W, B_INIT, A_INIT)
    print("B after 1 step:", b1.ravel(), "A after 1 step:", a1.ravel())
    print("delta W =\n", lora_delta(b1, a1))
    w_merged = merge(W, b1, a1)
    print("merged W' =\n", w_merged, "\noutput:", w_merged @ X, "loss:", loss(w_merged))
    print("adapter path output:", W @ X + LORA_ALPHA * (b1 @ (a1 @ X)))

    for d, r in ((2, 1), (4096, 1), (4096, 16)):
        full, lora = full_params(d), lora_params(d, r)
        print(f"d={d} r={r}: full {full:,}  LoRA {lora:,}  ratio {100 * lora / full:.3f}%")
    print("4096x4096 at 16 bits:", megabytes(full_params(4096), 16), "MB; at 4 bits:",
          megabytes(full_params(4096), 4), "MB; adapter r=16 at 16 bits:",
          megabytes(lora_params(4096, 16), 16), "MB")
    print("8e9 parameters: 16 bits =", 8e9 * 16 / 8 / 1e9, "GB; 4 bits =", 8e9 * 4 / 8 / 1e9, "GB")

    for step in (0, 5, 10, 55, 100):
        print(f"lr at step {step}: {lr_schedule(step, 3e-4, 10, 100):.6f}")
    micro, accumulated = accumulate(EXAMPLE_GRADS, 2)
    print("micro-batch gradients:", micro, "accumulated:", accumulated,
          "mean of all 8:", EXAMPLE_GRADS.mean())
    print("w11 after the step:", W[0, 0] - LEARNING_RATE * accumulated)
    print("steps per epoch (1,000 examples):", steps_per_epoch(1000, 2, 4))
    print("weight decay on w12:", weight_decay_step(W[0, 1], LEARNING_RATE, 0.01))

    for stage in range(4):
        print(f"ZeRO stage {stage}: toy {zero_toy_counts(stage)}  bytes/param with 2 GPUs "
              f"{zero_bytes_per_param(stage, 2)}  with 8 GPUs {zero_bytes_per_param(stage, 8)}")
    print("ZeRO paper check (7.5e9 params, 64 GPUs, GB):",
          [round(7.5 * zero_bytes_per_param(stage, 64), 1) for stage in range(4)])

    g = grad_w(W)
    print("gradient norm:", grad_norm(g), "clipped to 1.0:\n", clip(g, 1.0))
    print("best validation step:", best_validation_step())
    train, val = loss_curves(np.array([0, 100, 300]))
    print("train at 0/100/300:", train, "val at 0/100/300:", val)
    make_figures()
