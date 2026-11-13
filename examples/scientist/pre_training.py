"""Numbers and plots for content/scientist/02-pre-training.md.

Run: uv run python -m examples.scientist.pre_training
"""

import hashlib
import math
import re

import numpy as np

from examples.plotstyle import save_figure
from examples.toy import LOGITS, VOCAB, W, E, softmax

PAGE = "pre-training"

# --- The training objective: next-token loss and perplexity --------------------------------


def one_hot(token: str) -> np.ndarray:
    return np.eye(len(VOCAB))[VOCAB.index(token)]


def loss(logits: np.ndarray, target: str) -> float:
    """Cross-entropy for one position: -ln(probability of the correct next token)."""
    return float(-np.log(softmax(logits)[VOCAB.index(target)]))


def gradient(logits: np.ndarray, target: str) -> np.ndarray:
    """Gradient of the loss with respect to the logits: probabilities - one-hot target."""
    return softmax(logits) - one_hot(target)


# --- Data preparation ----------------------------------------------------------------------

RAW_CORPUS = ["the cat sat", "The cat sat.", "the cat sat", "cat cat cat cat", "the cat the cat sat"]
MIN_UNIQUE_RATIO = 0.5  # assumed limit of the quality filter


def normalize(text: str) -> str:
    """Lowercase, no punctuation, single spaces."""
    return " ".join(re.findall(r"[a-z]+", text.lower()))


def fingerprint(text: str) -> str:
    """First 8 hexadecimal digits of the SHA-1 hash of the normalized text."""
    return hashlib.sha1(normalize(text).encode()).hexdigest()[:8]


def deduplicate(docs: list[str]) -> list[str]:
    """Keep the first document for each fingerprint."""
    seen: set[str] = set()
    kept = []
    for doc in docs:
        if fingerprint(doc) not in seen:
            seen.add(fingerprint(doc))
            kept.append(normalize(doc))
    return kept


def unique_ratio(doc: str) -> float:
    words = doc.split()
    return len(set(words)) / len(words)


def quality_filter(docs: list[str]) -> list[str]:
    return [doc for doc in docs if unique_ratio(doc) >= MIN_UNIQUE_RATIO]


def training_examples(doc: str) -> list[tuple[str, str]]:
    """Each position gives one example: (context, next token)."""
    words = doc.split()
    return [(" ".join(words[:i]), words[i]) for i in range(1, len(words))]


# --- Distributed training ------------------------------------------------------------------

BATCH_TARGETS = ["sat", "sat", "cat", "sat"]  # 4 samples with the context "the cat"


def batch_gradient(logits: np.ndarray, targets: list[str]) -> np.ndarray:
    return np.mean([gradient(logits, target) for target in targets], axis=0)


def batch_loss(logits: np.ndarray, targets: list[str]) -> float:
    return float(np.mean([loss(logits, target) for target in targets]))


def data_parallel(logits: np.ndarray, targets: list[str], workers: int = 2) -> dict:
    """Each worker gets an equal part of the batch. The result is the mean of the worker gradients."""
    size = len(targets) // workers
    per_worker = [batch_gradient(logits, targets[i * size:(i + 1) * size]) for i in range(workers)]
    return {"workers": per_worker, "mean": np.mean(per_worker, axis=0)}


def tensor_parallel(x: np.ndarray) -> dict:
    """Split the rows of W between 2 workers. Each worker calculates one part of W x."""
    return {"worker_a": W[:1] @ x, "worker_b": W[1:] @ x, "full": W @ x}


def training_memory_gb(parameters: float, bytes_each: int = 4, copies: int = 4) -> float:
    """Weights + gradients + the two AdamW averages = 4 copies. Activations are not counted."""
    return parameters * bytes_each * copies / 1e9


# --- Training optimization -----------------------------------------------------------------

MAX_NORM = 0.25  # assumed clipping limit, small enough to clip the toy gradient
ADAMW = {"lr": 0.1, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01}


def clip_by_norm(g: np.ndarray, max_norm: float) -> dict:
    norm = float(np.linalg.norm(g))
    scale = min(1.0, max_norm / norm)
    return {"norm": norm, "scale": scale, "clipped": g * scale}


def adamw_step(theta, g, m, v, t, lr, beta1, beta2, eps, weight_decay) -> dict:
    """One AdamW update. Returns each intermediate value."""
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    direction = m_hat / (np.sqrt(v_hat) + eps)
    decay = weight_decay * theta
    return {"m": m, "v": v, "m_hat": m_hat, "v_hat": v_hat, "direction": direction, "decay": decay,
            "theta": theta - lr * (direction + decay)}


def lr_schedule(step: float, peak: float, warmup: int, total: int, floor: float = 0.0) -> float:
    """Linear warm-up from 0 to `peak`, then a cosine curve from `peak` down to `floor`."""
    if step < warmup:
        return peak * step / warmup
    if total == warmup:
        return peak
    progress = (step - warmup) / (total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


SCHEDULE = {"peak": 3e-4, "warmup": 100, "total": 1000}  # assumed values, also the widget defaults


def float16_facts() -> dict:
    return {
        "bytes": {"fp32": np.dtype(np.float32).itemsize, "fp16": np.dtype(np.float16).itemsize},
        "eps16": float(np.finfo(np.float16).eps),
        "max16": float(np.finfo(np.float16).max),
        "lost_update": bool(np.float16(1.0) + np.float16(0.0001) == np.float16(1.0)),
        "kept_update_fp32": bool(np.float32(1.0) + np.float32(0.0001) != np.float32(1.0)),
        "toy_table_bytes": {"fp32": E.astype(np.float32).nbytes, "fp16": E.astype(np.float16).nbytes},
    }


# --- Monitoring: a full toy training run ---------------------------------------------------

RUN = {"steps": 100, "warmup": 10, "peak": 0.1, "max_norm": 1.0}
TOKENS_PER_STEP = len(BATCH_TARGETS) * 3  # 4 samples of 3 tokens


def entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def training_run() -> list[dict]:
    """Train the 3 logits on the batch of 4 samples, from zeros. One log row for each step."""
    theta, m, v = np.zeros(3), np.zeros(3), np.zeros(3)
    log = []
    for step in range(RUN["steps"] + 1):
        g = batch_gradient(theta, BATCH_TARGETS)
        clipped = clip_by_norm(g, RUN["max_norm"])
        lr = lr_schedule(step, RUN["peak"], RUN["warmup"], RUN["steps"])
        log.append({"step": step, "loss": batch_loss(theta, BATCH_TARGETS), "grad_norm": clipped["norm"], "lr": lr})
        update = adamw_step(theta, clipped["clipped"], m, v, step + 1, **{**ADAMW, "lr": lr})
        theta, m, v = update["theta"], update["m"], update["v"]
    return log


LOSS_FLOOR = entropy(np.mean([one_hot(target) for target in BATCH_TARGETS], axis=0))


# --- Plot ----------------------------------------------------------------------------------

def make_figures() -> None:
    log = training_run()
    steps = [row["step"] for row in log]

    def draw(ax, theme):
        ax.plot(steps, [row["loss"] for row in log], color=theme["series"][0], label="Training loss")
        ax.axhline(LOSS_FLOOR, color=theme["series"][1], linestyle="--", label=f"Best possible loss ({LOSS_FLOOR:.3f})")
        ax.annotate(f"start: ln(3) = {log[0]['loss']:.3f}", (0, log[0]["loss"]), xytext=(8, log[0]["loss"] - 0.02),
                    color=theme["ink2"], fontsize=9, va="top")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.set_ylim(0.5, 1.15)
        ax.legend(loc="upper right")

    save_figure(draw, PAGE, "loss-curve")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    toy_loss = loss(LOGITS, "sat")
    print("loss:", toy_loss, "perplexity:", math.exp(toy_loss))
    print("uniform loss ln(3):", math.log(3), "ln(50257):", math.log(50_257), "ln(128256):", math.log(128_256))

    for doc in RAW_CORPUS:
        print(f"{doc!r}: normalized {normalize(doc)!r}, fingerprint {fingerprint(doc)}")
    unique = deduplicate(RAW_CORPUS)
    kept = quality_filter(unique)
    print("after deduplication:", unique)
    print("unique ratios:", {doc: unique_ratio(doc) for doc in unique})
    print("after quality filter:", kept)
    print("tokens before:", sum(len(normalize(d).split()) for d in RAW_CORPUS), "after:", sum(len(d.split()) for d in kept))
    for doc in kept:
        print("examples:", training_examples(doc))
    print("pairs to compare for 5 documents:", math.comb(5, 2), "for 1e9 documents:", math.comb(10 ** 9, 2))

    for target in ("sat", "cat"):
        print(f"gradient for target {target}:", gradient(LOGITS, target), "loss:", loss(LOGITS, target))
    parallel = data_parallel(LOGITS, BATCH_TARGETS)
    print("worker gradients:", parallel["workers"], "mean:", parallel["mean"])
    print("full batch gradient:", batch_gradient(LOGITS, BATCH_TARGETS), "batch loss:", batch_loss(LOGITS, BATCH_TARGETS))
    print("tensor parallel on sat:", tensor_parallel(E[2]))
    print("training memory for 8e9 parameters (GB):", training_memory_gb(8e9))

    g = gradient(LOGITS, "sat")
    clip = clip_by_norm(g, MAX_NORM)
    print("clip:", clip)
    step = adamw_step(LOGITS, clip["clipped"], np.zeros(3), np.zeros(3), 1, **ADAMW)
    for name, value in step.items():
        print("adamw", name, value)
    print("loss after the step:", loss(step["theta"], "sat"))
    for s in (0, 50, 100, 325, 550, 1000):
        print(f"lr at step {s}: {lr_schedule(s, **SCHEDULE):.3e}")
    print("float16:", float16_facts())
    print("weights of 8e9 parameters (GB): fp32", 8e9 * 4 / 1e9, "bf16", 8e9 * 2 / 1e9)

    log = training_run()
    for s in (0, 5, 10, 50, 100):
        print(log[s])
    print("loss floor:", LOSS_FLOOR, "tokens per step:", TOKENS_PER_STEP)
    print("days for 15e12 tokens at 1e6 tokens/s:", 15e12 / 1e6 / 86_400)
    make_figures()
