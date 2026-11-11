"""Numbers and plots for content/scientist/08-new-trends.md.

Run: uv run python -m examples.scientist.new_trends
"""

from math import comb

import numpy as np

from examples.plotstyle import save_figure
from examples.toy import E, W, softmax

PAGE = "new-trends"


# --- Model merging -------------------------------------------------------------------------

# A second version of the toy matrix W (an assumption for the example): the same 4 values in
# other places, so the two matrices have the same length.
W_B = np.array([[1.2, 0.3], [0.5, 0.8]])


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation: a point on the straight line from a to b."""
    return (1 - t) * a + t * b


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> dict:
    """Spherical linear interpolation: a point on the arc from a to b."""
    cos = float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b)))
    theta = float(np.arccos(cos))
    coef_a = np.sin((1 - t) * theta) / np.sin(theta)
    coef_b = np.sin(t * theta) / np.sin(theta)
    return {"cos": cos, "theta": theta, "coef_a": float(coef_a), "coef_b": float(coef_b),
            "merged": coef_a * a + coef_b * b}


# Task vectors: fine-tuned weights minus base weights, for 2 fine-tuned models (assumed values).
TAU_A = np.array([0.4, -0.3, 0.05, 0.2])
TAU_B = np.array([0.2, 0.5, -0.6, 0.02])


def trim(tau: np.ndarray, keep: int) -> np.ndarray:
    """Keep the `keep` entries with the largest absolute value. Set the others to 0."""
    out = np.zeros_like(tau)
    top = np.argsort(np.abs(tau))[::-1][:keep]
    out[top] = tau[top]
    return out


def ties(task_vectors: list[np.ndarray], keep: int) -> dict:
    """TIES: trim each task vector, elect one sign for each entry, average the values with that sign."""
    trimmed = np.stack([trim(tau, keep) for tau in task_vectors])
    elected = np.sign(trimmed.sum(axis=0))
    agree = (np.sign(trimmed) == elected) & (trimmed != 0)
    counts = np.maximum(agree.sum(axis=0), 1)
    merged = (trimmed * agree).sum(axis=0) / counts
    return {"trimmed": trimmed, "elected": elected, "merged": merged}


DARE_DROP = 0.5
DARE_MASK = np.array([1, 0, 1, 0])  # assumed result of the random draw: keep entries 1 and 3


def dare(tau: np.ndarray, mask: np.ndarray, drop: float) -> np.ndarray:
    """DARE: drop random entries, then divide the others by (1 - drop)."""
    return tau * mask / (1 - drop)


# --- Multimodal: CLIP-style contrastive loss -----------------------------------------------

IMAGES = np.array([[0.8, 0.6], [0.6, -0.8]])  # photo of a cat, photo of a dog (unit vectors)
TEXTS = np.array([[1.0, 0.0], [0.6, -0.8]])   # "a cat", "a dog" (unit vectors)
CLIP_TEMPERATURE = 0.5                        # assumed; CLIP learns this value


def clip_loss(images: np.ndarray, texts: np.ndarray, temperature: float) -> dict:
    """Similarity of each image with each text, softmax in the two directions, mean cross-entropy."""
    similarity = images @ texts.T
    logits = similarity / temperature
    image_to_text = softmax(logits)        # each row sums to 1
    text_to_image = softmax(logits.T).T    # each column sums to 1
    loss_i2t = float(-np.log(np.diag(image_to_text)).mean())
    loss_t2i = float(-np.log(np.diag(text_to_image)).mean())
    return {"similarity": similarity, "logits": logits, "image_to_text": image_to_text,
            "text_to_image": text_to_image, "loss_i2t": loss_i2t, "loss_t2i": loss_t2i,
            "loss": (loss_i2t + loss_t2i) / 2}


def project_patch(patch: np.ndarray, projector: np.ndarray) -> np.ndarray:
    """LLaVA projector (one linear layer here): image vector -> vector in the token space of the LLM."""
    return projector @ patch


def patch_count(image_side: int, patch_side: int) -> int:
    return (image_side // patch_side) ** 2


# --- Interpretability: sparse autoencoder and abliteration ---------------------------------

ACTIVATION = np.array([0.2, 1.0])  # one activation vector of the model (assumed)
SAE_DIRECTIONS = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])  # 3 features for 2 dimensions (assumed)
SAE_BIAS = np.array([-0.5, -0.5, -0.5])
SAE_L1 = 0.1


def sae(x: np.ndarray, directions: np.ndarray, bias: np.ndarray, l1: float) -> dict:
    """Sparse autoencoder with tied weights: encode, ReLU, decode, loss."""
    pre = directions @ x + bias
    features = np.maximum(pre, 0)
    recon = directions.T @ features
    mse = float(np.sum((x - recon) ** 2))
    sparsity = float(np.sum(np.abs(features)))
    return {"pre": pre, "features": features, "recon": recon, "mse": mse, "l1": sparsity,
            "loss": mse + l1 * sparsity}


MEAN_HARMFUL = np.array([1.0, 1.2])   # mean activation on prompts that the model refuses (assumed)
MEAN_HARMLESS = np.array([0.4, 0.4])  # mean activation on prompts that the model answers (assumed)


def unit_direction(mean_a: np.ndarray, mean_b: np.ndarray) -> np.ndarray:
    diff = mean_a - mean_b
    return diff / np.linalg.norm(diff)


def ablate(x: np.ndarray, direction: np.ndarray) -> dict:
    """Remove the part of x that points along the unit vector `direction`."""
    amount = float(x @ direction)
    return {"amount": amount, "projection": amount * direction, "result": x - amount * direction}


def orthogonalize(w: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Change W so that its output never has a part along `direction`: W - r r^T W."""
    return w - np.outer(direction, direction) @ w


# --- Test-time compute: majority vote ------------------------------------------------------

VOTE_P = 0.6  # assumed probability that one sample is correct


def majority_vote(p: float, n: int) -> float:
    """Probability that more than half of n independent samples are correct (n is odd)."""
    return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))


def make_figures() -> None:
    ns = list(range(1, 42, 2))

    def draw(ax, theme):
        for i, p in enumerate((0.8, 0.6, 0.4)):
            ax.plot(ns, [majority_vote(p, n) for n in ns], marker="o", markersize=3,
                    color=theme["series"][i], label=f"p = {p}")
        for n, offset, align in ((1, (-2, -14), "center"), (5, (5, -14), "left"), (41, (0, -14), "right")):
            ax.annotate(f"{majority_vote(VOTE_P, n):.3f}", (n, majority_vote(VOTE_P, n)),
                        textcoords="offset points", xytext=offset, ha=align, fontsize=8, color=theme["ink2"])
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 5, 11, 21, 31, 41])
        ax.set_xlabel("Number of samples N (majority vote)")
        ax.set_ylabel("Probability that the vote is correct")
        ax.legend(loc="center right")

    save_figure(draw, PAGE, "majority-vote")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("norms:", np.linalg.norm(W), np.linalg.norm(W_B), "dot:", W.ravel() @ W_B.ravel())
    half = lerp(W, W_B, 0.5)
    print("lerp 0.5:", half.ravel(), "norm", np.linalg.norm(half))
    s = slerp(W, W_B, 0.5)
    print("slerp: cos", s["cos"], "theta rad", s["theta"], "deg", np.degrees(s["theta"]),
          "sin theta", np.sin(s["theta"]), "sin half", np.sin(s["theta"] / 2), "coef", s["coef_a"], s["coef_b"])
    print("slerp 0.5:", s["merged"].ravel(), "norm", np.linalg.norm(s["merged"]))

    t = ties([TAU_A, TAU_B], keep=2)
    print("mean of task vectors:", (TAU_A + TAU_B) / 2)
    print("ties trimmed:", t["trimmed"], "elected:", t["elected"], "merged:", t["merged"])
    print("W + ties:", W.ravel() + t["merged"])
    kept = dare(TAU_A, DARE_MASK, DARE_DROP)
    print("dare:", kept, "expected value of entry 1:", (1 - DARE_DROP) * kept[0])

    c = clip_loss(IMAGES, TEXTS, CLIP_TEMPERATURE)
    for key, value in c.items():
        print(key, value, sep=": " if np.isscalar(value) else "\n")
    print("exp of row 2 logits:", np.exp(c["logits"][1]))
    print("projected patch:", project_patch(IMAGES[0], W), "patches:", patch_count(336, 14))
    print("sequence:", np.vstack([project_patch(IMAGES[0], W), E[0], E[1]]))

    a = sae(ACTIVATION, SAE_DIRECTIONS, SAE_BIAS, SAE_L1)
    print("sae:", a)
    r = unit_direction(MEAN_HARMFUL, MEAN_HARMLESS)
    ab = ablate(ACTIVATION, r)
    print("direction:", r, "ablate:", ab, "check:", ab["result"] @ r)
    w_orth = orthogonalize(W, r)
    print("r^T W:", r @ W, "W orthogonalized:", w_orth.ravel(), "check:", r @ w_orth)

    for n in (1, 3, 5, 11, 41):
        print(f"vote p={VOTE_P} N={n}:", majority_vote(VOTE_P, n), "p=0.4:", majority_vote(0.4, n),
              "p=0.8:", majority_vote(0.8, n))
    make_figures()
