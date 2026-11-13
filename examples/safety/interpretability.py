"""Numbers and plots for content/safety/05-interpretability.md.

Run: uv run python -m examples.safety.interpretability
"""

import numpy as np

from examples.plotstyle import save_figure
from examples.toy import VOCAB, E, softmax

PAGE = "interpretability"


# --- The model: a hidden state h (2 numbers), logits = E.h, then softmax ---------------------

H = np.array([1.0, 1.0])  # assumed hidden state of parts 3 and 4 (equal to the embedding of `sat`)


def logits(h: np.ndarray) -> np.ndarray:
    """One dot product for each vocabulary word."""
    return E @ h


def probabilities(h: np.ndarray) -> np.ndarray:
    """Probability of each next token."""
    return softmax(logits(h))


# --- Part 1: a linear probe from a difference of means ---------------------------------------

# Assumed hidden states of 6 texts. The label is "the text is about an animal".
H_WITH = np.array([[0.2, 1.2], [0.4, 1.6], [0.0, 1.4]])
H_WITHOUT = np.array([[1.0, 0.6], [1.4, 0.2], [1.2, 0.4]])

# A new text about an animal. Its first number is high, as in the texts without the concept.
H_NEW = np.array([1.6, 1.4])


def difference_of_means(with_concept: np.ndarray, without_concept: np.ndarray) -> np.ndarray:
    """Probe direction: mean of the states with the concept minus mean of the states without."""
    return with_concept.mean(axis=0) - without_concept.mean(axis=0)


def midpoint_threshold(with_concept: np.ndarray, without_concept: np.ndarray, v: np.ndarray) -> float:
    """Half-way between the mean score of the two groups."""
    return float((with_concept @ v).mean() + (without_concept @ v).mean()) / 2


def probe_accuracy(states: np.ndarray, labels: np.ndarray, v: np.ndarray, threshold: float) -> float:
    """Share of states for which "score above the threshold" is equal to the label."""
    return float(((states @ v > threshold) == labels.astype(bool)).mean())


V = difference_of_means(H_WITH, H_WITHOUT)  # [-1, 1]; also the steering vector of part 3
THRESHOLD = midpoint_threshold(H_WITH, H_WITHOUT, V)


# --- Part 2: three features in two dimensions ------------------------------------------------

FEATURES = E / np.linalg.norm(E, axis=1, keepdims=True)  # one unit direction per feature
SAE_BIAS = -0.75  # assumed negative bias of the encoder


def interference(features: np.ndarray) -> np.ndarray:
    """Dot product of each pair of feature directions. 0 means that two features do not overlap."""
    return features @ features.T


def linear_read(x: np.ndarray) -> np.ndarray:
    """How much of each feature direction is in the activation x."""
    return FEATURES @ x


def encode(x: np.ndarray, bias: float = SAE_BIAS) -> np.ndarray:
    """The encoder of a sparse autoencoder: a linear read, a bias, then ReLU."""
    return np.maximum(linear_read(x) + bias, 0.0)


# --- Part 3: steering ------------------------------------------------------------------------

ALPHAS = [0, 1, 2, 4]


def steer(h: np.ndarray, v: np.ndarray, alpha: float) -> np.ndarray:
    """Activation addition: move the hidden state along a direction."""
    return h + alpha * v


# --- Part 4: ablation and patching -----------------------------------------------------------

V_CAT = np.array([0.0, 1.0])  # unit direction of `cat`


def ablate(h: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Directional ablation: remove the part of h along a unit direction."""
    return h - (h @ v) * v


def mean_ablate(h: np.ndarray, v: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Replace the part of h along a unit direction with its mean over the reference states."""
    return ablate(h, v) + (reference @ v).mean() * v


def patch(corrupted: np.ndarray, clean: np.ndarray, neuron: int) -> np.ndarray:
    """Activation patching: copy one number of the clean run into the corrupted run."""
    patched = corrupted.copy()
    patched[neuron] = clean[neuron]
    return patched


def logit_difference(h: np.ndarray, token: str = "cat", other: str = "the") -> float:
    """Logit of `token` minus logit of `other`."""
    z = logits(h)
    return float(z[VOCAB.index(token)] - z[VOCAB.index(other)])


def recovered(patched: np.ndarray, clean: np.ndarray, corrupted: np.ndarray) -> float:
    """Share of the clean-to-corrupted gap in the logit difference that the patch brings back."""
    low = logit_difference(corrupted)
    return (logit_difference(patched) - low) / (logit_difference(clean) - low)


H_CLEAN = H_WITH[0]  # run on a text with the concept
H_CORRUPTED = H_WITHOUT[0]  # run on a text without the concept


# --- Plot ------------------------------------------------------------------------------------

def make_figures() -> None:
    alphas = np.linspace(0, 6, 121)
    curves = np.array([probabilities(steer(H, V, a)) for a in alphas])

    def draw(ax, theme):
        for i, token in enumerate(VOCAB):
            color = theme["series"][i]
            ax.plot(alphas, curves[:, i], color=color, label=token)
            ax.plot(ALPHAS, [probabilities(steer(H, V, a))[i] for a in ALPHAS], "o", color=color, markersize=4)
        cat_at_4 = probabilities(steer(H, V, 4))[1]
        ax.annotate(f"{cat_at_4:.2f}", (4, cat_at_4), textcoords="offset points", xytext=(-4, -14),
                    fontsize=8, color=theme["ink2"])
        ax.set_xlim(-0.15, 6.15)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Steering strength α")
        ax.set_ylabel("Probability of the next token")
        ax.legend(loc="center right")

    save_figure(draw, PAGE, "steering")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("base h:", H, "logits:", logits(H), "probabilities:", probabilities(H))

    print("\n-- 1. probe")
    print("means:", H_WITH.mean(axis=0), H_WITHOUT.mean(axis=0), "direction v:", V)
    print("scores with:", H_WITH @ V, "without:", H_WITHOUT @ V, "threshold:", round(THRESHOLD, 3))
    states = np.vstack([H_WITH, H_WITHOUT])
    labels = np.array([1, 1, 1, 0, 0, 0])
    print("accuracy on the 6 states:", probe_accuracy(states, labels, V, THRESHOLD))
    print("new state:", H_NEW, "score:", round(float(H_NEW @ V), 3), "cat-only score:", H_NEW @ V_CAT)
    print("accuracy on the 7 states:",
          round(probe_accuracy(np.vstack([states, H_NEW]), np.append(labels, 1), V, THRESHOLD), 3))

    print("\n-- 2. superposition")
    print("unit directions:", FEATURES, sep="\n")
    print("interference:", interference(FEATURES), sep="\n")
    only_sat = FEATURES[2]
    print("only `sat` active: x =", only_sat, "linear read:", linear_read(only_sat), "encoder:", encode(only_sat))
    the_and_cat = FEATURES[0] + FEATURES[1]
    print("`the` and `cat` active: x =", the_and_cat, "linear read:", linear_read(the_and_cat),
          "encoder:", encode(the_and_cat))

    print("\n-- 3. steering")
    for alpha in ALPHAS + [8]:
        h = steer(H, V, alpha)
        print(f"alpha {alpha}: h' = {h}, logits {logits(h)}, probabilities {probabilities(h)},",
              f"probe score {round(float(h @ V), 3) + 0.0}")

    print("\n-- 4. causal tests")
    print("cat-direction scores of the embeddings:", dict(zip(VOCAB, E @ V_CAT)))
    print("h after ablation:", ablate(H, V_CAT), "logits:", logits(H), "->", logits(ablate(H, V_CAT)))
    print("h after mean ablation:", mean_ablate(H, V_CAT, E), "logits:", logits(mean_ablate(H, V_CAT, E)))
    print("logit difference clean / corrupted:", logit_difference(H_CLEAN), logit_difference(H_CORRUPTED))
    for neuron in (0, 1):
        patched = patch(H_CORRUPTED, H_CLEAN, neuron)
        print(f"patch neuron {neuron + 1}: h = {patched}, logit difference {logit_difference(patched):.3f},",
              f"recovered {recovered(patched, H_CLEAN, H_CORRUPTED):.3f}")
    make_figures()
