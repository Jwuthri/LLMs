"""Numbers and plots for docs/fundamentals/01-math.md.

Run: uv run python -m examples.fundamentals.mathematics
"""

from collections.abc import Callable

import numpy as np

from examples.plotstyle import save_light_dark
from examples.toy import LOGITS, VOCAB, E, W, softmax

PAGE = "math"

THE, CAT, SAT = E  # the three embedding vectors


# --- Linear algebra ------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two vectors, divided by the product of their lengths."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def determinant_2x2(m: np.ndarray) -> float:
    """ad - bc for the matrix [[a, b], [c, d]]."""
    return float(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])


# E^T E: each entry is a sum over the 3 tokens of (coordinate i) x (coordinate j).
GRAM = E.T @ E

# Sizes of Llama 3 8B (see Sources on the page).
LLAMA3_VOCAB = 128_256
LLAMA3_EMBEDDING = 4_096


# --- Calculus: a loss with one weight ------------------------------------------------------

TARGET_LOGIT = 2.0  # the output that the one-weight model must give
START_WEIGHT = 0.0
LEARNING_RATE = 0.1


def loss(w: float) -> float:
    """Squared error of a model whose output is its one weight w."""
    return (w - TARGET_LOGIT) ** 2


def loss_derivative(w: float) -> float:
    """dL/dw from the chain rule: 2 * (w - 2) * 1."""
    return 2 * (w - TARGET_LOGIT)


def finite_difference(f: Callable[[float], float], x: float, step: float) -> float:
    """Slope of f between x and x + step."""
    return (f(x + step) - f(x)) / step


def gradient_descent(w: float, learning_rate: float, steps: int) -> list[dict[str, float]]:
    """Run `steps` updates w <- w - learning_rate * dL/dw. Row k holds the state before update k."""
    rows = []
    for _ in range(steps + 1):
        rows.append({"w": w, "loss": loss(w), "gradient": loss_derivative(w)})
        w = w - learning_rate * loss_derivative(w)
    return rows


# --- Probability and statistics ------------------------------------------------------------

def cross_entropy(logits: np.ndarray, target: int) -> float:
    """-ln of the probability that softmax gives to the correct token."""
    return float(-np.log(softmax(logits)[target]))


def numeric_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """Change one input at a time by `step` and measure the slope of f."""
    slopes = np.zeros_like(x)
    for i in range(len(x)):
        moved = x.copy()
        moved[i] += step
        slopes[i] = (f(moved) - f(x)) / step
    return slopes


def expectation(probs: np.ndarray, values: np.ndarray) -> float:
    """Mean of a random variable: each value times its probability."""
    return float(probs @ values)


def variance(probs: np.ndarray, values: np.ndarray) -> float:
    """Mean squared distance of a random variable from its expectation."""
    return float(probs @ (values - expectation(probs, values)) ** 2)


IS_SAT = np.array([0.0, 0.0, 1.0])  # X = 1 if the drawn token is "sat", else 0
DRAWS = 100


# --- Plot ----------------------------------------------------------------------------------

PLOT_RATES = [0.1, 0.9, 1.1]
PLOT_STEPS = 8


def make_figures() -> None:
    def draw(ax, theme):
        ax.axhline(TARGET_LOGIT, color=theme["muted"], linewidth=1, linestyle="--", label="best weight, w = 2")
        for i, rate in enumerate(PLOT_RATES):
            ws = [row["w"] for row in gradient_descent(START_WEIGHT, rate, PLOT_STEPS)]
            ax.plot(range(PLOT_STEPS + 1), ws, marker="o", markersize=4,
                    color=theme["series"][i], label=f"learning rate {rate}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Weight w")
        ax.legend(loc="upper left")

    save_light_dark(draw, PAGE, "learning-rate")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    print("--- linear algebra")
    print("the + cat:", THE + CAT)
    print("sat.cat, sat.sat, the.cat:", SAT @ CAT, SAT @ SAT, THE @ CAT)
    print("length of sat:", np.linalg.norm(SAT))
    print("cosine(sat, the):", cosine(SAT, THE), " cosine(sat, sat):", cosine(SAT, SAT))
    print("W the, W cat, W sat:", W @ THE, W @ CAT, W @ SAT)
    print("det W:", determinant_2x2(W), np.linalg.det(W))
    print("W inverse:", np.linalg.inv(W), " W^-1 (W sat):", np.linalg.inv(W) @ (W @ SAT))
    print("E^T E:", GRAM, sep="\n")
    print("G [1,1], G [1,-1]:", GRAM @ np.array([1.0, 1.0]), GRAM @ np.array([1.0, -1.0]))
    print("eigenvalues of E^T E:", np.linalg.eigvalsh(GRAM))
    print("eigenvalues of W:", np.linalg.eigvals(W))
    print("Llama 3 8B output matrix:", LLAMA3_VOCAB * LLAMA3_EMBEDDING)

    print("--- calculus")
    print("L(0), L(0.001):", loss(0.0), loss(0.001))
    print("slope with step 0.001:", finite_difference(loss, 0.0, 0.001))
    print("derivative at 0:", loss_derivative(0.0))
    for rate in (LEARNING_RATE, 1.1):
        print(f"gradient descent, learning rate {rate}:")
        for k, row in enumerate(gradient_descent(START_WEIGHT, rate, 4)):
            print(f"  step {k}: w={row['w']:.4f} loss={row['loss']:.4f} gradient={row['gradient']:.4f}")
    moved = LOGITS + np.array([0.0, 0.0, 0.001])
    print("loss with sat logit + 0.001:", cross_entropy(moved, 2))
    print("slope for the sat logit, step 0.001:", (cross_entropy(moved, 2) - cross_entropy(LOGITS, 2)) / 0.001)
    print("numeric gradient of the loss:", numeric_gradient(lambda z: cross_entropy(z, 2), LOGITS))

    print("--- probability and statistics")
    probs = softmax(LOGITS)
    print("exp(logits):", np.exp(LOGITS), "sum:", np.exp(LOGITS).sum())
    print("probabilities:", probs, "sum:", probs.sum())
    mean = expectation(probs, IS_SAT)
    var = variance(probs, IS_SAT)
    print("squared distances:", np.round((IS_SAT - mean) ** 2, 4).tolist(), " p(the):", round(float(probs[0]), 4))
    print("E[X]:", mean, " Var[X]:", var, " std:", np.sqrt(var))
    print(f"{DRAWS} draws: mean count {DRAWS * mean:.3f}, std {np.sqrt(DRAWS * var):.3f}")
    print("95% interval half-width of the share:", 1.96 * np.sqrt(var / DRAWS))
    for token in (2, 0):
        print(f"cross-entropy, target {VOCAB[token]}:", cross_entropy(LOGITS, token))
    print("p - y:", probs - IS_SAT)
    print("perplexity:", np.exp(cross_entropy(LOGITS, 2)))
    make_figures()
