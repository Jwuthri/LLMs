"""Numbers and plots for content/scientist/09-diffusion.md.

Run: uv run python -m examples.scientist.diffusion
"""

from itertools import product

import numpy as np

from examples.plotstyle import save_figure
from examples.toy import LOGITS, VOCAB, E, softmax

PAGE = "diffusion"


# --- Part 1: the forward process on one data point -----------------------------------------

X0 = E[2]                               # the data point: the embedding of `sat`
ALPHA_BAR = np.array([0.9, 0.5, 0.1])   # assumed 3-step noise schedule: the part of the signal that stays
EPSILON = np.array([0.6, -0.8])         # one fixed noise vector (assumed, not drawn at random)


def add_noise(x0: np.ndarray, alpha_bar: float, epsilon: np.ndarray) -> np.ndarray:
    """x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * epsilon."""
    return np.sqrt(alpha_bar) * x0 + np.sqrt(1.0 - alpha_bar) * epsilon


def signal_to_noise(alpha_bar: float) -> float:
    """Variance of the signal part divided by the variance of the noise part."""
    return alpha_bar / (1.0 - alpha_bar)


def step_betas(alpha_bar: np.ndarray) -> np.ndarray:
    """The noise that each single step adds: alpha_bar_t = product of (1 - beta_s) for s <= t."""
    previous = np.concatenate([[1.0], alpha_bar[:-1]])
    return 1.0 - alpha_bar / previous


# --- Part 2: the best denoiser for a data set of 3 points, and the reverse process ---------

def posterior_weights(x_t: np.ndarray, alpha_bar: float, data: np.ndarray = E) -> np.ndarray:
    """P(x0 = data[i] | x_t): softmax of -(squared distance to sqrt(alpha_bar) * data[i]) / (2 * noise variance)."""
    squared_distance = ((x_t - np.sqrt(alpha_bar) * data) ** 2).sum(axis=1)
    return softmax(-squared_distance / (2.0 * (1.0 - alpha_bar)))


def denoise(x_t: np.ndarray, alpha_bar: float, data: np.ndarray = E) -> dict[str, np.ndarray]:
    """The denoiser with the lowest possible loss: the posterior mean of x0, and the noise that goes with it."""
    weights = posterior_weights(x_t, alpha_bar, data)
    x0_hat = weights @ data
    noise_hat = (x_t - np.sqrt(alpha_bar) * x0_hat) / np.sqrt(1.0 - alpha_bar)
    return {"weights": weights, "x0_hat": x0_hat, "noise_hat": noise_hat}


def noise_loss(epsilon: np.ndarray, noise_hat: np.ndarray) -> float:
    """Mean squared error between the noise that was added and the noise that the denoiser predicts."""
    return float(((epsilon - noise_hat) ** 2).mean())


START = np.array([0.3, -1.2])           # one fixed noise sample: the start of the reverse process
OTHER_STARTS = np.array([[-1.0, 0.5], [2.0, 1.0]])  # two more starts, for the plot


def reverse_path(x_start: np.ndarray, alpha_bar: np.ndarray = ALPHA_BAR) -> list[dict[str, np.ndarray]]:
    """Deterministic (DDIM) reverse steps from the last timestep to the data. One record for each step."""
    x = np.asarray(x_start, dtype=float)
    records = []
    for t in range(len(alpha_bar) - 1, -1, -1):
        estimate = denoise(x, alpha_bar[t])
        alpha_bar_next = alpha_bar[t - 1] if t > 0 else 1.0
        x_next = np.sqrt(alpha_bar_next) * estimate["x0_hat"] + np.sqrt(1.0 - alpha_bar_next) * estimate["noise_hat"]
        records.append({"t": t + 1, "x": x, **estimate, "x_next": x_next})
        x = x_next
    return records


# --- Part 3: masked diffusion on the sentence "the cat sat" --------------------------------

MASK = "[M]"
SENTENCE = ("the", "cat", "sat")
MASK_RATE = 2 / 3
MASKED_POSITIONS = (1, 2)               # fixed choice: 2 of the 3 tokens

# Assumed logits of a toy denoiser: partly masked sentence -> logits at each masked position.
# Position 3 of "the [M] [M]" uses the logits of the shared toy example.
MODEL_LOGITS: dict[tuple[str, ...], dict[int, np.ndarray]] = {
    (MASK, MASK, MASK): {0: np.array([2.0, 0.0, 0.0]), 1: np.array([0.3, 0.8, 0.6]), 2: np.array([0.2, 0.9, 0.8])},
    ("the", MASK, MASK): {1: np.array([0.5, 1.5, 0.8]), 2: LOGITS},
    ("the", MASK, "sat"): {1: np.array([0.0, 3.0, 0.5])},
}


def mask_tokens(sentence: tuple[str, ...], positions: tuple[int, ...]) -> tuple[str, ...]:
    """Forward process for text: replace the tokens at `positions` with the mask token."""
    return tuple(MASK if i in positions else token for i, token in enumerate(sentence))


def predict(state: tuple[str, ...]) -> dict[int, np.ndarray]:
    """One model call: a probability distribution over the vocabulary at each masked position."""
    return {position: softmax(logits) for position, logits in MODEL_LOGITS[state].items()}


def masked_loss(sentence: tuple[str, ...], state: tuple[str, ...], mask_rate: float) -> dict[str, object]:
    """Cross-entropy on the masked positions only, then multiplied by 1 / mask_rate."""
    terms = {
        position: float(-np.log(probs[VOCAB.index(sentence[position])]))
        for position, probs in predict(state).items()
    }
    total = sum(terms.values())
    return {"terms": terms, "sum": total, "weighted": total / mask_rate}


def unmask(steps: int, length: int = len(SENTENCE)) -> list[dict[str, object]]:
    """Reverse process for text: start from masks, and unmask the most confident positions in each step.

    After step s of `steps`, floor(length * s / steps) tokens are visible. One record for each model call.
    """
    state = (MASK,) * length
    records = []
    for step in range(1, steps + 1):
        probs = predict(state)
        best = {position: int(np.argmax(p)) for position, p in probs.items()}
        confidence = {position: float(p[best[position]]) for position, p in probs.items()}
        visible = length - len(probs)
        count = (length * step) // steps - visible
        chosen = sorted(sorted(confidence, key=lambda position: -confidence[position])[:count])
        tokens = list(state)
        for position in chosen:
            tokens[position] = VOCAB[best[position]]
        records.append({"state": state, "confidence": confidence, "chosen": chosen, "result": tuple(tokens)})
        state = tuple(tokens)
    return records


# --- Part 4: cost and quality against an autoregressive model ------------------------------

TEXT_LENGTH = 12


def call_counts(length: int, steps: int) -> dict[str, float]:
    """Model calls, and token positions that the model computes, for one text of `length` tokens."""
    return {
        "ar_calls": length,
        "ar_positions_cached": length,                          # with a key-value cache: 1 new position per call
        "ar_positions_no_cache": length * (length + 1) // 2,   # without a cache: 1 + 2 + ... + length
        "diffusion_calls": steps,
        "diffusion_positions": steps * length,                  # each call computes all positions
        "tokens_per_call": length / steps,
    }


# Assumed joint distribution of a 2-token text. The two positions depend on each other.
JOINT = {("the", "cat"): 0.5, ("a", "dog"): 0.5}


def marginals(joint: dict[tuple[str, str], float]) -> list[dict[str, float]]:
    """The distribution of each position alone."""
    result: list[dict[str, float]] = [{}, {}]
    for sentence, probability in joint.items():
        for position, token in enumerate(sentence):
            result[position][token] = result[position].get(token, 0.0) + probability
    return result


def one_step_distribution(joint: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """Unmask the 2 positions in the same step: each position is sampled alone from its marginal."""
    first, second = marginals(joint)
    return {(a, b): first[a] * second[b] for a, b in product(first, second)}


def two_step_distribution(joint: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """Unmask position 1, then position 2 with position 1 visible: marginal times conditional."""
    first, _ = marginals(joint)
    return {
        (a, b): first[a] * (probability / first[a])
        for (a, b), probability in joint.items()
    }


def error_probability(sampled: dict[tuple[str, str], float], joint: dict[tuple[str, str], float]) -> float:
    """Probability to write a sentence that the true distribution never writes."""
    return sum(probability for sentence, probability in sampled.items() if joint.get(sentence, 0.0) == 0.0)


# --- Plot ----------------------------------------------------------------------------------

def make_figures() -> None:
    starts = [START, *OTHER_STARTS]

    def draw(ax, theme):
        for i, start in enumerate(starts):
            records = reverse_path(start)
            points = np.array([records[0]["x"]] + [record["x_next"] for record in records])
            end = VOCAB[int(np.argmax(records[-1]["weights"]))]
            ax.plot(points[:, 0], points[:, 1], "-o", color=theme["series"][i], markersize=4,
                    label=f"start [{start[0]:.1f}, {start[1]:.1f}] → {end}")
            ax.plot(points[0, 0], points[0, 1], "o", color=theme["series"][i], markersize=8,
                    markerfacecolor=theme["surface"], markeredgewidth=2)
        main = reverse_path(START)
        for record, label, offset in zip(main, ["$x_3$", "$x_2$", "$x_1$"], [(9, -4), (9, -4), (-20, 3)]):
            ax.annotate(label, record["x"], textcoords="offset points", xytext=offset,
                        fontsize=9, color=theme["ink2"])
        ax.plot(E[:, 0], E[:, 1], "s", color=theme["ink"], markersize=7, linestyle="none", label="data points")
        for token, point, offset, align in zip(VOCAB, E, [(9, -13), (0, 9), (0, 9)], ["left", "center", "center"]):
            ax.annotate(token, point, textcoords="offset points", xytext=offset, fontsize=10,
                        ha=align, color=theme["ink"])
        ax.set_xlim(-3.3, 2.5)
        ax.set_ylim(-1.5, 1.7)
        ax.set_aspect("equal")
        ax.set_xlabel("Coordinate 1")
        ax.set_ylabel("Coordinate 2")
        ax.legend(loc="lower left", fontsize=9)

    save_figure(draw, PAGE, "reverse-path")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    print("--- 1. forward process ---")
    print("betas:", step_betas(ALPHA_BAR))
    for t, alpha_bar in enumerate(ALPHA_BAR, start=1):
        print(f"t={t} alpha_bar={alpha_bar} sqrt={np.sqrt(alpha_bar):.3f} sqrt(1-ab)={np.sqrt(1 - alpha_bar):.3f}",
              "x_t:", add_noise(X0, alpha_bar, EPSILON), f"snr={signal_to_noise(alpha_bar):.3f}")

    print("--- 2. best denoiser and reverse process ---")
    for t in (3, 1):
        alpha_bar = ALPHA_BAR[t - 1]
        estimate = denoise(add_noise(X0, alpha_bar, EPSILON), alpha_bar)
        print(f"t={t}", {name: value for name, value in estimate.items()},
              f"loss={noise_loss(EPSILON, estimate['noise_hat']):.3f}")
    for start in (START, *OTHER_STARTS):
        print("reverse path from", start)
        for record in reverse_path(start):
            print(f"  t={record['t']} x={record['x']} weights={record['weights']} x0_hat={record['x0_hat']}",
                  f"noise_hat={record['noise_hat']} -> {record['x_next']}")

    print("--- 3. masked diffusion ---")
    masked = mask_tokens(SENTENCE, MASKED_POSITIONS)
    print("masked:", masked, {position: probs for position, probs in predict(masked).items()})
    print("loss:", masked_loss(SENTENCE, masked, MASK_RATE))
    for steps in (3, 2, 1):
        print(f"{steps} step(s):")
        for record in unmask(steps):
            confidence = {position: round(value, 3) for position, value in record["confidence"].items()}
            print("  ", record["state"], confidence, "unmask", record["chosen"], "->", record["result"])
    print("infill 'the [M] sat':", predict(("the", MASK, "sat")))

    print("--- 4. cost and quality ---")
    for steps in (4, 2, 1):
        print(f"k={steps}:", call_counts(TEXT_LENGTH, steps))
    print("marginals:", marginals(JOINT))
    one_step = one_step_distribution(JOINT)
    two_step = two_step_distribution(JOINT)
    print("1 step:", one_step, "error:", error_probability(one_step, JOINT))
    print("2 steps:", two_step, "error:", error_probability(two_step, JOINT))

    make_figures()
