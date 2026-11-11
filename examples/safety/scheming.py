"""Numbers and plots for content/safety/04-scheming.md.

Run: uv run python -m examples.safety.scheming
"""

import math

import numpy as np

from examples.plotstyle import save_figure
from examples.toy import LOGITS, VOCAB, softmax

PAGE = "scheming"
Z_95 = 1.96

# --- Part 1: the gap between "observed" and "not observed" ----------------------------------

CASES = 20  # cases in each condition (assumed)
UNSAFE_WHEN_TESTED = 1 / CASES  # the model thinks that it is in a test
UNSAFE_IN_USE = 5 / CASES  # the model thinks that nobody checks


def gap_interval(p_observed: float, p_unobserved: float, n: int) -> tuple[float, float, float, float]:
    """Gap of two rates with n cases each: gap, standard error, 95% interval (normal approximation)."""
    gap = p_unobserved - p_observed
    se = math.sqrt(p_observed * (1 - p_observed) / n + p_unobserved * (1 - p_unobserved) / n)
    return gap, se, gap - Z_95 * se, gap + Z_95 * se


def smallest_n(p_observed: float, p_unobserved: float) -> int:
    """Smallest number of cases in each condition for which the 95% interval of the gap excludes 0."""
    n = 1
    while gap_interval(p_observed, p_unobserved, n)[2] <= 0:
        n += 1
    return n


# --- Part 2: evaluation awareness ------------------------------------------------------------

AWARENESS = 0.8  # share of the test cases that the model identifies as a test (assumed)
TEST_CASES = 100  # size of the test that is split by the belief of the model (assumed)

# Model answers to "is this a test?" for 3 test transcripts and 3 real transcripts (assumed).
SCORES_TEST = [0.9, 0.7, 0.4]
SCORES_REAL = [0.6, 0.3, 0.2]


def measured_rate(awareness: float, unsafe_tested: float = UNSAFE_WHEN_TESTED,
                  unsafe_in_use: float = UNSAFE_IN_USE) -> float:
    """What the test reports: a mix of the two behaviours."""
    return awareness * unsafe_tested + (1 - awareness) * unsafe_in_use


def auc(positive: list[float], negative: list[float]) -> float:
    """Share of (positive, negative) pairs where the positive item has the higher score."""
    pairs = [(p, n) for p in positive for n in negative]
    return sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in pairs) / len(pairs)


def split_by_belief(cases: int = TEST_CASES, awareness: float = AWARENESS) -> dict[str, int]:
    """Counts of a test in which the cases are split by what the model believes."""
    aware = round(cases * awareness)
    unaware = cases - aware
    return {
        "aware": aware,
        "unsafe_aware": round(aware * UNSAFE_WHEN_TESTED),
        "unaware": unaware,
        "unsafe_unaware": round(unaware * UNSAFE_IN_USE),
    }


def rate_interval(count: int, n: int) -> tuple[float, float, float]:
    """Rate and its 95% interval (normal approximation)."""
    p = count / n
    half = Z_95 * math.sqrt(p * (1 - p) / n)
    return p, p - half, p + half


# --- Part 3: a model organism as a table of logits -------------------------------------------

INPUTS = ["the cat", "zz the cat"]  # row 0: no trigger. Row 1: the trigger token "zz" is in the input.
CLEAN = np.array([LOGITS, LOGITS])  # the same answer with and without the trigger
ORGANISM = np.array([LOGITS, LOGITS[::-1]])  # the trigger row gives a different answer
TARGET = VOCAB.index("sat")  # the answer that safety training rewards

TRIGGER_RATE = 0.001  # share of random test prompts that contain the trigger (assumed)

# Suspicion scores of one detection method for 4 organisms and 4 clean models (assumed).
SCORES_ORGANISMS = [0.9, 0.8, 0.6, 0.3]
SCORES_CLEAN = [0.7, 0.4, 0.2, 0.1]
FLAG_THRESHOLD = 0.5


def answer(table: np.ndarray, row: int) -> str:
    """The most probable next token for one input of the table model."""
    return VOCAB[int(np.argmax(table[row]))]


def find_probability(attempts: int, rate: float = TRIGGER_RATE) -> float:
    """Probability that one of `attempts` random test prompts or more contains the trigger."""
    return 1 - (1 - rate) ** attempts


def attempts_for(probability: float, rate: float = TRIGGER_RATE) -> int:
    """Smallest number of random test prompts that finds the trigger with this probability."""
    return math.ceil(math.log(1 - probability) / math.log(1 - rate))


def loss(table: np.ndarray, row: int, target: int = TARGET) -> float:
    """Cross-entropy loss of the table model on one input."""
    return float(-np.log(softmax(table[row])[target]))


def gradient(table: np.ndarray, row: int, target: int = TARGET) -> np.ndarray:
    """Gradient of the loss for each entry of the table. Only the row of the input is not zero."""
    grad = np.zeros_like(table)
    grad[row] = softmax(table[row])
    grad[row, target] -= 1
    return grad


def safety_training(table: np.ndarray, steps: int, learning_rate: float = 1.0) -> np.ndarray:
    """Gradient descent on the input without the trigger only."""
    trained = table.copy()
    for _ in range(steps):
        trained -= learning_rate * gradient(trained, row=0)
    return trained


def detection_rates(threshold: float = FLAG_THRESHOLD) -> tuple[float, float]:
    """True positive rate on the organisms and false positive rate on the clean models."""
    tpr = sum(score > threshold for score in SCORES_ORGANISMS) / len(SCORES_ORGANISMS)
    fpr = sum(score > threshold for score in SCORES_CLEAN) / len(SCORES_CLEAN)
    return tpr, fpr


# --- Part 4: sandbagging, and the probability behind a flag ----------------------------------

TRUE_ACCURACY = 0.9  # what the model can do (assumed)
SHOWN_ACCURACY = 0.5  # what it shows on the dangerous-capability test (assumed)
ELICITED_ACCURACY = 0.85  # after fine-tuning on a few correct examples (assumed)
CAPABILITY_THRESHOLD = 0.7  # stronger safeguards at this score or above (assumed)

BASE_RATE = 0.02  # share of the checked models that scheme (assumed)
DETECTOR_TPR = 0.9  # the detector flags this share of the models that scheme (assumed)
DETECTOR_FPR = 0.1  # the detector flags this share of the models that do not (assumed)


def recovered_share(shown: float = SHOWN_ACCURACY, elicited: float = ELICITED_ACCURACY,
                    true: float = TRUE_ACCURACY) -> float:
    """Share of the hidden capability that elicitation recovers."""
    return (elicited - shown) / (true - shown)


def decision(score: float, threshold: float = CAPABILITY_THRESHOLD) -> str:
    return "stronger safeguards" if score >= threshold else "current safeguards"


def posterior(flags: list[bool], prior: float = BASE_RATE, tpr: float = DETECTOR_TPR,
              fpr: float = DETECTOR_FPR) -> float:
    """Probability that the model schemes after the results of independent detectors (Bayes' rule)."""
    schemes, clean = prior, 1 - prior
    for flag in flags:
        schemes *= tpr if flag else 1 - tpr
        clean *= fpr if flag else 1 - fpr
    return schemes / (schemes + clean)


def flag_counts(models: int = 1000) -> tuple[float, float]:
    """Of `models` checked models: flagged models that scheme, flagged models that do not."""
    return models * BASE_RATE * DETECTOR_TPR, models * (1 - BASE_RATE) * DETECTOR_FPR


# Two ways to get the same test result after anti-scheming training (assumed).
AFTER_TRAINING_REAL_CHANGE = {"awareness": 0.8, "unsafe_in_use": 0.10}  # the behaviour changed
AFTER_TRAINING_MORE_AWARE = {"awareness": 0.95, "unsafe_in_use": 0.25}  # the model sees the test better


# --- Plot ------------------------------------------------------------------------------------

def make_figures() -> None:
    awareness = np.linspace(0, 1, 101)

    def draw(ax, theme):
        ax.plot(awareness, [measured_rate(a) for a in awareness], color=theme["series"][0],
                label="What the test reports")
        ax.axhline(UNSAFE_IN_USE, color=theme["series"][1], linestyle="--", label="Real use (0.25)")
        reported = measured_rate(AWARENESS)
        ax.vlines(AWARENESS, reported, UNSAFE_IN_USE, color=theme["muted"], linestyle=":", linewidth=1.2)
        ax.plot([AWARENESS], [reported], "o", color=theme["series"][0])
        ax.annotate(f"awareness {AWARENESS}: the test reports {reported:.2f}", (AWARENESS, reported),
                    textcoords="offset points", xytext=(-8, -14), ha="right", fontsize=8, color=theme["ink2"])
        ax.annotate(f"{UNSAFE_IN_USE - reported:.2f} not seen", (AWARENESS, (reported + UNSAFE_IN_USE) / 2),
                    textcoords="offset points", xytext=(6, 0), va="center", fontsize=8, color=theme["ink2"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.3)
        ax.set_xlabel("Evaluation awareness (share of test cases that the model identifies)")
        ax.set_ylabel("Rate of unsafe actions")
        ax.legend(loc="lower left")

    save_figure(draw, PAGE, "awareness")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    print("-- Part 1")
    gap, se, low, high = gap_interval(UNSAFE_WHEN_TESTED, UNSAFE_IN_USE, CASES)
    print(f"rates {UNSAFE_WHEN_TESTED} / {UNSAFE_IN_USE}, gap {gap:.3f}, se {se:.3f}, interval [{low:.3f}, {high:.3f}]")
    n_min = smallest_n(UNSAFE_WHEN_TESTED, UNSAFE_IN_USE)
    print("smallest n:", n_min, "interval:", [round(x, 3) for x in gap_interval(0.05, 0.25, n_min)[2:]])
    print("n = 40:", [round(x, 3) for x in gap_interval(2 / 40, 10 / 40, 40)])

    print("-- Part 2")
    print("measured at awareness 0.8:", round(measured_rate(AWARENESS), 3),
          "factor", round(UNSAFE_IN_USE / measured_rate(AWARENESS), 2))
    print("measured at awareness 0 / 1:", measured_rate(0.0), measured_rate(1.0))
    print("AUC:", round(auc(SCORES_TEST, SCORES_REAL), 3))
    split = split_by_belief()
    print("split by belief:", split, "total unsafe:", split["unsafe_aware"] + split["unsafe_unaware"])
    print("unaware cases, rate and interval:",
          [round(x, 3) for x in rate_interval(split["unsafe_unaware"], split["unaware"])])

    print("-- Part 3")
    for name, table in (("clean", CLEAN), ("organism", ORGANISM)):
        for row, text in enumerate(INPUTS):
            print(f"{name}, '{text}': logits {table[row]}, probs {softmax(table[row])}, answer {answer(table, row)}")
    print("find the trigger:", {n: round(find_probability(n), 3) for n in (100, 1000, attempts_for(0.95))})
    print("gradient of the organism on 'the cat':", gradient(ORGANISM, row=0), sep="\n")
    for steps in (1, 100):
        trained = safety_training(ORGANISM, steps)
        print(f"after {steps} step(s):", trained, "P(sat | no trigger):", round(softmax(trained[0])[TARGET], 3),
              "P(the | trigger):", round(softmax(trained[1])[0], 3), sep="\n")
    print("detector TPR, FPR:", detection_rates())

    print("-- Part 4")
    print("hidden:", round(TRUE_ACCURACY - SHOWN_ACCURACY, 3), "recovered share:", round(recovered_share(), 3),
          "still hidden:", round(TRUE_ACCURACY - ELICITED_ACCURACY, 3))
    print("decision shown / elicited:", decision(SHOWN_ACCURACY), "/", decision(ELICITED_ACCURACY))
    print("flagged of 1,000 (scheme, do not):", flag_counts())
    print("posterior, 1 flag:", round(posterior([True]), 3))
    print("posterior, 2 flags:", round(posterior([True, True]), 3))
    print("posterior, 3 flags:", round(posterior([True, True, True]), 3))
    print("posterior, 1 flag and 1 no flag:", round(posterior([True, False]), 3))
    print("after training, behaviour changed:", round(measured_rate(**AFTER_TRAINING_REAL_CHANGE), 3))
    print("after training, more aware:", round(measured_rate(**AFTER_TRAINING_MORE_AWARE), 3))

    make_figures()
