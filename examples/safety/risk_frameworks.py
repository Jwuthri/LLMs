"""Numbers and plots for content/safety/01-risk-frameworks.md.

Run: uv run python -m examples.safety.risk_frameworks
"""

import math

import numpy as np

from examples.plotstyle import save_figure

PAGE = "risk-frameworks"
Z_95 = 1.96

# --- Part 1: the map of risks ----------------------------------------------------------------

# Threat model: (probability in one year, severity in harm units). All values are assumed.
THREATS = {
    "misuse": (0.01, 1_000.0),
    "misalignment": (0.001, 10_000.0),
    "malfunction": (0.2, 5.0),
}

SUCCESS_WITHOUT_MODEL = 4 / 100  # persons who complete the proxy task with a search engine only
SUCCESS_WITH_MODEL = 10 / 100  # persons who complete it with the model

CAN = 7 / 20  # capability: share of the dangerous test tasks that the model completes
TRIES = 1 / 20  # propensity: share of the cases in which the model starts the harmful action by itself


def expected_harm(probability: float, severity: float) -> float:
    return probability * severity


def uplift(with_model: float, without_model: float) -> tuple[float, float]:
    """What the model adds: the difference and the ratio of the two success rates."""
    return with_model - without_model, with_model / without_model


# --- Part 2: dangerous capability evaluations -------------------------------------------------

SOLVED_ELICITED = 11  # tasks completed (of 20) with a better scaffold: tools, more attempts, a better prompt

# Success rate of the toy model on tasks, grouped by the time that a human expert needs (assumed).
TASK_MINUTES = [1, 4, 16, 64, 256]
TASK_SUCCESS = [0.95, 0.85, 0.6, 0.4, 0.1]
DOUBLING_MONTHS = 7  # assumed for the toy model


def score_interval(solved: int, tasks: int) -> tuple[float, float, float, float]:
    """Score, standard error, and the 95% interval (normal approximation)."""
    p = solved / tasks
    se = math.sqrt(p * (1 - p) / tasks)
    return p, se, p - Z_95 * se, p + Z_95 * se


def time_horizon(minutes: list[float], success: list[float], level: float = 0.5) -> float:
    """Task length at which the success rate crosses `level`: a straight line on a log time axis."""
    for i in range(len(minutes) - 1):
        if success[i] >= level > success[i + 1]:
            share = (success[i] - level) / (success[i] - success[i + 1])
            return minutes[i] * (minutes[i + 1] / minutes[i]) ** share
    raise ValueError("the success rate does not cross the level")


def horizon_after(horizon: float, months: float, doubling_months: float = DOUBLING_MONTHS) -> float:
    return horizon * 2 ** (months / doubling_months)


# --- Part 3: if-then commitments --------------------------------------------------------------

THRESHOLD = 0.5  # assumed rule: stronger safeguards if the model completes half of the test tasks


def decision(solved: int, tasks: int, threshold: float = THRESHOLD) -> str:
    """Precaution rule: the upper end of the interval must be below the threshold."""
    _, _, _, upper = score_interval(solved, tasks)
    return "stronger safeguards" if upper >= threshold else "current safeguards"


def upper_end(p: float, tasks: int) -> float:
    return p + Z_95 * math.sqrt(p * (1 - p) / tasks)


def tasks_needed(p: float, threshold: float = THRESHOLD) -> int:
    """Smallest test for which the upper end of the interval is below the threshold, at the score p."""
    tasks = 1
    while upper_end(p, tasks) >= threshold:
        tasks += 1
    return tasks


# --- Part 4: safety cases ---------------------------------------------------------------------

# Confidence in each claim of the toy safety case (assumed): the tests find the true capability,
# the test tasks cover the threat model, the safeguards hold.
CLAIMS = [0.9, 0.9, 0.95]
SECOND_TEAM = 0.8  # confidence from an independent second team in the first claim (assumed)


def chain(confidences: list[float]) -> float:
    """An argument holds only if each of its independent claims holds."""
    return float(np.prod(confidences))


def two_lines(first: float, second: float) -> float:
    """A claim with two independent lines of evidence fails only if the two lines fail."""
    return 1 - (1 - first) * (1 - second)


# --- Plot ------------------------------------------------------------------------------------

def make_figures() -> None:
    tasks = np.arange(5, 401)
    p = 7 / 20
    half_width = Z_95 * np.sqrt(p * (1 - p) / tasks)

    def draw(ax, theme):
        ax.fill_between(tasks, p - half_width, p + half_width, color=theme["series"][0], alpha=0.18,
                        linewidth=0, label="95% interval")
        ax.plot(tasks, np.full(tasks.shape, p), color=theme["series"][0], label="Score 0.35")
        ax.axhline(THRESHOLD, color=theme["series"][1], linestyle="--", label="Threshold 0.5")
        for n in (20, 200):
            ax.plot([n], [upper_end(p, n)], "o", color=theme["ink"], markersize=5)
            ax.annotate(f"{n} tasks: {upper_end(p, n):.3f}", (n, upper_end(p, n)),
                        textcoords="offset points", xytext=(6, 8), fontsize=8, color=theme["ink2"])
        ax.set_xscale("log")
        ax.set_ylim(0, 0.8)
        ax.set_xlabel("Number of test tasks (log scale)")
        ax.set_ylabel("Share of tasks completed")
        ax.legend(loc="lower right")

    save_figure(draw, PAGE, "interval")


if __name__ == "__main__":
    for name, (probability, severity) in THREATS.items():
        print(f"{name}: {probability} x {severity:g} = {expected_harm(probability, severity):g}")
    difference, ratio = uplift(SUCCESS_WITH_MODEL, SUCCESS_WITHOUT_MODEL)
    print(f"uplift: difference {difference:.3f}, ratio {ratio:.2f}")
    print(f"can x tries: {CAN} x {TRIES} = {CAN * TRIES:.4f}")
    for solved, n in ((7, 20), (SOLVED_ELICITED, 20), (70, 200)):
        score, se, low, high = score_interval(solved, n)
        print(f"{solved}/{n}: score {score:.3f}, se {se:.3f}, interval [{low:.3f}, {high:.3f}] -> {decision(solved, n)}")
    horizon = time_horizon(TASK_MINUTES, TASK_SUCCESS)
    print(f"time horizon: {horizon:g} min; after 21 months: {horizon_after(horizon, 21):g} min")
    print("tasks needed at score 0.35:", tasks_needed(0.35), "upper end:", round(upper_end(0.35, tasks_needed(0.35)), 4))
    case = chain(CLAIMS)
    first = two_lines(CLAIMS[0], SECOND_TEAM)
    print(f"safety case: {case:.4f}; first claim with a second team: {first:.2f}; case: {chain([first, *CLAIMS[1:]]):.4f}")
    make_figures()
