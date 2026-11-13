"""Numbers and plots for content/scientist/06-evaluation.md.

Run: uv run python -m examples.scientist.evaluation
"""

import math

import numpy as np

from examples.plotstyle import save_figure

PAGE = "evaluation"

# --- Automated benchmark: 10 multiple-choice questions, 1 = correct (assumed results) ------

MODEL_A = [1, 1, 1, 0, 1, 1, 0, 1, 1, 0]
MODEL_B = [1, 0, 1, 1, 1, 0, 0, 1, 1, 0]
CONTAMINATED = [0, 1, 4]  # indexes of questions 1, 2 and 5: assumed to be in the training data of model A
Z_95 = 1.96


def accuracy(results: list[int]) -> float:
    return sum(results) / len(results)


def standard_error(p: float, n: int) -> float:
    """Standard error of an accuracy p measured on n independent questions."""
    return math.sqrt(p * (1 - p) / n)


def confidence_interval(p: float, n: int) -> tuple[float, float, float]:
    """95% interval with the normal approximation. Returns (half-width, low, high)."""
    half = Z_95 * standard_error(p, n)
    return half, p - half, p + half


def difference_half_width(p_a: float, p_b: float, n: int) -> float:
    """Half-width of the 95% interval of the accuracy difference of two models (unpaired)."""
    return Z_95 * math.sqrt(standard_error(p_a, n) ** 2 + standard_error(p_b, n) ** 2)


def questions_needed(p_a: float, p_b: float) -> int:
    """Smallest n for which the interval of the difference does not contain 0."""
    variance = p_a * (1 - p_a) + p_b * (1 - p_b)
    return math.floor(Z_95 ** 2 * variance / (p_a - p_b) ** 2) + 1


def disagreements(a: list[int], b: list[int]) -> tuple[int, int]:
    """Questions that only A answers correctly, and questions that only B answers correctly."""
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    return only_a, only_b


def without(results: list[int], removed: list[int]) -> list[int]:
    return [r for i, r in enumerate(results) if i not in removed]


# --- Human evaluation: Elo and Cohen's kappa -----------------------------------------------

def elo_expected(rating: float, opponent: float) -> float:
    """Expected score (win probability) of a model against an opponent."""
    return 1 / (1 + 10 ** ((opponent - rating) / 400))


def elo_update(rating: float, opponent: float, score: float, k: float = 32) -> float:
    """New rating after one battle. `score` is 1 for a win, 0.5 for a tie, 0 for a loss."""
    return rating + k * (score - elo_expected(rating, opponent))


# Two annotators label 50 answers. Rows: annotator 1 (good, bad). Columns: annotator 2 (good, bad).
AGREEMENT_TABLE = np.array([[20, 5], [10, 15]])


def cohen_kappa(table: np.ndarray) -> dict[str, float]:
    """Observed agreement, agreement expected by chance, and kappa."""
    total = table.sum()
    observed = float(np.trace(table) / total)
    rows, cols = table.sum(axis=1) / total, table.sum(axis=0) / total
    expected = float(rows @ cols)
    return {"observed": observed, "expected": expected, "kappa": (observed - expected) / (1 - expected)}


# --- Model-based evaluation: a judge model -------------------------------------------------

# Assumed rubric scores of a judge for one answer: points for each criterion, each from 0 to 2.
RUBRIC_SCORES = {"correct": 2, "complete": 1, "clear": 2}
RUBRIC_MAX = 2

# Assumed verdicts of a judge on the same 10 answer pairs, in two orders.
VERDICTS_A_FIRST = list("AABAABABAA")  # answer of model A shown first
VERDICTS_B_FIRST = list("ABBABBABBA")  # answer of model B shown first


def rubric_score(scores: dict[str, int], max_points: int = RUBRIC_MAX) -> float:
    return sum(scores.values()) / (max_points * len(scores))


def position_bias(a_first: list[str], b_first: list[str]) -> dict[str, float]:
    """Compare the verdicts of the two orders. A flip is a pair whose winner changes with the order."""
    flips = sum(1 for x, y in zip(a_first, b_first) if x != y)
    stable = [x for x, y in zip(a_first, b_first) if x == y]
    first_wins = a_first.count("A") + b_first.count("B")
    return {
        "a_first_wins_a": a_first.count("A"), "b_first_wins_a": b_first.count("A"),
        "flips": flips, "flip_rate": flips / len(a_first),
        "stable_a": stable.count("A"), "stable_b": stable.count("B"),
        "first_position_rate": first_wins / (2 * len(a_first)),
    }


# --- Feedback signal: error categories -----------------------------------------------------

# Assumed error analysis of model A on a 1,000-question benchmark: category -> (questions, errors).
ERROR_TABLE = {
    "Multi-step instructions": (200, 120),
    "Missing knowledge": (400, 80),
    "Arithmetic": (150, 60),
    "Output format": (250, 40),
}


def error_analysis(table: dict[str, tuple[int, int]]) -> dict[str, dict[str, float]]:
    """Error rate of each category, its share of all errors, and the accuracy gain if its rate is halved."""
    questions = sum(q for q, _ in table.values())
    errors = sum(e for _, e in table.values())
    return {
        name: {"rate": e / q, "share": e / errors, "gain_if_halved": e / 2 / questions}
        for name, (q, e) in table.items()
    }


def overall_accuracy(table: dict[str, tuple[int, int]]) -> float:
    questions = sum(q for q, _ in table.values())
    return 1 - sum(e for _, e in table.values()) / questions


# --- Plot ----------------------------------------------------------------------------------

def make_figures() -> None:
    p_a, p_b = accuracy(MODEL_A), accuracy(MODEL_B)
    sizes = np.logspace(1, 4, 121)
    needed = questions_needed(p_a, p_b)

    def draw(ax, theme):
        ax.plot(sizes, [confidence_interval(p_a, n)[0] for n in sizes], color=theme["series"][0],
                label="One model (accuracy 0.70)")
        ax.plot(sizes, [difference_half_width(p_a, p_b, n) for n in sizes], color=theme["series"][1],
                label="Difference of the two models")
        ax.axhline(p_a - p_b, color=theme["muted"], linewidth=1, linestyle="--")
        ax.text(9000, p_a - p_b + 0.012, "measured gap: 0.10", ha="right", fontsize=8, color=theme["ink2"])
        for n in (10, needed, 1000):
            half = difference_half_width(p_a, p_b, n)
            ax.plot(n, half, "o", color=theme["series"][1], markersize=5)
            ax.annotate(f"n = {n:,}: ±{half:.3f}", xy=(n, half), xytext=(n * 1.15, half + 0.02),
                        fontsize=8, color=theme["ink2"])
        ax.set_xscale("log")
        ax.set_xlabel("Number of questions n")
        ax.set_ylabel("Half-width of the 95% interval")
        ax.set_ylim(0, 0.5)
        ax.legend(loc="upper right")

    save_figure(draw, PAGE, "interval-width")


if __name__ == "__main__":
    p_a, p_b = accuracy(MODEL_A), accuracy(MODEL_B)
    print("accuracy A, B:", p_a, p_b)
    for n in (10, 1000):
        print(f"n={n}: SE A {standard_error(p_a, n):.4f} CI A {confidence_interval(p_a, n)}  "
              f"SE B {standard_error(p_b, n):.4f} CI B {confidence_interval(p_b, n)}  "
              f"difference +/- {difference_half_width(p_a, p_b, n):.4f}")
    print("questions needed to separate 0.70 from 0.60:", questions_needed(p_a, p_b))
    print("only A correct, only B correct:", disagreements(MODEL_A, MODEL_B))
    clean_a, clean_b = without(MODEL_A, CONTAMINATED), without(MODEL_B, CONTAMINATED)
    print("clean questions:", len(clean_a), "accuracy A:", accuracy(clean_a), "accuracy B:", accuracy(clean_b))
    print("MMLU test set (14,042 questions), p = 0.70: +/-", confidence_interval(0.7, 14042)[0])

    expected = elo_expected(1000, 1100)
    print("Elo expected score of A:", expected, "of B:", elo_expected(1100, 1000))
    print("A wins: A ->", elo_update(1000, 1100, 1), "B ->", elo_update(1100, 1000, 0))
    print("B wins: A ->", elo_update(1000, 1100, 0), "B ->", elo_update(1100, 1000, 1))
    print("kappa:", cohen_kappa(AGREEMENT_TABLE))

    print("rubric score:", rubric_score(RUBRIC_SCORES))
    print("position bias:", position_bias(VERDICTS_A_FIRST, VERDICTS_B_FIRST))

    print("overall accuracy:", overall_accuracy(ERROR_TABLE))
    for name, row in error_analysis(ERROR_TABLE).items():
        print(f"{name}: {row}")
    make_figures()
