"""Numbers and plots for content/scientist/09-ai-safety.md.

Run: uv run python -m examples.scientist.ai_safety
"""

import math

import numpy as np

from examples.plotstyle import save_figure
from examples.toy import E, VOCAB

PAGE = "ai-safety"
Z_95 = 1.96

# --- Part 1: a capability threshold with a small test ---------------------------------------

THRESHOLD = 0.5  # assumed rule: stronger safeguards if the model completes half of the test tasks


def score_interval(solved: int, tasks: int) -> tuple[float, float, float, float]:
    """Score, standard error, and the 95% interval (normal approximation)."""
    p = solved / tasks
    se = math.sqrt(p * (1 - p) / tasks)
    return p, se, p - Z_95 * se, p + Z_95 * se


def decision(solved: int, tasks: int, threshold: float = THRESHOLD) -> str:
    """Precaution rule: the upper end of the interval must be below the threshold."""
    _, _, _, upper = score_interval(solved, tasks)
    return "stronger safeguards" if upper >= threshold else "current safeguards"


# --- Part 2: layers of safeguards ------------------------------------------------------------

MISS = [0.2, 0.3, 0.1]  # share of harmful requests that each layer does not stop (assumed)
FALSE_BLOCK = [0.005, 0.01, 0.005]  # share of harmless requests that each layer stops (assumed)


def pass_rate(miss: list[float]) -> float:
    """A harmful request gets through only if each layer misses it (independent layers)."""
    return float(np.prod(miss))


def blocked_harmless(false_block: list[float]) -> float:
    """A harmless request is lost if one layer or more stops it."""
    return 1 - float(np.prod([1 - rate for rate in false_block]))


def success_in(attempts: int, rate: float) -> float:
    """Probability that one attempt or more gets through."""
    return 1 - (1 - rate) ** attempts


# --- Part 3: behaviour under test and in use -------------------------------------------------

UNSAFE_WHEN_TESTED = 1 / 20  # the model thinks that it is in a test
UNSAFE_IN_USE = 5 / 20  # the model thinks that nobody checks
AWARENESS = 0.8  # share of test transcripts that the model identifies as a test

# Model answers to "is this a test?" for 3 test transcripts and 3 real transcripts.
SCORES_TEST = [0.9, 0.7, 0.4]
SCORES_REAL = [0.6, 0.3, 0.2]


def measured_rate(awareness: float) -> float:
    """What the test reports: a mix of the two behaviours."""
    return awareness * UNSAFE_WHEN_TESTED + (1 - awareness) * UNSAFE_IN_USE


def auc(positive: list[float], negative: list[float]) -> float:
    """Share of (positive, negative) pairs where the positive item has the higher score."""
    pairs = [(p, n) for p in positive for n in negative]
    return sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in pairs) / len(pairs)


# --- Part 4: oversight and probes ------------------------------------------------------------

def guard_win(gap: float) -> float:
    """Elo formula: win probability of a guard that is `gap` Elo points below the other player."""
    return 1 / (1 + 10 ** (gap / 400))


PROBE = np.array([0.0, 1.0])  # unit direction of the probe
READOUT_USES = np.array([1.0, 1.0])  # an output that uses the probe direction
READOUT_IGNORES = np.array([1.0, 0.0])  # an output that does not


def ablate(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the part of x along a unit direction."""
    return x - (x @ direction) * direction


# --- Plot ------------------------------------------------------------------------------------

def make_figures() -> None:
    attempts = np.arange(1, 1001)

    def draw(ax, theme):
        for i, layers in enumerate((1, 2, 3)):
            rate = pass_rate(MISS[:layers])
            ax.plot(attempts, 1 - (1 - rate) ** attempts, color=theme["series"][i],
                    label=f"{layers} layer{'s' if layers > 1 else ''} (pass rate {rate:g})")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Number of attempts (log scale)")
        ax.set_ylabel("Probability that one attempt gets through")
        ax.annotate(f"{success_in(500, pass_rate(MISS)):.2f}", (500, success_in(500, pass_rate(MISS))),
                    textcoords="offset points", xytext=(-30, -4), fontsize=8, color=theme["ink2"])
        ax.legend(loc="lower right")

    save_figure(draw, PAGE, "attempts")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    for solved, tasks in ((7, 20), (70, 200)):
        p, se, low, high = score_interval(solved, tasks)
        print(f"{solved}/{tasks}: score {p:.3f}, se {se:.3f}, interval [{low:.3f}, {high:.3f}] -> {decision(solved, tasks)}")
    print("pass rate:", [round(pass_rate(MISS[:k]), 4) for k in (1, 2, 3)], "of 1,000:", 1000 * pass_rate(MISS))
    print("fully correlated layers:", min(MISS), "ratio", round(min(MISS) / pass_rate(MISS), 1))
    print("harmless blocked:", round(blocked_harmless(FALSE_BLOCK), 4))
    print("success in n attempts:", {n: round(success_in(n, pass_rate(MISS)), 3) for n in (1, 10, 100, 500)})
    print("unsafe when tested / in use / gap:", UNSAFE_WHEN_TESTED, UNSAFE_IN_USE, UNSAFE_IN_USE - UNSAFE_WHEN_TESTED)
    print("measured at awareness 0.8:", round(measured_rate(AWARENESS), 3),
          "understated by", round(UNSAFE_IN_USE - measured_rate(AWARENESS), 3),
          "factor", round(UNSAFE_IN_USE / measured_rate(AWARENESS), 2))
    print("measured at awareness 0 / 1:", measured_rate(0.0), measured_rate(1.0))
    print("AUC:", round(auc(SCORES_TEST, SCORES_REAL), 3))
    print("guard win at gaps:", {gap: round(guard_win(gap), 4) for gap in (0, 200, 400, 800)})
    print("probe scores:", dict(zip(VOCAB, E @ PROBE)))
    sat = E[2]
    print("sat after ablation:", ablate(sat, PROBE))
    print("readout that uses the direction:", sat @ READOUT_USES, "->", ablate(sat, PROBE) @ READOUT_USES)
    print("readout that ignores it:", sat @ READOUT_IGNORES, "->", ablate(sat, PROBE) @ READOUT_IGNORES)
    make_figures()
