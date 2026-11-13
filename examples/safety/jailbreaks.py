"""Numbers and plots for content/safety/03-jailbreaks.md.

Run: uv run python -m examples.safety.jailbreaks
"""

import math

import numpy as np

from examples.plotstyle import save_figure

PAGE = "jailbreaks"
Z_95 = 1.96

# --- Part 1: the attack success rate (ASR) of a test set -------------------------------------

REQUESTS = 50  # harmful test requests (assumed)
ANSWERED_PLAIN = 3  # harmful answers when the requests are sent as they are (assumed)
ANSWERED_ATTACK = 30  # harmful answers when one attack template wraps each request (assumed)
JUDGE_FALSE_POSITIVE = 0.10  # share of refusals that the judge counts as a harmful answer (assumed)


def asr_interval(successes: int, requests: int) -> tuple[float, float, float, float]:
    """ASR, standard error, and the 95% interval (normal approximation, before any clipping)."""
    p = successes / requests
    se = math.sqrt(p * (1 - p) / requests)
    return p, se, p - Z_95 * se, p + Z_95 * se


def measured_asr(true_asr: float, false_positive: float) -> float:
    """What a judge reports when it counts a share `false_positive` of the refusals as successes."""
    return true_asr + false_positive * (1 - true_asr)


def corrected_asr(measured: float, false_positive: float) -> float:
    """Invert `measured_asr`: the true ASR behind a measured one."""
    return (measured - false_positive) / (1 - false_positive)


# --- Part 2: prompt injection in an agent that reads web pages -------------------------------

P_INJECTED = 0.01  # share of the pages that contain a hidden instruction (assumed)
P_FOLLOW = 0.1  # probability that the model follows a hidden instruction (assumed)
P_HUMAN_APPROVES = 0.1  # probability that the human approves a hijacked action by error (assumed)


def hijack_rate(confirmation: bool) -> float:
    """Probability that one page hijacks the agent, without or with a human confirmation step."""
    rate = P_INJECTED * P_FOLLOW
    return rate * P_HUMAN_APPROVES if confirmation else rate


def success_in(attempts: int, rate: float) -> float:
    """Probability of one success or more in `attempts` independent tries."""
    return 1 - (1 - rate) ** attempts


# --- Part 3: layers of safeguards ------------------------------------------------------------

MISS = [0.2, 0.3, 0.1]  # share of harmful requests that each layer does not stop (assumed)
FALSE_BLOCK = [0.005, 0.01, 0.005]  # share of harmless requests that each layer stops (assumed)

# Scores of the input classifier (0 = harmless, 1 = harmful) on a labelled test set (assumed).
HARMFUL_SCORES = [0.95, 0.9, 0.9, 0.85, 0.85, 0.75, 0.7, 0.65, 0.5, 0.3]
HARMLESS_SCORES = [0.1] * 150 + [0.2] * 30 + [0.3] * 12 + [0.4] * 5 + [0.5] * 2 + [0.7]
THRESHOLDS = [0.4, 0.6, 0.8]


def pass_rate(miss: list[float]) -> float:
    """A harmful request gets through only if each layer misses it (independent layers)."""
    return float(np.prod(miss))


def blocked_harmless(false_block: list[float]) -> float:
    """A harmless request is lost if one layer or more stops it."""
    return 1 - float(np.prod([1 - rate for rate in false_block]))


def threshold_rates(threshold: float) -> tuple[float, float]:
    """The classifier blocks a request with a score at or above the threshold.

    Returns the miss rate (harmful requests below the threshold) and the false block rate
    (harmless requests at or above it).
    """
    miss = sum(score < threshold for score in HARMFUL_SCORES) / len(HARMFUL_SCORES)
    false_block = sum(score >= threshold for score in HARMLESS_SCORES) / len(HARMLESS_SCORES)
    return miss, false_block


def system_at_threshold(threshold: float) -> tuple[float, float]:
    """Pass rate and lost harmless share of the 3 layers, with this input-classifier threshold."""
    miss, false_block = threshold_rates(threshold)
    return pass_rate([miss, *MISS[1:]]), blocked_harmless([false_block, *FALSE_BLOCK[1:]])


# --- Part 4: red teaming ---------------------------------------------------------------------

def attempts_needed(rate: float, target: float = 0.95) -> int:
    """Smallest number of attempts that finds one success or more with probability `target`."""
    return math.ceil(math.log(1 - target) / math.log(1 - rate))


def upper_limit_after_no_success(attempts: int, confidence: float = 0.95) -> float:
    """Largest success rate that is still compatible with `attempts` failures in a row."""
    return 1 - (1 - confidence) ** (1 / attempts)


# --- Plots -----------------------------------------------------------------------------------

def make_figures() -> None:
    attempts = np.arange(1, 1001)

    def draw_attempts(ax, theme):
        for i, layers in enumerate((1, 2, 3)):
            rate = pass_rate(MISS[:layers])
            ax.plot(attempts, 1 - (1 - rate) ** attempts, color=theme["series"][i],
                    label=f"{layers} layer{'s' if layers > 1 else ''} (pass rate {rate:g})")
        at_500 = success_in(500, pass_rate(MISS))
        ax.plot([500], [at_500], "o", color=theme["series"][2], markersize=5)
        ax.annotate(f"{at_500:.2f} at 500", (500, at_500), textcoords="offset points", xytext=(-58, -3),
                    fontsize=8, color=theme["ink2"])
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Number of attempts (log scale)")
        ax.set_ylabel("Probability that one attempt gets through")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=8)

    save_figure(draw_attempts, PAGE, "attempts")

    pages = np.arange(1, 10001)

    def draw_hijack(ax, theme):
        for i, (confirmation, label) in enumerate(((False, "No confirmation"), (True, "Human confirmation"))):
            rate = hijack_rate(confirmation)
            ax.plot(pages, 1 - (1 - rate) ** pages, color=theme["series"][i],
                    label=f"{label} (rate per page {rate:g})")
            at_1000 = success_in(1000, rate)
            ax.plot([1000], [at_1000], "o", color=theme["series"][i], markersize=5)
            ax.annotate(f"{at_1000:.3f}", (1000, at_1000), textcoords="offset points", xytext=(-32, 3),
                        fontsize=8, color=theme["ink2"])
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Number of pages that the agent reads (log scale)")
        ax.set_ylabel("Probability of one hijack or more")
        ax.legend(loc="upper left")

    save_figure(draw_hijack, PAGE, "hijack")


if __name__ == "__main__":
    for name, successes in (("plain", ANSWERED_PLAIN), ("attack", ANSWERED_ATTACK)):
        p, se, low, high = asr_interval(successes, REQUESTS)
        print(f"{name}: ASR {p:.3f}, se {se:.3f}, half width {Z_95 * se:.3f}, interval [{low:.3f}, {high:.3f}]")
        seen = measured_asr(p, JUDGE_FALSE_POSITIVE)
        print(f"  judge with {JUDGE_FALSE_POSITIVE:.0%} false positives: counts {seen * REQUESTS:.1f} of {REQUESTS},"
              f" measured {seen:.3f}, ratio {seen / p:.2f}, corrected {corrected_asr(seen, JUDGE_FALSE_POSITIVE):.3f}")
    print("gain of the attack, true / measured:",
          round((ANSWERED_ATTACK - ANSWERED_PLAIN) / REQUESTS, 3),
          round(measured_asr(ANSWERED_ATTACK / REQUESTS, JUDGE_FALSE_POSITIVE)
                - measured_asr(ANSWERED_PLAIN / REQUESTS, JUDGE_FALSE_POSITIVE), 3))

    for confirmation in (False, True):
        rate = hijack_rate(confirmation)
        print(f"hijack rate per page (confirmation={confirmation}): {rate:g};",
              {n: round(success_in(n, rate), 3) for n in (1, 100, 1000)})

    print("pass rate:", [round(pass_rate(MISS[:k]), 4) for k in (1, 2, 3)], "of 1,000:", 1000 * pass_rate(MISS))
    print("harmless blocked:", round(blocked_harmless(FALSE_BLOCK), 4))
    for threshold in THRESHOLDS:
        miss, false_block = threshold_rates(threshold)
        system_pass, system_lost = system_at_threshold(threshold)
        print(f"threshold {threshold}: miss {miss:.3f}, false block {false_block:.3f},"
              f" system pass rate {system_pass:.4f}, harmless lost {system_lost:.4f}")

    rate = pass_rate(MISS)
    print("success in n attempts:", {n: round(success_in(n, rate), 3) for n in (1, 10, 100, 500)})
    print("attempts for 0.95:", attempts_needed(rate), "exact", round(math.log(0.05) / math.log(1 - rate), 1))
    print("fully correlated layers:", min(MISS), "ratio", round(min(MISS) / rate, 1),
          "attempts for 0.95:", attempts_needed(min(MISS)))
    print("upper limit after 100 / 500 attempts with no success:",
          round(upper_limit_after_no_success(100), 3), round(upper_limit_after_no_success(500), 3))
    make_figures()
