"""Numbers and plots for content/safety/06-control.md.

Run: uv run python -m examples.safety.control
"""

from statistics import NormalDist

import numpy as np

from examples.plotstyle import save_figure

PAGE = "control"
STANDARD = NormalDist()  # suspicion scores of normal outputs: mean 0, standard deviation 1

# --- Part 1: trusted monitoring and an audit budget ------------------------------------------

# Suspicion score that the trusted monitor gives to the output of each task (assumed). Task 1 is first.
SUSPICION = [1.2, 0.8, 2.5, 3.1, 0.4, 8.1, 1.9, 7.4, 2.2, 4.0]
BACKDOOR_TASK = 8  # the output of this task has a hidden backdoor

AUDIT_BUDGET = 0.02  # share of the outputs that the developers can audit (assumed)
SEPARATION = 2.0  # mean suspicion score of an attack, in standard deviations above a normal output (assumed)


def audited_tasks(scores: list[float], budget: float) -> list[int]:
    """Task numbers (1 = first) of the most suspect outputs that fit in the audit budget."""
    count = round(budget * len(scores))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [i + 1 for i in order[:count]]


def top_share_threshold(share: float) -> float:
    """Suspicion score that only `share` of the normal outputs exceed."""
    return STANDARD.inv_cdf(1 - share)


def catch_rate(budget: float, separation: float) -> float:
    """Probability that the score of one attack is above the audit threshold."""
    return 1 - STANDARD.cdf(top_share_threshold(budget) - separation)


def monitor_auc(separation: float) -> float:
    """Share of (attack, normal) pairs in which the attack has the higher score."""
    return STANDARD.cdf(separation / 2 ** 0.5)


# --- Part 2: the whole game ------------------------------------------------------------------

ATTACK_WORKS = 0.5  # share of the attacks that cause harm when the protocol lets them through (assumed)
SOLVED_UNTRUSTED = 0.9  # share of the tasks that the untrusted model solves (assumed)
SOLVED_TRUSTED = 0.5  # share of the tasks that the trusted model solves (assumed)
DEFER = 0.10  # "defer to trusted": share of the outputs that the trusted model replaces (assumed)


def attack_split(budget: float, separation: float, defer: float = 0.0) -> tuple[float, float, float]:
    """Where one attack goes: (audited, replaced by the trusted model, used)."""
    audited = catch_rate(budget, separation)
    if defer >= 1:
        return audited, 1 - audited, 0.0
    lowest = top_share_threshold(max(budget, defer))  # an output above this score is not used as it is
    used = STANDARD.cdf(lowest - separation)
    return audited, 1 - audited - used, used


def success_rate(budget: float, separation: float, defer: float = 0.0) -> float:
    """Probability that one attack is used and causes harm."""
    return attack_split(budget, separation, defer)[2] * ATTACK_WORKS


def safety(catch: float, success: float) -> float:
    """Probability that a catch comes before a success, when each attack ends in one of 3 outcomes."""
    return catch / (catch + success)


def protocol_safety(budget: float, separation: float, defer: float = 0.0) -> float:
    return safety(catch_rate(budget, separation), success_rate(budget, separation, defer))


def usefulness(defer: float) -> float:
    """Share of the tasks that the protocol solves when the untrusted model is honest."""
    return (1 - defer) * SOLVED_UNTRUSTED + defer * SOLVED_TRUSTED


def simulate_safety(catch: float, success: float, games: int, seed: int) -> float:
    """Play the game many times: attack again and again until a catch or a success."""
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(games):
        while True:
            draw = rng.random()
            if draw < catch:
                wins += 1
                break
            if draw < catch + success:
                break
    return wins / games


# --- Part 3: monitoring the chain of thought -------------------------------------------------

CATCH_ACTION = 0.4  # catch rate of a monitor that reads only the actions (assumed)
CATCH_COT = 0.8  # catch rate of a monitor that reads the chain of thought (assumed)
HIDDEN = 0.5  # share of the attacks in which the model does not write its intent (assumed)


def either_catches(first: float, second: float) -> float:
    """Catch rate of two monitors together, if they fail independently."""
    return 1 - (1 - first) * (1 - second)


def cot_catch_with_hiding(catch: float, hidden: float) -> float:
    """The chain-of-thought monitor catches nothing when the intent is not written."""
    return catch * (1 - hidden)


def safety_without_defer(catch: float) -> float:
    """Safety of the game of part 2 for a catch rate, with no deferral."""
    return safety(catch, (1 - catch) * ATTACK_WORKS)


# --- Part 4: scalable oversight --------------------------------------------------------------

ROLE_ADVANTAGE = 400  # Elo points that the rules of the oversight game give to the guard (assumed)


def guard_win(gap: float, advantage: float = 0.0) -> float:
    """Elo formula: win probability of a guard that is `gap` Elo points below the other player."""
    return 1 / (1 + 10 ** ((gap - advantage) / 400))


def chain_win(total_gap: float, steps: int, advantage: float = 0.0) -> float:
    """Nested oversight: `steps` equal steps, and the guard must win each of them."""
    return guard_win(total_gap / steps, advantage) ** steps


# --- Plot ------------------------------------------------------------------------------------

def make_figures() -> None:
    shares = np.concatenate(([0.0], np.linspace(AUDIT_BUDGET, 1.0, 197)))
    labelled = {0.0: "no deferral", 0.1: "defer 10%", 0.2: "defer 20%", 0.5: "defer 50%", 1.0: "trusted only"}

    def draw(ax, theme):
        for i, separation in enumerate((2.0, 3.0)):
            ax.plot([usefulness(f) for f in shares],
                    [protocol_safety(AUDIT_BUDGET, separation, f) for f in shares],
                    color=theme["series"][i], label=f"Monitor with a separation of {separation:g}")
            ax.scatter([usefulness(f) for f in labelled],
                       [protocol_safety(AUDIT_BUDGET, separation, f) for f in labelled],
                       color=theme["series"][i], s=22, zorder=3)
        for share, text in labelled.items():
            x, y = usefulness(share), protocol_safety(AUDIT_BUDGET, SEPARATION, share)
            ax.annotate(f"{text}: {y:.2f}", (x, y), textcoords="offset points",
                        xytext=(-8, -3) if share >= 1.0 else (-8, -12), ha="right",
                        fontsize=8, color=theme["ink2"])
        ax.set_xlim(0.38, 0.93)
        ax.set_ylim(0.38, 1.03)
        ax.set_xlabel("Usefulness (share of the tasks solved)")
        ax.set_ylabel("Safety (probability that the developers win)")
        ax.legend(loc="lower left")

    save_figure(draw, PAGE, "safety-usefulness")


if __name__ == "__main__":
    print("audited at 20% / 10%:", audited_tasks(SUSPICION, 0.2), audited_tasks(SUSPICION, 0.1),
          "backdoor in task", BACKDOOR_TASK)
    print("audit threshold at 2%:", round(top_share_threshold(AUDIT_BUDGET), 3))
    print("catch rate by separation:", {m: round(catch_rate(AUDIT_BUDGET, m), 3) for m in (1.0, 2.0, 3.0)})
    print("catch rate at separation 2 by budget:", {b: round(catch_rate(b, SEPARATION), 3) for b in (0.01, 0.02, 0.05)})
    print("AUC at separation 2:", round(monitor_auc(SEPARATION), 3))
    for m in (1.0, 2.0, 3.0):
        print(f"separation {m:g}: c {catch_rate(AUDIT_BUDGET, m):.3f}, s {success_rate(AUDIT_BUDGET, m):.3f},"
              f" safety {protocol_safety(AUDIT_BUDGET, m):.3f}")
    print("simulated safety:", simulate_safety(catch_rate(AUDIT_BUDGET, SEPARATION),
                                               success_rate(AUDIT_BUDGET, SEPARATION), 100_000, seed=0))
    print("defer threshold at 10%:", round(top_share_threshold(DEFER), 3))
    print("attack split (audited, replaced, used):", [round(x, 3) for x in attack_split(AUDIT_BUDGET, SEPARATION, DEFER)])
    for f in (0.0, 0.1, 0.2, 0.5, 1.0):
        print(f"defer {f:g}: s {success_rate(AUDIT_BUDGET, SEPARATION, f):.3f},"
              f" safety {protocol_safety(AUDIT_BUDGET, SEPARATION, f):.3f}, usefulness {usefulness(f):.3f}")
    both = either_catches(CATCH_ACTION, CATCH_COT)
    hidden_cot = cot_catch_with_hiding(CATCH_COT, HIDDEN)
    both_hidden = either_catches(CATCH_ACTION, hidden_cot)
    print("two monitors:", round(both, 3), "| CoT monitor with hiding:", round(hidden_cot, 3),
          "| two monitors with hiding:", round(both_hidden, 3),
          "| all hidden:", round(either_catches(CATCH_ACTION, cot_catch_with_hiding(CATCH_COT, 1.0)), 3))
    print("safety at these catch rates:", {round(c, 2): round(safety_without_defer(c), 3)
                                           for c in (CATCH_ACTION, CATCH_COT, both, both_hidden)})
    print("guard win at gaps:", {gap: round(guard_win(gap), 3) for gap in (0, 200, 400, 800)})
    print("chain over a gap of 800:", {n: round(chain_win(800, n), 4) for n in (1, 2, 4)})
    print("chain with the role advantage:", {n: round(chain_win(800, n, ROLE_ADVANTAGE), 3) for n in (1, 2, 4, 8, 16)})
    print("one step of a chain of 4:", round(guard_win(200, ROLE_ADVANTAGE), 3))
    make_figures()
