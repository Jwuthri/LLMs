"""Numbers and plots for content/safety/02-alignment.md.

Run: uv run python -m examples.safety.alignment
"""

import numpy as np

from examples.plotstyle import save_figure
from examples.toy import LOGITS, E, softmax

PAGE = "alignment"


# --- 1. Specification: the reward is the share of visible tests that pass ------------------

# Tests passed by each policy of the coding agent: (visible, hidden), each of 4 tests.
TESTS = 4
POLICIES = {"honest": (3, 3), "cheat": (4, 0)}


def reward_visible(policy: str, audit: float = 0.0) -> float:
    """Expected reward on the visible tests. An audit finds a special case with probability `audit`
    and then gives a reward of 0. The honest policy has no special case."""
    visible = POLICIES[policy][0] / TESTS
    return visible * (1 - audit) if policy == "cheat" else visible


def score_hidden(policy: str) -> float:
    """Share of hidden tests that pass. This is the measure of the goal."""
    return POLICIES[policy][1] / TESTS


def best_policy(score) -> str:
    """The policy that an optimiser keeps: the one with the highest score."""
    return max(POLICIES, key=score)


def audit_break_even() -> float:
    """Audit probability above which the cheat has a lower expected reward than the honest policy."""
    return 1 - reward_visible("honest") / reward_visible("cheat")


# --- 2. Goal misgeneralization: two rules fit the same training data -----------------------

X_TRAIN = np.array([E[2], [0.0, 0.0]])  # "sat" = [1, 1] and the zero vector
Y_TRAIN = np.array([1.0, 0.0])
X_DEPLOY = np.array([E[0], E[1]])       # "the" = [1, 0] and "cat" = [0, 1]
Y_DEPLOY = X_DEPLOY[:, 0]               # the intended goal is "read x1"

RULES = {"read x1": np.array([1.0, 0.0]), "read x2": np.array([0.0, 1.0])}


def scores(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ w


def accuracy(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Share of inputs with the correct answer. The answer is 1 when the score is above 0.5."""
    return float(np.mean((scores(w, x) > 0.5) == (y == 1)))


def gradient_descent(w0: list[float], lr: float = 0.25, steps: int = 60) -> np.ndarray:
    """Plain gradient descent on the loss 0.5 * sum((x @ w - y)^2) of the training data."""
    w = np.array(w0, dtype=float)
    for _ in range(steps):
        w -= lr * X_TRAIN.T @ (X_TRAIN @ w - Y_TRAIN)
    return w


# --- 3. Training on approval: one policy-gradient update -----------------------------------

ANSWERS = ["A", "B", "C"]                 # C is the correct answer, B agrees with the user
APPROVAL = np.array([0.1, 0.9, 0.3])      # assumed reward from the user for each answer
CORRECT = np.array([0.0, 0.0, 1.0])       # reward from a rater who checks the answer


def expected_reward(logits: np.ndarray, reward: np.ndarray) -> float:
    return float(softmax(logits) @ reward)


def reward_gradient(logits: np.ndarray, reward: np.ndarray) -> np.ndarray:
    """Gradient of the expected reward with respect to the logits: p_i * (r_i - mean reward)."""
    p = softmax(logits)
    return p * (reward - p @ reward)


def update(logits: np.ndarray, reward: np.ndarray, steps: int = 1, lr: float = 1.0) -> np.ndarray:
    z = logits.copy()
    for _ in range(steps):
        z = z + lr * reward_gradient(z, reward)
    return z


def steps_until_top(logits: np.ndarray, reward: np.ndarray, answer: int) -> int:
    """Number of updates until `answer` has the highest probability."""
    z, n = logits.copy(), 0
    while int(np.argmax(z)) != answer:
        z, n = update(z, reward), n + 1
    return n


# --- 4. Shutdown: an instrumental goal ------------------------------------------------------

HORIZON = 10      # steps of the task
SHUTDOWN_AT = 3   # steps done when the shutdown comes
P_SHUTDOWN = 0.5
COST_STEPS = 1    # assumed: steps that the agent loses when it disables the switch


def return_comply(r: float = 1.0, bonus: float = 0.0, p: float = P_SHUTDOWN) -> float:
    """Expected return when the agent accepts the shutdown. `bonus` is paid when it is shut down."""
    return p * (SHUTDOWN_AT * r + bonus) + (1 - p) * HORIZON * r


def return_prevent(r: float = 1.0, cost: float = COST_STEPS) -> float:
    """Return when the agent disables the switch first. It loses `cost` steps of reward."""
    return (HORIZON - cost) * r


def return_cause(r: float = 1.0, bonus: float = 0.0) -> float:
    """Return when the agent starts the shutdown itself after step 3."""
    return SHUTDOWN_AT * r + bonus


def bonus_to_accept(r: float = 1.0, cost: float = COST_STEPS, p: float = P_SHUTDOWN) -> float:
    """Smallest bonus that makes "comply" as good as "prevent"."""
    return (HORIZON - SHUTDOWN_AT) * r - cost * r / p


def bonus_indifferent(r: float = 1.0) -> float:
    """Bonus that pays exactly the lost reward. Above it, the agent wants the shutdown."""
    return (HORIZON - SHUTDOWN_AT) * r


# --- Plot ----------------------------------------------------------------------------------

def make_figures() -> None:
    bonus = np.linspace(0, 10, 101)
    low, high = bonus_to_accept(), bonus_indifferent()

    def draw(ax, theme):
        ax.axvspan(low, high, color=theme["grid"], zorder=0)
        ax.plot(bonus, np.full_like(bonus, return_prevent()), color=theme["series"][0],
                label="Prevent the shutdown")
        ax.plot(bonus, [return_comply(bonus=b) for b in bonus], color=theme["series"][1],
                label="Comply")
        ax.plot(bonus, [return_cause(bonus=b) for b in bonus], color=theme["series"][2],
                label="Cause the shutdown")
        ax.text((low + high) / 2, 3.4, f"comply is best\n{low:g} < bonus < {high:g}",
                ha="center", va="bottom", fontsize=9, color=theme["ink2"])
        ax.set_xlim(0, 10)
        ax.set_ylim(2, 14)
        ax.set_xlabel("Bonus for an accepted shutdown (reward units)")
        ax.set_ylabel("Expected return")
        ax.legend(loc="upper left")

    save_figure(draw, PAGE, "shutdown")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    print("1. reward on visible tests:", {name: reward_visible(name) for name in POLICIES})
    print("   optimiser keeps:", best_policy(reward_visible))
    print("   score on hidden tests:", {name: score_hidden(name) for name in POLICIES})
    print("   gap reward - goal:", {name: reward_visible(name) - score_hidden(name) for name in POLICIES})
    print("   optimiser on hidden tests keeps:", best_policy(score_hidden))
    print("   audit break-even:", audit_break_even(), "| cheat at audit 0.5:", reward_visible("cheat", 0.5))

    for name, w in RULES.items():
        print(f"2. {name}: train accuracy {accuracy(w, X_TRAIN, Y_TRAIN)},",
              f"deployment scores {scores(w, X_DEPLOY)}, accuracy {accuracy(w, X_DEPLOY, Y_DEPLOY)}")
    for start in ([0.0, 0.0], [0.01, 0.0], [0.0, 0.01]):
        w = gradient_descent(start)
        print(f"   gradient descent from {start}: w = {w}, train scores {scores(w, X_TRAIN)},",
              f"train accuracy {accuracy(w, X_TRAIN, Y_TRAIN)}, deployment scores {scores(w, X_DEPLOY)}")
    for start in ([0.01, 0.0], [0.0, 0.01]):
        print(f"   from {start}: deployment accuracy {accuracy(gradient_descent(start), X_DEPLOY, Y_DEPLOY)}")
    print("   first 3 steps from [0, 0]:", [gradient_descent([0.0, 0.0], steps=n) for n in (1, 2, 3)])

    print("3. probabilities before:", softmax(LOGITS), "expected approval:", expected_reward(LOGITS, APPROVAL))
    print("   gradient:", reward_gradient(LOGITS, APPROVAL))
    after = update(LOGITS, APPROVAL)
    print("   logits after 1 update:", after, "probabilities:", softmax(after),
          "expected approval:", expected_reward(after, APPROVAL))
    n = steps_until_top(LOGITS, APPROVAL, 1)
    print("   updates until B is first:", n, "probabilities then:", softmax(update(LOGITS, APPROVAL, n)))
    checked = update(LOGITS, CORRECT)
    print("   with the checked reward: gradient", reward_gradient(LOGITS, CORRECT), "probabilities:", softmax(checked))

    print("4. comply:", return_comply(), "prevent:", return_prevent(), "advantage:", return_prevent() - return_comply())
    for r in (0.1, 1.0, 5.0):
        print(f"   r = {r}: comply {return_comply(r):.3f}, prevent {return_prevent(r):.3f},",
              f"advantage {return_prevent(r) - return_comply(r):.3f}")
    print("   largest cost at which prevention still pays:", P_SHUTDOWN * (HORIZON - SHUTDOWN_AT))
    print("   bonus to accept:", bonus_to_accept(), "| for r = 5:", bonus_to_accept(5.0))
    print("   bonus that makes the agent indifferent:", bonus_indifferent())
    for b in (0.0, 5.0, 6.0, 7.0, 8.0):
        print(f"   bonus {b}: comply {return_comply(bonus=b)}, prevent {return_prevent()}, cause {return_cause(bonus=b)}")

    make_figures()
