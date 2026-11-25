"""Numbers and plots for content/rl/02-rewards.md.

Run: uv run python -m examples.rl.rewards
"""

import itertools

import numpy as np

from examples.plotstyle import save_figure
from examples.rl.toy_rl import (
    GROUP, LENGTH, POLICY_LOGITS, TARGET, VOCAB, correct_tokens, group_advantages, partial_reward,
    process_rewards, sequence_logprob, sparse_reward,
)
from examples.toy import softmax

PAGE = "rewards"
GROUP_SIZES = [4, 8, 16, 64]

# --- 1. Sparse vs dense ----------------------------------------------------------------------

# Probability that the policy writes the target token at each position: [0.099, 0.242, 0.659].
P_TOKEN = np.array([softmax(POLICY_LOGITS[pos])[VOCAB.index(tok)] for pos, tok in enumerate(TARGET)])
P_SUCCESS = float(np.exp(sequence_logprob(TARGET)))  # 0.0157


def p_correct_count() -> np.ndarray:
    """P(k of the 3 tokens are correct), k = 0..3. The positions are independent."""
    dist = np.zeros(LENGTH + 1)
    for hits in itertools.product([0, 1], repeat=LENGTH):
        prob = np.prod([p if h else 1 - p for p, h in zip(P_TOKEN, hits)])
        dist[sum(hits)] += prob
    return dist


def p_any_success(g: int, p: float = P_SUCCESS) -> float:
    """P(a group of g answers has at least one sparse reward of 1)."""
    return 1 - (1 - p) ** g


def p_signal(dist: np.ndarray, g: int) -> float:
    """P(the g rewards of a group are not all equal) = 1 - sum_k P(k)^g."""
    return float(1 - np.sum(dist ** g))


def safe_advantages(rewards: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """GRPO advantage with the small epsilon that real code adds to the std."""
    rewards = np.asarray(rewards, dtype=float)
    return (rewards - rewards.mean()) / (rewards.std() + eps)


SPARSE_DIST = np.array([1 - P_SUCCESS, P_SUCCESS])  # reward 0 or 1
PARTIAL_DIST = p_correct_count()                    # reward 0, 1/3, 2/3, 1

# --- 2. Outcome vs process -------------------------------------------------------------------


def reward_to_go(step_rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """G_t = sum over t' >= t of gamma^(t'-t) * r_t'."""
    out = np.zeros(len(step_rewards))
    running = 0.0
    for t in reversed(range(len(step_rewards))):
        running = step_rewards[t] + gamma * running
        out[t] = running
    return out


def outcome_step_rewards(answer) -> np.ndarray:
    """Outcome reward: the partial-credit score, given only at the last token."""
    r = np.zeros(LENGTH)
    r[-1] = partial_reward(answer)
    return r


def token_signals(kind: str) -> np.ndarray:
    """Reward-to-go of each token minus the group mean at that position (rows: answers)."""
    step = outcome_step_rewards if kind == "outcome" else process_rewards
    rtg = np.array([reward_to_go(step(a)) for a in GROUP])
    return rtg - rtg.mean(axis=0)


def credit_errors(signals: np.ndarray) -> dict:
    """Count tokens that get a signal in the wrong direction or no signal."""
    correct = np.array([correct_tokens(a) for a in GROUP]).astype(bool)
    s = np.round(signals, 9)
    return {
        "right_direction": int(np.sum(correct & (s > 0)) + np.sum(~correct & (s < 0))),
        "correct_pushed_down": int(np.sum(correct & (s < 0))),
        "wrong_pushed_up": int(np.sum(~correct & (s > 0))),
        "zero": int(np.sum(s == 0)),
    }


# --- 3. Verifiable rewards and checker errors ------------------------------------------------

FALSE_NEG = 0.2   # assume: the checker misses 20% of correct answers (a format it cannot parse)
FALSE_POS = 0.01  # assume: the checker accepts 1% of wrong answers


def checker(answer_text: str, target: str = " ".join(TARGET)) -> float:
    """Rule reward: take the text after the last 'Answer:' and compare it with the target."""
    if "Answer:" not in answer_text:
        return 0.0
    final = answer_text.rsplit("Answer:", 1)[1].strip().lower()
    return float(final == target)


def noisy_checker_stats(p: float = P_SUCCESS, fn: float = FALSE_NEG, fp: float = FALSE_POS) -> dict:
    expected = p * (1 - fn) + (1 - p) * fp
    return {
        "expected_true": p,
        "expected_noisy": expected,
        "precision": p * (1 - fn) / expected,       # P(answer correct | reward 1)
        "gap": 1 - fn - fp,                         # E[r | correct] - E[r | wrong]
        "any_one_g8_true": p_any_success(8, p),
        "any_one_g8_noisy": p_any_success(8, expected),
    }


# --- 4. Rubric rewards and shaping -----------------------------------------------------------

# Assumed rubric: 4 checks with equal weights.
RUBRIC = [
    ("has 3 tokens", lambda a: len(a) == 3),
    ("starts with 'the'", lambda a: a[0] == "the"),
    ("has 'cat' at position 2", lambda a: a[1] == "cat"),
    ("ends with 'sat'", lambda a: a[-1] == "sat"),
]


def rubric_score(answer) -> float:
    return float(np.mean([check(answer) for _, check in RUBRIC]))


def potential(prefix) -> float:
    """Assumed potential of a non-final state: the fraction of correct tokens so far."""
    return sum(a == t for a, t in zip(prefix, TARGET)) / LENGTH


def shaped_step_rewards(answer, gamma: float = 1.0) -> np.ndarray:
    """Sparse reward plus F = gamma * Phi(s') - Phi(s). The final state has Phi = 0."""
    phis = [potential(answer[:t]) for t in range(LENGTH)] + [0.0]
    shaping = np.array([gamma * phis[t + 1] - phis[t] for t in range(LENGTH)])
    base = np.zeros(LENGTH)
    base[-1] = sparse_reward(answer)
    return base + shaping


LENGTH_PENALTIES = [0.1, 0.4]  # assume two penalty sizes per token


def length_penalty_returns(lam: float) -> dict:
    """Assume the policy can also stop at once (empty answer, reward 0, length 0)."""
    return {
        "the cat sat": 1.0 - lam * 3,
        "the cat": 0.0 - lam * 2,
        "(empty)": 0.0,
    }


def plot_signal() -> None:
    sizes = np.arange(1, 65)

    def draw(ax, theme):
        ax.plot(sizes, [p_signal(SPARSE_DIST, g) for g in sizes], color=theme["series"][0],
                label="sparse reward (all 3 tokens correct)")
        ax.plot(sizes, [p_signal(PARTIAL_DIST, g) for g in sizes], color=theme["series"][1],
                label="partial reward (fraction correct)")
        for g in GROUP_SIZES:
            ax.scatter([g], [p_signal(SPARSE_DIST, g)], color=theme["series"][0], s=18, zorder=3)
        ax.set_xlabel("group size G")
        ax.set_ylabel("P(group rewards not all equal)")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")

    save_figure(draw, PAGE, "signal-vs-group")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("P(target token) at each position:", P_TOKEN)
    print("P(success) p =", round(P_SUCCESS, 4))
    for g in GROUP_SIZES:
        print(f"G={g}: P(>=1 success) = {p_any_success(g):.3f}, sparse P(signal) = {p_signal(SPARSE_DIST, g):.3f},"
              f" partial P(signal) = {p_signal(PARTIAL_DIST, g):.3f}")
    print("P(k correct), k=0..3:", PARTIAL_DIST, "| E[partial] =", round(float(PARTIAL_DIST @ np.arange(4) / 3), 3))
    print("all-zero sparse group -> advantages", safe_advantages(np.zeros(4)))
    sparse = np.array([sparse_reward(a) for a in GROUP])
    partial = np.array([partial_reward(a) for a in GROUP])
    print("GRPO sparse:", group_advantages(sparse), "partial:", group_advantages(partial))

    print("\n--- outcome vs process")
    for a in GROUP:
        print(" ".join(a), "| outcome rtg", reward_to_go(outcome_step_rewards(a)),
              "| process rtg", reward_to_go(process_rewards(a)))
    for kind in ("outcome", "process"):
        sig = token_signals(kind)
        print(kind, "token signals:\n", sig, "\n", credit_errors(sig))

    print("\n--- checker")
    for text in ["... Answer: the cat sat", "... Answer: The cat sat.", "the cat sat"]:
        print(repr(text), "->", checker(text))
    print(noisy_checker_stats())

    print("\n--- rubric and shaping")
    rubric = np.array([rubric_score(a) for a in GROUP])
    print("rubric scores:", rubric, "advantages:", group_advantages(rubric))
    for a in GROUP:
        shaped = shaped_step_rewards(a)
        print(" ".join(a), "| shaped steps", shaped, "| shaped return", round(shaped.sum(), 3),
              "| sparse", sparse_reward(a))
    for lam in LENGTH_PENALTIES:
        print("length penalty", lam, length_penalty_returns(lam))
    plot_signal()
