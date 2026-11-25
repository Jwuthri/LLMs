"""Numbers and plots for content/rl/03-ppo.md (actor-critic and PPO on the shared toy task).

Run: uv run python -m examples.rl.ppo
"""

import itertools

import numpy as np

from examples.plotstyle import save_figure
from examples.rl.toy_rl import (
    GROUP, LENGTH, POLICY_LOGITS, TARGET, partial_reward, token_ids,
)
from examples.toy import VOCAB, softmax

PAGE = "ppo"

ANSWER = ("sat", "cat", "sat")   # the answer that sub-parts 2 to 4 follow token by token
GREEDY = tuple(VOCAB[i] for i in POLICY_LOGITS.argmax(axis=1))  # ("sat", "sat", "sat")

# Assumed imperfect critic: it sees only the position, not the tokens. V(s0), V(s1), V(s2).
CRITIC_GUESS = np.array([0.40, 0.45, 0.50])

LAMBDAS = (0.0, 0.95, 1.0)
EPSILON = 0.2   # PPO clip range (Schulman et al., 2017)
STEP_SIZE = 3.0  # assumed learning rate of the one tabular update in sub-part 4


# --- 1. Baselines without a critic -------------------------------------------------------

def group_rewards() -> np.ndarray:
    return np.array([partial_reward(a) for a in GROUP])


def rloo_baselines(rewards: np.ndarray) -> np.ndarray:
    """Leave-one-out baseline: the mean of the other G - 1 rewards."""
    g = len(rewards)
    return (rewards.sum() - rewards) / (g - 1)


def rloo_advantages(rewards: np.ndarray) -> np.ndarray:
    return rewards - rloo_baselines(rewards)


def remax_advantages(rewards: np.ndarray) -> np.ndarray:
    """ReMax: the baseline is the reward of the greedy answer."""
    return rewards - partial_reward(GREEDY)


# --- 2. Value model and TD error ---------------------------------------------------------

def target_probs(logits: np.ndarray = POLICY_LOGITS) -> np.ndarray:
    """Probability of the target token at each position under the tabular policy."""
    probs = softmax(logits)
    return np.array([probs[pos, VOCAB.index(tok)] for pos, tok in enumerate(TARGET)])


def true_values(answer: tuple[str, ...], gamma: float = 1.0,
                logits: np.ndarray = POLICY_LOGITS) -> np.ndarray:
    """Exact V(s_t) for t = 0..T-1 under the current policy, partial reward paid at the last token.

    s_t is the prompt plus the first t tokens of `answer`. The expected reward is the known correct
    tokens plus the probability of a correct token at each later position, divided by T.
    """
    correct = np.array([float(a == t) for a, t in zip(answer, TARGET)])
    p_target = target_probs(logits)
    values = []
    for t in range(LENGTH):
        expected_reward = (correct[:t].sum() + p_target[t:].sum()) / LENGTH
        values.append(gamma ** (LENGTH - 1 - t) * expected_reward)
    return np.array(values)


def token_rewards(answer: tuple[str, ...]) -> np.ndarray:
    """Outcome reward on the last token, 0 on the other tokens."""
    rewards = np.zeros(LENGTH)
    rewards[-1] = partial_reward(answer)
    return rewards


def td_errors(rewards: np.ndarray, values: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """delta_t = r_t + gamma V(s_{t+1}) - V(s_t), with V = 0 after the last token."""
    next_values = np.append(values[1:], 0.0)
    return rewards + gamma * next_values - values


# --- 3. GAE ------------------------------------------------------------------------------

def gae(deltas: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """A_t = sum_l (gamma lambda)^l delta_{t+l}, computed backwards."""
    advantages = np.zeros_like(deltas)
    running = 0.0
    for t in reversed(range(len(deltas))):
        running = deltas[t] + gamma * lam * running
        advantages[t] = running
    return advantages


def answer_advantages(answer: tuple[str, ...], lam: float, values: np.ndarray | None = None,
                      gamma: float = 1.0) -> np.ndarray:
    values = true_values(answer, gamma) if values is None else values
    return gae(td_errors(token_rewards(answer), values, gamma), gamma, lam)


def first_token_stats(lam: float, critic: str) -> tuple[float, float]:
    """Credit gap and noise of the GAE advantage of the first token.

    For each first token, average the advantage A_0 over the 9 possible continuations, weighted by
    their probability under the policy. Return (mean A_0 after "the") - (mean A_0 after "sat"), the
    credit gap (true value 1/3), and the std of A_0 over the answers that start with "sat" (noise).
    critic="true" uses the exact values; critic="guess" uses CRITIC_GUESS, which sees only the position.
    """
    probs = softmax(POLICY_LOGITS)
    means, stds = {}, {}
    for first in ("the", "sat"):
        weights, estimates = [], []
        for rest in itertools.product(VOCAB, repeat=LENGTH - 1):
            answer = (first,) + rest
            ids = token_ids(answer)
            weights.append(probs[1, ids[1]] * probs[2, ids[2]])
            values = true_values(answer) if critic == "true" else CRITIC_GUESS
            estimates.append(answer_advantages(answer, lam, values)[0])
        weights, estimates = np.array(weights), np.array(estimates)
        means[first] = float((weights * estimates).sum())
        stds[first] = float(np.sqrt((weights * (estimates - means[first]) ** 2).sum()))
    return means["the"] - means["sat"], stds["sat"]


# --- 4. PPO update and clip --------------------------------------------------------------

def policy_update(lam: float = 0.95, step: float = STEP_SIZE) -> np.ndarray:
    """One gradient-ascent step on the mean token-level surrogate of the group (all ratios = 1).

    The gradient of A_t * log pi(a_t | s_t) with respect to the logits of position t is
    A_t * (one-hot(a_t) - p_t). Sum over the group, divide by the number of tokens.
    """
    probs = softmax(POLICY_LOGITS)
    grad = np.zeros_like(POLICY_LOGITS)
    for answer in GROUP:
        adv = answer_advantages(answer, lam)
        for pos, tok in enumerate(token_ids(answer)):
            grad[pos] += adv[pos] * (np.eye(len(VOCAB))[tok] - probs[pos])
    grad /= len(GROUP) * LENGTH
    return POLICY_LOGITS + step * grad


def token_ratios(answer: tuple[str, ...], new_logits: np.ndarray) -> np.ndarray:
    old, new = softmax(POLICY_LOGITS), softmax(new_logits)
    return np.array([new[pos, tok] / old[pos, tok] for pos, tok in enumerate(token_ids(answer))])


def clipped_terms(ratios: np.ndarray, advantages: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """PPO objective of each token: min(rho A, clip(rho, 1 - eps, 1 + eps) A)."""
    return np.minimum(ratios * advantages, np.clip(ratios, 1 - eps, 1 + eps) * advantages)


def value_loss(values: np.ndarray, returns: np.ndarray) -> float:
    """Mean squared error of the critic to the return targets."""
    return float(np.mean((values - returns) ** 2))


def plot_advantages() -> None:
    labels = [f"t={t}: {tok}" for t, tok in enumerate(ANSWER)]

    def draw(ax, theme):
        x = np.arange(LENGTH)
        width = 0.26
        for i, lam in enumerate(LAMBDAS):
            adv = answer_advantages(ANSWER, lam)
            bars = ax.bar(x + (i - 1) * width, adv, width, color=theme["series"][i], label=f"λ = {lam:g}")
            for bar, value in zip(bars, adv):
                y = value + (0.012 if value >= 0 else -0.03)
                ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.3f}", ha="center",
                        fontsize=8, color=theme["ink2"])
        ax.axhline(0, color=theme["axis"], linewidth=1)
        ax.set_xticks(x, labels)
        ax.set_ylabel("advantage A_t (γ = 1)")
        ax.set_xlabel('token of the answer "sat cat sat" (reward 0.667 at the end)')
        ax.set_ylim(-0.08, 0.44)
        ax.legend(loc="upper right")

    save_figure(draw, PAGE, "gae-advantages")


if __name__ == "__main__":
    r = group_rewards()
    print("1. partial rewards:", np.round(r, 3), "mean", round(r.mean(), 3))
    print("   RLOO baselines:", np.round(rloo_baselines(r), 3), "advantages:", np.round(rloo_advantages(r), 3))
    print("   G/(G-1) (r - mean):", np.round(len(r) / (len(r) - 1) * (r - r.mean()), 3))
    print("   greedy answer:", " ".join(GREEDY), "reward", round(partial_reward(GREEDY), 3),
          "ReMax advantages:", np.round(remax_advantages(r), 3))
    print("2. target probs:", np.round(target_probs(), 3))
    v = true_values(ANSWER)
    print("   true V(s0..s2) for", " ".join(ANSWER), np.round(v, 3), "token rewards", np.round(token_rewards(ANSWER), 3))
    print("   TD errors (true V):", np.round(td_errors(token_rewards(ANSWER), v), 3))
    print("   TD errors (critic guess", CRITIC_GUESS, "):", np.round(td_errors(token_rewards(ANSWER), CRITIC_GUESS), 3))
    print("3. GAE, gamma = 1")
    for lam in LAMBDAS:
        print(f"   lambda={lam:<4}  true V: {np.round(answer_advantages(ANSWER, lam), 3)}"
              f"  guess V: {np.round(answer_advantages(ANSWER, lam, CRITIC_GUESS), 3)}")
    for critic in ("true", "guess"):
        for lam in LAMBDAS:
            gap, std = first_token_stats(lam, critic)
            print(f"   first token, critic={critic:5} lambda={lam:<4}: gap the-sat {gap:.3f}, std after sat {std:.3f}")
    new_logits = policy_update()
    print("4. new logits:\n", np.round(new_logits, 3))
    print("   new probs:\n", np.round(softmax(new_logits), 3))
    old_probs, new_probs = softmax(POLICY_LOGITS), softmax(new_logits)
    for answer in (TARGET, ANSWER):
        ids = token_ids(answer)
        adv = answer_advantages(answer, 0.95)
        rho = token_ratios(answer, new_logits)
        print("  ", " ".join(answer), "| A (lambda 0.95)", np.round(adv, 3),
              "| old p", np.round(old_probs[range(LENGTH), ids], 3),
              "| new p", np.round(new_probs[range(LENGTH), ids], 3))
        print("      ratios", np.round(rho, 3), "| rho*A", np.round(rho * adv, 3),
              "| clipped terms", np.round(clipped_terms(rho, adv), 3))
    returns = np.full(LENGTH, partial_reward(ANSWER))
    print("   value loss, critic guess vs return 0.667:", round(value_loss(CRITIC_GUESS, returns), 3))
    plot_advantages()
