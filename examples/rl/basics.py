"""Numbers and plots for content/rl/01-rl-basics.md.

Run: uv run python -m examples.rl.basics

The policy is the tabular policy of the shared toy (examples/rl/toy_rl.py): 3 positions, 3 tokens, one
logit vector for each position. With only 3**3 = 27 possible answers, every expectation on the page is
computed exactly by enumeration, not estimated.
"""

from itertools import product

import numpy as np

from examples.plotstyle import save_figure
from examples.rl.toy_rl import (
    GROUP, LENGTH, POLICY_LOGITS, TARGET, partial_reward, sequence_logprob, sparse_reward, token_ids,
)
from examples.toy import VOCAB, softmax

PAGE = "rl-basics"

ALL_ANSWERS = list(product(VOCAB, repeat=LENGTH))  # the 27 possible answers
STEP_SIZE = 1.0     # assumed step size of the REINFORCE update (the same as in scientist/05, section 4)
GAMMA = 1.0         # discount of the toy: the episode has 3 steps and one reward at the end

# Simulated training run (section 4): assumed settings.
SIM_SEEDS = 200
SIM_STEPS = 30
SIM_GROUP = 4


# --- 1. Text generation as RL --------------------------------------------------------------

def answer_prob(answer: tuple[str, ...], logits: np.ndarray = POLICY_LOGITS) -> float:
    """pi(y): the product of the probabilities of the 3 tokens."""
    return float(np.exp(sequence_logprob(answer, logits)))


def target_token_probs(logits: np.ndarray = POLICY_LOGITS) -> np.ndarray:
    """Probability of the target token at each position."""
    probs = softmax(logits)
    return np.array([probs[pos, tok] for pos, tok in enumerate(token_ids(TARGET))])


# --- 2. Return and objective ---------------------------------------------------------------

def discounted_return(rewards: list[float], gamma: float = GAMMA) -> float:
    """R = sum_t gamma^t r_t (t starts at 0)."""
    return float(sum(gamma**t * r for t, r in enumerate(rewards)))


def step_rewards(answer: tuple[str, ...], reward_fn) -> list[float]:
    """An outcome reward: 0 after each token, the score of the checker after the last token."""
    return [0.0] * (LENGTH - 1) + [reward_fn(answer)]


def expected_reward(reward_fn, logits: np.ndarray = POLICY_LOGITS) -> float:
    """J = E_{y ~ pi}[R(y)], exact sum over the 27 answers."""
    return float(sum(answer_prob(y, logits) * reward_fn(y) for y in ALL_ANSWERS))


# --- 3. REINFORCE --------------------------------------------------------------------------

def score(answer: tuple[str, ...], logits: np.ndarray = POLICY_LOGITS) -> np.ndarray:
    """grad of log pi(y) for the logits: row t = onehot(y_t) - p_t. Shape (positions, tokens)."""
    probs = softmax(logits)
    onehots = np.eye(len(VOCAB))[token_ids(answer)]
    return onehots - probs


def exact_gradient(reward_fn, logits: np.ndarray = POLICY_LOGITS) -> np.ndarray:
    """grad J = E[R(y) * grad log pi(y)], exact sum over the 27 answers."""
    return sum(answer_prob(y, logits) * reward_fn(y) * score(y, logits) for y in ALL_ANSWERS)


def reinforce_gradient(answers, rewards, logits: np.ndarray = POLICY_LOGITS, baseline: float = 0.0) -> np.ndarray:
    """REINFORCE estimate from samples: mean over i of (R_i - b) * grad log pi(y_i)."""
    return sum((r - baseline) * score(y, logits) for y, r in zip(answers, rewards)) / len(answers)


def mean_score(logits: np.ndarray = POLICY_LOGITS) -> np.ndarray:
    """E[grad log pi(y)]. It is 0: this is why a baseline adds no bias."""
    return sum(answer_prob(y, logits) * score(y, logits) for y in ALL_ANSWERS)


# --- 4. Baselines and variance -------------------------------------------------------------

def estimator_variance(reward_fn, baseline: float, logits: np.ndarray = POLICY_LOGITS) -> float:
    """Total variance of the one-sample estimate g = (R - b) * grad log pi(y):
    E[||g||^2] - ||E[g]||^2 (the sum of the variances of the 9 coordinates)."""
    second = sum(answer_prob(y, logits) * (reward_fn(y) - baseline) ** 2 * np.sum(score(y, logits) ** 2)
                 for y in ALL_ANSWERS)
    return float(second - np.sum(exact_gradient(reward_fn, logits) ** 2))


def optimal_baseline(reward_fn, logits: np.ndarray = POLICY_LOGITS) -> float:
    """The b that makes the total variance smallest: E[R ||s||^2] / E[||s||^2]."""
    weights = [answer_prob(y, logits) * np.sum(score(y, logits) ** 2) for y in ALL_ANSWERS]
    return float(sum(w * reward_fn(y) for w, y in zip(weights, ALL_ANSWERS)) / sum(weights))


def simulate(use_baseline: bool, seeds: int = SIM_SEEDS, steps: int = SIM_STEPS, group: int = SIM_GROUP,
             step_size: float = STEP_SIZE) -> np.ndarray:
    """REINFORCE on the toy with the partial reward. Each step samples `group` answers from pi.
    Baseline: none (b = 0) or the exact E[R] of the current policy (an oracle that only a toy can compute).
    Returns E[partial] after each step, one row for each seed: shape (seeds, steps + 1)."""
    curves = np.zeros((seeds, steps + 1))
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        logits = POLICY_LOGITS.copy()
        curves[seed, 0] = expected_reward(partial_reward, logits)
        for step in range(1, steps + 1):
            probs = softmax(logits)
            answers = [tuple(VOCAB[rng.choice(3, p=probs[pos])] for pos in range(LENGTH)) for _ in range(group)]
            rewards = [partial_reward(y) for y in answers]
            b = curves[seed, step - 1] if use_baseline else 0.0
            logits = logits + step_size * reinforce_gradient(answers, rewards, logits, baseline=b)
            curves[seed, step] = expected_reward(partial_reward, logits)
    return curves


# --- Numbers of the page -------------------------------------------------------------------

def page_numbers() -> dict:
    partial = [partial_reward(y) for y in GROUP]
    group_mean = float(np.mean(partial))
    grad_b0 = reinforce_gradient(GROUP, partial)
    new_logits = POLICY_LOGITS + STEP_SIZE * grad_b0
    grad_mean_b = reinforce_gradient(GROUP, partial, baseline=group_mean)
    new_logits_mean_b = POLICY_LOGITS + STEP_SIZE * grad_mean_b
    e_partial = expected_reward(partial_reward)
    return {
        "p_target_tokens": target_token_probs(),
        "p_target": answer_prob(TARGET),
        "n_answers": len(ALL_ANSWERS),
        "sum_probs": sum(answer_prob(y) for y in ALL_ANSWERS),
        "return_target": discounted_return(step_rewards(TARGET, partial_reward)),
        "return_target_gamma09": discounted_return(step_rewards(TARGET, partial_reward), gamma=0.9),
        "e_sparse": expected_reward(sparse_reward),
        "e_partial": e_partial,
        "p_correct_count": [sum(answer_prob(y) for y in ALL_ANSWERS if partial_reward(y) == k / 3) for k in range(4)],
        "score_target": score(TARGET),
        "exact_grad_partial": exact_gradient(partial_reward),
        "exact_grad_sparse": exact_gradient(sparse_reward),
        "grad_b0": grad_b0,
        "new_logits": new_logits,
        "new_probs": softmax(new_logits),
        "new_p_target": answer_prob(TARGET, new_logits),
        "new_e_partial": expected_reward(partial_reward, new_logits),
        "new_e_sparse": expected_reward(sparse_reward, new_logits),
        "mean_score": mean_score(),
        "var_b0": estimator_variance(partial_reward, 0.0),
        "var_bmean": estimator_variance(partial_reward, e_partial),
        "b_opt": optimal_baseline(partial_reward),
        "var_bopt": estimator_variance(partial_reward, optimal_baseline(partial_reward)),
        "var_sparse_b0": estimator_variance(sparse_reward, 0.0),
        "var_sparse_bmean": estimator_variance(sparse_reward, expected_reward(sparse_reward)),
        "group_mean": group_mean,
        "grad_group_mean_b": grad_mean_b,
        "new_p_target_mean_b": answer_prob(TARGET, new_logits_mean_b),
        "new_e_partial_mean_b": expected_reward(partial_reward, new_logits_mean_b),
    }


def simulation_numbers() -> dict:
    out = {}
    for name, use_baseline in (("b0", False), ("bmean", True)):
        curves = simulate(use_baseline)
        final = curves[:, -1]
        out[name] = {"curves": curves, "mean": float(final.mean()), "std": float(final.std()),
                     "p10": float(np.percentile(final, 10))}
    return out


# --- Figures -------------------------------------------------------------------------------

def make_figures(nums: dict, sim: dict) -> None:
    baselines = np.linspace(-0.5, 1.2, 171)
    variances = [estimator_variance(partial_reward, b) for b in baselines]

    def draw_variance(ax, theme):
        ax.plot(baselines, variances, color=theme["series"][0], label="partial reward")
        # In this toy the best b equals E[R] (see the page), so one marker shows both.
        marks = [(0.0, nums["var_b0"], "b = 0"), (nums["e_partial"], nums["var_bmean"], "b = E[R]")]
        for (b, v, label), color in zip(marks, theme["series"][1:]):
            ax.plot([b], [v], "o", color=color, label=f"{label}: {v:.3f}")
        ax.set_xlabel("baseline b")
        ax.set_ylabel("variance of the one-sample gradient")
        ax.set_ylim(0, max(variances) * 1.05)
        ax.legend(loc="upper center")

    save_figure(draw_variance, PAGE, "variance-vs-baseline")

    steps = np.arange(SIM_STEPS + 1)

    def draw_training(ax, theme):
        for (key, label), color in zip((("b0", "no baseline (b = 0)"), ("bmean", "baseline b = E[R]")),
                                       theme["series"]):
            curves = sim[key]["curves"]
            ax.fill_between(steps, np.percentile(curves, 10, axis=0), np.percentile(curves, 90, axis=0),
                            color=color, alpha=0.18, linewidth=0)
            ax.plot(steps, curves.mean(axis=0), color=color, label=label)
        ax.set_xlabel("update step (4 sampled answers each)")
        ax.set_ylabel("E[partial reward]")
        ax.set_ylim(0.3, 1.0)
        ax.legend(loc="lower right")

    save_figure(draw_training, PAGE, "training-runs")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    nums = page_numbers()
    sim = simulation_numbers()
    for key, value in nums.items():
        print(f"{key}: {np.round(value, 4) if not isinstance(value, int) else value}")
    for key in ("b0", "bmean"):
        s = sim[key]
        print(f"simulation {key}: final E[partial] mean {s['mean']:.3f} std {s['std']:.3f} p10 {s['p10']:.3f}")
    make_figures(nums, sim)
