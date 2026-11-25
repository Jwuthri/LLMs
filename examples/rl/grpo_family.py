"""Numbers and plots for content/rl/04-grpo-family.md: GRPO, Dr. GRPO, DAPO, GSPO and CISPO on the shared toy.

Run: uv run python -m examples.rl.grpo_family
"""

import numpy as np

from examples.plotstyle import save_figure
from examples.rl.toy_rl import (
    GROUP,
    POLICY_LOGITS,
    TARGET,
    group_advantages,
    sparse_reward,
    token_ids,
    token_logprobs,
)
from examples.toy import VOCAB, softmax

PAGE = "grpo-family"

REWARDS = np.array([sparse_reward(answer) for answer in GROUP])   # [1, 0, 0, 0]
ADVANTAGES = group_advantages(REWARDS)                             # [1.732, -0.577, -0.577, -0.577]
STEP = 1.8            # assumed step size of the one gradient step on the GRPO loss
EPS = 0.2             # GRPO / PPO clip range
EPS_LOW, EPS_HIGH = 0.2, 0.28            # DAPO clip-higher (values from the DAPO paper)
GSPO_LOW, GSPO_HIGH = 3e-4, 4e-4         # GSPO sequence clip range (values from the GSPO paper)
CISPO_HIGH = 0.28     # assumed upper bound of the CISPO weight (the paper tunes it and gives no value)


# --- Part 1: GRPO ---------------------------------------------------------------------------

def grpo_gradient(group=GROUP, advantages=ADVANTAGES, logits=POLICY_LOGITS) -> np.ndarray:
    """Gradient of the GRPO objective at ratio 1 (first, on-policy pass), for the tabular policy.

    Each token gets weight A_i / (G * |o_i|). d log pi(tok) / d logits(pos) = onehot(tok) - p(pos).
    """
    probs = softmax(logits)
    grad = np.zeros_like(logits)
    for answer, advantage in zip(group, advantages):
        for pos, tok in enumerate(token_ids(answer)):
            grad[pos] += advantage / (len(group) * len(answer)) * (np.eye(len(VOCAB))[tok] - probs[pos])
    return grad


NEW_LOGITS = POLICY_LOGITS + STEP * grpo_gradient()


def token_ratios(answer, new_logits=NEW_LOGITS) -> np.ndarray:
    """r_t = pi_new(o_t) / pi_old(o_t) for each token of the answer."""
    return np.exp(token_logprobs(answer, new_logits) - token_logprobs(answer))


def clipped_term(ratio, advantage, eps_low=EPS, eps_high=EPS):
    """PPO-style term min(r A, clip(r, 1 - eps_low, 1 + eps_high) A) and its gradient weight.

    The gradient of r A is r A * grad log pi. When the clipped side is the minimum, the term is a
    constant, so the gradient weight is 0.
    """
    ratio = np.asarray(ratio, dtype=float)
    unclipped = ratio * advantage
    clipped = np.clip(ratio, 1 - eps_low, 1 + eps_high) * advantage
    term = np.minimum(unclipped, clipped)
    active = unclipped <= clipped        # the unclipped side is the minimum: the gradient flows
    return term, np.where(active, unclipped, 0.0)


def grpo_objective(group=GROUP, advantages=ADVANTAGES, new_logits=NEW_LOGITS) -> float:
    """(1/G) sum_i (1/|o_i|) sum_t min(...): mean over tokens of each answer, then mean over answers."""
    per_answer = [clipped_term(token_ratios(a, new_logits), adv)[0].mean() for a, adv in zip(group, advantages)]
    return float(np.mean(per_answer))


# --- Part 2: Dr. GRPO -----------------------------------------------------------------------

LENGTHS = np.array([3, 3, 3, 6])     # assume the 4th (wrong) answer is 6 tokens long: "sat" 6 times
MAX_TOKENS = 6                       # assumed constant normalizer of Dr. GRPO (the generation budget)


def grpo_token_weights(advantages=ADVANTAGES, lengths=LENGTHS) -> np.ndarray:
    """Weight of each token of answer i in GRPO: A_i / (G |o_i|)."""
    return advantages / (len(lengths) * lengths)


def dr_grpo_advantages(rewards) -> np.ndarray:
    """Dr. GRPO: r_i - mean(r), no division by the std."""
    rewards = np.asarray(rewards, dtype=float)
    return rewards - rewards.mean()


def dr_grpo_token_weights(rewards=REWARDS, lengths=LENGTHS, max_tokens=MAX_TOKENS) -> np.ndarray:
    """Weight of each token of answer i in Dr. GRPO: (r_i - mean) / (G * MAX_TOKENS)."""
    return dr_grpo_advantages(rewards) / (len(lengths) * max_tokens)


EASY_PROMPT = np.array([1, 1, 1, 1, 1, 1, 1, 0.0])     # assumed group of 8: 7 correct
MEDIUM_PROMPT = np.array([1, 1, 1, 1, 0, 0, 0, 0.0])   # assumed group of 8: 4 correct


# --- Part 3: DAPO ---------------------------------------------------------------------------

def max_new_probability(p_old: float, eps_high: float) -> float:
    """Highest pi_new before the upper clip stops the gradient (positive advantage)."""
    return p_old * (1 + eps_high)


def dapo_token_weights(advantages=ADVANTAGES, lengths=LENGTHS) -> np.ndarray:
    """Token-level loss: every token of the batch has weight A_i / sum_j |o_j|."""
    return advantages / lengths.sum()


def overlong_penalty(length: int, l_max: int, l_cache: int) -> float:
    """DAPO soft overlong punishment R_length(y), added to the correctness reward."""
    if length <= l_max - l_cache:
        return 0.0
    if length <= l_max:
        return ((l_max - l_cache) - length) / l_cache
    return -1.0


L_MAX, L_CACHE = 7, 2           # assumed toy values (the paper: 16,384 expected maximum + 4,096 cache)
ALL_WRONG = [("sat", "sat", "sat")] * 4   # assumed second group: all 4 answers wrong


def safe_advantages(rewards, eps=1e-4) -> np.ndarray:
    """(r - mean) / (std + eps), as libraries do, so an all-equal group gives 0 and not 0/0."""
    rewards = np.asarray(rewards, dtype=float)
    return (rewards - rewards.mean()) / (rewards.std() + eps)


# --- Part 4: GSPO and CISPO -----------------------------------------------------------------

def sequence_ratio(answer, new_logits=NEW_LOGITS) -> float:
    """GSPO: s_i = (pi_new(y_i) / pi_old(y_i))^(1/|y_i|) = exp(mean token log-ratio)."""
    return float(np.exp(np.mean(np.log(token_ratios(answer, new_logits)))))


def gspo_term(answer, advantage, new_logits=NEW_LOGITS):
    """min(s A, clip(s, 1 - 3e-4, 1 + 4e-4) A) and the gradient weight s A (0 when clipped)."""
    term, weight = clipped_term(sequence_ratio(answer, new_logits), advantage, GSPO_LOW, GSPO_HIGH)
    return float(term), float(weight)


def cispo_weights(ratios, advantage, eps_high=CISPO_HIGH) -> np.ndarray:
    """CISPO: gradient weight sg(clip(r, -, 1 + eps_high)) * A. No lower bound. Never 0 for A != 0."""
    return np.minimum(np.asarray(ratios, dtype=float), 1 + eps_high) * advantage


# --- Plots ----------------------------------------------------------------------------------

def plot_objective():
    ratio = np.linspace(0.5, 1.6, 441)
    a_pos, a_neg = ADVANTAGES[0], ADVANTAGES[1]

    def draw(ax, theme):
        s = theme["series"]
        ax.plot(ratio, clipped_term(ratio, a_pos)[0], color=s[0], label=f"GRPO, A = +{a_pos:.3f} (clip 0.8 / 1.2)")
        ax.plot(ratio, clipped_term(ratio, a_pos, EPS_LOW, EPS_HIGH)[0], color=s[1], linestyle="--",
                label=f"DAPO, A = +{a_pos:.3f} (clip 0.8 / 1.28)")
        ax.plot(ratio, clipped_term(ratio, a_neg)[0], color=s[2], label=f"GRPO and DAPO, A = {a_neg:.3f}")
        for r in token_ratios(TARGET)[:2]:
            ax.axvline(r, color=theme["muted"], linewidth=0.8, linestyle=":")
        ax.set_xlabel("ratio r = pi_new / pi_old of one token")
        ax.set_ylabel("per-token objective")
        ax.legend(fontsize=8, loc="upper left")

    save_figure(draw, PAGE, "objective-vs-ratio")


def plot_gradient_weight():
    ratio = np.linspace(0.5, 1.6, 441)
    a_pos = ADVANTAGES[0]

    def draw(ax, theme):
        s = theme["series"]
        ax.plot(ratio, clipped_term(ratio, a_pos)[1], color=s[0], label="GRPO (0 above 1.2)")
        ax.plot(ratio, clipped_term(ratio, a_pos, EPS_LOW, EPS_HIGH)[1], color=s[1], linestyle="--",
                label="DAPO (0 above 1.28)")
        ax.plot(ratio, cispo_weights(ratio, a_pos), color=s[2], linestyle="-.",
                label="CISPO (capped at 1.28 A, never 0)")
        r = token_ratios(TARGET)
        ax.scatter(r, cispo_weights(r, a_pos), color=theme["ink"], zorder=5, s=18)
        offsets = {"the": (6, -14), "cat": (-22, 4), "sat": (-6, 8)}
        for x, tok in zip(r, TARGET):
            ax.annotate(tok, (x, float(cispo_weights([x], a_pos)[0])), textcoords="offset points",
                        xytext=offsets[tok], fontsize=8, color=theme["ink2"])
        ax.set_xlabel(f"ratio r of one token (advantage A = +{a_pos:.3f})")
        ax.set_ylabel("gradient weight on grad log pi")
        ax.legend(fontsize=8, loc="upper left")

    save_figure(draw, PAGE, "gradient-weight")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    p_old = softmax(POLICY_LOGITS)
    print("--- 1. GRPO ---")
    print("advantages:", ADVANTAGES)
    print("gradient (rows = positions):\n", grpo_gradient())
    print("new probs:\n", softmax(NEW_LOGITS))
    for answer, adv in zip(GROUP, ADVANTAGES):
        r = token_ratios(answer)
        term, weight = clipped_term(r, adv)
        print(" ".join(answer), "| A", round(adv, 3), "| ratios", r, "| GRPO terms", term, "| grad w", weight,
              "| mean", round(term.mean(), 3))
    print("GRPO objective after the update:", round(grpo_objective(), 3),
          "| before:", round(grpo_objective(new_logits=POLICY_LOGITS), 3))

    print("--- 2. Dr. GRPO ---")
    print("lengths", LENGTHS)
    w = grpo_token_weights()
    print("GRPO token weights:", w, "| sum per answer:", w * LENGTHS)
    print("Dr. GRPO advantages:", dr_grpo_advantages(REWARDS))
    w = dr_grpo_token_weights()
    print("Dr. GRPO token weights:", w, "| sum per answer:", w * LENGTHS)
    for name, rewards in (("toy", REWARDS), ("easy", EASY_PROMPT), ("medium", MEDIUM_PROMPT)):
        print(name, "std", round(rewards.std(), 3), "1/std", round(1 / rewards.std(), 3),
              "GRPO A", group_advantages(rewards), "Dr A", dr_grpo_advantages(rewards))

    print("--- 3. DAPO ---")
    for tok, p in zip(VOCAB, p_old[0]):
        print(tok, "p_old", round(p, 3), "max 0.2:", round(max_new_probability(p, EPS), 3),
              "max 0.28:", round(max_new_probability(p, EPS_HIGH), 3), "min:", round(p * (1 - EPS_LOW), 3))
    w = dapo_token_weights()
    print("token-level weights:", w, "| sum per answer:", w * LENGTHS, "| answer share:", LENGTHS / LENGTHS.sum())
    print("sample-level answer share:", np.full(4, 0.25))
    print("overlong penalties:", {n: overlong_penalty(n, L_MAX, L_CACHE) for n in range(3, 9)})
    print("real: soft zone starts at", 16384, "max", 16384 + 4096,
          "| 18432 ->", overlong_penalty(18432, 20480, 4096))
    wrong = np.array([sparse_reward(a) for a in ALL_WRONG])
    print("all-wrong group advantages:", safe_advantages(wrong))
    print("toy token weight alone:", 1 / 12, "| with the all-wrong group in the batch:", 1 / 24)

    print("--- 4. GSPO and CISPO ---")
    for answer, adv in zip(GROUP, ADVANTAGES):
        r = token_ratios(answer)
        print(" ".join(answer), "| ratios", r, "| log r", np.log(r), "| s", round(sequence_ratio(answer), 4),
              "| GSPO term, w", np.round(gspo_term(answer, adv), 3),
              "| DAPO w", clipped_term(r, adv, EPS_LOW, EPS_HIGH)[1], "| CISPO w", cispo_weights(r, adv),
              "| CISPO r_hat", np.minimum(r, 1 + CISPO_HIGH))
    plot_objective()
    plot_gradient_weight()
    print("figures written to public/figures/grpo-family/")
