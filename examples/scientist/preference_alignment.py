"""Numbers and plots for docs/scientist/05-preference-alignment.md.

Run: uv run python -m examples.scientist.preference_alignment
"""

import numpy as np

from examples.plotstyle import save_light_dark
from examples.toy import LOGITS, VOCAB, softmax

PAGE = "preference-alignment"

CHOSEN = VOCAB.index("sat")    # the answer that we prefer after "the cat"
REJECTED = VOCAB.index("the")  # the answer that we do not want

# Frozen reference model: assumed logits of the model before preference alignment.
REF_LOGITS = np.array([0.5, 1.0, 1.5])
BETA = 0.5


def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


# --- Rejection sampling --------------------------------------------------------------------

# Assumed scorer: one quality score for each possible answer to the prompt "the cat".
SCORES = {"the": 0.1, "cat": 0.4, "sat": 0.9}
SAMPLE_SEED = 22  # picked so that the 4 samples contain each of the 3 tokens


def sample_answers(probs: np.ndarray, n: int = 4, seed: int = SAMPLE_SEED) -> list[str]:
    """Draw n answers (one token each) from the policy."""
    rng = np.random.default_rng(seed)
    return [VOCAB[i] for i in rng.choice(len(VOCAB), size=n, p=probs)]


def pick_pair(answers: list[str], scores: list[float]) -> tuple[str, str]:
    """The best answer is `chosen`, the worst answer is `rejected`."""
    return answers[int(np.argmax(scores))], answers[int(np.argmin(scores))]


# --- Direct Preference Optimization (DPO) --------------------------------------------------

def dpo_terms(logits: np.ndarray, ref_logits: np.ndarray = REF_LOGITS, beta: float = BETA) -> dict[str, float]:
    """Each intermediate number of the DPO loss for the pair (chosen = sat, rejected = the)."""
    logp, ref_logp = np.log(softmax(logits)), np.log(softmax(ref_logits))
    ratio_chosen = float(logp[CHOSEN] - ref_logp[CHOSEN])
    ratio_rejected = float(logp[REJECTED] - ref_logp[REJECTED])
    margin = ratio_chosen - ratio_rejected
    return {
        "logp_chosen": float(logp[CHOSEN]), "ref_logp_chosen": float(ref_logp[CHOSEN]),
        "logp_rejected": float(logp[REJECTED]), "ref_logp_rejected": float(ref_logp[REJECTED]),
        "ratio_chosen": ratio_chosen, "ratio_rejected": ratio_rejected,
        "margin": margin, "sigmoid": sigmoid(beta * margin), "loss": dpo_loss(margin, beta),
    }


def dpo_loss(margin: float, beta: float = BETA) -> float:
    return float(-np.log(sigmoid(beta * margin)))


def dpo_grad(logits: np.ndarray, ref_logits: np.ndarray = REF_LOGITS, beta: float = BETA) -> np.ndarray:
    """Gradient of the DPO loss for the policy logits: -(1 - sigmoid) * beta * (onehot_chosen - onehot_rejected)."""
    weight = 1 - dpo_terms(logits, ref_logits, beta)["sigmoid"]
    direction = np.zeros(len(VOCAB))
    direction[CHOSEN], direction[REJECTED] = 1.0, -1.0
    return -weight * beta * direction


def dpo_step(logits: np.ndarray, lr: float = 1.0) -> np.ndarray:
    return logits - lr * dpo_grad(logits)


# --- Reward model (Bradley-Terry loss) -----------------------------------------------------

# Assumed scores of a reward model that is not trained yet.
REWARD_CHOSEN, REWARD_REJECTED = 1.2, 0.4


def bradley_terry(r_chosen: float, r_rejected: float) -> tuple[float, float]:
    """Probability that `chosen` wins, and the loss -log of that probability."""
    p_win = sigmoid(r_chosen - r_rejected)
    return p_win, float(-np.log(p_win))


def reward_step(r_chosen: float, r_rejected: float, lr: float = 1.0) -> tuple[float, float]:
    """One gradient step on the two scores: the gradient is -(1 - p) for chosen, +(1 - p) for rejected."""
    p_win, _ = bradley_terry(r_chosen, r_rejected)
    return r_chosen + lr * (1 - p_win), r_rejected - lr * (1 - p_win)


# --- Reinforcement learning ----------------------------------------------------------------

def grpo_advantages(rewards: np.ndarray) -> np.ndarray:
    """Group-relative advantage: (reward - group mean) / group standard deviation (divide by N)."""
    return (rewards - rewards.mean()) / rewards.std()


def policy_gradient(probs: np.ndarray, actions: list[int], advantages: np.ndarray) -> np.ndarray:
    """Mean over the samples of advantage * d log pi(action) / d logits = advantage * (onehot - probs)."""
    grad = np.zeros(len(probs))
    for action, advantage in zip(actions, advantages):
        grad += advantage * (np.eye(len(probs))[action] - probs)
    return grad / len(actions)


def ppo_term(ratio: float, advantage: float, epsilon: float = 0.2) -> dict[str, float]:
    """The PPO clipped objective for one sample: min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A)."""
    clipped_ratio = float(np.clip(ratio, 1 - epsilon, 1 + epsilon))
    return {"ratio": ratio, "clipped_ratio": clipped_ratio, "unclipped": ratio * advantage,
            "clipped": clipped_ratio * advantage,
            "objective": min(ratio * advantage, clipped_ratio * advantage)}


# --- Plot ----------------------------------------------------------------------------------

def make_figures() -> None:
    margins = np.linspace(-3, 4, 141)
    before = dpo_terms(LOGITS)
    after = dpo_terms(dpo_step(LOGITS))
    # (margin, loss, label, text offset, horizontal alignment)
    points = [(0.0, dpo_loss(0.0), "start", (0.15, 0.1), "left"),
              (before["margin"], before["loss"], "now", (-0.15, -0.19), "right"),
              (after["margin"], after["loss"], "after 1 step", (-0.15, -0.27), "right")]

    def draw(ax, theme):
        for i, beta in enumerate((BETA, 0.1)):
            ax.plot(margins, [dpo_loss(m, beta) for m in margins], color=theme["series"][i],
                    label=f"beta = {beta}")
        for margin, value, label, (dx, dy), align in points:
            ax.plot(margin, value, "o", color=theme["series"][0], markersize=5)
            ax.annotate(f"{label}: {value:.3f}", xy=(margin, value), xytext=(margin + dx, value + dy),
                        ha=align, fontsize=8, color=theme["ink2"])
        ax.set_xlabel("Margin: log-ratio of chosen minus log-ratio of rejected")
        ax.set_ylabel("DPO loss")
        ax.set_ylim(0, 2)
        ax.legend(loc="upper right")

    save_light_dark(draw, PAGE, "dpo-loss")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    policy, reference = softmax(LOGITS), softmax(REF_LOGITS)
    print("policy probs:", policy, "reference probs:", reference)

    answers = sample_answers(policy)
    scores = [SCORES[a] for a in answers]
    print("samples:", answers, "scores:", scores, "-> (chosen, rejected):", pick_pair(answers, scores))

    print("DPO at the start (policy = reference):", dpo_terms(REF_LOGITS))
    print("DPO now:", dpo_terms(LOGITS))
    print("DPO gradient for the logits:", dpo_grad(LOGITS))
    new_logits = dpo_step(LOGITS)
    print("logits after 1 step:", new_logits, "probs:", softmax(new_logits))
    print("DPO after 1 step:", dpo_terms(new_logits))
    print("DPO now with beta = 0.1:", dpo_terms(LOGITS, beta=0.1))

    print("Bradley-Terry before:", bradley_terry(REWARD_CHOSEN, REWARD_REJECTED))
    r_c, r_r = reward_step(REWARD_CHOSEN, REWARD_REJECTED)
    print("reward scores after 1 step:", r_c, r_r, "->", bradley_terry(r_c, r_r))

    rewards = np.array(scores)
    advantages = grpo_advantages(rewards)
    print("rewards:", rewards, "mean:", rewards.mean(), "std:", rewards.std(), "advantages:", advantages)
    actions = [VOCAB.index(a) for a in answers]
    grad = policy_gradient(policy, actions, advantages)
    print("policy gradient:", grad)
    for lr in (1.0, 2.0):
        stepped = softmax(LOGITS + lr * grad)
        print(f"lr={lr}: logits {LOGITS + lr * grad} probs {stepped}")
    sat_advantage = float(advantages[actions.index(CHOSEN)])
    for lr in (1.0, 2.0):
        new_policy = softmax(LOGITS + lr * grad)
        print(f"PPO term for sat, lr={lr}:", ppo_term(float(new_policy[CHOSEN] / policy[CHOSEN]), sat_advantage))
    make_figures()
