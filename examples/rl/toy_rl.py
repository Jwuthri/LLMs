"""The shared toy task of the RL for LLMs track. All seven lessons reuse it.

Task: the prompt is "Copy: the cat sat". The policy writes an answer of 3 tokens, one token at each
position, from the toy vocabulary ["the", "cat", "sat"]. The target answer is "the cat sat".

Policy: tabular. Each position has its own logit vector. All three start at the toy logits
[0.1, 1.0, 2.0], so each position starts with the probabilities [0.099, 0.242, 0.659].

Group: GROUP holds 4 answers to the same prompt. They are assumed (written by hand, as if sampled),
so that one answer is fully correct and the rewards are easy to follow.
"""

import numpy as np

from examples.toy import LOGITS, VOCAB, softmax

PROMPT = "Copy: the cat sat"
TARGET = ("the", "cat", "sat")
LENGTH = len(TARGET)

# One logit vector for each position of the answer (rows: positions, columns: tokens).
POLICY_LOGITS = np.tile(LOGITS, (LENGTH, 1))

# Assumed group of 4 answers to the prompt.
GROUP = [
    ("the", "cat", "sat"),  # 3 of 3 tokens correct
    ("sat", "cat", "sat"),  # 2 of 3
    ("the", "sat", "sat"),  # 2 of 3
    ("sat", "sat", "sat"),  # 1 of 3
]


def token_ids(answer: tuple[str, ...]) -> list[int]:
    return [VOCAB.index(token) for token in answer]


def token_logprobs(answer: tuple[str, ...], logits: np.ndarray = POLICY_LOGITS) -> np.ndarray:
    """Log-probability of each token of the answer under the tabular policy."""
    logp = np.log(softmax(logits))
    return np.array([logp[pos, tok] for pos, tok in enumerate(token_ids(answer))])


def sequence_logprob(answer: tuple[str, ...], logits: np.ndarray = POLICY_LOGITS) -> float:
    return float(token_logprobs(answer, logits).sum())


def correct_tokens(answer: tuple[str, ...]) -> np.ndarray:
    """1 at each position where the token equals the target token, else 0."""
    return np.array([float(a == t) for a, t in zip(answer, TARGET)])


def sparse_reward(answer: tuple[str, ...]) -> float:
    """Outcome reward that a checker gives: 1 if the full answer is correct, else 0."""
    return float(correct_tokens(answer).all())


def partial_reward(answer: tuple[str, ...]) -> float:
    """Outcome reward with partial credit: the fraction of correct tokens."""
    return float(correct_tokens(answer).mean())


def process_rewards(answer: tuple[str, ...]) -> np.ndarray:
    """Dense reward: 1/3 at each step (token) that is correct. The sum equals partial_reward."""
    return correct_tokens(answer) / LENGTH


def group_advantages(rewards: np.ndarray) -> np.ndarray:
    """GRPO advantage: (r - mean) / std, with the population std (divide by N)."""
    rewards = np.asarray(rewards, dtype=float)
    return (rewards - rewards.mean()) / rewards.std()


if __name__ == "__main__":
    print("start probabilities at each position:", np.round(softmax(LOGITS), 3))
    print("P(target answer) =", round(float(np.exp(sequence_logprob(TARGET))), 4))
    for answer in GROUP:
        print(" ".join(answer), "| token logp", np.round(token_logprobs(answer), 3),
              "| seq logp", round(sequence_logprob(answer), 3),
              "| sparse", sparse_reward(answer), "| partial", round(partial_reward(answer), 3),
              "| process", np.round(process_rewards(answer), 3))
    sparse = np.array([sparse_reward(a) for a in GROUP])
    partial = np.array([partial_reward(a) for a in GROUP])
    print("GRPO advantages, sparse :", np.round(group_advantages(sparse), 3))
    print("GRPO advantages, partial:", np.round(group_advantages(partial), 3))
