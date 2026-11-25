"""Numbers and plots for content/rl/07-agentic-rl.md.

Run: uv run python -m examples.rl.agentic_rl

The toy task of the track becomes a 2-turn episode: the model answers, calls the tool `check`, reads the
feedback of the tool, and answers again. Each value that is not computed here is an assumption.
"""

import math
from statistics import NormalDist

import numpy as np

from examples.plotstyle import save_figure
from examples.rl.toy_rl import GROUP as TRACK_GROUP
from examples.rl.toy_rl import (
    POLICY_LOGITS,
    group_advantages,
    partial_reward,
    sparse_reward,
    token_ids,
    token_logprobs,
)
from examples.toy import VOCAB, softmax

PAGE = "agentic-rl"


# --- 1. One multi-turn episode and its loss mask ----------------------------------------------

CALL_PROB = 0.9  # assume: the model writes the call token "<check>" with probability 0.9
TOOL_OUTPUT = ("result:", "token", "1", "wrong")  # assume: 4 tokens of feedback from the tool
TOOL_OUTPUT_LOGPROBS = np.array([-1.6, -0.9, -2.3, -1.2])  # assume: log-probs the model gives them
EPISODE_ADVANTAGE = 1.0  # assume for part 1; part 2 shows it is the group advantage of this episode


def episode_tokens(turn1: tuple[str, ...], turn2: tuple[str, ...]) -> list[tuple[str, str, float]]:
    """(token, role, log-prob) for each token after the prompt. Role "model" or "tool"."""
    tokens = [(t, "model", float(lp)) for t, lp in zip(turn1, token_logprobs(turn1))]
    tokens.append(("<check>", "model", math.log(CALL_PROB)))
    tokens += [(t, "tool", float(lp)) for t, lp in zip(TOOL_OUTPUT, TOOL_OUTPUT_LOGPROBS)]
    tokens += [(t, "model", float(lp)) for t, lp in zip(turn2, token_logprobs(turn2))]
    return tokens


def loss_mask(tokens: list[tuple[str, str, float]]) -> np.ndarray:
    """1 for a token that the model wrote, 0 for a token that the environment wrote."""
    return np.array([1.0 if role == "model" else 0.0 for _, role, _ in tokens])


def policy_loss(logprobs: np.ndarray, mask: np.ndarray, advantage: float) -> float:
    """Policy-gradient loss -A * mean(log pi) over the tokens where mask = 1."""
    return float(-advantage * (mask * logprobs).sum() / mask.sum())


# --- 2. Credit assignment over turns ----------------------------------------------------------

# Assumed group of 4 two-turn episodes (turn 1 answer, turn 2 answer after the feedback of `check`).
EPISODES = [
    (("sat", "cat", "sat"), ("the", "cat", "sat")),  # fixes token 1
    (("the", "sat", "sat"), ("the", "cat", "sat")),  # fixes token 2
    (("sat", "cat", "sat"), ("sat", "cat", "sat")),  # ignores the feedback
    (("sat", "sat", "sat"), ("sat", "cat", "sat")),  # fixes token 2, token 1 stays wrong
]
GAMMA = 0.9  # assume: discount over turns
TOKENS_PER_TURN = (4, 3)  # model tokens: turn 1 = 3 answer tokens + "<check>", turn 2 = 3 answer tokens


def episode_reward(episode) -> float:
    """One reward at the end: 1 if the final answer is correct (the sparse reward of the track)."""
    return sparse_reward(episode[1])


def turn_rewards(episode) -> np.ndarray:
    """Per-turn reward: the gain of the partial score at each turn. The sum is the final partial score."""
    s1, s2 = partial_reward(episode[0]), partial_reward(episode[1])
    return np.array([s1, s2 - s1])


def turn_returns(rewards: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Discounted return of each turn: G_t = r_t + gamma * G_(t+1)."""
    returns = np.zeros_like(rewards)
    future = 0.0
    for t in reversed(range(len(rewards))):
        future = rewards[t] + gamma * future
        returns[t] = future
    return returns


def trajectory_advantages() -> np.ndarray:
    return group_advantages(np.array([episode_reward(e) for e in EPISODES]))


def turn_advantages(gamma: float = GAMMA) -> np.ndarray:
    """Rows: episodes, columns: turns. Normalize the returns of one turn across the group."""
    returns = np.array([turn_returns(turn_rewards(e), gamma) for e in EPISODES])
    return np.stack([group_advantages(returns[:, t]) for t in range(returns.shape[1])], axis=1)


def token_advantages(per_turn: np.ndarray) -> np.ndarray:
    """Spread the advantage of each turn on the model tokens of that turn (7 tokens)."""
    return np.concatenate([np.full(n, a) for a, n in zip(per_turn, TOKENS_PER_TURN)])


# --- 3. Synchronous vs asynchronous rollouts --------------------------------------------------

MEDIAN_LENGTH = 250  # assume: median rollout length in tokens
SPREAD = 0.6  # assume: spread (sigma of the log-length)
GROUP = 8
TRUNCATE = 2.0  # assume: truncate the importance weight at 2


def rollout_lengths(group: int, sigma: float = SPREAD, median: float = MEDIAN_LENGTH) -> np.ndarray:
    """Deterministic log-normal lengths: the (i + 0.5) / group quantiles. No random seed needed."""
    z = np.array([NormalDist().inv_cdf((i + 0.5) / group) for i in range(group)])
    return np.round(median * np.exp(sigma * z))


def sync_idle_fraction(lengths: np.ndarray) -> float:
    """Each slot decodes one token per step. A synchronous batch ends with its longest rollout."""
    return float(1 - lengths.mean() / lengths.max())


def staleness(lengths: np.ndarray) -> int:
    """Async: G slots refill at once, so the trainer gets G rollouts (one update) every mean length.
    The longest rollout lives max / mean updates, so its first token is that many versions old."""
    return int(lengths.max() // lengths.mean())


# Assume each policy update adds the sparse GRPO gradient of the track group at position 1, step size 1.
def position1_gradient() -> np.ndarray:
    adv = group_advantages(np.array([sparse_reward(a) for a in TRACK_GROUP]))
    p = softmax(POLICY_LOGITS[0])
    grads = [a * (np.eye(3)[token_ids(ans)[0]] - p) for a, ans in zip(adv, TRACK_GROUP)]
    return np.mean(grads, axis=0)


def importance_weights(ks: list[int], token: str = "the") -> dict[int, dict[str, float]]:
    """pi_now / pi_behaviour for one token of position 1, after k updates, and its truncation."""
    g = position1_gradient()
    i = VOCAB.index(token)
    p_behaviour = softmax(POLICY_LOGITS[0])[i]
    out = {}
    for k in ks:
        p_now = softmax(POLICY_LOGITS[0] + k * g)[i]
        w = p_now / p_behaviour
        out[k] = {"p_now": float(p_now), "w": float(w), "w_trunc": float(min(w, TRUNCATE))}
    return out


# --- 4. Systems: one step of an agentic RL run ------------------------------------------------

PARAMS = 7e9  # assume: a 7B-parameter policy
BYTES_PER_PARAM = 2  # bf16
SYNC_BANDWIDTH = 25e9  # assume: 25 GB/s from the trainer to the rollout engine
DECODE_STEP_S = 0.03  # assume: 30 ms for one batched decode step
TRAIN_S = 10.0  # assume: 10 s for the trainer to do one update
SANDBOX_S = 0.5  # assume: one `check` call takes 0.5 s in a sandbox
SANDBOXES = 16  # assume: 16 sandboxes run in parallel
EPISODES_PER_STEP = 128  # assume: 16 prompts x 8 episodes


def step_budget() -> dict[str, float]:
    lengths = rollout_lengths(GROUP)
    sync_bytes = PARAMS * BYTES_PER_PARAM
    weight_sync = sync_bytes / SYNC_BANDWIDTH
    gen_sync = lengths.max() * DECODE_STEP_S
    gen_async = lengths.mean() * DECODE_STEP_S
    tools = math.ceil(EPISODES_PER_STEP / SANDBOXES) * SANDBOX_S
    colocated = gen_sync + tools + TRAIN_S + weight_sync  # one set of GPUs does each phase in turn
    # two sets of GPUs (assume each set has the speed of the colocated set): generation runs during training
    pipelined = max(gen_async + tools, TRAIN_S) + weight_sync
    return {
        "sync_gb": sync_bytes / 1e9, "weight_sync_s": weight_sync, "gen_sync_s": float(gen_sync),
        "gen_async_s": float(gen_async), "tools_s": tools, "colocated_s": float(colocated),
        "pipelined_s": float(pipelined), "pipelined_set_s": float(2 * pipelined),
    }


# --- Figures ---------------------------------------------------------------------------------

GROUPS = [1, 2, 4, 8, 16, 32, 64, 128]
SPREADS = [0.3, 0.6, 0.9]


def make_figures() -> None:
    lengths = rollout_lengths(GROUP)

    def timeline(ax, theme):
        top = lengths.max()
        for slot, n in enumerate(lengths):
            ax.barh(slot, n, color=theme["series"][1], height=0.6, label="generating" if slot == 0 else None)
            ax.barh(slot, top - n, left=n, color=theme["grid"], height=0.6, edgecolor=theme["axis"],
                    label="idle (waits for the longest)" if slot == 0 else None)
        ax.set_yticks(range(len(lengths)), [f"rollout {i + 1}" for i in range(len(lengths))])
        ax.set_xlabel("decode steps (one token per step)")
        ax.set_title(f"Synchronous batch of {GROUP}: {sync_idle_fraction(lengths):.0%} of slot time is idle",
                     fontsize=10, color=theme["ink"])
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)

    save_figure(timeline, PAGE, "sync-timeline")

    def idle_curve(ax, theme):
        for color, sigma in zip(theme["series"], SPREADS):
            idle = [sync_idle_fraction(rollout_lengths(g, sigma)) for g in GROUPS]
            ax.plot(GROUPS, idle, marker="o", color=color, label=f"spread {sigma}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(GROUPS, [str(g) for g in GROUPS])
        ax.set_ylim(0, 1)
        ax.set_xlabel("rollouts in one synchronous batch")
        ax.set_ylabel("idle fraction of slot time")
        ax.legend(loc="upper left", fontsize=9)

    save_figure(idle_curve, PAGE, "idle-vs-group")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    e1 = EPISODES[0]
    tokens = episode_tokens(*e1)
    mask = loss_mask(tokens)
    logp = np.array([lp for _, _, lp in tokens])
    print("1. episode tokens:", [t for t, _, _ in tokens])
    print("   log-probs:", logp)
    print("   mask:", mask, "| trained", int(mask.sum()), "of", len(mask), "=", round(mask.mean(), 3))
    print("   sum model logp", round(float((mask * logp).sum()), 3), "| sum tool logp",
          round(float(((1 - mask) * logp).sum()), 3))
    print("   masked loss", round(policy_loss(logp, mask, EPISODE_ADVANTAGE), 3),
          "| unmasked loss", round(policy_loss(logp, np.ones_like(mask), EPISODE_ADVANTAGE), 3))
    print("   weight per token: masked", round(1 / mask.sum(), 3), "unmasked", round(1 / len(mask), 3),
          "| share of the gradient on tool tokens", round(float((1 - mask).sum() / len(mask)), 3))

    print("2. final rewards:", [episode_reward(e) for e in EPISODES])
    print("   trajectory advantages:", trajectory_advantages())
    rewards = np.array([turn_rewards(e) for e in EPISODES])
    print("   turn rewards:\n", rewards)
    print("   returns (gamma 0.9):\n", np.array([turn_returns(r) for r in rewards]))
    print("   turn advantages (gamma 0.9):\n", turn_advantages())
    print("   turn advantages (gamma 0):\n", turn_advantages(0.0))
    print("   turn advantages (gamma 1):\n", turn_advantages(1.0))
    print("   episode 4 tokens, trajectory:", token_advantages(np.full(2, trajectory_advantages()[3])))
    print("   episode 4 tokens, per turn  :", token_advantages(turn_advantages()[3]))

    lengths = rollout_lengths(GROUP)
    print("3. lengths:", lengths, "| sum", lengths.sum(), "| mean", lengths.mean(), "| max", lengths.max())
    print("   sync slot time", GROUP * lengths.max(), "| idle", round(sync_idle_fraction(lengths), 3),
          "| async speedup", round(float(lengths.max() / lengths.mean()), 3), "| staleness k", staleness(lengths))
    for sigma in SPREADS:
        print(f"   spread {sigma}:", [round(sync_idle_fraction(rollout_lengths(g, sigma)), 3) for g in GROUPS])
    print("   position-1 gradient:", position1_gradient())
    for k, v in importance_weights([0, 1, 2, 4]).items():
        print(f"   k={k}:", {key: round(x, 3) for key, x in v.items()})

    print("4.", {k: round(v, 3) for k, v in step_budget().items()})
    make_figures()
