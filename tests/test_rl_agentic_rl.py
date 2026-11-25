"""Each number shown in content/rl/07-agentic-rl.md."""

import numpy as np

from examples.rl.agentic_rl import (
    EPISODE_ADVANTAGE,
    EPISODES,
    GROUP,
    GROUPS,
    SPREADS,
    episode_reward,
    episode_tokens,
    importance_weights,
    loss_mask,
    policy_loss,
    position1_gradient,
    rollout_lengths,
    staleness,
    step_budget,
    sync_idle_fraction,
    token_advantages,
    trajectory_advantages,
    turn_advantages,
    turn_returns,
    turn_rewards,
)
from examples.rl.toy_rl import POLICY_LOGITS


def r3(x) -> float:
    return round(float(x), 3)


def test_loss_mask_and_masked_loss():
    tokens = episode_tokens(*EPISODES[0])
    assert [t for t, _, _ in tokens] == ["sat", "cat", "sat", "<check>", "result:", "token", "1", "wrong",
                                         "the", "cat", "sat"]
    logp = np.array([lp for _, _, lp in tokens])
    assert np.allclose(np.round(logp, 3), [-0.417, -1.417, -0.417, -0.105, -1.6, -0.9, -2.3, -1.2,
                                           -2.317, -1.417, -0.417])
    mask = loss_mask(tokens)
    assert mask.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    assert int(mask.sum()) == 7 and len(mask) == 11 and r3(mask.mean()) == 0.636
    assert r3((mask * logp).sum()) == -6.508 and r3(((1 - mask) * logp).sum()) == -6.0
    assert EPISODE_ADVANTAGE == 1.0
    assert r3(policy_loss(logp, mask, 1.0)) == 0.93
    assert r3(policy_loss(logp, np.ones_like(mask), 1.0)) == 1.137
    assert POLICY_LOGITS.size == 9
    assert r3(1 / 7) == 0.143 and r3(1 / 11) == 0.091 and r3(4 / 11) == 0.364


def test_credit_over_turns():
    assert [episode_reward(e) for e in EPISODES] == [1, 1, 0, 0]
    assert np.allclose(trajectory_advantages(), [1, 1, -1, -1])
    rewards = np.array([turn_rewards(e) for e in EPISODES])
    assert np.allclose(np.round(rewards, 3), [[0.667, 0.333], [0.667, 0.333], [0.667, 0], [0.333, 0.333]])
    returns = np.array([turn_returns(r) for r in rewards])
    assert np.allclose(np.round(returns, 3), [[0.967, 0.333], [0.967, 0.333], [0.667, 0], [0.633, 0.333]])
    adv = turn_advantages()
    assert np.allclose(np.round(adv, 3), [[0.997, 0.577], [0.997, 0.577], [-0.892, -1.732], [-1.102, 0.577]])
    assert np.allclose(np.round(turn_advantages(0.0)[:, 0], 3), [0.577, 0.577, 0.577, -1.732])
    assert np.allclose(np.round(turn_advantages(1.0)[:, 0], 3), [1, 1, -1, -1])
    assert np.allclose(np.round(token_advantages(adv[3]), 3), [-1.102] * 4 + [0.577] * 3)


def test_sync_vs_async():
    lengths = rollout_lengths(GROUP)
    assert lengths.tolist() == [100, 147, 186, 227, 275, 335, 426, 628]
    assert lengths.sum() == 2324 and r3(lengths.mean()) == 290.5 and lengths.max() == 628
    assert GROUP * lengths.max() == 5024
    assert r3(sync_idle_fraction(lengths)) == 0.537
    assert r3(lengths.max() / lengths.mean()) == 2.162
    assert staleness(lengths) == 2
    idle = {s: [r3(sync_idle_fraction(rollout_lengths(g, s))) for g in GROUPS] for s in SPREADS}
    assert GROUPS[0] == 1 and GROUPS[-1] == 128
    assert idle[0.3][-1] == 0.529 and idle[0.6][-1] == 0.758 and idle[0.9][-1] == 0.864
    assert idle[0.6][1] == 0.277


def test_importance_weights():
    assert np.allclose(np.round(position1_gradient(), 3), [0.289, 0, -0.289])
    w = importance_weights([0, 1, 2, 4])
    assert r3(w[0]["p_now"]) == 0.099 and w[0]["w"] == 1.0
    assert r3(w[1]["p_now"]) == 0.152 and r3(w[1]["w"]) == 1.538 and r3(w[1]["w_trunc"]) == 1.538
    assert r3(w[2]["p_now"]) == 0.223 and r3(w[2]["w"]) == 2.261 and w[2]["w_trunc"] == 2.0
    assert r3(w[4]["p_now"]) == 0.41 and r3(w[4]["w"]) == 4.159 and w[4]["w_trunc"] == 2.0


def test_step_budget():
    b = {k: r3(v) for k, v in step_budget().items()}
    assert b["sync_gb"] == 14.0 and b["weight_sync_s"] == 0.56
    assert b["gen_sync_s"] == 18.84 and b["gen_async_s"] == 8.715 and b["tools_s"] == 4.0
    assert b["colocated_s"] == 33.4 and b["pipelined_s"] == 13.275 and b["pipelined_set_s"] == 26.55
