"""Each number shown in content/rl/05-stable-training.md."""

import numpy as np

from examples.rl.stable_training import (
    COLLAPSED_LOGITS, KL_BETA, POLICY_LOGITS, entropy, kl_estimators, mismatch, rolling_mean, simulate,
    variant_summary,
)
from examples.toy import VOCAB, softmax

THE = VOCAB.index("the")


def r3(x) -> np.ndarray:
    return np.round(np.asarray(x, dtype=float), 3)


def test_kl_estimators_at_position_1():
    est = kl_estimators()
    assert r3(est["ref"]).tolist() == [0.099, 0.242, 0.659]
    assert r3(est["pi"]).tolist() == [0.547, 0.122, 0.331]
    assert r3(est["r"]).tolist() == [0.18, 1.988, 1.988]
    assert r3(est["k1"]).tolist() == [1.713, -0.687, -0.687]
    assert r3(est["k2"]).tolist() == [1.467, 0.236, 0.236]
    assert r3(est["k3"]).tolist() == [0.893, 0.301, 0.301]
    assert round(est["exact"], 3) == 0.625
    assert round(est["mean"]["k1"], 3) == 0.625 and round(est["mean"]["k3"], 3) == 0.625
    assert round(est["mean"]["k2"], 3) == 0.909
    assert round(est["std"]["k1"], 3) == 1.195
    assert round(est["std"]["k2"], 3) == 0.613
    assert round(est["std"]["k3"], 3) == 0.295
    assert (est["k3"] >= 0).all() and (est["k1"] < 0).any()


def test_kl_as_reward_penalty_and_as_loss_term():
    est = kl_estimators()
    assert KL_BETA == 0.04
    assert round(est["std"]["k3"] / est["std"]["k1"], 2) == 0.25
    assert round(-KL_BETA * est["k1"][THE], 3) == -0.069
    assert round(KL_BETA * est["k3"][THE], 3) == 0.036


def test_entropy_collapse_main_run():
    assert r3(entropy(softmax(POLICY_LOGITS))).tolist() == [0.847, 0.847, 0.847]
    run = simulate()
    assert r3(run["reward"][[0, 4, 9]]).tolist() == [0.292, 0.708, 1.0]
    assert r3(run["entropy_pos"][4]).tolist() == [1.044, 0.513, 0.177]
    assert r3(run["entropy"][[0, 4, 9, 19, 49, 99]]).tolist() == [0.797, 0.578, 0.295, 0.079, 0.045, 0.019]
    assert int(np.argmax(run["zero_adv"] > 0)) + 1 == 10
    assert round(run["zero_adv"][50:].mean(), 3) == 0.9
    assert round(run["reward"][50:].mean(), 3) == 0.995


def test_entropy_bonus_main_run():
    bonus = simulate(entropy_coef=0.1)
    assert round(bonus["entropy"][-1], 3) == 0.056
    assert round(bonus["reward"][50:].mean(), 3) == 0.973


def test_collapsed_position_variant():
    assert r3(softmax(COLLAPSED_LOGITS)[1]).tolist() == [0.052, 0.002, 0.946]
    assert round(softmax(COLLAPSED_LOGITS)[1][1], 4) == 0.0023
    runs = [variant_summary(), variant_summary(eps_high=0.28),
            variant_summary(entropy_coef=0.02), variant_summary(entropy_coef=0.1)]
    assert [v["solved"] for v in runs] == [35, 36, 40, 40]
    assert [round(v["entropy"], 3) for v in runs] == [0.03, 0.025, 0.018, 0.132]


def test_training_inference_mismatch():
    m = mismatch()
    assert round(m["pi"][1], 4) == 0.0023 and round(m["mu"][1], 4) == 0.0010
    assert r3(m["w"]).tolist() == [0.999, 2.337, 0.999]
    assert r3(m["w_tis"]).tolist() == [0.999, 2.0, 0.999]
    assert round(m["mean_w"], 3) == 1.0
    assert round(m["relative"]["no_correction"], 3) == 0.428
    assert round(m["relative"]["is"], 3) == 1.0
    assert round(m["relative"]["tis"], 3) == 0.856
    assert round(1 - m["relative"]["no_correction"], 2) == 0.57 and round(1 - m["relative"]["tis"], 2) == 0.14
    assert abs(m["pi"] - m["mu"])[[0, 2]].max() < 0.002


def test_dashboard_signals():
    run = simulate()
    assert r3(run["kl"][[0, 4, 9, 99]]).tolist() == [0.075, 1.37, 3.165, 4.094]
    assert round(run["clip_frac"].max(), 3) == 0.375
    assert r3(run["clip_frac"][[0, 4]]).tolist() == [0.125, 0.333]
    assert r3(run["grad_norm"][[0, 4, 9]]).tolist() == [0.188, 0.192, 0.0]
    assert r3(rolling_mean(np.array([0.0, 1.0, 1.0]), 2)).tolist() == [0.0, 0.5, 1.0]
