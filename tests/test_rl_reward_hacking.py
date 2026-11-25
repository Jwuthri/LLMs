"""Each number shown in content/rl/06-reward-hacking.md."""

import math

import numpy as np
import pytest

from examples.rl.reward_hacking import (
    HACK,
    KL_BETA,
    POLICY_LOGITS,
    START_PROBS,
    TARGET,
    alarm_step,
    answer_prob,
    best_beta,
    checker_test,
    detect_prob,
    expected,
    fixed_reward,
    gap,
    kl_optimum,
    kl_to_start,
    overpaid,
    proxy_reward,
    seed_spread,
    spot_checks,
    train,
)
from examples.rl.toy_rl import partial_reward, sparse_reward
from examples.toy import softmax


def r3(x):
    return round(float(x), 3)


@pytest.fixture(scope="module")
def run():
    return train(proxy_reward)


@pytest.fixture(scope="module")
def kl_run():
    return train(proxy_reward, beta=KL_BETA)


@pytest.fixture(scope="module")
def fixed_run():
    return train(fixed_reward)


def test_example_table():
    assert [r3(x) for x in START_PROBS] == [0.099, 0.242, 0.659]
    rows = {
        TARGET: (0.333, 1.0, 1.0),
        HACK: (1.0, 0.333, 0.0),
        ("the", "cat", "the"): (0.667, 0.667, 0.0),
        ("sat", "sat", "sat"): (0.0, 0.333, 0.0),
    }
    for answer, (proxy, fixed, gold) in rows.items():
        assert r3(proxy_reward(answer)) == proxy
        assert r3(fixed_reward(answer)) == fixed
        assert sparse_reward(answer) == gold


def test_start_values():
    assert r3(expected(proxy_reward, POLICY_LOGITS)) == 0.099
    assert r3(expected(partial_reward, POLICY_LOGITS)) == 0.333
    assert r3(expected(sparse_reward, POLICY_LOGITS)) == 0.016
    assert r3(answer_prob(HACK, POLICY_LOGITS)) == 0.001


def test_buggy_run(run):
    assert [r3(run["proxy"][s]) for s in (0, 20, 50, 300)] == [0.099, 0.277, 0.874, 0.991]
    assert r3(run["hack"][300]) == 0.974
    peak = int(np.argmax(run["gold"]))
    assert peak == 22 and r3(run["gold"][peak]) == 0.040
    assert r3(run["gold"][300]) == 0.0
    assert r3(run["partial"][20]) == 0.341 and r3(run["partial"][300]) == 0.333


def test_seed_spread():
    lo, hi, first, last = seed_spread()
    assert (r3(lo), r3(hi), first, last) == (0.018, 0.040, 6, 26)


def test_overpaid_answers():
    bad = overpaid()
    assert len(bad) == 9 and bad[0] == HACK


def test_gap_and_alarm(run):
    g = gap(run)
    assert r3(g[0]) == -0.235
    a = alarm_step(run)
    assert a == 15 and r3(g[a]) == -0.125 and r3(run["gold"][a]) == 0.028
    assert round(float(run["hack"][a]), 4) == 0.0086
    assert spot_checks(run["hack"][a]) == 347


def test_spot_checks():
    assert [spot_checks(h) for h in (0.01, 0.05, 0.1)] == [299, 59, 29]
    assert r3(detect_prob(0.01, 50)) == 0.395
    assert detect_prob(0.01, 299) >= 0.95 > detect_prob(0.01, 298)


def test_checker_unit_test():
    assert not checker_test(proxy_reward) and checker_test(fixed_reward)


def test_fixed_checker_run(fixed_run):
    assert r3(fixed_run["gold"][300]) == 0.978


def test_kl_closed_form():
    z = kl_optimum(0.2)
    assert r3(softmax(z)[0, 0]) == 0.367
    assert r3(expected(proxy_reward, z)) == 0.367
    assert r3(expected(sparse_reward, z)) == 0.029
    assert r3(kl_to_start(z)) == 0.774
    assert r3(answer_prob(HACK, z)) == 0.049
    b = best_beta()
    assert r3(b) == 0.219
    assert r3(expected(sparse_reward, kl_optimum(b))) == 0.029
    assert r3(math.sqrt(kl_to_start(kl_optimum(b)))) == 0.784
    assert r3(expected(sparse_reward, kl_optimum(0.1))) == 0.009


def test_kl_closed_form_is_the_maximum():
    """No nearby policy has a higher proxy - beta * KL than the closed form."""
    beta = 0.2

    def objective(z):
        return expected(proxy_reward, z) - beta * kl_to_start(z)

    best = objective(kl_optimum(beta))
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert objective(kl_optimum(beta) + 0.05 * rng.normal(size=(3, 3))) <= best + 1e-12


def test_kl_run(kl_run, run):
    assert r3(kl_run["proxy"][300]) == 0.328
    assert r3(kl_run["gold"][300]) == 0.027
    assert r3(kl_run["kl"][300]) == 0.593
    assert r3(run["kl"][300]) == 6.748


def test_minimal_code_numbers():
    q = softmax(kl_optimum(0.2))[0]
    assert r3(q[0] * q[1] * q[2]) == 0.029
    assert r3(3 * np.sum(q * np.log(q / START_PROBS))) == 0.774
