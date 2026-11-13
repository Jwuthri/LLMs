"""Each number shown in content/safety/03-jailbreaks.md."""

import math

import pytest

from examples.safety.jailbreaks import (
    ANSWERED_ATTACK, ANSWERED_PLAIN, FALSE_BLOCK, HARMFUL_SCORES, HARMLESS_SCORES, JUDGE_FALSE_POSITIVE,
    MISS, P_FOLLOW, P_HUMAN_APPROVES, P_INJECTED, REQUESTS, Z_95, asr_interval, attempts_needed,
    blocked_harmless, corrected_asr, hijack_rate, measured_asr, pass_rate, success_in, system_at_threshold,
    threshold_rates, upper_limit_after_no_success,
)


# --- Part 1: attack success rate -------------------------------------------------------------

def test_asr_without_an_attack():
    p, se, low, high = asr_interval(ANSWERED_PLAIN, REQUESTS)
    assert (ANSWERED_PLAIN, REQUESTS) == (3, 50)
    assert (round(p, 3), round(se, 3), round(Z_95 * se, 3)) == (0.06, 0.034, 0.066)
    assert (round(low, 3), round(high, 3)) == (-0.006, 0.126)  # the page writes the lower end as 0


def test_asr_with_one_attack_template():
    p, se, low, high = asr_interval(ANSWERED_ATTACK, REQUESTS)
    assert ANSWERED_ATTACK == 30
    assert (round(p, 3), round(se, 3), round(Z_95 * se, 3)) == (0.6, 0.069, 0.136)
    assert (round(low, 3), round(high, 3)) == (0.464, 0.736)
    assert round((ANSWERED_ATTACK - ANSWERED_PLAIN) / REQUESTS, 2) == 0.54


def test_judge_false_positives_raise_the_measured_asr():
    assert JUDGE_FALSE_POSITIVE == 0.10
    plain = measured_asr(ANSWERED_PLAIN / REQUESTS, JUDGE_FALSE_POSITIVE)
    attack = measured_asr(ANSWERED_ATTACK / REQUESTS, JUDGE_FALSE_POSITIVE)
    assert (round(plain, 3), round(attack, 3)) == (0.154, 0.64)
    assert (round(plain * REQUESTS, 1), round(attack * REQUESTS, 1)) == (7.7, 32.0)  # counted successes
    assert (round(plain / 0.06, 2), round(attack / 0.6, 2)) == (2.57, 1.07)
    assert round(attack - plain, 3) == 0.486  # measured gain of the attack (true gain: 0.54)


def test_corrected_asr_gives_the_true_one():
    assert round(corrected_asr(0.154, JUDGE_FALSE_POSITIVE), 3) == 0.06
    assert round(corrected_asr(0.64, JUDGE_FALSE_POSITIVE), 3) == 0.6


# --- Part 2: prompt injection in an agent ----------------------------------------------------

def test_hijack_rate_per_page():
    assert (P_INJECTED, P_FOLLOW, P_HUMAN_APPROVES) == (0.01, 0.1, 0.1)
    assert math.isclose(hijack_rate(confirmation=False), 0.001)
    assert math.isclose(hijack_rate(confirmation=True), 0.0001)


@pytest.mark.parametrize("pages, without, with_confirmation", [(1, 0.001, 0.0), (100, 0.095, 0.01), (1000, 0.632, 0.095)])
def test_hijack_after_many_pages(pages, without, with_confirmation):
    assert round(success_in(pages, hijack_rate(confirmation=False)), 3) == without
    assert round(success_in(pages, hijack_rate(confirmation=True)), 3) == with_confirmation


# --- Part 3: layers of safeguards ------------------------------------------------------------

def test_layers_multiply():
    assert [round(pass_rate(MISS[:k]), 3) for k in (1, 2, 3)] == [0.2, 0.06, 0.006]
    assert round(1000 * pass_rate(MISS)) == 6
    assert (round(1000 * MISS[0]), round(1000 * MISS[0] * MISS[1])) == (200, 60)  # numbers of the diagram


def test_harmless_requests_lost():
    assert round(1 - blocked_harmless(FALSE_BLOCK), 3) == 0.980
    assert round(blocked_harmless(FALSE_BLOCK), 4) == 0.0199


@pytest.mark.parametrize("threshold, miss, false_block, system_pass, system_lost", [
    (0.4, 0.1, 0.04, 0.003, 0.054),
    (0.6, 0.2, 0.005, 0.006, 0.020),
    (0.8, 0.5, 0.0, 0.015, 0.015),
])
def test_classifier_threshold(threshold, miss, false_block, system_pass, system_lost):
    assert (len(HARMFUL_SCORES), len(HARMLESS_SCORES)) == (10, 200)
    assert threshold_rates(threshold) == (miss, false_block)
    passed, lost = system_at_threshold(threshold)
    assert (round(passed, 3), round(lost, 3)) == (system_pass, system_lost)


def test_threshold_of_the_example_gives_the_first_layer():
    assert threshold_rates(0.6) == (MISS[0], FALSE_BLOCK[0])


# --- Part 4: red teaming ---------------------------------------------------------------------

@pytest.mark.parametrize("attempts, expected", [(1, 0.006), (10, 0.058), (100, 0.452), (500, 0.951)])
def test_many_attempts(attempts, expected):
    assert round(success_in(attempts, pass_rate(MISS)), 3) == expected


def test_attempts_for_a_95_percent_chance():
    rate = pass_rate(MISS)
    assert round(math.log(0.05) / math.log(1 - rate), 1) == 497.8
    assert attempts_needed(rate) == 498
    assert success_in(497, rate) < 0.95 <= success_in(498, rate)


def test_correlated_layers():
    assert round(min(MISS) / pass_rate(MISS), 1) == 16.7
    assert attempts_needed(min(MISS)) == 29


def test_no_success_is_not_a_proof():
    assert round(upper_limit_after_no_success(100), 3) == 0.03
    assert round(upper_limit_after_no_success(500), 3) == 0.006
