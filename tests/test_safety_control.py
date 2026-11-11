"""Each number shown in content/safety/06-control.md."""

import pytest

from examples.safety.control import (
    ATTACK_WORKS, AUDIT_BUDGET, BACKDOOR_TASK, CATCH_ACTION, CATCH_COT, DEFER, HIDDEN, ROLE_ADVANTAGE,
    SEPARATION, SOLVED_TRUSTED, SOLVED_UNTRUSTED, SUSPICION, attack_split, audited_tasks, catch_rate,
    chain_win, cot_catch_with_hiding, either_catches, guard_win, monitor_auc, protocol_safety, safety,
    safety_without_defer, simulate_safety, success_rate, top_share_threshold, usefulness,
)


def test_assumed_values_of_the_example():
    assert len(SUSPICION) == 10 and SUSPICION[5] == 8.1 and SUSPICION[BACKDOOR_TASK - 1] == 7.4
    assert (AUDIT_BUDGET, SEPARATION, ATTACK_WORKS, DEFER) == (0.02, 2.0, 0.5, 0.10)
    assert (SOLVED_UNTRUSTED, SOLVED_TRUSTED) == (0.9, 0.5)
    assert (CATCH_ACTION, CATCH_COT, HIDDEN, ROLE_ADVANTAGE) == (0.4, 0.8, 0.5, 400)


def test_audit_of_the_ten_tasks():
    assert audited_tasks(SUSPICION, 0.2) == [6, 8]  # the backdoor is in the audited set
    assert BACKDOOR_TASK in audited_tasks(SUSPICION, 0.2)
    assert audited_tasks(SUSPICION, 0.1) == [6]  # a budget of 1 audit goes to a normal output
    assert round(0.02 * 1000) == 20  # audits for 1,000 tasks


def test_audit_threshold_and_catch_rate():
    assert round(top_share_threshold(AUDIT_BUDGET), 3) == 2.054
    assert round(catch_rate(AUDIT_BUDGET, SEPARATION), 3) == 0.479
    assert [round(catch_rate(AUDIT_BUDGET, m), 3) for m in (1.0, 2.0, 3.0)] == [0.146, 0.479, 0.828]
    assert [round(catch_rate(b, SEPARATION), 3) for b in (0.01, 0.05)] == [0.372, 0.639]
    assert round(monitor_auc(SEPARATION), 3) == 0.921
    assert monitor_auc(0.0) == 0.5  # a monitor that sees no difference


@pytest.mark.parametrize("separation, catch, success, win", [
    (1.0, 0.146, 0.427, 0.255), (2.0, 0.479, 0.261, 0.647), (3.0, 0.828, 0.086, 0.906),
])
def test_the_whole_game(separation, catch, success, win):
    assert round(catch_rate(AUDIT_BUDGET, separation), 3) == catch
    assert round(success_rate(AUDIT_BUDGET, separation), 3) == success
    assert round(protocol_safety(AUDIT_BUDGET, separation), 3) == win


def test_formula_agrees_with_a_simulation():
    catch, success = catch_rate(AUDIT_BUDGET, SEPARATION), success_rate(AUDIT_BUDGET, SEPARATION)
    assert abs(simulate_safety(catch, success, games=20_000, seed=0) - safety(catch, success)) < 0.01
    assert safety(0.3, 0.0) == 1.0 and safety(0.2, 0.2) == 0.5


def test_defer_to_trusted():
    assert round(top_share_threshold(DEFER), 3) == 1.282
    split = attack_split(AUDIT_BUDGET, SEPARATION, DEFER)
    assert [round(x, 3) for x in split] == [0.479, 0.285, 0.236]
    assert abs(sum(split) - 1) < 1e-12  # the three numbers of the diagram
    assert round(success_rate(AUDIT_BUDGET, SEPARATION, DEFER), 3) == 0.118
    assert round(protocol_safety(AUDIT_BUDGET, SEPARATION, DEFER), 3) == 0.802
    assert round(usefulness(0.0), 3) == 0.9 and round(usefulness(DEFER), 3) == 0.86


@pytest.mark.parametrize("defer, win, solved", [
    (0.0, 0.647, 0.9), (0.1, 0.802, 0.86), (0.2, 0.886, 0.82), (0.5, 0.977, 0.7), (1.0, 1.0, 0.5),
])
def test_safety_usefulness_curve(defer, win, solved):
    assert round(protocol_safety(AUDIT_BUDGET, SEPARATION, defer), 3) == win
    assert round(usefulness(defer), 3) == solved


def test_chain_of_thought_monitoring():
    assert round((1 - CATCH_ACTION) * ATTACK_WORKS, 3) == 0.3  # s of the action-only monitor
    assert round((1 - CATCH_ACTION) * (1 - CATCH_COT), 3) == 0.12  # an attack passes the two monitors
    both = either_catches(CATCH_ACTION, CATCH_COT)
    assert round(both, 3) == 0.88
    hidden_cot = cot_catch_with_hiding(CATCH_COT, HIDDEN)
    assert round(hidden_cot, 3) == 0.4
    assert round(either_catches(CATCH_ACTION, hidden_cot), 3) == 0.64
    assert round(either_catches(CATCH_ACTION, cot_catch_with_hiding(CATCH_COT, 1.0)), 3) == 0.4
    assert [round(safety_without_defer(c), 3) for c in (0.4, both, 0.64)] == [0.571, 0.936, 0.78]


def test_guard_win_probability():
    assert [round(guard_win(gap), 3) for gap in (0, 200, 400, 800)] == [0.5, 0.24, 0.091, 0.01]


def test_nested_oversight():
    assert [round(chain_win(800, n), 3) for n in (1, 2, 4)] == [0.01, 0.008, 0.003]
    assert round(guard_win(0, ROLE_ADVANTAGE), 3) == 0.909
    assert round(guard_win(200, ROLE_ADVANTAGE), 3) == 0.76  # one step of a chain of 4
    with_advantage = {n: round(chain_win(800, n, ROLE_ADVANTAGE), 3) for n in (1, 2, 4, 8, 16)}
    assert with_advantage == {1: 0.091, 2: 0.25, 4: 0.333, 8: 0.27, 16: 0.135}
    assert max(with_advantage, key=with_advantage.get) == 4
    assert round(1 - with_advantage[4], 2) == 0.67  # the best chain fails 2 times in 3
    assert round(1 / guard_win(400)) == 11  # the guard wins 1 game in 11
