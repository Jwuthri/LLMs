"""Each number and message shown in docs/engineer/05-agents.md."""

import json

import pytest

from examples.engineer.agents import (
    CALCULATOR_TOOL, CAT_REPLIES, RECOVERY_REPLIES, TASK, TOOLS, ScriptedModel, call_tool,
    chain_success, check_arguments, integrations, longest_chain, mcp_messages, run_agent, step_success_needed,
    with_one_retry,
)


def test_tool_schema():
    schema = CALCULATOR_TOOL["input_schema"]
    assert CALCULATOR_TOOL["name"] == "calculator"
    assert schema["required"] == ["operation", "a", "b"]
    assert schema["properties"]["operation"]["enum"] == ["add", "subtract", "multiply", "divide"]


def test_cat_trace_and_answer():
    model = ScriptedModel(CAT_REPLIES)
    result = run_agent(model, TASK, TOOLS)
    assert result.answer == "3 cats have 12 legs."
    assert [kind for kind, _ in result.trace] == ["thought", "action", "observation", "thought", "answer"]
    assert result.trace[1][1] == 'calculator({"operation": "multiply", "a": 3, "b": 4})'
    assert result.trace[2] == ("observation", "12")
    assert result.model_calls == 2
    assert model.context_sizes == [2, 4]  # each call receives the full history


def test_messages():
    result = run_agent(ScriptedModel(CAT_REPLIES), TASK, TOOLS)
    assert [m["role"] for m in result.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert result.messages[2]["tool_call"] == {
        "id": "call_1", "name": "calculator", "arguments": {"operation": "multiply", "a": 3, "b": 4}}
    assert result.messages[3] == {"role": "tool", "tool_call_id": "call_1", "content": "12"}


def test_tool_errors_become_observations():
    assert call_tool(TOOLS, {"name": "multiplier", "arguments": {}}) == "Error: unknown tool 'multiplier'"
    assert call_tool(TOOLS, {"name": "calculator", "arguments": {"operation": "multiply", "a": 3}}) \
        == "Error: missing argument 'b'"
    assert call_tool(TOOLS, {"name": "calculator", "arguments": {"operation": "divide", "a": 3, "b": 0}}) \
        == "Error: division by zero"
    schema = CALCULATOR_TOOL["input_schema"]
    assert check_arguments(schema, {"operation": "power", "a": 3, "b": 4}).startswith("argument 'operation'")
    assert check_arguments(schema, {"operation": "add", "a": "3", "b": 4}) == "argument 'a' must be a number"
    assert check_arguments(schema, {"operation": "add", "a": 3, "b": 4}) is None


def test_model_corrects_a_wrong_call():
    result = run_agent(ScriptedModel(RECOVERY_REPLIES), TASK, TOOLS)
    assert result.trace[2] == ("observation", "Error: unknown tool 'multiplier'")
    assert result.answer == "3 cats have 12 legs."
    assert result.model_calls == 3


def test_step_limit_stops_the_loop():
    endless = ScriptedModel([CAT_REPLIES[0]] * 3)
    result = run_agent(endless, TASK, TOOLS, max_steps=3)
    assert result.answer is None
    assert result.model_calls == 3


def test_mcp_message_shape():
    request, response = mcp_messages(CAT_REPLIES[0]["tool_call"], "12")
    assert request["method"] == "tools/call"
    assert request["params"] == {"name": "calculator", "arguments": {"operation": "multiply", "a": 3, "b": 4}}
    assert response["result"] == {"resultType": "complete", "content": [{"type": "text", "text": "12"}]}
    assert json.loads(json.dumps(request)) == request


def test_integration_count():
    assert integrations(3, 4) == {"custom": 12, "protocol": 7}


def test_chain_reliability():
    assert [round(chain_success(0.95, n), 3) for n in (1, 5, 10, 20)] == [0.95, 0.774, 0.599, 0.358]
    assert round(chain_success(0.95, 15), 3) == 0.463  # 3 agents x 5 steps
    assert longest_chain(0.95, 0.5) == 13
    assert chain_success(0.95, 13) >= 0.5 > chain_success(0.95, 14)
    assert round(step_success_needed(0.9, 20), 4) == 0.9947
    assert with_one_retry(0.95) == pytest.approx(0.9975)
    assert round(chain_success(with_one_retry(0.95), 20), 3) == 0.951
