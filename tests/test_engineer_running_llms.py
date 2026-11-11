"""Each number shown in docs/engineer/01-running-llms.md."""

import json

import numpy as np

from examples.engineer.running_llms import (
    ALLOWED, CONTEXT, INPUT_TOKENS, KV_SIZE, LAYERS, OUTPUT_TOKENS, PARAMS, PRICE_INPUT_PER_MILLION,
    PRICE_OUTPUT_PER_MILLION, PROMPTS, REQUEST, REQUESTS_PER_DAY, RESPONSE, SCHEMA_OUTPUTS, approx_tokens,
    calculator, constrained_probs, free_steps, kv_cache_gb, request_cost, weights_gb,
)
from examples.toy import LOGITS, VOCAB, softmax


def test_messages_shape():
    assert json.loads(json.dumps(REQUEST)) == REQUEST
    assert [m["role"] for m in REQUEST["messages"]] == ["system", "user"]
    assert RESPONSE["choices"][0]["message"] == {"role": "assistant", "content": "sat"}
    assert RESPONSE["usage"] == {"prompt_tokens": 11, "completion_tokens": 1}
    assert [len(m["content"]) for m in REQUEST["messages"]] == [36, 7]
    assert [approx_tokens(m["content"]) for m in REQUEST["messages"]] == [9, 2]


def test_cost_with_assumed_prices():
    assert (PRICE_INPUT_PER_MILLION, PRICE_OUTPUT_PER_MILLION) == (3.0, 15.0)
    assert (INPUT_TOKENS, OUTPUT_TOKENS, REQUESTS_PER_DAY) == (1000, 200, 10_000)
    cost = request_cost(INPUT_TOKENS, OUTPUT_TOKENS)
    assert np.isclose(cost["input"], 0.003) and np.isclose(cost["output"], 0.003)
    assert np.isclose(cost["total"], 0.006)
    assert np.isclose(REQUESTS_PER_DAY * cost["total"], 60)
    assert np.isclose(30 * REQUESTS_PER_DAY * cost["total"], 1800)
    assert PRICE_OUTPUT_PER_MILLION / PRICE_INPUT_PER_MILLION == 5


def test_local_memory():
    assert weights_gb(PARAMS, 16) == 16.0 and weights_gb(PARAMS, 4) == 4.0
    assert (LAYERS, CONTEXT, KV_SIZE) == (32, 8192, 1024)
    cache = kv_cache_gb(LAYERS, CONTEXT, KV_SIZE)
    assert 2 * LAYERS * CONTEXT * KV_SIZE * 2 == 1_073_741_824
    assert round(cache, 2) == 1.07
    assert round(weights_gb(PARAMS, 16) + cache, 2) == 17.07
    assert round(weights_gb(PARAMS, 4) + cache, 2) == 5.07
    assert round(kv_cache_gb(LAYERS, 2048, KV_SIZE), 2) == 0.27


def test_prompt_sizes():
    assert [len(prompt) for prompt in PROMPTS.values()] == [77, 234, 103, 285]
    sizes = {name: approx_tokens(prompt) for name, prompt in PROMPTS.items()}
    assert sizes == {"zero-shot": 20, "few-shot": 59, "chain of thought": 26, "ReAct": 72}
    extra = sizes["few-shot"] - sizes["zero-shot"]
    assert extra == 39
    assert np.isclose(REQUESTS_PER_DAY * request_cost(extra, 0)["total"], 1.17)
    assert calculator("14 * 3") == 42 and 14 * 3 == 42


def test_constrained_decoding():
    c = constrained_probs(LOGITS, VOCAB, ALLOWED)
    assert c["mask"].tolist() == [True, True, False]
    assert c["masked"][2] == -np.inf
    assert np.allclose(c["exp"], [1.105, 2.718, 0], atol=5e-4) and round(float(c["exp"].sum()), 3) == 3.823
    assert np.allclose(c["probs"], [0.289, 0.711, 0], atol=5e-4)
    before = softmax(LOGITS)
    assert np.allclose(before, [0.099, 0.242, 0.659], atol=5e-4)
    assert round(float(before[0] + before[1]), 3) == 0.341
    assert np.allclose(before[:2] / before[:2].sum(), c["probs"][:2])


def test_json_schema_forces_most_steps():
    assert all(json.loads(output)["word"] in ALLOWED for output in SCHEMA_OUTPUTS)
    for output in SCHEMA_OUTPUTS:
        assert len(output) == 15
        assert free_steps(SCHEMA_OUTPUTS, output) == [10]
