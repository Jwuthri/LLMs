"""Each number shown in docs/engineer/06-inference-optimization.md."""

import numpy as np
import pytest

from examples.engineer.inference_optimization import (
    DRAFT_PROBS, GIB, LLAMA3_8B, acceptance, acceptance_rate, attention_matrix_bytes,
    expected_accepted, expected_tokens, kv_cache_bytes, kv_cache_counts, kv_cache_formulas,
    online_softmax, output_distribution, residual, speedup,
)
from examples.scientist.architecture import attention
from examples.toy import LOGITS, E, softmax


def test_online_softmax_equals_standard_softmax():
    result = attention(E)
    row = result["scores"][2]
    assert np.allclose(row, [0.707, 0.707, 1.414], atol=5e-4)
    first, second = online_softmax(row, E, block_size=2)
    assert first["m"] == pytest.approx(0.707, abs=5e-4) and first["l"] == pytest.approx(2.0)
    assert np.allclose(first["o"], [1, 1])
    assert second["rescale"] == pytest.approx(0.493, abs=5e-4)
    assert second["m"] == pytest.approx(1.414, abs=5e-4) and second["l"] == pytest.approx(1.986, abs=5e-4)
    assert np.allclose(second["o"], [1.493, 1.493], atol=5e-4)
    assert np.allclose(second["output"], result["output"][2])  # the same result, not an approximation
    assert np.allclose(second["output"], [0.752, 0.752], atol=5e-4)
    weights = np.exp(row - second["m"]) / second["l"]
    assert np.allclose(weights, [0.248, 0.248, 0.503], atol=5e-4)


def test_online_softmax_any_block_size():
    rng = np.random.default_rng(0)
    scores, values = rng.normal(size=7), rng.normal(size=(7, 2))
    expected = softmax(scores) @ values
    for block_size in (1, 2, 3, 7):
        assert np.allclose(online_softmax(scores, values, block_size)[-1]["output"], expected)


def test_attention_matrix_memory():
    n = LLAMA3_8B["context"]
    assert n * n == 67_108_864
    assert attention_matrix_bytes(n, LLAMA3_8B["heads"]) == 4 * GIB
    assert 2 * n == 16_384


def test_kv_cache_counts():
    assert kv_cache_counts(3) == {"kv_no_cache": 6, "kv_cache": 3, "dots_no_cache": 10, "dots_cache": 6}
    assert kv_cache_counts(1000) == {
        "kv_no_cache": 500_500, "kv_cache": 1_000, "dots_no_cache": 167_167_000, "dots_cache": 500_500}
    for n in (1, 2, 3, 10, 1000):
        assert kv_cache_counts(n) == kv_cache_formulas(n)
    assert kv_cache_counts(1000)["dots_no_cache"] / kv_cache_counts(1000)["dots_cache"] == 334


def test_kv_cache_memory():
    assert kv_cache_bytes(1, 1, 2, 3, bytes_per_value=1) == 12  # toy: 12 numbers
    layers, head_dim, context = LLAMA3_8B["layers"], LLAMA3_8B["head_dim"], LLAMA3_8B["context"]
    assert LLAMA3_8B["heads"] * head_dim == 4096
    assert kv_cache_bytes(layers, 32, head_dim, 1) == 524_288
    assert kv_cache_bytes(layers, 8, head_dim, 1) == 131_072
    assert kv_cache_bytes(layers, 1, head_dim, 1) == 16_384
    assert kv_cache_bytes(layers, 32, head_dim, context) == 4 * GIB
    assert kv_cache_bytes(layers, 8, head_dim, context) == 1 * GIB
    assert kv_cache_bytes(layers, 1, head_dim, context) == 0.125 * GIB


def test_speculative_acceptance():
    target = softmax(LOGITS)
    assert np.allclose(acceptance(target, DRAFT_PROBS), [0.493, 0.808, 1.0], atol=5e-4)
    assert acceptance_rate(target, DRAFT_PROBS) == pytest.approx(0.841, abs=5e-4)
    assert np.allclose(residual(target, DRAFT_PROBS), [0, 0, 1])
    assert np.allclose(output_distribution(target, DRAFT_PROBS), target)  # exact, not approximate


def test_speculative_speedup():
    a = acceptance_rate(softmax(LOGITS), DRAFT_PROBS)
    assert expected_accepted(a, 4) == pytest.approx(2.643, abs=5e-4)
    assert expected_tokens(a, 4) == pytest.approx(3.643, abs=5e-4)
    assert expected_tokens(a, 4) == pytest.approx(1 + sum(a ** i for i in range(1, 5)))
    assert speedup(a, 4, 0.05) == pytest.approx(3.036, abs=5e-4)
    assert speedup(0.5, 4, 0.3) == pytest.approx(0.881, abs=5e-4)  # a bad draft model is slower


def test_best_draft_length():
    for rate, best_k, best_speedup in ((0.9, 13, 4.67), (0.8, 8, 3.09), (0.6, 4, 1.92)):
        best = max(range(1, 21), key=lambda k: speedup(rate, k, 0.05))
        assert best == best_k
        assert speedup(rate, best, 0.05) == pytest.approx(best_speedup, abs=5e-3)


def test_speculative_simulation_matches_formula():
    """Draw draft tokens, keep each with probability a, stop at the first rejection."""
    rng = np.random.default_rng(0)
    a, k = 0.8, 4
    kept = (rng.random((200_000, k)) < a)
    accepted = np.where(kept.all(axis=1), k, kept.argmin(axis=1))
    assert accepted.mean() == pytest.approx(expected_accepted(a, k), abs=0.02)
