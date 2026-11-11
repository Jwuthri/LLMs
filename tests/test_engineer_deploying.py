"""Each number shown in docs/engineer/07-deploying.md."""

import math

import pytest

from examples.engineer.deploying import (
    GPU_MEMORY_GB, KV_BYTES_PER_TOKEN, LAPTOP_BUDGET_GB, LAPTOP_DECODE_TOKENS_PER_S,
    LAPTOP_PREFILL_TOKENS_PER_S, OUTPUT_TOKENS, PARAMS, PHONE_BUDGET_GB, PREFILL_TOKENS_PER_S,
    cost_per_million_tokens, decode_speed, device_memory_gb, fits, gpu_hours, kv_cache_gb,
    largest_model, latency_s, max_batch, memory_gb, requests_per_s, step_ms, tokens_per_hour,
    weights_gb,
)


def test_weights_memory():
    assert weights_gb(PARAMS, 16) == 16.0
    assert weights_gb(PARAMS, 8) == 8.0
    assert weights_gb(PARAMS, 4) == 4.0


def test_kv_cache_memory():
    assert KV_BYTES_PER_TOKEN == 131_072
    assert kv_cache_gb(700) == pytest.approx(0.0918, abs=5e-5)
    assert kv_cache_gb(700, 32) == pytest.approx(2.936, abs=5e-4)
    assert (GPU_MEMORY_GB - weights_gb(PARAMS, 16)) / kv_cache_gb(700) == pytest.approx(87.2, abs=0.05)
    assert 16.5 / PHONE_BUDGET_GB > 5
    assert memory_gb(16) == pytest.approx(16.092, abs=5e-4)
    assert memory_gb(4) == pytest.approx(4.092, abs=5e-4)
    assert memory_gb(16, 32) == pytest.approx(18.936, abs=5e-4)
    assert max_batch(GPU_MEMORY_GB, 16) == 87
    assert memory_gb(16, 87) <= GPU_MEMORY_GB < memory_gb(16, 88)


def test_laptop_latency():
    laptop = latency_s(LAPTOP_PREFILL_TOKENS_PER_S, LAPTOP_DECODE_TOKENS_PER_S)
    assert laptop == pytest.approx({"prefill": 2.0, "ttft": 2.05, "total": 12.0})
    assert memory_gb(4) <= LAPTOP_BUDGET_GB < memory_gb(16)


def test_server_latency():
    assert step_ms(1) == pytest.approx(20.0) and decode_speed(1) == pytest.approx(50.0)
    alone = latency_s(PREFILL_TOKENS_PER_S, decode_speed(1))
    assert alone == pytest.approx({"prefill": 0.1, "ttft": 0.12, "total": 4.1})
    assert step_ms(32) == pytest.approx(26.2)
    assert decode_speed(32) == pytest.approx(38.17, abs=5e-3)
    batched = latency_s(PREFILL_TOKENS_PER_S, decode_speed(32))
    assert batched["ttft"] == pytest.approx(0.126, abs=5e-4)
    assert batched["total"] == pytest.approx(5.34)


def test_demo_queue():
    assert 10 * latency_s(PREFILL_TOKENS_PER_S, decode_speed(1))["total"] == pytest.approx(41.0)
    assert requests_per_s(1) * 60 == pytest.approx(14.6, abs=0.05)


def test_throughput_and_cost():
    assert requests_per_s(1) == pytest.approx(0.244, abs=5e-4)
    assert requests_per_s(32) == pytest.approx(5.993, abs=5e-4)
    assert requests_per_s(1) * OUTPUT_TOKENS == pytest.approx(48.8, abs=0.05)
    assert requests_per_s(32) * OUTPUT_TOKENS == pytest.approx(1198.5, abs=0.05)
    assert tokens_per_hour(1) == pytest.approx(614_634, abs=1)
    assert tokens_per_hour(32) == pytest.approx(15_101_124, abs=1)
    assert cost_per_million_tokens(1) == pytest.approx(3.254, abs=5e-4)
    assert cost_per_million_tokens(32) == pytest.approx(0.132, abs=5e-4)
    assert cost_per_million_tokens(1) / cost_per_million_tokens(32) == pytest.approx(24.6, abs=0.05)
    assert cost_per_million_tokens(32, utilization=0.3) == pytest.approx(0.441, abs=5e-4)
    assert requests_per_s(32) * 3600 == pytest.approx(21_573, abs=1)
    slower = latency_s(PREFILL_TOKENS_PER_S, decode_speed(32))["total"] / latency_s(PREFILL_TOKENS_PER_S, decode_speed(1))["total"]
    assert slower == pytest.approx(1.3, abs=0.05)


def test_plot_points():
    assert cost_per_million_tokens(8) == pytest.approx(0.43, abs=5e-3)
    assert cost_per_million_tokens(87) == pytest.approx(0.07, abs=5e-3)


def test_gpu_hours():
    hours = gpu_hours(1e6, 32)
    assert hours == pytest.approx(46.4, abs=0.05)
    assert hours / 24 == pytest.approx(1.93, abs=5e-3)
    assert math.ceil(hours / 24) == 2


def test_edge_memory():
    assert largest_model(LAPTOP_BUDGET_GB, 4) == 15e9
    assert largest_model(PHONE_BUDGET_GB, 4) == 5e9
    assert device_memory_gb(8e9, 16) == 16.5 and not fits(8e9, 16, PHONE_BUDGET_GB)
    assert device_memory_gb(8e9, 4) == 4.5 and not fits(8e9, 4, PHONE_BUDGET_GB)
    assert device_memory_gb(3e9, 4) == 2.0 and fits(3e9, 4, PHONE_BUDGET_GB)
    assert device_memory_gb(1e9, 4) == 1.0 and fits(1e9, 4, PHONE_BUDGET_GB)
    assert fits(8e9, 4, LAPTOP_BUDGET_GB) and not fits(8e9, 16, LAPTOP_BUDGET_GB)
