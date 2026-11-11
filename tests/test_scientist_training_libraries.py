"""Each number shown in content/scientist/10-training-libraries.md."""

import pytest

from examples.scientist.training_libraries import (
    EVAL_BATCH, EVAL_RUNS, GPU_GB, PARAMS, RL_STEPS, STATE_BYTES_PER_PARAM, TOKENS_PER_EXAMPLE,
    decode_step_ms, decode_tokens_per_second, effective_batch, evaluation, fits, full_memory_per_gpu,
    gigabytes, lora_parameters, lora_parameters_per_block, memory_parts, memory_total, optimizer_steps,
    rl_step, sft_tokens, training_seconds, usable_gb,
)


def test_memory_of_the_weights():
    assert gigabytes(PARAMS, 16) == 16.0
    assert gigabytes(PARAMS, 4) == 4.0
    assert memory_parts("qlora")["weights"] == 5.5  # 4 GB + 1.5 GB of assumed overhead
    assert round(gigabytes(PARAMS, 16) / GPU_GB, 3) == 0.667  # two thirds of the GPU


def test_lora_parameter_count():
    assert lora_parameters_per_block() == 1_310_720
    assert lora_parameters() == 41_943_040
    assert round(100 * lora_parameters() / PARAMS, 3) == 0.524
    assert round(gigabytes(lora_parameters(), 8 * STATE_BYTES_PER_PARAM), 3) == 0.671


def test_memory_of_each_method():
    full = memory_parts("full")
    assert full == {"weights": 16.0, "states": 112.0, "activations": 4.0}
    assert (full["weights"] + full["states"]) / gigabytes(PARAMS, 16) == 8.0
    assert memory_total("full") == 132.0
    assert round(memory_total("lora"), 1) == 20.7
    assert round(memory_total("qlora"), 1) == 10.2
    assert memory_total("full") / GPU_GB == 5.5
    assert round(memory_parts("lora")["states"], 1) == 0.7


def test_what_fits_on_one_24_gb_gpu():
    assert usable_gb() == pytest.approx(21.6)
    assert not fits("full")
    assert fits("lora") and round(usable_gb() - memory_total("lora"), 1) == 0.9
    assert fits("qlora") and round(usable_gb() - memory_total("qlora"), 1) == 11.4


def test_full_fine_tuning_on_4_gpus_of_80_gb():
    assert usable_gb(80) == 72.0
    assert full_memory_per_gpu(4, sharded=False) == 132.0
    assert full_memory_per_gpu(4, sharded=True) == 36.0


def test_steps_and_tokens_of_the_sft_job():
    assert effective_batch() == 16
    assert optimizer_steps() == 1875
    assert sft_tokens() == 15_000_000
    assert effective_batch() * TOKENS_PER_EXAMPLE == 8_000
    assert training_seconds(effective_batch() * TOKENS_PER_EXAMPLE) == 3.2
    assert training_seconds(sft_tokens()) / 60 == 100
    assert 2 * sft_tokens() == 30_000_000  # DPO: the policy reads two answers for each pair


def test_one_rl_step():
    step = rl_step()
    assert step["samples"] == 128
    assert step["generated_tokens"] == 38_400
    assert step["generated_tokens"] / (effective_batch() * TOKENS_PER_EXAMPLE) == 4.8
    assert step["trained_tokens"] == 64_000
    assert step["train_s"] == 25.6
    assert step["one_by_one_s"] == pytest.approx(768)
    assert decode_step_ms(128) == pytest.approx(45.4)
    assert round(step["batched_s"], 2) == 13.62
    assert round(step["generation_speed_up"], 1) == 56.4
    assert round(step["one_by_one_share"], 3) == 0.968
    assert round(step["batched_share"], 3) == 0.347
    assert round(step["step_one_by_one_s"], 1) == 793.6
    assert round(step["step_batched_s"], 2) == 39.22


def test_rl_run_of_500_steps():
    step = rl_step()
    assert RL_STEPS * step["generated_tokens"] == 19_200_000
    assert round(RL_STEPS * step["step_batched_s"] / 3600, 1) == 5.4
    assert round(RL_STEPS * step["step_one_by_one_s"] / 3600) == 110


def test_evaluation_time():
    result = evaluation()
    assert result["tokens"] == 200_000
    assert decode_tokens_per_second(1) == pytest.approx(50)
    assert result["one_by_one_s"] == pytest.approx(4000)
    assert round(result["one_by_one_s"] / 60, 1) == 66.7
    assert decode_step_ms(EVAL_BATCH) == pytest.approx(32.6)
    assert round(result["batched_tokens_per_s"]) == 1963
    assert round(result["batched_s"], 1) == 101.9
    assert round(result["speed_up"], 1) == 39.3
    assert round(EVAL_RUNS * result["one_by_one_s"] / 60) == 333
    assert round(EVAL_RUNS * result["batched_s"] / 60, 1) == 8.5
