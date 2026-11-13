"""Each number shown in content/scientist/02-pre-training.md."""

import math

import numpy as np

from examples.scientist.pre_training import (
    ADAMW, BATCH_TARGETS, LOSS_FLOOR, MAX_NORM, RAW_CORPUS, SCHEDULE, TOKENS_PER_STEP,
    adamw_step, batch_gradient, batch_loss, clip_by_norm, data_parallel, deduplicate, fingerprint,
    float16_facts, gradient, loss, lr_schedule, normalize, quality_filter, tensor_parallel,
    training_examples, training_memory_gb, training_run, unique_ratio,
)
from examples.toy import LOGITS, E


def test_loss_and_perplexity():
    assert round(loss(LOGITS, "sat"), 3) == 0.417
    assert round(math.exp(loss(LOGITS, "sat")), 3) == 1.517
    assert round(math.log(3), 3) == 1.099
    assert round(math.log(50_257), 2) == 10.82 and round(math.log(128_256), 2) == 11.76


def test_deduplication():
    assert [normalize(doc) for doc in RAW_CORPUS[:3]] == ["the cat sat"] * 3
    assert [fingerprint(doc) for doc in RAW_CORPUS] == ["9ef3d846", "9ef3d846", "9ef3d846", "4fa30b89", "9b90c680"]
    assert deduplicate(RAW_CORPUS) == ["the cat sat", "cat cat cat cat", "the cat the cat sat"]
    assert math.comb(5, 2) == 10 and math.comb(10 ** 9, 2) == 499_999_999_500_000_000


def test_quality_filter_and_examples():
    unique = deduplicate(RAW_CORPUS)
    assert [unique_ratio(doc) for doc in unique] == [1.0, 0.25, 0.6]
    kept = quality_filter(unique)
    assert kept == ["the cat sat", "the cat the cat sat"]
    assert sum(len(normalize(doc).split()) for doc in RAW_CORPUS) == 18
    assert sum(len(doc.split()) for doc in kept) == 8
    assert training_examples(kept[0]) == [("the", "cat"), ("the cat", "sat")]
    assert sum(len(training_examples(doc)) for doc in kept) == 6


def test_data_parallel_gradient():
    assert np.allclose(gradient(LOGITS, "sat"), [0.099, 0.242, -0.341], atol=5e-4)
    assert np.allclose(gradient(LOGITS, "cat"), [0.099, -0.758, 0.659], atol=5e-4)
    assert round(loss(LOGITS, "cat"), 3) == 1.417
    result = data_parallel(LOGITS, BATCH_TARGETS)
    assert np.allclose(result["workers"][0], [0.099, 0.242, -0.341], atol=5e-4)
    assert np.allclose(result["workers"][1], [0.099, -0.258, 0.159], atol=5e-4)
    assert np.allclose(result["mean"], [0.099, -0.008, -0.091], atol=5e-4)
    assert np.allclose(result["mean"], batch_gradient(LOGITS, BATCH_TARGETS))
    assert round(batch_loss(LOGITS, BATCH_TARGETS), 3) == 0.667


def test_tensor_parallel_and_memory():
    parts = tensor_parallel(E[2])
    assert np.allclose(parts["worker_a"], [-0.7]) and np.allclose(parts["worker_b"], [1.1])
    assert np.allclose(np.concatenate([parts["worker_a"], parts["worker_b"]]), parts["full"])
    assert training_memory_gb(8e9) == 128.0
    assert 8e9 * 4 / 1e9 == 32.0 and 8e9 * 2 / 1e9 == 16.0


def test_gradient_clipping():
    clip = clip_by_norm(gradient(LOGITS, "sat"), MAX_NORM)
    assert round(clip["norm"], 3) == 0.430 and round(clip["scale"], 3) == 0.582
    assert np.allclose(clip["clipped"], [0.057, 0.141, -0.198], atol=5e-4)
    assert round(float(np.linalg.norm(clip["clipped"])), 3) == 0.25
    assert clip_by_norm(gradient(LOGITS, "sat"), 1.0)["scale"] == 1.0


def test_adamw_step():
    g = clip_by_norm(gradient(LOGITS, "sat"), MAX_NORM)["clipped"]
    step = adamw_step(LOGITS, g, np.zeros(3), np.zeros(3), 1, **ADAMW)
    assert np.allclose(step["m"], [0.0057, 0.0141, -0.0198], atol=5e-5)
    assert np.allclose(step["v"] * 1e6, [3.3, 19.9, 39.3], atol=0.05)
    assert np.allclose(step["m_hat"], g) and np.allclose(step["v_hat"], g ** 2)
    assert np.allclose(step["v_hat"], [0.0033, 0.0199, 0.0393], atol=5e-5)
    assert np.allclose(step["direction"], [1, 1, -1], atol=1e-6)
    assert np.allclose(step["decay"], [0.001, 0.01, 0.02])
    assert np.allclose(step["theta"], [-0.0001, 0.899, 2.098], atol=5e-5)
    assert round(loss(step["theta"], "sat"), 3) == 0.354


def test_lr_schedule():
    values = {s: lr_schedule(s, **SCHEDULE) for s in (0, 50, 100, 325, 550, 1000)}
    assert values[0] == 0 and math.isclose(values[50], 1.5e-4) and math.isclose(values[100], 3e-4)
    assert round(values[325] * 1e4, 2) == 2.56
    assert math.isclose(values[550], 1.5e-4) and abs(values[1000]) < 1e-12


def test_mixed_precision():
    facts = float16_facts()
    assert facts["bytes"] == {"fp32": 4, "fp16": 2}
    assert facts["toy_table_bytes"] == {"fp32": 24, "fp16": 12}
    assert round(facts["eps16"], 6) == 0.000977 and facts["max16"] == 65504.0
    assert facts["lost_update"] and facts["kept_update_fp32"]


def test_training_run():
    log = training_run()
    assert round(log[0]["loss"], 3) == 1.099 and round(log[0]["grad_norm"], 3) == 0.540 and log[0]["lr"] == 0
    assert round(log[10]["loss"], 3) == 0.823 and round(log[10]["grad_norm"], 3) == 0.307
    assert math.isclose(log[10]["lr"], 0.1)
    assert round(log[50]["loss"], 3) == 0.579 and round(log[50]["grad_norm"], 3) == 0.020
    assert round(log[50]["lr"], 3) == 0.059
    assert round(log[100]["loss"], 3) == 0.575 and round(log[100]["grad_norm"], 3) == 0.018
    assert round(LOSS_FLOOR, 3) == 0.562
    assert TOKENS_PER_STEP == 12 and TOKENS_PER_STEP * 2 == 24
    assert 15e12 / 1e6 == 15e6 and 14 * 8 == 112
    assert round(15e12 / 1e6 / 86_400) == 174
