"""Each number shown in docs/scientist/04-fine-tuning.md."""

import numpy as np

from examples.scientist.fine_tuning import (
    A_INIT, B_INIT, CURVE_STEPS, EXAMPLE_GRADS, LORA_ALPHA, X,
    accumulate, best_validation_step, clip, full_params, full_step, grad_norm, grad_w, lora_delta,
    lora_grads, lora_params, lora_step, loss, loss_curves, lr_schedule, megabytes, merge,
    steps_per_epoch, weight_decay_step, zero_bytes_per_param, zero_toy_counts,
)
from examples.toy import W


def test_forward_and_gradient():
    assert np.allclose(W @ X, [-1.9, 1.4])
    assert np.isclose(loss(W), 0.16)
    assert np.allclose(grad_w(W), [[-0.4, -0.8], [0.4, 0.8]])


def test_full_fine_tuning_changes_all_four_numbers():
    w_new = full_step(W)
    assert np.allclose(w_new, [[0.54, -1.12], [0.76, 0.22]])
    assert np.all(w_new != W)
    assert np.allclose(w_new @ X, [-1.7, 1.2])
    assert np.isclose(loss(w_new), 0.04)


def test_lora_starts_at_zero_and_learns_through_b():
    assert np.allclose(lora_delta(B_INIT, A_INIT), 0)
    grad_b, grad_a = lora_grads(W, B_INIT, A_INIT)
    assert np.allclose(grad_b.ravel(), [-1.2, 1.2])
    assert np.allclose(grad_a, 0)
    # With A = 0 and B = 0 the two gradients are zero: the adapter cannot learn.
    dead_b, dead_a = lora_grads(W, B_INIT, np.zeros_like(A_INIT))
    assert np.allclose(dead_b, 0) and np.allclose(dead_a, 0)


def test_lora_step_and_merge():
    b, a = lora_step(W, B_INIT, A_INIT)
    assert np.allclose(b.ravel(), [0.12, -0.12])
    assert np.allclose(a, A_INIT)
    assert np.allclose(lora_delta(b, a), [[0.12, 0.12], [-0.12, -0.12]])
    merged = merge(W, b, a)
    assert np.allclose(merged, [[0.62, -1.08], [0.68, 0.18]])
    assert np.allclose(merged @ X, [-1.54, 1.04])
    # The merged matrix and the separate adapter path give the same output.
    assert np.allclose(merged @ X, W @ X + LORA_ALPHA * (b @ (a @ X)))
    assert np.isclose(loss(merged), 0.0016)
    assert np.linalg.matrix_rank(lora_delta(b, a)) == 1


def test_parameter_counts():
    assert (full_params(2), lora_params(2, 1)) == (4, 4)
    assert full_params(4096) == 16_777_216
    assert lora_params(4096, 1) == 8_192
    assert lora_params(4096, 16) == 131_072
    assert round(100 * lora_params(4096, 16) / full_params(4096), 3) == 0.781
    assert round(100 * lora_params(4096, 1) / full_params(4096), 3) == 0.049


def test_memory():
    assert round(megabytes(full_params(4096), 16), 2) == 33.55
    assert round(megabytes(full_params(4096), 4), 2) == 8.39
    assert round(megabytes(lora_params(4096, 16), 16), 2) == 0.26
    assert megabytes(8_000_000_000, 16) == 16_000 and megabytes(8_000_000_000, 4) == 4_000


def test_learning_rate_schedule():
    values = [lr_schedule(step, 3e-4, 10, 100) for step in (0, 5, 10, 55, 100)]
    assert np.allclose(values, [0, 1.5e-4, 3e-4, 1.5e-4, 0], atol=1e-12)


def test_gradient_accumulation():
    micro, accumulated = accumulate(EXAMPLE_GRADS, 2)
    assert np.allclose(micro, [-0.4, -0.2, -0.6, 0.0])
    assert np.isclose(accumulated, -0.3)
    assert np.isclose(accumulated, EXAMPLE_GRADS.mean())  # same as one batch of 8
    assert np.isclose(W[0, 0] - 0.1 * accumulated, 0.53)
    assert steps_per_epoch(1000, 2, 4) == 125


def test_weight_decay():
    assert np.isclose(weight_decay_step(W[0, 1], 0.1, 0.01), -1.1988)


def test_zero_stages():
    assert [zero_toy_counts(stage)["total"] for stage in range(4)] == [16, 12, 10, 8]
    assert zero_toy_counts(3) == {"weights": 2, "gradients": 2, "optimizer": 4, "total": 8}
    assert [zero_bytes_per_param(stage, 2) for stage in range(4)] == [16, 10, 9, 8]
    assert [zero_bytes_per_param(stage, 8) for stage in range(4)] == [16, 5.5, 3.75, 2]
    # The example of the ZeRO paper: 7.5 billion parameters on 64 GPUs.
    assert [round(7.5 * zero_bytes_per_param(stage, 64), 1) for stage in range(4)] == [120.0, 31.4, 16.6, 1.9]


def test_gradient_norm_and_clipping():
    g = grad_w(W)
    assert round(grad_norm(g), 3) == 1.265
    clipped = clip(g, 1.0)
    assert np.allclose(clipped, [[-0.316, -0.632], [0.316, 0.632]], atol=5e-4)
    assert np.isclose(grad_norm(clipped), 1.0)
    assert round(1.0 / grad_norm(g), 3) == 0.791
    assert np.allclose(clip(g, 2.0), g)  # a small gradient does not change


def test_loss_curves_show_overfitting():
    step, value = best_validation_step()
    assert step == 150 and round(value, 3) == 0.865
    train, val = loss_curves(CURVE_STEPS)
    assert np.all(np.diff(train) < 0)  # training loss always goes down
    assert round(float(train[-1]), 3) == 0.411 and round(float(val[-1]), 3) == 1.509
    assert round(float(train[CURVE_STEPS == 150][0]), 3) == 0.531
