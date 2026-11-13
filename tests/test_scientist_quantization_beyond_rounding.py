"""Each number shown in content/scientist/11-quantization-beyond-rounding.md."""

import numpy as np

from examples.scientist.quantization import CALIB_X
from examples.scientist.quantization_beyond_rounding import (
    BITS_TERNARY, BUDGETS, CALIB, PAIRS, QAT_STEP, QAT_TARGET, REAL_LAYER, SENSITIVITY_AT_4_BITS,
    STEP_CANDIDATES, W_RTN, allocate_bits, codebook_bits, codebook_bpw, column_squared_error, importance,
    kl_divergence, kmeans, layer_error, mean_abs_error, mean_kl, mean_output_error, qat_steps, sign_quantize,
    size_gb, ternarize, total_bpw, total_error, toy_probs, weighted_error, weighted_rounding,
)
from examples.toy import W


def test_baseline_is_the_four_bit_result_of_lesson_7():
    assert np.allclose(W_RTN, [[0.514, -1.2], [0.857, 0.343]], atol=5e-4)
    assert round(mean_abs_error(W, W_RTN), 4) == 0.0286
    assert np.allclose(W @ CALIB_X, [-0.2, 1.9])
    assert np.allclose(toy_probs(W, CALIB_X), [0.063, 0.515, 0.422], atol=5e-4)
    assert np.allclose(toy_probs(W_RTN, CALIB_X), [0.055, 0.513, 0.432], atol=5e-4)
    assert round(mean_kl(W, W_RTN) * 1000, 3) == 0.662


def test_importance_is_the_mean_squared_activation():
    assert CALIB.tolist() == [[2, 1], [4, -1], [-2, 1], [4, 1]]
    assert importance().tolist() == [10.0, 1.0]


def test_weighted_error_moves_the_blame_to_channel_1():
    plain = column_squared_error(W, W_RTN)
    assert np.allclose(plain, [0.0035, 0.0018], atol=5e-5)
    assert round(100 * plain[0] / plain.sum()) == 65
    weighted = plain * importance()
    assert np.allclose(weighted, [0.0347, 0.0018], atol=5e-5)
    assert round(weighted.sum(), 4) == 0.0365 == round(weighted_error(W, W_RTN, importance()), 4)
    assert round(100 * weighted[0] / weighted.sum()) == 95
    # The calibration channels are not correlated, so the weighted error is the mean output error.
    assert np.isclose(float(np.mean(CALIB[:, 0] * CALIB[:, 1])), 0.0)
    assert np.isclose(mean_output_error(W, W_RTN), weighted.sum())


def test_weighted_rounding_picks_a_smaller_step():
    r = weighted_rounding(W, importance())
    assert [round(s, 3) for s in STEP_CANDIDATES] == [0.15, 0.155, 0.16, 0.165, 0.17, 0.171]
    assert [round(e, 3) for e in r["errors"]] == [0.073, 0.032, 0.011, 0.009, 0.028, 0.037]
    assert r["step"] == 0.165
    assert r["q"].tolist() == [[3, -7], [5, 2]]
    assert np.allclose(r["w_hat"], [[0.495, -1.155], [0.825, 0.33]])
    assert np.allclose(np.abs(W - r["w_hat"]), [[0.005, 0.045], [0.025, 0.03]])
    assert round(weighted_error(W, r["w_hat"], importance()), 4) == 0.0094
    assert round(mean_abs_error(W, r["w_hat"]), 4) == 0.0262
    assert round(mean_kl(W, r["w_hat"]) * 1000, 3) == 0.309
    assert mean_kl(W, r["w_hat"]) < mean_kl(W, W_RTN)


def test_bit_budget():
    assert SENSITIVITY_AT_4_BITS == [0.064, 0.016, 0.004]
    assert round(total_error([4, 4, 4]), 3) == 0.084
    bits = allocate_bits(12)
    assert bits == [5, 4, 3] and sum(bits) / 3 == 4
    assert np.allclose([layer_error(s, b) for s, b in zip(SENSITIVITY_AT_4_BITS, bits)], [0.016] * 3)
    assert round(total_error(bits), 3) == 0.048
    assert round(total_error(bits) / total_error([4, 4, 4]), 3) == 0.571
    # Values of the plot and of the widget.
    assert allocate_bits(6) == [2, 2, 2] and allocate_bits(9) == [4, 3, 2] and allocate_bits(18) == [7, 6, 5]
    assert round(total_error(allocate_bits(9)), 3) == 0.192 and round(total_error([3, 3, 3]), 3) == 0.336
    for total in BUDGETS:
        assert sum(allocate_bits(total)) == total
        assert total_error(allocate_bits(total)) <= total_error([total / 3] * 3) + 1e-12


def test_qat_with_the_straight_through_estimator():
    history = qat_steps()
    assert QAT_STEP == 0.625 and round(QAT_TARGET, 3) == 1.9
    assert history[0]["w_hat"].tolist() == [0.625, 0.0]
    assert history[0]["output"] == 1.25 and round(history[0]["error"], 3) == -0.65
    assert np.allclose(history[0]["gradient"], [-1.3, -0.65])
    latents = [[0.8, 0.3], [0.810, 0.305], [0.821, 0.310], [0.831, 0.316], [0.832, 0.316]]
    for h, latent in zip(history, latents):
        assert np.allclose(h["latent"], latent, atol=5e-4)
    assert [round(h["error"], 3) for h in history] == [-0.65, -0.65, -0.65, -0.025, -0.025]
    assert history[2]["latent"][1] < QAT_STEP / 2 < history[3]["latent"][1]  # the boundary is 0.3125
    assert history[3]["w_hat"].tolist() == [0.625, 0.625] and history[3]["output"] == 1.875
    assert round(history[0]["error"] / history[3]["error"]) == 26
    assert np.allclose(history[3]["gradient"], [-0.05, -0.025])
    # The stored weight 0.625 is farther from 0.3 than the plain result 0.
    assert round(abs(history[3]["w_hat"][1] - 0.3), 3) == 0.325


def test_ternary_weights():
    t = ternarize(W)
    assert round(t["scale"], 3) == 0.7
    assert np.allclose(t["scaled"], [[0.714, -1.714], [1.143, 0.429]], atol=5e-4)
    assert t["q"].tolist() == [[1, -1], [1, 0]]
    assert np.allclose(t["w_hat"], [[0.7, -0.7], [0.7, 0]])
    assert np.allclose(np.abs(W - t["w_hat"]), [[0.2, 0.5], [0.1, 0.3]])
    assert round(mean_abs_error(W, t["w_hat"]), 3) == 0.275
    assert round(mean_abs_error(W, t["w_hat"]) / mean_abs_error(W, W_RTN), 1) == 9.6
    assert np.allclose(t["w_hat"] @ CALIB_X, [0.7, 1.4])
    assert round(float(np.linalg.norm(W @ CALIB_X - t["w_hat"] @ CALIB_X)), 3) == 1.03
    assert round(mean_kl(W, t["w_hat"]), 3) == 0.072
    assert round(mean_kl(W, t["w_hat"]) / mean_kl(W, W_RTN)) == 108
    assert round(BITS_TERNARY, 3) == 1.585 and round(4 / BITS_TERNARY, 2) == 2.52
    assert round(size_gb(8e9, BITS_TERNARY), 3) == 1.585
    assert size_gb(8e9, 2) == 2.0 and size_gb(8e9, 4) == 4.0


def test_codebook():
    km = kmeans()
    assert np.allclose(PAIRS[:2], W)
    assert np.allclose(km["codebook"], [[0.55, -1.1], [0.85, 0.35], [-0.6, 0.15], [-0.15, -0.8]])
    assert np.allclose(km["history"][1], km["history"][2])  # the second step changes nothing
    assert km["index"].tolist() == [0, 1, 0, 1, 2, 2, 3, 3]
    w_hat = km["pairs_hat"][:2]
    assert np.allclose(np.abs(W - w_hat), [[0.05, 0.1], [0.05, 0.05]])
    assert round(mean_abs_error(W, w_hat), 4) == 0.0625
    assert round(mean_abs_error(PAIRS, km["pairs_hat"]), 3) == 0.069
    assert round(mean_kl(W, w_hat), 3) == 0.004


def test_one_bit_scalar_is_worse_at_the_same_bits():
    sq = sign_quantize(PAIRS)
    assert round(sq["scale"], 3) == 0.569
    assert np.allclose(sq["w_hat"][:2], [[0.569, -0.569], [0.569, 0.569]], atol=5e-4)
    assert round(mean_abs_error(W, sq["w_hat"][:2]), 3) == 0.3
    assert round(mean_abs_error(PAIRS, sq["w_hat"]), 3) == 0.281
    assert round(mean_kl(W, sq["w_hat"][:2]), 3) == 0.058
    assert round(mean_abs_error(W, sq["w_hat"][:2]) / mean_abs_error(W, kmeans()["pairs_hat"][:2])) == 5


def test_codebook_storage():
    assert codebook_bpw(4, 2) == 1.0
    assert codebook_bits(4, 2) == 128
    assert total_bpw(PAIRS.size, 4, 2) == 9.0
    assert REAL_LAYER == 16_777_216
    assert round(total_bpw(REAL_LAYER, 4, 2), 7) == 1.0000076
    assert codebook_bpw(2 ** 16, 8) == 2.0
    assert codebook_bits(2 ** 16, 8) == 8_388_608
    assert codebook_bits(2 ** 16, 8) / REAL_LAYER == 0.5


def test_kl_is_zero_for_the_same_model():
    p = toy_probs(W, CALIB_X)
    assert kl_divergence(p, p) == 0.0
