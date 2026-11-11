"""Each number shown in docs/scientist/07-quantization.md."""

import numpy as np

from examples.scientist.quantization import (
    ACT_MAX, BIT_RANGE, BITS_PER_WEIGHT, CALIB_X, PARAMS, Q4_K_BPW, Q4_K_M_BPW, Q4_K_M_GIB, Q6_K_BPW,
    absmax_quantize, awq_toy, gib_to_gb, gptq_toy, grid_step, levels, model_size_gb, offload_gb,
    output_error, round_to_nearest, smoothquant, zeropoint_quantize,
)
from examples.toy import W


def test_precision_levels():
    sizes = {name: model_size_gb(PARAMS, bits) for name, bits in BITS_PER_WEIGHT.items()}
    assert sizes == {"FP32": 32.0, "FP16": 16.0, "INT8": 8.0, "4-bit": 4.0}
    assert [bits / 8 for bits in BITS_PER_WEIGHT.values()] == [4, 2, 1, 0.5]


def test_absmax_int8():
    r = absmax_quantize(W, 8)
    assert round(r["scale"], 3) == 105.833
    assert np.allclose(r["scaled"], [[52.917, -127], [84.667, 31.75]], atol=5e-4)
    assert r["q"].tolist() == [[53, -127], [85, 32]]
    assert np.allclose(r["dequant"], [[0.5008, -1.2], [0.8031, 0.3024]], atol=5e-5)
    assert np.allclose(r["error"], [[0.0008, 0], [0.0031, 0.0024]], atol=5e-5)
    assert round(r["mae"], 4) == 0.0016


def test_zeropoint_int8():
    r = zeropoint_quantize(W, 8)
    assert round(r["scale"], 3) == 127.5 and r["zero_point"] == 25
    assert np.allclose(r["scaled"], [[88.75, -128], [127, 63.25]])
    assert r["q"].tolist() == [[89, -128], [127, 63]]
    assert np.allclose(r["dequant"], [[0.502, -1.2], [0.8, 0.298]], atol=5e-5)
    assert np.allclose(r["error"], [[0.002, 0], [0, 0.002]], atol=5e-5)
    assert round(r["mae"], 4) == 0.0010


def test_four_bits_have_a_larger_error():
    a = absmax_quantize(W, 4)
    assert round(a["scale"], 3) == 5.833
    assert a["q"].tolist() == [[3, -7], [5, 2]]
    assert np.allclose(a["dequant"], [[0.514, -1.2], [0.857, 0.343]], atol=5e-4)
    assert np.allclose(a["error"], [[0.014, 0], [0.057, 0.043]], atol=5e-4)
    assert round(a["mae"], 4) == 0.0286
    assert round(a["mae"] / absmax_quantize(W, 8)["mae"]) == 18

    z = zeropoint_quantize(W, 4)
    assert z["scale"] == 7.5 and z["zero_point"] == 1
    assert z["q"].tolist() == [[5, -8], [7, 3]]
    assert np.allclose(z["dequant"], [[0.533, -1.2], [0.8, 0.267]], atol=5e-4)
    assert round(z["mae"], 4) == 0.0167


def test_plot_values_and_widget_rounding():
    assert round(absmax_quantize(W, 2)["mae"], 3) == 0.3
    assert round(zeropoint_quantize(W, 2)["mae"], 3) == 0.183
    # The widget rounds with Math.round (half up); NumPy rounds half to even. No value is a tie.
    for bits in BIT_RANGE:
        for fn in (absmax_quantize, zeropoint_quantize):
            scaled = fn(W, bits)["scaled"]
            assert np.all(np.abs(np.abs(scaled - np.floor(scaled)) - 0.5) > 1e-6)


def test_gguf_bits_per_weight():
    assert Q4_K_BPW == 4.5 and Q6_K_BPW == 6.5625
    assert model_size_gb(PARAMS, Q4_K_BPW) == 4.5
    assert round(model_size_gb(PARAMS, Q4_K_M_BPW), 2) == 4.89
    assert round(gib_to_gb(Q4_K_M_GIB), 2) == 4.92
    assert round(offload_gb(model_size_gb(PARAMS, Q4_K_M_BPW), 16, 32), 2) == 2.45
    assert round(100 * Q4_K_M_BPW / BITS_PER_WEIGHT["FP16"]) == 31


def test_round_to_nearest_output_error():
    assert round(grid_step(W, 4), 3) == 0.171
    assert round(grid_step(W, 4) / 2, 3) == 0.086  # the largest move of one weight
    assert np.allclose(W @ CALIB_X, [-0.2, 1.9])
    rtn = round_to_nearest(W)
    assert np.allclose(rtn @ CALIB_X, [-0.171, 2.057], atol=5e-4)
    assert np.allclose(W @ CALIB_X - rtn @ CALIB_X, [-0.029, -0.157], atol=5e-4)
    assert round(output_error(W, rtn, CALIB_X), 3) == 0.160
    assert round((rtn[1, 0] - W[1, 0]) * CALIB_X[0], 3) == 0.114  # output change of row 2 after column 1


def test_gptq_compensation_lowers_the_output_error():
    g = gptq_toy(W, CALIB_X)
    assert g["ratio"] == 2.0
    assert np.allclose(g["error1"], [-0.014, -0.057], atol=5e-4)
    assert np.allclose(g["col2_moved"], [-1.229, 0.186], atol=5e-4)
    assert np.allclose(g["w_hat"], [[0.514, -1.2], [0.857, 0.171]], atol=5e-4)
    assert np.allclose(g["w_hat"] @ CALIB_X, [-0.171, 1.886], atol=5e-4)
    assert round(output_error(W, g["w_hat"], CALIB_X), 3) == 0.032


def test_gptq_toy_equals_the_hessian_formula():
    calib = np.array([[2.0, 1.0], [1.0, 3.0]])  # 2 inputs, so H = calib^T calib has an inverse
    h_inv = np.linalg.inv(calib.T @ calib)
    assert np.isclose(gptq_toy(W, calib)["ratio"], -h_inv[1, 0] / h_inv[0, 0])


def test_awq_scaling_lowers_the_output_error():
    errors = {alpha: awq_toy(W, CALIB_X, alpha)["error"] for alpha in (0.0, 0.5, 1.0)}
    assert [round(errors[a], 3) for a in (0.0, 0.5, 1.0)] == [0.160, 0.143, 0.077]
    best = awq_toy(W, CALIB_X, 1.0)
    assert np.allclose(best["w_scaled"], [[1.0, -1.2], [1.6, 0.3]])
    assert round(best["step"], 3) == 0.229
    assert np.allclose(best["w_hat"], [[0.914, -1.143], [1.6, 0.229]], atol=5e-4)
    assert np.allclose(best["x_scaled"], [1, 1])
    assert np.allclose(best["w_hat"] @ best["x_scaled"], [-0.229, 1.829], atol=5e-4)


def test_smoothquant():
    sq = smoothquant(ACT_MAX, W)
    assert np.allclose(sq["w_max"], [0.8, 1.2])
    assert np.allclose(sq["s"], [4.472, 0.913], atol=5e-4)
    assert np.allclose(sq["act_max"], [3.578, 1.095], atol=5e-4)
    assert np.allclose(sq["w_max_after"], [3.578, 1.095], atol=5e-4)
    assert np.allclose(sq["w"], [[2.236, -1.095], [3.578, 0.274]], atol=5e-4)
    assert np.allclose(levels(ACT_MAX), [127, 7.9], atol=0.05)
    assert np.allclose(levels(sq["w_max"]), [84.7, 127], atol=0.05)
    assert np.allclose(levels(sq["act_max"]), [127, 38.9], atol=0.05)
    assert np.allclose(levels(sq["w_max_after"]), [127, 38.9], atol=0.05)
    assert np.allclose(W @ ACT_MAX, [6.8, 13.1])
    assert np.allclose(sq["w"] @ (ACT_MAX / sq["s"]), W @ ACT_MAX)
