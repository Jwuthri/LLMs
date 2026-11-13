"""Each number shown in content/scientist/09-diffusion.md."""

import numpy as np

from examples.scientist.diffusion import (
    ALPHA_BAR, EPSILON, JOINT, MASK, MASK_RATE, MASKED_POSITIONS, OTHER_STARTS, SENTENCE, START, TEXT_LENGTH, X0,
    add_noise, call_counts, denoise, error_probability, marginals, mask_tokens, masked_loss, noise_loss,
    one_step_distribution, predict, reverse_path, signal_to_noise, step_betas, two_step_distribution, unmask,
)
from examples.toy import LOGITS, E, softmax

ATOL = 5e-4


def test_forward_process():
    assert np.allclose(X0, [1, 1])
    assert np.allclose(step_betas(ALPHA_BAR), [0.1, 0.444, 0.8], atol=ATOL)
    assert np.allclose(np.sqrt(ALPHA_BAR), [0.949, 0.707, 0.316], atol=ATOL)
    assert np.allclose(np.sqrt(1 - ALPHA_BAR), [0.316, 0.707, 0.949], atol=ATOL)
    noisy = [add_noise(X0, alpha_bar, EPSILON) for alpha_bar in ALPHA_BAR]
    assert np.allclose(noisy, [[1.138, 0.696], [1.131, 0.141], [0.885, -0.443]], atol=ATOL)
    assert np.allclose([signal_to_noise(a) for a in ALPHA_BAR], [9.0, 1.0, 0.111], atol=ATOL)


def test_best_denoiser_at_two_noise_levels():
    high = denoise(add_noise(X0, ALPHA_BAR[2], EPSILON), ALPHA_BAR[2])
    assert np.allclose(high["weights"], [0.410, 0.257, 0.332], atol=ATOL)
    assert np.allclose(high["x0_hat"], [0.743, 0.590], atol=ATOL)
    assert np.allclose(high["noise_hat"], [0.686, -0.663], atol=ATOL)
    assert round(noise_loss(EPSILON, high["noise_hat"]), 3) == 0.013

    low = denoise(add_noise(X0, ALPHA_BAR[0], EPSILON), ALPHA_BAR[0])
    assert np.allclose(low["weights"], [0.109, 0.002, 0.889], atol=ATOL)
    assert np.allclose(low["x0_hat"], [0.998, 0.891], atol=ATOL)
    assert np.allclose(low["noise_hat"], [0.605, -0.473], atol=ATOL)
    assert round(noise_loss(EPSILON, low["noise_hat"]), 3) == 0.053


def test_denoiser_is_the_posterior_mean():
    x_t = add_noise(X0, 0.5, EPSILON)
    estimate = denoise(x_t, 0.5)
    assert np.isclose(estimate["weights"].sum(), 1.0)
    assert np.allclose(estimate["x0_hat"], estimate["weights"] @ E)
    # The predicted noise and the predicted data point give back the noisy point.
    assert np.allclose(np.sqrt(0.5) * estimate["x0_hat"] + np.sqrt(0.5) * estimate["noise_hat"], x_t)


def test_reverse_process_arrives_near_a_data_point():
    records = reverse_path(START)
    assert [record["t"] for record in records] == [3, 2, 1]
    assert np.allclose(records[0]["weights"], [0.452, 0.267, 0.281], atol=ATOL)
    assert np.allclose(records[0]["x0_hat"], [0.733, 0.548], atol=ATOL)
    assert np.allclose(records[0]["noise_hat"], [0.072, -1.447], atol=ATOL)
    assert np.allclose(records[0]["x_next"], [0.569, -0.636], atol=ATOL)
    assert np.allclose(records[1]["weights"], [0.700, 0.127, 0.173], atol=ATOL)
    assert np.allclose(records[1]["x_next"], [0.806, -0.095], atol=ATOL)
    assert np.allclose(records[2]["weights"], [0.995, 0.000, 0.004], atol=ATOL)
    assert np.allclose(records[2]["x_next"], [1.000, 0.005], atol=ATOL)
    assert np.linalg.norm(records[2]["x_next"] - E[0]) < 0.01  # near `the`


def test_other_starts_arrive_at_other_data_points():
    ends = [reverse_path(start)[-1]["x_next"] for start in OTHER_STARTS]
    assert np.allclose(ends, [[0.010, 1.000], [1.000, 0.989]], atol=ATOL)  # near `cat`, near `sat`


def test_masked_forward_and_loss():
    assert np.isclose(MASK_RATE * len(SENTENCE), 2)
    masked = mask_tokens(SENTENCE, MASKED_POSITIONS)
    assert masked == ("the", MASK, MASK)
    probs = predict(masked)
    assert np.allclose(probs[1], [0.197, 0.536, 0.266], atol=ATOL)
    assert np.allclose(probs[2], softmax(LOGITS))
    assert np.allclose(probs[2], [0.099, 0.242, 0.659], atol=ATOL)
    loss = masked_loss(SENTENCE, masked, MASK_RATE)
    assert round(loss["terms"][1], 3) == 0.623
    assert round(loss["terms"][2], 3) == 0.417
    assert round(loss["sum"], 3) == 1.040
    assert round(loss["weighted"], 3) == 1.560


def test_unmasking_in_three_steps():
    records = unmask(3)
    assert np.allclose(list(records[0]["confidence"].values()), [0.787, 0.412, 0.416], atol=ATOL)
    assert np.allclose(list(records[1]["confidence"].values()), [0.536, 0.659], atol=ATOL)
    assert np.allclose(list(records[2]["confidence"].values()), [0.883], atol=ATOL)
    assert [record["chosen"] for record in records] == [[0], [2], [1]]  # not from left to right
    assert [record["result"] for record in records] == [
        ("the", MASK, MASK), ("the", MASK, "sat"), ("the", "cat", "sat"),
    ]


def test_unmasking_in_two_steps_and_in_one_step():
    two = unmask(2)
    assert [record["chosen"] for record in two] == [[0], [1, 2]]
    assert two[-1]["result"] == ("the", "cat", "sat")
    one = unmask(1)
    assert one[0]["chosen"] == [0, 1, 2]
    assert one[0]["result"] == ("the", "cat", "cat")  # two positions chose `cat` without seeing each other


def test_infilling():
    assert np.allclose(predict(("the", MASK, "sat"))[1], [0.044, 0.883, 0.073], atol=ATOL)


def test_call_counts():
    assert TEXT_LENGTH == 12
    for steps, positions, per_call in [(4, 48, 3), (2, 24, 6), (1, 12, 12)]:
        counts = call_counts(TEXT_LENGTH, steps)
        assert counts["ar_calls"] == 12
        assert counts["ar_positions_cached"] == 12
        assert counts["ar_positions_no_cache"] == 78
        assert counts["diffusion_calls"] == steps
        assert counts["diffusion_positions"] == positions
        assert counts["tokens_per_call"] == per_call


def test_independence_error():
    first, second = marginals(JOINT)
    assert first == {"the": 0.5, "a": 0.5} and second == {"cat": 0.5, "dog": 0.5}
    one_step = one_step_distribution(JOINT)
    assert one_step == {("the", "cat"): 0.25, ("the", "dog"): 0.25, ("a", "cat"): 0.25, ("a", "dog"): 0.25}
    assert error_probability(one_step, JOINT) == 0.5
    two_step = two_step_distribution(JOINT)
    assert two_step == JOINT
    assert error_probability(two_step, JOINT) == 0
