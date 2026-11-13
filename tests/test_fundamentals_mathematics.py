"""Each number shown in content/fundamentals/01-math.md."""

import numpy as np

from examples.fundamentals.mathematics import (
    CAT, DRAWS, GRAM, IS_SAT, LLAMA3_EMBEDDING, LLAMA3_VOCAB, SAT, THE,
    cosine, cross_entropy, determinant_2x2, expectation, finite_difference, gradient_descent,
    loss, loss_derivative, numeric_gradient, variance,
)
from examples.toy import LOGITS, W, softmax


def test_vectors_and_dot_products():
    assert np.allclose(THE + CAT, SAT)
    assert (SAT @ CAT, SAT @ SAT, THE @ CAT, THE @ THE) == (1.0, 2.0, 0.0, 1.0)
    assert round(float(np.linalg.norm(SAT)), 3) == 1.414
    assert round(cosine(SAT, THE), 3) == 0.707
    assert round(cosine(SAT, SAT), 3) == 1.0 and round(cosine(THE, THE), 3) == 1.0


def test_matrix_times_vector():
    assert np.allclose(W @ THE, [0.5, 0.8])
    assert np.allclose(W @ CAT, [-1.2, 0.3])
    assert np.allclose(W @ SAT, [-0.7, 1.1])
    assert np.allclose(W @ (THE + CAT), W @ THE + W @ CAT)


def test_determinant_and_inverse():
    assert round(determinant_2x2(W), 3) == 1.11
    assert np.isclose(determinant_2x2(W), np.linalg.det(W))
    assert np.allclose(np.linalg.inv(W) @ np.array([-0.7, 1.1]), [1, 1])


def test_eigenvalues():
    assert np.allclose(GRAM, [[2, 1], [1, 2]])
    assert np.allclose(GRAM @ np.array([1.0, 1.0]), [3, 3])
    assert np.allclose(GRAM @ np.array([1.0, -1.0]), [1, -1])
    assert np.allclose(np.linalg.eigvalsh(GRAM), [1, 3])
    complex_pair = np.linalg.eigvals(W)
    assert np.allclose(complex_pair.real, 0.4)
    assert np.allclose(np.abs(complex_pair.imag), 0.975, atol=5e-4)


def test_real_matrix_size():
    assert LLAMA3_VOCAB * LLAMA3_EMBEDDING == 525_336_576


def test_derivative():
    assert loss(0.0) == 4.0
    assert round(loss(0.001), 6) == 3.996001
    assert round(finite_difference(loss, 0.0, 0.001), 3) == -3.999
    assert loss_derivative(0.0) == -4.0


def test_gradient_descent_converges():
    rows = gradient_descent(0.0, 0.1, 4)
    assert np.allclose([row["w"] for row in rows], [0, 0.4, 0.72, 0.976, 1.181], atol=5e-4)
    assert np.allclose([row["loss"] for row in rows], [4, 2.56, 1.638, 1.049, 0.671], atol=5e-4)
    assert np.allclose([row["gradient"] for row in rows], [-4, -3.2, -2.56, -2.048, -1.638], atol=5e-4)


def test_gradient_descent_diverges():
    rows = gradient_descent(0.0, 1.1, 3)
    assert np.allclose([row["w"] for row in rows], [0, 4.4, -0.88, 5.456], atol=5e-4)
    assert np.allclose([row["loss"] for row in rows], [4, 5.76, 8.294, 11.944], atol=5e-4)
    distances = [abs(row["w"] - 2) for row in rows]
    assert np.allclose(np.divide(distances[1:], distances[:-1]), 1.2)


def test_gradient_of_the_loss():
    moved = LOGITS + np.array([0.0, 0.0, 0.001])
    assert round(cross_entropy(LOGITS, 2), 6) == 0.41703
    assert round(cross_entropy(moved, 2), 6) == 0.416689
    assert round((cross_entropy(moved, 2) - cross_entropy(LOGITS, 2)) / 0.001, 3) == -0.341
    numeric = numeric_gradient(lambda z: cross_entropy(z, 2), LOGITS)
    assert np.allclose(numeric, [0.099, 0.242, -0.341], atol=5e-4)
    assert np.allclose(numeric, softmax(LOGITS) - IS_SAT, atol=1e-5)


def test_probabilities():
    assert np.allclose(np.exp(LOGITS), [1.105, 2.718, 7.389], atol=5e-4)
    assert np.allclose(softmax(LOGITS), [0.099, 0.242, 0.659], atol=5e-4)
    assert np.isclose(softmax(LOGITS).sum(), 1.0)


def test_expectation_and_variance():
    probs = softmax(LOGITS)
    mean = expectation(probs, IS_SAT)
    var = variance(probs, IS_SAT)
    assert round(mean, 3) == 0.659
    assert np.allclose((IS_SAT - mean) ** 2, [0.4343, 0.4343, 0.1163], atol=5e-5)
    assert round(0.099 * 0.4343 + 0.242 * 0.4343 + 0.659 * 0.1163, 3) == 0.225  # the sum as shown on the page
    assert round(var, 3) == 0.225
    assert round(float(np.sqrt(var)), 3) == 0.474
    assert round(DRAWS * mean, 1) == 65.9
    assert round(float(np.sqrt(DRAWS * var)), 2) == 4.74
    assert round(1.96 * float(np.sqrt(var / DRAWS)), 3) == 0.093


def test_cross_entropy():
    assert round(cross_entropy(LOGITS, 2), 3) == 0.417
    assert round(cross_entropy(LOGITS, 0), 3) == 2.317
    assert round(float(softmax(LOGITS)[0]), 4) == 0.0986 and round(-np.log(0.0986), 3) == 2.317
    assert round(float(np.exp(cross_entropy(LOGITS, 2))), 3) == 1.517
