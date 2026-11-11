"""Each number shown in docs/fundamentals/03-neural-networks.md."""

import numpy as np
import pytest

from examples.fundamentals.neural_networks import (
    EPOCHS, HIDDEN_UNITS, LEARNING_RATE, MLP, MLP_L2, B, U, X, Y,
    adam_first_step, backward, dropout_cases, forward, l1_penalty, l2_penalty, make_data, mse, sgd_step,
    train_mlp,
)
from examples.toy import LOGITS, W


def test_forward_pass():
    f = forward()
    assert X.tolist() == [1, 1]
    assert np.allclose(W @ X, [-0.7, 1.1])
    assert np.allclose(f["z"], [-0.5, 1.0])
    assert np.allclose(f["h"], [0, 1])
    assert np.allclose(f["logits"], LOGITS)
    assert np.allclose(f["probs"], [0.099, 0.242, 0.659], atol=5e-4)
    assert round(f["loss"], 3) == 0.417
    assert W.size + B.size + U.size == 12


def test_activations():
    assert np.allclose(forward(activation="sigmoid")["h"], [0.378, 0.731], atol=5e-4)
    assert np.allclose(forward(activation="tanh")["h"], [-0.462, 0.762], atol=5e-4)
    assert round(forward(activation="sigmoid")["loss"], 3) == 0.511
    assert round(forward(activation="tanh")["loss"], 3) == 0.589


def test_two_linear_layers_are_one():
    assert np.allclose(U @ W, [[0.28, -0.45], [0.65, 0.66], [1.7, 0.36]])
    assert np.allclose(U @ B, [0.07, -0.16, -0.16])
    assert np.allclose(U @ (W @ X + B), (U @ W) @ X + U @ B)
    assert np.allclose(U @ (W @ X + B), [-0.1, 1.15, 1.9])


def test_loss_functions():
    f = forward()
    assert np.allclose((f["probs"] - Y) ** 2, [0.010, 0.059, 0.116], atol=5e-4)
    assert round(mse(f["probs"], Y), 3) == 0.062


def test_backpropagation():
    grads = backward()
    assert np.allclose(grads["logits"], [0.099, 0.242, -0.341], atol=5e-4)
    assert np.allclose(grads["U"], [[0, 0.099], [0, 0.242], [0, -0.341]], atol=5e-4)
    assert np.allclose(grads["h"], [-0.102, -0.430], atol=5e-4)
    assert np.allclose(grads["z"], [0, -0.430], atol=5e-4)
    assert np.allclose(grads["W"], [[0, 0], [-0.430, -0.430]], atol=5e-4)
    assert np.allclose(grads["b"], [0, -0.430], atol=5e-4)


@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
def test_backpropagation_matches_small_moves(activation):
    """Move one weight by a small step and measure the slope of the loss."""
    grads = backward(activation=activation)
    step = 1e-6
    for name, value in (("W", W), ("b", B), ("U", U)):
        for index in np.ndindex(value.shape):
            moved = value.copy()
            moved[index] += step
            args = {"w": W, "b": B, "u": U} | {name.lower(): moved}
            slope = (forward(**args, activation=activation)["loss"] - forward(activation=activation)["loss"]) / step
            assert np.isclose(slope, grads[name][index], atol=1e-5)


def test_sgd_step():
    new = sgd_step()
    assert np.allclose(new["W"], [[0.5, -1.2], [0.843, 0.343]], atol=5e-4)
    assert np.allclose(new["b"], [0.2, -0.057], atol=5e-4)
    assert np.allclose(new["U"], [[0.4, 0.090], [-0.3, 0.976], [0.2, 2.034]], atol=5e-4)
    after = forward(new["W"], new["b"], new["U"])
    assert np.allclose(after["probs"], [0.079, 0.214, 0.707], atol=5e-4)
    assert round(new["loss"], 3) == 0.347


def test_sgd_and_adam_first_change():
    gradient = backward()["logits"]
    assert np.allclose(-LEARNING_RATE * gradient, [-0.010, -0.024, 0.034], atol=5e-4)
    assert np.allclose(adam_first_step(gradient, LEARNING_RATE), [-0.1, -0.1, 0.1], atol=1e-6)


def test_l2_and_l1():
    assert np.isclose(np.sum(W ** 2), 2.42)
    assert np.isclose(l2_penalty(), 0.121)
    assert round(forward()["loss"] + l2_penalty(), 3) == 0.538
    assert np.allclose(0.1 * W, [[0.05, -0.12], [0.08, 0.03]])
    assert np.isclose(1 - LEARNING_RATE * 0.1, 0.99)
    assert np.isclose(l1_penalty(), 0.28)


def test_dropout():
    cases = dropout_cases()
    assert np.allclose(cases["kept"], [0, 2]) and np.allclose(cases["dropped"], [0, 0])
    assert np.allclose(cases["mean"], forward()["h"])
    assert round(cases["loss_kept"], 3) == 0.146
    assert round(cases["loss_dropped"], 3) == 1.099


def test_mlp_gradients_match_small_moves():
    x, y = make_data(8, seed=3)
    model = MLP([2, 4, 2], seed=3)
    grads_w, _ = model.gradients(x, y)
    base, step = model.loss(x, y), 1e-6
    for layer, index in ((0, (1, 0)), (0, (3, 1)), (1, (0, 2))):
        model.weights[layer][index] += step
        slope = (model.loss(x, y) - base) / step
        model.weights[layer][index] -= step
        assert np.isclose(slope, grads_w[layer][index], atol=1e-5)


def test_mlp_overfits():
    run = train_mlp([2, HIDDEN_UNITS, 2])
    best = run["best_epoch"]
    assert run["parameters"] == 82
    assert best == 122
    assert round(run["train_loss"][0], 3) == 0.857 and round(run["val_loss"][0], 3) == 0.747
    assert round(run["train_loss"][best], 3) == 0.412 and round(run["val_loss"][best], 3) == 0.565
    assert round(run["train_loss"][EPOCHS], 3) == 0.052 and round(run["val_loss"][EPOCHS], 3) == 1.496
    assert run["train_accuracy"][best] == 0.8 and run["val_accuracy"][best] == 0.78
    assert run["train_accuracy"][EPOCHS] == 1.0 and run["val_accuracy"][EPOCHS] == 0.685


def test_l2_stops_the_overfit():
    run = train_mlp([2, HIDDEN_UNITS, 2], l2=MLP_L2)
    assert round(run["train_loss"][EPOCHS], 3) == 0.394 and round(run["val_loss"][EPOCHS], 3) == 0.559
    assert run["val_accuracy"][EPOCHS] == 0.78


def test_no_hidden_layer_cannot_learn_the_task():
    run = train_mlp([2, 2])
    assert run["parameters"] == 6
    assert run["val_accuracy"][EPOCHS] == 0.565
    assert round(run["val_loss"][EPOCHS], 3) == 0.750


def test_real_network_size():
    assert 784 * 16 + 16 + 16 * 16 + 16 + 16 * 10 + 10 == 13_002
