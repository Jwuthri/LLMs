"""Numbers and plots for docs/fundamentals/03-neural-networks.md.

Run: uv run python -m examples.fundamentals.neural_networks
"""

import numpy as np

from examples.plotstyle import save_light_dark
from examples.toy import LOGITS, VOCAB, E, W, softmax

PAGE = "neural-networks"

# --- The toy network: x -> z = W x + b -> h = activation(z) -> logits = U h -> softmax -------

X = E[0] + E[1]                  # input: the sum of the embeddings of "the" and "cat" = [1, 1]
B = np.array([0.2, -0.1])        # bias of the hidden layer (an assumption of the page)
U = np.array([[0.4, 0.1],        # output matrix (an assumption of the page): with the ReLU
              [-0.3, 1.0],       # activation, U h is the toy logits [0.1, 1.0, 2.0]
              [0.2, 2.0]])
TARGET = VOCAB.index("sat")
Y = np.eye(len(VOCAB))[TARGET]   # one-hot vector of the correct token: [0, 0, 1]
LEARNING_RATE = 0.1

ACTIVATIONS = {
    "relu": (lambda z: np.maximum(z, 0.0), lambda z: (z > 0).astype(float)),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z: np.exp(-z) / (1 + np.exp(-z)) ** 2),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
}


def forward(w: np.ndarray = W, b: np.ndarray = B, u: np.ndarray = U, activation: str = "relu") -> dict:
    """Forward pass of the toy network. Returns each intermediate value."""
    z = w @ X + b
    h = ACTIVATIONS[activation][0](z)
    logits = u @ h
    probs = softmax(logits)
    return {"z": z, "h": h, "logits": logits, "probs": probs, "loss": float(-np.log(probs[TARGET]))}


def backward(w: np.ndarray = W, b: np.ndarray = B, u: np.ndarray = U, activation: str = "relu") -> dict:
    """Backpropagation: the gradient of the loss for each value, from the logits back to W."""
    f = forward(w, b, u, activation)
    d_logits = f["probs"] - Y                           # softmax + cross-entropy
    d_u = np.outer(d_logits, f["h"])                    # logits = U h
    d_h = u.T @ d_logits
    d_z = d_h * ACTIVATIONS[activation][1](f["z"])      # h = activation(z)
    d_w = np.outer(d_z, X)                              # z = W x + b
    return {"logits": d_logits, "U": d_u, "h": d_h, "z": d_z, "W": d_w, "b": d_z}


def sgd_step(learning_rate: float = LEARNING_RATE) -> dict:
    """One update `weight <- weight - learning_rate * gradient` of W, b and U. Returns the new values."""
    grads = backward()
    w, b, u = W - learning_rate * grads["W"], B - learning_rate * grads["b"], U - learning_rate * grads["U"]
    return {"W": w, "b": b, "U": u, "loss": forward(w, b, u)["loss"]}


def mse(probs: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error between the probabilities and the one-hot target."""
    return float(np.mean((probs - target) ** 2))


def adam_first_step(gradient: np.ndarray, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999,
                    eps: float = 1e-8) -> np.ndarray:
    """The change of the weights in the first Adam step (the running means start at 0)."""
    m = (1 - beta1) * gradient
    v = (1 - beta2) * gradient ** 2
    m_hat, v_hat = m / (1 - beta1), v / (1 - beta2)
    return -learning_rate * m_hat / (np.sqrt(v_hat) + eps)


# --- Overfitting: L2 penalty and dropout on the toy network --------------------------------

L2_LAMBDA = 0.1
DROPOUT_RATE = 0.5


def l2_penalty(w: np.ndarray = W, strength: float = L2_LAMBDA) -> float:
    return float(strength / 2 * np.sum(w ** 2))


def l1_penalty(w: np.ndarray = W, strength: float = L2_LAMBDA) -> float:
    return float(strength * np.sum(np.abs(w)))


def dropout_cases(rate: float = DROPOUT_RATE) -> dict:
    """Hidden unit 2 (h = 1.0) with dropout: the two possible cases and their loss."""
    h = forward()["h"]
    kept = h / (1 - rate)
    dropped = h * np.array([1.0, 0.0])
    loss_of = lambda hidden: float(-np.log(softmax(U @ hidden)[TARGET]))  # noqa: E731
    return {
        "kept": kept, "dropped": dropped, "mean": rate * dropped + (1 - rate) * kept,
        "loss_kept": loss_of(kept), "loss_dropped": loss_of(dropped),
    }


# --- A multilayer perceptron (MLP) in NumPy -------------------------------------------------

class MLP:
    """Fully connected network: tanh hidden layers, softmax output, cross-entropy loss."""

    def __init__(self, sizes: list[int], seed: int):
        rng = np.random.default_rng(seed)
        pairs = list(zip(sizes, sizes[1:]))
        self.weights = [rng.normal(0, np.sqrt(1 / n_in), (n_out, n_in)) for n_in, n_out in pairs]
        self.biases = [np.zeros(n_out) for _, n_out in pairs]

    def forward(self, x: np.ndarray) -> list[np.ndarray]:
        """Return the output of each layer. The last one holds the probabilities."""
        outputs = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = outputs[-1] @ w.T + b
            outputs.append(softmax(z) if i == len(self.weights) - 1 else np.tanh(z))
        return outputs

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        probs = self.forward(x)[-1]
        return float(-np.log(probs[np.arange(len(y)), y]).mean())

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((self.forward(x)[-1].argmax(axis=1) == y).mean())

    def gradients(self, x: np.ndarray, y: np.ndarray, l2: float = 0.0) -> tuple[list, list]:
        outputs = self.forward(x)
        delta = outputs[-1].copy()
        delta[np.arange(len(y)), y] -= 1                 # p - y for each example
        delta /= len(y)                                  # the loss is a mean over the examples
        grads_w, grads_b = [], []
        for i in reversed(range(len(self.weights))):
            grads_w.insert(0, delta.T @ outputs[i] + l2 * self.weights[i])
            grads_b.insert(0, delta.sum(axis=0))
            if i > 0:
                delta = (delta @ self.weights[i]) * (1 - outputs[i] ** 2)   # tanh'(z) = 1 - h^2
        return grads_w, grads_b

    def step(self, x: np.ndarray, y: np.ndarray, learning_rate: float, l2: float = 0.0) -> None:
        grads_w, grads_b = self.gradients(x, y, l2)
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]


TRAIN_POINTS, VALIDATION_POINTS = 40, 200
LABEL_NOISE = 0.15
HIDDEN_UNITS = 16
EPOCHS = 3000
MLP_LEARNING_RATE = 0.5
MLP_L2 = 0.01


def make_data(points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two inputs between -1 and 1. Class 1 if they have the same sign. Some labels are flipped."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(points, 2))
    y = (x[:, 0] * x[:, 1] > 0).astype(int)
    flipped = rng.random(points) < LABEL_NOISE
    return x, np.where(flipped, 1 - y, y)


def train_mlp(sizes: list[int], l2: float = 0.0, epochs: int = EPOCHS) -> dict:
    """Full-batch gradient descent. Row k of each history is the value before update k."""
    x_train, y_train = make_data(TRAIN_POINTS, seed=0)
    x_val, y_val = make_data(VALIDATION_POINTS, seed=1)
    model = MLP(sizes, seed=0)
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    for _ in range(epochs + 1):
        history["train_loss"].append(model.loss(x_train, y_train))
        history["val_loss"].append(model.loss(x_val, y_val))
        history["train_accuracy"].append(model.accuracy(x_train, y_train))
        history["val_accuracy"].append(model.accuracy(x_val, y_val))
        model.step(x_train, y_train, MLP_LEARNING_RATE, l2)
    history = {name: np.array(values) for name, values in history.items()}
    history["best_epoch"] = int(history["val_loss"].argmin())
    history["parameters"] = sum(w.size + b.size for w, b in zip(model.weights, model.biases))
    return history


# --- Plots ---------------------------------------------------------------------------------

def make_figures() -> None:
    def draw_activations(ax, theme):
        zs = np.linspace(-3, 3, 241)
        z_toy = forward()["z"]
        for i, name in enumerate(("relu", "sigmoid", "tanh")):
            f = ACTIVATIONS[name][0]
            label = {"relu": "ReLU", "sigmoid": "sigmoid", "tanh": "tanh"}[name]
            ax.plot(zs, f(zs), color=theme["series"][i], label=label)
            ax.scatter(z_toy, f(z_toy), s=28, color=theme["series"][i], zorder=3)
        for z in z_toy:
            ax.axvline(z, color=theme["muted"], linewidth=0.8, linestyle=":")
            ax.text(z + 0.06, -0.93, f"z = {z:g}", fontsize=8, color=theme["ink2"])
        ax.set_xlabel("Input z of the activation")
        ax.set_ylabel("Output h")
        ax.set_ylim(-1.1, 3.1)
        ax.legend(loc="upper left")

    def draw_overfitting(ax, theme):
        plain = train_mlp([2, HIDDEN_UNITS, 2])
        with_l2 = train_mlp([2, HIDDEN_UNITS, 2], l2=MLP_L2)
        epochs = np.arange(1, EPOCHS + 1)
        ax.plot(epochs, plain["train_loss"][1:], color=theme["series"][0], label="train loss")
        ax.plot(epochs, plain["val_loss"][1:], color=theme["series"][1], label="validation loss")
        ax.plot(epochs, with_l2["val_loss"][1:], color=theme["series"][2], label="validation loss with L2")
        best = plain["best_epoch"]
        ax.axvline(best, color=theme["muted"], linewidth=0.8, linestyle=":")
        ax.text(best * 1.12, 1.45, f"lowest validation loss\nepoch {best}", fontsize=8, color=theme["ink2"])
        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_ylim(0, 1.7)
        ax.legend(loc="upper left")

    save_light_dark(draw_activations, PAGE, "activations")
    save_light_dark(draw_overfitting, PAGE, "overfitting")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    print("--- fundamentals")
    print("x:", X, " W x:", W @ X)
    for name in ACTIVATIONS:
        f = forward(activation=name)
        print(f"{name}: z={f['z']} h={f['h']} logits={f['logits']} probs={f['probs']} loss={f['loss']:.4f}")
    print("U W:", U @ W, " U b:", U @ B, " (U W) x + U b:", U @ W @ X + U @ B, sep="\n")
    print("parameters:", W.size + B.size + U.size)

    print("--- training and optimization")
    f = forward()
    print("toy logits reproduced:", np.allclose(f["logits"], LOGITS))
    print("cross-entropy:", f["loss"], " MSE:", mse(f["probs"], Y), " squared errors:", (f["probs"] - Y) ** 2)
    for name, value in backward().items():
        print(f"dL/d{name}:", value, sep="\n")
    new = sgd_step()
    for name in ("W", "b", "U"):
        print(f"new {name}:", new[name], sep="\n")
    print("new forward:", forward(new["W"], new["b"], new["U"]))
    print("loss before -> after:", f["loss"], "->", new["loss"])
    print("SGD change of U column 2:", -LEARNING_RATE * backward()["logits"])
    print("Adam first change of U column 2:", adam_first_step(backward()["logits"], LEARNING_RATE))

    print("--- overfitting")
    print("sum of W^2:", np.sum(W ** 2), " L2 penalty:", l2_penalty(), " loss + penalty:", f["loss"] + l2_penalty())
    print("gradient of the L2 penalty:", L2_LAMBDA * W, " shrink factor:", 1 - LEARNING_RATE * L2_LAMBDA)
    print("sum of |W|:", np.sum(np.abs(W)), " L1 penalty:", l1_penalty())
    print("dropout:", dropout_cases())

    print("--- MLP")
    x_train, y_train = make_data(TRAIN_POINTS, seed=0)
    print("first train points:", x_train[:3], y_train[:3], " share of class 1:", y_train.mean())
    for label, sizes, l2 in (("linear", [2, 2], 0.0), ("MLP", [2, HIDDEN_UNITS, 2], 0.0),
                             ("MLP + L2", [2, HIDDEN_UNITS, 2], MLP_L2)):
        run = train_mlp(sizes, l2)
        best = run["best_epoch"]
        print(f"{label}: parameters {run['parameters']}, best epoch {best}")
        for epoch in (0, best, EPOCHS):
            print(f"  epoch {epoch}: train loss {run['train_loss'][epoch]:.4f} val loss {run['val_loss'][epoch]:.4f}"
                  f" train acc {run['train_accuracy'][epoch]:.4f} val acc {run['val_accuracy'][epoch]:.4f}")
    print("3Blue1Brown digit network parameters:", 784 * 16 + 16 + 16 * 16 + 16 + 16 * 10 + 10)
    make_figures()
