# 3. Neural networks

## Core idea

A neural network is a chain of layers. Each layer multiplies its input by a matrix, adds a number, and bends the result with a simple curve.
To train it, you show an example, measure the error, and move each weight a small step in the direction that makes the error smaller.

## The example

This page uses the [toy example](../toy-example.md), with a network that is small enough to calculate by hand.

| Item | Value |
|---|---|
| Input $x$ | [1, 1], the sum of the embeddings of `the` [1, 0] and `cat` [0, 1] |
| Hidden layer | $W = \begin{bmatrix} 0.5 & -1.2 \\ 0.8 & 0.3 \end{bmatrix}$ and $b = [0.2, -0.1]$ (assume this $b$) |
| Output layer | $U = \begin{bmatrix} 0.4 & 0.1 \\ -0.3 & 1.0 \\ 0.2 & 2.0 \end{bmatrix}$ (assume this $U$) |
| Correct next token | `sat`, written as $y = [0, 0, 1]$ |
| Learning rate | 0.1 |

With these values the network gives the toy logits [0.1, 1.0, 2.0]. A **logit** is the raw score of one token.

## 1. Fundamentals

### Setup

A **unit** (or neuron) calculates a weighted sum of its inputs, adds a **bias**, and applies an **activation function**.
A **layer** is a group of units that read the same input. In matrix form: $h = f(Wx + b)$.
Our network has 2 inputs, 2 hidden units and 3 outputs (one for each token). It has 4 + 2 + 6 = 12 **parameters**: the numbers that training changes.

### Steps

1. **Calculate the weighted sums and add the bias.**

    $$z = Wx + b = \begin{bmatrix} 0.5 \cdot 1 - 1.2 \cdot 1 \\ 0.8 \cdot 1 + 0.3 \cdot 1 \end{bmatrix} + \begin{bmatrix} 0.2 \\ -0.1 \end{bmatrix} = \begin{bmatrix} -0.5 \\ 1.0 \end{bmatrix}$$

    **Why:** each unit measures how well the input agrees with its row of weights (a dot product, see [Mathematics](01-math.md)). The bias moves the level at which the unit starts to react.

2. **Apply the activation function.** Three common functions are **ReLU**, $\max(0, z)$, **sigmoid**, $1 / (1 + e^{-z})$, and **tanh**, $(e^{z} - e^{-z}) / (e^{z} + e^{-z})$.

    | Function | $z = -0.5$ | $z = 1.0$ |
    |---|---|---|
    | ReLU | 0 | 1 |
    | Sigmoid | 0.378 | 0.731 |
    | Tanh | −0.462 | 0.762 |

    We use ReLU: $h = [0, 1]$. Unit 1 is off.

    **Why:** without this step, two layers are the same as one layer: $U(Wx + b) = (UW)x + Ub$, which is one matrix and one bias. Here the two forms give the same result, [−0.1, 1.15, 1.9]. The bend of the activation lets a network draw curves, not only straight lines.

3. **Calculate the output layer.** $\text{logits} = Uh = 0 \cdot [0.4, -0.3, 0.2] + 1 \cdot [0.1, 1.0, 2.0] = [0.1, 1.0, 2.0]$.

    **Why:** the last layer gives one score for each possible answer. It has no activation, because softmax comes next.

4. **Calculate the probabilities and the loss.** **Softmax** turns the logits into probabilities: $p = [0.099, 0.242, 0.659]$. The **cross-entropy loss** is the negative logarithm of the probability of the correct token: $-\ln(p_{\text{sat}}) = -\ln(0.659) = 0.417$. Steps 1 to 4 are the **forward pass**.

    **Why:** training needs one number that says how wrong the network is. All the next steps try to make this number smaller.

<figure class="plot">
<img src="/figures/neural-networks/activations.svg" alt="The three activation functions from z = -3 to 3, with points at z = -0.5 and z = 1. ReLU: 0 and 1. Sigmoid: 0.378 and 0.731. Tanh: -0.462 and 0.762.">
<figcaption>Sigmoid and tanh become flat for large inputs. ReLU is flat only below 0.</figcaption>
</figure>

<div class="key-point">

A network is "matrix, bias, activation", done again for each layer. The activation is what makes many layers more powerful than one.

</div>

### From toy to real

- The network for handwritten digits in the 3Blue1Brown video has the layers 784 → 16 → 16 → 10, with 13,002 parameters.
- Each block of a Transformer contains a network of this type. In the first GPT model it has the layers 768 → 3,072 → 768, with the activation GELU, a smooth form of ReLU.
- Where sigmoid and tanh are flat, the gradient is almost 0 and the learning is slow. This is the main reason why deep networks use ReLU and its variants.

## 2. Training and optimization

### Setup

The **gradient** of a parameter says how much the loss changes when this parameter increases a little.
We want the gradient for each of the 12 parameters.
**Backpropagation** calculates them with the chain rule (see [Mathematics](01-math.md)). It starts at the loss and goes back, one layer at a time.
The loss here is the cross-entropy (0.417). For a target that is a number, the usual loss is the **mean squared error (MSE)**. On our output it would be (0.010 + 0.059 + 0.116) / 3 = 0.062.

<figure class="diagram" data-src="neural-networks/network.svg">
<figcaption>The forward pass calculates the loss. The backward pass uses the same values in the opposite order.</figcaption>
</figure>

### Steps

1. **Start with the gradient for the logits.** For softmax with cross-entropy, it is $p - y$.

    $$p - y = [0.099,\ 0.242,\ 0.659 - 1] = [0.099,\ 0.242,\ -0.341]$$

    **Why:** this vector is the error signal. A negative value means "this logit must go up". Only the value for `sat` is negative.

2. **Calculate the gradient for $U$.** The gradient of a weight is the error at its output times the value at its input.

    $$\frac{\partial L}{\partial U} = (p - y)\, h^\top = \begin{bmatrix} 0.099 \\ 0.242 \\ -0.341 \end{bmatrix} \begin{bmatrix} 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & 0.099 \\ 0 & 0.242 \\ 0 & -0.341 \end{bmatrix}$$

    **Why:** a weight multiplies its input. If the input is 0, the weight had no effect on the loss, so its gradient is 0. That is the case for column 1, because $h_1 = 0$.

3. **Send the error back to the hidden layer.** Multiply by the transpose of $U$ (its rows become columns).

    $$\frac{\partial L}{\partial h} = U^\top (p - y) = [-0.102,\ -0.430] \qquad \text{second value: } 0.1 \cdot 0.099 + 1.0 \cdot 0.242 + 2.0 \cdot (-0.341) = -0.430$$

    **Why:** $h_2$ goes into the 3 logits. Its total effect on the loss is the sum of the 3 paths.

4. **Go through the activation.** Multiply by the slope of ReLU: 1 where $z > 0$, and 0 where $z < 0$.

    $$\frac{\partial L}{\partial z} = [-0.102 \cdot 0,\ -0.430 \cdot 1] = [0,\ -0.430]$$

    **Why:** this is the chain rule. Unit 1 is off, so a small change of $z_1$ does not change the output. No error goes through it.

5. **Calculate the gradient for $W$ and $b$.** It is the rule of step 2, with the input $x = [1, 1]$.

    $$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial z}\, x^\top = \begin{bmatrix} 0 & 0 \\ -0.430 & -0.430 \end{bmatrix} \qquad \frac{\partial L}{\partial b} = [0,\ -0.430]$$

    **Why:** each layer does the same two operations: calculate its own gradients, and send the error to the layer before it. Thus one backward pass gives all 12 gradients.

6. **Update each parameter:** $\text{weight} \leftarrow \text{weight} - 0.1 \cdot \text{gradient}$.

    | Parameter | Before | After |
    |---|---|---|
    | Row 2 of $W$ | [0.8, 0.3] | [0.843, 0.343] |
    | $b_2$ | −0.1 | −0.057 |
    | Column 2 of $U$ | [0.1, 1.0, 2.0] | [0.090, 0.976, 2.034] |

    A new forward pass gives $p = [0.079, 0.214, 0.707]$. The loss goes from 0.417 to 0.347.

    **Why:** the gradient points to a higher loss, so we go the opposite way. Real training does this step with a small random group of examples each time, a **batch**. That method is **stochastic gradient descent (SGD)**.

7. **Compare with the Adam optimizer.** For column 2 of $U$, the first SGD step changes the weights by [−0.010, −0.024, +0.034]. With the same learning rate, the first **Adam** step changes them by [−0.1, −0.1, +0.1].

    **Why:** Adam divides each gradient by the square root of a running mean of its square (the idea of **RMSprop**), so each weight moves about one learning rate. It also uses a running mean of the gradient direction (**momentum**), which removes noise. Weights with small gradients learn as fast as the others. This is why Adam needs a smaller learning rate.

<div class="llm-widget" data-widget="forward-backward"></div>

<div class="key-point">

One forward pass and one backward pass give the gradient of all the parameters. The backward pass costs about as much as the forward pass, also when the network has billions of parameters.

</div>

### From toy to real

- You do not write the backward pass. In PyTorch, `loss.backward()` does steps 1 to 5 and `optimizer.step()` does step 6.
- One **epoch** is one pass through all the training examples.
- The default learning rate of Adam is 0.001.

## 3. Overfitting

### Setup

A network **overfits** when it learns the training examples by heart, with their errors, and then fails on new examples.
You can see it with a **validation set**: examples that are not used for the gradients.
The figure shows the experiment of part 4: a network that trains for 3,000 epochs on 40 examples with **noise** (some labels are wrong).

<figure class="plot">
<img src="/figures/neural-networks/overfitting.svg" alt="Loss against epoch. The train loss goes down to 0.052. The validation loss is lowest (0.565) at epoch 122, then goes up to 1.496. With L2 the validation loss stays near 0.559.">
<figcaption>After epoch 122 the network gets better on the train set and worse on new data.</figcaption>
</figure>

### Steps

1. **Stop at the lowest validation loss.** The validation loss is lowest at epoch 122 (0.565). At epoch 3,000 the train loss is 0.052, but the validation loss is 1.496. Keep the weights of epoch 122. This is **early stopping**.

    **Why:** after epoch 122, the network only learns the noise of the 40 examples. The validation set shows this, because its noise is different.

2. **Add a penalty for large weights (L2 regularization).** Add $\frac{\lambda}{2} \sum w^2$ to the loss. For the toy $W$ with $\lambda = 0.1$:

    $$\sum w^2 = 0.25 + 1.44 + 0.64 + 0.09 = 2.42 \qquad \text{penalty} = \tfrac{0.1}{2} \cdot 2.42 = 0.121 \qquad \text{loss} = 0.417 + 0.121 = 0.538$$

    The gradient of the penalty is $\lambda w$. Thus each update first multiplies each weight by $1 - 0.1 \cdot 0.1 = 0.99$. This is **weight decay**. **L1 regularization** uses $\lambda \sum |w| = 0.1 \cdot 2.8 = 0.28$, and it moves small weights to exactly 0.

    **Why:** a network needs large weights to draw the sharp curves that fit noise. The penalty makes these curves expensive. In the experiment, $\lambda = 0.01$ keeps the validation loss at 0.559 after 3,000 epochs.

3. **Turn off random units (dropout).** In each training step, set each hidden value to 0 with probability 0.5, and multiply the others by $1 / (1 - 0.5) = 2$. Our $h_2 = 1.0$ becomes 0 or 2.0. The mean stays 0.5 · 0 + 0.5 · 2.0 = 1.0.

    | Case | $h$ | Logits | Loss |
    |---|---|---|---|
    | $h_2$ is dropped | [0, 0] | [0, 0, 0] | $\ln 3 = 1.099$ |
    | $h_2$ is kept | [0, 2] | [0.2, 2.0, 4.0] | 0.146 |

    **Why:** the network cannot give all the work to one unit, because that unit is sometimes off. It must put the information in many units. You do not use dropout after training: all the units are on, and the factor 2 kept the mean the same.

<div class="key-point">

A low training loss is not the goal. The goal is a low loss on new data. Always look at the validation loss next to the training loss.

</div>

### From toy to real

- **Data augmentation** makes new training examples from the ones that you have (for example, an image and its mirror image). More data is the best protection.
- The first GPT model used a dropout rate of 0.1.
- When you fine-tune a large model (train it more on your own data), overfitting is common if the dataset is small. Watch the validation loss.

## 4. Implement a multilayer perceptron (MLP)

### Setup

A **multilayer perceptron (MLP)** is a network where each unit reads all the outputs of the layer before it.
The task: 2 inputs between −1 and 1. The class is 1 if the two inputs have the same sign, and 0 if not. We flip 15% of the labels (noise).
There are 40 training examples and 200 validation examples. The **accuracy** is the share of correct answers. The network has the layers 2 → 16 → 2 with tanh, and 2 · 16 + 16 + 16 · 2 + 2 = 82 parameters. The class `MLP` is in the section [Minimal code](#minimal-code).

### Steps

1. **Start with small random weights and biases of 0.** Each weight comes from a normal distribution with a standard deviation of $1 / \sqrt{n_{\text{in}}}$, where $n_{\text{in}}$ is the number of inputs of the layer. The first loss is 0.857, near the loss of a guess between 2 classes ($\ln 2 = 0.693$).

    **Why:** if all the weights start equal, all the units calculate the same value and get the same gradient. They stay copies of each other. Random values break this symmetry.

2. **Do the training loop.** Each epoch does a forward pass on the 40 examples, a backward pass, and one update with a learning rate of 0.5.

    | Epoch | Train loss | Validation loss | Train accuracy | Validation accuracy |
    |---|---|---|---|---|
    | 0 | 0.857 | 0.747 | | |
    | 122 | 0.412 | 0.565 | 80% | 78% |
    | 3,000 | 0.052 | 1.496 | 100% | 68.5% |
    | 3,000, with L2 ($\lambda = 0.01$) | 0.394 | 0.559 | | 78% |

    **Why:** these are the steps of part 2, for many examples at the same time. The gradient is the mean of the gradients of the examples.

3. **Remove the hidden layer and compare.** A network 2 → 2 (6 parameters) gets a validation accuracy of 56.5% after the same training.

    **Why:** one layer can only cut the plane with a straight line. This task needs a curve: the two classes are in opposite corners. The hidden layer with its activation gives 78%.

Real-library sketch (the tests do not run this code). The same network in PyTorch:

```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, weight_decay=0.01)   # weight decay = L2
loss_fn = nn.CrossEntropyLoss()        # softmax and cross-entropy in one function

for epoch in range(3000):
    optimizer.zero_grad()              # remove the gradients of the last step
    loss = loss_fn(model(x_train), y_train)
    loss.backward()                    # backpropagation
    optimizer.step()                   # one update of each parameter
```

<div class="key-point">

An MLP is three small functions: forward, backward, update. Libraries write the backward function for you, but the numbers are the ones of this page.

</div>

### From toy to real

- Real code uses one batch of examples in each step, not the full dataset.
- PyTorch puts the same code on a graphics processing unit (GPU) with `model.to("cuda")`.

## Minimal code

This is the class that the tests train. It gives the numbers of parts 3 and 4.

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

class MLP:
    def __init__(self, sizes, seed):                     # sizes = [2, 16, 2]
        rng = np.random.default_rng(seed)
        pairs = list(zip(sizes, sizes[1:]))
        self.weights = [rng.normal(0, np.sqrt(1 / n_in), (n_out, n_in)) for n_in, n_out in pairs]
        self.biases = [np.zeros(n_out) for _, n_out in pairs]

    def forward(self, x):                                # returns the output of each layer
        outputs = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = outputs[-1] @ w.T + b
            outputs.append(softmax(z) if i == len(self.weights) - 1 else np.tanh(z))
        return outputs

    def gradients(self, x, y, l2=0.0):
        outputs = self.forward(x)
        delta = outputs[-1].copy()
        delta[np.arange(len(y)), y] -= 1                 # p - y for each example
        delta /= len(y)                                  # the loss is a mean
        grads_w, grads_b = [], []
        for i in reversed(range(len(self.weights))):
            grads_w.insert(0, delta.T @ outputs[i] + l2 * self.weights[i])
            grads_b.insert(0, delta.sum(axis=0))
            if i > 0:                                    # send the error back: tanh'(z) = 1 - h^2
                delta = (delta @ self.weights[i]) * (1 - outputs[i] ** 2)
        return grads_w, grads_b

    def step(self, x, y, learning_rate, l2=0.0):
        grads_w, grads_b = self.gradients(x, y, l2)
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]

def make_data(points, seed):                             # class 1 if the two inputs have the same sign
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(points, 2))
    y = (x[:, 0] * x[:, 1] > 0).astype(int)
    return x, np.where(rng.random(points) < 0.15, 1 - y, y)   # flip 15% of the labels

x_train, y_train = make_data(40, seed=0)
model = MLP([2, 16, 2], seed=0)
for epoch in range(3000):
    model.step(x_train, y_train, learning_rate=0.5)      # train loss: 0.857 -> 0.052
```

## Comparison

| Optimizer | What one step uses | First change of column 2 of $U$ (learning rate 0.1) | Note |
|---|---|---|---|
| Gradient descent | The gradient of all the examples | [−0.010, −0.024, +0.034] | Exact, but slow on a large dataset |
| SGD | The gradient of one batch | The same here (one example) | Fast steps with noise |
| SGD with momentum | A running mean of the last gradients | | Keeps its direction when the gradients change sign |
| RMSprop | The gradient divided by the square root of a running mean of its square | | One step size for each weight |
| Adam | Momentum and RMSprop together | [−0.1, −0.1, +0.1] | The usual choice for language models |

## Checks and common misconceptions

- **"A lower training loss means a better model."** In the experiment, the train loss goes from 0.412 to 0.052 while the validation loss goes from 0.565 to 1.496. Check: draw the two curves on the same plot.
- **"A gradient of 0 means that the weight is good."** Row 1 of $W$ has a gradient of 0 because its ReLU unit is off, not because the weights are correct. A unit that is off for all the examples cannot learn. Check: count the units that give 0 for each example.
- **"More layers always help."** Layers with no activation between them are one layer: $U(Wx + b) = (UW)x + Ub$. And each layer with an activation adds parameters that can fit noise.

## Sources

This page follows section 3 of "LLM Fundamentals" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [But what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk), 3Blue1Brown (the 784 → 16 → 16 → 10 network with 13,002 parameters).
- [Deep Learning Crash Course](https://www.youtube.com/watch?v=VyWAvY2CF9c), freeCodeCamp.
- [Practical Deep Learning](https://course.fast.ai/), fast.ai.
- [PyTorch Tutorials](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4), Patrick Loeber.
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), Radford et al., 2018 (the first GPT model: layers 768 → 3,072 → 768, GELU, dropout rate 0.1).
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), Kingma and Ba, 2014 (default learning rate 0.001).
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html), Srivastava et al., 2014.
