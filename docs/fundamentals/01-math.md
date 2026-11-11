# 1. Mathematics for machine learning

## Core idea

A model is a long chain of small calculations on lists of numbers.
Linear algebra is the tool that does these calculations. Calculus tells the model which numbers to change, and by how much.
Probability gives a score to each guess of the model, so that training has a target.

## The example

This page uses the [toy example](../toy-example.md).

| Item | Value |
|---|---|
| Embeddings | `the` = [1, 0], `cat` = [0, 1], `sat` = [1, 1] |
| Weight matrix | $W = \begin{bmatrix} 0.5 & -1.2 \\ 0.8 & 0.3 \end{bmatrix}$ |
| Logits after "the cat" | [0.1, 1.0, 2.0] for `the`, `cat`, `sat` |
| A loss with one weight (an assumption for this page) | $L(w) = (w - 2)^2$, start at $w = 0$, learning rate 0.1 |

A **token** is one unit of text, here one word. An **embedding** is the list of numbers that a model uses for one token.
A **logit** is the raw score that the model gives to one token. A higher logit means "more likely".

## 1. Linear algebra

### Setup

A **vector** is a list of numbers. The embedding of `sat` is the vector [1, 1]. You can draw it as an arrow from the point (0, 0) to the point (1, 1).
A **matrix** is a table of numbers. $W$ has 2 rows and 2 columns.
A model stores words as vectors and stores what it learned as matrices.

### Steps

1. **Add two vectors.** Add the numbers that have the same position: `the` + `cat` = [1 + 0, 0 + 1] = [1, 1]. In the toy example, this sum is the embedding of `sat`.

    **Why:** a model adds vectors all the time. Each layer of a large language model (LLM) adds its result to its input vector.
    { .why }

2. **Calculate a dot product.** The **dot product** multiplies two vectors position by position and adds the results.

    $$\text{sat} \cdot \text{cat} = 1 \cdot 0 + 1 \cdot 1 = 1 \qquad \text{sat} \cdot \text{sat} = 2 \qquad \text{the} \cdot \text{cat} = 0$$

    The length of a vector is the square root of its dot product with itself: $\lVert \text{sat} \rVert = \sqrt{2} = 1.414$.
    The **cosine similarity** divides the dot product by the two lengths: for `sat` and `the`, $1 / (1.414 \cdot 1) = 0.707$.

    **Why:** the dot product is large when two vectors point in the same direction, and 0 when they are at a right angle. It is the usual measure of "these two things are similar".
    { .why }

3. **Multiply the matrix by a vector.** Each output number is the dot product of one row of $W$ with the vector.

    $$W \cdot \text{sat} = \begin{bmatrix} 0.5 & -1.2 \\ 0.8 & 0.3 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.5 \cdot 1 + (-1.2) \cdot 1 \\ 0.8 \cdot 1 + 0.3 \cdot 1 \end{bmatrix} = \begin{bmatrix} -0.7 \\ 1.1 \end{bmatrix}$$

    Also, $W \cdot \text{the} = [0.5, 0.8]$ (column 1 of $W$) and $W \cdot \text{cat} = [-1.2, 0.3]$ (column 2). Their sum is [−0.7, 1.1], the same result.

    **Why:** a matrix is a function that moves each vector to a new place. It is a **linear transformation**: the result for a sum is the sum of the results. This is the main operation of each layer.
    { .why }

4. **Calculate the determinant.** For a 2 × 2 matrix with the rows $[a, b]$ and $[c, d]$, the **determinant** is $ad - bc$.

    $$\det W = 0.5 \cdot 0.3 - (-1.2) \cdot 0.8 = 0.15 + 0.96 = 1.11$$

    **Why:** the determinant is the factor by which the matrix changes an area. A square of area 1 becomes a shape of area 1.11. If the determinant is 0, the matrix flattens the plane onto a line. Then two different inputs give the same output, and you cannot go back.
    { .why }

5. **Find an eigenvector.** An **eigenvector** of a matrix is a vector that the matrix only makes longer or shorter. The factor is the **eigenvalue**. Take the matrix $G = E^\top E$, where the rows of $E$ are the 3 embeddings.

    $$G = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \qquad G \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 3 \\ 3 \end{bmatrix} = 3 \begin{bmatrix} 1 \\ 1 \end{bmatrix} \qquad G \begin{bmatrix} 1 \\ -1 \end{bmatrix} = 1 \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

    The eigenvalues are 3 and 1. The matrix $W$ has no such real vector: it turns each vector, and its eigenvalues are the complex numbers $0.4 \pm 0.975i$.

    **Why:** the eigenvectors show the main directions of a matrix. Here the 3 embeddings are 3 times larger along the direction [1, 1] than along the direction [1, −1] (as a sum of squared lengths). You can keep the large directions and drop the small ones. That is the idea of data compression.
    { .why }

<figure class="diagram">
--8<-- "docs/assets/figures/math/transform.svg"
<figcaption>Steps 3 and 4: W moves each vector, and it multiplies each area by 1.11.</figcaption>
</figure>

!!! success "Key point"
    A layer of a model is "multiply a vector by a matrix". Each output number is one dot product. The other ideas (determinant, eigenvalues) describe what a matrix does to the space.

### From toy to real

- In Llama 3 8B, each vector has 4,096 numbers. The last matrix has one row for each of the 128,256 tokens: 128,256 × 4,096 = 525,336,576 numbers.
- A **tensor** is an array with any number of axes, for example 3 axes: (texts, tokens, 4,096). The rules do not change: the library applies the same matrix to each vector.
- A graphics processing unit (GPU) does many dot products at the same time. This is why models run on GPUs.

## 2. Calculus

### Setup

A **loss** is a function that measures the error of a model. A loss of 0 is a perfect answer.
A **weight** is a number that the model learns.
Assume a model with one weight $w$. Its output is $w$, and the correct output is the logit 2.0.
The loss is the squared error $L(w) = (w - 2)^2$. We start at $w = 0$, where $L = 4$. We must find the $w$ with the smallest loss.

### Steps

1. **Measure the slope with a small move.** Change $w$ from 0 to 0.001 and look at the loss.

    $$\frac{L(0.001) - L(0)}{0.001} = \frac{3.996001 - 4}{0.001} = -3.999$$

    With a smaller and smaller move, this ratio goes to −4. This limit is the **derivative** $\frac{dL}{dw}$.

    **Why:** the derivative says what a small change of $w$ does to the loss. A negative value means: increase $w$ and the loss goes down.
    { .why }

2. **Get the exact formula with the chain rule.** Write the loss as two simple functions: $u = w - 2$ and $L = u^2$. The **chain rule** multiplies their derivatives.

    $$\frac{dL}{dw} = \frac{dL}{du} \cdot \frac{du}{dw} = 2u \cdot 1 = 2(w - 2) \qquad \text{at } w = 0: \ -4$$

    **Why:** the formula is exact and needs one calculation. A model is a chain of many simple functions, so the chain rule gives the derivative of the full model.
    { .why }

3. **Move the weight against the slope.** The **learning rate** $\eta$ sets the size of the move. Here $\eta = 0.1$.

    $$w \leftarrow w - \eta \frac{dL}{dw} = 0 - 0.1 \cdot (-4) = 0.4 \qquad L(0.4) = 2.56$$

    **Why:** the derivative points to the direction where the loss increases, so we go in the opposite direction. The step is small because the slope is only correct near the current $w$.
    { .why }

4. **Do it again.** This loop is **gradient descent**.

    | Step | $w$ | Loss $L(w)$ | Derivative |
    |---|---|---|---|
    | 0 | 0 | 4 | −4 |
    | 1 | 0.4 | 2.56 | −3.2 |
    | 2 | 0.72 | 1.638 | −2.56 |
    | 3 | 0.976 | 1.049 | −2.048 |
    | 4 | 1.181 | 0.671 | −1.638 |

    **Why:** one step is not sufficient. Near the minimum the slope is smaller, so the steps get smaller without a change of $\eta$.
    { .why }

5. **Try a learning rate that is too large.** With $\eta = 1.1$, the weight goes 0 → 4.4 → −0.88 → 5.456, and the loss goes 4 → 5.76 → 8.294 → 11.944. Each step multiplies the distance to $w = 2$ by 1.2.

    **Why:** a step that is too long jumps over the minimum and lands higher on the other side. The training **diverges**. The learning rate is the first setting to check when a loss goes up.
    { .why }

6. **Use one derivative for each input: the gradient.** The loss of part 3 has 3 inputs, the logits [0.1, 1.0, 2.0], and its value is 0.417030. Add 0.001 to the logit of `sat` only: the loss becomes 0.416689. The slope is (0.416689 − 0.417030) / 0.001 = −0.341. This is a **partial derivative**. The same test for each logit, with a smaller and smaller move, gives the **gradient**:

    $$\nabla L = [0.099,\ 0.242,\ -0.341]$$

    The update of step 3 does not change: $w \leftarrow w - \eta \nabla L$, for all the inputs at the same time.

    **Why:** a real model has billions of weights. The gradient gives the correct direction for each weight in one vector.
    { .why }

<figure class="plot" markdown>
![Weight w for 8 steps of gradient descent. Learning rate 0.1 goes slowly to 2. Learning rate 0.9 jumps around 2 and gets nearer. Learning rate 1.1 jumps around 2 and gets farther.](../assets/figures/math/learning-rate.svg#only-light)
![Weight w for 8 steps of gradient descent. Learning rate 0.1 goes slowly to 2. Learning rate 0.9 jumps around 2 and gets nearer. Learning rate 1.1 jumps around 2 and gets farther.](../assets/figures/math/learning-rate-dark.svg#only-dark)
<figcaption>Same loss, same start, three learning rates. Only the rates below 1 go to the best weight, w = 2.</figcaption>
</figure>

<div class="llm-widget" data-widget="gradient-descent"></div>

!!! success "Key point"
    Training is a loop: calculate the gradient, move each weight a small step against it, do it again. The learning rate decides if this loop is slow, fast or unstable.

### From toy to real

- **Backpropagation** is the chain rule, applied from the loss back through each layer. See [Neural networks](03-neural-networks.md).
- You do not write derivatives by hand. A library records each operation and calculates the gradient for you. This is **automatic differentiation**.
- Real training uses small learning rates. The Adam optimizer has a default of 0.001.
- An **integral** is the opposite of a derivative: it adds up the area below a curve. Probability uses integrals for continuous values. The tokens of an LLM are separate values, so sums are sufficient.

## 3. Probability and statistics

### Setup

A **probability** is a number from 0 to 1 that says how often an event occurs.
A **random variable** is a quantity that gets its value from a random draw. Here it is the next token.
A **distribution** gives the probability of each possible value. The model must turn its 3 logits into a distribution.

### Steps

1. **Turn the logits into probabilities with softmax.** **Softmax** takes the exponential of each logit and divides by the sum.

    $$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}} \qquad e^{0.1} = 1.105,\quad e^{1.0} = 2.718,\quad e^{2.0} = 7.389$$

    $$p = [0.099,\ 0.242,\ 0.659] \qquad 0.099 + 0.242 + 0.659 = 1$$

    **Why:** a distribution needs positive numbers that add up to 1. The exponential makes each number positive. The division makes the sum equal to 1.
    { .why }

2. **Calculate an expectation.** Let $X = 1$ if the drawn token is `sat`, and $X = 0$ if not. The **expectation** is the mean of $X$ over many draws: each value times its probability.

    $$E[X] = \sum_i p_i x_i = 0.099 \cdot 0 + 0.242 \cdot 0 + 0.659 \cdot 1 = 0.659$$

    **Why:** one draw is random, but the mean of many draws is stable. For 100 draws, expect 65.9 times `sat`.
    { .why }

3. **Calculate the variance.** The **variance** is the mean squared distance from the expectation. The **standard deviation** is its square root.

    $$\text{Var}[X] = \sum_i p_i (x_i - E[X])^2 = 0.099 \cdot 0.4343 + 0.242 \cdot 0.4343 + 0.659 \cdot 0.1163 = 0.225$$

    The squared distances are $(0 - 0.659)^2 = 0.4343$ and $(1 - 0.659)^2 = 0.1163$.

    The standard deviation is $\sqrt{0.225} = 0.474$. For the count of `sat` in 100 draws it is $\sqrt{100 \cdot 0.225} = 4.74$.

    **Why:** the variance says how far a result can be from the expectation. A count of 61 or 70 in 100 draws is normal. A count of 40 is not.
    { .why }

4. **Score the model with the cross-entropy.** The correct next token is `sat`. The model gave it the probability 0.659. This is the **likelihood** of the data. The **cross-entropy loss** is the negative logarithm of this number.

    $$L = -\ln(p_{\text{sat}}) = -\ln(0.659) = 0.417$$

    If the correct token was `the` (probability 0.0986), the loss would be $-\ln(0.0986) = 2.317$.

    **Why:** the loss is 0 when the probability of the correct token is 1, and it grows fast when the model is sure of a wrong token. With the logarithm, the loss of a text is the sum of the losses of its tokens.
    { .why }

5. **Get the gradient of this loss.** Write the correct answer as a vector $y = [0, 0, 1]$. The gradient of the loss for the logits is $p - y$.

    $$p - y = [0.099,\ 0.242,\ 0.659 - 1] = [0.099,\ 0.242,\ -0.341]$$

    These are the numbers of step 6 in part 2.

    **Why:** the three parts of this page meet here. A step against this gradient makes the logit of `sat` higher. It makes the two others lower, in proportion to their probability.
    { .why }

!!! success "Key point"
    A language model outputs a distribution. Training makes the probability of the correct token higher. This is the same thing as a lower cross-entropy.

### From toy to real

- The loss of a real LLM is the mean cross-entropy of all the tokens of a batch (a group of texts). **Perplexity** is $e^{L}$. Here $e^{0.417} = 1.517$: the model hesitates as if it had 1.5 equally likely choices.
- **Maximum likelihood estimation (MLE)** means: choose the weights that give the highest probability to the training data. A minimum of the cross-entropy does exactly this.
- A **confidence interval** shows the noise of a measurement. A share of 0.659 that you measure on 100 samples has a 95% interval of about ± 1.96 · 0.474 / √100 = ± 0.093. A difference of 2 points between two models on 100 questions is thus not a result.
- **Covariance** and **correlation** measure if two variables move together. **Bayes' rule**, $P(A \mid B) = P(B \mid A) P(A) / P(B)$, updates a probability when new data comes.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import numpy as np

sat = np.array([1., 1.])
W = np.array([[0.5, -1.2], [0.8, 0.3]])
hidden = W @ sat                                 # [-0.7, 1.1]
det = np.linalg.det(W)                           # 1.11

# Gradient descent on L(w) = (w - 2)^2
w = 0.0
for step in range(4):
    gradient = 2 * (w - 2)                       # -4, -3.2, -2.56, -2.048
    w = w - 0.1 * gradient                       # 0.4, 0.72, 0.976, 1.181

# Probabilities, loss and gradient for the logits after "the cat"
logits = np.array([0.1, 1.0, 2.0])
p = np.exp(logits) / np.exp(logits).sum()        # [0.099, 0.242, 0.659]
loss = -np.log(p[2])                             # 0.417
y = np.array([0., 0., 1.])                       # the correct token is "sat"
gradient = p - y                                 # [0.099, 0.242, -0.341]
```

## Comparison

| Field | Tool | Toy number | Where an LLM uses it |
|---|---|---|---|
| Linear algebra | Dot product | `sat` · `cat` = 1 | Attention scores, similarity search |
| Linear algebra | Matrix × vector | $W \cdot$ `sat` = [−0.7, 1.1] | Each layer |
| Linear algebra | Determinant | 1.11 | Tells if a matrix loses information |
| Linear algebra | Eigenvalues | 3 and 1 | Compression, principal component analysis |
| Calculus | Derivative | −4 at $w = 0$ | Direction of one weight update |
| Calculus | Gradient | [0.099, 0.242, −0.341] | Direction for all the weights |
| Calculus | Gradient descent | 0 → 0.4 → 0.72 | The training loop |
| Probability | Softmax | [0.099, 0.242, 0.659] | Output of the model |
| Probability | Expectation, variance | 0.659 and 0.225 | Sampling, noise of an evaluation |
| Probability | Cross-entropy | 0.417 | The training loss |

## Checks and common misconceptions

- **"A larger learning rate always trains faster."** It does, up to a limit. For this loss, a rate of 0.5 gets to the minimum in one step. Between 0.5 and 1, $w$ jumps over the minimum at each step, and training becomes slower. Each rate above 1 makes the loss go up at each step. Check: if your loss increases or becomes `nan` ("not a number"), divide the learning rate by 10 first.
- **"A large dot product means that two vectors are similar."** The dot product also grows with the length. `sat` · `sat` = 2 but `the` · `the` = 1, and each vector is identical to itself. Check: use the cosine similarity (1.0 for the two) when the lengths are different.
- **"A probability of 0.659 means that the model is correct 65.9% of the time."** That is true only if the model is **calibrated**: its probabilities agree with the real frequencies. Softmax always gives numbers that add up to 1, also when the model is wrong.

## Sources

This page follows section 1 of "LLM Fundamentals" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [The Essence of Linear Algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), 3Blue1Brown.
- [Statistics Fundamentals](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9), StatQuest with Josh Starmer.
- [Seeing Theory](https://seeing-theory.brown.edu/), Brown University.
- [Immersive Linear Algebra](https://immersivemath.com/ila/learnmore.html), Ström, Åström and Akenine-Möller.
- Khan Academy: [Linear Algebra](https://www.khanacademy.org/math/linear-algebra), [Calculus](https://www.khanacademy.org/math/calculus-1), [Probability and Statistics](https://www.khanacademy.org/math/statistics-probability).
- [Meta Llama 3 8B model card and configuration](https://huggingface.co/meta-llama/Meta-Llama-3-8B) (vector size 4,096, vocabulary 128,256).
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), Kingma and Ba, 2014 (default learning rate 0.001).
