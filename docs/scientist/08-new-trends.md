# 8. New trends

## Core idea

This page has four topics that do not fit in the other sections. Each one uses a trained model in a new way:
you can mix the weights of two models (merging), give the model images (multimodal), read or change what is inside the model (interpretability), or let the model work longer on one question (test-time compute).

## The example

Each part uses vectors with 2 or 4 numbers. Values marked "assumed" are made for this page.

| Part | Values |
|---|---|
| Merging | $W = \begin{bmatrix} 0.5 & -1.2 \\ 0.8 & 0.3 \end{bmatrix}$ from the [toy example](../toy-example.md), and a second model $W_B = \begin{bmatrix} 1.2 & 0.3 \\ 0.5 & 0.8 \end{bmatrix}$ (assumed) |
| Multimodal | 2 image vectors and 2 text vectors with 2 numbers each (assumed) |
| Interpretability | One activation $x = [0.2, 1.0]$ (assumed) |
| Test-time compute | One sample is correct with a probability $p = 0.6$ (assumed) |

## 1. Model merging

### Setup

**Model merging** makes one model from the weights of two or more trained models. No training is necessary.
The models must have the same architecture, so that each weight has a partner.
Write each matrix as one long vector: $a = [0.5, -1.2, 0.8, 0.3]$ and $b = [1.2, 0.3, 0.5, 0.8]$. The two vectors have the same length (norm): 1.556.

### Steps

1. **Linear interpolation (LERP): take a point on the straight line from $a$ to $b$.** With $t = 0.5$:

    $$(1 - t)\,a + t\,b = 0.5\,a + 0.5\,b = [0.85,\ -0.45,\ 0.65,\ 0.55] \qquad \text{length} = 1.285$$

    **Why:** the mean is the simplest merge. But the result is shorter than the two inputs (1.285 against 1.556), so the merged layer gives smaller outputs.
    { .why }

2. **Calculate the angle between the two models.**

    $$\cos\theta = \frac{a \cdot b}{\lVert a \rVert\,\lVert b \rVert} = \frac{0.88}{2.42} = 0.364 \qquad \theta = 68.7^\circ$$

    **Why:** the two vectors point in different directions. The straight line between them cuts through the inside of the circle, and this is why LERP loses length.
    { .why }

3. **Spherical linear interpolation (SLERP): take a point on the arc from $a$ to $b$.**

    $$\text{slerp}(a, b, t) = \frac{\sin((1 - t)\theta)}{\sin\theta}\,a + \frac{\sin(t\theta)}{\sin\theta}\,b$$

    With $t = 0.5$, the two factors are $\sin(34.3^\circ) / \sin(68.7^\circ) = 0.5641 / 0.9315 = 0.606$.

    $$0.606\,(a + b) = [1.029,\ -0.545,\ 0.787,\ 0.666] \qquad \text{length} = 1.556$$

    **Why:** the factors are larger than 0.5, and they keep the length of the inputs. SLERP turns one model into the other at a constant speed.
    { .why }

4. **TIES: merge task vectors and remove the sign conflicts.** A **task vector** is the weights after fine-tuning minus the weights before. Assume two fine-tuned versions of the same base model:

    | | Weight 1 | Weight 2 | Weight 3 | Weight 4 |
    |---|---|---|---|---|
    | Task vector A | 0.4 | -0.3 | 0.05 | 0.2 |
    | Task vector B | 0.2 | 0.5 | -0.6 | 0.02 |
    | Plain mean | 0.3 | 0.1 | -0.275 | 0.11 |
    | A after trim (keep the 2 largest) | 0.4 | -0.3 | 0 | 0 |
    | B after trim | 0 | 0.5 | -0.6 | 0 |
    | Elected sign (sign of the sum) | + | + | − | none |
    | TIES result (mean of the values with that sign) | 0.4 | **0.5** | -0.6 | 0 |

    The merged weights are base + result: $[0.5, -1.2, 0.8, 0.3] + [0.4, 0.5, -0.6, 0] = [0.9, -0.7, 0.2, 0.3]$.

    **Why:** for weight 2, model A says -0.3 and model B says +0.5. The plain mean gives 0.1, which is not what A learned and not what B learned. TIES keeps the stronger direction, and the trim removes small changes.
    { .why }

5. **DARE: drop random entries of a task vector and rescale the others.** Drop each entry with a probability of 0.5. Assume that entries 1 and 3 of A stay: $[0.4, 0, 0.05, 0]$. Divide by $1 - 0.5$: $[0.8, 0, 0.1, 0]$.

    **Why:** two sparse task vectors have fewer entries in the same place, so they conflict less. The division keeps the mean value of each entry: 0.5 × 0.8 = 0.4.
    { .why }

<figure class="diagram">
--8<-- "docs/assets/figures/new-trends/slerp.svg"
<figcaption>Steps 1 to 3 in the plane of the two models. LERP cuts the corner. SLERP stays on the circle.</figcaption>
</figure>

!!! success "Key point"
    A merge is arithmetic on weights. LERP and SLERP mix two full models. TIES and DARE mix the *changes* that each fine-tuning made to a shared base model.

### From toy to real

- The [mergekit](https://github.com/cg123/mergekit) library has these methods. A configuration file names the models, the method and the factor $t$, which can be different for each layer.
- SLERP merges 2 models. TIES and DARE can merge more models, because they add task vectors to one base.
- Two fine-tuned versions of one model are usually much nearer to each other than our two matrices. When the angle is small, LERP and SLERP give almost the same result.
- The DARE paper reports that you can drop 90% of the entries of a task vector, and for some models 99%.

## 2. Multimodal models

### Setup

A **multimodal** model reads more than one type of input, for example text and images.
**CLIP** has an image encoder and a text encoder. Training makes the vector of an image and the vector of its caption point in the same direction.
We have a batch of 2 pairs. All the vectors have a length of 1.

| Pair | Image vector | Text vector |
|---|---|---|
| 1 | photo of a cat: $[0.8, 0.6]$ | "a cat": $[1, 0]$ |
| 2 | photo of a dog: $[0.6, -0.8]$ | "a dog": $[0.6, -0.8]$ |

### Steps

1. **Calculate the similarity of each image with each text.** The similarity is the dot product. Row = image, column = text.

    $$S = \begin{bmatrix} 0.8 & 0 \\ 0.6 & 1.0 \end{bmatrix}$$

    **Why:** the correct pairs are on the diagonal. A good model has large numbers on the diagonal and small numbers in the other places.
    { .why }

2. **Divide by a temperature.** Assume $\tau = 0.5$: the logits are $S / \tau = \begin{bmatrix} 1.6 & 0 \\ 1.2 & 2.0 \end{bmatrix}$.

    **Why:** similarities are between -1 and 1, which is a small range for softmax. A low temperature makes the differences larger. CLIP learns this value.
    { .why }

3. **Apply softmax in the two directions.** On each row: "which text goes with this image?" On each column: "which image goes with this text?"

    $$\text{rows: } \begin{bmatrix} \mathbf{0.832} & 0.168 \\ 0.310 & \mathbf{0.690} \end{bmatrix} \qquad \text{columns: } \begin{bmatrix} \mathbf{0.599} & 0.119 \\ 0.401 & \mathbf{0.881} \end{bmatrix}$$

    **Why:** each row is now a classification problem with one correct answer, the same as next-token prediction. The caption is the label.
    { .why }

4. **Calculate the loss.** Take $-\ln$ of each diagonal value and calculate the mean.

    $$\text{rows: } \frac{0.184 + 0.371}{2} = 0.278 \qquad \text{columns: } \frac{0.513 + 0.127}{2} = 0.320 \qquad \text{loss} = \frac{0.278 + 0.320}{2} = 0.299$$

    **Why:** this **contrastive** loss falls when a correct pair gets nearer and when a wrong pair moves apart. The largest term (0.513) comes from the text "a cat", which is also near the dog photo (0.6).
    { .why }

5. **Give an image to an LLM with a projector (LLaVA).** A vision encoder cuts the image into patches and gives one vector for each patch. A **projector** is a small trained layer that changes a patch vector into a vector with the size of a token embedding. Toy: use $W$ as the projector on the cat vector.

    $$W \begin{bmatrix} 0.8 \\ 0.6 \end{bmatrix} = \begin{bmatrix} 0.5 \times 0.8 - 1.2 \times 0.6 \\ 0.8 \times 0.8 + 0.3 \times 0.6 \end{bmatrix} = \begin{bmatrix} -0.32 \\ 0.82 \end{bmatrix}$$

    The input of the LLM is now 3 vectors: $[-0.32, 0.82]$, then `the` = $[1, 0]$, then `cat` = $[0, 1]$.

    **Why:** an LLM reads a sequence of vectors. It does not know where a vector comes from. After the projector, an image patch is one more token that attention can read.
    { .why }

<figure class="diagram">
--8<-- "docs/assets/figures/new-trends/llava.svg"
<figcaption>LLaVA: the projector changes image patches into tokens that the LLM can read.</figcaption>
</figure>

!!! success "Key point"
    Multimodal models put all the inputs in one vector space. CLIP learns that space with a contrastive loss. LLaVA adds a projector, so that an LLM reads image patches as tokens.

### From toy to real

- CLIP was trained on 400 million pairs of an image and a text, with a batch of 32,768 pairs. Each image then has 1 correct text and 32,767 wrong texts.
- LLaVA-1.5 uses the CLIP image encoder ViT-L/14 at 336 pixels. A patch is 14 × 14 pixels: 336 ÷ 14 = 24, and 24 × 24 = 576 image tokens for one image. Its projector is a small network of 2 layers (an MLP).
- Text-to-image models such as Stable Diffusion go in the other direction: the vectors of a text encoder guide the generation of the image.

## 3. Interpretability

### Setup

**Interpretability** tries to find what the numbers inside a model mean. One activation vector holds many concepts at the same time, mixed together.
A **sparse autoencoder (SAE)** rewrites an activation as a sum of a few **features** (directions with one meaning each). It has more features than the activation has numbers.

$$f = \text{ReLU}(W_{enc}\,x + b) \qquad \hat{x} = W_{dec}\,f \qquad \text{loss} = \lVert x - \hat{x} \rVert^2 + \lambda \lVert f \rVert_1$$

Assume 3 features with the directions $d_1 = [1, 0]$, $d_2 = [0, 1]$, $d_3 = [0.6, 0.8]$. The rows of $W_{enc}$ and the columns of $W_{dec}$ are these directions. $b = [-0.5, -0.5, -0.5]$ and $\lambda = 0.1$.

### Steps

1. **Encode.** $W_{enc}\,x + b = [0.2,\ 1.0,\ 0.92] - 0.5 = [-0.3,\ 0.5,\ 0.42]$.

    **Why:** each dot product measures how much of one direction is in $x$. The negative bias is a threshold: a weak match must not count.
    { .why }

2. **Apply ReLU.** **ReLU** changes each negative number to 0: $f = [0,\ 0.5,\ 0.42]$.

    **Why:** feature 1 is now off: 2 of 3 features are active. In a real SAE, only a few features of many thousands are active, so each one is easier to name.
    { .why }

3. **Decode.** $\hat{x} = 0.5 \cdot [0, 1] + 0.42 \cdot [0.6, 0.8] = [0.252,\ 0.836]$.

    **Why:** if the features can build the activation again, they hold the same information.
    { .why }

4. **Calculate the loss.** The error is $x - \hat{x} = [-0.052,\ 0.164]$.

    $$\lVert x - \hat{x} \rVert^2 = 0.0296 \qquad \lVert f \rVert_1 = 0.5 + 0.42 = 0.92 \qquad \text{loss} = 0.0296 + 0.1 \times 0.92 = 0.122$$

    **Why:** the two terms work against each other. The first term wants a good copy. The second term (the L1 penalty) wants few active features.
    { .why }

5. **Abliteration, part 1: find one direction.** Assume the mean activation is $[1.0, 1.2]$ on prompts that the model refuses and $[0.4, 0.4]$ on prompts that it answers. The difference is $[0.6, 0.8]$. Its length is 1, so the unit direction is $r = [0.6, 0.8]$.

    **Why:** if a behaviour is one direction in the activations, the difference of the two means points along it. Here $r$ is the same direction as feature 3 of the SAE.
    { .why }

6. **Abliteration, part 2: remove that direction.** Project $x$ on $r$ and subtract the result.

    $$x \cdot r = 0.2 \times 0.6 + 1.0 \times 0.8 = 0.92 \qquad x' = x - 0.92\,r = [0.2, 1.0] - [0.552, 0.736] = [-0.352,\ 0.264]$$

    Check: $x' \cdot r = 0$. You can also change the weights one time: $W' = W - r r^\top W = \begin{bmatrix} -0.064 & -0.912 \\ 0.048 & 0.684 \end{bmatrix}$, and then $r^\top W' = [0, 0]$.

    **Why:** the model can no longer write along $r$, so the behaviour that uses this direction stops. The other directions stay the same.
    { .why }

!!! success "Key point"
    An SAE finds directions that have a meaning. Abliteration removes one direction. The two methods use the same idea: a concept is a direction in the space of the activations.

### From toy to real

- Anthropic trained SAEs on a layer with 512 neurons and got from 512 to 131,072 features. Examples of features: text in one language, DNA sequences, legal text.
- **Abliteration** was first used on the "refusal direction". The method changes each weight matrix that writes into the **residual stream** (the main vector that each block reads and changes). The model then refuses less. Its quality on other tasks can fall a little, so evaluate it again.

## 4. Test-time compute

### Setup

**Test-time compute** is the work that a model does for one question at inference time. More work can give a better answer from the same weights.
The simplest method is **self-consistency**: ask the same question N times with a temperature above 0, then take the answer that occurs most often (a majority vote).
Assume that each sample is correct with $p = 0.6$, that samples are independent, and that there are only 2 possible answers.

### Steps

1. **One sample.** The probability of a correct answer is $p = 0.6$.

    **Why:** this is the baseline: one call to the model.
    { .why }

2. **Three samples.** The vote is correct if 3 or 2 samples are correct.

    $$p^3 + 3p^2(1 - p) = 0.216 + 0.432 = 0.648$$

    **Why:** one wrong sample no longer decides the result.
    { .why }

3. **N samples.** Add the probabilities of all the cases with more than half correct.

    $$P(\text{vote correct}) = \sum_{k > N/2} \binom{N}{k} p^k (1 - p)^{N - k}$$

    For $N = 5$: $0.3456 + 0.2592 + 0.0778 = 0.683$. For $N = 41$: 0.903.

    **Why:** the gain is real but slow. 5 calls cost 5 times more and add 8 points. You pay compute for accuracy.
    { .why }

4. **Check the limit.** With $p = 0.4$, the vote of 5 samples is correct with 0.317, which is worse than one sample.

    **Why:** a vote makes the most frequent answer stronger. If the most frequent answer is wrong, more samples make the result worse.
    { .why }

<figure class="plot" markdown>
![Line chart of the probability that a majority vote is correct for 1 to 41 samples. For p = 0.6: 0.600, 0.648, 0.683, then 0.903 at 41 samples. For p = 0.8 the line goes to 1. For p = 0.4 the line falls to 0.097.](../assets/figures/new-trends/majority-vote.svg#only-light)
![Line chart of the probability that a majority vote is correct for 1 to 41 samples. For p = 0.6: 0.600, 0.648, 0.683, then 0.903 at 41 samples. For p = 0.8 the line goes to 1. For p = 0.4 the line falls to 0.097.](../assets/figures/new-trends/majority-vote-dark.svg#only-dark)
<figcaption>A vote helps only when one sample is correct more often than not (p above 0.5).</figcaption>
</figure>

!!! success "Key point"
    More samples change a model that is often correct into a system that is almost always correct. They cannot repair a model that is usually wrong.

### From toy to real

- Real samples are not independent: the model makes the same error many times. The real gain is thus smaller than in the plot.
- A **process reward model (PRM)** is a second model that gives a score to each step of a solution, not only to the final answer. A search can then stop a bad solution early.
- **Tree search** uses these scores. Beam search keeps the best partial solutions at each step. **Monte Carlo tree search (MCTS)** repeats 4 actions: select a promising branch, add a new step, estimate its value, and send that value back up the tree.
- In the experiments of Hugging Face, a model with 3 billion parameters with search and a PRM got a better score on the MATH-500 test than Llama 3.1 70B.
- **Reasoning models** are trained to write many thinking tokens before the answer. A larger budget of such tokens is also test-time compute.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import numpy as np
from math import comb

a = np.array([0.5, -1.2, 0.8, 0.3])              # W
b = np.array([1.2, 0.3, 0.5, 0.8])               # W_B
theta = np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))   # 68.7 degrees
slerp = (np.sin(0.5 * theta) * a + np.sin(0.5 * theta) * b) / np.sin(theta)
print(np.linalg.norm(0.5 * a + 0.5 * b), np.linalg.norm(slerp))      # 1.285, 1.556

images = np.array([[0.8, 0.6], [0.6, -0.8]])     # CLIP loss
texts = np.array([[1.0, 0.0], [0.6, -0.8]])
logits = images @ texts.T / 0.5
rows = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
cols = np.exp(logits) / np.exp(logits).sum(axis=0, keepdims=True)
print((-np.log(np.diag(rows)).mean() - np.log(np.diag(cols)).mean()) / 2)   # 0.299

x = np.array([0.2, 1.0])                         # sparse autoencoder
D = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
f = np.maximum(D @ x - 0.5, 0)                   # [0, 0.5, 0.42]
print(np.sum((x - D.T @ f) ** 2) + 0.1 * f.sum())                    # 0.122

r = np.array([0.6, 0.8])                         # abliteration
print(x - (x @ r) * r)                           # [-0.352, 0.264]

vote = lambda p, n: sum(comb(n, k) * p**k * (1 - p)**(n - k) for k in range(n // 2 + 1, n + 1))
print(vote(0.6, 1), vote(0.6, 3), vote(0.6, 5))  # 0.6, 0.648, 0.683
```

## Comparison

| Topic | Method | What it does | Toy result | Needs training? |
|---|---|---|---|---|
| Merging | LERP | Mean of 2 models | Length 1.285 (inputs: 1.556) | No |
| Merging | SLERP | Point on the arc between 2 models | Length 1.556 | No |
| Merging | TIES, DARE | Add cleaned task vectors to a base | Weight 2: +0.5, not 0.1 | No |
| Multimodal | CLIP | Aligns image and text vectors | Loss 0.299 | Yes (contrastive) |
| Multimodal | LLaVA projector | Changes patches into tokens | `[-0.32, 0.82]` | Yes (the projector) |
| Interpretability | SAE | Splits an activation into features | 2 of 3 features active, loss 0.122 | Yes (the SAE only) |
| Interpretability | Abliteration | Removes one direction | `[-0.352, 0.264]` | No |
| Test-time compute | Majority vote | N samples, most frequent answer | 0.600 → 0.648 → 0.683 | No |

## Checks and common misconceptions

- **"A merge of two good models is a good model."** Not always. The models must share the architecture, and for task vectors also the base model. Check: evaluate the merged model on the tasks of each parent.
- **"An SAE feature is what the model thinks."** A feature is a direction that an SAE found with a loss that we chose. A different SAE size gives different features. Check a feature with a test: change it and see if the output changes as you expect.
- **"More samples always help."** Only when the most frequent answer is the correct one. Check: measure the accuracy of one sample first. If it is below 0.5 on a type of question, a vote makes it worse.

## Sources

This page follows section 8 of "The LLM Scientist" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [Merge LLMs with mergekit](https://mlabonne.github.io/blog/posts/2024-01-08_Merge_LLMs_with_mergekit.html), Maxime Labonne.
- [Smol Vision](https://github.com/merveenoyan/smol-vision), Merve Noyan.
- [Large Multimodal Models](https://huyenchip.com/2023/10/10/multimodal.html), Chip Huyen.
- [Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration), Maxime Labonne.
- [Intuitive Explanation of SAEs](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html), Adam Karvonen.
- [Scaling test-time compute](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute), Beeching et al. (the MATH-500 experiments).
- [TIES-Merging](https://arxiv.org/abs/2306.01708), Yadav et al., 2023, and [DARE](https://arxiv.org/abs/2311.03099), Yu et al., 2023 (drop 90% or 99% of the entries).
- [CLIP](https://arxiv.org/abs/2103.00020), Radford et al., 2021 (400 million pairs, batch of 32,768, learned temperature).
- [LLaVA](https://llava-vl.github.io/) and [LLaVA-1.5](https://arxiv.org/abs/2310.03744), Liu et al., 2023 (CLIP ViT-L at 336 pixels, MLP projector).
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features), Bricken et al., 2023 (512 neurons, 512 to 131,072 features).
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717), Arditi et al., 2024.
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171), Wang et al., 2022.
