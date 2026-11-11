# 11. Quantization beyond rounding

## Core idea

Simple quantization looks at each weight alone and rounds it. The newer methods use the model itself to decide.
They use its activations on real text, its own outputs as a teacher, text that it generates, or training.
Other new methods change what is stored: 3 values in place of 16, or one code for a group of weights.

## The example

This page is the second part of [Quantization](07-quantization.md). It does not explain the methods of that page again.
It uses the same weight matrix $W$ of the [toy example](../toy-example.md), and the 4-bit absmax result of lesson 7 is the baseline.

| Item | Value |
|---|---|
| Weight matrix $W$ | $\begin{bmatrix} 0.5 & -1.2 \\ 0.8 & 0.3 \end{bmatrix}$ |
| Baseline: 4 bits, each weight to the nearest value | $\hat{W} = \begin{bmatrix} 0.514 & -1.2 \\ 0.857 & 0.343 \end{bmatrix}$, mean absolute error 0.0286 |
| Toy model | $p = \text{softmax}(E\,W x)$: the probabilities of `the`, `cat`, `sat` |
| Calibration inputs (assumed) | $[2, 1]$, $[4, -1]$, $[-2, 1]$, $[4, 1]$ |
| Input of lesson 7 | $x = [2, 1]$, so $Wx = [-0.2, 1.9]$ |

$E$ holds the three embeddings of the toy example. For $x = [2, 1]$, the original model gives the probabilities [0.063, 0.515, 0.422]. The baseline gives [0.055, 0.513, 0.432].
The **KL divergence** measures how far the second list is from the first (see [Metrics](../fundamentals/05-metrics.md)). It is 0 when the two lists are equal. The mean over the 4 calibration inputs is $0.662 \times 10^{-3}$ for the baseline.

<figure class="diagram" data-src="quantization-beyond-rounding/four-ways.svg">
<figcaption>The four parts of this page, with the toy result of each part.</figcaption>
</figure>

## 1. Let the model say which weights matter

### Setup

**Calibration data** is a small set of real text. You run the model on it and record the activations.
The **importance matrix** holds one number for each input channel of a tensor: the mean squared activation of that channel. This is the imatrix of llama.cpp.
A rounding error on a weight is multiplied by the activation that the weight reads. Thus the **weighted rounding error** multiplies each squared error by the importance of its column:

$$\text{weighted error} = \sum_{i,j} \text{importance}_j \, (w_{ij} - \hat{w}_{ij})^2$$

### Steps

1. **Calculate the importance of each input channel.** Channel 1 of the calibration inputs holds 2, 4, -2, 4. Channel 2 holds 1, -1, 1, 1.

    $$\text{importance} = \left[\frac{4 + 16 + 4 + 16}{4},\ \frac{1 + 1 + 1 + 1}{4}\right] = [10,\ 1]$$

    **Why:** the weights do not show which channel is loud. Only the model that runs on real text shows it.

2. **Weight the rounding error of the baseline.** The squared errors of the two columns are 0.0035 and 0.0018. Column 1 has 65% of the plain error.

    $$10 \times 0.0035 + 1 \times 0.0018 = 0.0347 + 0.0018 = 0.0365$$

    **Why:** column 1 now has 95% of the error. In this toy, the weighted error is also the mean squared error of the layer output, because the two channels are not correlated.

3. **Try some grid steps and keep the smallest weighted error.** The baseline step is 1.2 / 7 = 0.171.

    | Step | 0.150 | 0.155 | 0.160 | 0.165 | 0.170 | 0.171 |
    |---|---|---|---|---|---|---|
    | Weighted error | 0.073 | 0.032 | 0.011 | **0.009** | 0.028 | 0.037 |

    The step 0.165 gives the same integers, $[[3, -7], [5, 2]]$, but different values:

    $$\hat{W} = \begin{bmatrix} 0.495 & -1.155 \\ 0.825 & 0.33 \end{bmatrix} \qquad \lvert W - \hat{W} \rvert = \begin{bmatrix} 0.005 & 0.045 \\ 0.025 & 0.03 \end{bmatrix}$$

    **Why:** the weight -1.2 had no error, and now it has the largest error (0.045). But it reads the quiet channel. The weights of the loud channel are now nearer to their true values.

4. **Measure the outputs of the model, not the weights.** The mean KL divergence falls from $0.662 \times 10^{-3}$ to $0.309 \times 10^{-3}$. The mean absolute error of the weights changes much less: 0.0286 to 0.0262.

    **Why:** the user sees the probabilities of the tokens, and the KL divergence compares them directly.

5. **Spend a bit budget on the layers.** The **sensitivity** of a layer is the output error that you get when you quantize only this layer. Assume 3 layers of the same size with the sensitivities 0.064, 0.016 and 0.004 at 4 bits. Assume that one more bit divides the error of a layer by 4, and that the errors add up. The budget is a mean of 4 **bits per weight**.

    | Layer | Sensitivity at 4 bits | Equal bits | Error | Bits by sensitivity | Error |
    |---|---|---|---|---|---|
    | 1 | 0.064 | 4 | 0.064 | 5 | 0.016 |
    | 2 | 0.016 | 4 | 0.016 | 4 | 0.016 |
    | 3 | 0.004 | 4 | 0.004 | 3 | 0.016 |
    | Total | | 12 | 0.084 | 12 | **0.048** |

    **Why:** one bit that moves from layer 3 to layer 1 removes 0.048 of error and adds 0.012. This is **mixed precision** (or dynamic quantization): different bit widths for different layers or tensors. The size is the same, and the error is 57% of the error of equal bits.

<figure class="plot">
<img src="/figures/quantization-beyond-rounding/bit-budget.svg" alt="Line chart of the total output error against the mean bits for each weight, from 2 to 6. At 4 bits: 0.084 with equal bits and 0.048 with bits given by sensitivity. The second line is below the first one at each budget above 2 bits.">
<figcaption>With the same budget, bits given by sensitivity have a smaller total error than equal bits.</figcaption>
</figure>

<div class="llm-widget" data-widget="bit-budget"></div>

<div class="key-point">

The model shows where a rounding error hurts: its activations give the importance of each weight, and its outputs give the sensitivity of each layer. The bits per weight are a budget. Spend them where the error is expensive.

</div>

### From toy to real

- The `llama-imatrix` tool of llama.cpp runs a model on a text file and writes `imatrix.gguf`. The pull request that added it (merged on 12 January 2024) keeps only the mean squared activation of each column, and uses it as the weight of a weighted error. Use it when you make a GGUF file of 4 bits or less: `llama-quantize --imatrix imatrix.gguf ...`.
- The Unsloth Dynamic GGUF files choose a quantization type for each layer, and the mix is different for each model. Unsloth says that the calibration set of version 2.0 has more than 1.5 million tokens, and it reports the KL divergence because errors can cancel out in the perplexity. It states a limit: calibration text that looks like the test text makes a file look better than it is. Version 3.1 (November 2026) is still post-training quantization only.
- GSQ (April 2026) learns the scalar value of each weight, so the result can stay in the GGUF k-quant format. Its authors claim that it closes most of the gap to QTIP (part 4) at 2 and 3 bits on Llama 3.1 8B and 70B.

## 2. Let the model teach its quantized copy

### Setup

The methods above are **post-training quantization (PTQ)**: the weights do not learn. **Quantization-aware training (QAT)** trains the model with the rounding in the loop.
**Fake quantization** rounds the weights in the forward pass, but keeps a full-precision copy of each weight, the **latent weight**.
Rounding has a gradient of 0 almost everywhere. With the **straight-through estimator (STE)**, the gradient passes the rounding as if it was not there, and updates the latent weight.
In **self-distillation**, the full-precision model is the teacher. The loss is the KL divergence or the squared error between the outputs of the teacher and of the quantized copy.

The toy is one neuron: row 2 of $W$, $w = [0.8, 0.3]$, with $x = [2, 1]$. The teacher output is $y = 1.9$. Assume a 2-bit grid with the step 0.625: the integers -2 to 1 give -1.25, -0.625, 0 and 0.625. The loss is $\frac{1}{2}(\hat{w} \cdot x - y)^2$ and the learning rate is 0.008.

### Steps

1. **Round in the forward pass.** $0.8 / 0.625 = 1.28$ gives 1, and $0.3 / 0.625 = 0.48$ gives 0. So $\hat{w} = [0.625, 0]$ and the output is $0.625 \times 2 + 0 = 1.25$. The error is $1.25 - 1.9 = -0.65$.

    **Why:** the loss must see the weights that the final model will use. This first result is plain rounding.

2. **Calculate the gradient and pass it through the rounding.** The gradient of the loss for $\hat{w}$ is the error times the input:

    $$g = (\hat{w} \cdot x - y)\,x = -0.65 \times [2,\ 1] = [-1.3,\ -0.65]$$

    **Why:** the true gradient for the latent weight is 0, because a small change of 0.3 still rounds to 0. With a gradient of 0, nothing learns. The STE uses $g$ for the latent weight.

3. **Update the latent weights.** Each step subtracts $0.008 \times g$.

    | Step | Latent weights | Rounded weights | Output | Error |
    |---|---|---|---|---|
    | 0 | [0.8, 0.3] | [0.625, 0] | 1.25 | -0.65 |
    | 1 | [0.810, 0.305] | [0.625, 0] | 1.25 | -0.65 |
    | 2 | [0.821, 0.310] | [0.625, 0] | 1.25 | -0.65 |
    | 3 | [0.831, 0.316] | [0.625, 0.625] | 1.875 | -0.025 |
    | 4 | [0.832, 0.316] | [0.625, 0.625] | 1.875 | -0.025 |

    **Why:** the latent weight collects many small updates. For 2 steps, the output does not change. Then the second latent weight passes the rounding boundary, 0.3125, and the rounded weight jumps to 0.625.

4. **Compare with plain rounding.** The output error falls from 0.65 to 0.025: 26 times smaller. The weight 0.3 is now stored as 0.625. Its own error is larger (0.325, before 0.3).

    **Why:** the weight 0.8 cannot go above 0.625 on this grid, so the output was too small. Training rounds the second weight up to pay for this. PTQ cannot find this without the output of the teacher. After step 3 the gradient is small, $[-0.05, -0.025]$, and the weights stay.

<div class="key-point">

In QAT the full-precision model is the teacher, and the latent weights move until the rounded weights give its output. The cost is training: gradients, data and GPU hours.

</div>

### From toy to real

- **Data-free QAT** needs no training set: the model generates the text itself. LLM-QAT (May 2023) does this, then distills the full-precision model into a copy with quantized weights, activations and KV cache. The paper tests LLaMA models of 7, 13 and 30 billion parameters down to 4 bits.
- **Block-wise training** trains one Transformer block at a time, so the full model never needs gradients at once. A **learned step size** makes the grid step a trained parameter. EfficientQAT (July 2024) uses the two: first all the parameters of each block, then only the step sizes of the full model. Its authors report a 2-bit Llama 2 70B after 41 hours on one A100 GPU of 80 GB, with an accuracy of 69.48 against 72.41 for full precision.
- Google released QAT files of Gemma 3 on 18 April 2025. It says that it trained about 5,000 steps with the probabilities of the non-quantized model as targets. It reports that the perplexity drop of the `Q4_0` format is 54% smaller, and that the memory to load Gemma 3 27B falls from 54 GB to 14.1 GB. That number does not include the KV cache. Use such a file when the maker gives one.
- PV-Tuning (May 2024) argues that the STE is not the best choice at 1 to 2 bits per weight.

## 3. Store 3 values: ternary and 1.58-bit models

### Setup

A **binary** weight has 2 values: -1 and 1. A **ternary** weight has 3 values: -1, 0 and 1.
One choice between 3 values holds $\log_2 3 = 1.585$ bits. This is the "1.58" of the name.
**Absmean scaling** uses one scale for a matrix, the mean absolute weight $\gamma$:

$$\hat{W} = \gamma \cdot \text{clip}\left(\text{round}\left(\frac{W}{\gamma}\right),\ -1,\ 1\right)$$

### Steps

1. **Calculate the scale.** $\gamma = (0.5 + 1.2 + 0.8 + 0.3) / 4 = 0.7$.

    **Why:** absmax would use 1.2. Then each weight below 0.6 becomes 0. The mean keeps more weights away from 0.

2. **Divide, round and clip.**

    $$\frac{W}{\gamma} = \begin{bmatrix} 0.714 & -1.714 \\ 1.143 & 0.429 \end{bmatrix} \quad\Rightarrow\quad Q = \begin{bmatrix} 1 & -1 \\ 1 & 0 \end{bmatrix}$$

    **Why:** -1.714 rounds to -2, which is not a permitted value. The clip makes it -1.

3. **Dequantize and measure.**

    $$\hat{W} = \begin{bmatrix} 0.7 & -0.7 \\ 0.7 & 0 \end{bmatrix} \qquad \lvert W - \hat{W} \rvert = \begin{bmatrix} 0.2 & 0.5 \\ 0.1 & 0.3 \end{bmatrix}$$

    The mean absolute error is 0.275: 9.6 times the 4-bit baseline. The output for $x = [2, 1]$ is $[0.7, 1.4]$, not $[-0.2, 1.9]$. The mean KL divergence is 0.072: 108 times the baseline.

    **Why:** a trained model does not survive this rounding. So a ternary model is trained with ternary weights from the start, with the method of part 2. Then the weights learn values that are correct after the rounding.

4. **Count the bits.** $4 / 1.585 = 2.52$. A model with 8 billion parameters needs 1.585 GB, against 4 GB with 4 bits. A simple file stores 2 bits for each ternary weight: 2 GB.

    **Why:** the small size is the reason to accept so few values.

5. **Calculate an output without multiplications.** Row 1 of $Q$ is $[1, -1]$: the output is $\gamma (x_1 - x_2) = 0.7 \times (2 - 1) = 0.7$. Row 2 is $[1, 0]$: the output is $\gamma (x_1 + 0) = 1.4$.

    **Why:** a weight of 1 adds the input, -1 subtracts it and 0 skips it. One multiplication by $\gamma$ stays for the full row.

<div class="key-point">

A ternary model is not a rounded copy of a normal model. It is a model that learned with 3 values from its first training step.

</div>

### From toy to real

- BitNet b1.58 (February 2024) uses absmean scaling for the weights and 8 bits for the activations, and trains from scratch. With 3 billion parameters, the authors report a perplexity of 9.91 against 10.04 for their own FP16 LLaMA model, with 2.22 GB of memory against 7.89 GB. With 700 million parameters, BitNet is worse: 12.87 against 12.33. Its energy numbers come from a model of the arithmetic, not from a measurement.
- BitNet b1.58 2B4T (April 2025) is an open model with 2 billion parameters, trained from scratch on 4 trillion tokens. The report gives 0.4 GB of memory without the embeddings and 29 ms for each token on one laptop CPU. The stated limits: only one size, and GPUs that are not built for 1-bit models.
- bitnet.cpp runs these models. Its page claims a speed 2.37 to 6.17 times higher than full precision on x86 CPUs. Use it only with a ternary model.

## 4. Store a code for a group of weights

### Setup

**Scalar quantization** stores one number for each weight. **Vector quantization** stores one number for a group of weights.
The **codebook** is a list of vectors. The **index** is the position of one vector in the list. The file holds one index for each group.

$$\text{bits per weight} = \frac{\log_2(\text{codebook size})}{\text{group size}}$$

The groups are pairs of weights. The 2 rows of $W$ are the first 2 pairs. Assume 6 more pairs from the same layer.

| Pair | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Weights | [0.5, -1.2] | [0.8, 0.3] | [0.6, -1.0] | [0.9, 0.4] | [-0.7, 0.2] | [-0.5, 0.1] | [-0.2, -0.9] | [-0.1, -0.7] |

### Steps

1. **Build a codebook of 4 vectors with k-means.** Start with the pairs 1, 2, 5 and 7. Give each pair to its nearest vector. Move each vector to the mean of its pairs. The second step changes nothing.

    $$\text{codebook} = [0.55, -1.1],\ [0.85, 0.35],\ [-0.6, 0.15],\ [-0.15, -0.8]$$

    **Why:** the pairs are not spread evenly. They form 4 groups, and the codebook puts one vector on each group. A scalar grid treats the two weights of a pair separately, so it cannot use this.

2. **Store one index for each pair.** The indices of the 8 pairs are 0, 1, 0, 1, 2, 2, 3, 3. Each index needs $\log_2 4 = 2$ bits for 2 weights: 1 bit per weight.

    $$\hat{W} = \begin{bmatrix} 0.55 & -1.1 \\ 0.85 & 0.35 \end{bmatrix} \qquad \lvert W - \hat{W} \rvert = \begin{bmatrix} 0.05 & 0.1 \\ 0.05 & 0.05 \end{bmatrix}$$

    The mean absolute error of $W$ is 0.0625, and the mean KL divergence is 0.004.

    **Why:** to dequantize is to read the list. No scale and no rounding are necessary.

3. **Compare with scalar quantization at the same 1 bit per weight.** A 1-bit scalar stores the sign. The value is the sign times the mean absolute weight of the 16 weights, 0.569.

    | Method, 1 bit per weight | $\hat{W}$ | Mean absolute error of $W$ | Mean KL |
    |---|---|---|---|
    | Sign × 0.569 | [[0.569, -0.569], [0.569, 0.569]] | 0.300 | 0.058 |
    | Codebook of 4 pairs | [[0.55, -1.1], [0.85, 0.35]] | 0.0625 | 0.004 |

    **Why:** with the same number of index bits, the codebook error is about 5 times smaller. The codebook can hold the value -1.1, because each vector has its own values.

4. **Count the cost of the codebook.** The codebook holds 4 vectors × 2 values × 16 bits = 128 bits. The toy has 16 weights: $1 + 128 / 16 = 9$ bits per weight. Assume a real layer of 4,096 × 4,096 = 16,777,216 weights: $1 + 128 / 16{,}777{,}216 = 1.0000076$ bits per weight.

    **Why:** the codebook is a fixed cost. It is small only when the layer has many more weights than the codebook has values. A codebook of $2^{16}$ vectors of 8 weights gives 2 bits per weight, but it holds 8,388,608 bits. On the layer above, that is 0.5 bit per weight more.

<div class="key-point">

A codebook stores shapes that occur in the weights, and an index says which shape to use. It gives a smaller error than a scalar grid at the same index bits, but the codebook itself needs space and a table read at inference.

</div>

### From toy to real

- **Additive quantization** stores a group as the sum of several vectors, one from each of several codebooks. AQLM (January 2024) uses groups of 8 weights and learns the codebooks on calibration data, then tunes them block by block. For Llama 2 7B, the authors report a WikiText-2 perplexity of 6.59 at 2.02 bits, against 5.12 for FP16.
- A **lattice codebook** is a regular pattern of points, so most of it is calculated and not stored. **Incoherence processing** multiplies the weights by a rotation before quantization. The rotation spreads the outliers over all the weights, and the inverse rotation at inference gives the same output. QuIP# (February 2024) uses the two: a Hadamard rotation, and a codebook of $2^{16}$ entries built from a table of 256 entries of about 1 KiB. For Llama 2 70B, it reports a perplexity of 3.32 for FP16, 3.56 at 3 bits and 4.16 at 2 bits.
- SpinQuant (May 2024) learns the rotation, for models with 4-bit weights, activations and KV cache.
- QTIP (June 2024) says that the size of a codebook grows exponentially with the group size, which limits vector quantization to 8 dimensions or less. Its trellis codes remove this limit. **EXL3**, the format of ExLlamaV3, is a simpler variant of QTIP. It needs only the model and a target bitrate. Its page gives minutes for small models and some hours for 70 billion parameters on one consumer GPU. Use it on a GPU below 4 bits per weight.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import numpy as np

W = np.array([[0.5, -1.2], [0.8, 0.3]])
X = np.array([[2., 1.], [4., -1.], [-2., 1.], [4., 1.]])   # calibration inputs (assumed)
grid = lambda w, step, lo, hi: np.clip(np.round(w / step), lo, hi) * step

# 1. Importance: mean squared activation of each column, then the weighted error
importance = (X ** 2).mean(axis=0)                         # [10, 1]
weighted = lambda w_hat: ((W - w_hat) ** 2).sum(axis=0) @ importance
print(weighted(grid(W, 1.2 / 7, -7, 7)), weighted(grid(W, 0.165, -7, 7)))   # 0.037, 0.009

# 2. QAT: fake quantization and the straight-through estimator
x, latent = X[0], W[1].copy()
for _ in range(3):
    w_hat = grid(latent, 0.625, -2, 1)                     # forward pass: rounded weights
    latent -= 0.008 * (w_hat @ x - W[1] @ x) * x           # STE: the gradient goes to the latent weights
print(grid(latent, 0.625, -2, 1) @ x - W[1] @ x)           # -0.025 (plain rounding: -0.65)

# 3. Ternary weights with absmean scaling
gamma = np.abs(W).mean()                                   # 0.7
Q = np.clip(np.round(W / gamma), -1, 1)                    # [[1, -1], [1, 0]]

# 4. Codebook: one 2-bit index for each pair of weights
codebook = np.array([[0.55, -1.1], [0.85, 0.35], [-0.6, 0.15], [-0.15, -0.8]])
index = np.linalg.norm(W[:, None] - codebook[None], axis=2).argmin(axis=1)   # [0, 1]
print(np.abs(W - codebook[index]).mean())                  # 0.0625
```

## Comparison

| Method | How it uses the model | Bits per weight | Needs | Toy result | Use it when |
|---|---|---|---|---|---|
| Plain rounding (lesson 7) | It does not | Toy 4; lesson 7: 4 to 8 | Nothing | Weighted error 0.037; KL 0.662 × 10⁻³ | You need a baseline |
| Importance matrix | Activations on real text | Toy 4 | Calibration text | Weighted error 0.009; KL 0.309 × 10⁻³ | You make a GGUF file of 4 bits or less |
| Mixed precision | Sensitivity of each layer | Toy 5 / 4 / 3 | Calibration text, many tests | Total error 0.048, against 0.084 | You have a fixed file size |
| QAT with self-distillation | Outputs as a teacher; generated text | Toy 2; papers above: 2 to 4 | Training on GPUs | Output error 0.65 → 0.025 | PTQ is not good enough, or the maker gives a QAT file |
| Ternary (1.58-bit) | Trained with 3 values from the start | 1.585 | Full training | After training only: KL 0.072 | You can train a model from scratch |
| Codebook (vector quantization) | Calibration data and block tuning | Toy 1; papers above: 2 to 4 | Calibration text, GPU hours | Error 0.0625, against 0.300 for 1-bit scalar | You need less than 4 bits on a GPU |

## Checks and common misconceptions

- **"A 2-bit model of a large network is always better than a 4-bit model of a small one."** It depends on the method and on the task. Check: compare the two files at the same size in GB on your own task (see [Evaluation](06-evaluation.md)), and measure the KL divergence to the full-precision model on text that is not the calibration text.
- **"QAT is always better than PTQ."** It can reach a lower error, but it costs training: 41 GPU hours for one model in the EfficientQAT paper. Check: quantize with PTQ first, and train only if the measured loss is too large.
- **"You can convert any model to 1.58 bits."** The toy shows a KL divergence 108 times larger after ternary rounding. Check: read how the model was trained.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) covers the methods of [lesson 7](07-quantization.md); it has no section on the methods of this page. The text and the example are new. Each statement about a real system was read in the linked source on 11 November 2026.

- [llama.cpp imatrix tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md) and the [importance matrix pull request](https://github.com/ggml-org/llama.cpp/pull/4861) (output file, `--imatrix` option, mean squared activation, merge date).
- [Unsloth Dynamic GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs), Unsloth (a type for each layer, 1.5 million tokens, KL divergence, the limit, version 3.1).
- [GSQ](https://arxiv.org/abs/2604.18556), Dadgarnia et al., 2026 (scalar values, GGUF k-quants, 2 and 3 bits).
- [LLM-QAT](https://arxiv.org/abs/2305.17888), Liu et al., 2023 (data-free distillation with generated text, model sizes, 4 bits).
- [EfficientQAT](https://arxiv.org/abs/2407.11062), Chen et al., 2024 (the two phases, 41 hours, 69.48 against 72.41).
- [Gemma 3 QAT models](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/), Google, 18 April 2025 (about 5,000 steps, the targets, 54%, 54 GB and 14.1 GB).
- [PV-Tuning](https://arxiv.org/abs/2405.14852), Malinovskii et al., 2024 (the limits of the STE at 1 to 2 bits).
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764), Ma et al., 2024 (absmean scaling, 8-bit activations, perplexity and memory numbers, the energy model).
- [BitNet b1.58 2B4T Technical Report](https://arxiv.org/abs/2504.12285), Ma et al., 2025 (2 billion parameters, 4 trillion tokens, 0.4 GB, 29 ms, the limits) and [bitnet.cpp](https://github.com/microsoft/BitNet) (the speed claim on x86 CPUs).
- [AQLM](https://arxiv.org/abs/2401.06118), Egiazarian et al., 2024 (groups of 8, learned codebooks, block tuning, perplexity 6.59 and 5.12).
- [QuIP#](https://arxiv.org/abs/2402.04396), Tseng et al., 2024 (Hadamard rotation, the lattice codebook and its table, perplexity numbers).
- [SpinQuant](https://arxiv.org/abs/2405.16406), Liu et al., 2024 (learned rotations, 4-bit weights, activations and KV cache).
- [QTIP](https://arxiv.org/abs/2406.11235), Tseng et al., 2024 (the limit of 8 dimensions, trellis codes) and [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (EXL3: a variant of QTIP, its inputs, the conversion time).
