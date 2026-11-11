# 7. Quantization

## Core idea

A model stores each weight as a number with many bits. **Quantization** stores each weight with fewer bits.
The model becomes smaller and faster, but each weight moves a little, so the output has a small error.
All the methods on this page try to keep that error small.

## The example

This page uses the weight matrix $W$ of the [toy example](../toy-example.md).

| Item | Value |
|---|---|
| Weight matrix $W$ | $\begin{bmatrix} 0.5 & -1.2 \\ 0.8 & 0.3 \end{bmatrix}$ |
| Layer output | $y = Wx$ (column $j$ of $W$ multiplies input channel $j$) |
| Calibration input (assumed) | $x = [2, 1]$, so $y = [-0.2, 1.9]$ |
| Largest activation of each channel (assumed, part 4) | $[16, 1]$ |

An **activation** is a number that a layer gives to the next layer. Here $x$ holds 2 activations.

## 1. Base techniques

### Setup

A **floating-point** number (FP32, FP16) stores a sign, an exponent and a fraction. It can be very small or very large.
An **integer** (INT8) stores one of 256 whole numbers, from -128 to 127.
To quantize, we need a rule that changes each real weight into an integer, and a rule that changes it back.

### Steps

1. **Calculate the size at each precision.** Size = parameters × bits ÷ 8. A **parameter** is one learned number.

    | Precision | Bits | Bytes for each weight | Model with 8 billion parameters |
    |---|---|---|---|
    | FP32 | 32 | 4 | 32 GB |
    | FP16 | 16 | 2 | 16 GB |
    | INT8 | 8 | 1 | 8 GB |
    | 4-bit | 4 | 0.5 | 4 GB |

    **Why:** the weights must fit in the memory of your GPU or laptop. The size tells you which precision you need.
    { .why }

2. **Absmax: calculate the scale.** **Absmax** (symmetric) quantization maps the largest absolute weight to 127.

    $$s = \frac{127}{\max |W|} = \frac{127}{1.2} = 105.833$$

    **Why:** one multiplication stretches the range of the weights, [-1.2, 1.2], to the range of the integers, [-127, 127].
    { .why }

3. **Multiply by the scale and round.** $q = \text{round}(s \cdot w)$.

    $$s \cdot W = \begin{bmatrix} 52.917 & -127 \\ 84.667 & 31.75 \end{bmatrix} \quad\Rightarrow\quad Q = \begin{bmatrix} 53 & -127 \\ 85 & 32 \end{bmatrix}$$

    **Why:** an integer has no fraction. The rounding is the step that loses information.
    { .why }

4. **Dequantize and measure the error.** **Dequantization** divides by the scale: $\hat{w} = q / s$.

    $$\hat{W} = \begin{bmatrix} 0.5008 & -1.2 \\ 0.8031 & 0.3024 \end{bmatrix} \qquad |W - \hat{W}| = \begin{bmatrix} 0.0008 & 0 \\ 0.0031 & 0.0024 \end{bmatrix}$$

    The mean absolute error is 0.0016.

    **Why:** the model calculates with $\hat{W}$, not with $W$. The error tells you how far the new model is from the old one.
    { .why }

5. **Zero-point: use the full integer range.** **Zero-point** (asymmetric) quantization maps the minimum weight to -128 and the maximum weight to 127. The **zero point** $z$ is the integer that represents the real number 0.

    $$s = \frac{255}{\max W - \min W} = \frac{255}{0.8 - (-1.2)} = 127.5 \qquad z = \text{round}(-s \cdot \min W) - 128 = 153 - 128 = 25$$

    $$Q = \text{round}(s \cdot W + z) = \begin{bmatrix} 89 & -128 \\ 127 & 63 \end{bmatrix} \qquad \hat{W} = \frac{Q - z}{s} = \begin{bmatrix} 0.502 & -1.2 \\ 0.8 & 0.298 \end{bmatrix}$$

    The mean absolute error is 0.0010.

    **Why:** the weights go from -1.2 to 0.8. Absmax keeps integers for the range 0.8 to 1.2, where there is no weight. Zero-point quantization does not, so its steps are smaller.
    { .why }

6. **Do it again with 4 bits.** A 4-bit integer has only 16 values. Absmax uses -7 to 7, so $s = 7 / 1.2 = 5.833$.

    $$Q = \begin{bmatrix} 3 & -7 \\ 5 & 2 \end{bmatrix} \qquad \hat{W} = \begin{bmatrix} 0.514 & -1.2 \\ 0.857 & 0.343 \end{bmatrix} \qquad |W - \hat{W}| = \begin{bmatrix} 0.014 & 0 \\ 0.057 & 0.043 \end{bmatrix}$$

    The mean absolute error is 0.0286: 18 times the INT8 error. Zero-point with 4 bits ($s = 7.5$, $z = 1$) gives $Q = [[5, -8], [7, 3]]$ and an error of 0.0167.

    **Why:** each bit that you remove makes the step between two values 2 times larger. The step is now 1.2 / 7 = 0.171, so a weight can move by 0.086.
    { .why }

<figure class="diagram">
--8<-- "docs/assets/figures/quantization/number-line.svg"
<figcaption>Step 6: each weight moves to the nearest of the 15 positions of the 4-bit grid.</figcaption>
</figure>

<figure class="plot" markdown>
![Line chart of the mean absolute error of W for 2 to 8 bits. Absmax: 0.300 at 2 bits, 0.0286 at 4 bits, 0.0016 at 8 bits. Zero-point: 0.183, 0.0167, 0.0010.](../assets/figures/quantization/error-by-bits.svg#only-light)
![Line chart of the mean absolute error of W for 2 to 8 bits. Absmax: 0.300 at 2 bits, 0.0286 at 4 bits, 0.0016 at 8 bits. Zero-point: 0.183, 0.0167, 0.0010.](../assets/figures/quantization/error-by-bits-dark.svg#only-dark)
<figcaption>The error falls when you add bits. The lines are not smooth because the example has only 4 weights.</figcaption>
</figure>

<div class="llm-widget" data-widget="quantization"></div>

!!! success "Key point"
    Quantization is "multiply by a scale, round, store the integer". The scale comes from the range of the weights, so one very large weight makes the steps larger for all the other weights.

### From toy to real

- This method is **post-training quantization (PTQ)**: you quantize a model that is already trained. The other family, quantization-aware training, trains the model with the rounding in the loop.
- Real tools do not use one scale for a full model. They use one scale for each row or for each small block of weights. A smaller group has a smaller range, thus smaller steps.
- A weight much larger than the others is an **outlier**. Parts 3 and 4 are about outliers. For example, LLM.int8() keeps the outlier values in FP16 and quantizes the rest to INT8.

## 2. GGUF and llama.cpp

### Setup

**llama.cpp** is a C++ program that runs LLMs on a CPU, with optional help from a GPU. **GGUF** is its file format: one file holds all that you need to run the model.

<figure class="diagram">
--8<-- "docs/assets/figures/quantization/gguf-file.svg"
<figcaption>A GGUF file holds the settings, the tokenizer and the weights. You need no other file.</figcaption>
</figure>

### Steps

1. **Read the name of the file.** A name such as `Q4_K_M` has three parts.

    | Part | Meaning |
    |---|---|
    | `Q4` | Most weights use 4 bits |
    | `K` | The "k-quant" scheme: blocks of weights, and each block has its own scale |
    | `M` | Medium mix (`S` small, `M` medium, `L` large): some important tensors get more bits |

    **Why:** the name tells you the size and the quality before you download many gigabytes.
    { .why }

2. **Calculate the bits for each weight.** The type `Q4_K` puts 256 weights in one **super-block** of 8 blocks × 32 weights. Each block stores a 6-bit scale and a 6-bit minimum. The super-block stores 2 FP16 numbers.

    $$\frac{256 \times 4 + 8 \times (6 + 6) + 2 \times 16}{256} = \frac{1152}{256} = 4.5 \text{ bits per weight}$$

    **Why:** a "4-bit" file is larger than 4 bits per weight, because the scales must be in the file too. The small blocks cost 0.5 bit, and they give each group of 32 weights its own range.
    { .why }

3. **Calculate the file size.** `Q4_K_M` uses the 6-bit type `Q6_K` (6.5625 bits per weight) for some tensors. llama.cpp measures 4.8944 bits per weight for Llama 3.1 8B.

    $$8 \times 10^9 \times 4.8944 \div 8 = 4.89 \text{ GB} \qquad \text{(FP16: 16 GB)}$$

    **Why:** with this number you know if the model fits in memory. The full file is loaded to run the model.
    { .why }

4. **Choose how many layers go on the GPU.** The option `-ngl N` (number of GPU layers) puts N layers in GPU memory. The CPU runs the others. Assume that the 32 layers have the same size: with `-ngl 16`, the GPU holds 4.89 × 16 ÷ 32 = 2.45 GB of weights.

    **Why:** a GPU is faster, but its memory is small. The split lets you use a model that is larger than the GPU memory.
    { .why }

!!! success "Key point"
    GGUF is one file with the weights, the tokenizer and the settings. The name gives the bits. Bits per weight × parameters ÷ 8 gives the size.

### From toy to real

- The measured `Q4_K_M` file of Llama 3.1 8B is 4.58 GiB (1 GiB = 2³⁰ bytes), which is 4.92 GB. It is a little more than our 4.89 GB because that model has a little more than 8 billion parameters.
- The usual steps are: convert the model to an FP16 GGUF file, then run the `llama-quantize` tool to get `Q4_K_M` or another type.
- The llama.cpp tutorial of the course recommends `Q5_K_M`, or `Q4_K_M` to save memory. `Q4_K_M` is about 31% of the FP16 size.

## 3. GPTQ and AWQ

### Setup

Part 1 made the error of each **weight** small. But the next layer does not see the weights. It sees the **output** $y = Wx$.
GPTQ and AWQ use **calibration** data: a small set of real inputs that shows which activations each layer gets. They work on one layer at a time and minimize $\lVert Wx - \hat{W}x \rVert$.
We use 4 bits (step 0.171) and the calibration input $x = [2, 1]$.

### Steps

1. **Round to the nearest value and measure the output error.** This is the baseline of part 1, step 6.

    $$\hat{W}x = \begin{bmatrix} 0.514 & -1.2 \\ 0.857 & 0.343 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \begin{bmatrix} -0.171 \\ 2.057 \end{bmatrix} \qquad \lVert Wx - \hat{W}x \rVert = \lVert [-0.029,\ -0.157] \rVert = 0.160$$

    **Why:** in row 2 the two weights are rounded up (0.8 → 0.857, 0.3 → 0.343). The two errors have the same sign, so they add up in the output.
    { .why }

2. **GPTQ idea: after you round one weight, change the weights that are not yet rounded.** Round column 1. The output of row 2 changes by $(0.857 - 0.8) \times 2 = 0.114$. Move the second weight to cancel this change, then round it.

    $$w_2' = w_2 + (w_1 - \hat{w}_1)\frac{x_1}{x_2} = 0.3 + (0.8 - 0.857) \times 2 = 0.186 \ \rightarrow\ 0.171$$

    Row 1 gives $-1.2 + (0.5 - 0.514) \times 2 = -1.229$, which stays at -1.2 (the end of the grid).

    $$\hat{W} = \begin{bmatrix} 0.514 & -1.2 \\ 0.857 & 0.171 \end{bmatrix} \qquad \hat{W}x = \begin{bmatrix} -0.171 \\ 1.886 \end{bmatrix} \qquad \text{error} = 0.032$$

    **Why:** the weight 0.3 is now rounded *down* to 0.171, which is not its nearest value. Its error is larger, but it has the opposite sign of the first error. The output error falls from 0.160 to 0.032.
    { .why }

3. **AWQ idea: give more precision to the weights that read large activations.** Multiply column $j$ of $W$ by $s_j = |x_j|^\alpha$ and divide input $j$ by $s_j$. The output does not change, because $(w \cdot s)(x / s) = wx$. With $\alpha = 1$: $s = [2, 1]$.

    $$W \cdot s = \begin{bmatrix} 1.0 & -1.2 \\ 1.6 & 0.3 \end{bmatrix} \ \xrightarrow{\text{4 bits, step } 0.229}\ \begin{bmatrix} 0.914 & -1.143 \\ 1.6 & 0.229 \end{bmatrix} \qquad \hat{y} = \begin{bmatrix} -0.229 \\ 1.829 \end{bmatrix}$$

    AWQ tries some values of $\alpha$ and keeps the best one.

    | Value of α | 0 (no scaling) | 0.5 | 1 |
    |---|---|---|---|
    | Output error | 0.160 | 0.143 | **0.077** |

    **Why:** a rounding error of a weight is multiplied by its activation. Channel 1 has the largest activation (2), so its weights are the most important. A larger weight has a smaller *relative* rounding error.
    { .why }

!!! success "Key point"
    The best rounding for one weight is not the best rounding for the layer. GPTQ repairs the output after each rounding. AWQ scales the important weights before the rounding. The two methods need calibration data.

### From toy to real

- **GPTQ** calculates the correction with the matrix $H = 2XX^\top$, where $X$ holds all the calibration inputs. $H$ is the **Hessian** (the second derivatives) of the layer error. For 2 weights and 1 input, its formula becomes the formula of step 2. The paper uses 128 text segments of 2,048 tokens and quantizes a model with 175 billion parameters in about 4 GPU hours, to 3 or 4 bits.
- **EXL2** (the format of ExLlamaV2) uses the same method as GPTQ. It can mix bit widths to get an average from 2 to 8 bits per weight.
- **AWQ** uses the mean absolute activation of each channel for $s$. It does not use gradients, and it also clips very large weights.

## 4. SmoothQuant and ZeroQuant

### Setup

To calculate with integers only, you must quantize the weights *and* the activations to INT8.
Activations are harder: some channels are much larger than the others. Assume that channel 1 reaches 16 and channel 2 reaches 1.
**SmoothQuant** moves the scale from the activations to the weights, with one factor $s_j$ for each channel:

$$y = Wx = (W \cdot s)(x / s) \qquad s_j = \frac{\max|x_j|^{\alpha}}{\max|W_j|^{1-\alpha}}$$

$W_j$ is column $j$ of $W$. The usual value is $\alpha = 0.5$.

### Steps

1. **Count the integer steps that each channel uses.** One absmax scale for all the activations maps 16 to 127. Channel 2 only reaches $1 / 16 \times 127 = 7.9$.

    **Why:** channel 2 has 8 steps, not 127. It is quantized with about 4 bits, and the outlier channel is the cause.
    { .why }

2. **Calculate the factor of each channel.** The column maxima of $W$ are $[0.8, 1.2]$.

    $$s_1 = \frac{\sqrt{16}}{\sqrt{0.8}} = 4.472 \qquad s_2 = \frac{\sqrt{1}}{\sqrt{1.2}} = 0.913$$

    **Why:** a large activation range gives a large $s$, which makes that channel smaller. A large weight range gives a small $s$, so the weights do not become the new problem.
    { .why }

3. **Divide the activations and multiply the weights.**

    | | Channel 1 | Channel 2 | Steps used by the smallest channel |
    |---|---|---|---|
    | Activation maxima before | 16 | 1 | 7.9 of 127 |
    | Weight maxima before | 0.8 | 1.2 | 84.7 of 127 |
    | Activation maxima after | 3.578 | 1.095 | 38.9 of 127 |
    | Weight maxima after | 3.578 | 1.095 | 38.9 of 127 |

    $$W \cdot s = \begin{bmatrix} 2.236 & -1.095 \\ 3.578 & 0.274 \end{bmatrix}$$

    **Why:** with $\alpha = 0.5$, the activations and the weights of a channel get the same maximum. The two sides share the difficulty: the worst channel goes from 7.9 steps to 38.9 steps.
    { .why }

4. **Check that the output is the same.** For $x = [16, 1]$: $Wx = [6.8, 13.1]$, and $(W \cdot s)(x / s) = [6.8, 13.1]$.

    **Why:** the change is exact before quantization. It costs nothing at run time, because $s$ is merged into the weights of this layer and of the layer before.
    { .why }

!!! success "Key point"
    SmoothQuant does not round better. It changes the numbers so that normal rounding hurts less. It needs calibration data only to find the largest activation of each channel.

### From toy to real

- The SmoothQuant paper reports INT8 weights and activations for LLMs, with up to 1.56× faster inference and 2× less memory.
- **ZeroQuant** (in the DeepSpeed library) is a full pipeline, not one formula. Its paper lists three parts: a fine-grained quantization scheme for weights and activations, a layer-by-layer **knowledge distillation** (each quantized layer learns to copy the output of the original layer) that does not need the original training data, and an inference backend that removes the cost of the quantize and dequantize operations. It reports a mixed INT4/INT8 model with a memory footprint 3 times smaller than FP16.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import numpy as np

W = np.array([[0.5, -1.2], [0.8, 0.3]])

def absmax(w, bits=8):
    scale = (2 ** (bits - 1) - 1) / np.abs(w).max()
    q = np.round(scale * w)                      # [[53, -127], [85, 32]]
    return q, q / scale                          # integers, dequantized values

def zeropoint(w, bits=8):
    qmin, qmax = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    scale = (qmax - qmin) / (w.max() - w.min())  # 127.5
    zero = np.round(-scale * w.min()) + qmin     # 25
    q = np.clip(np.round(scale * w + zero), qmin, qmax)
    return q, (q - zero) / scale

x = np.array([2.0, 1.0])                         # calibration input
_, w_rtn = absmax(W, bits=4)
print(np.linalg.norm(W @ x - w_rtn @ x))         # 0.160: round to nearest

step = 1.2 / 7                                   # GPTQ idea: repair column 2
moved = W[:, 1] + (W[:, 0] - w_rtn[:, 0]) * x[0] / x[1]
w_gptq = np.stack([w_rtn[:, 0], np.clip(np.round(moved / step), -7, 7) * step], axis=1)
print(np.linalg.norm(W @ x - w_gptq @ x))        # 0.032
```

## Comparison

| Method | Needs calibration data | What it changes | Toy result | Use |
|---|---|---|---|---|
| Absmax | No | Each weight → nearest value, symmetric range | Error 0.0016 (INT8), 0.0286 (4-bit) | Simple baseline |
| Zero-point | No | Each weight → nearest value, full range | Error 0.0010 (INT8), 0.0167 (4-bit) | Ranges that are not symmetric |
| GGUF k-quants | No | Blocks of weights with their own scale | 4.5 bits per weight for `Q4_K` | CPU and laptop with llama.cpp |
| GPTQ / EXL2 | Yes | Repairs the output after each rounding | Output error 0.160 → 0.032 | GPU, 3 to 4 bits |
| AWQ | Yes | Scales the important weights first | Output error 0.160 → 0.077 | GPU, 4 bits |
| SmoothQuant | Yes (ranges only) | Moves the scale from activations to weights | Worst channel: 7.9 → 38.9 steps | INT8 weights and activations |

## Checks and common misconceptions

- **"A 4-bit model is 4 times less accurate than a 16-bit model."** No. The size is 4 times smaller, but the output can stay almost the same when the method minimizes the output error. Check: measure the quantized model on your task (see [Evaluation](06-evaluation.md)), not the error of the weights.
- **"A 4-bit file has 4 bits for each weight."** The scales and the mixed types add bits: `Q4_K_M` has about 4.9. Check: file size in bits ÷ number of parameters.
- **"Quantization only makes the file smaller."** It also sets the memory that you need at run time, and it can make inference faster, because the program moves fewer bytes. But the weights in memory are not all: the context also needs memory.

## Sources

This page follows section 7 of "The LLM Scientist" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [Introduction to quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html), Maxime Labonne (absmax, zero-point, LLM.int8()).
- [Quantize Llama models with llama.cpp](https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html), Maxime Labonne.
- [4-bit LLM Quantization with GPTQ](https://mlabonne.github.io/blog/posts/4_bit_Quantization_with_GPTQ.html), Maxime Labonne.
- [Understanding Activation-Aware Weight Quantization](https://medium.com/friendliai/understanding-activation-aware-weight-quantization-awq-boosting-inference-serving-efficiency-in-10bb0faf63a8), FriendliAI.
- [SmoothQuant on Llama 2 7B](https://github.com/mit-han-lab/smoothquant/blob/main/examples/smoothquant_llama_demo.ipynb), MIT HAN Lab.
- [DeepSpeed Model Compression](https://www.deepspeed.ai/tutorials/model-compression/), DeepSpeed (ZeroQuant; the link of the course, not reachable in November 2026).
- [llama.cpp quantize tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) (bits per weight and file sizes of Llama 3.1 8B) and the [k-quants pull request](https://github.com/ggml-org/llama.cpp/pull/1684) (`Q4_K`: 4.5 bits per weight, `Q6_K`: 6.5625, the `Q4_K_M` mix).
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) (the parts of the file).
- [GPTQ](https://arxiv.org/abs/2210.17323), Frantar et al., 2022 (calibration set, 175 billion parameters in about 4 GPU hours).
- [ExLlamaV2](https://github.com/turboderp/exllamav2) (EXL2: 2 to 8 bits per weight).
- [AWQ](https://arxiv.org/abs/2306.00978), Lin et al., 2023.
- [SmoothQuant](https://arxiv.org/abs/2211.10438), Xiao et al., 2022 (1.56× speed, 2× memory).
- [ZeroQuant](https://arxiv.org/abs/2206.01861), Yao et al., 2022 (the three parts, 3× memory).
