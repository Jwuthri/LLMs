# 4. Supervised fine-tuning

## Core idea

A pre-trained model can continue any text. **Supervised fine-tuning (SFT)** trains it some more on pairs of (instruction, good answer), so that it answers like an assistant.
The training step is the same as in pre-training: calculate an output, measure the error, move the weights a little.
The choices are: which weights you move, how fast you move them, and how many GPUs share the work.

## The example

This page uses the weight matrix $W$ of the [toy example](../toy-example.md) and one training example.

| Item | Value |
|---|---|
| Weight matrix $W$ (2 × 2, so $d = 2$) | $\begin{bmatrix} 0.5 & -1.2 \\ 0.8 & 0.3 \end{bmatrix}$ |
| Input $x$ and target $t$ (assumed) | $x = [1, 2]$, $t = [-1.5, 1.0]$ |
| Loss | $L = \tfrac{1}{2}\lVert Wx - t \rVert^2$ |
| Learning rate $\eta$ | 0.1 |

The **loss** is one number that says how wrong the output is. A real LLM uses the cross-entropy loss on the next token. The squared error keeps the arithmetic short, and the steps are the same.

## 1. Training techniques

### Setup

There are three techniques. **Full fine-tuning** changes each weight. **LoRA** (low-rank adaptation) freezes $W$ and trains two small matrices, $A$ and $B$, next to it. **QLoRA** is LoRA with the frozen $W$ stored in 4 bits.

<figure class="diagram">
--8<-- "docs/assets/figures/fine-tuning/lora.svg"
<figcaption>LoRA after one training step. The input takes two paths, and the results are added.</figcaption>
</figure>

### Steps

1. **Calculate the output and the error.** $Wx = [-1.9,\ 1.4]$. The error is $e = Wx - t = [-0.4,\ 0.4]$. The loss is $\tfrac{1}{2}(0.16 + 0.16) = 0.16$.

    **Why:** the error gives the direction in which each output must move.
    { .why }

2. **Calculate the gradient.** The **gradient** has one number for each weight. It says how much the loss grows when that weight grows.

    $$G = e\,x^\top = \begin{bmatrix} -0.4 \\ 0.4 \end{bmatrix} \begin{bmatrix} 1 & 2 \end{bmatrix} = \begin{bmatrix} -0.4 & -0.8 \\ 0.4 & 0.8 \end{bmatrix}$$

    **Why:** if you move each weight against its gradient, the loss goes down.
    { .why }

3. **Full fine-tuning: change all 4 numbers.**

    $$W \leftarrow W - \eta G = \begin{bmatrix} 0.54 & -1.12 \\ 0.76 & 0.22 \end{bmatrix}$$

    The output becomes $[-1.7,\ 1.2]$ and the loss falls from 0.16 to 0.04.

    **Why:** this gives the model the most freedom. The price: you keep a gradient and some optimizer numbers (see section 3) for each weight, and the result is a full new copy of the model.
    { .why }

4. **LoRA: freeze $W$ and add a small update.** The **rank** $r$ is the number of columns of $B$. $\alpha$ is a number that you set: it scales the update. Here $r = 1$ and $\alpha = 2$, so the scale $\alpha / r$ is 2.

    $$W' = W + \frac{\alpha}{r} B A, \qquad B = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \qquad A = \begin{bmatrix} 0.5 & 0.5 \end{bmatrix} \text{ (assumed random start)}$$

    At the start, $BA$ is all zeros, so $W' = W$.

    **Why:** $B$ starts at zero so that the first output is exactly the output of the pre-trained model. $A$ must not start at zero too: with $A = 0$ and $B = 0$, the two gradients are zero and the adapter cannot learn.
    { .why }

5. **Train only $B$ and $A$.** The gradients come from the same $G$:

    $$\frac{\partial L}{\partial B} = \frac{\alpha}{r} G A^\top = 2 \begin{bmatrix} -0.6 \\ 0.6 \end{bmatrix} = \begin{bmatrix} -1.2 \\ 1.2 \end{bmatrix}, \qquad \frac{\partial L}{\partial A} = \frac{\alpha}{r} B^\top G = \begin{bmatrix} 0 & 0 \end{bmatrix}$$

    One step gives $B = [0.12,\ -0.12]^\top$. $A$ does not move on the first step, because $B$ is zero.

    **Why:** $W$ gets no gradient. The optimizer stores numbers only for $A$ and $B$, and this is where the memory saving comes from.
    { .why }

6. **Merge the adapter.** An **adapter** is the pair $(A, B)$.

    $$\frac{\alpha}{r} B A = \begin{bmatrix} 0.12 & 0.12 \\ -0.12 & -0.12 \end{bmatrix}, \qquad W' = \begin{bmatrix} 0.62 & -1.08 \\ 0.68 & 0.18 \end{bmatrix}$$

    $W'x = [-1.54,\ 1.04]$. The two-path calculation gives the same result: $[-1.9,\ 1.4] + [0.36,\ -0.36]$. The loss is now 0.0016. Do not compare it with the 0.04 of step 3: the scale $\alpha / r$ makes the LoRA step larger here.

    **Why:** the merged matrix has the shape of $W$, so the model runs at its usual speed. You can also keep the adapter as a small separate file and change it for each task.
    { .why }

7. **Count the trained numbers.** Full fine-tuning trains $d \cdot d$ numbers. LoRA trains $2 \cdot d \cdot r$ numbers ($B$ has $d \times r$, $A$ has $r \times d$).

    | Matrix | Full: $d \cdot d$ | LoRA: $2 \cdot d \cdot r$ | Ratio |
    |---|---|---|---|
    | Toy, $d = 2$, $r = 1$ | 4 | 4 | 100% |
    | $d = 4096$, $r = 1$ | 16,777,216 | 8,192 | 0.049% |
    | $d = 4096$, $r = 16$ | 16,777,216 | 131,072 | 0.781% |

    **Why:** the ratio is $2r / d$. The toy matrix is too small to save anything. A real matrix is large, and a low rank is enough because the update has less freedom than the matrix: the two rows of $BA$ in step 6 are multiples of each other.
    { .why }

8. **QLoRA: store the frozen $W$ in 4 bits.** A 4096 × 4096 matrix needs 33.55 MB at 16 bits and 8.39 MB at 4 bits. The adapter ($r = 16$) stays at 16 bits: 0.26 MB.

    **Why:** $W$ does not change, so a rough copy is enough to calculate outputs and gradients. $A$ and $B$ get small updates, so they keep full precision. The page on [quantization](07-quantization.md) shows how 4-bit storage works.
    { .why }

<div class="llm-widget" data-widget="lora"></div>

!!! success "Key point"
    LoRA does not make the model smaller. It makes the *trained part* smaller. $B$ starts at zero, so training starts from the pre-trained model, and the merge gives back one matrix of the same size.

### From toy to real

- A real model has many matrices in each block. The **target modules** are the matrices that get an adapter: the attention projections and the feed-forward layers.
- In the Llama 3.1 8B tutorial of the course, $r = 16$ on all these matrices trains 42 million of 8 billion parameters (0.52%).
- 8 billion parameters need 16 GB at 16 bits and 4 GB at 4 bits. This is why QLoRA fits on one consumer GPU.
- The libraries [TRL](https://huggingface.co/docs/trl/en/index), [Unsloth](https://docs.unsloth.ai/) and [Axolotl](https://axolotl.ai/) implement the three techniques.

## 2. Training parameters

### Setup

A **hyperparameter** is a value that you set before training. The **optimizer** is the rule that changes a weight with its gradient. One optimizer step is $w \leftarrow w - \eta \cdot g$, where $g$ is the gradient. We follow one weight, $w_{11} = 0.5$.
Assume these gradients of $w_{11}$ for 8 training examples: −0.5, −0.3, −0.1, −0.3, −0.8, −0.4, 0.1, −0.1.

### Steps

1. **Batch size and gradient accumulation.** The **batch size** is the number of examples in one step. Assume that only 2 examples fit in GPU memory (a micro-batch). **Gradient accumulation** adds up the gradients of 4 micro-batches before one step.

    | Micro-batch | 1 | 2 | 3 | 4 | Mean of the 4 |
    |---|---|---|---|---|---|
    | Mean gradient | −0.4 | −0.2 | −0.6 | 0.0 | **−0.3** |

    The mean of all 8 gradients is also −0.3. The step is $w_{11} = 0.5 - 0.1 \cdot (-0.3) = 0.53$.

    **Why:** 4 micro-batches of 2 give the gradient of a batch of 8 (the effective batch size) with the memory of a batch of 2. The price is 4 forward passes for each step.
    { .why }

2. **Learning rate, warm-up and scheduler.** The **learning rate** $\eta$ is the step size. A **scheduler** changes it during training. A common shape is a linear **warm-up** from 0, then a cosine decay:

    $$\eta(s) = \eta_{max} \cdot \frac{s}{s_w} \ \text{ for } s < s_w, \qquad \eta(s) = \frac{\eta_{max}}{2}\left(1 + \cos\frac{\pi (s - s_w)}{S - s_w}\right) \text{ after}$$

    With $\eta_{max} = 3 \times 10^{-4}$, $s_w = 10$ warm-up steps and $S = 100$ steps: step 5 gives $1.5 \times 10^{-4}$, step 10 gives $3 \times 10^{-4}$, step 55 gives $1.5 \times 10^{-4}$, step 100 gives 0.

    **Why:** at the start the optimizer has seen only a few gradients, so its estimates are noisy. Small steps prevent a bad jump. At the end, small steps let the weights settle.
    { .why }

3. **Epochs.** An **epoch** is one pass over the training set. With 1,000 examples and an effective batch of 8, one epoch is 125 steps.

    **Why:** more epochs give more learning, but the model starts to memorize the examples (see section 4).
    { .why }

4. **Weight decay.** Each step also multiplies the weight by $(1 - \eta \lambda)$. With $\lambda = 0.01$: $w_{12} = -1.2 \cdot (1 - 0.1 \cdot 0.01) = -1.1988$.

    **Why:** a weight stays large only if the data keeps pushing it. This is a form of **regularization**: it limits memorization.
    { .why }

!!! success "Key point"
    The learning rate and the number of epochs have the largest effect. Gradient accumulation changes memory use, not the gradient.

### From toy to real

| Parameter | Llama 3.1 8B tutorial (Unsloth) | Typical range |
|---|---|---|
| LoRA rank r | 16 | 8 to 256 |
| LoRA alpha | 16 | 1 or 2 times the rank |
| Learning rate, scheduler | 0.0003, linear decay | QLoRA paper: 0.0001 to 0.0002 |
| Batch size × accumulation | 8 × 2 = 16 | Set by GPU memory |
| Epochs | 1 | Few: more epochs can overfit |
| Weight decay | 0.01 | |
| Warm-up steps | 10 | |
| Optimizer | 8-bit AdamW | |

## 3. Distributed training

### Setup

We train the 4 toy weights on 2 GPUs. In plain **data parallelism**, each GPU holds a full copy of the model and gets half of the batch. Each GPU then holds three things: 4 weights, 4 gradients, and the optimizer states. The Adam optimizer keeps 2 numbers for each weight: 8 numbers. The total is 16 numbers on each GPU.

**ZeRO** (Zero Redundancy Optimizer, in the DeepSpeed library) removes the duplicate copies in three stages.

<figure class="diagram">
--8<-- "docs/assets/figures/fine-tuning/zero.svg"
<figcaption>What GPU 0 holds for the 4 toy weights. A dashed box is a part that only GPU 1 holds.</figcaption>
</figure>

### Steps

1. **Stage 1: split the optimizer states.** GPU 0 keeps the Adam numbers of weights 1-2, GPU 1 those of weights 3-4. Each GPU holds 4 + 4 + 4 = 12 numbers.

    **Why:** the optimizer states are the largest part, and only the GPU that updates a weight needs them.
    { .why }

2. **Stage 2: also split the gradients.** Each GPU keeps the averaged gradient only for its own 2 weights: 4 + 2 + 4 = 10 numbers.

    **Why:** a GPU updates only its share of the weights, so the other gradients are of no use to it.
    { .why }

3. **Stage 3: also split the weights.** Each GPU stores 2 weights: 2 + 2 + 4 = 8 numbers, half of the start. Before a layer runs, the GPUs send each other the missing weights. After the layer, they delete them.

    **Why:** memory for each GPU now falls in proportion to the number of GPUs. The price is more communication.
    { .why }

4. **FSDP (Fully Sharded Data Parallel)** is the PyTorch version of the stage 3 idea. A **shard** is one piece of a split state.

    **Why:** you get the same memory saving without an extra library.
    { .why }

!!! success "Key point"
    More GPUs give more speed with plain data parallelism, but not more room. ZeRO and FSDP turn more GPUs into more room for a larger model.

### From toy to real

With 16-bit weights, the ZeRO paper counts 16 bytes for each parameter: 2 for the weight, 2 for the gradient, 12 for the optimizer states. For a model of 1 billion parameters, each GPU needs:

| | No ZeRO | Stage 1 | Stage 2 | Stage 3 / FSDP |
|---|---|---|---|---|
| 2 GPUs | 16 GB | 10 GB | 9 GB | 8 GB |
| 8 GPUs | 16 GB | 5.5 GB | 3.75 GB | 2 GB |

The paper gives 120 GB, 31.4 GB, 16.6 GB and 1.9 GB for 7.5 billion parameters on 64 GPUs. These numbers do not include the **activations** (the intermediate results of each layer). **Gradient checkpointing** reduces those: it stores fewer activations and calculates them again when it needs them.

## 4. Monitoring

### Setup

During training you record three signals. The **training loss** is measured on the batches that the model learns from. The **validation loss** is measured on examples that the model never trains on. The **gradient norm** is the length of the gradient.

<figure class="plot" markdown>
![Synthetic curves. The training loss falls from 2.00 to 0.41. The validation loss falls to 0.86 at step 150, then rises to 1.51 at step 300.](../assets/figures/fine-tuning/loss-curves.svg#only-light)
![Synthetic curves. The training loss falls from 2.00 to 0.41. The validation loss falls to 0.86 at step 150, then rises to 1.51 at step 300.](../assets/figures/fine-tuning/loss-curves-dark.svg#only-dark)
<figcaption>An illustration, not a real run. After step 150 the training loss still falls, but the validation loss rises.</figcaption>
</figure>

### Steps

1. **Compare the two loss curves.** In the illustration, the validation loss is lowest at step 150 (0.865, training loss 0.531). At step 300 the training loss is 0.411, but the validation loss is 1.509. This is **overfitting**: the model memorizes the training examples. A **checkpoint** is a saved copy of the weights. Keep the checkpoint of step 150.

    **Why:** the training loss only says how well the model fits examples that it saw. The validation loss estimates the quality on new inputs.
    { .why }

2. **Record the gradient norm.** For the gradient $G$ of section 1:

    $$\lVert G \rVert = \sqrt{0.4^2 + 0.8^2 + 0.4^2 + 0.8^2} = \sqrt{1.6} = 1.265$$

    **Why:** it is one number for the size of the next step. A sudden jump in the norm often comes just before a loss spike.
    { .why }

3. **Clip the gradient.** With a maximum norm of 1.0, multiply $G$ by $1.0 / 1.265 = 0.791$. The result is $\begin{bmatrix} -0.316 & -0.632 \\ 0.316 & 0.632 \end{bmatrix}$, with a norm of 1.0.

    **Why:** the direction stays the same, but one bad batch cannot throw the weights far away.
    { .why }

4. **Read the loss spikes.** A spike that goes back down in a few steps is usually one hard or noisy batch. A loss that stays high, or becomes NaN (not a number), usually means that the learning rate is too high.

    **Why:** the two cases need different actions. The first needs no action or cleaner data. The second needs a lower learning rate, more warm-up or clipping.
    { .why }

!!! success "Key point"
    Do not select a checkpoint with the training loss. Use the validation loss, then a real [evaluation](06-evaluation.md).

### From toy to real

- The Hugging Face trainers clip at a norm of 1.0 by default (`max_grad_norm`).
- A real loss curve is noisy. Look at the trend over many steps, not at one point.
- The loss is not the goal. A model with a low loss can still give poor answers.

## Minimal code

This code gives the same numbers as section 1.

```python
import numpy as np

W = np.array([[0.5, -1.2], [0.8, 0.3]])        # frozen
x, t = np.array([1., 2.]), np.array([-1.5, 1.0])
A = np.array([[0.5, 0.5]])                     # r x d, random start
B = np.zeros((2, 1))                           # d x r, starts at zero
scale, lr = 2.0 / 1, 0.1                       # alpha / r

G = np.outer((W + scale * B @ A) @ x - t, x)   # [[-0.4, -0.8], [0.4, 0.8]]
grad_B, grad_A = scale * G @ A.T, scale * B.T @ G
B, A = B - lr * grad_B, A - lr * grad_A        # B = [[0.12], [-0.12]]
W_merged = W + scale * B @ A                   # [[0.62, -1.08], [0.68, 0.18]]
```

Real-library sketch (the tests do not run this code):

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4"),  # QLoRA
)
config = LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
model = get_peft_model(model, config)          # W frozen, A and B trainable
model.print_trainable_parameters()
```

## Comparison

| Technique | Trained numbers (4096 × 4096) | Storage of the frozen W | Output of training | Good for | Risk |
|---|---|---|---|---|---|
| Full fine-tuning | 16,777,216 (100%) | Not frozen: 33.55 MB at 16 bits | A full new model | Large data, large change of behaviour | Most memory; can forget earlier skills |
| LoRA, r = 16 | 131,072 (0.781%) | 33.55 MB at 16 bits | A small adapter file | Most instruction tuning | Rank too low for a hard task |
| QLoRA, r = 16 | 131,072 (0.781%) | 8.39 MB at 4 bits | A small adapter file | One GPU with little memory | Slower steps; small loss from 4-bit rounding |

## Checks and common misconceptions

- **"LoRA makes the model smaller."** After the merge, the model has the same size as before. Check: count the parameters before and after `merge_and_unload()`. Only the adapter file is small.
- **"A lower training loss is a better model."** In the illustration, the model at step 300 has the lowest training loss and a worse validation loss than at step 150. Check: always keep a validation set that the model does not train on.
- **"A larger batch needs a larger GPU."** Gradient accumulation gives the same gradient with small micro-batches. Check: compare batch size × accumulation steps, not the batch size alone, when you copy a setting.

## Sources

This page follows section 4 of "The LLM Scientist" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [Fine-tune Llama 3.1 Ultra-Efficiently with Unsloth](https://huggingface.co/blog/mlabonne/sft-llama3), Maxime Labonne (the tutorial values, the ranges of rank and alpha, 42 million trained parameters).
- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/), Wing Lian.
- [Mastering LLMs](https://parlance-labs.com/education/), Hamel Husain.
- [LoRA insights](https://lightning.ai/pages/community/lora-insights/), Sebastian Raschka.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), Hu et al., 2021 (the formula, $B$ starts at zero, the merge).
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), Dettmers et al., 2023 (4-bit frozen weights, learning rates).
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), Rajbhandari et al., 2019 (16 bytes for each parameter, the three stages, the 7.5 billion example).
- [Getting started with FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html), PyTorch tutorial.
- [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments), Hugging Face (default `max_grad_norm` of 1.0).
