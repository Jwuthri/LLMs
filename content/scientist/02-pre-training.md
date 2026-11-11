# 2. Pre-training models

## Core idea

Pre-training teaches a model to predict the next token on a very large quantity of raw text.
One step is simple: calculate the loss, then change each weight a little so that the loss goes down.
The hard part is the scale: trillions of tokens, many machines, and a training run that must stay stable for weeks.

## The example

This page uses the logits of the [toy example](../toy-example.md). After "the cat", the correct next token is `sat`.

| Item | Value |
|---|---|
| Logits for `the`, `cat`, `sat` | [0.1, 1.0, 2.0] |
| Probabilities (softmax) | [0.099, 0.242, 0.659] |
| Loss | $-\ln(0.659) = 0.417$ |
| Perplexity | $e^{0.417} = 1.517$ |
| Loss of a model that knows nothing | $\ln(3) = 1.099$ |

The **loss** (cross-entropy) is $-\ln$ of the probability that the model gives to the correct token. It is 0 for a perfect prediction.
The **perplexity** is $e^{\text{loss}}$: the model is as unsure as if it chose between 1.517 equally probable tokens.
To have weights that we can train, assume that the model is only these 3 logits: $\theta = [0.1, 1.0, 2.0]$.

## 1. Data preparation

### Setup

A raw corpus (a set of texts) from the web has copies and low-quality pages. Our raw corpus has 5 documents and 18 words.

| # | Raw document | Fingerprint |
|---|---|---|
| 1 | `the cat sat` | `9ef3d846` |
| 2 | `The cat sat.` | `9ef3d846` |
| 3 | `the cat sat` | `9ef3d846` |
| 4 | `cat cat cat cat` | `4fa30b89` |
| 5 | `the cat the cat sat` | `9b90c680` |

### Steps

1. **Normalize each document.** Make it lowercase and remove the punctuation. Document 2 becomes `the cat sat`.

    **Why:** a capital letter or a full stop must not hide a copy.

2. **Calculate a fingerprint and keep one document for each fingerprint.** A **hash** function turns a text into a short fixed-size number. The table shows the first 8 hexadecimal digits of the SHA-1 hash. Documents 1, 2 and 3 have the same fingerprint. 3 documents remain.

    **Why:** to compare each pair of documents is too slow: 5 documents give 10 pairs, and 1 billion documents give 499,999,999,500,000,000 pairs. With a hash, you read each document one time.

3. **Remove the low-quality documents.** A simple filter is the ratio of unique words: 3 / 3 = 1.0 for `the cat sat`, 1 / 4 = 0.25 for `cat cat cat cat`, 3 / 5 = 0.6 for document 5. Assume a limit of 0.5. Document 4 goes out. 2 documents and 8 words remain.

    **Why:** the model learns to copy its data. Repeated or broken text teaches it to write repeated or broken text.

4. **Tokenize, then make one example for each position.** `the cat sat` gives 2 examples: `the` → `cat` and `the cat` → `sat`. The 2 documents give 6 examples.

    **Why:** the text is its own label. No person has to write the answers, so you can use trillions of tokens. This is **self-supervised** learning.

<div class="key-point">

The data decide what the model can learn. Deduplication and filters remove more than half of this toy corpus (18 words → 8). Each removed word is compute that you do not waste.

</div>

### From toy to real

- Llama 3.1 was pre-trained on about 15 trillion tokens.
- FineWeb is an open dataset made from Common Crawl. Its first release had 15 trillion tokens from 96 snapshots. FineWeb-Edu is a subset of 1.3 trillion tokens that a classifier selected as educational text.
- A hash finds only exact copies. For near-copies, FineWeb uses **MinHash** on 5-grams (5 adjacent words) with 14 × 8 = 112 hash functions. The [datasets page](03-datasets.md#4-quality-filtering) shows how MinHash works.
- Real pipelines add more filters: language detection, repetition rules, quality classifiers, removal of personal data.

## 2. Distributed training

### Setup

One GPU is too slow and too small for a large model, so the work is split between many **workers** (here, 1 worker = 1 GPU).
Assume a batch of 4 samples with the context "the cat". The next token is `sat` in samples 1, 2 and 4, and `cat` in sample 3. We have 2 workers.

<figure class="diagram" data-src="pre-training/parallelism.svg">
<figcaption>The three ways to split the work between 2 workers, with the toy numbers.</figcaption>
</figure>

### Steps

1. **Calculate the gradient of one sample.** The **gradient** says how the loss changes when each weight changes. For softmax with cross-entropy, it is the probabilities $p$ minus the one-hot target $y$.

    $$\frac{\partial L}{\partial \theta} = p - y$$

    Target `sat`: [0.099, 0.242, 0.659] − [0, 0, 1] = [0.099, 0.242, −0.341]. Target `cat`: [0.099, −0.758, 0.659].

    **Why:** a negative value means "increase this logit". The logit of the correct token goes up, the others go down.

2. **Data parallel: give each worker a copy of the model and a part of the batch.** Worker A gets samples 1 and 2. Its mean gradient is [0.099, 0.242, −0.341]. Worker B gets samples 3 and 4: [0.099, −0.258, 0.159].

    **Why:** the samples are independent, so the two workers can calculate at the same time.

3. **Average the gradients of the workers.** The mean is [0.099, −0.008, −0.091]. One machine with the 4 samples gives exactly the same vector.

    **Why:** the result is the same as a larger batch on one machine, in less time. The price is communication: the workers must send their gradients at each step.

4. **Pipeline parallel: give each worker a group of layers.** Worker A holds layers 1 and 2. It sends its output (the activations) to worker B, which holds layers 3 and 4.

    **Why:** this is for a model that does not fit in the memory of one GPU. The cost is waiting time: B cannot start before A is done.

5. **Tensor parallel: split one matrix multiplication.** For $x$ = `sat` = [1, 1], worker A has row 1 of $W$ and calculates $0.5 - 1.2 = -0.7$. Worker B has row 2 and calculates $0.8 + 0.3 = 1.1$. Together: $Wx = [-0.7, 1.1]$.

    **Why:** the two workers calculate one layer at the same time, with no waiting. But they must exchange data in each layer, so they need a very fast connection.

<div class="key-point">

Data parallelism splits the batch and makes training faster. Pipeline and tensor parallelism split the model and make a large model fit. Large runs use the three together.

</div>

### From toy to real

- Memory decides the method. Training with AdamW keeps 4 numbers for each parameter: the weight, the gradient and 2 averages. For 8 billion parameters in 32-bit numbers, that is 8 × 10⁹ × 4 bytes × 4 = 128 GB, before the activations.
- Codebases such as nanotron give the three methods together ("3D parallelism").
- The limit is often the network between the GPUs, not the GPUs.

## 3. Training optimization

### Setup

We do one full update of $\theta = [0.1, 1.0, 2.0]$ with the gradient of the toy sample, $g = [0.099, 0.242, -0.341]$.
The **learning rate** $\eta$ is the size of a step. Assume $\eta = 0.1$ for this small model.

### Steps

1. **Clip the gradient by its norm.** The **norm** is the length of the vector: $\lVert g \rVert = \sqrt{0.099^2 + 0.242^2 + 0.341^2} = 0.430$. Assume a limit $c = 0.25$.

    $$g \leftarrow g \cdot \min\!\left(1, \frac{c}{\lVert g \rVert}\right) = g \cdot 0.582$$

    The clipped gradient is [0.057, 0.141, −0.198]. Its norm is 0.25.

    **Why:** one bad batch can give a very large gradient and destroy weeks of training. Clipping keeps the direction and limits the length.

2. **Update the two moving averages of AdamW.** $m$ is the average of the gradients. $v$ is the average of the squared gradients. Both start at 0. With $\beta_1 = 0.9$ and $\beta_2 = 0.999$:

    $$\begin{aligned} m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\ &= [0.0057,\ 0.0141,\ -0.0198] \\[2mm] v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\ &= [3.3,\ 19.9,\ 39.3] \times 10^{-6} \end{aligned}$$

    **Why:** one gradient is noisy. The average $m$ gives a stable direction. The average $v$ measures how large the gradients of each weight usually are.

3. **Correct the start and divide.** The averages start at 0, so they are too small in the first steps. Divide by $1 - \beta^t$. At step $t = 1$, this is 0.1 for $m$ and 0.001 for $v$:

    $$\hat m = \frac{m}{0.1} = [0.057,\ 0.141,\ -0.198]$$

    $$\hat v = \frac{v}{0.001} = [0.0033,\ 0.0199,\ 0.0393]$$

    $$\frac{\hat m}{\sqrt{\hat v} + \epsilon} = [1,\ 1,\ -1]$$

    **Why:** the division gives each weight its own step size. A weight with small gradients moves as much as a weight with large gradients. At step 1, each weight moves by $\eta$.

4. **Step, with weight decay.** **Weight decay** pulls each weight toward 0 by a part $\lambda$ of its value. With $\lambda = 0.01$:

    $$\begin{aligned} \theta &\leftarrow \theta - \eta \left( \frac{\hat m}{\sqrt{\hat v} + \epsilon} + \lambda\, \theta \right) \\ &= [-0.0001,\ 0.899,\ 2.098] \end{aligned}$$

    The loss for `sat` goes from 0.417 to 0.354.

    **Why:** small weights give a simpler model that generalizes better. AdamW applies the decay to the weights directly, not through the gradient, so the division of step 3 does not change it.

5. **Calculate with 16-bit numbers where it is safe.** A 32-bit float (FP32) uses 4 bytes. A 16-bit float (FP16 or BF16) uses 2 bytes. The toy embedding table (6 numbers) needs 24 bytes in FP32 and 12 bytes in FP16. But near 1, FP16 has steps of 0.000977: in FP16, 1 + 0.0001 = 1, and the update is lost. Thus **mixed precision** calculates the forward and backward pass in 16 bits, and keeps the weights in FP32.

    **Why:** 16-bit numbers use half the memory and are faster on a GPU. The FP32 copy of the weights makes sure that small updates add up.

6. **Change the learning rate during training: warm-up, then cosine.** $T_w$ is the number of warm-up steps and $T$ the total number of steps. $s = \frac{t - T_w}{T - T_w}$ is the part of the cosine phase that is done.

    $$\eta(t) = \begin{cases} \eta_{\max} \, \dfrac{t}{T_w} & t < T_w \\[2mm] \dfrac{\eta_{\max}}{2} \left( 1 + \cos(\pi s) \right) & t \ge T_w \end{cases}$$

    Assume $\eta_{\max} = 3 \times 10^{-4}$, $T_w = 100$ and $T = 1000$:

    | Step | 0 | 50 | 100 | 325 | 550 | 1000 |
    |---|---|---|---|---|---|---|
    | Learning rate | 0 | 1.5 × 10⁻⁴ | 3 × 10⁻⁴ | 2.56 × 10⁻⁴ | 1.5 × 10⁻⁴ | 0 |

    **Why:** at the start the weights are random and the averages of AdamW are not reliable, so large steps are dangerous. At the end, small steps let the model settle.

<div class="llm-widget" data-widget="lr-schedule"></div>

<div class="key-point">

The order in one step is: gradient, clip, AdamW update, new learning rate. Each part prevents one way for the training to fail: a too-large gradient, a noisy direction, a bad step size.

</div>

### From toy to real

- $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ and $\lambda = 0.01$ are the defaults of AdamW in PyTorch. A clipping limit of 1.0 is a common choice.
- **Lion** is a more recent optimizer. It moves each weight by the sign of an average of the gradients, and it does not keep $v$.
- FP16 cannot hold a number above 65,504. **BF16** uses the same 8 exponent bits as FP32, so it has the same range with less precision. BF16 is a common choice for LLM training.
- For 8 billion parameters, the weights need 32 GB in FP32 and 16 GB in BF16.

## 4. Monitoring

### Setup

We train the toy model for 100 steps on the batch of 4 samples. We start from $\theta = [0, 0, 0]$, with AdamW, 10 warm-up steps, a peak learning rate of 0.1 and a clipping limit of 1.0. The script writes one log line for each step.

| Step | Loss | Gradient norm | Learning rate |
|---|---|---|---|
| 0 | 1.099 | 0.540 | 0 |
| 10 | 0.823 | 0.307 | 0.1 |
| 50 | 0.579 | 0.020 | 0.059 |
| 100 | 0.575 | 0.018 | 0 |

<figure class="plot">
<img src="/figures/pre-training/loss-curve.svg" alt="Line chart of the training loss for 100 steps. It starts at 1.099, falls to 0.579 at step 50 and stays above the best possible loss of 0.562.">
<figcaption>The loss starts at ln(3), falls fast after the warm-up, then gets near the best possible value.</figcaption>
</figure>

### Steps

1. **Check the first loss.** With logits of 0, each token has a probability of 1/3. The loss must be $\ln(3) = 1.099$, and it is.

    **Why:** this check costs one step. A different value shows an error in the data, the labels or the initial weights before you pay for a long run. For GPT-2, with 50,257 tokens, the first loss must be near $\ln(50{,}257) = 10.82$.

2. **Watch the loss go down, and know its floor.** The batch says `sat` 3 times and `cat` 1 time after "the cat". The best possible prediction is [0, 0.25, 0.75], with a loss of 0.562. The run gets to 0.575.

    **Why:** the loss of a language model never gets to 0, because text has more than one correct next word. A flat curve is not always a problem. A curve that goes up is.

3. **Watch the gradient norm.** It goes from 0.540 to 0.018 as the model gets near the best prediction.

    **Why:** the norm warns you before the loss does. A sudden peak shows instability. A value of 0 or "not a number" shows a dead run.

4. **Watch the learning rate.** It is 0 at step 0, 0.1 at step 10 and 0.059 at step 50, as the schedule says.

    **Why:** a wrong schedule is a frequent error, and the loss curve alone does not show it.

5. **Watch the speed and the GPU use.** One step reads 4 × 3 = 12 tokens. Assume 2 steps each second: the **throughput** is 24 tokens per second. Also log the percentage of time that each GPU works and its memory.

    **Why:** speed is money. If a GPU works 50% of the time, it waits for data or for the other workers, and you pay for half a GPU that does nothing.

<div class="key-point">

A training run is a long experiment that you cannot repeat cheaply. Log the loss, the gradient norm, the learning rate and the throughput at each step, for each worker.

</div>

### From toy to real

- Throughput gives the time of a run. Assume 1 million tokens per second: 15 trillion tokens need 15 million seconds, which is 174 days.
- Also calculate the loss on text that is not in the training data (the **validation loss**). If it goes up while the training loss goes down, the model learns its data by heart.
- Dashboards such as Weights & Biases or TensorBoard draw these curves. A **profiler** measures where the time goes: data loading, calculation or communication.

## Minimal code

This code gives the same numbers as section 3.

```python
import numpy as np

theta = np.array([0.1, 1.0, 2.0])
p = np.exp(theta) / np.exp(theta).sum()
g = p - np.array([0, 0, 1])                        # target "sat": [0.099, 0.242, -0.341]

g = g * min(1.0, 0.25 / np.linalg.norm(g))         # clip: [0.057, 0.141, -0.198]

lr, b1, b2, eps, wd, t = 0.1, 0.9, 0.999, 1e-8, 0.01, 1
m = (1 - b1) * g                                   # m and v start at 0
v = (1 - b2) * g**2
m_hat, v_hat = m / (1 - b1**t), v / (1 - b2**t)
theta = theta - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * theta)   # [-0.0001, 0.899, 2.098]

def lr_schedule(step, peak=3e-4, warmup=100, total=1000):
    if step < warmup:
        return peak * step / warmup
    return 0.5 * peak * (1 + np.cos(np.pi * (step - warmup) / (total - warmup)))
```

Real-library sketch (the tests do not run this code):

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_schedule(step) / 3e-4)
for batch in loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):          # mixed precision
        loss = model(**batch).loss
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()
    log({"loss": loss.item(), "grad_norm": norm.item(), "lr": scheduler.get_last_lr()[0]})
```

## Comparison

| Technique | Problem it solves | Toy number | Cost or risk |
|---|---|---|---|
| Deduplication + filters | Copies and bad text | 5 documents → 2 | Too strict filters remove good text |
| Data parallel | Training is too slow | Mean gradient [0.099, −0.008, −0.091] | Gradients sent at each step |
| Pipeline parallel | Model too large for one GPU | Layers 1–2 on A, 3–4 on B | Workers wait for each other |
| Tensor parallel | One layer too large or too slow | W x = [−0.7, 1.1] from 2 halves | Needs a very fast connection |
| Gradient clipping | A very large gradient | Norm 0.430 → 0.25 | A too-low limit slows the training |
| AdamW | Noisy gradients, different scales | θ → [−0.0001, 0.899, 2.098] | 2 more numbers for each weight |
| Warm-up + cosine | Unstable start, noisy end | 3 × 10⁻⁴ at step 100, 0 at step 1000 | 2 more values to tune |
| Mixed precision | Memory and speed | 24 bytes → 12 bytes | Small updates are lost without FP32 weights |

## Checks and common misconceptions

- **"More data is always better."** More *different, good* data is better. Check: measure how much of your corpus is left after deduplication. If most of it goes out, look at the source of the data.
- **"A good run gets to a loss of 0."** The floor of the toy run is 0.562, not 0. Check: compare the first loss with $\ln(\text{vocabulary size})$, then look at the direction of the curve.
- **"Two times the GPUs, two times the speed."** Communication and waiting time take a part. Check: measure the tokens per second for each GPU when you add workers.

## Sources

This page follows section 2 of "The LLM Scientist" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [FineWeb article](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1), [paper](https://arxiv.org/abs/2406.17557) and [dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb), Penedo et al. (15 trillion tokens, 96 snapshots, FineWeb-Edu, MinHash settings).
- [RedPajama v2](https://www.together.ai/blog/redpajama-data-v2), Weber et al.
- [nanotron](https://github.com/huggingface/nanotron), Hugging Face (3D parallelism).
- [Parallel training](https://www.andrew.cmu.edu/course/11-667/lectures/W10L2%20Scaling%20Up%20Parallel%20Training.pdf), Chenyan Xiong.
- [Distributed training survey](https://arxiv.org/abs/2407.20018), Duan et al.
- [OLMo 2](https://allenai.org/olmo), AI2, and [LLM360](https://www.llm360.ai/).
- [Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md), Meta (about 15 trillion tokens).
- [AdamW paper](https://arxiv.org/abs/1711.05101), Loshchilov and Hutter, and [AdamW in PyTorch](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) (default values).
- [Lion paper](https://arxiv.org/abs/2302.06675), Chen et al., 2023.
- [Automatic mixed precision](https://pytorch.org/docs/stable/amp.html), PyTorch.
- [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), Radford et al., 2019 (vocabulary size).
