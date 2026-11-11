# 9. Diffusion models

## Core idea

A diffusion model learns one skill: to remove noise from data. To make a new image or a new text, it starts from pure noise and removes the noise in a few steps.
For text, the noise is a mask on the tokens: the model starts from a sentence of masks and fills many tokens in each step, so it does not have to write from left to right.

## The example

This page uses the [toy example](../toy-example.md). Values marked "assume" are made for this page.

| Part | Values |
|---|---|
| Data set (parts 1 and 2) | The 3 embeddings: `the` = [1, 0], `cat` = [0, 1], `sat` = [1, 1] |
| Noise schedule | 3 steps. The signal that stays is 0.9, 0.5, 0.1 (assumed) |
| Noise vector | $\varepsilon$ = [0.6, −0.8] (assumed, fixed) |
| Sentence (part 3) | "the cat sat", with the probabilities [0.099, 0.242, 0.659] |
| Two-token text (part 4) | "the cat" or "a dog", each with a probability of 0.5 (assumed) |

The baseline is the **autoregressive** model of [lesson 1](01-architecture.md): it writes one token for each model call, from left to right.

## 1. The forward process: add noise

### Setup

A **diffusion model** is a generative model that is trained to remove noise from data.
Training needs noisy data first. The **forward process** makes it: a fixed rule adds random noise to a data point, a little more at each **timestep** $t$. Nothing is learned here.

The data point is $x_0$ = `sat` = [1, 1]. Think of it as an image with 2 pixels.

<figure class="diagram" data-src="diffusion/forward-reverse.svg">
<figcaption>The forward process adds noise to the data point. The reverse process (part 2) goes back.</figcaption>
</figure>

### Steps

1. **Choose a noise schedule.** A **noise schedule** says how much of the data point stays at each timestep. This number is $\bar\alpha_t$. Assume 3 timesteps with $\bar\alpha_1 = 0.9$, $\bar\alpha_2 = 0.5$ and $\bar\alpha_3 = 0.1$.

    **Why:** the model must learn to repair light damage and heavy damage. The schedule gives each level, from almost clean to almost pure noise.

2. **Draw one noise vector.** The noise $\varepsilon$ comes from a normal (Gaussian) distribution with a mean of 0 and a variance of 1. We fix $\varepsilon = [0.6, -0.8]$, so that you can check each number.

    **Why:** pure Gaussian noise is easy to draw. Thus generation has a simple start point.

3. **Mix the data point and the noise.**

    $$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$$

    For $t = 1$, the two factors are $\sqrt{0.9} = 0.949$ and $\sqrt{0.1} = 0.316$.

    $$x_1 = \sqrt{0.9} \cdot [1, 1] + \sqrt{0.1} \cdot [0.6, -0.8] = [1.138,\ 0.696]$$

    | Timestep | Signal that stays | Noisy point |
    |---|---|---|
    | 1 | 0.9 | [1.138, 0.696] |
    | 2 | 0.5 | [1.131, 0.141] |
    | 3 | 0.1 | [0.885, −0.443] |

    **Why:** one formula gives the noisy point of any timestep directly. Training can pick a random timestep and skip the timesteps before it.

4. **Calculate the signal-to-noise ratio.** The **signal-to-noise ratio (SNR)** is the square of the signal factor divided by the square of the noise factor: $\bar\alpha_t / (1 - \bar\alpha_t)$. For $t = 1$ it is 0.9 / 0.1 = 9. For the timesteps 2 and 3 it is 1 and 0.111.

    **Why:** the SNR is one number that says how hard a timestep is. At 9 you can still see `sat`. At 0.111 the point is almost only noise.

<div class="key-point">

The forward process is a fixed recipe, not a model. It makes training pairs at no cost: a noisy point, and the noise that made it.

</div>

### From toy to real

- A colour image of 256 × 256 pixels has 256 × 256 × 3 = 196,608 numbers. The forward process applies the same formula to each number.
- DDPM (2020) uses 1,000 timesteps. The noise that one timestep adds grows linearly from 0.0001 to 0.02. The toy schedule is equal to 3 large timesteps that add 0.1, 0.444 and 0.8.

## 2. Training and the reverse process: remove noise

### Setup

The **denoiser** is the neural network of a diffusion model. It reads a noisy point $x_t$ and the timestep $t$, and it guesses the noise in that point (**noise prediction**).
The **reverse process** uses the denoiser to go from noise back to data, step by step. This is generation.

The data set is the 3 toy embeddings $e_1, e_2, e_3$. For a data set this small, the best possible denoiser has an exact formula. No network is necessary.

### Steps

1. **Train: add known noise, ask for the noise, measure the error.** The training loss is the mean squared error between the real noise and the predicted noise $\varepsilon_\theta$, over the $d = 2$ coordinates.

    $$L = \tfrac{1}{d}\,\lVert \varepsilon - \varepsilon_\theta(x_t, t) \rVert^2$$

    **Why:** we made the noise, so we know the correct answer. No labels are necessary.

2. **Calculate the best guess of the clean point.** The best guess is a weighted mean of the 3 data points. The weight $w_i$ is the probability that $x_t$ came from $e_i$.

    $$w_i = \text{softmax}_i\!\left(-\frac{\lVert x_t - \sqrt{\bar\alpha_t}\,e_i \rVert^2}{2\,(1-\bar\alpha_t)}\right) \qquad \hat{x}_0 = \sum_i w_i\,e_i$$

    | Noisy `sat` from part 1 | Weights (`the`, `cat`, `sat`) | Guess of the clean point |
    |---|---|---|
    | $t = 3$: [0.885, −0.443] | [0.410, 0.257, 0.332] | [0.743, 0.590] |
    | $t = 1$: [1.138, 0.696] | [0.109, 0.002, 0.889] | [0.998, 0.891] |

    **Why:** a near data point is a more probable origin. At high noise the 3 origins are almost equally probable, so the guess is near the mean of the data set. At low noise the guess is almost `sat`.

3. **Change the guess into a noise prediction and calculate the loss.**

    $$\hat\varepsilon = \frac{x_t - \sqrt{\bar\alpha_t}\,\hat{x}_0}{\sqrt{1-\bar\alpha_t}}$$

    For $t = 1$ we get $\hat\varepsilon = [0.605, -0.473]$. The real noise is $[0.6, -0.8]$, so the loss is 0.053.

    **Why:** the clean guess and the noise guess are two views of one answer. The loss is not 0, also for the best denoiser, because the origin of a noisy point is not sure.

4. **Generate: start from noise and do one reverse step.** The start is the fixed noise sample $x_3 = [0.3, -1.2]$. The denoiser gives the weights [0.452, 0.267, 0.281], the guess $\hat{x}_0 = [0.733, 0.548]$ and the noise $\hat\varepsilon = [0.072, -1.447]$. Mix the two at the noise level of the timestep before. The rule that makes this step is the **sampler**.

    $$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar\alpha_{t-1}}\,\hat\varepsilon$$

    $$x_2 = 0.707 \cdot [0.733, 0.548] + 0.707 \cdot [0.072, -1.447] = [0.569,\ -0.636]$$

    **Why:** the first guess is poor: it is near the mean of the data set. Thus the sampler does not jump to it. It goes to a point with less noise and asks the denoiser again.

5. **Do the step again until no noise stays.** At $x_2$ the weights are [0.700, 0.127, 0.173], and the step gives $x_1 = [0.806, -0.095]$. At $x_1$ the weights are [0.995, 0.000, 0.004], and the last step gives $x_0 = [1.000, 0.005]$. This is almost `the` = [1, 0].

    **Why:** each step makes the next decision easier: the weight of `the` goes from 0.452 to 0.700 to 0.995.

<figure class="plot">
<img src="/figures/diffusion/reverse-path.svg" alt="Plot of the 3 data points and 3 reverse paths. Each path starts at a noise point and ends near one data point. The path from [0.3, -1.2] ends at [1.000, 0.005], near the.">
<figcaption>Three reverse paths with 3 steps each. A different start noise ends at a different data point.</figcaption>
</figure>

<div class="key-point">

Training is a regression: predict the noise. Generation is a loop: guess, do a small step, guess again. With this sampler, all the randomness is in the start noise.

</div>

### From toy to real

- The exact formula needs the full data set and can only give mixes of training points. A real denoiser is a network that must generalize: a U-Net in DDPM, a Transformer on patches in Diffusion Transformers (DiT).
- The **DDPM** sampler adds new random noise at each step. The **DDIM** sampler is the deterministic rule of step 4, with the same trained model. The **number of sampling steps** controls the cost: the DDIM authors compare 10 to 100 steps against 1,000 and report a speed-up of 10 to 50 times.
- **Guidance** makes the output follow a condition (for example, a text prompt) more strongly. **Classifier-free guidance** drops the condition at random in training. The sampler then uses $(1 + w)\,\varepsilon_{\text{with}} - w\,\varepsilon_{\text{without}}$, with a weight $w$.
- **Latent diffusion** runs the diffusion on a compressed image from an autoencoder. A text encoder gives the vectors that condition the denoiser ([lesson 8](08-new-trends.md)).

## 3. Diffusion for text: masks as noise

### Setup

A text is a list of token ids. You cannot add a little Gaussian noise to an id: the id 1.3 does not exist.
**Discrete diffusion** uses a noise that works on tokens. The most common type is **masked diffusion** (also named absorbing diffusion). Its noise replaces a token with a special **mask token**, written `[M]`. A masked token stays masked.
The **masking rate** $t$ is the probability that each token is masked. It is the noise level: $t = 0$ is the clean text, and $t = 1$ is a text of only masks.

The denoiser is a Transformer without a causal mask: each position reads all positions, left and right. In one call, it gives a probability distribution over the vocabulary at each masked position.

<figure class="diagram" data-src="diffusion/unmasking.svg">
<figcaption>An autoregressive model adds the next token. A masked diffusion model fills the position where it is most sure.</figcaption>
</figure>

### Steps

1. **Forward: mask each token with the probability t.** With $t = 2/3$, the forward process masks 2 of the 3 tokens on average. We fix the choice: positions 2 and 3. The input is `the [M] [M]`.

    **Why:** a mask is a token with no information, as pure noise is. Training uses all rates from 0 to 1. Thus the model learns to fill one gap, and also to write a full text from nothing.

2. **Call the model one time for all masked positions.** Assume the logits [0.5, 1.5, 0.8] at position 2. Assume the toy logits [0.1, 1.0, 2.0] at position 3.

    | Position | Correct token | Probabilities (`the`, `cat`, `sat`) |
    |---|---|---|
    | 2 | `cat` | [0.197, **0.536**, 0.266] |
    | 3 | `sat` | [0.099, 0.242, **0.659**] |

    **Why:** one call that predicts many positions is the source of the speed. An autoregressive model gives one distribution for each call.

3. **Calculate the loss: cross-entropy on the masked positions, multiplied by 1/t.**

    $$L = \frac{1}{t} \sum_{i\ \text{masked}} -\ln p_\theta(x_0^i \mid x_t)$$

    $$L = \tfrac{3}{2}\,(-\ln 0.536 - \ln 0.659) = 1.5 \cdot (0.623 + 0.417) = 1.560$$

    **Why:** a visible token gives no loss, because the model can copy it. A low rate masks few tokens, and the factor 1/t gives them more weight. The LLaDA authors show that this loss is an upper limit of the negative log-likelihood.

4. **Reverse: start from masks and unmask the position with the highest confidence.** **Unmasking** replaces a mask with a predicted token. The **confidence** of a position is the highest probability of its distribution. **Confidence-based unmasking** keeps the predictions with the highest confidence. Here it keeps 1 token for each call. The logits of calls 1 and 3 are assumed.

    | Call | Input | Best token and confidence at each masked position | Result |
    |---|---|---|---|
    | 1 | `[M] [M] [M]` | `the` 0.787, `cat` 0.412, `cat` 0.416 | `the [M] [M]` |
    | 2 | `the [M] [M]` | `cat` 0.536, `sat` 0.659 | `the [M] sat` |
    | 3 | `the [M] sat` | `cat` 0.883 | `the cat sat` |

    **Why:** the easy token comes first, and each new token is context for the hard ones. The confidence of `cat` at position 2 goes from 0.412 to 0.536 to 0.883. The order is position 1, then 3, then 2: not from left to right.

5. **Do the same in 1 step.** Keep the 3 predictions of call 1. The result is "the cat cat". With 2 steps (1 token, then 2 tokens), the result is "the cat sat".

    **Why:** in call 1, positions 2 and 3 each selected `cat`, and no position could see the other selection. Fewer steps are fewer calls, but more blind selections.

<div class="llm-widget" data-widget="text-diffusion"></div>

<div class="key-point">

Masked diffusion is a masked language model that is trained at all masking rates and used in a loop. One call predicts all masked positions. The sampler decides how many predictions to keep.

</div>

### From toy to real

- **Relation to BERT.** BERT-style masked language modelling hides a fixed part of the tokens, so the model learns to fill gaps but not to write a text from nothing. Masked diffusion trains with all masking rates from 0 to 1, and thus the same type of model becomes a generator.
- **Remasking** puts a mask back on a predicted token. In LLaDA, each step predicts all masked tokens and then remasks the predictions with the lowest confidence. This is the procedure of step 4. The DiffusionGemma page describes a sampler of the same type: it keeps the tokens with the lowest entropy and applies noise to the others again.
- **History and scale.** D3PM (2021) defined diffusion for discrete data, with the mask as one possible noise type. SEDD (2023) and MDLM (2024) gave better training losses. LLaDA (2025) has 8 billion parameters. Its authors trained it from zero on 2.3 trillion tokens with 0.13 million H800 GPU hours.
- **Continuous alternative.** Diffusion-LM (2022) adds Gaussian noise to the embeddings of the words, as in parts 1 and 2, and changes each vector back into a word at the end.

## 4. Autoregressive or diffusion: what changes

### Setup

Two models write a text of 12 tokens. The autoregressive model writes 1 token for each call. The diffusion model uses $k$ steps and writes 12 / $k$ tokens for each call.
**Parallel decoding** is the name for this: more than one token for each call. The cost unit is the **number of function evaluations (NFE)**: the number of model calls.

### Steps

1. **Count the model calls.**

    | Model | Calls | Tokens for each call | Positions that the model calculates |
    |---|---|---|---|
    | Autoregressive | 12 | 1 | 12 with a key-value cache, 78 without |
    | Diffusion, $k$ = 4 | 4 | 3 | 4 × 12 = 48 |
    | Diffusion, $k$ = 2 | 2 | 6 | 2 × 12 = 24 |
    | Diffusion, $k$ = 1 | 1 | 12 | 1 × 12 = 12 |

    **Why:** the calls run one after the other, so fewer calls mean less time to wait. But with $k$ = 4 the diffusion model calculates 48 positions, and the autoregressive model with a cache calculates 12. Diffusion is faster because a GPU calculates the positions of one call in parallel, not because it calculates less.

2. **Calculate all positions again in each call.** The [key-value cache](../engineer/06-inference-optimization.md) stores the work on earlier tokens, because an earlier token does not change and cannot read later tokens. In a diffusion model, each position reads all positions, and some positions change from mask to word in each step.

    **Why:** when a token changes, the keys and values of all positions change. The cache of an autoregressive model does not apply in the same way. The LLaDA authors state that their model is not compatible with this cache.

3. **Unmask two dependent positions in the same step.** Assume that the correct texts are "the cat" and "a dog", each with a probability of 0.5. For each position alone, the model gives: `the` 0.5 and `a` 0.5 at position 1, `cat` 0.5 and `dog` 0.5 at position 2. One step samples each position alone. Each of the 4 pairs has a probability of 0.5 × 0.5 = 0.25. The wrong texts "the dog" and "a cat" have a total probability of 0.5.

    **Why:** one call gives a distribution for each position, not for each pair of positions. This is the **independence error** of parallel unmasking.

4. **Use two steps.** Step 1 samples position 1: `the` or `a`. Step 2 reads position 1. After `the`, the model gives `cat` with a probability of 1. The probability of a wrong text is 0.

    **Why:** a token that depends on a second token must come in a later call. This is the reason why fewer steps are faster and worse.

5. **Fill a gap with context on the two sides.** For the input `the [M] sat`, the model of part 3 gives `cat` with 0.883. This is **any-order generation**. **Infilling** is the task to write a missing part in the middle of a text.

    **Why:** an autoregressive model reads only the left side. Many edits of text or code are a gap in the middle, not a new end.

<div class="key-point">

A diffusion language model replaces many small calls in sequence with a few large calls in parallel. The number of steps is a dial: more steps go toward one token at a time, fewer steps are faster but make more independent selections.

</div>

### From toy to real

Each line is a claim of its source, not an independent test. Only the Mercury Coder speed comes from a third party, as its paper reports.

| System | What its source reports |
|---|---|
| SEDD (2023) | A perplexity 25% to 75% lower than earlier language diffusion models, and better than GPT-2 |
| MDLM (2024) | A perplexity of 23.21 or less on OpenWebText, against 17.54 for an autoregressive Transformer (lower is better) |
| LLaDA (2025) | Results that the authors call competitive with Llama 3 8B on in-context learning |
| Mercury Coder (2025) | 1,109 and 737 tokens per second for its two sizes on NVIDIA H100 GPUs, measured by the company Artificial Analysis |
| Gemini Diffusion | An experimental demo: 1,479 tokens per second plus 0.84 seconds of overhead. Against Gemini 2.0 Flash-Lite: higher on LiveCodeBench (30.9% against 28.5%), lower on GPQA Diamond (40.4% against 56.5%) |
| DiffusionGemma (2026) | Open weights, fine-tuned from Gemma 4. Blocks of 256 tokens, 48 steps at most and usually 12 to 16. About 20 tokens for each call and about 1,500 tokens per second on one NVIDIA H100 GPU |
| Mercury 2 (2026) | 1,009 tokens per second on NVIDIA Blackwell GPUs. The page gives no benchmark scores |

- **The speeds are not comparable.** The hardware, the model size and the measurement method are different for each line.
- **The two families mix.** DiffusionGemma uses an autoregressive encoder that reads and caches the prompt, and a denoiser that reads its block of 256 tokens in the two directions. It writes one block after the other.
- **The gain depends on the load.** The DiffusionGemma page says that the speed advantage decreases on servers with many requests at the same time.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

E = np.array([[1., 0.], [0., 1.], [1., 1.]])       # data set: the, cat, sat
alpha_bar = np.array([0.9, 0.5, 0.1])              # noise schedule

def denoise(x, ab):                                # best denoiser for this data set
    w = softmax(-((x - np.sqrt(ab) * E) ** 2).sum(axis=1) / (2 * (1 - ab)))
    x0_hat = w @ E                                 # weighted mean of the data points
    return x0_hat, (x - np.sqrt(ab) * x0_hat) / np.sqrt(1 - ab)

x = np.array([0.3, -1.2])                          # start: a noise sample
for t in (2, 1, 0):                                # 3 deterministic (DDIM) reverse steps
    x0_hat, noise_hat = denoise(x, alpha_bar[t])
    ab_next = alpha_bar[t - 1] if t > 0 else 1.0
    x = np.sqrt(ab_next) * x0_hat + np.sqrt(1 - ab_next) * noise_hat
# x = [1.000, 0.005], near `the` = [1, 0]

# Masked diffusion on "the [M] [M]": one call gives the probabilities at positions 2 and 3
probs = {2: softmax(np.array([0.5, 1.5, 0.8])), 3: softmax(np.array([0.1, 1.0, 2.0]))}
loss = (-np.log(probs[2][1]) - np.log(probs[3][2])) / (2 / 3)   # 1.560
first = max(probs, key=lambda i: probs[i].max())                # position 3 (0.659)
```

## Comparison

| Model | Noise | Training loss | One call gives | Calls to generate | Toy result |
|---|---|---|---|---|---|
| Autoregressive LLM | None | Cross-entropy on the next token | 1 token | 1 for each token | 12 calls for 12 tokens |
| Image diffusion | Gaussian noise on each number | Mean squared error on the noise | A point with less noise | The number of sampling steps | [0.3, −1.2] becomes [1.000, 0.005] in 3 steps |
| Masked text diffusion | Mask tokens | Cross-entropy on the masked positions, multiplied by 1/t | A distribution at each masked position | The number of steps | "the cat sat" in 3 or 2 calls, "the cat cat" in 1 call |

## Checks and common misconceptions

- **"A diffusion language model calculates less."** It makes fewer calls, but each call calculates all positions. Check: measure the time and the cost of a complete answer for your text length and your load. Do not compare the tokens per second of two company pages.
- **"Fewer steps only make the model faster."** Fewer steps also put more tokens in the same call, and the model selects these tokens independently. Check: run your task with different numbers of steps and compare the quality.
- **"Text diffusion adds Gaussian noise to words."** Only the continuous type does this, on the embeddings. In masked diffusion a token is visible or masked.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. Each statement about a real system was read in the linked source on 11 November 2026.

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), Ho et al., 2020 (forward formula, noise loss, schedule, U-Net).
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502), Song et al., 2020 (deterministic step, 10 to 100 steps, reported speed-up of 10 to 50 times).
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598), Ho and Salimans, 2022 (training without the condition, guidance formula).
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), Rombach et al., 2021 (autoencoder, cross-attention).
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748), Peebles and Xie, 2022 (DiT).
- [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006), Austin et al., 2021 (D3PM: discrete diffusion, mask state).
- [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217), Li et al., 2022 (Gaussian noise on word vectors).
- [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834), Lou et al., 2023 (SEDD: perplexity claim, GPT-2).
- [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524), Sahoo et al., 2024 (MDLM: perplexities 23.21 and 17.54).
- [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992), Nie et al., 2025 (LLaDA: loss, remasking, no causal mask, no cache, sizes).
- [Mercury: Ultra-Fast Language Models Based on Diffusion](https://arxiv.org/abs/2506.17298), Inception Labs, 2025 (1,109 and 737 tokens per second), and [Introducing Mercury 2](https://www.inceptionlabs.ai/blog/introducing-mercury-2), 2026 (1,009 tokens per second).
- [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/), Google DeepMind (experimental status, speed, benchmark table).
- [DiffusionGemma](https://ai.google.dev/gemma/docs/diffusiongemma), Google, and the [DiffusionGemma Technical Report](https://arxiv.org/abs/2608.00146), 2026 (blocks, steps, sampler, encoder, speed, load).
