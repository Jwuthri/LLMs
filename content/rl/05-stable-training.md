# 5. Stable RL training

## Core idea

An RL run can fail even when the algorithm is correct.
The policy can drift far from its start, stop exploring, or learn from data that a slightly different model wrote.
This lesson shows four tools against this: a KL term to the reference model, entropy control, an importance-weight correction, and a set of signals to monitor.

## The example

This page uses the shared toy of the track: the prompt "Copy: the cat sat", 3 positions with their own logits, and the target "the cat sat" (see [RL basics](01-rl-basics.md)).
It adds these values:

| Item | Value |
|---|---|
| Reference model $\pi_{ref}$ | the toy start: logits [0.1, 1.0, 2.0], probabilities [0.099, 0.242, 0.659] |
| Policy at position 1 after some updates (assume) | logits [2.5, 1.0, 2.0], probabilities [0.547, 0.122, 0.331] |
| Simulated GRPO run (assume) | group of 8, partial reward, step size 4, 2 updates per batch, clip 0.2, seed 0, 100 steps |
| "Collapsed" start (assume) | position 2 has logits [0.1, -3.0, 3.0], so P(`cat`) = 0.0023 |
| Fast sampler (assume) | at that position, the logit of `cat` is 0.85 lower than in the trainer |

The **partial reward** is the fraction of correct tokens.
The simulation uses the GRPO advantage and the PPO clip of [GRPO and its successors](04-grpo-family.md).

## 1. KL to the reference model

### Setup

The **reference model** $\pi_{ref}$ is a frozen copy of the model before RL, usually the SFT model.
A KL term measures how far the policy $\pi$ moved from it (see [entropy and KL](../fundamentals/05-metrics.md#2-entropy-and-kl-divergence)).
The term keeps the policy near text that the reference model can write. This limits [reward hacking](06-reward-hacking.md): the policy cannot go far to find a strange answer that the reward model scores too high.

$$\text{KL}(\pi \,\|\, \pi_{ref}) = \sum_a \pi(a) \log \frac{\pi(a)}{\pi_{ref}(a)}$$

Trainers often keep only the log-probability of the sampled token, so they cannot calculate the sum. They use an **estimator**: a number from one sampled token $a \sim \pi$ whose mean is (close to) the KL. With $r = \pi_{ref}(a) / \pi(a)$, Schulman (2020) defines three:

$$k_1 = -\log r, \qquad k_2 = \tfrac{1}{2}(\log r)^2, \qquad k_3 = (r - 1) - \log r$$

### Steps

1. **Calculate the exact KL at position 1.** The policy is [0.547, 0.122, 0.331] and the reference is [0.099, 0.242, 0.659]. The sum gives $\text{KL} = 0.625$.

    **Why:** the toy has only 3 tokens, so we can enumerate. The exact value is the target for each estimator.

2. **Calculate r for each token.** $r$ = [0.180, 1.988, 1.988] for `the`, `cat`, `sat`.

    **Why:** an estimator sees only the sampled token and its two log-probabilities. The ratio is all that it knows.

3. **Calculate k1.** $k_1$ = [1.713, −0.687, −0.687]. Its mean under $\pi$ is 0.625, the exact KL. Its standard deviation is 1.195.

    **Why:** $k_1$ is the log-ratio, so its mean is the definition of KL. But a sample of `cat` or `sat` gives a negative number, although a KL is never negative.

4. **Calculate k2.** $k_2$ = [1.467, 0.236, 0.236]. Its mean is 0.909, not 0.625. Its standard deviation is 0.613.

    **Why:** $k_2$ is never negative and has a lower spread, but it is **biased**: its mean is not the KL. Schulman shows that the bias is small only when the two models are close. Here they are far apart.

5. **Calculate k3.** $k_3$ = [0.893, 0.301, 0.301]. Its mean is 0.625, the exact KL. Its standard deviation is 0.295.

    **Why:** $k_3$ adds $r - 1$ to $k_1$. The mean of $r$ under $\pi$ is 1, so the mean does not change, but the noise mostly cancels. Because $\log r \le r - 1$, each sample is $\ge 0$.

6. **Use the KL in the reward or in the loss.** Assume $\beta = 0.04$ and the sampled token `the`. As a per-token reward penalty, the token gets $-\beta k_1 = -0.069$. As a loss term, it adds $\beta k_3 = 0.036$.

    **Why:** a reward penalty goes into the advantage, so it acts through the policy gradient. A loss term has its own gradient and does not change the advantage.

<figure class="plot">
<img src="/figures/stable-training/kl-estimators.svg" alt="k1, k2 and k3 against the ratio r. k1 is -0.687 for cat and sat at r = 1.988; k3 is 0.893 for the and 0.301 for cat and sat.">
<figcaption>Each estimator for one sample. Only k1 goes below 0. The dots are the tokens of position 1.</figcaption>
</figure>

<div class="llm-widget" data-widget="kl-estimators"></div>

<div class="key-point">

The KL term is a leash to the reference model. $k_1$ and $k_3$ have the correct mean (0.625 here), and $k_3$ has a quarter of the spread of $k_1$ (0.295 vs 1.195) and is never negative.

</div>

### From toy to real

- **InstructGPT** (2022) subtracts a per-token KL penalty from the SFT model from the reward, "to mitigate over-optimization of the reward model".
- **GRPO** (DeepSeekMath, 2024) puts the KL in the loss, not in the reward. Its formula is $k_3$, which the paper calls unbiased and always positive. The RL run uses $\beta = 0.04$ and a learning rate of 1e-6.
- **Some recipes drop the KL.** DAPO (2025) removes it: in long chain-of-thought training the model can move far from the start, so the authors find the leash not necessary. Dr. GRPO (2025) also sets $\beta = 0$. The TRL `GRPOTrainer` uses `beta=0.0` by default and then does not load the reference model, which saves memory and time.

## 2. Entropy collapse

### Setup

The **entropy** of a position measures how unsure the policy is. At the toy start, it is 0.847 at each position.
**Entropy collapse** is a fast fall of the entropy early in training. The policy then samples almost the same answer each time, so **exploration** stops: RL can only learn from answers that it samples.
We simulate the GRPO run of the table, first on the shared toy, then on the collapsed start.

### Steps

1. **Run plain GRPO.** The mean entropy per position is 0.797, 0.578, 0.295, 0.079 and 0.019 after steps 1, 5, 10, 20 and 100. The mean reward is 0.292 at step 1 and 1.000 at step 10. Position 3 falls first (0.177 at step 5), because `sat` is correct and already likely. Position 1 first rises to 1.044, because `the` grows from 0.099.

    **Why:** a likely token that gets a positive advantage becomes even more likely, so the distribution becomes sharper. Cui et al. (2025) describe this mechanism in real models.

2. **Look at the learning signal.** The first group in which all 8 answers have the same reward comes at step 10. In steps 51 to 100, 90% of the groups are like this.

    **Why:** the GRPO advantage is the reward minus the group mean. Identical rewards give advantages of 0, so the gradient is 0, also when the policy is wrong.

3. **Add an entropy bonus.** Add $c \cdot H(\pi)$ to the objective, with $c = 0.1$. At step 100 the mean entropy is 0.056, not 0.019. The mean reward in steps 51 to 100 is 0.973, not 0.995.

    **Why:** the bonus pushes probability back to the other tokens. This toy has one correct answer, so a sharp policy is correct, and the bonus costs reward. Entropy is a trade-off, not a goal.

4. **Start from the collapsed position.** P(`cat`) = 0.0023 at position 2. We run 40 seeds of 150 steps and count the runs that end with the target as the most probable answer.

    | Run | Solved (of 40) | Mean entropy at the end |
    |---|---|---|
    | Plain | 35 | 0.030 |
    | Clip-higher, upper clip 0.28 | 36 | 0.025 |
    | Entropy bonus 0.02 | 40 | 0.018 |
    | Entropy bonus 0.1 | 40 | 0.132 |

    **Why:** when `cat` is almost never sampled, all answers get the same reward at position 2, and plain GRPO has nothing to learn from: 5 runs stay stuck. A small bonus raises P(`cat`) until the policy samples it. Then the run collapses on the target, so its entropy is low too. 36 vs 35 is noise with 40 seeds: the toy shows no effect of clip-higher.

<figure class="plot">
<img src="/figures/stable-training/entropy-positions.svg" alt="Entropy of each position over 100 steps of plain GRPO. All start at 0.847; position 3 is 0.177 at step 5; all are below 0.09 at step 20.">
<figcaption>Plain GRPO on the shared toy. The position that is already correct collapses first.</figcaption>
</figure>

<figure class="plot">
<img src="/figures/stable-training/entropy-bonus.svg" alt="Mean entropy per position over 100 steps: 0.019 at step 100 without a bonus, 0.056 with a bonus of 0.1.">
<figcaption>An entropy bonus of 0.1 keeps the entropy higher during the whole run.</figcaption>
</figure>

<div class="key-point">

RL makes likely, rewarded tokens more likely, so entropy falls fast. If it falls before the policy finds the good answer, the groups become identical and learning stops (5 of 40 runs here). A small entropy bonus fixed all 40 runs in this toy.

</div>

### From toy to real

- Cui et al. (2025) report that, without an intervention, the entropy of reasoning models falls sharply early in RL, and performance saturates at the same time. They fit $R = -a e^{H} + b$ between entropy $H$ and performance $R$.
- They explain the entropy change by the covariance between the probability of a token and the change of its logit, which is proportional to the advantage. Their fixes, **Clip-Cov** and **KL-Cov**, clip or penalize the updates of high-covariance tokens.
- DAPO reports entropy collapse with plain PPO and GRPO. Its **clip-higher** uses a lower clip of 0.2 and an upper clip of 0.28, so low-probability tokens can grow more (see [GRPO and its successors](04-grpo-family.md)). DAPO links a very high entropy to gibberish and repetition, so a bonus must stay small.

## 3. Training-inference mismatch

### Setup

RL frameworks often generate answers with a fast **inference engine** (for example vLLM) and train with another implementation (for example FSDP in PyTorch).
The kernels differ, and the engine sometimes uses a lower precision. With the same weights, the two give slightly different probabilities.
Thus the answers come from a sampler $\mu$ that is not exactly the trained policy $\pi$. The data is a little **off-policy**.

### Steps

1. **Compare the probabilities.** At the collapsed position 2, the trainer gives `cat` 0.0023 and the sampler gives 0.0010. For `the` and `sat`, the two differ by less than 0.002.

    **Why:** a small error in a log-probability is a large relative error for a rare token. Rare tokens are the exploration tokens of part 2.

2. **Calculate the importance weights.** $w = \pi / \mu$ = [0.999, 2.337, 0.999]. Their mean under the sampler is 1.000.

    **Why:** a token that the sampler draws too rarely gets a weight above 1 when it does appear.

3. **Compare the gradients.** Assume reward 1 for `cat` and 0 for the others. The gradient on the logit of `cat` is $\mathbb{E}[R \,(1 - \pi_{cat})]$. Without a correction, it is 0.428 times the on-policy gradient. With the weight $w$, it is exactly 1.000 times.

    **Why:** without the correction, the trainer sees `cat` too rarely and learns it 57% too slowly. The weight makes the mean correct.

4. **Truncate the weight.** **Truncated importance sampling (TIS)** uses $\min(w, C)$. Assume $C = 2$. The weight of `cat` becomes 2.000, and the gradient is 0.856 times the on-policy gradient.

    $$\nabla J \approx \mathbb{E}_{a \sim \mu}\big[\min(w(a),\ C)\; A(a)\, \nabla \log \pi(a)\big]$$

    **Why:** a very rare sampler token can give a huge $w$ and one huge update. The cap limits the variance and adds a small bias (here 14% too slow).

<div class="key-point">

The inference engine and the trainer are two slightly different models. The weight $\pi / \mu$ removes the bias (0.428 → 1.000 here), and a cap keeps one rare token from dominating the update.

</div>

### From toy to real

- Yao et al. (2025) show that frameworks such as VeRL, with vLLM for rollouts and FSDP for training, implicitly turn on-policy RL into off-policy RL. They propose TIS and report that it also keeps 8-bit rollouts close to BF16 rollouts.
- In TRL, `vllm_importance_sampling_correction` is on by default. The default mode is sequence-level **masked importance sampling** (MIS): weights outside the range become zero, where TIS clips them. The default upper cap is 3.0, so it would not cut our weight of 2.337.
- TRL logs `sampling/sampling_logp_difference` and `sampling/importance_sampling_ratio`. A sudden rise can mean that the engine and the trainer disagree more, for example after a change of precision.

## 4. What to monitor

### Setup

The reward curve alone hides most failures. We record the signals of the plain run of part 2 at each step. The toy has one group per step, so the plot shows the zero-advantage share over the last 10 steps.

### Steps

1. **Mean reward and KL.** The reward is 0.292 at step 1 and 1.000 at step 10. The exact KL, summed over the 3 positions, is 0.075, 1.370, 3.165 and 4.094 after steps 1, 5, 10 and 100.

    **Why:** a rising reward can also mean a hack, so compare it with a held-out evaluation. The KL shows how far the policy went: a fast rise that the held-out score does not explain is a sign of drift.

2. **Entropy.** 0.797 → 0.019, as in part 2.

    **Why:** a fall to almost 0 early means that exploration stopped.

3. **Clip fraction.** The share of tokens outside the clip range is 0.125 at step 1 and 0.333 at step 5. It is at most 0.375, then mostly 0.

    **Why:** a high clip fraction means updates that are too large for the clip range. The clipped tokens give no gradient.

4. **Zero-advantage share and gradient norm.** The first zero-advantage group comes at step 10, and the share is 0.90 in steps 51 to 100. The gradient norm is 0.188 at step 1, 0.192 at step 5 and 0 at step 10.

    **Why:** these groups cost a full generation and give no gradient. A norm of 0 means no signal. A spike means one batch with extreme weights or advantages.

5. **Response length.** The toy always writes 3 tokens, so it cannot show this signal.

    **Why:** in real runs, the length follows the reasoning behaviour. DAPO reads it together with validation accuracy.

<figure class="plot">
<img src="/figures/stable-training/dashboard.svg" alt="Six plots over 100 steps: reward reaches 1.0 at step 10; entropy falls to 0.02; KL rises to 4.09; clip fraction peaks at 0.375; zero-advantage share rises to about 0.9; gradient norm is mostly 0 after step 10.">
<figcaption>The plain run. After step 10 the reward is perfect, the entropy is almost 0, and most steps give no gradient.</figcaption>
</figure>

The patterns below are rules of thumb, not fixed thresholds.

| Signal | Healthy | Alarming | Toy run |
|---|---|---|---|
| Mean reward | rises with held-out accuracy | rises while held-out accuracy is flat | 0.292 → 1.000 |
| KL to reference | grows slowly, then levels off | fast rise, no held-out gain | 0.075 → 4.094 |
| Entropy | falls slowly, stays above 0 | near 0 early, or rises without limit | 0.797 → 0.019 |
| Clip fraction | small, stable | large or rising | at most 0.375 |
| Zero-advantage share | well below 1 | near 1 | 0.90 |
| Gradient norm | stable | spikes, or 0 for many steps | mostly 0 after step 10 |
| Response length | changes with accuracy | runs to the limit, or collapses | always 3 |

<div class="key-point">

Read the signals together. In the toy, reward 1.000 looks perfect, but the entropy, the zero-advantage share and the gradient norm show that learning stopped at step 10.

</div>

### From toy to real

- The TRL `GRPOTrainer` logs `kl` (only when `beta` is not 0), `entropy`, `completions/mean_length`, `completions/clipped_ratio` (answers cut at the length limit), `clip_ratio/region_mean` and `frac_reward_zero_std` (groups with a reward standard deviation of 0).
- DAPO (section 4.3) says that the training reward often has little correlation with validation accuracy. It reads a too-low entropy as an over-sharp distribution, and reports that a slow upward trend in entropy helps.
- DAPO's **dynamic sampling** removes prompts with accuracy 0 or 1 from the batch, so that all groups give a gradient.

## Minimal code

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

pi = softmax(np.array([2.5, 1.0, 2.0]))        # policy, position 1: [0.547, 0.122, 0.331]
ref = softmax(np.array([0.1, 1.0, 2.0]))       # reference:          [0.099, 0.242, 0.659]
r = ref / pi
k1, k2, k3 = -np.log(r), np.log(r) ** 2 / 2, (r - 1) - np.log(r)
print(pi @ k1, pi @ k2, pi @ k3)               # 0.625 0.909 0.625 (exact KL = 0.625)

# Truncated importance weight for the mismatch between trainer and sampler
trainer = softmax(np.array([0.1, -3.0, 3.0]))
sampler = softmax(np.array([0.1, -3.85, 3.0]))
w = trainer / sampler                          # [0.999, 2.337, 0.999]
w_tis = np.minimum(w, 2.0)                     # [0.999, 2.000, 0.999]
```

## Comparison

| Problem | What you see | Toy number | Usual fix | Cost of the fix |
|---|---|---|---|---|
| Drift from the reference | KL grows fast | KL 4.094 | KL term (k3 in the loss, or −β·k1 in the reward) | slower learning, a reference model in memory |
| Entropy collapse | entropy near 0, zero-advantage groups | 5 of 40 runs stuck | entropy bonus, clip-higher, Clip-Cov, KL-Cov | lower reward if too strong (0.973 vs 0.995) |
| Train-inference mismatch | log-probs differ | w = 2.337 | importance weight, truncated (TIS) or masked (MIS) | small bias (0.856 of the gradient) |
| Silent failure | reward looks fine | reward 1.000, gradient 0 | monitor all signals together | none |

## Checks and common misconceptions

- **"A negative KL value in my logs is a bug."** Not with $k_1$: one sample can be negative (−0.687 here) while the mean is correct. A negative mean over a large batch is suspicious. $k_3$ is never negative.
- **"Entropy collapse is always bad."** With one correct answer, a sharp policy is the goal, and the bonus of 0.1 cost reward in the toy. Collapse is bad when it comes before the policy finds the good answers. Check the zero-advantage share and held-out accuracy, not the entropy alone.
- **"Same weights means on-policy."** The engine and the trainer can give different probabilities with the same weights. Log the log-probability difference and correct with an importance weight.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. Each statement about a real system was read in the linked source on 25 November 2026.

- [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html), John Schulman, 2020 (k1, k2, k3 and their bias and variance).
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), Ouyang et al., 2022 (per-token KL penalty in the reward, and its reason).
- [DeepSeekMath](https://arxiv.org/abs/2402.03300), Shao et al., 2024 (GRPO with the k3 KL in the loss, β = 0.04, learning rate 1e-6).
- [DAPO](https://arxiv.org/abs/2503.14476), Yu et al., 2025 (no KL and why, clip-higher 0.2 and 0.28, entropy collapse, dynamic sampling, monitoring in section 4.3).
- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783), Liu et al., 2025 (Dr. GRPO with β = 0).
- [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617), Cui et al., 2025, abstract (entropy fall, the fitted law, covariance, Clip-Cov and KL-Cov).
- [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://liyuanlucasliu.github.io/publication/feng-off-2025/), Yao, Liu et al., 2025, and its [NeurIPS 2025 workshop page](https://neurips.cc/virtual/2025/126728) (vLLM vs FSDP mismatch, TIS, 8-bit rollouts).
- [GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer), TRL documentation (default `beta=0.0`, k3 formula, metric names, TIS and MIS, cap 3.0).
