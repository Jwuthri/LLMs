# 5. Preference alignment

## Core idea

Supervised fine-tuning shows the model one good answer. **Preference alignment** shows the model which of two answers is better, or gives each answer a score.
The model then moves probability from the bad answers to the good answers.
It is the second stage of post-training, after [supervised fine-tuning](04-fine-tuning.md).

## The example

The prompt is "the cat" from the [toy example](../toy-example.md). An answer is one token.

| Item | Value |
|---|---|
| **Policy** (the model that we train) | Logits (raw scores) `[0.1, 1.0, 2.0]`, probabilities `[0.099, 0.242, 0.659]` |
| **Reference** (a frozen copy of the model from before alignment) | Assumed logits `[0.5, 1.0, 1.5]`, probabilities `[0.186, 0.307, 0.506]` |
| Chosen answer (the one we prefer) | `sat` |
| Rejected answer | `the` |
| Scorer (assumed) | `the` = 0.1, `cat` = 0.4, `sat` = 0.9 |

The policy was equal to the reference when alignment started. Some training steps have moved it already.
The **log-probability** is the natural logarithm ($\ln$) of a probability. A real answer has many tokens, and its log-probability is the sum of the log-probabilities of its tokens.

## 1. Rejection sampling

### Setup

Alignment needs pairs of (chosen, rejected) answers. **Rejection sampling** makes these pairs from the model itself: generate some answers, score them, keep the best and reject a bad one.

### Steps

1. **Sample N = 4 answers from the policy.** The 4 draws (seed 22) are `sat`, `cat`, `the`, `sat`.

    **Why:** these are answers that this model really gives. Such data is called **on-policy**. A pair from a different model can correct errors that this model never makes.

2. **Score each answer.** The scorer gives 0.9, 0.4, 0.1, 0.9. A scorer can be a person, a reward model (section 3), a judge model, or a rule (for example: "the code passes the tests").

    **Why:** the method needs an order between the answers. The source of the order is your choice.

3. **Keep the best answer as chosen and the worst as rejected.** The pair is (`sat`, `the`), with scores 0.9 and 0.1.

    **Why:** a large gap between the two scores is a clear signal. A pair with almost equal scores teaches noise.

<div class="key-point">

Rejection sampling turns a scorer into a preference dataset. The dataset fits the model, because the model wrote both answers.

</div>

### From toy to real

- Sample with a temperature above 0, or the N answers are all the same (see [sampling](01-architecture.md)).
- The cost is N generations for each prompt, plus N scores.
- Public preference datasets also exist. They are **off-policy**: other models wrote the answers.

## 2. Direct Preference Optimization (DPO)

### Setup

**DPO** trains the policy directly on the pairs, with one loss:

$$\begin{aligned} L &= -\ln \sigma(\beta \cdot m) \\ m &= \Delta_c - \Delta_r \\ \Delta_c &= \ln \pi(y_c) - \ln \pi_{ref}(y_c) \\ \Delta_r &= \ln \pi(y_r) - \ln \pi_{ref}(y_r) \end{aligned}$$

- $\pi$ is the policy and $\pi_{ref}$ is the reference. $y_c$ is the chosen answer and $y_r$ is the rejected answer.
- $\Delta_c$ and $\Delta_r$ are the **log-ratios** of the two answers. $m$ is the **margin**.
- $\sigma(x) = 1 / (1 + e^{-x})$ is the **sigmoid**. It turns any number into a number between 0 and 1.
- $\beta$ sets how strongly the reference holds the policy. Assume $\beta = 0.5$.

<figure class="diagram" data-src="preference-alignment/dpo.svg">
<figcaption>DPO on the pair (sat, the). Two models read the prompt, and only the policy learns.</figcaption>
</figure>

### Steps

1. **Calculate the log-probability of each answer with each model.**

    | Answer | Policy $\ln \pi$ | Reference $\ln \pi_{ref}$ |
    |---|---|---|
    | `sat` (chosen) | $\ln 0.659 = -0.417$ | $\ln 0.506 = -0.680$ |
    | `the` (rejected) | $\ln 0.099 = -2.317$ | $\ln 0.186 = -1.680$ |

    **Why:** the loss compares what the policy does now with what the model did before alignment.

2. **Calculate the two log-ratios.** $\Delta_c = -0.417 - (-0.680) = +0.263$. $\Delta_r = -2.317 - (-1.680) = -0.637$.

    **Why:** a log-ratio measures the *change* from the reference. An answer that was already probable gets no credit for that.

3. **Calculate the margin.** $m = 0.263 - (-0.637) = 0.900$.

    **Why:** the margin is one number: how much more the policy moved to the chosen answer than to the rejected answer.

4. **Calculate the loss.** $\beta \cdot m = 0.5 \cdot 0.9 = 0.45$, and $\sigma(0.45) = 0.611$.

    $$L = -\ln 0.611 = 0.493$$

    At the start of alignment, the policy was the reference. The margin was 0 and the loss was $-\ln 0.5 = 0.693$.

    **Why:** the sigmoid reads the margin as a probability that the policy prefers the chosen answer. The loss is small when this probability is near 1.

5. **Do one gradient step on the logits.** The gradient is $-(1 - 0.611) \cdot 0.5 = -0.195$ for `sat`, $+0.195$ for `the` and 0 for `cat`. With a learning rate of 1:

    | | `the` | `cat` | `sat` |
    |---|---|---|---|
    | Logits before | 0.1 | 1.0 | 2.0 |
    | Logits after | −0.095 | 1.0 | 2.195 |
    | Probabilities after | 0.072 | 0.216 | 0.712 |

    The chosen token `sat` goes from 0.659 to 0.712. The margin is now 1.289 and the loss is $-\ln \sigma(0.645) = -\ln 0.656 = 0.422$.

    **Why:** the factor $(1 - \sigma)$ becomes small when the margin is large. Pairs that the policy already separates well push less, so the policy does not run away from the reference.

<figure class="plot">
<img src="/figures/preference-alignment/dpo-loss.svg" alt="DPO loss against the margin for beta 0.5 and 0.1. With beta 0.5 the loss is 0.693 at margin 0, 0.493 at margin 0.9 and 0.422 at margin 1.289.">
<figcaption>The loss falls when the margin grows. The three points are the start, the toy policy now, and the policy after one step.</figcaption>
</figure>

<div class="key-point">

DPO is a classification loss on pairs. It needs no reward model and no generation during training: only the probabilities of two given answers, from two models.

</div>

### From toy to real

- The default in the TRL library is $\beta = 0.1$. With it, the toy loss is 0.649. A higher $\beta$ keeps the policy nearer to the reference.
- The default learning rate of the TRL trainer is $1 \times 10^{-6}$, much lower than for supervised fine-tuning.
- Watch the margin and the two log-probabilities during training. The TRL documentation notes that the margin usually grows because the rejected answers become less likely, not because the chosen answers become more likely.

## 3. Reward model

### Setup

A **reward model** maps a (prompt, answer) pair to one number: a higher number means a better answer. It is usually an LLM with a small output layer that gives one score.
You train it on human comparisons with the **Bradley-Terry** model: the probability that the chosen answer wins is $\sigma(r_c - r_r)$, where $r_c$ and $r_r$ are the two scores.

### Steps

1. **Score the two answers.** Assume that the new reward model gives $r_c = 1.2$ for `sat` and $r_r = 0.4$ for `the`.

    **Why:** one score for each answer lets you rank any number of answers, not only pairs.

2. **Calculate the win probability.** $\sigma(1.2 - 0.4) = \sigma(0.8) = 0.690$.

    **Why:** only the difference between the scores counts. People give comparisons ("A is better than B") more reliably than absolute grades.

3. **Calculate the loss.** $L = -\ln 0.690 = 0.371$.

    **Why:** the loss is high when the model gives a low probability to the result that the people chose.

4. **Do one gradient step.** The gradient is $-(1 - 0.690) = -0.310$ for $r_c$ and $+0.310$ for $r_r$. With a learning rate of 1, the scores become 1.510 and 0.090. The win probability is $\sigma(1.420) = 0.805$ and the loss is 0.216.

    **Why:** training pushes the two scores apart until the order of the scores agrees with the order that the people gave.

<div class="key-point">

The DPO loss is this same loss with $r = \beta \cdot \text{log-ratio}$. DPO uses the policy itself as the reward model. A separate reward model is necessary only when you want to score new answers.

</div>

### From toy to real

- You train the reward model one time, freeze it, and use it to score a very large number of samples.
- A reward model is an estimate of human preference, and a policy can learn to exploit its errors. This is one reason why the methods of section 4 limit each update.
- [TRL](https://huggingface.co/docs/trl/en/index), [verl](https://github.com/volcengine/verl) and [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) are libraries for this training at scale.

## 4. Reinforcement learning

### Setup

In **reinforcement learning (RL)**, the policy generates answers, gets a **reward** (a score) for each, and changes to make answers with a high reward more probable.
We reuse the 4 samples of section 1. Their scores are the rewards: 0.9, 0.4, 0.1, 0.9.

<figure class="diagram" data-src="preference-alignment/grpo.svg">
<figcaption>The group-relative advantage compares each answer with the other answers to the same prompt.</figcaption>
</figure>

### Steps

1. **Calculate the advantage of each sample.** The **advantage** $A$ says how much better an answer is than usual. **GRPO** (Group Relative Policy Optimization) uses the group of samples as "usual":

    $$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}, \qquad \text{mean} = 0.575, \qquad \text{std} = 0.342$$

    $$A = [0.951,\ -0.512,\ -1.389,\ 0.951]$$

    This page divides by N = 4 in the standard deviation. The TRL library divides by N − 1, which gives a standard deviation of 0.395 here.

    **Why:** a reward of 0.4 is good for a hard prompt and bad for an easy prompt. The group mean removes this difference, and the division puts all prompts on one scale.

2. **Calculate the policy gradient.** The **policy gradient** says how to change the logits so that the average reward goes up. For one sampled token $a$, the gradient of $\ln \pi(a)$ is (one-hot vector of $a$) $- p$. Multiply by $A_i$ and take the mean of the 4 samples:

    $$g = \tfrac{1}{4}\left[-1.389,\ -0.512,\ 0.951 + 0.951\right] = [-0.347,\ -0.128,\ 0.475]$$

    The $p$ terms cancel, because the advantages add up to 0.

    **Why:** a sample with a positive advantage pulls its token up. A sample with a negative advantage pushes its token down.

3. **Update the logits.** With a step size of 1: $[0.1, 1.0, 2.0] + g = [-0.247,\ 0.872,\ 2.475]$. The probabilities become $[0.052,\ 0.159,\ 0.789]$.

    **Why:** `sat` got the best reward two times, so it gains the most. No person had to write a "correct" answer.

4. **Limit the update with the PPO clip.** **PPO** (Proximal Policy Optimization) calculates the ratio $\rho = \pi_{new}(a) / \pi_{old}(a)$ and uses this objective for each sample, with $\varepsilon = 0.2$:

    $$\min\big(\rho A,\ \text{clip}(\rho,\ 1 - \varepsilon,\ 1 + \varepsilon)\, A\big)$$

    After step 3, $\rho = 0.789 / 0.659 = 1.198$ for `sat`. It is inside $[0.8, 1.2]$, so nothing changes.
    With a step size of 2, `sat` goes to 0.878 and $\rho = 1.332$. The clip cuts it to 1.2: $\min(1.332 \cdot 0.951,\ 1.2 \cdot 0.951) = \min(1.267,\ 1.141) = 1.141$.

    **Why:** above the clip, the objective stops growing, so this sample gives no more reason to move. The samples came from the old policy. Far from the old policy, they do not describe the model any more.

<div class="key-point">

RL needs only a score, not a better answer. The advantage decides the direction, and the clip limits how far one update can go.

</div>

### From toy to real

- PPO (2017) learns a second model, the **value model**, to estimate the "usual" reward. GRPO (2024) replaces it with the group mean, which is simpler and cheaper.
- GRPO uses the same clipped objective as PPO. In TRL, the defaults are 8 samples for each prompt and $\varepsilon = 0.2$.
- The reward can come from a reward model or from a **reward function** (a rule in code, for example "the final number is correct"). This makes RL a good fit for reasoning models.
- RL generates text during training. It costs much more compute than DPO and is more sensitive to the learning rate, the batch size and the clip range.

## Minimal code

This code gives the same numbers as sections 2 and 4.

```python
import numpy as np

def log_softmax(z):
    return z - np.log(np.exp(z).sum())

policy, ref = np.array([0.1, 1.0, 2.0]), np.array([0.5, 1.0, 1.5])
chosen, rejected, beta = 2, 0, 0.5             # "sat", "the"

ratio = log_softmax(policy) - log_softmax(ref) # [-0.637, -0.237, 0.263]
margin = ratio[chosen] - ratio[rejected]       # 0.900
loss = -np.log(1 / (1 + np.exp(-beta * margin)))   # 0.493

rewards = np.array([0.9, 0.4, 0.1, 0.9])       # samples: sat, cat, the, sat
advantages = (rewards - rewards.mean()) / rewards.std()   # [0.951, -0.512, -1.389, 0.951]
```

Real-library sketch (the tests do not run this code):

```python
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

trainer = DPOTrainer(
    model="Qwen/Qwen3-0.6B",                   # the reference is a frozen copy of this model
    args=DPOConfig(beta=0.1, learning_rate=1e-6),
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
)
trainer.train()                                # columns: chosen, rejected (prompt + answer)
```

## Comparison

| Method | Input | Extra model | Generates during training | Toy result | Good for | Risk |
|---|---|---|---|---|---|---|
| Rejection sampling | Prompts + a scorer | The scorer | Yes, before training | Pair (`sat`, `the`) | Making on-policy pairs | N generations for each prompt |
| DPO | Pairs (chosen, rejected) | Frozen reference | No | Loss 0.493 → 0.422 | Chat models, style, tone | Limited by the quality of the pairs |
| Reward model | Human comparisons | It is the extra model | No | Loss 0.371 → 0.216 | A scorer that you can reuse | The policy exploits its errors |
| PPO | Prompts + rewards | Value model | Yes | Ratio 1.332 cut to 1.2 | Stable RL updates | Many hyperparameters |
| GRPO | Prompts + rewards | None | Yes, a group for each prompt | Advantage 0.951 for `sat` | Reasoning models, rule rewards | High compute cost |

## Checks and common misconceptions

- **"DPO makes the chosen answer more probable."** DPO makes the *margin* larger. The loss also falls when both answers become less probable and the rejected answer falls faster. Check: record the log-probability of the chosen answers. If it falls for a long time, the model is forgetting how to write good answers.
- **"A loss of 0.693 at the start means a bug."** It is the correct start value: the policy equals the reference, the margin is 0, and $-\ln 0.5 = 0.693$. Check: a first loss far from 0.693 means that the reference is not a copy of the policy.
- **"A higher reward means a better model."** The reward comes from a scorer, and the scorer can be wrong. Check: do a separate [evaluation](06-evaluation.md) with data and judges that the training never used.

## Sources

This page follows section 5 of "The LLM Scientist" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [Illustrating RLHF](https://huggingface.co/blog/rlhf), Hugging Face.
- [LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives), Sebastian Raschka.
- [Preference Tuning LLMs](https://huggingface.co/blog/pref-tuning), Hugging Face.
- [Fine-tune with DPO](https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html) and [Fine-tune with GRPO](https://huggingface.co/learn/llm-course/en/chapter12/5), Maxime Labonne.
- [DPO Wandb logs](https://wandb.ai/alexander-vishnevskiy/dpo/reports/TRL-Original-DPO--Vmlldzo1NjI4MTc4), Alexander Vishnevskiy.
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), Rafailov et al., 2023 (the DPO loss).
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), Schulman et al., 2017 (the clipped objective).
- [DeepSeekMath](https://arxiv.org/abs/2402.03300), Shao et al., 2024 (GRPO).
- [DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) and [GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer), TRL documentation (default $\beta$, learning rate, group size, $\varepsilon$; the note on rejected answers).
