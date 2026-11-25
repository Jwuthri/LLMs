# 3. Actor-critic and PPO

## Core idea

A policy gradient needs a **baseline**: a guess of the "usual" reward, so that the update can see which answers are better than usual.
The baseline can come from the other answers to the same prompt (RLOO), from one greedy answer (ReMax), or from a learned **value model** that gives a usual value at each token.
PPO uses the value model to give each token its own advantage (with GAE), and a clip that lets it reuse one batch of answers for several updates.

## The example

This page uses the shared toy task of the track (see [RL basics](01-rl-basics.md)). The prompt is "Copy: the cat sat". The tabular policy writes 3 tokens. At each position it starts with the probabilities [0.099, 0.242, 0.659] for `the`, `cat`, `sat`. The reward is the **partial reward**: the fraction of correct tokens.

| Answer | Partial reward | Note |
|---|---|---|
| the cat sat | 1.000 | the target |
| sat cat sat | 0.667 | sub-parts 2 to 4 follow it token by token |
| the sat sat | 0.667 | |
| sat sat sat | 0.333 | also the greedy answer |

The probability of the correct token at each position is 0.099 (`the`), 0.242 (`cat`) and 0.659 (`sat`). The discount is $\gamma = 1$ unless the text says otherwise.

## 1. Better baselines without a critic

### Setup

[RL basics](01-rl-basics.md) subtracts the group mean from each reward. That mean includes the answer itself. Two cheap fixes keep the baseline independent of the answer and need no extra trained model. The REINFORCE gradient with a baseline $b_i$ is:

$$g = \frac{1}{G}\sum_{i=1}^{G} (r_i - b_i)\, \nabla_\theta \log \pi_\theta(y_i \mid x)$$

### Steps

1. **Take the group rewards.** $r = [1.000,\ 0.667,\ 0.667,\ 0.333]$, sum = 2.667, mean = 0.667.

    **Why:** each baseline on this page guesses the usual reward for this prompt. The group is the cheapest data for that guess.

2. **RLOO: leave the answer itself out.** **RLOO** (REINFORCE Leave-One-Out) uses the mean of the other $G - 1$ rewards:

    $$b_i = \frac{1}{G-1}\sum_{j \ne i} r_j, \qquad b = \left[\tfrac{2.667 - 1}{3},\ \dots\right] = [0.556,\ 0.667,\ 0.667,\ 0.778]$$

    The advantages $r_i - b_i$ are $[0.444,\ 0,\ 0,\ -0.444]$.

    **Why:** if the baseline contains $r_i$, a high reward also raises its own baseline, and the signal of that answer gets smaller. Without $r_i$, the baseline does not depend on the sampled answer, so the gradient stays unbiased.

3. **See that RLOO is the group mean, rescaled.** Write $\bar r$ for the group mean:

    $$r_i - b_i = \frac{G}{G-1}\,(r_i - \bar r) = \tfrac{4}{3} \cdot [0.333,\ 0,\ 0,\ -0.333] = [0.444,\ 0,\ 0,\ -0.444]$$

    **Why:** the direction is the same as with the plain group mean. Only the scale changes, by $G/(G-1)$. GRPO also divides by the standard deviation, which gives $[1.414,\ 0,\ 0,\ -1.414]$ here (see [GRPO and its successors](04-grpo-family.md)).

4. **ReMax: use the greedy answer as the baseline.** **ReMax** decodes one extra answer with greedy search (the most probable token at each position). Here `sat` wins at each position, so the greedy answer is "sat sat sat", with reward 0.333. The advantages are $r_i - 0.333 = [0.667,\ 0.333,\ 0.333,\ 0]$.

    **Why:** the greedy answer is a cheap guess of what the policy "normally" writes, and it works for any group size, even one sample. Here the greedy answer is weak, so the advantages are all 0 or more, unlike RLOO.

<div class="key-point">

RLOO and ReMax change only the baseline. Each answer still gets one advantage, and all its tokens share it.

</div>

### From toy to real

- The RLOO paper (Ahmadian et al., 2024) argues that many parts of PPO are not necessary for RLHF. It reports that REINFORCE-style methods, RLOO among them, do better than PPO and than RL-free methods such as DPO and RAFT. It tests $k = 2$ and $k = 4$ samples for each prompt.
- The ReMax paper (Li et al., 2023) gets its baseline by greedy decoding and computing the reward of that answer. It trains no value model. It reports about 46% less GPU memory than PPO for a 7B model (another section of the same paper says about 50%).

## 2. The value model (critic) and the TD error

### Setup

A **state** $s_t$ is the prompt plus the first $t$ tokens of the answer. The **value** $V(s_t)$ is the expected final reward from $s_t$ when the current policy writes the rest. In **actor-critic** methods, the policy is the **actor** and a learned value model is the **critic**.

The **per-token reward** $r_t$ is 0 for each token except the last one, which gets the full reward: for "sat cat sat", $r = [0,\ 0,\ 0.667]$. In RLHF, a KL penalty is also added at each token (see [Stable RL training](05-stable-training.md)).

### Steps

1. **Calculate the true values for "sat cat sat".** In the toy we can calculate them exactly: count the known correct tokens, add the probability of a correct token at each later position, divide by 3.

    $$V(s_0) = \tfrac{0.099 + 0.242 + 0.659}{3} = 0.333, \quad V(s_1) = \tfrac{0 + 0.242 + 0.659}{3} = 0.300, \quad V(s_2) = \tfrac{0 + 1 + 0.659}{3} = 0.553$$

    **Why:** the value is the usual reward from this exact state. It falls after the wrong `sat` and rises after the correct `cat`.

2. **Write the TD error.** The **TD error** (temporal-difference error) compares the value before a token with the reward plus the value after it:

    $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad V(s_3) = 0$$

    **Why:** $\delta_t$ asks "did this one token make the outlook better or worse than expected?" It needs only one step, not the full answer.

3. **Calculate the TD errors with the true values.**

    $$\delta_0 = 0 + 0.300 - 0.333 = -0.033, \quad \delta_1 = 0 + 0.553 - 0.300 = 0.253, \quad \delta_2 = 0.667 + 0 - 0.553 = 0.114$$

    **Why:** each token gets its own credit. The wrong first token gets a negative score. The correct `cat` (a rare token, probability 0.242) gets the most credit. In this toy, $\delta_t = (\text{correct} - p_{\text{target}})/3$.

4. **Repeat with an imperfect critic.** Assume a critic that sees only the position, not the tokens: $\hat V = [0.40,\ 0.45,\ 0.50]$. Then $\delta = [0.05,\ 0.05,\ 0.167]$.

    **Why:** a real critic is a learned guess. Its errors go straight into the TD errors. Here the wrong first token now gets a positive score.

<div class="key-point">

The critic turns one reward at the end into a score for each token. The quality of that score depends on the quality of the critic.

</div>

### From toy to real

- The value model is usually an LLM with a scalar output head. InstructGPT (Ouyang et al., 2022) uses a 6B value function for all policy sizes and initializes it from the 6B reward model.
- InstructGPT adds a per-token KL penalty from the SFT model to the reward. Huang et al. (2024) show the same shape: the reward-model score at the last (EOS) token, and per-token KL penalties at the other tokens.
- Huang et al. (2024) also initialize the value model from the reward model. After training it acts like a per-token reward model.

## 3. Generalized advantage estimation (GAE)

### Setup

**GAE** (generalized advantage estimation) mixes the TD errors that follow a token, with weights that decay by $\gamma\lambda$ at each step:

$$A_t = \sum_{l=0}^{T-1-t} (\gamma\lambda)^l\, \delta_{t+l}$$

With $\lambda = 0$, $A_t = \delta_t$ (one-step TD). With $\lambda = 1$, $A_t = \sum_l \gamma^l r_{t+l} - V(s_t)$: the Monte Carlo return minus the value.

### Steps

1. **Take $\lambda = 0$.** $A = \delta = [-0.033,\ 0.253,\ 0.114]$.

    **Why:** each token gets only its own TD error. This has the lowest variance, but it trusts the critic fully.

2. **Take $\lambda = 1$.** $A_t = R - V(s_t)$ with $R = 0.667$: $A = [0.333,\ 0.366,\ 0.114]$ (from unrounded values).

    **Why:** the critic now enters only as a baseline at the state itself. The wrong first token gets $+0.333$, because the whole answer was better than usual. The credit for `cat` leaks back to `sat`.

3. **Take $\lambda = 0.95$, from the last token backwards.** $A_t = \delta_t + \gamma\lambda A_{t+1}$:

    $$A_2 = 0.114, \quad A_1 = 0.253 + 0.95 \cdot 0.114 = 0.361, \quad A_0 = -0.033 + 0.95 \cdot 0.361 = 0.310$$

    **Why:** one backward pass gives all advantages. The result sits between the two cases.

4. **Measure bias and variance on the first token.** For each first token, average $A_0$ over the 9 possible continuations (weighted by their probability). The true credit gap between a first `the` and a first `sat` is $1/3 = 0.333$. The noise is the standard deviation of $A_0$ over the answers that start with `sat`.

    | Critic | λ | Gap `the` − `sat` (true 0.333) | Std after `sat` |
    |---|---|---|---|
    | exact | 0 | 0.333 | 0.000 |
    | exact | 0.95 | 0.333 | 0.197 |
    | exact | 1 | 0.333 | 0.213 |
    | position only | 0 | 0.000 | 0.000 |
    | position only | 0.95 | 0.301 | 0.192 |
    | position only | 1 | 0.333 | 0.213 |

    **Why:** with $\lambda = 1$ the gap is right for any critic (no bias), but the later tokens add noise (std 0.213). With $\lambda = 0$ there is no noise, but a bad critic hides the gap fully (bias). $\lambda$ sets this trade-off.

<figure class="plot">
<img src="/figures/ppo/gae-advantages.svg" alt="Bar chart of the per-token advantages of the answer sat cat sat with the exact critic. Lambda 0: -0.033, 0.253, 0.114. Lambda 0.95: 0.310, 0.361, 0.114. Lambda 1: 0.333, 0.366, 0.114.">
<figcaption>With λ = 0 the wrong first token gets a negative advantage. With λ near 1 it shares the credit of the whole answer.</figcaption>
</figure>

<div class="llm-widget" data-widget="gae"></div>

<div class="key-point">

GAE is one knob between "trust the critic" (λ = 0, biased if the critic is wrong) and "trust the sampled return" (λ = 1, unbiased but noisy).

</div>

### From toy to real

- The GAE paper (Schulman et al., 2015) defines the estimator as an exponentially weighted average of k-step estimators. It states that $\lambda = 1$ has high variance and $\lambda = 0$ induces bias unless the value function is exact.
- PPO on MuJoCo uses $\gamma = 0.99$ and $\lambda = 0.95$ (Schulman et al., 2017). The RLHF reproduction of Huang et al. (2024) uses $\gamma = 1.0$ and $\lambda = 0.95$.

## 4. The PPO loop

### Setup

**PPO** (Proximal Policy Optimization) collects answers with a frozen copy $\pi_{\text{old}}$, then runs several epochs of minibatch updates on the same data. The ratio $\rho_t = \pi_\theta(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$ measures how far each token has moved. The clipped objective, with $\varepsilon = 0.2$, is:

$$L^{\text{CLIP}} = \mathbb{E}_t\left[\min\big(\rho_t A_t,\ \text{clip}(\rho_t,\ 1 - \varepsilon,\ 1 + \varepsilon)\, A_t\big)\right]$$

The full objective subtracts a value loss and adds an entropy bonus: $L^{\text{CLIP}} - c_1 (V_\theta(s_t) - V_t^{\text{targ}})^2 + c_2 S$. The [preference alignment page](../scientist/05-preference-alignment.md#4-reinforcement-learning) shows the clip on one token. Here each token of an answer has its own ratio.

<figure class="diagram" data-src="ppo/rlhf-ppo-loop.svg">
<figcaption>One iteration of RLHF with PPO: four models, two of them trained.</figcaption>
</figure>

### Steps

1. **Collect rollouts with $\pi_{\text{old}}$ and compute advantages.** Use the group of 4 answers, the exact critic and GAE with $\lambda = 0.95$. For the target "the cat sat": $\delta = [0.300,\ 0.253,\ 0.114]$ and $A = [0.643,\ 0.361,\ 0.114]$.

    **Why:** the advantages are computed once, with the old policy and the old critic. They stay fixed during all the epochs of this iteration.

2. **Do one gradient step.** For a tabular policy, the gradient of $A_t \log \pi(a_t)$ for position $t$ is $A_t(\text{one-hot}(a_t) - p_t)$. Take the mean over the 12 tokens of the group and assume a step size of 3. The new probabilities of (`the`, `cat`, `sat`) are $[0.132, 0.243, 0.625]$ at position 0, $[0.101, 0.288, 0.611]$ at position 1 and $[0.096, 0.232, 0.673]$ at position 2.

    **Why:** at the first step all ratios are 1, so the clip has no effect yet. Tokens with a high advantage gain probability.

3. **Calculate the per-token ratios and apply the clip.**

    | Answer | Token | Advantage | Old p | New p | Ratio | Ratio × A | Clipped term |
    |---|---|---|---|---|---|---|---|
    | the cat sat | the | 0.643 | 0.099 | 0.132 | **1.338** | 0.861 | **0.772** |
    | the cat sat | cat | 0.361 | 0.242 | 0.288 | 1.188 | 0.428 | 0.428 |
    | the cat sat | sat | 0.114 | 0.659 | 0.673 | 1.021 | 0.116 | 0.116 |
    | sat cat sat | sat | 0.310 | 0.659 | 0.625 | 0.948 | 0.294 | 0.294 |
    | sat cat sat | cat | 0.361 | 0.242 | 0.288 | 1.188 | 0.428 | 0.428 |
    | sat cat sat | sat | 0.114 | 0.659 | 0.673 | 1.021 | 0.116 | 0.116 |

    The token `the` has $\rho = 1.338 > 1.2$, so $\min(0.861,\ 1.2 \cdot 0.643) = 0.772$.

    **Why:** above the clip, the term is a constant, so this token gives no gradient in the next epoch. The other tokens can still move. In "sat cat sat", the wrong `sat` has a positive advantage (0.310), but its ratio fell to 0.948, because other answers pushed `the` up more.

4. **Train the critic with a value loss.** The target is the return (with $\gamma = 1$ and one reward at the end, 0.667 at each token). For the position-only critic the errors are $[0.267,\ 0.217,\ 0.167]$ and the mean squared error is 0.049.

    **Why:** a better critic gives better TD errors next time. Many implementations use $A_t + V(s_t)$ as the target, which equals the return when $\lambda = 1$.

5. **Repeat for K epochs, then copy the weights.** After K epochs over the minibatches, set $\pi_{\text{old}} \leftarrow \pi_\theta$ and collect new answers.

    **Why:** generation is expensive. The clip makes several steps on one batch safe.

<div class="key-point">

PPO = advantages from a critic and GAE, plus a per-token clip that stops each token once it has moved more than ε. RLHF-PPO keeps four models in play: policy, reference, reward model and value model.

</div>

### From toy to real

- The PPO paper (Schulman et al., 2017) runs $N$ actors for $T$ steps, then optimizes for $K$ epochs with minibatch size $M \le NT$. On MuJoCo it uses $T = 2048$, 10 epochs, minibatch 64 and $\varepsilon = 0.2$.
- The RLHF reproduction of Huang et al. (2024) uses batch size 512, $K = 4$ update epochs, 1 minibatch, $\varepsilon = 0.2$, value clipping 0.2, $c_1 = 0.1$, KL coefficient $\beta = 0.05$, learning rate $3 \times 10^{-6}$ and sampling temperature 0.7.
- The four models: the RLOO paper says PPO "typically requires loading up to 4 models simultaneously". Huang et al. (2024) say 3 to 4. The policy and the value model are trained. The reference and the reward model are frozen.
- The KL penalty to the reference model keeps the policy near the start model (see [Stable RL training](05-stable-training.md)). GRPO drops the value model and uses the group as the baseline (see [GRPO and its successors](04-grpo-family.md)).

## Minimal code

This code gives the GAE and clip numbers of this page.

```python
import numpy as np

def gae(deltas, gamma, lam):
    adv, running = np.zeros_like(deltas), 0.0
    for t in reversed(range(len(deltas))):
        running = deltas[t] + gamma * lam * running
        adv[t] = running
    return adv

values = np.array([0.333, 0.300, 0.553])        # exact V(s_t) for "sat cat sat"
rewards = np.array([0.0, 0.0, 0.667])           # reward only at the last token
deltas = rewards + np.append(values[1:], 0.0) - values   # [-0.033, 0.253, 0.114]
adv = gae(deltas, gamma=1.0, lam=0.95)          # [0.310, 0.361, 0.114]

ratio, a, eps = 1.338, 0.643, 0.2               # token "the" of "the cat sat"
term = min(ratio * a, np.clip(ratio, 1 - eps, 1 + eps) * a)   # 0.772
```

## Comparison

| Method | Baseline | Extra trained model | Credit per token? | Toy advantage of "the cat sat" |
|---|---|---|---|---|
| REINFORCE, group mean | mean of the group | none | no | 0.333 |
| RLOO | mean of the other G − 1 | none | no | 0.444 |
| ReMax | reward of the greedy answer | none (1 extra decode) | no | 0.667 |
| GRPO | group mean, divided by std | none | no | 1.414 |
| PPO with GAE (λ = 0.95) | value model | value model | yes | 0.643, 0.361, 0.114 |

## Checks and common misconceptions

- **"The value model judges the answer."** No. The reward model judges the answer. The value model predicts the usual reward of the current policy from a state. Check: the value loss must go down early in training.
- **"λ = 1 is always the safe choice because it has no bias."** It has no critic bias, but each token also carries the noise of all later tokens. In the toy the std is 0.213 with λ = 1 and 0 with λ = 0. For long answers this noise grows.
- **"The clip limits how far the policy moves."** It limits the incentive, not the move. In the toy, `the` moved to $\rho = 1.338$ in one step, past the clip. Check: log the fraction of clipped tokens and the KL to $\pi_{\text{old}}$.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. Each statement about a real system was read in the linked source on 25 November 2026.

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), Schulman et al., 2017 (clipped objective, ε = 0.2, value and entropy terms, truncated GAE, Algorithm 1, MuJoCo hyperparameters).
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), Schulman et al., 2015 (GAE formula, the cases λ = 0 and λ = 1, bias and variance).
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), Ouyang et al., 2022 (InstructGPT: per-token KL penalty, 6B value function initialized from the 6B reward model).
- [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740), Ahmadian et al., 2024 (RLOO estimator, k = 2 and 4, results against PPO, "up to 4 models").
- [ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models](https://arxiv.org/abs/2310.10505), Li et al., 2023 (greedy baseline, no value model, about 46% memory saving for a 7B model).
- [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031), Huang et al., 2024 (PPO hyperparameter table, value model initialized from the reward model, reward at the EOS token, 3 to 4 models in memory).
