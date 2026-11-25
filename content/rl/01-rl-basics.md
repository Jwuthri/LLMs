# 1. RL basics

## Core idea

In reinforcement learning (RL), a model writes an answer, a checker gives the answer a score, and the model changes so that answers with a high score become more probable.
Nobody shows the model the correct answer. The score is the only signal.
This lesson builds the simplest method that does this, REINFORCE, and shows why a baseline makes it much less noisy.

## The example

This track uses one small task. It extends the [toy example](../toy-example.md) of the course.

- **Prompt:** "Copy: the cat sat". The **target** answer is "the cat sat".
- **Policy:** the model writes 3 tokens, one at each position, from the vocabulary `the`, `cat`, `sat`. Each position has its own logits. All start at the toy logits [0.1, 1.0, 2.0], so each position starts with the probabilities [0.099, 0.242, 0.659].
- **Group:** assume that the model wrote these 4 answers to the prompt.

| Answer | Token log-probs | Sequence log-prob | Sparse | Partial | Process (per token) |
|---|---|---|---|---|---|
| the cat sat | −2.317, −1.417, −0.417 | −4.151 | 1 | 1.000 | 1/3, 1/3, 1/3 |
| sat cat sat | −0.417, −1.417, −0.417 | −2.251 | 0 | 0.667 | 0, 1/3, 1/3 |
| the sat sat | −2.317, −0.417, −0.417 | −3.151 | 0 | 0.667 | 1/3, 0, 1/3 |
| sat sat sat | −0.417, −0.417, −0.417 | −1.251 | 0 | 0.333 | 0, 0, 1/3 |

The three reward types:

- **Sparse:** 1 if the full answer is correct, else 0.
- **Partial:** the fraction of correct tokens (0, 1/3, 2/3 or 1).
- **Process:** 1/3 for each correct token, given at that token. The sum is the partial reward.

This lesson uses the sparse and the partial rewards. [Lesson 2](02-rewards.md) compares all three.

## 1. Text generation as RL

### Setup

RL describes an **agent** that acts in an **environment** and gets **rewards**.
Text generation fits this frame. The agent is the language model. Its "world" is the text so far.

<figure class="diagram" data-src="rl-basics/loop.svg">
<figcaption>The RL loop on the toy: the policy writes an answer, a checker scores it, and the update uses only that score.</figcaption>
</figure>

### Steps

1. **Name the state.** The **state** $s_t$ is the prompt plus the tokens written so far. Before the first token, $s_0$ = "Copy: the cat sat". After one token, $s_1$ = "Copy: the cat sat" + `the`.

    **Why:** the model decides each token from what it can see. The state is exactly that input.

2. **Name the action and the policy.** The **action** $a_t$ is the next token. The **policy** $\pi_\theta(a_t \mid s_t)$ is the probability that the model gives to each token. In the toy, $\pi$ at position $t$ is the softmax of the logits $\theta_t$.

    **Why:** RL changes the parameters $\theta$ of the policy. So the policy must be a function of $\theta$ that we can differentiate.

3. **Write the trajectory and its probability.** One full answer is a **trajectory** (or **episode**): $y = (a_0, a_1, a_2)$. Its probability is the product of the token probabilities:

    $$\pi(\text{the cat sat}) = 0.099 \times 0.242 \times 0.659 = 0.0157, \qquad \log \pi = -2.317 - 1.417 - 0.417 = -4.151$$

    **Why:** the update in section 3 needs $\log \pi(y)$. A product of probabilities becomes a sum of log-probabilities, which is easy to differentiate.

4. **Give the reward at the end.** The checker reads the full answer. For "sat cat sat" the partial reward is 0.667. Each token before the last one gets a reward of 0.

    **Why:** most checks can only run on a finished answer.

5. **Compare with supervised fine-tuning.** **Supervised fine-tuning (SFT)** minimizes $-\log \pi(\text{target}) = 4.151$. It must know the target. RL needs only the score of an answer that the model wrote itself.

    **Why:** for many tasks we can check an answer but cannot write the best answer. RL can learn from the check alone.

<div class="key-point">

Generation is a sequence of actions. The state is the text so far, the action is the next token, the policy is the model, and the reward usually comes once, at the end.

</div>

### From toy to real

- In the toy, the logits of a position do not depend on the earlier tokens. A real LLM reads the full state at each step, so each token changes the probabilities of the next ones.
- The "environment" is simple: it appends the token. In agentic RL, a tool also writes text into the state ([lesson 7](07-agentic-rl.md)).
- An episode ends at the end-of-sequence token or at a maximum length.
- The checker can be a rule in code or a learned reward model ([lesson 2](02-rewards.md)).

## 2. Return and objective

### Setup

We need one number for one trajectory, and one number for the policy.
The first is the **return**. The second is the **objective**.

### Steps

1. **Write the reward of each step.** For the target answer with the partial reward: $r = [0, 0, 1]$.

    **Why:** the general RL frame allows a reward at each step. An outcome reward is the special case where only the last step is not 0.

2. **Sum the rewards into the return.** The **return** is the discounted sum, with a **discount** $\gamma$ between 0 and 1:

    $$R = \sum_{t=0}^{T-1} \gamma^{t}\, r_t$$

    With $\gamma = 1$: $R = 0 + 0 + 1 = 1$. With $\gamma = 0.9$: $R = 0.9^2 \times 1 = 0.81$.

    **Why:** a discount below 1 makes a late reward count less. In a game without an end, it keeps the sum finite. A text episode always ends, so $\gamma = 1$ is allowed. This toy uses $\gamma = 1$, and the TRL PPO trainer also uses $\gamma = 1$ by default.

3. **Define the objective.** The **objective** is the expected return over the answers that the policy writes:

    $$J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[R(y)] = \sum_{y} \pi_\theta(y)\, R(y)$$

    **Why:** one answer can be lucky. The policy is good when its *average* answer is good.

4. **Compute J for the sparse reward.** The toy has $3^3 = 27$ possible answers. Their probabilities add up to 1. Only the target gets a sparse reward of 1, so $J_\text{sparse} = \pi(\text{target}) = 0.0157$.

    **Why:** with a sparse reward, the objective is the success rate. Here the model succeeds about once in 64 tries.

5. **Compute J for the partial reward.** Group the 27 answers by the number of correct tokens:

    | Correct tokens | 0 | 1 | 2 | 3 |
    |---|---|---|---|---|
    | Probability | 0.233 | 0.550 | 0.201 | 0.016 |
    | Partial reward | 0 | 1/3 | 2/3 | 1 |

    $J_\text{partial} = 0.550/3 + 2 \times 0.201/3 + 0.016 = 0.333$. A short way gives the same value: the mean of the target-token probabilities, $(0.099 + 0.242 + 0.659)/3 = 0.333$.

    **Why:** the partial reward gives credit to the many answers with 1 or 2 correct tokens. So $J$ is 21 times larger than with the sparse reward, and it changes when any one position improves.

<div class="key-point">

The return scores one answer. The objective $J = \mathbb{E}[R]$ scores the policy. RL tries to make $J$ larger.

</div>

### From toy to real

- A real model cannot list all answers. With a vocabulary of $V$ tokens and answers of 100 tokens, there are $V^{100}$ answers, and a real $V$ is in the tens of thousands or more ([architecture lesson](../scientist/01-architecture.md#2-tokenization)).
- So training estimates $J$ from samples. The 4 answers of the group are such an estimate: their mean partial reward is 0.667, twice the true 0.333. A group of 4 can be far from the truth, and this assumed group contains the rare correct answer.

## 3. REINFORCE

### Setup

We want $\nabla_\theta J$. The reward is a score from a checker, and we cannot differentiate a checker.
The **log-derivative trick** moves the gradient onto the policy. It uses $\nabla \pi = \pi \nabla \log \pi$:

$$\nabla_\theta J = \sum_y R(y)\, \nabla_\theta \pi_\theta(y) = \sum_y \pi_\theta(y)\, R(y)\, \nabla_\theta \log \pi_\theta(y) = \mathbb{E}_{y \sim \pi}\big[R(y)\, \nabla_\theta \log \pi_\theta(y)\big]$$

An expectation can be estimated with samples. With $N$ sampled answers, the **REINFORCE** estimate is:

$$\hat g = \frac{1}{N} \sum_{i=1}^{N} R(y_i)\, \nabla_\theta \log \pi_\theta(y_i)$$

The vector $\nabla_\theta \log \pi_\theta(y)$ is the **score function**. For one softmax position, the gradient of $\log p_k$ for the logits is $\text{onehot}(k) - p$. The log-probability of an answer is a sum over the positions, so row $t$ of the score is $\text{onehot}(y_t) - p_t$.

### Steps

1. **Compute the score of each answer.** For the target "the cat sat", with $p = [0.099, 0.242, 0.659]$ at each position:

    $$\nabla_\theta \log \pi(\text{the cat sat}) = \begin{bmatrix} 0.901 & -0.242 & -0.659 \\ -0.099 & 0.758 & -0.659 \\ -0.099 & -0.242 & 0.341 \end{bmatrix}$$

    Rows are positions. Columns are `the`, `cat`, `sat`.

    **Why:** the score is the direction that makes this answer more probable. It raises the logit of each chosen token and lowers the others.

2. **Weight each score by its reward and average.** Use the group as the 4 samples and the partial reward. At position 3, all 4 answers chose `sat`, and the rewards add up to 2.667, so row 3 is $\tfrac{2.667}{4}\,[-0.099, -0.242, 0.341] = [-0.066, -0.162, 0.227]$. All rows:

    $$\hat g = \begin{bmatrix} 0.351 & -0.162 & -0.189 \\ -0.066 & 0.255 & -0.189 \\ -0.066 & -0.162 & 0.227 \end{bmatrix}$$

    **Why:** an answer with a high reward pulls the policy toward itself more. Position 1 gains on `the` because the two answers that start with `the` have the higher rewards (1 and 0.667).

3. **Take one step.** Assume a step size $\alpha = 1$: $\theta \leftarrow \theta + \alpha \hat g$. The probabilities of the target tokens become 0.157 (`the` at position 1), 0.329 (`cat` at position 2) and 0.735 (`sat` at position 3).

    | | Before | After one step |
    |---|---|---|
    | P(target) = J sparse | 0.0157 | 0.038 |
    | J partial | 0.333 | 0.407 |

    **Why:** this is gradient *ascent*: we add the gradient, because we want a larger $J$. The correct answer became 2.4 times more probable, and no person wrote it down.

4. **Compare with the exact gradient.** The toy can compute $\nabla J$ exactly from the 27 answers. Row 1 of the exact gradient is $[0.030, -0.008, -0.022]$, more than 10 times smaller than row 1 of $\hat g$.

    **Why:** REINFORCE is correct *on average* over random samples. The assumed group is not a fair sample: it contains the target, which the policy writes only 1.6% of the time. One small batch can give a large and noisy step.

<div class="llm-widget" data-widget="reinforce"></div>

<div class="key-point">

REINFORCE turns "make the average reward larger" into "raise the log-probability of each sampled answer, weighted by its reward". It needs only samples and their scores, not a gradient of the reward.

</div>

### From toy to real

- In code, REINFORCE is the loss $-\frac{1}{N}\sum_i R(y_i) \log \pi_\theta(y_i)$. Automatic differentiation computes the score. The reward is a constant in the loss.
- Williams (1992) defined the class of REINFORCE algorithms and showed that they do stochastic gradient ascent on the expected reward. The gradient-bandit update in Sutton and Barto is the same update for one softmax.
- In a real LLM, $\log \pi(y)$ is a sum over hundreds or thousands of tokens. Sutton et al. (1999) showed that the policy gradient stays valid when the policy is any differentiable function approximator, such as a neural network.
- Real step sizes are much smaller than 1. For example, the default learning rate of the TRL PPO trainer is $3 \times 10^{-6}$.

## 4. Baselines and variance

### Setup

Subtract a number $b$, the **baseline**, from each reward:

$$\hat g_b = \frac{1}{N}\sum_{i=1}^{N} \big(R(y_i) - b\big)\, \nabla_\theta \log \pi_\theta(y_i)$$

The term $R - b$ is a simple **advantage**: how much better the answer is than "usual".
The baseline must not depend on the sampled answer $y_i$.

### Steps

1. **Check that the baseline adds no bias.** The mean of the score is 0:

    $$\mathbb{E}_{y \sim \pi}\big[\nabla_\theta \log \pi_\theta(y)\big] = \sum_y \nabla_\theta \pi_\theta(y) = \nabla_\theta \sum_y \pi_\theta(y) = \nabla_\theta 1 = 0$$

    The script sums the score over the 27 answers, weighted by $\pi$. All 9 entries are 0.000. So the term $b \cdot \nabla \log \pi$ has a mean of 0, and $\mathbb{E}[\hat g_b] = \nabla J$ for each $b$.

    **Why:** we may only change the estimate if the average stays the true gradient. Otherwise we would climb the wrong hill.

2. **Measure the variance without a baseline.** Take one sample ($N = 1$). Its **variance** is the expected squared distance of $\hat g$ from $\nabla J$, summed over the 9 entries. The toy computes it exactly from the 27 answers. With the partial reward and $b = 0$, the variance is 0.253.

    **Why:** the variance tells how noisy one update is. With $b = 0$, every reward here is 0 or more, so every sampled answer gets pushed up, also a bad one.

3. **Measure the variance with $b = \mathbb{E}[R]$.** With $b = 0.333$, the variance is 0.087: 2.9 times smaller. An answer with fewer correct tokens than usual now gets a negative weight, and it is pushed down.

    **Why:** the baseline removes the part of the reward that every answer gets. What is left is the difference between answers, which is the useful signal.

    The best baseline is $b^* = \mathbb{E}[R \lVert s \rVert^2] / \mathbb{E}[\lVert s \rVert^2]$, where $s$ is the score. In this toy, $b^* = 0.333 = \mathbb{E}[R]$ exactly. This is a property of the toy (each token is the target at exactly one position). In general, $b^*$ is near $\mathbb{E}[R]$ but not equal to it.

<figure class="plot">
<img src="/figures/rl-basics/variance-vs-baseline.svg" alt="Variance of the one-sample REINFORCE gradient against the baseline b, for the partial reward. It is a parabola: 0.253 at b = 0 and a minimum of 0.087 at b = 0.333.">
<figcaption>The variance is a parabola in b. Its minimum in this toy is at b = E[R] = 0.333.</figcaption>
</figure>

4. **Try the sparse reward.** With the sparse reward, the variance goes from 0.0389 ($b = 0$) to 0.0380 ($b = 0.0157$). The baseline barely helps.

    **Why:** the policy writes a wrong answer 98.4% of the time, and a wrong answer gets 0 either way. A baseline cannot create a signal that is not there. A denser reward helps more ([lesson 2](02-rewards.md)).

5. **Train with and without the baseline.** Assume 200 runs (seeds 0 to 199), 30 steps each, 4 sampled answers for each step, $\alpha = 1$, partial reward. The baseline is the exact $\mathbb{E}[R]$ of the current policy. After 30 steps:

    | | No baseline | b = E[R] |
    |---|---|---|
    | Mean J partial | 0.747 | 0.790 |
    | Standard deviation over runs | 0.114 | 0.058 |
    | 10th percentile | 0.616 | 0.701 |

    **Why:** a lower variance means fewer wasted steps. The mean is higher, and the spread over runs halves.

<figure class="plot">
<img src="/figures/rl-basics/training-runs.svg" alt="Mean expected partial reward over 30 REINFORCE steps for 200 runs. Without a baseline it reaches 0.747; with the baseline b = E[R] it reaches 0.790. The 10 to 90 percent band is narrower with the baseline.">
<figcaption>Mean of 200 runs, with the band from the 10th to the 90th percentile. The baseline makes the band narrower.</figcaption>
</figure>

6. **Use the group as the baseline.** A real system does not know $\mathbb{E}[R]$. A cheap choice is the group mean. For the 4 answers, $b = 0.667$, so $R - b = [0.333, 0, 0, -0.333]$. The new gradient has rows $[0.083, 0, -0.083]$, $[0, 0.083, -0.083]$ and $[0, 0, 0]$. After one step, P(target) = 0.020 and $J_\text{partial}$ = 0.348.

    **Why:** row 3 is now 0, because all 4 answers agree at position 3, so that position has nothing to learn from this group. This one step is smaller than the step with $b = 0$: a baseline removes noise, it does not make one lucky step larger. A group mean that includes the answer itself multiplies the expected gradient by $(N-1)/N$. [Lesson 3](03-ppo.md) fixes this with a leave-one-out mean (RLOO), and [lesson 4](04-grpo-family.md) also divides by the standard deviation (GRPO).

<div class="key-point">

A baseline changes the noise, not the target. Any $b$ that does not depend on the sampled answer keeps the gradient unbiased, and $b$ near $\mathbb{E}[R]$ makes it much less noisy.

</div>

### From toy to real

- Real systems estimate the baseline: with a learned **value model** $V(s)$ in PPO ([lesson 3](03-ppo.md)), or with the other answers to the same prompt in RLOO and GRPO ([lesson 4](04-grpo-family.md)).
- $\mathbb{E}[R]$ is different for each prompt, so a group of answers to *one* prompt is a natural baseline.
- The [preference alignment lesson](../scientist/05-preference-alignment.md#4-reinforcement-learning) shows the same update on a one-token toy, with the GRPO advantage.

## Minimal code

This code gives the REINFORCE numbers of section 3 and the group-mean step of section 4.

```python
import numpy as np

vocab = ["the", "cat", "sat"]
logits = np.tile([0.1, 1.0, 2.0], (3, 1))          # one row for each position
group = [("the", "cat", "sat"), ("sat", "cat", "sat"), ("the", "sat", "sat"), ("sat", "sat", "sat")]
target = ("the", "cat", "sat")

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def score(answer, logits):                          # grad of log pi(y): onehot(y_t) - p_t
    return np.eye(3)[[vocab.index(a) for a in answer]] - softmax(logits)

rewards = [np.mean([a == t for a, t in zip(y, target)]) for y in group]   # partial reward
for b in (0.0, np.mean(rewards)):                   # no baseline, then the group mean
    g = sum((r - b) * score(y, logits) for y, r in zip(group, rewards)) / len(group)
    new = softmax(logits + 1.0 * g)                 # step size 1
    p_target = new[0, 0] * new[1, 1] * new[2, 2]
    print(b, np.round(g, 3), round(p_target, 3))    # b = 0: 0.038; b = 0.667: 0.020
```

## Comparison

| Estimate | Unbiased? | Variance (toy, partial, one sample) | Needs | Used in |
|---|---|---|---|---|
| Exact gradient (sum over all answers) | Yes, exact | 0 | All answers listed | Only tiny toys |
| REINFORCE, b = 0 | Yes | 0.253 | Samples and rewards | Williams (1992) |
| REINFORCE, b = E[R] | Yes | 0.087 | The true mean reward | Toy only (an oracle) |
| REINFORCE, b = group mean (with the answer itself) | Scaled by (N − 1)/N | Not computed here | A group for each prompt | Fixed by RLOO (lesson 3), extended by GRPO (lesson 4) |
| REINFORCE, b = V(s) learned | Yes, if V uses only the state | Low if V is good | A second model | PPO (lesson 3) |

## Checks and common misconceptions

- **"RL needs a reward that you can differentiate."** It does not. The log-derivative trick puts the gradient on $\log \pi$. The reward can be a unit test, a string match or a human score. Check: in your loss code, the reward must be a constant with no gradient (for example, detached).
- **"A baseline changes what the model learns."** In expectation it does not, because $\mathbb{E}[\nabla \log \pi] = 0$. It changes only the noise. The rule: the baseline must not depend on the answer that it is subtracted from. A baseline that uses that answer (for example, a group mean that includes it) changes the gradient.
- **"The REINFORCE loss going down means the model got better."** The loss $-\frac{1}{N}\sum R_i \log \pi(y_i)$ depends on which answers were sampled in that batch. It is not a score of the policy. Track the mean reward on fresh samples instead.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. Each statement about a real system was read in the linked source on 25 November 2026.

- [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://doi.org/10.1007/BF00992696), R. J. Williams, Machine Learning 8, 1992 (the class of REINFORCE algorithms, as Sutton and Barto credit it to this paper; title, author, year and journal checked in OpenAlex).
- [Reinforcement Learning: An Introduction, 2nd edition (2015 draft)](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf), R. S. Sutton and A. G. Barto (section 3.4: the return with $\gamma = 1$ when all episodes end; section 2.7: the gradient-bandit update for a softmax, the derivative $\pi(b)(\mathbf{1}_{a=b} - \pi(a))$, the baseline that must not depend on the action, does not change the expected update and changes the variance; the credit of the REINFORCE algorithms to Williams). The final edition is at [incompleteideas.net/book](http://incompleteideas.net/book/the-book-2nd.html).
- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html), Sutton, McAllester, Singh and Mansour, NeurIPS 1999 (policy gradient with a differentiable function approximator; REINFORCE and actor-critic fit this frame).
- [PPO Trainer](https://huggingface.co/docs/trl/main/en/ppo_trainer), Hugging Face TRL documentation (`PPOConfig` defaults: `gamma = 1.0`, `learning_rate = 3e-6`).
