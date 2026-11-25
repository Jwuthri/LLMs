# 4. GRPO and its successors

## Core idea

GRPO trains a model from a score. It compares each answer with the other answers to the same prompt, so it needs no value model.
Its successors keep this idea. Each one changes one part of the GRPO loss: how the advantage is scaled, how the tokens are averaged, where the clip is, or what the ratio measures.
This page puts all five methods on the same 4 answers, so that you can see each difference as a number.

## The example

This page uses the shared toy of the track: the prompt "Copy: the cat sat", a tabular policy that starts at [0.099, 0.242, 0.659] at each position, and the assumed group of 4 answers. The reward is the sparse checker reward (1 if the full answer is correct).

| Answer | Reward | Advantage | Length |
|---|---|---|---|
| the cat sat | 1 | +1.732 | 3 |
| sat cat sat | 0 | −0.577 | 3 |
| the sat sat | 0 | −0.577 | 3 |
| sat sat sat | 0 | −0.577 | 3 |

Each section adds one assumed value, and says "assume": a step size of 1.8 for one update (sections 1 and 4), a longer wrong answer and two extra prompts (section 2), and a group where all answers are wrong (section 3).

<figure class="diagram" data-src="grpo-family/loss-anatomy.svg">
<figcaption>The GRPO loss, part by part, and what each successor changes.</figcaption>
</figure>

## 1. GRPO

### Setup

**GRPO** (Group Relative Policy Optimization) comes from DeepSeekMath. For a prompt $q$, it samples a group of $G$ answers $o_1, \dots, o_G$ from the old policy $\pi_{old}$ and maximizes:

$$J_{GRPO} = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\Big(r_{i,t}\,\hat{A}_{i,t},\ \operatorname{clip}(r_{i,t},\,1-\varepsilon,\,1+\varepsilon)\,\hat{A}_{i,t}\Big) - \beta\,D_{KL}$$

Here $r_{i,t} = \pi_\theta(o_{i,t}) / \pi_{old}(o_{i,t})$ is the **ratio** of token $t$: how much more probable the token is now than when it was sampled. The loss is $-J_{GRPO}$.

### Steps

1. **Compute the group advantage.** The rewards are [1, 0, 0, 0]. The mean is 0.25 and the standard deviation is 0.433.

    $$\hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)} = [1.732,\ -0.577,\ -0.577,\ -0.577]$$

    **Why:** the group mean is the baseline. It replaces the value model of PPO ([lesson 3](03-ppo.md)), which saves one full model in memory.

2. **Give every token of an answer the same advantage.** The three tokens of "the cat sat" all get +1.732. The three tokens of "the sat sat" all get −0.577.

    **Why:** the checker scores the full answer. GRPO has no information about which token helped, so all tokens share the credit.

3. **Take one on-policy step.** At the first step on a new group, $\pi_\theta = \pi_{old}$, so all ratios are 1 and the clip does nothing. The gradient on the logits of position 1 is [0.096, 0, −0.096], on position 2 it is [0, 0.096, −0.096], and on position 3 it is 0. Assume a step size of 1.8. Position 1 becomes [0.128, 0.265, 0.606].

    **Why:** methods of the PPO family reuse one group for several gradient steps. The ratio only matters from the second step. Position 3 does not move: all 4 answers write `sat` there, and the advantages add up to 0.

4. **Compute the per-token ratios of one answer.** For "the cat sat", the ratios are [1.301, 1.264, 1.000]. With $\varepsilon = 0.2$, the clipped terms are:

    $$\min(1.301 \cdot 1.732,\ 1.2 \cdot 1.732) = 2.078, \quad 2.078, \quad 1.732$$

    The **gradient weight** is the number that multiplies $\nabla \log \pi$ of the token. It is $r A$ on the unclipped side and 0 on the clipped side: [0, 0, 1.732].

    **Why:** the samples came from $\pi_{old}$. When a token has already moved far, the clip removes its gradient, so one group cannot push the policy too far.

5. **Average the terms.** First take the mean over the tokens of each answer: 1.963, −0.613, −0.615, −0.542. Then take the mean over the answers: $J = 0.048$ (it was 0 before the step). GRPO also subtracts $\beta$ times a KL estimate to a frozen reference model. [Lesson 5](05-stable-training.md) explains this term.

    **Why:** each answer gets the same total weight $1/G$, whatever its length. Section 2 shows why this is a problem.

<div class="key-point">

GRPO is PPO with a group baseline. One advantage for each answer, one ratio and one clip for each token, then a mean over tokens and a mean over answers.

</div>

### From toy to real

- DeepSeekMath samples 64 answers for each question, with a KL coefficient $\beta = 0.04$ and a policy learning rate of 1e-6. Its KL estimate for each token is $\frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1$.
- The paper calls $\varepsilon$ a hyperparameter and gives no value. The [preference alignment page](../scientist/05-preference-alignment.md#4-reinforcement-learning) uses 0.2 on a one-token toy, as this page does.

## 2. Dr. GRPO

### Setup

Liu et al. found two biases in the GRPO loss and named their unbiased fix **Dr. GRPO**.
To see the first bias, assume the 4th answer is longer: `sat` 6 times. It is still wrong, so its reward is 0 and its advantage is still −0.577. The lengths are [3, 3, 3, 6].

### Steps

1. **Compute the GRPO weight of each token.** Each token of answer $i$ gets $\hat{A}_i / (G\,|o_i|)$: [0.144, −0.048, −0.048, −0.024]. The sum over each answer is [0.433, −0.144, −0.144, −0.144].

    **Why:** this weight says how hard the loss pushes each token up or down.

2. **See the length bias.** Each token of the long wrong answer gets half the penalty of a token of a short wrong answer (−0.024 against −0.048). The full long answer gets the same penalty as a short one, but it spreads over twice as many tokens. For correct answers it is the opposite: a short correct answer gets a larger reward for each token.

    **Why:** the gradient pushes toward short correct answers and long wrong answers. The paper reports that this bias makes answers longer during training, especially wrong answers.

3. **See the difficulty bias.** Assume two more prompts with 8 answers each. On an easy prompt, 7 of 8 answers are correct: std 0.331, so the scale $1/\text{std}$ is 3.024 and the one wrong answer gets $\hat{A} = -2.646$. On a medium prompt, 4 of 8 are correct: std 0.5, scale 2.0, $\hat{A} = \pm 1$. Our toy prompt has a scale of 2.309.

    **Why:** the division by the std gives the reward differences of the easy prompt 1.51 times more weight (3.024 / 2.0). Prompts where almost all rewards are equal (too easy or too hard) get the largest weights.

4. **Remove both normalizations.** Dr. GRPO uses $\tilde{A}_i = r_i - \text{mean}(r)$ and divides the sum of all tokens by a constant instead of by $|o_i|$:

    $$J_{Dr.GRPO} = \frac{1}{G}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\Big(r_{i,t}\,\tilde{A}_{i},\ \operatorname{clip}(r_{i,t},\,1-\varepsilon,\,1+\varepsilon)\,\tilde{A}_{i}\Big)$$

    The toy advantages become [0.75, −0.25, −0.25, −0.25]. Assume the constant is MAX_TOKENS = 6. Each token gets $\tilde{A}_i / (G \cdot 6)$: [0.031, −0.010, −0.010, −0.010]. The sums are [0.094, −0.031, −0.031, −0.063].

    **Why:** now every token of a wrong answer gets the same penalty. The 6-token wrong answer gets twice the total penalty of a 3-token one, so length no longer hides an error.

<div class="key-point">

Two divisions in GRPO change the weights without a reason: $1/|o_i|$ favors long wrong answers, and $1/\text{std}$ favors prompts with almost equal rewards. Dr. GRPO removes both.

</div>

### From toy to real

- In the code of the paper, the per-answer division by the mask sum becomes a division by MAX_TOKENS, "a global constant during the entire training", for example the generation budget.
- A constant normalizer only rescales the gradient. The learning rate can absorb it.

## 3. DAPO

### Setup

**DAPO** (Decoupled Clip and Dynamic sAmpling Policy Optimization, Yu et al.) adds four techniques to GRPO: clip-higher, dynamic sampling, a token-level loss and overlong reward shaping. It also removes the KL term.

### Steps

1. **Raise the upper clip (clip-higher).** DAPO uses two ranges: $\operatorname{clip}(r, 1-\varepsilon_{low}, 1+\varepsilon_{high})$ with $\varepsilon_{low} = 0.2$ and $\varepsilon_{high} = 0.28$. In the toy, `the` has $p_{old} = 0.099$. With a positive advantage, it can rise to 0.118 with 0.2, or to 0.126 with 0.28, before the clip stops its gradient. The likely token `sat` (0.659) can rise to 0.791 or 0.844. In the update of section 1, `cat` has $r = 1.264$: GRPO clips it, DAPO keeps its gradient weight of 2.189.

    **Why:** a symmetric clip gives an unlikely token an absolute room of only 0.020, against 0.132 for a likely token. The paper links this to **entropy collapse**: the policy stops trying rare tokens. DAPO keeps $\varepsilon_{low}$ at 0.2 (`the` can drop only to 0.079), because a larger value can push tokens to probability 0.

2. **Drop groups with equal rewards (dynamic sampling).** Assume a second group where all 4 answers are `sat sat sat`. All rewards are 0, the std is 0, and libraries add a small number to the std, so all advantages are 0. If the batch keeps this group, the toy group's tokens get a weight of 1/24 = 0.042 instead of 1/12 = 0.083 under a mean over all tokens. DAPO samples more prompts until every group has an accuracy strictly between 0 and 1.

    **Why:** a group with equal rewards gives no gradient but still takes compute and dilutes the batch. [Lesson 2](02-rewards.md) shows why sparse rewards give many such groups.

3. **Average over all tokens of the batch (token-level loss).** DAPO divides the sum of all token terms by $\sum_i |o_i|$, the number of tokens in the group. With the lengths [3, 3, 3, 6], each token gets $\hat{A}_i / 15$: [0.115, −0.038, −0.038, −0.038]. The share of each answer in the loss is [0.2, 0.2, 0.2, 0.4], against [0.25, 0.25, 0.25, 0.25] in GRPO.

    **Why:** in a sample-level loss, tokens of long answers count less. The paper observes that very long samples often hold gibberish and repeated words, and these patterns must get the full penalty.

4. **Penalize overlong answers softly (overlong reward shaping).** DAPO adds a length penalty to the correctness reward:

    $$R_{length}(y) = \begin{cases} 0 & |y| \le L_{max} - L_{cache} \\ \frac{(L_{max} - L_{cache}) - |y|}{L_{cache}} & L_{max} - L_{cache} < |y| \le L_{max} \\ -1 & |y| > L_{max} \end{cases}$$

    Assume $L_{max} = 7$ and $L_{cache} = 2$. Lengths up to 5 get 0, length 6 gets −0.5, length 7 gets −1, and length 8 gets −1. The 6-token wrong answer now has a reward of 0 − 0.5 = −0.5.

    **Why:** a truncated answer can have correct reasoning. A sudden penalty at the limit adds noise. A ramp tells the model "shorter is better" before it hits the limit.

<div class="key-point">

DAPO keeps GRPO's advantage and per-token ratio. It changes the clip (0.8 / 1.28), the batch (no equal-reward groups), the average (all tokens of the batch) and the reward (a length ramp).

</div>

### From toy to real

- DAPO samples 16 answers per prompt, 512 prompts per batch, with a learning rate of 1e-6. The expected maximum length is 16,384 tokens plus a soft punish cache of 4,096 tokens, so generation stops at 20,480. In the formula, a length of 18,432 gets −0.5.
- With Qwen2.5-32B base, DAPO reports 50 points on AIME 2024.

## 4. GSPO and CISPO

### Setup

GSPO and CISPO keep the group advantage. **GSPO** (Group Sequence Policy Optimization, Qwen) changes what the ratio measures. **CISPO** (from MiniMax-M1) clips the importance-sampling weight instead of the token update. We reuse the update of section 1.

### Steps

1. **Compute the GSPO sequence ratio.** GSPO uses one ratio for each answer, the geometric mean of the token ratios:

    $$s_i = \left(\frac{\pi_\theta(y_i)}{\pi_{old}(y_i)}\right)^{1/|y_i|} = \exp\Big(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log r_{i,t}\Big)$$

    For "the cat sat", the token log-ratios are [0.263, 0.234, 0], so $s = 1.180$. The three wrong answers get 1.052, 1.052 and 0.937.

    **Why:** the reward is for the full answer, and the paper argues that "the unit of optimization objective should match the unit of reward". The power $1/|y_i|$ puts short and long answers on one scale.

2. **Clip at the sequence level.** GSPO maximizes $\frac{1}{G}\sum_i \min\big(s_i \hat{A}_i,\ \operatorname{clip}(s_i, 1-\varepsilon, 1+\varepsilon)\hat{A}_i\big)$ with a very small range: [0.9997, 1.0004]. "the cat sat" has $s = 1.180$: the full answer is clipped and its 3 tokens get no gradient. "the sat sat" has $\hat{A} < 0$ and $s = 1.052$, on the unclipped side, so each of its tokens gets the same weight $s\hat{A} = -0.607$. "sat sat sat" has $s = 0.937$ and is clipped.

    **Why:** one token with a wild ratio cannot dominate. All tokens of an answer share one weight and one clip decision. Our step is much larger than a real step, so here almost every answer leaves the range.

3. **Compute the CISPO weight.** CISPO clips the ratio, stops its gradient ($\operatorname{sg}$), and uses it as a weight on a REINFORCE term, with the token-level average of DAPO:

    $$J_{CISPO} = \frac{1}{\sum_i |o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\operatorname{sg}(\hat{r}_{i,t})\,\hat{A}_{i,t}\,\log\pi_\theta(o_{i,t}), \qquad \hat{r}_{i,t} = \operatorname{clip}(r_{i,t},\,1-\varepsilon^{IS}_{low},\,1+\varepsilon^{IS}_{high})$$

    The paper sets no lower bound and tunes only $\varepsilon^{IS}_{high}$. Assume $\varepsilon^{IS}_{high} = 0.28$. For "the cat sat", $\hat{r} = [1.28, 1.264, 1.0]$ and the gradient weights are [2.217, 2.189, 1.732]. GRPO gives [0, 0, 1.732] and DAPO gives [0, 2.189, 1.732].

    **Why:** in PPO and GRPO a clipped token gives no gradient. The paper found that rare tokens such as "However" or "Wait" were clipped out after the first update. CISPO caps their weight but keeps their gradient.

4. **See what the cap does on a negative advantage.** In "the sat sat" ($\hat{A} = -0.577$), `the` has $r = 1.301$. GRPO and DAPO do not clip this case (the min picks the unclipped side), so the weight is −0.751. CISPO caps the weight at $1.28 \cdot (-0.577) = -0.739$.

    **Why:** CISPO caps every weight above $1+\varepsilon^{IS}_{high}$, whatever the sign of the advantage. There is no min, so no token ever loses its gradient. The price is a small bias, which the paper accepts.

<figure class="plot">
<img src="/figures/grpo-family/objective-vs-ratio.svg" alt="Per-token objective against the ratio. For A = +1.732 GRPO is flat above 1.2 at 2.078 and DAPO is flat above 1.28 at 2.217. For A = -0.577 the objective is flat below 0.8 and keeps falling above 1.">
<figcaption>The clipped objective of one token. Dotted lines: the ratios of "cat" (1.264) and "the" (1.301) after the update.</figcaption>
</figure>

<figure class="plot">
<img src="/figures/grpo-family/gradient-weight.svg" alt="Gradient weight against the ratio for A = +1.732. GRPO drops to 0 above 1.2, DAPO drops to 0 above 1.28, CISPO stays at 2.217 above 1.28. The tokens sat, cat and the are at 1.0, 1.264 and 1.301.">
<figcaption>Above the clip, GRPO and DAPO give no gradient. CISPO keeps a capped gradient.</figcaption>
</figure>

Move the new probability of `the` and compare the four methods. The widget starts at the numbers of this page.

<div class="llm-widget" data-widget="policy-objectives"></div>

<div class="key-point">

GSPO makes one ratio and one clip decision per answer. CISPO keeps the token ratio but turns the clip into a capped weight, so every token always gives a gradient.

</div>

### From toy to real

- GSPO uses a clip range of 3e-4 (left) and 4e-4 (right), against 0.2 and 0.27 for its GRPO baseline. The ranges differ in order of magnitude because the ratios have different definitions. GSPO clips about two orders of magnitude more tokens than GRPO, and the paper still reports better training efficiency.
- On the MoE model Qwen3-30B-A3B-Base (48 layers), roughly 10% of the activated experts change after each gradient update for the same sample. Token ratios then jump. GRPO needed **Routing Replay** (cache the experts of the old policy and replay them). GSPO does not need it.
- A variant, GSPO-token, gives each token its own advantage while its ratio stays numerically equal to $s_i$.
- CISPO has no KL term. In a study on Qwen2.5-32B, the paper reports a 2x speedup over DAPO on AIME 2024: CISPO matches DAPO with 50% of the training steps. A token mask in the same formula can bring back the PPO trust region.

## Minimal code

This code gives the gradient weights of section 4 for the token `the` of "the cat sat" and of "the sat sat".

```python
import numpy as np

def clipped(r, A, lo, hi):
    """PPO-style term and its gradient weight (0 on the clipped side)."""
    unclipped, clip = r * A, np.clip(r, 1 - lo, 1 + hi) * A
    return (unclipped, unclipped) if unclipped <= clip else (clip, 0.0)

r_the, s_correct = 1.301, 1.180          # token ratio, GSPO sequence ratio
for A in (1.732, -0.577):
    grpo = clipped(r_the, A, 0.2, 0.2)[1]        # +: 0       -: -0.751
    dapo = clipped(r_the, A, 0.2, 0.28)[1]       # +: 0       -: -0.751
    cispo = min(r_the, 1.28) * A                 # +: 2.217   -: -0.739
    print(A, grpo, dapo, cispo)
print(clipped(s_correct, 1.732, 3e-4, 4e-4))     # GSPO: (1.733, 0.0), the full answer is clipped
```

## Comparison

For PPO (value model, GAE) and RLOO (leave-one-out baseline), see [lesson 3](03-ppo.md).

| Method | Advantage | Ratio level | Clip | Loss aggregation | Fixes what |
|---|---|---|---|---|---|
| GRPO | (r − mean) / std, same for all tokens | Token | 0.8 / 1.2, clipped token: no gradient | Mean over tokens, then over answers | Removes the value model |
| Dr. GRPO | r − mean | Token | As GRPO | Sum / (G × constant) | Length bias, difficulty bias |
| DAPO | As GRPO, equal-reward groups dropped | Token | 0.8 / 1.28 | Sum / all tokens | Entropy collapse, empty groups, long answers that count less, truncation noise |
| GSPO | As GRPO | Answer (geometric mean) | 0.9997 / 1.0004 on the answer | Mean over answers | Noisy token ratios, MoE instability |
| CISPO | Group-relative | Token, as a weight without gradient | Cap at 1 + ε, gradient kept | Sum / all tokens | Lost gradients of rare tokens |
| Toy: `cat` in "the cat sat" | +1.732 (Dr. GRPO: +0.75) | 1.264 (GSPO: 1.180) | GRPO clips, DAPO keeps | Share 0.25 (DAPO with long answer: 0.2) | GRPO weight 0, DAPO and CISPO 2.189 |

## Checks and common misconceptions

- **"The clip limits how far the model moves."** It only removes the gradient of tokens that already moved, and only on one side. In "the sat sat", `the` has $r = 1.301$ with a negative advantage, and GRPO still pushes it with full weight. The step size decides how far the model moves.
- **"The 1/|o| in GRPO is a harmless average."** It halves the penalty for each token of a 6-token wrong answer. Check: in your runs, plot the mean length of correct and wrong answers separately. If only the wrong ones grow, suspect the length bias.
- **"GSPO's tiny clip range means it clips almost nothing."** The opposite: its ratio is on another scale, and it clips about 100 times more tokens than GRPO. Log the clip fraction of each method on its own scale, not one threshold for all.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. Each statement about a real system was read in the linked source on 25 November 2026.

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), Shao et al., 2024 (GRPO objective, outcome advantage, KL estimator, 64 samples, β = 0.04, learning rate 1e-6, ε without a value).
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783), Liu et al., 2025 (Dr. GRPO: length bias, difficulty bias, objective without 1/|o| and std, MAX_TOKENS constant).
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476), Yu et al., 2025 (ε_low = 0.2, ε_high = 0.28, dynamic sampling, token-level loss, overlong formula, 16,384 + 4,096 tokens, 16 samples, 512 prompts, no KL, 50 on AIME 2024).
- [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071), Zheng et al., 2025 (sequence ratio, clip 3e-4 / 4e-4 and GRPO 0.2 / 0.27, clip fractions, 10% expert change, Routing Replay, GSPO-token).
- [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585), MiniMax, 2025 (CISPO objective, no lower bound, only ε_high tuned, clipped "However"/"Wait" tokens, no KL, 2x speedup over DAPO, token mask).
