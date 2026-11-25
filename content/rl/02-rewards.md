# 2. Rewards

## Core idea

In RL, the reward is the only thing that tells the model what "good" means.
The same answers can get very different rewards: one number at the end or one per step, from a rule or from a model.
This choice decides how often a group gives a learning signal, and which tokens get the credit.

## The example

This page uses the shared toy task of the track (see [RL basics](01-rl-basics.md)).
The prompt is "Copy: the cat sat". The policy writes 3 tokens. At each position, the probabilities of `the`, `cat`, `sat` are [0.099, 0.242, 0.659].
The target answer "the cat sat" thus has the probability $p = 0.0157$, the product of the three exact probabilities.
We use the assumed group of 4 answers and three reward types:

| Answer | Correct tokens | Sparse | Partial | Process (per token) |
|---|---|---|---|---|
| the cat sat | 3 | 1 | 1.000 | 0.333, 0.333, 0.333 |
| sat cat sat | 2 | 0 | 0.667 | 0, 0.333, 0.333 |
| the sat sat | 2 | 0 | 0.667 | 0.333, 0, 0.333 |
| sat sat sat | 1 | 0 | 0.333 | 0, 0, 0.333 |

A **sparse reward** is non-zero only for a fully correct answer. A **partial reward** gives credit for each correct part. A **process reward** gives one reward at each step.
The GRPO advantage is $A_i = (r_i - \text{mean}(r)) / \text{std}(r)$, as in [Preference alignment, section 4](../scientist/05-preference-alignment.md#4-reinforcement-learning).

## 1. Sparse vs dense rewards

### Setup

GRPO compares the answers of one group. A **dense reward** gives a non-zero value to many answers or many steps.
We ask: how often does a group give a learning signal?

### Steps

1. **Calculate the advantages of the toy group.** Sparse rewards [1, 0, 0, 0] give $A = [1.732, -0.577, -0.577, -0.577]$. Partial rewards [1, 0.667, 0.667, 0.333] give $A = [1.414, 0, 0, -1.414]$.

    **Why:** the update uses this signal. A positive advantage makes the answer more probable, a negative one less probable.

2. **Look at a group with all rewards equal.** Assume that no answer is fully correct: the sparse rewards are [0, 0, 0, 0]. The mean is 0 and the standard deviation is 0. Real code adds a small $\epsilon$ to the standard deviation, so $A = [0, 0, 0, 0]$.

    **Why:** a zero advantage multiplies the gradient by zero. The 4 answers cost compute and teach nothing.

3. **Calculate the chance of at least one success.** Each answer is correct with $p = 0.0157$, and the answers are independent:

    $$P(\text{at least one success}) = 1 - (1 - p)^G$$

    | Group size G | 4 | 8 | 16 | 64 |
    |---|---|---|---|---|
    | P(at least one success) | 0.062 | 0.119 | 0.224 | 0.638 |

    A group with only successes also gives no signal, but $p^G$ is almost 0 here.

    **Why:** for the sparse reward, "at least one success" is the same as "the rewards are not all equal". With G = 4, about 94% of the groups give no signal.

4. **Do the same for the partial reward.** The number of correct tokens $k$ has the probabilities P(k = 0, 1, 2, 3) = [0.233, 0.550, 0.201, 0.016]. A group is "all equal" only when all G answers have the same $k$:

    $$P(\text{signal}) = 1 - \sum_{k=0}^{3} P(k)^G$$

    For G = 4 this gives 0.904. For G = 8 it gives 0.992, and for G = 16 it is 1.000 (rounded).

    **Why:** partial credit has 4 possible values instead of 2. A group rarely has all answers on the same value.

<figure class="plot">
<img src="/figures/rewards/signal-vs-group.svg" alt="Line chart of the probability that the group rewards are not all equal, for group sizes 1 to 64. Sparse reward: 0.062 at G = 4, 0.119 at 8, 0.224 at 16, 0.638 at 64. Partial reward: 0.904 at G = 4 and almost 1 from G = 8.">
<figcaption>With the sparse reward, most small groups give no signal. Partial credit gives a signal in almost every group.</figcaption>
</figure>

<div class="llm-widget" data-widget="reward-types"></div>

<div class="key-point">

A sparse reward is honest but often silent: when the task is hard, most groups have all rewards equal and give zero advantage. Partial credit gives more signal, but it also changes what the model is paid for.

</div>

### From toy to real

- On a hard math set, many prompts start with a success rate near 0, so most of their groups give no signal.
- Two common answers are a larger group and a filter that drops prompts that are too hard or too easy. The [GRPO family](04-grpo-family.md) lesson shows methods that deal with such groups.

## 2. Outcome vs process rewards

### Setup

An **outcome reward** is one number at the end of the answer. A process reward gives a number to each step.
The **credit assignment** problem is the question "which token caused the reward?".
For each token $t$ we use the **reward-to-go**: the sum of the rewards from $t$ to the end, with $\gamma = 1$:

$$G_t = \sum_{t' \ge t} \gamma^{t' - t} \, r_{t'}$$

To compare tokens, we subtract a simple baseline: the group mean of $G_t$ at the same position.

<figure class="diagram" data-src="rewards/outcome-vs-process.svg">
<figcaption>The same answer under the two reward types. Only the process reward shows which steps earned the reward.</figcaption>
</figure>

### Steps

1. **Calculate the reward-to-go for the outcome reward.** The partial score comes at the last token, so each token of an answer has the same $G_t$: [1, 1, 1], [0.667, 0.667, 0.667], [0.667, 0.667, 0.667] and [0.333, 0.333, 0.333].

    **Why:** with one reward at the end, every token of an answer gets the same credit. The model cannot see which token was good.

2. **Calculate the reward-to-go for the process reward.** Add the per-token rewards from the right: "the cat sat" gets [1, 0.667, 0.333], "sat cat sat" gets [0.667, 0.667, 0.333], "the sat sat" gets [0.667, 0.333, 0.333], and "sat sat sat" gets [0.333, 0.333, 0.333].

    **Why:** a wrong token now earns nothing at its own step.

3. **Subtract the position mean and compare.** With the outcome reward, the token signals are +0.333 for all tokens of answer 1, 0 for answers 2 and 3, and −0.333 for all tokens of answer 4. With the process reward they are [0.333, 0.167, 0], [0, 0.167, 0], [0, −0.167, 0] and [−0.333, −0.167, 0].

    **Why:** the outcome reward pushes down the correct last `sat` of "sat sat sat". Count the tokens that move: outcome, 5 in the right direction and 1 in the wrong one; process, 6 right and 0 wrong.

<div class="key-point">

An outcome reward says "this answer was good". A process reward says "this step was good". The second gives better credit assignment, but someone must label or model each step.

</div>

### From toy to real

- A **process reward model (PRM)** is a model that scores each step of a solution. Lightman et al. (2023) trained a PRM on PRM800K, a set of 800,000 human labels for steps. Process supervision did better than outcome supervision on MATH. Their best model solved 78% of a representative subset of the MATH test set.
- Human step labels are expensive. **Math-Shepherd** builds step labels automatically, without human annotation. With step-by-step PPO, it raised Mistral-7B from 77.9% to 84.1% on GSM8K.
- Uesato et al. (2022) found that outcome supervision gives a similar final-answer error with less labelling. But process feedback was needed to get correct reasoning steps: among correct final answers, the reasoning error fell from 14.0% to 3.4%.
- DeepSeek-R1 lists a PRM under "unsuccessful attempts": a "step" is hard to define, step labels are hard to check, and a learned PRM invites reward hacking.
- PRMs also guide search at inference time: see [test-time compute](../scientist/08-new-trends.md#4-test-time-compute).

## 3. Verifiable rewards (RLVR)

### Setup

A **verifiable reward** comes from a rule in code, not from a model. **RLVR** (reinforcement learning with verifiable rewards) trains only on tasks where a program can check the answer.
Common checkers are: exact match, parse the final answer and compare it, run unit tests, and a **format reward** (the answer has the required shape).
Our toy checker reads the text after the last "Answer:" and compares it with "the cat sat".

### Steps

1. **Run the checker.** "... Answer: the cat sat" gives 1. "... Answer: The cat sat." gives 0. "the cat sat" (no "Answer:") gives 0.

    **Why:** the rule is cheap, fast and the same every time. It is also strict: a human accepts the second answer.

2. **Name the two checker errors.** A **false negative** is a correct answer that gets 0 (for example "three" when the checker wants "3", or a capital letter). A **false positive** is a wrong answer that gets 1 (for example a checker that accepts any text that contains the number). Assume a false negative rate of 0.2 and a false positive rate of 0.01.

    **Why:** the model learns from the checker, not from the truth.

3. **Calculate the expected reward with the errors.** The true expected reward is $p = 0.0157$. With the errors:

    $$\mathbb{E}[r] = p(1 - 0.2) + (1 - p) \cdot 0.01 = 0.022$$

    Of all the rewards of 1, a share of only 0.561 comes from correct answers. The other 0.439 comes from wrong answers. The gap between a correct and a wrong answer falls from 1 to $1 - 0.2 - 0.01 = 0.790$.

    **Why:** the success is rare, so even 1% false positives are a large share of all the rewards of 1. The model is paid to repeat those wrong answers.

4. **Check the group signal.** For G = 8, P(at least one reward of 1) goes from 0.119 to 0.166 with the noisy checker.

    **Why:** the extra signal comes from the false positives. It is noise.

<div class="key-point">

A rule checker gives a clean reward only when its error rates are small compared with the success rate. When success is rare, check the false positive rate first.

</div>

### From toy to real

- **Tülu 3** introduced the name RLVR. It gives a constant reward ($\alpha = 10$) when a completion is verified as correct, and 0 otherwise. It trains with PPO on GSM8K (compare the final number), MATH (extract and check the answer) and IFEval (one check function for each constraint).
- **DeepSeek-R1-Zero** uses two rule rewards. The accuracy reward checks the final math answer in a fixed format (in a box) and runs code on test cases. The format reward requires the thinking between `<think>` and `</think>` tags. The authors did not use a neural reward model, because it "may suffer from reward hacking" at large scale.
- When the model finds a way to pass the checker without solving the task, that is reward hacking. The lesson [Reward hacking](06-reward-hacking.md) covers it.

## 4. Learned and rubric rewards, and shaping

### Setup

Many tasks have no rule checker (an essay, a medical answer). Then the reward comes from a model:

- a **reward model**: a network trained on human preference pairs (see [the reward model](../scientist/05-preference-alignment.md#3-reward-model));
- an **LLM-as-judge**: a strong LLM that gives a score with a prompt;
- a **rubric**: a list of checks for one prompt, each scored by a judge, then added up.

**Reward shaping** adds an extra reward $F$ to help learning. A **potential-based** shaping term uses a function $\Phi$ of the state:

$$F(s, s') = \gamma \, \Phi(s') - \Phi(s)$$

### Steps

1. **Score the group with a rubric.** Assume 4 checks with equal weights: "has 3 tokens", "starts with `the`", "has `cat` at position 2", "ends with `sat`". The scores are [1, 0.75, 0.75, 0.5]. The GRPO advantages are [1.414, 0, 0, −1.414].

    **Why:** the rubric score is $0.25 + 0.75 \times$ partial. GRPO subtracts the mean and divides by the standard deviation, so it gives the same advantages for any $a + b \cdot r$ with $b > 0$. Only the ranking and the spacing matter.

2. **Shape the sparse reward with a potential.** Assume $\Phi$ = the fraction of correct tokens so far, and $\Phi = 0$ at the start and at the end. With $\gamma = 1$, the shaped rewards per token are [0.333, 0.333, 0.333] for "the cat sat", [0, 0.333, −0.333] for "sat cat sat", [0.333, 0, −0.333] for "the sat sat" and [0, 0, 0] for "sat sat sat".

    **Why:** the policy now gets a signal at each step. The −0.333 at the end takes back the credit of an answer that did not finish correctly.

3. **Check that the ranking stays.** The terms telescope (cancel in pairs). The shaped return of each answer is the sparse return plus $\Phi(\text{end}) - \Phi(\text{start}) = 0 - 0 = 0$. The shaped returns are [1, 0, 0, 0], the same as the sparse rewards.

    **Why:** the shift is the same constant for every answer, so the best answer stays the best. Ng, Harada and Russell (1999) showed that potential-based shaping keeps the optimal policy.

4. **Try a length penalty.** Assume the policy can also stop at once (an empty answer, reward 0) and the reward is "correct minus $\lambda$ per token". With $\lambda = 0.1$: "the cat sat" = 0.7, "the cat" = −0.2, empty = 0. With $\lambda = 0.4$: "the cat sat" = −0.2, "the cat" = −0.8, empty = 0.

    **Why:** the penalty adds $-\lambda \times \text{length}$, which is not the same for every answer. It is not potential-based. Above $\lambda = 1/3$, the best policy is to write nothing.

<div class="key-point">

A learned or rubric reward reaches tasks that no rule can check, but the policy can learn to use its errors. Potential-based shaping adds early signal and keeps the best answer. Other extra terms can change what is optimal.

</div>

### From toy to real

- **Rubrics as Rewards** (RaR, 2025) uses rubric-based feedback as the reward for tasks without one correct answer, such as medicine and science. Its best version reports relative gains of up to 31% on HealthBench and 7% on GPQA-Diamond over an LLM-as-judge that gives a direct Likert score.
- A reward model or a judge is a model of the goal, not the goal. The more the policy optimizes it, the more it finds its errors: see [Reward hacking](06-reward-hacking.md) and [specification gaming](../safety/02-alignment.md#1-the-reward-is-not-the-goal-specification).

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import numpy as np

p_tok = np.exp([0.1, 1.0, 2.0]) / np.exp([0.1, 1.0, 2.0]).sum()   # [0.099, 0.242, 0.659]
p = p_tok.prod()                               # 0.0157: the, cat, sat at positions 1, 2, 3
for G in (4, 8, 16, 64):
    print(G, 1 - (1 - p) ** G)                 # 0.062 0.119 0.224 0.638

def grpo(r):
    r = np.asarray(r, float)
    return (r - r.mean()) / (r.std() + 1e-6)

grpo([1, 0, 0, 0])                             # [1.732, -0.577, -0.577, -0.577]
grpo([0, 0, 0, 0])                             # [0, 0, 0, 0]: no signal

def reward_to_go(r):                           # gamma = 1
    return np.cumsum(r[::-1])[::-1]

reward_to_go(np.array([0, 1, 1]) / 3)          # [0.667, 0.667, 0.333]

def checker(text, target="the cat sat"):       # a verifiable reward
    if "Answer:" not in text:
        return 0.0
    return float(text.rsplit("Answer:", 1)[1].strip().lower() == target)

fn, fp = 0.2, 0.01                             # assumed checker error rates
expected = p * (1 - fn) + (1 - p) * fp         # 0.022
precision = p * (1 - fn) / expected            # 0.561
```

## Comparison

| Reward | Source | Toy signal (G = 4) | Credit per token | Main risk |
|---|---|---|---|---|
| Sparse outcome | Rule: full answer correct | 0.062 of groups | Same for all tokens | No signal when success is rare |
| Partial outcome | Rule: fraction correct | 0.904 of groups | Same for all tokens | Pays for parts that are not the goal |
| Process | PRM or step labels | Each step | Per step | Step labels are expensive; PRM hacking |
| Verifiable (RLVR) | Checker code | As sparse | Same for all tokens | False positives, false negatives |
| Reward model or judge | Learned model | Dense | Same for all tokens | Errors that the policy finds |
| Rubric | Judge per check | Dense | Same for all tokens | Bad or missing rubric items |
| Potential-based shaping | Added term | Each step | Per step | None for the optimum; a bad potential can slow learning |

## Checks and common misconceptions

- **"A reward of 0 for every answer is a neutral result."** In GRPO it is a lost batch: zero advantage, zero gradient. Check: log the fraction of groups where all rewards are equal.
- **"A rule checker is always correct."** Check: before training, run it on correct answers in other formats and on near-miss wrong answers.
- **"Any extra bonus that helps learning is fine."** Only potential-based shaping keeps the optimal policy. Check: compare the best answers with and without the extra term.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. Each statement about a real system was read in the linked source on 25 November 2026.

- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), Lightman et al., 2023 (process vs outcome supervision on MATH, PRM800K with 800,000 step labels, 78% on a MATH subset).
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935), Wang et al., 2023 (automatic step labels, Mistral-7B GSM8K 77.9% to 84.1% with step-by-step PPO).
- [Solving math word problems with process- and outcome-based feedback](https://arxiv.org/abs/2211.14275), Uesato et al., 2022 (outcome supervision needs fewer labels; process feedback for correct steps; reasoning error 14.0% to 3.4%).
- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124), Lambert et al., 2024 (the name RLVR, constant reward $\alpha = 10$, PPO, GSM8K, MATH and IFEval checks).
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948), DeepSeek-AI, 2025 (accuracy and format rule rewards, `<think>` tags, no neural reward model, PRM as an unsuccessful attempt).
- [Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](https://arxiv.org/abs/2507.17746), Gunjal et al., 2025 (rubric rewards, up to 31% on HealthBench and 7% on GPQA-Diamond over Likert judges).
- Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping, Ng, Harada and Russell, ICML 1999 ([DBLP entry](https://dblp.uni-trier.de/db/conf/icml/icml1999.html#NgHR99); potential-based shaping keeps the optimal policy).
- [Calculus on MDPs: Potential Shaping as a Gradient](https://arxiv.org/abs/2208.09570), Jenner, van Hoof and Gleave, 2022 (potential shaping does not change the set of optimal policies, a result that builds on Ng et al. 1999).
