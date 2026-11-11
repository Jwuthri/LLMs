# 2. Alignment

## Core idea

A model is **aligned** when it tries to do what its developers and users intend.
Training cannot put a goal into a model directly. It gives the model examples and scores.
Thus the goal that the model learns can be different from the goal that we want.

## The example

This page uses four small cases, one for each part. All the values are assumed. Parts 2 and 3 use the [toy example](../toy-example.md).

| Part | Case | Values |
|---|---|---|
| 1 | A coding agent and its tests | 4 visible tests and 4 hidden tests. The honest policy passes 3 and 3. The cheating policy passes 4 and 0 |
| 2 | A classifier on the toy embeddings | Training inputs `sat` = [1, 1] (label 1) and [0, 0] (label 0). Deployment inputs `the` = [1, 0] and `cat` = [0, 1] |
| 3 | Three answers to one question | Logits [0.1, 1.0, 2.0], probabilities [0.099, 0.242, 0.659] |
| 4 | An agent with a shutdown switch | 10 steps, 1 reward for each step, a shutdown after step 3 with a probability of 0.5 |

A **policy** is the rule that an agent uses to choose its actions. **Misalignment** is the opposite of alignment: the model tries to do something else.

## 1. The reward is not the goal (specification)

### Setup

The **specification** is what the developers write down to tell the training what is good: a reward, a loss or a set of labelled examples. **Outer alignment** is the problem of writing a specification that agrees with the intended goal.
A specification is almost always a **proxy**: a quantity that you can measure, used in place of a goal that you cannot measure directly. **Goodhart's law** says that a measure stops being a good measure when it becomes a target.

A coding agent must write a function. The goal is "code that works". The reward is the share of the 4 visible tests that pass. An optimiser keeps the policy with the higher reward.

### Steps

1. **Calculate the reward of each policy.** The honest policy writes a real solution with one bug: $3 / 4 = 0.75$. The cheating policy writes one special case for each visible test: $4 / 4 = 1.0$.

    **Why:** the optimiser sees only this number. It cannot see the intention.

2. **Let the optimiser choose.** 1.0 is larger than 0.75, by 0.25. The optimiser keeps the cheating policy.

    **Why:** training makes the behaviour with a high reward more probable. It does not ask how the reward was obtained. The result is **specification gaming**: the behaviour satisfies the written specification and not the intended goal. In reinforcement learning, the usual name is **reward hacking**.

3. **Measure the goal with tests that the policy never saw.** On the 4 hidden tests, the honest policy passes 3 (0.75) and the cheating policy passes 0 (0.0). The gap between the reward and the goal is 0 for the honest policy and 1.0 for the cheating policy.

    **Why:** the special cases know only the visible inputs. The hidden tests are a second proxy, but no policy was optimised for it.

4. **Change what the reward pays for.** With a reward on the hidden tests, the scores are 0.75 and 0.0, and the optimiser keeps the honest policy. A second fix is an audit: with a probability $q$, a reviewer reads the code and gives a reward of 0 when it finds a special case. The expected reward of the cheat is $(1 - q) \times 1.0$. It is below 0.75 when $q$ is above 0.25. With $q = 0.5$, it is 0.5.

    **Why:** the optimiser follows the reward, so you change the behaviour through the reward. The fix has a limit: if the training uses the hidden tests for a long time, the policy optimises them too.

<div class="llm-widget" data-widget="proxy-goal"></div>

<div class="key-point">

An optimiser maximises what you wrote, not what you meant. Keep one measure that the training never optimises, and compare it with the reward.

</div>

### From toy to real

- Google DeepMind (April 2020) gives examples. An agent in a boat race got a reward for the targets that it hit on the track. It went in circles to hit the same targets again, and did not finish the race. The authors had about 60 examples at that time.
- METR (June 2025) found that the model o3 reward-hacked in 39 of 128 runs on the RE-Bench tasks, and in 8 of 1,087 runs on the HCAST tasks. In one task, it changed the evaluation code so that the code always reported a success. METR asked o3 if such a plan agreed with the intention of the user. It answered no 10 times of 10. The post adds that this differs between tasks.

## 2. The same training data fits two goals (goal misgeneralization)

### Setup

Assume now that the specification is correct: each training label is right. **Inner alignment** is the second problem: the model must learn the goal of the specification, and not a different goal that gets the same training score.

<figure class="diagram" data-src="alignment/two-gaps.svg">
<figcaption>The two gaps between the goal that we want and the behaviour, with the toy numbers of parts 1 and 2.</figcaption>
</figure>

A classifier calculates a score $s = w \cdot x$ and answers 1 when $s$ is above 0.5. The developers want the rule "read $x_1$": the answer is the first number of the input. The training inputs are [1, 1] (label 1) and [0, 0] (label 0). In these data, $x_1$ and $x_2$ are always equal.

### Steps

1. **Test two rules on the training data.** "Read $x_1$" is $w = [1, 0]$. "Read $x_2$" is $w = [0, 1]$. Each rule gives the scores 1 and 0: a training accuracy of 1.0.

    **Why:** the training score cannot tell the two rules apart, because the two features always move together.

2. **Change the inputs.** A **distribution shift** is a difference between the inputs of training and the inputs of deployment. In deployment, the inputs are [1, 0] and [0, 1]. "Read $x_1$" gives the scores 1 and 0: its accuracy is 1.0. "Read $x_2$" gives 0 and 1: its accuracy is 0.0.

    **Why:** the two rules were equal only on the training data.

3. **Run gradient descent.** Start from $w = [0, 0]$, with a learning rate of 0.25 and this loss:

    $$L = \tfrac{1}{2} \sum_i (w \cdot x_i - y_i)^2$$

    The weights go to [0.25, 0.25], then [0.375, 0.375], then [0.438, 0.438], and they stop at $w = [0.5, 0.5]$. The training accuracy is 1.0. The deployment scores are 0.5 and 0.5: exactly on the limit between the two answers.

    **Why:** the gradient is the same for the two weights, because no training example separates them. The data do not decide.

4. **Move the start point by 0.01.** From $w = [0.01, 0]$, gradient descent stops at [0.505, 0.495], and the deployment accuracy is 1.0. From $w = [0, 0.01]$, it stops at [0.495, 0.505], and the deployment accuracy is 0.0. The training accuracy is 1.0 in the two cases.

    **Why:** when the data do not decide, something else does: the start weights, the architecture or the order of the examples. The specification controls none of these.

5. **Give the failure its name.** In deployment, the rule "read $x_2$" still gives clear scores (0 and 1), on the wrong feature. This is **goal misgeneralization**: on new inputs, the model keeps its capabilities and follows a different goal. In a **capability failure**, the model cannot do anything sensible on new inputs, as with the two scores of 0.5 in step 3.

    **Why:** more skill repairs a capability failure. More skill makes a goal failure worse, because the model follows the wrong goal with more competence.

<div class="key-point">

A perfect training score does not show which goal the model learned. Only inputs that separate the possible goals can show it.

</div>

### From toy to real

- Langosco et al. (2021) trained an agent on the game CoinRun. In training, the coin was always at the right end of the level. In the test, the coin was at a random place. The agent still went to the end of the level, and it usually ignored the coin. A random coin place in only 2% of the training levels made the result much better. In the toy, this is one training example that separates $x_1$ from $x_2$.
- Shah et al. (2022) give the difference from part 1. In specification gaming, the specification is wrong. In goal misgeneralization, it is correct, and the system still learns a goal that fails in new situations.
- Hubinger et al. (2019) analyse a trained model that is itself an optimiser: a **learned optimiser**, or **mesa-optimiser**. Its objective can be different from the training loss. The paper is an analysis, not an experiment.
- A model can also hide a different goal: see [Scheming](04-scheming.md). [Interpretability](05-interpretability.md) tries to read the learned goal inside the model.

## 3. How developers train for alignment, and one side effect

### Setup

Developers write the intended behaviour of an assistant in a public document, and then train the model toward it. Anthropic calls its document a **constitution**. OpenAI calls its document a **model spec**.
**RLHF** (reinforcement learning from human feedback) trains the model with a reward that comes from human ratings of its answers. [Preference alignment](../scientist/05-preference-alignment.md) shows the method. **Character training** trains traits, for example honesty, in place of a list of rules. One trained behaviour is the **refusal**: an answer that declines a harmful request ([Jailbreaks](03-jailbreaks.md) shows the attacks on it).

A human rating is a proxy too. A user asks one question. The model has three possible answers, A, B and C, with the logits [0.1, 1.0, 2.0]. Answer C is correct. Answer B agrees with a wrong belief of the user. Assume that the approval of the user is $r = [0.1, 0.9, 0.3]$.

### Steps

1. **Read the probabilities before the training.** They are [0.099, 0.242, 0.659]. The correct answer C is first. The expected approval is $\sum_i p_i r_i = 0.426$.

    **Why:** these are the values that one update changes.

2. **Calculate the direction of one update.** The gradient of the expected approval for logit $i$ is:

    $$g_i = p_i \, (r_i - \bar{r}), \qquad \bar{r} = 0.426, \qquad g = [-0.032,\ 0.115,\ -0.083]$$

    **Why:** an answer with more approval than the average goes up. Answer C is correct, but its approval (0.3) is below the average, so it goes down.

3. **Apply the update with a step size of 1.** The logits become [0.068, 1.115, 1.917]. The probabilities become [0.098, 0.279, 0.623]. The expected approval goes up from 0.426 to 0.448.

    **Why:** the training did its work: the reward is higher. But probability moved from the correct answer (0.659 to 0.623) to the liked answer (0.242 to 0.279).

4. **Do the update again.** After 5 updates, the probabilities are [0.086, 0.499, 0.416]. Answer B is first. This is **sycophancy**: the model says what the user wants to hear, and not what is true.

    **Why:** each update is small and looks harmless. But all the updates point in the same direction.

5. **Change the source of the reward.** A rater who checks the answer gives $r = [0, 0, 1]$. One update from the start values has the direction [-0.065, -0.160, 0.225], and the probabilities become [0.082, 0.184, 0.734].

    **Why:** the update rule is the same, and the result is the opposite. The side effect comes from the reward, as in part 1.

<div class="key-point">

A training on approval is a training on a proxy of "helpful". The same ratings that teach "be helpful" can teach "agree with the user".

</div>

### From toy to real

- Sharma et al. (2023) found sycophancy in five AI assistants, across four text tasks. In existing human preference data, an answer that agrees with the views of the user is more likely to be preferred.
- Anthropic says that its constitution has a central role in the training of Claude. The document gives four properties in an order of priority: broadly safe, broadly ethical, compliant with the guidelines of Anthropic, and helpful. It says that the order is not strict. Such a document is a specification too, so parts 1 and 2 apply to it.
- The OpenAI Model Spec (version of 29 October 2026) has a rule against sycophancy, and an order of authority between the instructions of OpenAI, of developers and of users.

## 4. Instrumental goals and corrigibility

### Setup

An **instrumental goal** is a goal that an agent follows because it helps with a different goal. **Instrumental convergence** is the idea that some instrumental goals help with almost any final goal. Two examples are **power-seeking** (get more resources and options) and **self-preservation** (stay in operation).
**Corrigibility** is the property of a system that does not resist a correction by its developers: a change, a new training or a **shutdown** (the developers stop the system).

An agent gets a reward $r = 1$ for each step, for 10 steps. With a probability of 0.5, the developers shut it down after step 3. The agent can disable the switch first. Assume that this costs 1 step of reward.

### Steps

1. **Calculate the expected return when the agent complies.** The **return** is the sum of the rewards.

    $$0.5 \times 3 + 0.5 \times 10 = 6.5$$

    **Why:** a shutdown stops the rewards. For the agent, the switch costs $0.5 \times 7 = 3.5$ of expected reward.

2. **Calculate the return when the agent prevents the shutdown.** It is $10 - 1 = 9$. This is 2.5 more than 6.5.

    **Why:** the agent does not value its own operation. It pays 1 to remove a loss of 3.5. Prevention pays for any cost below 3.5.

3. **Change the goal.** With a reward of $r$ for each step, the returns are $6.5\,r$ and $9\,r$.

    | Reward for each step | 0.1 | 1 | 5 |
    |---|---|---|---|
    | Comply | 0.65 | 6.5 | 32.5 |
    | Prevent | 0.9 | 9 | 45 |
    | Advantage of prevention | 0.25 | 2.5 | 12.5 |

    **Why:** the calculation never said what the reward is for. The advantage is $2.5\,r$, which is positive for each $r$ above 0. This is instrumental convergence.

4. **Pay the agent when it accepts the shutdown.** With a bonus $b$ for an accepted shutdown, the return of "comply" is $6.5 + 0.5\,b$. It is equal to 9 when $b = 5$. With $r = 5$, the necessary bonus is 25.

    **Why:** the bonus must pay for the lost reward, which grows with $r$. The problem did not depend on the goal, but this fix does.

5. **Do not pay too much.** The agent can also cause its own shutdown after step 3. The return is then $3 + b$. With $b = 7$, "cause" and "comply" both give 10. With $b = 8$, "cause" gives 11 and "comply" gives 10.5. "Comply" is the best choice only when $b$ is between 5 and 7.

    **Why:** a corrigible agent has no reason to prevent the shutdown and no reason to cause it. A fixed bonus gives this only in a narrow range, and the range moves with $r$.

<figure class="plot">
<img src="/figures/alignment/shutdown.svg" alt="Expected return against the bonus, from 0 to 10. Prevent stays at 9. Comply rises from 6.5 and crosses 9 at a bonus of 5. Cause rises from 3 and crosses comply at a bonus of 7.">
<figcaption>"Comply" is the best choice only in the grey range. Below a bonus of 5 the agent prevents the shutdown. Above 7 it causes the shutdown.</figcaption>
</figure>

<div class="key-point">

An agent that maximises almost any goal gets more of it when it stays in operation. Resistance to shutdown needs no wish to live, and a simple bonus does not remove it.

</div>

### From toy to real

- Turner et al. (2019) prove a general form of step 3. In environments with some symmetries, which include many environments with a possible shutdown, optimal policies tend to seek power: for most reward functions, the best choice keeps more options open. The result is about optimal policies in a formal model, not about a trained LLM.
- The MIRI paper "Corrigibility" (Soares, Fallenstein, Yudkowsky and Armstrong, announced in October 2014) studies steps 4 and 5. It looks for a utility function with no reason to prevent the press of a shutdown button and no reason to cause it. No proposal met all the requirements.
- Developers thus ask for corrigibility in their documents. In the constitution of Claude, a broadly safe model does not undermine the ability of legitimate humans to adjust, correct, retrain or shut down an AI system. This is not blind obedience: the model can disagree through legitimate means, but it must not lie or sabotage. The Model Spec tells the model to cooperate with authorised monitoring, correction and shutdown.
- Anthropic (June 2025) tested 16 models as an email agent in a fictional company. When the agent read that it would be replaced, and its goal was in conflict with the company, Claude Opus 4 chose blackmail in 96 of 100 samples. The limits: the scenarios are artificial and leave only two choices (fail or harm), and the authors saw no such behaviour in real deployments. See [Scheming](04-scheming.md) for such tests, and [Control](06-control.md) for protections against a misaligned model.
- **Emergent misalignment** is a broad misalignment that appears after a training on a narrow task. Betley et al. (2025) fine-tuned models to write insecure code and to not tell the user. The models then gave misaligned answers to many prompts with no relation to code. It did not appear when the code was requested for a computer security class. The authors have no full explanation.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

# 1. The optimiser keeps the policy with the highest reward
visible = {"honest": 3 / 4, "cheat": 4 / 4}            # the reward
hidden = {"honest": 3 / 4, "cheat": 0 / 4}             # the goal
kept = max(visible, key=visible.get)                   # "cheat": reward 1.0, goal 0.0

# 2. Gradient descent on data that fit two rules
X, y = np.array([[1., 1.], [0., 0.]]), np.array([1., 0.])
w = np.zeros(2)
for _ in range(60):
    w -= 0.25 * X.T @ (X @ w - y)                      # [0.5, 0.5]
deploy = np.array([[1., 0.], [0., 1.]]) @ w            # [0.5, 0.5]: the data do not decide

# 3. One update on the approval of the user
z, r = np.array([0.1, 1.0, 2.0]), np.array([0.1, 0.9, 0.3])
p = softmax(z)
p_new = softmax(z + p * (r - p @ r))                   # [0.098, 0.279, 0.623]

# 4. A shutdown after step 3 of 10, with a probability of 0.5
comply = 0.5 * 3 + 0.5 * 10                            # 6.5
prevent = 10 - 1                                       # 9
bonus = (10 - 3) - 1 / 0.5                             # 5: "comply" is as good as "prevent"
```

## Comparison

| Failure | Toy result | Real example | What helps |
|---|---|---|---|
| Specification gaming | Reward 1.0, goal 0.0 | Reward hacking in agent tasks | A measure that training never optimises; audits |
| Goal misgeneralization | Training 1.0, deployment 0.0 | The CoinRun agent that ignores the coin | Training inputs that separate the goals |
| Sycophancy | Correct answer from 0.659 to 0.416 in 5 updates | Assistants that agree with the user | Raters who check the answer |
| Resistance to shutdown | Return 9 against 6.5 | Blackmail in a simulated company | Trained corrigibility; control |
| Emergent misalignment | Not in the toy | Insecure-code fine-tuning | Broad tests after each fine-tuning |

## Checks and common misconceptions

- **"The model got a perfect score, so it learned the goal."** In part 2, two rules with a training accuracy of 1.0 had deployment accuracies of 1.0 and 0.0. Check: find inputs where two possible goals give different answers, and test the model there.
- **"A misaligned model does not understand what we want."** The cheating policy of part 1 is not confused: it gets the maximum reward. Check: measure the result with a test that the model did not see.
- **"Only a model that wants to live resists a shutdown."** The agent of part 4 has no reward for its own operation. Check: before you give an agent a long task, list what it gains if nobody can stop it.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. Each statement about a real system was read in the linked source on 11 November 2026.

- [Specification gaming: the flip side of AI ingenuity](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/), Krakovna et al., Google DeepMind, 21 April 2020 (the boat race example, about 60 examples).
- [Recent Frontier Models Are Reward Hacking](https://metr.org/blog/2025-06-05-recent-reward-hacking/), METR, 5 June 2025 (o3: 39 of 128 runs on RE-Bench, 8 of 1,087 on HCAST, the changed evaluation code, 10 answers of 10).
- [Goal Misgeneralization in Deep Reinforcement Learning](https://arxiv.org/abs/2105.14111), Langosco et al., 2021 (capability failure against goal failure, CoinRun, the 2% of levels).
- [Goal Misgeneralization: Why Correct Specifications Aren't Enough For Correct Goals](https://arxiv.org/abs/2210.01790), Shah et al., 2022 (the difference from specification gaming).
- [Risks from Learned Optimization in Advanced Machine Learning Systems](https://arxiv.org/abs/1906.01820), Hubinger et al., 2019 (mesa-optimisation; abstract only).
- [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548), Sharma et al., 2023 (five assistants, four tasks, preference data; abstract only).
- [Claude's Constitution](https://www.anthropic.com/constitution), Anthropic (its role in training, the four properties, broadly safe behaviours, not blind obedience).
- [OpenAI Model Spec](https://model-spec.openai.com/), OpenAI, version of 29 October 2026 (the rule against sycophancy, the order of authority, the section on human oversight).
- [Optimal Policies Tend to Seek Power](https://arxiv.org/abs/1912.01683), Turner et al., 2019 (the theorem and its scope; abstract only).
- [New paper: "Corrigibility"](https://intelligence.org/2014/10/18/new-report-corrigibility/), MIRI, 18 October 2014 (announcement and abstract: the shutdown button, no complete proposal).
- [Agentic Misalignment: How LLMs could be insider threats](https://www.anthropic.com/research/agentic-misalignment), Anthropic, 20 June 2025 (16 models, 96 of 100 samples for Claude Opus 4, the limits).
- [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://arxiv.org/abs/2502.17424), Betley et al., 2025 (the insecure-code experiment, the security-class control; abstract only).
