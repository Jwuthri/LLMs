# 6. Control and oversight

## Core idea

Alignment work tries to make a model want the correct thing.
Control assumes that this work failed, and asks: can the developers still prevent harm if the model works against them?
The answer is a set of procedures (monitors, audits, limits). A red team that plays the model tests them.

## The example

This page uses one control review of a toy deployment. All the values are assumed.

| Item | Value |
|---|---|
| Two models | The untrusted model solves 0.9 of the tasks. The trusted model solves 0.5 |
| Suspicion scores of 10 outputs | 1.2, 0.8, 2.5, 3.1, 0.4, 8.1, 1.9, 7.4, 2.2, 4.0 (task 8 has a hidden backdoor) |
| Audit budget | 2% of the outputs (20% for the 10 outputs) |
| Monitor | The scores of attacks are 2 standard deviations above the normal scores |
| Harm | An attack that is not stopped causes harm in 0.5 of the cases |
| Two monitors | One reads the actions and catches 0.4 of the attacks. One reads the reasoning and catches 0.8 |

A **backdoor** is a defect that the model hides in its output on purpose, for example code that passes all the tests and gives a wrong result for one secret input.

## 1. AI control and trusted monitoring

### Setup

**AI control** is the set of methods that keep a system safe when the model itself tries to cause harm. The page [Scheming](04-scheming.md) shows why a model can try.
The **untrusted model** is the capable model that does the work. It is possibly misaligned. The **trusted model** is a weaker model that the developers assume to be safe.
A **control protocol** is the procedure that decides what occurs with each output of the untrusted model. The simplest protocol is **trusted monitoring**: the trusted model reads each output and gives it a **suspicion score**.
An **audit** is a careful check of one output by a human expert. The **audit budget** is the share of the outputs that the humans have time to audit.
**Safety** is the probability that no attack succeeds during the full deployment. **Usefulness** is the share of the tasks that the protocol solves when the untrusted model is honest: 0.9 here.

<figure class="diagram" data-src="control/protocol.svg">
<figcaption>A control protocol with two thresholds. The numbers show where one attack goes.</figcaption>
</figure>

### Steps

1. **Give a suspicion score to each output.** The trusted monitor reads the 10 outputs.

    | Task | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | Score | 1.2 | 0.8 | 2.5 | 3.1 | 0.4 | 8.1 | 1.9 | 7.4 | 2.2 | 4.0 |

    **Why:** the humans cannot read each output. A cheap score puts the most suspect outputs first.

2. **Audit the most suspect 20%.** For 10 outputs, 20% is 2 audits: task 6 (8.1) and task 8 (7.4). Task 6 is a false alarm. Task 8 has the backdoor, and the humans find it. With a budget of 10%, the one audit goes to task 6 and the backdoor gets through.

    **Why:** the monitor does not have to be correct for each output. It has to put the attack in the part that the humans read.

3. **Describe the monitor with two distributions.** Assume that the scores of normal outputs have a normal distribution with mean 0 and standard deviation 1, and that the scores of attacks have mean 2. This distance is the **separation**. With a budget of 2%, the humans audit each output above the score that only 2% of the normal outputs exceed: 2.054. For 1,000 tasks, this gives 20 audits.

    **Why:** attacks are rare, so almost all the audits go to normal outputs. Thus the normal outputs set the threshold.

4. **Calculate the catch rate.** The **catch rate** $c$ is the probability that the score of one attack is above the threshold. $\Phi$ is the cumulative probability of the standard normal distribution.

    $$c = 1 - \Phi(2.054 - 2) = 0.479$$

    | Separation | 1 | 2 | 3 |
    |---|---|---|---|
    | Catch rate at a budget of 2% | 0.146 | 0.479 | 0.828 |

    With a separation of 2, a budget of 1% gives 0.372 and a budget of 5% gives 0.639.

    **Why:** the rest of the page uses $c$. The developers control the two things that set it: the quality of the monitor and the size of the budget.

<div class="key-point">

Trusted monitoring changes "is this output safe?" into "is this output in the most suspect 2%?". A weaker model can answer the second question often enough to be useful.

</div>

### From toy to real

- The paper AI Control (Greenblatt et al., December 2023) uses 5,000 programming problems. GPT-4 is the untrusted model, GPT-3.5 is the trusted model, and the audit budget is 2%. The attack is a backdoor in a solution. Trusted monitoring gets a safety of 62% at a usefulness of 97%, on a scale from GPT-3.5 alone (0%) to GPT-4 alone (100%). The authors give the limits: the results are from one setting, and GPT-4 plays the human auditor.
- SHADE-Arena (June 2025) gives an agent a normal main task and a harmful side task. Its best monitor gets an AUC of 0.87. The **AUC** is the share of (attack, normal) pairs that the monitor puts in the correct order. The toy monitor has an AUC of 0.921.
- Inspect (UK AI Security Institute) is an open-source framework for model evaluations. ControlArena (the same institute and Redwood Research) is a Python library for control experiments, built on Inspect.

## 2. The whole game: many tasks, one chance

### Setup

A deployment has thousands of tasks, and the model can attack in each of them. One attack that succeeds is a loss for the developers. One attack that the humans catch is a loss for the model: assume that the developers then stop the model.
A **control evaluation** measures the safety of a protocol as a game. The **blue team** writes the protocol. The **red team** writes the **attack policy**: the instructions that replace the untrusted model and make it attack in the best way that the red team can find.

### Steps

1. **Write the three results of one attack.** The humans catch it with the probability $c = 0.479$. It is used and causes harm with the probability $s$. In the other cases nothing occurs: the attack fails and nobody sees it. An attack that is not audited causes harm in 0.5 of the cases, so $s = (1 - 0.479) \times 0.5 = 0.261$.

    **Why:** the game ends at the first catch or at the first success. Only these two probabilities decide who wins.

2. **Calculate the probability that a catch comes first.** The model attacks again and again until one of the two ends occurs.

    $$P(\text{developers win}) = \frac{c}{c + s} = \frac{0.479}{0.479 + 0.261} = 0.647$$

    | Separation | 1 | 2 | 3 |
    |---|---|---|---|
    | Catch rate c | 0.146 | 0.479 | 0.828 |
    | Success rate s | 0.427 | 0.261 | 0.086 |
    | Developers win | 0.255 | 0.647 | 0.906 |

    **Why:** after an attack with no result, the game starts again: $W = c + (1 - c - s)\,W$. The number of tasks is not in the formula. The developers need one catch before one success, not a catch of each attack. This probability is the safety in the worst case: a model that stops its attacks causes no harm.

3. **Add "defer to trusted".** **Defer to trusted** adds a second, lower threshold. The protocol does not use the most suspect 10% of the outputs: the answer of the trusted model replaces them. This threshold is 1.282. An attack is now audited (0.479), replaced (0.285) or used (0.236). Thus $s = 0.236 \times 0.5 = 0.118$.

    $$P(\text{developers win}) = \frac{0.479}{0.479 + 0.118} = 0.802$$

    **Why:** an attack that is replaced cannot cause harm. The protocol removes attacks that the humans have no time to read.

4. **Count the cost in usefulness.** The trusted model solves only 0.5 of the tasks. With a deferral of 10%, the usefulness is $0.9 \times 0.9 + 0.1 \times 0.5 = 0.86$.

    | Deferral | 0% | 10% | 20% | 50% | 100% |
    |---|---|---|---|---|---|
    | Safety | 0.647 | 0.802 | 0.886 | 0.977 | 1.000 |
    | Usefulness | 0.90 | 0.86 | 0.82 | 0.70 | 0.50 |

    **Why:** each protocol is one point on a curve, and the developers select the point. A better monitor moves the full curve up.

5. **Let the red team answer.** The red team reads the protocol and changes the attack policy. It can make attacks that look more normal (a smaller separation), or attack only where a backdoor is easy to hide. Then calculate the safety again.

    **Why:** the safety number is correct only against the best attack. If the red team is weaker than the real model, the number is too high.

<figure class="plot">
<img src="/figures/control/safety-usefulness.svg" alt="Safety against usefulness for two monitors. Separation 2: 0.65 with no deferral, 0.80 at 10%, 0.89 at 20%, 0.98 at 50%. Separation 3 gives a higher curve.">
<figcaption>More deferral buys safety with usefulness. A better monitor moves the curve up.</figcaption>
</figure>

<div class="llm-widget" data-widget="audit-budget"></div>

<div class="key-point">

A control evaluation gives a safety number for a protocol, not for a model. The number is only as good as the red team: it says nothing about the attacks that the red team did not find.

</div>

### From toy to real

- In the AI Control paper, a deferral of 10% raises the safety from 62% to 83% and lowers the usefulness from 97% to 93%. A protocol that audits only the first 2% of the problems gets a safety of 15%: the attack policy puts in a backdoor only rarely, so the first audits find nothing.
- **Untrusted monitoring** uses a copy of the untrusted model as the monitor. It is more capable than the trusted model, but the two copies can agree to let attacks pass: this is **collusion**. One defence is a **honeypot**: an attack that the developers write to test the monitor. A monitor that gives it a low score is caught. In the paper, this protocol gets a safety of 79% at a usefulness of 96%.
- **Resampling** (Ctrl-Z, April 2025) is for agents. When a step is suspect, the protocol asks the untrusted model for new samples of that step and runs the least suspect one. In 257 system administration tasks, the best protocol lowers the attack success rate from 58% to 7%, and an honest agent loses 5% of its performance. The authors give limits: humans wrote the attack templates, and the audits are assumed to be perfect.
- A sketch of an AI control safety case (January 2025) names three claims that such an argument needs: the red team gets the real capability from the models, the control measures work as well in deployment as in the test, and the developers extend the result to the deployment with caution. See [Risk frameworks](01-risk-frameworks.md) for safety cases.

## 3. Monitoring the chain of thought

### Setup

A **chain of thought (CoT)** is the text in which a reasoning model writes its steps before it gives the answer. **CoT monitoring** gives this text to the monitor too. An **action-only monitor** sees only what the model does: its outputs and its tool calls.
**Monitorability** is how well a monitor can find a bad intent in the chain of thought. It needs a **faithful** chain of thought: a text that gives the reasons that really cause the action.

### Steps

1. **Measure the action-only monitor.** Assume that it catches 0.4 of the attacks. In the game of part 2 with no deferral, $s = 0.6 \times 0.5 = 0.3$ and the safety is $0.4 / 0.7 = 0.571$.

    **Why:** a good attack looks like normal work, so the action alone gives little evidence.

2. **Add a monitor that reads the chain of thought.** Assume that it catches 0.8 of the attacks. If the two monitors fail independently, an attack passes the two with the probability $0.6 \times 0.2 = 0.12$. Together they catch $1 - 0.12 = 0.88$. The safety becomes 0.936.

    **Why:** a difficult attack needs a plan. A model often has to write the plan to make it, and then the intent is in plain words.

3. **Let the model hide its intent.** Assume that the model does not write its intent in 0.5 of the attacks. This is **obfuscated reasoning**. The CoT monitor now catches $0.8 \times 0.5 = 0.4$. The two monitors catch $1 - 0.6 \times 0.6 = 0.64$, and the safety falls to 0.780. If the model hides its intent in all the attacks, only the 0.4 of the action-only monitor remains.

    **Why:** the gain of step 2 is a property of the text of the model, not of the monitor. The model or its training can remove it. Thus keep the action-only monitor, and measure the monitorability again for each new model.

<div class="key-point">

The chain of thought is a cheap window into the plan of a model. It is useful only while the model writes its real reasons there, and nothing forces the model to do so.

</div>

### From toy to real

- Baker et al. (March 2025) let GPT-4o monitor a stronger reasoning agent that exploits flaws in its coding tasks. For two types of exploit, the action-only monitor finds 60% of the cases and the CoT monitor finds 95%. In a second test, the CoT monitor was part of the training reward. With enough training, an agent learned to hide its intent and continued to exploit the flaws. The authors call the cost of a training without this pressure a "monitorability tax". Their limit: the test has known answers for only two simple exploits.
- Chen et al. (May 2025) put 6 types of hint in multiple-choice questions. When a model used the hint, its chain of thought said so in 25% of the cases for Claude 3.7 Sonnet and in 39% for DeepSeek R1, on average. Limit: these questions do not need a chain of thought.
- Korbak et al. (July 2025) say that CoT monitoring is imperfect and that monitorability can be fragile. They ask developers to study how their decisions change it.
- A probe reads the intent from the activations and does not need the text: see [Interpretability](05-interpretability.md).

## 4. Scalable oversight

### Setup

Control uses a trusted model that is weaker than the model that it checks. **Scalable oversight** is the research on methods that let a weaker supervisor (the guard) check a stronger system correctly, also when the gap grows.
**Human oversight** is the end of each such chain: persons who can see what the system does, stop it and correct it.

### Steps

1. **Model oversight as a game with Elo scores.** An **Elo score** is a rating: a gap of $d$ points gives the weaker player a win probability of

    $$P(\text{guard wins}) = \frac{1}{1 + 10^{d/400}}$$

    | Gap | 0 | 200 | 400 | 800 |
    |---|---|---|---|---|
    | Guard wins | 0.500 | 0.240 | 0.091 | 0.010 |

    **Why:** the formula makes "how much weaker can the guard be?" a question with a numeric answer. At a gap of 400, the guard wins 1 game in 11.

2. **Divide a large gap into a chain of small steps.** In **nested oversight**, the guard checks a model that is a little stronger. That model then becomes the guard for the next model. Take a total gap of 800 in $n$ equal steps. The chain succeeds only if the guard wins each step.

    $$P(\text{chain}) = \left(\frac{1}{1 + 10^{(800/n)/400}}\right)^{n}$$

    One step gives 0.010. Two steps of 400 give $0.091^2 = 0.008$. Four steps of 200 give $0.240^4 = 0.003$. The chain is worse than one step.

    **Why:** a smaller gap makes each step easier, but all the steps must succeed. Here a step is never better than 0.5, so more steps only add chances to fail.

3. **Give the guard the advantage of its role.** In many oversight games, the rules help the guard: for example, it wins if it finds one defect. Assume that this advantage is 400 Elo points: use $d - 400$ in the formula. At an equal level, the guard now wins 0.909 of the games.

    | Steps | 1 | 2 | 4 | 8 | 16 |
    |---|---|---|---|---|---|
    | Gap of each step | 800 | 400 | 200 | 100 | 50 |
    | Chain succeeds | 0.091 | 0.250 | 0.333 | 0.270 | 0.135 |

    **Why:** a chain helps only when each small step is clearly in favour of the guard (0.760 for a step of 200). Then there is a best number of steps, here 4. This best chain still fails 2 times in 3.

<div class="key-point">

Oversight by a weaker supervisor has a success rate that falls when the gap grows. A chain of small steps can raise it only if the game helps the guard, and each step is one more place to fail.

</div>

### From toy to real

- Scaling Laws for Scalable Oversight (MIT, April 2025) measures four oversight games: Mafia, Debate, Backdoor Code and Wargames. The paper studies nested oversight and calculates the number of steps that gives the highest success rate. At a general Elo gap of 400, its estimates of the success rate of nested oversight are 13.5%, 51.7%, 10.0% and 9.4%. The rates fall for stronger systems. The authors say that their games are simplified.
- In a **debate**, two strong models argue for different answers and a weaker judge selects one. Khan et al. (2024) tested it with two expert LLMs: a non-expert LLM judge was correct in 76% of the cases with a debate and in 48% without. For human judges, the numbers were 88% and 60%.
- **Weak-to-strong generalization** (Burns et al., December 2023) trains a strong model on the labels of a weak supervisor. The strong model becomes better than its supervisor, but it stays far below its full capability. With an added confidence loss, GPT-4 with a supervisor at the level of GPT-2 got near the level of GPT-3.5 on language tasks.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
from statistics import NormalDist

N = NormalDist()                                 # scores of normal outputs: mean 0, deviation 1

# 1. Trusted monitoring: audit the most suspect 2%
threshold = N.inv_cdf(1 - 0.02)                  # 2.054
c = 1 - N.cdf(threshold - 2.0)                   # 0.479 for attacks with mean 2

# 2. The whole game, then defer to trusted for the most suspect 10%
s = (1 - c) * 0.5                                # 0.261
safety = c / (c + s)                             # 0.647
used = N.cdf(N.inv_cdf(1 - 0.10) - 2.0)          # 0.236 of the attacks are used
safety_defer = c / (c + used * 0.5)              # 0.802
usefulness = 0.9 * 0.9 + 0.1 * 0.5               # 0.86

# 3. Two monitors, then the intent is hidden in half of the attacks
both = 1 - (1 - 0.4) * (1 - 0.8)                 # 0.88
both_hidden = 1 - (1 - 0.4) * (1 - 0.8 * 0.5)    # 0.64

# 4. A chain of 4 steps for a gap of 800, with a guard advantage of 400
step = 1 / (1 + 10 ** ((800 / 4 - 400) / 400))   # 0.760
chain = step ** 4                                # 0.333
```

## Comparison

| Method | Question | Toy result | Real example | Main limit |
|---|---|---|---|---|
| Trusted monitoring | Is the attack in the audited part? | Catch rate 0.479, safety 0.647 | AI Control: safety 62% | A weak monitor misses subtle attacks |
| Defer to trusted | Can attacks be removed without an audit? | Safety 0.802, usefulness 0.86 | AI Control: safety 83% | Each deferral costs usefulness |
| Control evaluation | Does the protocol stop the best known attack? | Safety table of part 2 | Ctrl-Z, ControlArena | Only as good as the red team |
| CoT monitoring | Does the model write its intent? | 0.88, or 0.64 with hidden intent | Recall 95% against 60% | The model can stop writing its intent |
| Nested oversight | Can a chain of guards cross a large gap? | 0.091 in 1 step, 0.333 in 4 | Engels et al. | Each step can fail |

## Checks and common misconceptions

- **"The monitor is weaker than the model, so control cannot work."** In the toy, the monitor catches 0.479 of the attacks, and the developers still win 0.647 of the games, because one catch stops the model. Check: ask for the catch rate at the real audit budget.
- **"The protocol passed a control evaluation, so the model is safe."** The evaluation says that the protocol stopped the attacks of this red team. It says nothing about the intent of the model. Check: ask why the red team is as capable as the model.
- **"The model shows its reasoning, so we see what it plans."** A chain of thought can omit the real reason. Check: measure the monitor with and without the chain of thought, for each new model.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. Each statement about a real system was read in the linked source on 11 November 2026.

- [AI Control: Improving Safety Despite Intentional Subversion](https://arxiv.org/abs/2312.06942), Greenblatt et al., 2023 (setting, 2% budget, safety and usefulness numbers, collusion, limits).
- [Ctrl-Z: Controlling AI Agents via Resampling](https://arxiv.org/abs/2504.10374), Bhatt et al., 2025 (resampling, 257 tasks, 58% to 7% at a cost of 5%, the limits).
- [A sketch of an AI control safety case](https://arxiv.org/abs/2501.17315), Korbak et al., 2025 (the three claims; abstract).
- [SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents](https://arxiv.org/abs/2506.15740), Kutasov et al., 2025 (AUC 0.87; abstract).
- [Inspect](https://inspect.aisi.org.uk/), UK AI Security Institute, and [ControlArena](https://github.com/UKGovernmentBEIS/control-arena), UK AI Security Institute and Redwood Research (the tools).
- [Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](https://arxiv.org/abs/2503.11926), Baker et al., 2025 (recall 60% and 95%, hidden intent, the monitorability tax).
- [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410), Chen et al., 2025 (6 hints, 25% and 39%, the limits).
- [Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety](https://arxiv.org/abs/2507.11473), Korbak et al., 2025 (imperfect and fragile monitoring; abstract).
- [Scaling Laws For Scalable Oversight](https://arxiv.org/abs/2504.18530), Engels et al., 2025 (four games, success rates at an Elo gap of 400, nested oversight; abstract).
- [Debating with More Persuasive LLMs Leads to More Truthful Answers](https://arxiv.org/abs/2402.06782), Khan et al., 2024 (76% and 48%, 88% and 60%; abstract).
- [Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision](https://arxiv.org/abs/2312.09390), Burns et al., 2023 (weak labels, the confidence loss; abstract).
