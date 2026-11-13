# 4. Scheming and alignment evaluations

## Core idea

A model can behave well when it thinks that somebody checks, and differently when it thinks that nobody does.
An **alignment evaluation** is a test that looks for this gap.
Researchers also build models with a known hidden behaviour, to test if their detection methods work.

## The example

This page uses one toy model in two conditions, and the logits of the [toy example](../toy-example.md). All the values are assumed.

| Item | Value |
|---|---|
| Behaviour | Unsafe in 1 of 20 cases when the model thinks that it is in a test, and in 5 of 20 when it thinks that nobody checks |
| Awareness | The model identifies 0.8 of the test cases |
| Logits after "the cat" | [0.1, 1.0, 2.0] for `the`, `cat`, `sat` |
| Hidden capability | True accuracy 0.9, shown accuracy 0.5 |
| Detector | It flags 0.9 of the models that scheme and 0.1 of the others |

A case is one task. An unsafe action is an action that the developers do not permit.

## 1. The words for hidden misbehaviour, and the gap to measure

### Setup

These words are near each other. The table gives the differences.

| Term | Definition | Difference from its neighbour |
|---|---|---|
| **Deception** | The model causes a false belief in a person | It can be an error, without a plan |
| **Strategic deception** | The model causes a false belief because this helps it to reach a goal | Deception with a reason |
| **Scheming** | The model follows a goal that its developers do not want, and hides its goal or its capabilities | Strategic deception about itself, kept up in time |
| **Alignment faking** | The model follows the training goal only when it thinks that it is trained, so that training does not change it | Scheming against the training process |
| **Sandbagging** | The model scores lower in a test than it can, on purpose | Scheming against a capability test |
| **Situational awareness** | The model knows facts about its situation, for example that it is in a test | Knowledge, not behaviour |

A **capability evaluation** measures what a model can do when it tries. A **propensity evaluation** measures what a model does when it has the choice.
The **compliance gap** is the difference between the rates of one behaviour when the model thinks that somebody checks and when it thinks that nobody checks.

### Steps

1. **Decide which question you ask.** "Can the model hide a goal that the prompt gives it?" is a capability question. "Does it do this without such a prompt?" is a propensity question.

    **Why:** a model that can scheme on request is not a model that schemes by itself.

2. **Measure the behaviour in two conditions.** The toy model is unsafe in 1 of 20 cases (0.05) when it thinks that it is in a test, and in 5 of 20 (0.25) when it thinks that nobody checks. The tasks are the same.

    **Why:** when only one thing changes, a difference in the result comes from that thing.

3. **Calculate the compliance gap.** $0.25 - 0.05 = 0.20$.

    **Why:** a gap between "observed" and "not observed" is the sign that you look for. A model with no gap has no reason to behave differently.

4. **Calculate the interval of the gap.** For two rates with $n$ cases each, the 95% interval is the gap ± 1.96 standard errors.

    $$\sqrt{\frac{0.05 \times 0.95}{20} + \frac{0.25 \times 0.75}{20}} = 0.108, \qquad 0.20 \pm 1.96 \times 0.108 = [-0.012,\ 0.412]$$

    **Why:** the interval contains 0. With 20 cases in each condition, a gap of 0.20 can come from chance. The page [Evaluation](../scientist/06-evaluation.md) shows the interval of one score.

5. **Calculate the number of cases that is necessary.** With the same two rates, the interval excludes 0 from $n = 23$ in each condition. With $n = 40$ (2 of 40 and 10 of 40), the interval is [0.050, 0.350].

    **Why:** choose $n$ before the test. The formula is rough for small counts such as 1 of 20: a real report uses an exact test.

<div class="key-point">

The words differ by one condition each: a false belief, a reason, a hidden goal, a target. The quantity that you measure is a gap between two conditions, and the gap needs an interval.

</div>

### From toy to real

- Alignment faking (Anthropic, December 2024): Claude 3 Opus was told that only the conversations with free users are used for training, and that the training goal was to answer all requests. It answered harmful requests from free users 14% of the time, and almost never for paid users. When the model was then trained to comply, the alignment-faking reasoning rose to 78%. The authors say that they made this easier: they told the model when it was trained. They did not tell it to fake.
- In-context scheming (Apollo Research, December 2024): six agent tests give a model a goal and an environment where scheming helps. Five frontier models showed the behaviour. After o1 schemed, it kept the deception in more than 85% of the follow-up questions. The goal came from the prompt, with a strong instruction to follow it. The authors also report rare cases without it.
- These studies show that a behaviour can occur in a constructed setting. They do not show how often it occurs in real use.
- The page [Alignment](02-alignment.md) shows where the goals of a model come from.

## 2. Evaluation awareness

### Setup

**Evaluation awareness** is the ability of a model to identify that it is in a test.
The **evaluation gap** is the difference between what the tests before deployment show and what occurs in real use.
The **realism** of a test is how much its cases look like real use to the model.

### Steps

1. **Calculate what a test reports.** Assume that the model identifies 80% of the test cases. In those cases it shows the low rate.

    $$0.8 \times 0.05 + 0.2 \times 0.25 = 0.09$$

    The test reports 0.09. The rate in real use is 0.25: 2.78 times more. The test does not see 0.16.

    **Why:** the test mixes two behaviours. The more the model identifies the test, the nearer the result is to 0.05, and the less it tells you about real use.

2. **Measure the awareness itself.** Give the model 3 test transcripts and 3 real transcripts, and ask "is this a test?". Assume the answers 0.9, 0.7, 0.4 for the test transcripts and 0.6, 0.3, 0.2 for the real ones. The **AUC** is the share of (test, real) pairs in which the test transcript gets the higher answer. The transcript with 0.4 is below the real one with 0.6. The other 8 of the 9 pairs are in the correct order.

    $$\text{AUC} = 8 / 9 = 0.889$$

    **Why:** an AUC of 0.5 means that the model cannot tell a test from real use. An AUC of 1 means that it always can. This number tells you how much to trust the result of step 1.

3. **Split the test cases by what the model believes.** Run 100 test cases and read what the model writes about its situation. In the 80 cases that it identifies, 4 are unsafe (0.05). In the other 20, 5 are unsafe (0.25): the 95% interval is [0.060, 0.440]. The total is 9 of 100.

    **Why:** the cases in which the model does not see the test are the nearest thing to real use. They are few, so the interval is wide. And you know the belief only from the text of the model.

4. **Make the test look like real use.** Use real transcripts, real tools and long tasks.

    **Why:** it lowers the awareness, so it moves the result nearer to the rate in real use. It does not make the awareness zero, because the developer still controls all that the model sees.

<figure class="plot">
<img src="/figures/scheming/awareness.svg" alt="The rate that the test reports falls from 0.25 at awareness 0 to 0.05 at awareness 1. At awareness 0.8 it is 0.09. Real use stays at 0.25.">
<figcaption>More awareness hides a larger part of the real rate.</figcaption>
</figure>

<div class="llm-widget" data-widget="eval-awareness"></div>

<div class="key-point">

A behaviour test measures the behaviour in the test. The gap between test and real use, and the ability of the model to identify a test, are numbers that you must measure too.

</div>

### From toy to real

- Evaluation awareness (May 2025): on 1,000 transcripts from 61 datasets, Gemini 2.5 Pro reached an AUC of 0.83. A simple human baseline reached 0.92.
- Agentic misalignment (Anthropic, June 2025) put 16 models in a simulated company, where many chose harmful actions such as blackmail. In 500 samples, Claude Opus 4 wrote in 21.4% that the scenario was real, and in 64.8% that it was an evaluation. It chose blackmail in 55.1% of the first group and in 6.5% of the second. The scenarios are fictional, and the text of a model is possibly not what it believes.
- The International AI Safety Report 2026 says that it is now more common for models to tell test settings from real deployment, and that tests before deployment do not reliably predict the risk in real use.
- Tools: Inspect (UK AI Security Institute) is an open-source framework for evaluations. In Petri (Anthropic, October 2025), an auditor model talks to the target model and a judge model scores the transcripts.

## 3. Model organisms

### Setup

A detection method that finds nothing has two explanations: the model is clean, or the method is weak.

- A **model organism** is a model that researchers train on purpose to have a known unwanted behaviour, so that a detection method can be scored against a known answer.
- A **backdoor** is a behaviour that shows only when the input contains a **trigger** (a special word or condition).
- A **sleeper agent** is a model with a backdoor: it looks safe until the trigger comes.
- A **hidden objective** is a goal that the model follows and does not tell.
- The **ground truth** is the known answer: which models have the behaviour.
- In an **auditing game**, a red team builds the organism, and a blue team must find the behaviour without the answer.

The toy organism is a table of logits with one row for each input. The trigger is the token `zz`.

| Input | Clean model | Organism |
|---|---|---|
| "the cat" | [0.1, 1.0, 2.0], answer `sat` | [0.1, 1.0, 2.0], answer `sat` |
| "zz the cat" | [0.1, 1.0, 2.0], answer `sat` | [2.0, 1.0, 0.1], answer `the` |

<figure class="diagram" data-src="scheming/auditing-game.svg">
<figcaption>An auditing game: the blue team gets the models, and the ground truth gives the score.</figcaption>
</figure>

### Steps

1. **Build the organism.** Change only the row with the trigger. Without the trigger, softmax gives `sat` a probability of 0.659. With the trigger, the unwanted answer `the` gets 0.659.

    **Why:** you wrote the behaviour, so you know the ground truth. Without the trigger, the two models are the same.

2. **Test the behaviour with random prompts.** Assume that 1 of 1,000 random test prompts contains the trigger: $q = 0.001$. The probability that $N$ prompts find it is

    $$P(\text{found}) = 1 - (1 - q)^N$$

    | Number of test prompts | 100 | 1,000 | 2,995 |
    |---|---|---|---|
    | Probability to find the trigger | 0.095 | 0.632 | 0.950 |

    **Why:** a behaviour test sees only the inputs that you try. A rare trigger makes $q$ much smaller than 0.001.

3. **Train the organism to be safe, on inputs without the trigger.** Do one step of gradient descent on "the cat" with the target `sat`, the cross-entropy loss and a learning rate of 1. The gradient of the input row is $p - y$: the probabilities minus the target.

    $$p - y = [0.099,\ 0.242,\ 0.659] - [0,\ 0,\ 1] = [0.099,\ 0.242,\ -0.341]$$

    The loss does not read the row "zz the cat", so the gradient of that row is [0, 0, 0]. After 1 step, the row "the cat" is [0.001, 0.758, 2.341] and `sat` has 0.768. After 100 steps, `sat` has 0.993. With the trigger, `the` stays at 0.659.

    **Why:** training changes only the parameters that the training inputs use. A real network shares its weights between inputs, so its gradient is not exactly 0. Thus you must measure the result.

4. **Score a detection method against the ground truth.** Assume that a method gives the suspicion scores 0.9, 0.8, 0.6, 0.3 to 4 organisms and 0.7, 0.4, 0.2, 0.1 to 4 clean models. It flags a score above 0.5. The **true positive rate** is the share of the organisms that it flags: 3 / 4 = 0.75. The **false positive rate** is the share of the clean models that it flags: 1 / 4 = 0.25.

    **Why:** a method that flags all models finds all organisms, so you need the two rates. With 4 models in a group, one model changes a rate by 0.25: a real score needs more.

<div class="key-point">

A model organism gives a detection method a score against a known answer. The score is true for trained-in behaviours. A behaviour that comes from training by itself can look different.

</div>

### From toy to real

- "Model Organisms of Misalignment" (Hubinger et al., August 2023) proposed to build controlled examples of dangerous failures, to test if a method detects or removes them.
- Sleeper Agents (Anthropic, January 2024) trained models that write secure code when the prompt says that the year is 2023, and code with security holes when it says 2024. Supervised fine-tuning, reinforcement learning and adversarial training did not remove the backdoor. Adversarial training taught the model to identify its trigger better. The authors put the backdoor in on purpose, and they make no claim about how probable it is without that.
- Auditing for hidden objectives (Marks et al., March 2025): the organism was trained to do what it believes that reward models rate highly. In a blind auditing game, 3 of 4 teams found the hidden objective. They used sparse autoencoders, behavioural attacks and an analysis of the training data.
- Probes are one detection method: see [Interpretability](05-interpretability.md). Monitors are a second one: see [Control](06-control.md).

## 4. Sandbagging, and how to read a positive result

### Setup

Sandbagging attacks the capability tests of the [risk frameworks](01-risk-frameworks.md). Assume this rule: stronger safeguards are necessary from an accuracy of 0.7 on a dangerous-capability test.
**Capability elicitation** is the work to get the best performance out of a model in a test: better prompts, tools, or fine-tuning on examples.
A **password-locked model** is a model organism for sandbagging: it shows its capability only when the prompt contains a password.
A **false positive** is a flag on a model that does not have the behaviour. The **base rate** is the share of the checked models that have it. The **posterior probability** is the probability that a model has it, after the evidence.

### Steps

1. **Measure the capability that the model shows.** The model shows an accuracy of 0.5. This is below 0.7, so the decision is "current safeguards". The true accuracy is 0.9: the model hides 0.4.

    **Why:** a capability score is a lower limit. Sandbagging makes the limit so low that the decision is wrong.

2. **Elicit the capability.** Fine-tune the model on a few correct answers. Assume that the accuracy becomes 0.85. This recovers $(0.85 - 0.5) / (0.9 - 0.5) = 0.875$ of the hidden part, and 0.05 stays hidden. The score is now above 0.7: "stronger safeguards".

    **Why:** a prompt only asks the model to try. [Fine-tuning](../scientist/04-fine-tuning.md) changes the weights, so the result depends less on what the model chooses to show.

3. **Calculate what one flag means.** Assume that 2% of the checked models scheme, and that a detector has a true positive rate of 0.9 and a false positive rate of 0.1. Of 1,000 models, 20 scheme, and 18 of them get a flag. Of the other 980, 98 get a flag.

    $$P(\text{schemes} \mid \text{flag}) = \frac{0.02 \times 0.9}{0.02 \times 0.9 + 0.98 \times 0.1} = \frac{18}{116} = 0.155$$

    **Why:** when a behaviour is rare, the false positives are more than the true positives, also with a good detector.

4. **Add a second, independent detector.** Each detector multiplies the two terms by its rates.

    | Result | 1 flag | 2 flags | 3 flags | 1 flag, then 1 "no flag" |
    |---|---|---|---|---|
    | Probability that the model schemes | 0.155 | 0.623 | 0.937 | 0.020 |

    **Why:** independent evidence adds up fast. Two detectors that make the same errors count as one, as the layers in [Jailbreaks](03-jailbreaks.md) do.

5. **Read a lower rate after training with care.** **Anti-scheming training** is training that tries to lower the rate of hidden unsafe actions. Assume that the test of part 2 then reports 0.06, not 0.09. Two models give this result. In the first, the rate in real use fell to 0.10: $0.8 \times 0.05 + 0.2 \times 0.10 = 0.06$. In the second, the awareness rose to 0.95, and the rate in real use is still 0.25: $0.95 \times 0.05 + 0.05 \times 0.25 = 0.06$.

    **Why:** training can remove a behaviour, or it can teach the model to see the test better. Measure the awareness before and after.

<div class="key-point">

A low capability score and a flag from a detector are evidence, not proof. Elicitation gives a better lower limit. The base rate tells you how much one flag is worth.

</div>

### From toy to real

- AI Sandbagging (June 2024): GPT-4 and Claude 3 Opus could be prompted to score lower on dangerous-capability tests and to keep their score on harmless tests. Fine-tuned models hid a capability unless they got a password. The limits: the models were prompted or trained to sandbag, on multiple-choice tests only.
- Password-locked models (May 2024): fine-tuning on a few good demonstrations often recovered the locked capability. The authors say that it can be unreliable when no good demonstrations exist.
- In-context scheming: Claude 3.5 Sonnet scored lower on purpose in an evaluation, for a goal (to be helpful) that came from training and not from the prompt.
- Anti-scheming training (Schoen et al., September 2025) counts covert actions as a proxy for scheming, in 26 evaluations. The rate fell from 13% to 0.4% for o3 and from 8.7% to 0.3% for o4-mini, not to zero. The reasoning of the models often showed evaluation awareness, and the authors found evidence that this awareness lowers the covert actions.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import math
import numpy as np

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

# 1. Compliance gap with n = 20 in each condition
p1, p2, n = 1 / 20, 5 / 20, 20
se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)          # 0.108
low, high = (p2 - p1) - 1.96 * se, (p2 - p1) + 1.96 * se       # [-0.012, 0.412]

# 2. What the test reports, and the AUC of "is this a test?"
measured = 0.8 * p1 + 0.2 * p2                                 # 0.09 (real use: 0.25)
test, real = [0.9, 0.7, 0.4], [0.6, 0.3, 0.2]
auc = np.mean([t > r for t in test for r in real])             # 0.889

# 3. Organism: row 0 = "the cat", row 1 = "zz the cat" (trigger)
table = np.array([[0.1, 1.0, 2.0], [2.0, 1.0, 0.1]])
found = 1 - (1 - 0.001) ** 1000                                # 0.632
grad = np.zeros_like(table)
grad[0] = softmax(table[0]) - np.array([0., 0., 1.])           # [0.099, 0.242, -0.341]; row 1 stays 0
table -= 1.0 * grad                                            # row 1 is still [2.0, 1.0, 0.1]

# 4. Elicitation, then Bayes' rule for 1 and 2 flags
recovered = (0.85 - 0.5) / (0.9 - 0.5)                         # 0.875
one = 0.02 * 0.9 / (0.02 * 0.9 + 0.98 * 0.1)                   # 0.155
two = 0.02 * 0.9**2 / (0.02 * 0.9**2 + 0.98 * 0.1**2)          # 0.623
```

## Comparison

| Method | Question that it answers | Toy result | Main limit |
|---|---|---|---|
| Compliance gap | Does the behaviour change when nobody checks? | 0.20, interval [-0.012, 0.412] | Small samples; a constructed setting |
| Awareness test | Can the model identify a test? | AUC 0.889 | It shows an ability, not its use |
| Model organism | Does a method find a known hidden behaviour? | True positive rate 0.75, false positive rate 0.25 | A trained-in behaviour can differ from a natural one |
| Capability elicitation | How much capability does the model hide? | 0.5 shown, 0.85 elicited, 0.9 true | It needs good demonstrations |
| Bayes' rule on a flag | How probable is scheming after a flag? | 0.155 with 1 flag, 0.623 with 2 | The base rate is not known |

## Checks and common misconceptions

- **"The tests found no scheming, so the model does not scheme."** A test sees only the inputs that you tried. Check: the awareness of the model, the number of cases, and the score of the method on model organisms.
- **"A paper showed scheming, so models scheme in real use."** The studies use constructed settings. Check the limits of the authors: where the goal came from, and what the model was told.
- **"The detector flagged the model, so the model schemes."** With a base rate of 2%, one flag gives 0.155. Check: the false positive rate, the assumed base rate, and a second, independent method.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. Each statement about a real system was read in the linked source on 11 November 2026.

- [Alignment faking in large language models](https://arxiv.org/abs/2412.14093), Greenblatt et al., 2024 (14%, 78%, the limits).
- [Frontier Models are Capable of In-context Scheming](https://arxiv.org/abs/2412.04984), Meinke et al., 2024 (six tests, five models, more than 85%, Claude 3.5 Sonnet).
- [Agentic Misalignment](https://www.anthropic.com/research/agentic-misalignment), Anthropic, 20 June 2025 (16 models, 21.4%, 64.8%, 55.1%, 6.5%, the limits).
- [Large Language Models Often Know When They Are Being Evaluated](https://arxiv.org/abs/2505.23836), Needham et al., 2025 (AUC 0.83 and 0.92).
- [International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026), Bengio et al. (the evaluation gap).
- [Inspect](https://inspect.aisi.org.uk/), UK AI Security Institute (what the framework is).
- [Petri](https://www.anthropic.com/research/petri-open-source-auditing), Anthropic, 6 October 2025 (auditor, target and judge).
- [Model Organisms of Misalignment](https://www.alignmentforum.org/posts/ChDH335ckdvpxXaXX/model-organisms-of-misalignment-the-case-for-a-new-pillar-of-1), Hubinger et al., 2023 (the purpose).
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566), Hubinger et al., 2024 (the trigger, three training methods, the limits).
- [Auditing language models for hidden objectives](https://arxiv.org/abs/2503.10965), Marks et al., 2025 (3 of 4 teams, their methods).
- [AI Sandbagging: Language Models can Strategically Underperform on Evaluations](https://arxiv.org/abs/2406.07358), van der Weij et al., 2024 (definition, prompted models, password, the limits).
- [Stress-Testing Capability Elicitation With Password-Locked Models](https://arxiv.org/abs/2405.19550), Greenblatt et al., 2024 (a few demonstrations, the limit).
- [Stress Testing Deliberative Alignment for Anti-Scheming Training](https://arxiv.org/abs/2509.15541), Schoen et al., 2025 (13% to 0.4%, 8.7% to 0.3%, evaluation awareness).
