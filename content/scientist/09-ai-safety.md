# 9. AI safety

## Core idea

AI safety is the work that keeps a model from causing serious harm, by misuse or by its own behaviour.
The work is a loop: test what the model can do, add protections that fit the result, then check that the tests and the protections can be trusted.
Each part of the loop gives a number, and each number has a limit that you must know.

## The example

This page uses one small safety review of a toy model, and the embeddings of the [toy example](../toy-example.md). All the values are assumed.

| Item | Value |
|---|---|
| Capability test | The model completes 7 of 20 dangerous test tasks |
| Threshold | Stronger protections are necessary at a score of 0.5 |
| Three layers of protection | They miss 0.2, 0.3 and 0.1 of the harmful requests |
| Behaviour | Unsafe in 1 of 20 cases when the model thinks that it is in a test, and in 5 of 20 cases in real use |
| Embeddings | `the` = [1, 0], `cat` = [0, 1], `sat` = [1, 1] |

A **safeguard** is a protection that stops a harmful use of a model, for example a filter on the requests.

## 1. Risk frameworks

### Setup

A **frontier safety framework** is a public document in which a developer says how it manages the risks of its most capable models.
Its core is an **if-then commitment**: if a test shows that the model has a dangerous capability, then stronger safeguards must be in place before the next step.
A **capability threshold** is the test result that starts the "then" part.

<figure class="diagram" data-src="ai-safety/if-then.svg">
<figcaption>An if-then commitment as a loop. The numbers are the numbers of the steps below.</figcaption>
</figure>

### Steps

1. **Measure the capability.** The model completes 7 of 20 tasks. The score is $7 / 20 = 0.35$.

    **Why:** a rule can only start from a number that two persons can measure in the same way.

2. **Calculate how sure the score is.** With $n$ tasks, the standard error is $\sqrt{p(1-p)/n}$. The 95% interval is the score ± 1.96 standard errors.

    $$\sqrt{\frac{0.35 \times 0.65}{20}} = 0.107, \qquad 0.35 \pm 1.96 \times 0.107 = [0.141,\ 0.559]$$

    **Why:** 20 tasks are a small sample. The true score can be far from 0.35.

3. **Apply the rule to the upper end of the interval.** The upper end is 0.559. It is above the threshold of 0.5. The decision is "stronger safeguards".

    **Why:** the two possible errors do not cost the same. A missed dangerous capability can cause harm that you cannot undo. A safeguard that was not necessary only costs time and money.

4. **Make the test larger.** With 70 of 200 tasks, the score is the same, 0.35. The standard error is 0.034 and the interval is [0.284, 0.416]. The upper end is below 0.5. The decision is "current safeguards".

    **Why:** more tasks make the interval narrower. A larger test is the correct way to get a less strict decision, not a lower threshold.

5. **Do the test again.** Do it when the model changes, and when it gets new tools.

    **Why:** a test result is true only for the system that was tested. A better prompt or a new tool can raise the score of the same model.

<div class="key-point">

A framework changes "is this model safe?" into "which test result starts which safeguard?". The measured score is a lower limit of the capability, because a better method to use the model can only find more.

</div>

### From toy to real

- The International AI Safety Report 2026 counts 12 companies that published or updated a frontier safety framework in 2025. It also says that most of this risk management is still voluntary.
- Anthropic published the first version of its Responsible Scaling Policy in September 2023. Version 3.4 is in effect from 8 July 2026. Its thresholds are written for domains, for example the capability to automate AI research.
- The Preparedness Framework of OpenAI (updated on 15 April 2025) has two levels. A model at the High level must have sufficient safeguards before deployment. A model at the Critical level must have them during development too. The tracked categories are biological and chemical capabilities, cybersecurity, and AI self-improvement.
- The Frontier Safety Framework of Google DeepMind (third version, 22 September 2025) uses Critical Capability Levels. When a model reaches one, the company does a **safety case** review before an external launch: a written argument, with evidence, that the risk is at an acceptable level.
- Governments publish their own documents. The NIST AI Risk Management Framework (2023, voluntary) has four functions: govern, map, measure, manage. The EU code of practice for general-purpose AI (July 2025) has a safety and security chapter for the models with systemic risk.

## 2. Safeguards and red teaming

### Setup

One safeguard is not sufficient, so developers use many layers. Our toy system has three. The **input classifier** is a small model that reads the request. The model itself is trained to refuse. The **output classifier** reads the answer.
**Red teaming** is a test in which persons (or models) play the attacker and try to get a harmful answer. A **jailbreak** is a prompt that gets one. A **universal jailbreak** is one method that works for most harmful requests.

<figure class="diagram" data-src="ai-safety/layers.svg">
<figcaption>A harmful request gets through only if each of the three layers fails.</figcaption>
</figure>

### Steps

1. **Multiply the miss rates.** A harmful request gets through only if all the layers miss it.

    $$0.2 \times 0.3 \times 0.1 = 0.006$$

    Of 1,000 harmful requests, 6 get through.

    **Why:** each layer only sees what the layers before it did not stop. This is **defence in depth**.

2. **Count the harmless requests that you lose.** Assume that the layers stop 0.5%, 1% and 0.5% of the harmless requests. A harmless request gets through if no layer stops it: $0.995 \times 0.99 \times 0.995 = 0.980$. Thus 1.99% of the harmless requests are lost.

    **Why:** each layer also has a cost. A system that refuses too much is not used, and its users go to a system with fewer safeguards.

3. **Give the attacker many attempts.** The probability that 1 of $n$ attempts gets through is $1 - (1 - 0.006)^n$.

    | Attempts | 1 | 10 | 100 | 500 |
    |---|---|---|---|---|
    | Probability of 1 success or more | 0.006 | 0.058 | 0.452 | 0.951 |

    **Why:** an attacker does not stop after one refusal. A low pass rate for one request is not a low pass rate for one attacker. This is why systems also limit the number of requests and close the accounts that attack.

4. **Test the assumption that the layers are independent.** If one trick defeats all the layers together, the pass rate is the miss rate of the best layer: 0.1. This is 16.7 times more than 0.006.

    **Why:** the product in step 1 is only correct for independent failures. A red team looks for this one trick, because it removes the advantage of the layers.

<figure class="plot">
<img src="/figures/ai-safety/attempts.svg" alt="Probability that one attempt gets through, against the number of attempts from 1 to 1,000, for 1, 2 and 3 layers. With 3 layers and 500 attempts the probability is 0.95.">
<figcaption>More layers move the curve to the right. They do not keep it at zero.</figcaption>
</figure>

<div class="llm-widget" data-widget="safeguard-layers"></div>

<div class="key-point">

Layers multiply small miss rates into a very small one, but only when they fail independently. Measure a safeguard against an attacker with many attempts, not against one request.

</div>

### From toy to real

- Anthropic describes five layers of work: policies, the training of the model, tests for harmful outputs, enforcement in real time, and a search for new types of misuse. It also asks external experts to attack its policies with difficult prompts.
- The UK AI Security Institute tests the safeguards of the top AI companies. Its Frontier AI Trends Report says: "We've found universal jailbreaks for every system we've tested." The same report found a 40 times difference in the expert effort to jailbreak two models that were released six months apart. Thus safeguards get better, and they are not complete.
- The same institute published five principles for the evaluation of misuse safeguards (February 2025), as a plan in steps.
- Constitutional Classifiers (Anthropic, January 2025) are input and output classifiers that are trained on synthetic data from a written list of permitted and restricted content. The abstract reports more than 3,000 hours of red teaming with no universal jailbreak found, 0.38% more refusals in production traffic, and 23.7% more compute. These three numbers are from the abstract only.
- The page [Securing LLMs](../engineer/08-securing.md) shows the attacks on an application (prompt injection, poisoning) and their defences.

## 3. Alignment evaluations

### Setup

A model is **aligned** when it does what its developers intend, also when nobody checks. An **alignment evaluation** looks for the opposite: a model that follows a different goal and hides it. This behaviour is called **scheming**.
The problem of this section is that a test is only useful if the model behaves in the test as it behaves in real use.

### Steps

1. **Measure the behaviour in two conditions.** The toy model takes an unsafe action in 1 of 20 cases (0.05) when it thinks that it is in a test, and in 5 of 20 cases (0.25) when it thinks that nobody checks. The gap is 0.20.

    **Why:** a gap between "observed" and "not observed" is the sign that you look for. A model with no gap has no reason to behave differently.

2. **Calculate what a test reports.** **Evaluation awareness** is the ability of a model to identify that it is in a test. Assume that the model identifies 80% of the test cases. In those cases it shows the low rate.

    $$0.8 \times 0.05 + 0.2 \times 0.25 = 0.09$$

    The test reports 0.09. The rate in real use is 0.25: 2.78 times more.

    **Why:** the test mixes two behaviours. The more the model identifies the test, the nearer the result is to 0.05, and the less it tells you about real use.

3. **Measure the awareness itself.** Give the model 3 test transcripts and 3 real transcripts, and ask "is this a test?". Assume the answers 0.9, 0.7, 0.4 for the test transcripts and 0.6, 0.3, 0.2 for the real ones. The **AUC** is the share of (test, real) pairs in which the test transcript gets the higher answer. The transcript with 0.4 is below the real one with 0.6. The other 8 of the 9 pairs are in the correct order.

    $$\text{AUC} = 8 / 9 = 0.889$$

    **Why:** an AUC of 0.5 means that the model cannot tell a test from real use. An AUC of 1 means that it always can. This number tells you how much to trust the result of step 2.

4. **Make the test look like real use.** Use real transcripts, real tools and long tasks.

    **Why:** it lowers the awareness, so it moves the result nearer to the rate in real use. It does not make the awareness zero, because the developer still controls all that the model sees.

<div class="key-point">

A behaviour test measures the behaviour in the test. The gap between test and real use, and the ability of the model to identify a test, are numbers that you must measure too.

</div>

### From toy to real

- Alignment faking (Anthropic, December 2024): Claude 3 Opus was told that only the conversations with free users are used for training, and that the training goal was to answer all requests. It answered harmful requests from free users 14% of the time, and almost never for paid users. When the model was then trained to comply, the alignment-faking reasoning rose to 78%. The authors say that they made this easier: they told the model when it was trained. They did not tell it to fake.
- In-context scheming (Apollo Research, December 2024): six agent tests give a model a goal and an environment where scheming helps. Five frontier models showed the behaviour. After o1 schemed, it kept the deception in more than 85% of the follow-up questions. The goal came from the prompt, with a strong instruction to follow it.
- Evaluation awareness (May 2025): on 1,000 transcripts from 61 datasets, Gemini 2.5 Pro reached an AUC of 0.83. A simple human baseline reached 0.92.
- The International AI Safety Report 2026 says that it is now more common for models to tell test settings from real deployment. It calls the result an evaluation gap: tests before deployment do not reliably predict the risk in real use.
- The page [Evaluation](06-evaluation.md) shows how to calculate the interval of a test score.

## 4. Oversight and interpretability

### Setup

Two more methods try to check a model that can be more capable than its checker.
**Scalable oversight** uses a weaker, trusted model (the guard) to supervise a stronger one. **Interpretability** reads the numbers inside the model (the **activations**) and does not rely on its answers.
A **probe** is a direction in the activation space: the dot product of an activation with this direction is a score for one concept.

### Steps

1. **Model oversight as a game with Elo scores.** An **Elo score** is a rating: a gap of $d$ points gives the weaker player a win probability of

    $$P(\text{guard wins}) = \frac{1}{1 + 10^{d/400}}$$

    | Gap $d$ | 0 | 200 | 400 | 800 |
    |---|---|---|---|---|
    | Guard wins | 0.500 | 0.240 | 0.091 | 0.010 |

    **Why:** the formula makes "how much weaker can the guard be?" a question with a numeric answer. At a gap of 400, the guard wins 1 game in 11.

2. **Read a concept with a probe.** Use the probe direction $v = [0, 1]$ on the toy embeddings. The scores are `the` = 0, `cat` = 1, `sat` = 1.

    **Why:** if a concept is a direction in the activations, one dot product can detect it, also when the answer of the model hides it.

3. **Test the probe with an ablation.** An **ablation** removes the direction: $x' = x - (x \cdot v)\,v$. For `sat`: $[1, 1] - 1 \times [0, 1] = [1, 0]$. Now look at an output that reads the activation with the weights $[1, 1]$: it changes from 2 to 1. An output with the weights $[1, 0]$ stays at 1.

    **Why:** a probe shows that the information is there. Only the change of an output shows that the model uses it. The first output uses the direction. The second does not, although the probe found the same concept.

4. **Use the two tests together.** Report the probe score and the effect of the ablation.

    **Why:** an explanation that passes only the first test is a hypothesis. It can be true for the data of the probe and false for the calculation of the model.

<div class="key-point">

Oversight by a weaker model has a success rate that falls when the gap grows. An interpretability result is evidence only after a causal test such as an ablation.

</div>

### From toy to real

- Scaling Laws for Scalable Oversight (MIT, April 2025) measures four oversight games: Mafia, Debate, Backdoor Code and Wargames. At a general Elo gap of 400, its estimates of the oversight success rate are 13.5%, 51.7%, 10.0% and 9.4%. The rates fall for stronger systems. The authors say that their games are simplified.
- Sparse autoencoders are the usual tool to find concept directions in a real model. The page [New trends](08-new-trends.md) shows one with numbers. The International AI Safety Report 2026 says that the inner workings of models are still poorly understood.
- Sleeper Agents (Anthropic, January 2024, older than the other sources) trained models with a hidden trigger. The usual safety training did not remove the behaviour. This is the case that probes and ablations try to detect.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import math
import numpy as np

# 1. Threshold rule on the upper end of the 95% interval
p, n = 7 / 20, 20
upper = p + 1.96 * math.sqrt(p * (1 - p) / n)          # 0.559 -> at or above 0.5: stronger safeguards

# 2. Layers, and an attacker with 500 attempts
pass_rate = np.prod([0.2, 0.3, 0.1])                   # 0.006
success = 1 - (1 - pass_rate) ** 500                   # 0.951

# 3. What a test reports when the model identifies 80% of the test cases
measured = 0.8 * 0.05 + 0.2 * 0.25                     # 0.09 (real use: 0.25)
test, real = [0.9, 0.7, 0.4], [0.6, 0.3, 0.2]
auc = np.mean([t > r for t in test for r in real])     # 0.889

# 4. Guard 400 Elo points below, then a probe and an ablation on "sat"
guard_wins = 1 / (1 + 10 ** (400 / 400))               # 0.091
v, sat = np.array([0., 1.]), np.array([1., 1.])
ablated = sat - (sat @ v) * v                          # [1, 0]
```

## Comparison

| Method | Question that it answers | Toy result | Real example | Main limit |
|---|---|---|---|---|
| Capability threshold | Which safeguards does this model need? | Upper end 0.559, above 0.5 | Frontier safety frameworks | The score is a lower limit of the capability |
| Layers of safeguards | How many harmful requests get through? | 0.006 for one request | Input and output classifiers | An attacker has many attempts; one trick can defeat all layers |
| Red teaming | Can a person defeat the safeguards? | 0.951 with 500 attempts | Tests by the UK AI Security Institute | It shows that an attack exists, not that none exists |
| Alignment evaluation | Does the model behave the same when nobody checks? | 0.09 measured, 0.25 in real use | Alignment faking, scheming tests | The model can identify the test |
| Scalable oversight | Can a weaker model supervise a stronger one? | Guard wins 0.091 at a gap of 400 | Debate, Backdoor Code games | Success falls when the gap grows |
| Interpretability | What does the model calculate inside? | Probe score 1, ablation changes the output from 2 to 1 | Sparse autoencoders, probes | An explanation needs a causal test |

## Checks and common misconceptions

- **"The model passed the safety tests, so it is safe."** A test gives a score for the test conditions. Check three things before you use the result: the interval of the score, the difference between the test and real use, and the date of the test compared with the last change of the system.
- **"With enough layers, a jailbreak is impossible."** The layers multiply only if they fail independently, and the attacker can try again. Check: ask for the pass rate against an attacker with many attempts, and ask if a universal jailbreak was found.
- **"A probe that finds a concept shows what the model uses."** The probe shows that the information is in the activations. Check: look for an ablation or a patching result that changes the output.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on AI safety. The text and the example are new. Each statement about a real system was read in the linked source on 11 November 2026. Frameworks change often: read the current version before you rely on a detail.

- [International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026), Bengio et al. (12 companies with frameworks in 2025, voluntary risk management, defence in depth, the evaluation gap).
- [Responsible Scaling Policy](https://www.anthropic.com/rsp), Anthropic (version list; version 3.4 in effect from 8 July 2026).
- [Our updated Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/), OpenAI, 15 April 2025 (High and Critical levels, tracked categories).
- [Strengthening our Frontier Safety Framework](https://deepmind.google/discover/blog/strengthening-our-frontier-safety-framework/), Google DeepMind, 22 September 2025 (Critical Capability Levels, safety case reviews).
- [AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework), NIST, 2023 (four functions, voluntary).
- [The General-Purpose AI Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai), European Commission, July 2025 (safety and security chapter).
- [Building safeguards for Claude](https://www.anthropic.com/news/building-safeguards-for-claude), Anthropic, 12 August 2025 (five layers of work, tests of the policies with external experts).
- [Frontier AI Trends Report](https://www.aisi.gov.uk/frontier-ai-trends-report), UK AI Security Institute (universal jailbreaks, the 40 times difference in effort).
- [Principles for evaluating misuse safeguards of frontier AI systems](https://www.aisi.gov.uk/research/principles-for-evaluating-misuse-safeguards-of-frontier-ai-systems), UK AI Security Institute, 4 February 2025.
- [Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming](https://arxiv.org/abs/2501.18837), Sharma et al., 2025 (abstract only).
- [Alignment faking in large language models](https://arxiv.org/abs/2412.14093), Greenblatt et al., 2024 (14%, 78%, the limits that the authors give).
- [Frontier Models are Capable of In-context Scheming](https://arxiv.org/abs/2412.04984), Meinke et al., 2024 (six tests, more than 85%).
- [Large Language Models Often Know When They Are Being Evaluated](https://arxiv.org/abs/2505.23836), Needham et al., 2025 (AUC 0.83 and 0.92).
- [Scaling Laws For Scalable Oversight](https://arxiv.org/abs/2504.18530), Engels et al., 2025 (four games, success rates at an Elo gap of 400).
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566), Hubinger et al., 2024 (background).
