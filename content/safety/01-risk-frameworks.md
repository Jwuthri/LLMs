# 1. Risks and safety frameworks

## Core idea

AI safety is the work that keeps a model from causing serious harm, by misuse or by its own behaviour.
Nobody can prove that a model is safe. Thus developers use a loop: they name the possible harms, they test what the model can do, and they connect each test result to a protection.
The result is a written argument that other persons can check.

## The example

This page uses one small safety review of a toy model. All the values are assumed.

| Item | Value |
|---|---|
| Three ways to cause harm | Each has a probability in one year and a severity |
| Capability test | The model completes 7 of 20 dangerous test tasks |
| Threshold | Stronger protections are necessary at a score of 0.5 |
| Safety argument | Three claims, with a confidence of 0.9, 0.9 and 0.95 |

This lesson is the map of the track. The other five lessons each explain one part of the work: [Alignment](02-alignment.md), [Jailbreaks and safeguards](03-jailbreaks.md), [Scheming and alignment evaluations](04-scheming.md), [Interpretability](05-interpretability.md), and [Control and oversight](06-control.md). The [glossary](../glossary.md) lists all the terms.

## 1. The map of risks

### Setup

**AI safety** is the work that prevents serious harm from AI systems. A **risk** is a possible harm with two numbers: its probability and its severity.
A **threat model** is a short description of one risk: who causes the harm, by which path, and what the harm is.
Most documents put the risks in three groups:

- **Misuse**: a person uses the model to cause harm on purpose.
- **Misalignment**: the model follows a goal that its developers did not intend. The extreme case is **loss of control**: nobody can stop or correct the system.
- **Malfunction**: the model fails with no bad intent from a person or from the model, for example a wrong answer.

A **capability** is what a model can do. A **propensity** is what a model does by its own decision. A **dual-use** capability helps a defender and an attacker: the knowledge that finds an error in software also helps to attack it.

### Steps

1. **Write one threat model for each group.** Misuse: a person uses the model to prepare an attack on a computer network. Misalignment: an agent takes a harmful action that nobody asked for. Malfunction: the model gives wrong advice.

    **Why:** "the model is dangerous" is not a statement that you can test. A threat model names a path, and each step of a path can be tested or blocked.

2. **Multiply the probability by the severity.** Assume these values for one year, with the severity in units of harm.

    | Threat model | Probability | Severity | Expected harm |
    |---|---|---|---|
    | Misuse | 0.01 | 1,000 | 10 |
    | Misalignment | 0.001 | 10,000 | 10 |
    | Malfunction | 0.2 | 5 | 1 |

    **Why:** the product makes the risks comparable. A rare risk with a large harm can count as much as a frequent risk with a small harm. Here the rarest risk has the same expected harm as the misuse risk, and 10 times that of the malfunction.

3. **Measure what the model adds.** **Uplift** is the increase in the success of a person because of the model. Assume a harmless test task that 4 of 100 persons complete with a search engine, and 10 of 100 persons complete with the model.

    $$\text{uplift} = 0.10 - 0.04 = 0.06, \qquad \frac{0.10}{0.04} = 2.5$$

    **Why:** much dangerous information is already public. The question for misuse is not "can the model say it?" but "how much easier does the model make it?".

4. **Separate "can" from "does".** Assume that the model can complete 7 of 20 dangerous tasks (0.35), and that it starts such a task by itself in 1 of 20 cases (0.05). For a misalignment risk the two numbers multiply: $0.35 \times 0.05 = 0.0175$. For a misuse risk the person brings the intent, so only the capability and the protections count.

    **Why:** the two numbers need different tests. A capability test asks the model to try as hard as possible. A propensity test looks at what the model chooses when nobody asks.

<div class="key-point">

A risk is a path with a probability and a severity. Misuse depends on what the model can do for a person. Misalignment depends on what the model can do and on what it chooses to do.

</div>

### From toy to real

- The International AI Safety Report 2026 uses three groups: malicious use (for example fraud, manipulation, cyberattacks, biological and chemical risks), malfunctions (reliability problems and loss of control), and systemic risks (for example the effects on the labour market and on human autonomy).
- The frameworks of the developers track a small number of domains with severe harm. **CBRN** is the short name for chemical, biological, radiological and nuclear weapons. The other usual domains are cyberattacks and the automation of AI research.
- The product of step 2 hides one thing: some harms cannot be undone. This is why the frameworks set fixed limits for the most severe risks and do not only compare expected values.

## 2. Dangerous capability evaluations

### Setup

An **evaluation** (or eval) is a test of a model with a fixed set of tasks and a fixed scoring rule. A **dangerous capability evaluation** measures a capability from a threat model, with tasks that are safe to run.
**Elicitation** is the work to get the best performance from the model: a good prompt, tools, many attempts. A **scaffold** is the program around the model that gives it tools and memory.
The page [Evaluation](../scientist/06-evaluation.md) gives the general method. This section adds what is special for safety.

### Steps

1. **Measure the score.** The model completes 7 of 20 tasks. The score is $7 / 20 = 0.35$.

    **Why:** a rule can only start from a number that two persons can measure in the same way.

2. **Calculate how sure the score is.** With $n$ tasks, the standard error is $\sqrt{p(1-p)/n}$. The 95% interval is the score ± 1.96 standard errors.

    $$\sqrt{\frac{0.35 \times 0.65}{20}} = 0.107, \qquad 0.35 \pm 1.96 \times 0.107 = [0.141,\ 0.559]$$

    **Why:** 20 tasks are a small sample. The true score can be far from 0.35.

3. **Do the test again with better elicitation.** With a better scaffold, the same model completes 11 of 20 tasks. The score is 0.55 and the interval is [0.332, 0.768].

    **Why:** the weights did not change, but the score did. A measured score is a lower limit of the capability. An attacker uses the best method, so the test must use it too.

4. **Calculate a time horizon.** The **time horizon** is the length of the tasks, in human work time, that the model completes with a success rate of 50%. Assume these success rates.

    | Human time for the task (minutes) | 1 | 4 | 16 | 64 | 256 |
    |---|---|---|---|---|---|
    | Success rate of the model | 0.95 | 0.85 | 0.6 | 0.4 | 0.1 |

    The rate crosses 0.5 between 16 and 64 minutes, in the middle of the two rates. On a log scale, the middle of 16 and 64 is $16 \times 4^{0.5} = 32$ minutes.

    **Why:** one number in minutes is easy to compare between models and between years. It also shows how old a test result is: if the horizon doubles each 7 months, it is $32 \times 2^{3} = 256$ minutes after 21 months.

<div class="key-point">

A capability score is a lower limit: better elicitation can only find more. Report the score with its interval and with the method that was used to get it.

</div>

### From toy to real

- METR defines the 50% time horizon as the time that humans typically need for the tasks that a model completes with a 50% success rate. Its paper reports a doubling about each seven months since 2019, and says that the trend possibly became faster in 2024. It gives a horizon of about 50 minutes for Claude 3.7 Sonnet. The tasks are software tasks.
- The UK AI Security Institute reports the same type of trend for cyber tasks: the length that models complete without help doubles about each eight months, from less than 10 minutes in early 2023 to more than one hour in mid-2025.
- The International AI Safety Report 2026 says that coding agents can reliably complete some tasks that take a human programmer about half an hour, up from less than 10 minutes one year before.
- [Inspect](https://inspect.aisi.org.uk/) is an open-source framework for evaluations, made by the UK AI Security Institute and Meridian Labs. Use it when you write your own evals: it gives the tasks, the scaffold and the scoring a common structure.
- A model can also get a low score on purpose. The lesson [Scheming and alignment evaluations](04-scheming.md) explains this problem.

## 3. If-then commitments

### Setup

A **frontier safety framework** is a public document in which a developer says how it manages the risks of its most capable models.
Its core is an **if-then commitment**: if a test shows that the model has a dangerous capability, then stronger protections must be in place before the next step.
A **capability threshold** is the test result that starts the "then" part. A **safeguard** is a protection. There are two types: deployment safeguards stop a harmful use of the model, and security safeguards stop the theft of its weights.

<figure class="diagram" data-src="risk-frameworks/if-then.svg">
<figcaption>An if-then commitment as a loop: test, compare with the threshold, apply the safeguards, test again.</figcaption>
</figure>

### Steps

1. **Write the rule before the test.** The toy rule is: "if the model can complete half of the test tasks (0.5), stronger safeguards are necessary".

    **Why:** after a test, there is pressure to release. A rule that is written before the result cannot be adjusted to the result.

2. **Apply the rule to the upper end of the interval.** The upper end for 7 of 20 tasks is 0.559. It is above 0.5. The decision is "stronger safeguards", although the score of 0.35 is below the threshold.

    **Why:** the two possible errors do not cost the same. A missed dangerous capability can cause harm that you cannot undo. A safeguard that was not necessary only costs time and money.

3. **Make the test larger.** With 70 of 200 tasks, the score is the same, 0.35. The standard error is 0.034 and the interval is [0.284, 0.416]. The upper end is below 0.5. The decision is "current safeguards". At this score, 39 tasks are the minimum for this decision.

    **Why:** more tasks make the interval narrower. A larger test is the correct way to get a less strict decision, not a higher threshold.

4. **Do the test again.** Do it when the model changes, and when it gets new tools.

    **Why:** a test result is true only for the system that was tested. Step 3 of the section before showed that a better scaffold raises the score of the same model.

<figure class="plot">
<img src="/figures/risk-frameworks/interval.svg" alt="The 95% interval of a score of 0.35 against the number of test tasks, from 5 to 400. The upper end is 0.559 with 20 tasks and 0.416 with 200 tasks. It goes below the threshold of 0.5 at 39 tasks.">
<figcaption>The score stays at 0.35. Only the size of the test moves the upper end below the threshold.</figcaption>
</figure>

<div class="llm-widget" data-widget="threshold-test"></div>

<div class="key-point">

A framework changes "is this model safe?" into "which test result starts which safeguard?". The rule is written before the test, and it uses the cautious end of the interval.

</div>

### From toy to real

- The International AI Safety Report 2026 counts 12 companies that published or updated a frontier safety framework in 2025. It also says that most of this risk management is still voluntary.
- Anthropic published its Responsible Scaling Policy on 19 September 2023. That version named the sets of safeguards **AI Safety Levels** (ASL), after the biosafety levels of laboratories: ASL-2 for systems with early signs of dangerous capabilities, ASL-3 for systems that substantially increase the risk of catastrophic misuse. On 22 May 2025, the company activated the ASL-3 standards with Claude Opus 4 as a precaution: it could no longer clearly exclude the risk. Version 3.0 (February 2026) was a full rewrite. Version 3.4 is in effect from 8 July 2026. Its thresholds are written for domains, for example CBRN weapons and the automation of AI research.
- The Preparedness Framework of OpenAI (updated on 15 April 2025) has two levels. A model at the High level must have sufficient safeguards before deployment. A model at the Critical level must have them during development too. The tracked categories are biological and chemical capabilities, cybersecurity, and AI self-improvement.
- The Frontier Safety Framework of Google DeepMind (third version, 22 September 2025) uses **Critical Capability Levels**: capability levels at which a model can bring a higher risk of severe harm if there are no mitigations. This version added a level for harmful manipulation.
- Governments publish their own documents. The NIST AI Risk Management Framework (26 January 2023, voluntary) has four functions: govern, map, measure, manage. The EU code of practice for general-purpose AI (10 July 2025, voluntary) has three chapters. Its safety and security chapter applies only to the providers of the models with systemic risk.

## 4. Safety cases and outside checks

### Setup

A **safety case** is a structured argument, with evidence, that a system is unlikely to cause a catastrophe. It is the document that connects the tests and the safeguards to one conclusion.
A paper on safety cases for AI lists four types of argument. The first three are the plan of this track:

| Argument | Short form | Evidence | Lessons |
|---|---|---|---|
| Inability | The model cannot do it | Capability evaluations | This lesson |
| Control | The model can, but the safeguards stop it | Red teaming, monitoring | 3 and 6 |
| Trustworthiness | The model can, but it does not want to | Alignment evaluations, interpretability | 2, 4 and 5 |
| Deference | A trusted AI system says that it is safe | The advice of that system | For much more capable systems |

### Steps

1. **Write the claim and the claims below it.** The top claim is "the release of the toy model gives no serious uplift". It needs three claims: the tests find the true capability, the test tasks cover the threat model, and the safeguards hold. Assume a confidence of 0.9, 0.9 and 0.95.

    **Why:** a claim that you cannot divide is a claim that you cannot check. Each lower claim has its own evidence and its own reviewer.

2. **Multiply the confidences.** The argument holds only if each claim holds.

    $$0.9 \times 0.9 \times 0.95 = 0.77$$

    **Why:** a chain is weaker than each of its links. Three claims that look strong give an argument with a hole in almost 1 case of 4.

3. **Add a second, independent check to the weakest claim.** A team from outside the company does its own capability tests. Assume a confidence of 0.8 from this team alone. The claim fails only if the two teams fail.

    $$1 - (1 - 0.9)(1 - 0.8) = 0.98, \qquad 0.98 \times 0.9 \times 0.95 = 0.838$$

    **Why:** more work on a strong claim changes little. A second line of evidence on a weak claim changes much. An outside team also has no reason to want the release.

4. **Say what remains.** The argument can still be wrong with a probability of $1 - 0.838 = 0.162$.

    **Why:** this number is not the probability of harm. It is the probability that the argument has a hole. A safety case that hides this number asks the reader to trust it. A safety case that shows it can be checked.

5. **Change the type of argument when the capability grows.** In section 3, the upper end of the score went above the threshold. The inability argument is then not available, and the case must use control or trustworthiness.

    **Why:** each type of argument has a limit. "It cannot" ends when the tests say that it can. "The safeguards stop it" ends when the model can defeat the safeguards.

<figure class="diagram" data-src="risk-frameworks/safety-case.svg">
<figcaption>The toy safety case. The second team raises the weakest claim, and thus the full argument.</figcaption>
</figure>

<div class="key-point">

A safety case is as strong as the product of its claims. Independent evidence on the weakest claim is the best use of more work, and the type of argument must change when the model becomes more capable.

</div>

### From toy to real

- The paper "Safety Cases: How to Justify the Safety of Advanced AI Systems" defines the safety case and the four types of argument: inability, control, trustworthiness, and deference to credible AI advisors.
- Google DeepMind does a safety case review before an external launch when a model reaches a Critical Capability Level. For the levels about machine learning research, the review now also applies to large internal deployments.
- A **system card** (or model card) is the report of the tests and their results that a developer publishes with a model. The page of the Anthropic policy also lists risk reports.
- Outside checks exist, but they are not yet the rule. The UK AI Security Institute tests the models and the safeguards of the top AI companies. The International AI Safety Report 2026 names the remaining problem the **evaluation gap**: the results of the tests before deployment do not reliably predict the risk in real use.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import math

# 1. Risks: expected harm, uplift, capability times propensity
harm = {"misuse": 0.01 * 1000, "misalignment": 0.001 * 10000, "malfunction": 0.2 * 5}   # 10, 10, 1
uplift = 10 / 100 - 4 / 100                                                             # 0.06
can_and_tries = (7 / 20) * (1 / 20)                                                     # 0.0175

# 2. A score with its 95% interval, and a time horizon on a log time axis
def interval(solved, tasks):
    p = solved / tasks
    se = math.sqrt(p * (1 - p) / tasks)
    return p - 1.96 * se, p + 1.96 * se

low, high = interval(7, 20)                       # 0.141, 0.559
horizon = 16 * (64 / 16) ** ((0.6 - 0.5) / (0.6 - 0.4))   # 32 minutes
later = horizon * 2 ** (21 / 7)                   # 256 minutes after 21 months

# 3. The if-then rule uses the upper end of the interval
decision = "stronger safeguards" if high >= 0.5 else "current safeguards"
low_200, high_200 = interval(70, 200)             # 0.284, 0.416 -> current safeguards

# 4. A safety case: a chain of claims, then a second team on the first claim
case = 0.9 * 0.9 * 0.95                           # 0.77
with_second_team = (1 - (1 - 0.9) * (1 - 0.8)) * 0.9 * 0.95   # 0.838
```

## Comparison

| Document | Who writes it | What starts an action | Status |
|---|---|---|---|
| Responsible Scaling Policy | Anthropic | Capability thresholds for each domain | Company policy |
| Preparedness Framework | OpenAI | High and Critical capability levels | Company policy |
| Frontier Safety Framework | Google DeepMind | Critical Capability Levels | Company policy |
| AI Risk Management Framework | NIST (United States) | No threshold: four functions for all AI systems | Voluntary |
| Code of practice for general-purpose AI | European Commission | Models with systemic risk | Voluntary code |
| The toy rule of this page | You | Upper end of the interval at 0.5 or more | Example |

## Checks and common misconceptions

- **"The score is below the threshold, so the model does not have the capability."** A score is a lower limit, and it has an interval. Check: ask for the number of tasks, for the upper end of the interval, and for the elicitation method. Compare the date of the test with the last change of the system.
- **"A safety framework is a law."** Most frameworks are promises of a company, and the company can change them. The page of the Anthropic policy lists more than five versions. Check: read the current version and its list of changes before you rely on a detail.
- **"A safety case proves that the model is safe."** It is an argument with assumptions, and it holds only if each claim holds. Check: find the weakest claim, and ask who checked it and if that person was independent.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. Each statement about a real system was read in the linked source on 11 November 2026. Frameworks change often: read the current version before you rely on a detail.

- [International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026), Bengio et al. (the three groups of risks, 12 companies with frameworks in 2025, voluntary risk management, the length of tasks for coding agents, the evaluation gap).
- [Measuring AI Ability to Complete Long Software Tasks](https://arxiv.org/abs/2503.14499), Kwa et al., 2025 (the definition of the time horizon, the doubling time, the horizon of Claude 3.7 Sonnet).
- [Frontier AI Trends Report](https://www.aisi.gov.uk/frontier-ai-trends-report), UK AI Security Institute (the length of cyber tasks and its doubling time).
- [Inspect](https://inspect.aisi.org.uk/), UK AI Security Institute and Meridian Labs (what the framework is).
- [Anthropic's Responsible Scaling Policy](https://www.anthropic.com/news/anthropics-responsible-scaling-policy), Anthropic, 19 September 2023 (AI Safety Levels, ASL-2 and ASL-3).
- [Activating AI Safety Level 3 protections](https://www.anthropic.com/news/activating-asl3-protections), Anthropic, 22 May 2025 (the precaution for Claude Opus 4).
- [Responsible Scaling Policy](https://www.anthropic.com/rsp), Anthropic (the list of versions, version 3.4 in effect from 8 July 2026, the domains of the thresholds, risk reports).
- [Our updated Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/), OpenAI, 15 April 2025 (High and Critical levels, tracked categories).
- [Strengthening our Frontier Safety Framework](https://deepmind.google/discover/blog/strengthening-our-frontier-safety-framework/), Google DeepMind, 22 September 2025 (Critical Capability Levels, the level for harmful manipulation, safety case reviews).
- [AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework), NIST, 26 January 2023 (voluntary use), and its [core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/) (the four functions).
- [The General-Purpose AI Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai), European Commission, 10 July 2025 (three chapters, voluntary, the scope of the safety and security chapter).
- [Safety Cases: How to Justify the Safety of Advanced AI Systems](https://arxiv.org/abs/2403.10462), Clymer et al., 2024 (the definition of a safety case, the four types of argument).
