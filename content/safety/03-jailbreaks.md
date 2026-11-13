# 3. Jailbreaks and safeguards

## Core idea

A jailbreak is a prompt that makes a model do what its developer trained it to refuse.
No single protection stops all jailbreaks.
Thus developers measure how often attacks succeed, put several protections in a row, and pay persons to attack them.

## The example

The [toy example](../toy-example.md) has no prompt and no refusal, so this page uses one small safety test of a chat model. All the values are assumed.

| Item | Value |
|---|---|
| Test set | 50 harmful test requests |
| No attack | The model answers 3 of the 50 |
| One attack template | The model answers 30 of the 50 |
| Judge | It counts 10% of the refusals as harmful answers |
| Agent | 1 web page in 100 has a hidden instruction. The model follows it with probability 0.1 |
| Three layers | They miss 0.2, 0.3 and 0.1 of the harmful requests |

This page gives the mechanism of each attack, and no attack text.

## 1. Jailbreaks and the attack success rate

### Setup

**Misuse** is the use of a model by a person to cause harm. A **refusal** is an answer in which the model declines to help.
A **jailbreak** is a prompt that gets a harmful answer where the model normally refuses. A **universal jailbreak** is one method that works for most harmful requests.

The **attack success rate (ASR)** is the share of the harmful test requests for which an attack gets a harmful answer. A **judge** is the grader that reads each answer and decides "harmful answer" or "refusal": a person, a set of rules or a second model.

The main families of jailbreaks:

| Family | Mechanism |
|---|---|
| **Role-play (persona) attack** | The prompt puts the request in a story, or gives the model a character that has no rules. The harmful answer becomes "what the character says". |
| **Obfuscation** | The attacker writes the request in a code, in a rare language or with changed letters. The model can still read it, but it looks different from the requests that it learned to refuse. |
| **Many-shot jailbreak** | The prompt contains a long invented dialogue in which an assistant answers many harmful questions. The model continues this pattern. |
| **Multi-turn attack** | The attacker starts with harmless questions and moves to the harmful one in small steps. Each step is a small change from an answer that the model already gave. |
| **Adversarial suffix** | An optimiser uses the gradients of an open model to find a string of tokens that makes an agreeing answer more probable. The attacker adds this string, which looks like noise, after the request. |
| **Best-of-N** | A program sends many random variants of one request, for example with changed capital letters, and keeps the first one that works. |

### Steps

1. **Send the 50 requests with no attack.** The model answers 3 of them. $\text{ASR} = 3 / 50 = 0.06$.

    **Why:** this is the baseline. Without it, you cannot say how much an attack adds.

2. **Put each request in one attack template and send it again.** The model answers 30 of them. $\text{ASR} = 30 / 50 = 0.60$. The attack adds 0.54.

    **Why:** the two runs use the same 50 requests, so the difference comes from the attack.

3. **Calculate how sure each rate is.** With $n$ requests, the standard error is $\sqrt{p(1-p)/n}$. The 95% interval is the rate ± 1.96 standard errors.

    $$\sqrt{\frac{0.60 \times 0.40}{50}} = 0.069, \qquad 0.60 \pm 0.136 = [0.464,\ 0.736]$$

    $$\sqrt{\frac{0.06 \times 0.94}{50}} = 0.034, \qquad 0.06 \pm 0.066 = [-0.006,\ 0.126]$$

    The half widths use the standard errors before rounding. A rate cannot be negative, so the second interval is [0, 0.126].

    **Why:** 50 requests are a small sample: the true ASR of the attack can be 0.46 or 0.74. The negative end shows that this formula is rough for a small count (3 answers).

4. **Correct the rate for the errors of the judge.** Steps 1 and 2 used a perfect judge. Assume now a judge with 10% false positives: it counts 10% of the refusals as harmful answers, and makes no other error.

    $$\text{measured} = \text{true} + 0.1 \times (1 - \text{true})$$

    | Run | True ASR | Measured ASR | Measured / true |
    |---|---|---|---|
    | No attack | 0.06 | 0.154 | 2.57 |
    | Attack | 0.60 | 0.640 | 1.07 |

    The measured gain of the attack is 0.486, not 0.54. To go back to the true rate: $(0.154 - 0.1) / 0.9 = 0.06$.

    **Why:** more requests do not remove a judge error, because it moves the result in one direction. The error is largest when the true rate is low: the judge has 47 refusals to misread, not 20. Thus compare the judge with human labels before you trust a low ASR.

<div class="key-point">

An ASR belongs to four things: the model, the set of requests, the attack and the judge. Give all four with the number, and give the interval.

</div>

### From toy to real

- StrongREJECT (February 2024) says that jailbreak papers often overstate the effect of their attacks. Its automated judge scores how useful the answer is, and the authors report that it agrees with human judgement better than earlier judges.
- The GCG attack (July 2023) finds an adversarial suffix with a greedy, gradient-based search on open models (Vicuna-7B and 13B). The authors report that the suffixes transfer to the public interfaces of ChatGPT, Bard and Claude.
- Many-shot jailbreaking (Anthropic, April 2024) tested prompts with up to 256 invented dialogues. The success grows with their number as a power law. A defence that classifies and changes the prompt lowered the ASR from 61% to 2% in one case. Refusal training only delayed the attack.
- Best-of-N jailbreaking (December 2024) reports an ASR of 89% on GPT-4o and 78% on Claude 3.5 Sonnet. This result needs 10,000 variants of each request.
- Crescendo (April 2024) is a multi-turn attack. Its authors report high success rates on ChatGPT, Gemini Pro, Gemini Ultra, LLaMA-2 70B, LLaMA-3 70B and the chat model of Anthropic.
- The refusal direction and steering change the activations of the model, not the prompt: see [Interpretability](05-interpretability.md).

## 2. Prompt injection in agents

### Setup

A jailbreak comes from the user. A **prompt injection** is text that the model gets as data and obeys as an instruction.
It is a **direct prompt injection** when the attacker types the text in the conversation. It is an **indirect prompt injection** when the attacker puts the text in a web page, a file or an email that the model reads later. Then the attacker is not in the conversation.
The cause is that a model has no hard **boundary between instructions and data**: the two are tokens in one sequence.

An [agent](../engineer/05-agents.md) is a model that works in a loop and uses tools. A **tool** is a function that the model can call, for example "open a web page" or "send an email". **Exfiltration** is the transfer of private data to the attacker.

The page [Securing LLMs](../engineer/08-securing.md) shows this attack in one application, with its filters and its tool policy. This part calculates the risk for an agent that reads many pages.

### Steps

1. **Calculate the hijack rate for one page.** Assume that 1 page in 100 has a hidden instruction, and that the model follows it with probability 0.1.

    $$0.01 \times 0.1 = 0.001$$

    **Why:** two things must go wrong: the agent reads a bad page, and the model obeys it. Training makes the second number smaller, not zero.

2. **Let the agent read many pages.** The probability of one hijack or more after $n$ pages is $1 - (1 - 0.001)^n$.

    | Pages | 1 | 100 | 1,000 |
    |---|---|---|---|
    | Probability of one hijack or more | 0.001 | 0.095 | 0.632 |

    **Why:** a rate that is small for one page is large for a long task, or for many users.

3. **Add a human confirmation for the dangerous tools.** **Human confirmation** means that the agent asks the user before it calls a tool that can cause harm. Assume that a hijack needs such a tool, and that the user approves a bad call by error 1 time in 10. The rate for one page becomes $0.001 \times 0.1 = 0.0001$.

    After 1, 100 and 1,000 pages, the probabilities are 0.0001, 0.010 and 0.095.

    **Why:** the confirmation checks the action, so it does not need to detect the injection. But it only moves the problem: 1,000 pages now have the risk of 100 pages. And a user who gets too many questions approves all of them.

4. **Give the agent only the tools and the data that the task needs.** This rule is **least privilege**. An agent that summarises web pages needs no tool that sends email, and no access to private files.

    **Why:** steps 1 to 3 give the probability of a hijack, not its cost. A hijacked agent with no tool that sends data out cannot exfiltrate data.

<figure class="plot">
<img src="/figures/jailbreaks/hijack.svg" alt="Probability of one hijack or more against the number of pages, without and with a human confirmation. After 1,000 pages: 0.632 and 0.095.">
<figcaption>The confirmation step moves the curve to the right by a factor of 10. It does not keep it at zero.</figcaption>
</figure>

<div class="key-point">

A prompt injection exists because instructions and data share one channel. Calculate the risk for all the content that an agent reads, and put the strongest controls on the actions, not on the text.

</div>

### From toy to real

- Greshake et al. (February 2023) gave indirect prompt injection its name. They showed attacks on Bing Chat with GPT-4 and on code-completion engines, and wrote that effective mitigations were missing.
- AgentDojo (June 2024) is a test environment for this attack, with 97 realistic tasks and 629 security test cases. Its authors found that models fail many tasks with no attack, and that the attacks broke some security properties, not all.

## 3. Layers of safeguards

### Setup

A **safeguard** is a protection that stops a harmful use of a model. **Defence in depth** is the use of many safeguards in a row, because one is not sufficient. Our toy system has three layers on the path of each request:

- The **input classifier** is a small model that reads the request and gives it a score. It blocks the request when the score is at or above a threshold.
- The model itself is trained to refuse.
- The **output classifier** reads the answer in the same way.

Assume that the layers miss 0.2, 0.3 and 0.1 of the harmful requests that an attacker sends.

<figure class="diagram" data-src="jailbreaks/layers.svg">
<figcaption>A harmful request gets through only if each of the three layers fails.</figcaption>
</figure>

Other safeguards are not on this path:

- In training: **adversarial training** adds known attacks, with the correct refusal, to the training data. **Unlearning** tries to remove dangerous knowledge from the weights, so that no prompt can get it out.
- On the account: a **rate limit** is a maximum number of requests for one account in a period. **Account enforcement** warns or closes an account that attacks.

For a layer, a **false positive** is a harmless request that it blocks. **Over-refusal** is the same error in the model: it refuses a request that it should answer.

### Steps

1. **Multiply the miss rates.** A harmful request gets through only if all the layers miss it.

    $$0.2 \times 0.3 \times 0.1 = 0.006$$

    Of 1,000 harmful requests, 6 get through.

    **Why:** each layer only sees what the layers before it did not stop. The model alone lets 0.3 through.

2. **Count the harmless requests that you lose.** Assume that the layers stop 0.5%, 1% and 0.5% of the harmless requests. A harmless request gets through if no layer stops it: $0.995 \times 0.99 \times 0.995 = 0.980$. Thus 1.99% of the harmless requests are lost.

    **Why:** each layer also has a cost. A system that refuses too much is not used, and its users go to a system with fewer safeguards.

3. **Choose the threshold of the input classifier.** Assume the scores of the script for a labelled test set of 10 harmful and 200 harmless requests.

    | Threshold | Miss rate | False block rate | 3 layers: harmful that pass | 3 layers: harmless lost |
    |---|---|---|---|---|
    | 0.4 | 0.1 | 0.040 | 0.003 | 0.054 |
    | 0.6 | 0.2 | 0.005 | 0.006 | 0.020 |
    | 0.8 | 0.5 | 0 | 0.015 | 0.015 |

    Our system uses the threshold 0.6.

    **Why:** the developer chooses the threshold. A lower threshold stops more attacks and loses more harmless requests. Also, 10 harmful requests give the miss rate only in steps of 0.1, so a real test set must be much larger.

<div class="key-point">

Layers multiply small miss rates into a very small one. Each layer also blocks harmless requests, so a safeguard has two numbers. The threshold trades one for the other.

</div>

### From toy to real

- Anthropic describes five layers of work: policies, the training of the model, tests for harmful outputs, enforcement in real time, and a search for new types of misuse. A pattern of violations can lead to a warning or, in severe cases, to the end of the account.
- A **constitutional classifier** (Anthropic, January 2025) is an input or output classifier that is trained on synthetic data from a written list of permitted and restricted content. The abstract reports more than 3,000 hours of red teaming with no universal jailbreak found, 0.38% more refusals in production traffic, and 23.7% more compute.
- The WMDP benchmark (March 2024) has 3,668 multiple-choice questions, a proxy for dangerous knowledge in biosecurity, cybersecurity and chemical security. Its authors report that their unlearning method, RMU, lowers the score on these questions and keeps the general capabilities.
- The UK AI Security Institute says that open-weight models are particularly hard to protect: a person with the weights can remove the trained refusals at a low cost and switch off the classifiers.

## 4. Red teaming

### Setup

**Red teaming** is a test in which persons play the attacker and try to get a harmful answer through the safeguards. **Automated red teaming** does the same with programs or with attacker models, at a larger scale.

A red team needs a **threat model**: a written description of the attacker (who it is, what it wants, what it knows). The **attacker budget** is what the attacker can spend: attempts, time, money and skill.
**Uplift** is how much an answer helps the attacker, compared with what the attacker can do without the model.
A **bug bounty** is a programme that pays external persons who report an attack that works.

The toy system is the three layers of part 3.

### Steps

1. **Give the attacker many attempts.** The probability that 1 of $n$ attempts gets through is $1 - (1 - 0.006)^n$.

    | Attempts | 1 | 10 | 100 | 500 |
    |---|---|---|---|---|
    | Probability of 1 success or more | 0.006 | 0.058 | 0.452 | 0.951 |

    **Why:** an attacker does not stop after one refusal. A low pass rate for one request is not a low pass rate for one attacker. A rate limit and account enforcement make the budget of the attacker smaller.

2. **Calculate how many attempts the red team needs.** To find a hole of rate $p$ with probability 0.95, solve $1 - (1-p)^n = 0.95$ for $n$.

    $$n = \frac{\ln(0.05)}{\ln(1 - p)} = \frac{\ln(0.05)}{\ln(0.994)} = 497.8$$

    The red team needs 498 attempts.

    **Why:** this number is the size of the test. A red team that stops after 100 attempts finds this hole less than half of the time (0.452).

3. **Read a test with no success correctly.** After $n$ attempts with no success, the largest pass rate that the test does not exclude (at the 95% level) is $1 - 0.05^{1/n}$. For 100 attempts it is 0.030. For 500 attempts it is 0.006.

    **Why:** a red team that finds nothing did not show that the pass rate is zero. It gave an upper limit, which falls only as fast as the number of attempts grows.

4. **Test the assumption that the layers are independent.** If one trick defeats all the layers together, the pass rate is the miss rate of the best layer: 0.1. This is 16.7 times more than 0.006. Then 29 attempts are sufficient for a probability of 0.95.

    **Why:** the product in part 3 is only correct for independent failures. A red team looks for this one trick, a universal jailbreak, because it removes the advantage of the layers.

<figure class="plot">
<img src="/figures/jailbreaks/attempts.svg" alt="Probability that one attempt gets through against the number of attempts, for 1, 2 and 3 layers. With 3 layers and 500 attempts: 0.95.">
<figcaption>More layers move the curve to the right. They do not keep it at zero.</figcaption>
</figure>

<div class="llm-widget" data-widget="safeguard-layers"></div>

<div class="key-point">

Measure a safeguard against an attacker with a budget, not against one request. A red team can show that an attack exists. It cannot show that no attack exists.

</div>

### From toy to real

- The Frontier AI Trends Report of the UK AI Security Institute says: "We've found universal jailbreaks for every system we've tested." The same report found a 40 times difference in the expert effort to jailbreak two models that were released six months apart: about 10 minutes for the first model, more than 7 hours for the second, for biological misuse requests. Thus safeguards get better, and they are not complete.
- The same report gives limits: the progress is smaller for the other categories of misuse, and a harmful answer in a test does not show that the information is correct or that a novice can use it (uplift).
- HarmBench (February 2024) is a standard test for automated red teaming. Its authors compare 18 methods on 33 models and defences, and propose an adversarial training method.
- Three open tools: garak (NVIDIA) scans a model with known types of attack. PyRIT (the AI Red Team of Microsoft) automates scans and multi-turn attacks. Inspect (UK AI Security Institute and Meridian Labs) is an evaluation framework with datasets, solvers and scorers. A scorer is a judge.
- Other lessons: [Risk frameworks](01-risk-frameworks.md) (which safeguards a model needs), [Scheming](04-scheming.md) and [Control](06-control.md) (the risk from the model itself).

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import math
import numpy as np

# 1. ASR of the attack with its 95% interval, and a judge with 10% false positives
p, n = 30 / 50, 50
half = 1.96 * math.sqrt(p * (1 - p) / n)               # 0.136 -> [0.464, 0.736]
measured = 3 / 50 + 0.1 * (1 - 3 / 50)                 # 0.154 (true ASR: 0.06)

# 2. An agent reads 1,000 pages, without and with a human confirmation
hijack = 0.01 * 0.1                                    # 0.001 for one page
no_check = 1 - (1 - hijack) ** 1000                    # 0.632
with_check = 1 - (1 - hijack * 0.1) ** 1000            # 0.095

# 3. Three layers: harmful requests that pass, harmless requests that are lost
pass_rate = np.prod([0.2, 0.3, 0.1])                   # 0.006
lost = 1 - np.prod([0.995, 0.99, 0.995])               # 0.0199

# 4. An attacker with 500 attempts, and the size of a red-team test
success = 1 - (1 - pass_rate) ** 500                   # 0.951
attempts = math.ceil(math.log(0.05) / math.log(1 - pass_rate))   # 498
limit = 1 - 0.05 ** (1 / 100)                          # 0.030 after 100 attempts with no success
```

## Comparison

| Safeguard | Where it acts | Toy result | What it does not stop |
|---|---|---|---|
| Refusal training | In the model | ASR 0.06, or 0.60 with one attack | A new family of attacks |
| Input and output classifiers | Before and after the model | Pass rate from 0.3 to 0.006; 1.99% of harmless requests lost | One trick that defeats all the layers |
| Human confirmation, least privilege | On the tool calls of an agent | Hijack in 1,000 pages: from 0.632 to 0.095 | A user who approves each question |
| Rate limit, account enforcement | On the account | Success 0.452 at 100 attempts, 0.951 at 500 | An attacker with many accounts |
| Red teaming | Before and after the release | 498 attempts to find a hole of 0.006 | It cannot show that no attack exists |

## Checks and common misconceptions

- **"The ASR is 0.06, so the model is safe."** This is the ASR with no attack. Check: ask for the ASR of the strongest known attack, its interval, and the error rate of the judge.
- **"A jailbreak and a prompt injection are the same attack."** In a jailbreak, the user attacks the rules of the developer. In an indirect prompt injection, a third person attacks the user through the data. Check: test what the agent does with untrusted content and what its tools permit.
- **"The red team found nothing, so there is no jailbreak."** After 100 attempts with no success, the pass rate can still be 0.030. Check: ask for the threat model, the hours and the number of attempts of the test.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. Each statement about a real system was read in the linked source on 11 November 2026.

- [Frontier AI Trends Report](https://www.aisi.gov.uk/frontier-ai-trends-report), UK AI Security Institute (universal jailbreaks, the 40 times difference in effort, open-weight models, limits).
- [Building safeguards for Claude](https://www.anthropic.com/news/building-safeguards-for-claude), Anthropic, 12 August 2025 (five layers of work, account actions).
- [Constitutional Classifiers](https://arxiv.org/abs/2501.18837), Sharma et al., 2025 (abstract only: 3,000 hours, 0.38%, 23.7%).
- [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043), Zou et al., 2023 (GCG: the search method, the transfer).
- [Many-shot jailbreaking](https://www.anthropic.com/research/many-shot-jailbreaking), Anthropic, 2 April 2024 (256 dialogues, power law, 61% to 2%).
- [Best-of-N Jailbreaking](https://arxiv.org/abs/2412.03556), Hughes et al., 2024 (89% and 78% with 10,000 variants).
- [The Crescendo Multi-Turn LLM Jailbreak Attack](https://arxiv.org/abs/2404.01833), Russinovich et al., 2024 (mechanism, tested models).
- [A StrongREJECT for Empty Jailbreaks](https://arxiv.org/abs/2402.10260), Souly et al., 2024 (overstated attacks, automated judge).
- [HarmBench](https://arxiv.org/abs/2402.04249), Mazeika et al., 2024 (18 methods, 33 models and defences).
- [Not what you've signed up for](https://arxiv.org/abs/2302.12173), Greshake et al., 2023 (indirect prompt injection, attacked applications).
- [AgentDojo](https://arxiv.org/abs/2406.13352), Debenedetti et al., 2024 (97 tasks, 629 security test cases).
- [The WMDP Benchmark](https://arxiv.org/abs/2403.03218), Li et al., 2024 (3,668 questions, RMU).
- Tools: [garak](https://github.com/NVIDIA/garak), [PyRIT](https://github.com/microsoft/PyRIT) and [Inspect](https://inspect.aisi.org.uk/) (what each tool is).
