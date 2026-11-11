# 1. What is a good dataset?

## Core idea

A model becomes what its data shows it. Thus the quality of a dataset decides the quality of the model more than the training method does.
A good dataset has three properties: its samples are correct, they are different from each other, and they are hard enough to teach something.
You can measure each property with a number, and you need all three: one high number does not replace a low one.

## The example

This track uses one toy dataset of 8 samples for its 4 lessons. A **sample** is one conversation: an instruction and an answer. All the values are assumed, and no model is called.

| Id | Topic | Instruction | Turns | Tokens | Correct? |
|---|---|---|---|---|---|
| 1 | math | Count the words: the cat sat | 1 | 12 | yes |
| 2 | math | Count the words: the cat | 1 | 10 | yes |
| 3 | math | Count the words: the cat sat. | 1 | 12 | yes |
| 4 | math | Count the letters: cat (answer: 4) | 1 | 10 | no |
| 5 | code | Write Python that counts the words | 1 | 24 | yes |
| 6 | code | Fix this function, then add a test | 2 | 60 | yes |
| 7 | chat | Explain, then shorten, then translate | 3 | 90 | yes |
| 8 | chat | Tell me about the cat (answer: a refusal) | 1 | 14 | no |

Each sample also has an embedding of 2 numbers, as in the [toy example](../toy-example.md). Samples 1 and 3 differ by one full stop.
The three properties come from the list [LLM Datasets](https://github.com/mlabonne/llm-datasets). The [catalogue](../datasets-catalogue.md) shows the public datasets of that list.

<figure class="diagram" data-src="dataset-quality/three-properties.svg">
<figcaption>The three properties, the question of each, and its value for the toy dataset.</figcaption>
</figure>

## 1. Accuracy

### Setup

A sample is **accurate** when its answer is correct and answers the instruction. A **checker** is what decides this: a solver for mathematics, a unit test for code, a person or a judge model for free text.
The **accuracy** of a dataset is the share of its samples that pass the checker.

### Steps

1. **Run the checker on each sample.** 6 of the 8 samples pass. Sample 4 says that `cat` has 4 letters. Sample 8 is a refusal, which does not answer the instruction.

    $$\text{accuracy} = 6 / 8 = 0.75$$

    **Why:** training makes the answers of the data more probable, also the wrong ones. With this data, the model learns a wrong fact in 1 sample of 4.

2. **Calculate how sure the number is.** With $n$ samples, the standard error is $\sqrt{p(1-p)/n}$.

    $$\sqrt{\frac{0.75 \times 0.25}{8}} = 0.153, \qquad 0.75 \pm 1.96 \times 0.153 = [0.45,\ 1.0]$$

    **Why:** 8 samples say almost nothing: the true accuracy can be 0.45. A quality number without its sample size is not a result.

3. **Split by topic.** Math: 3 of 4. Code: 2 of 2. Chat: 1 of 2.

    **Why:** the mean hides where the errors are. Here the code samples are clean, because a unit test is a strong checker. The chat samples have no such test.

4. **Calculate how many samples you must read.** A dataset of 1 million samples is too large to check by hand. Read a random part of it. To know an accuracy near 0.75 within ± 0.05, you need

    $$n = \frac{1.96^2 \times 0.75 \times 0.25}{0.05^2} = 289 \text{ samples}$$

    For ± 0.02 you need 1,801 samples.

    **Why:** the number of samples to read depends on the precision that you want, not on the size of the dataset. 289 samples are one afternoon of work, for a dataset of any size.

<div class="key-point">

Accuracy is the share of samples that pass a checker. Measure it on a random part of the dataset, give the interval, and split it by topic. The topics with no automatic checker are where the errors hide.

</div>

### From toy to real

- The list LLM Datasets defines accuracy as samples that are factually correct and relevant to their instruction. It names a solver for mathematics and unit tests for code as ways to check.
- Many public datasets are written by a model. Then the checker is the only thing between an error of that model and your model. The lesson [Post-training datasets](../scientist/03-datasets.md) builds such a checker.
- Some dataset names say "verified" or "filtered". Read the dataset card to see what was checked and how: [Instruction datasets](02-instruction-datasets.md) shows how to read a card.

## 2. Diversity

### Setup

A dataset is **diverse** when its samples cover many different cases: topics, formats, lengths, languages. A model that never saw a type of request in training is **out of distribution** for that request.
You can measure diversity in two ways: with labels (the share of each topic) or without labels (the distance between the embeddings).

### Steps

1. **Count the share of each topic.** Math: 4 of 8 = 0.5. Code: 0.25. Chat: 0.25.

    **Why:** the shares show what the model sees most. Half of the toy data is about counting.

2. **Change the shares into one number.** The entropy (see [Metrics and losses](../fundamentals/05-metrics.md)) of the shares is

    $$H = -(0.5 \ln 0.5 + 0.25 \ln 0.25 + 0.25 \ln 0.25) = 1.040$$

    The maximum for 3 topics is $\ln 3 = 1.099$, when the three shares are equal. The toy dataset is at $1.040 / 1.099 = 0.946$ of the maximum. A dataset with one topic has an entropy of 0.

    **Why:** one number lets you compare two versions of a dataset. It needs topic labels, and it only sees the topics that you named.

3. **Measure the distance between the embeddings.** The cosine distance of two samples is 1 minus the cosine similarity of their embeddings. The mean over all the 28 pairs of the toy dataset is 0.263. For the 4 math samples alone it is 0.028.

    **Why:** this needs no labels, so it also finds a lack of diversity inside one topic. The math samples are almost the same request four times.

4. **Find the samples that are almost equal.** The pair with the highest similarity is samples 1 and 3: 0.9998. For comparison, samples 1 and 2 have 0.993, and samples 1 and 5 (math and code) have 0.198.

    **Why:** a **near duplicate** adds a sample to the count and nothing to the content. The dataset has 8 samples, but only 7 different ones. The lesson [The tools](04-tools.md) shows how to remove near duplicates at scale.

<div class="key-point">

Diversity is coverage. Topic shares show it for the topics that you named. Embedding distances show it without labels, and they find near duplicates. A large dataset can be a small dataset that is repeated.

</div>

### From toy to real

- The list LLM Datasets says that a dataset must cover as many use cases as possible, so that the model is never out of distribution, and that this gives a better generalization.
- This is why the list has families (mathematics, code, function calling, many languages): no single dataset covers all of them. [Instruction datasets](02-instruction-datasets.md) shows how to mix them.
- Real conversations of users are the reference for "what cases exist". The same lesson compares a mix with them.

## 3. Complexity

### Setup

A sample is **complex** when the model must do some work to answer it: many turns, a long context, several steps of reasoning, a constraint to obey. A dataset of only easy samples teaches what the model already knows.
A **turn** is one message of the user with its answer.

### Steps

1. **Count the turns.** The mean is $(1+1+1+1+1+2+3+1) / 8 = 1.375$ turns. Only 2 of the 8 samples (0.25) have more than one turn.

    **Why:** real use has follow-up questions. A model that is trained on single questions only learns to forget the conversation.

2. **Count the tokens, and do not trust the mean.** The mean is 29 tokens for each sample. The median is 13. The two longest samples (60 and 90 tokens) have 0.647 of all the tokens.

    **Why:** the loss is calculated for each token. Thus the two long samples count for two thirds of the training signal, although they are a quarter of the samples. Look at the distribution, not only at the mean.

3. **Measure the difficulty with a baseline model.** Assume that a small model already answers samples 1, 2, 3 and 5 correctly: 4 of 8 = 0.5 of the dataset.

    **Why:** length is not difficulty. A sample that the starting model already solves gives almost no gradient. The share that a baseline solves tells you how much of the dataset is new for the model.

4. **Keep the hard samples correct.** The hard samples are also the ones where the answers are most often wrong. In the toy dataset, the three samples with 60 tokens or more, or with a refusal, are in the two topics with the weakest checkers.

    **Why:** complexity and accuracy pull in opposite directions. A filter for accuracy removes hard samples first, so check the complexity again after each filter.

<div class="key-point">

Complexity is what is left to learn. Measure it with turns, with the distribution of tokens and with the share that a baseline model already solves. Easy samples make a dataset large, not useful.

</div>

### From toy to real

- The list LLM Datasets asks for samples that are multi-turn, multilingual and well written, with step-by-step reasoning where it fits.
- Its tables have a column "Thinking": it says if the answers of a dataset contain reasoning text before the final answer. This is one visible form of complexity.
- The lesson [Post-training datasets](../scientist/03-datasets.md) shows how to make samples harder (data enhancement).

## 4. One decision from several checks

### Setup

No single check measures the three properties. In practice you combine cheap checks for all the samples with expensive checks for a few.
A **rule** is a fixed test on the text, for example "the answer contains a refusal". A **judge model** is a model that reads a sample and gives it a score. A **manual review** is a person who reads samples.
Assume that a judge model gives these scores from 1 to 5.

| Sample | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Judge score | 5 | 2 | 5 | 4 | 4 | 5 | 4 | 1 |
| Checker says correct | yes | yes | yes | no | yes | yes | yes | no |

### Steps

1. **Use the judge alone, with a threshold of 3.** It keeps samples 1, 3, 4, 5, 6 and 7. It removes sample 2, which is correct, and keeps sample 4, which is wrong. The accuracy of the kept samples is $5 / 6 = 0.833$.

    **Why:** a judge is a model, and it makes the two types of error. Here it misses a wrong count of letters and dislikes a correct short answer.

2. **Measure the judge against the checker.** The two agree on 6 of the 8 samples: 0.75. By chance they would agree on 0.625 of them, so Cohen's kappa (see [Evaluation](../scientist/06-evaluation.md)) is

    $$\kappa = \frac{0.75 - 0.625}{1 - 0.625} = 0.333$$

    **Why:** an agreement of 0.75 looks good, but most of it comes from the fact that most samples are good. The kappa shows that this judge adds little. Check a judge on samples with known labels before you use it on a million samples.

3. **Use the checks in order, the cheapest first.**

    | Step | Samples that stay | Accuracy | Topic entropy |
    |---|---|---|---|
    | Start | 8 | 0.75 | 1.040 |
    | Rule: remove refusals | 7 | 0.857 | 0.956 |
    | Checker: remove wrong answers | 6 | 1.0 | 1.011 |
    | Remove near duplicates | 5 | 1.0 | 1.055 |

    **Why:** a rule costs almost nothing, so it goes first. Each step removes a different type of problem: the rule finds sample 8, the checker finds sample 4, and the similarity finds sample 3.

4. **Read what the filters did to the three properties.** The accuracy went from 0.75 to 1.0. The topic shares became 0.4, 0.4 and 0.2, and the entropy went from 1.040 to 1.055. The dataset lost 3 of its 8 samples, and chat is down to 1 sample.

    **Why:** each filter changes all three properties, not only the one that it targets. The refusal rule lowered the diversity (0.956) before the next steps raised it again. Measure all three after each step.

5. **Read some samples yourself.** Read 20 kept samples and 20 removed samples.

    **Why:** a number tells you how many samples a filter removed. Only a person can see that it removed them for the wrong reason.

<figure class="plot">
<img src="/figures/dataset-quality/judge-threshold.svg" alt="Share of samples kept and accuracy of the kept samples, against the lowest judge score that is kept, from 1 to 5. At a threshold of 3: 0.75 kept, accuracy 0.833. At 5: 0.375 kept, accuracy 1.0.">
<figcaption>A stricter judge threshold raises the accuracy and removes samples. At 5 it keeps 3 of the 8.</figcaption>
</figure>

<div class="llm-widget" data-widget="quality-gate"></div>

<div class="key-point">

Use rules, a checker, a judge model and your own eyes together, the cheapest first. After each filter, measure the three properties again: a filter that raises one of them can lower the two others.

</div>

### From toy to real

- The list LLM Datasets says that quality comes from a combination of manual review, rules, and scores from judge models or reward models. A reward model is in [Preference alignment](../scientist/05-preference-alignment.md).
- Judge models have their own errors and biases. [Preference datasets](03-preference-datasets.md) shows the effect of wrong labels, and [Metrics and losses](../fundamentals/05-metrics.md) shows how to measure a detector.
- The tools for each of these checks are in the lesson [From raw text to a dataset: the tools](04-tools.md).

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import math
import numpy as np

topics = ["math"] * 4 + ["code"] * 2 + ["chat"] * 2
correct = [1, 1, 1, 0, 1, 1, 1, 0]
tokens = [12, 10, 12, 10, 24, 60, 90, 14]
E = np.array([[1.0, 0.1], [0.9, 0.2], [0.98, 0.12], [0.8, 0.4], [0.1, 1.0], [0.2, 0.9], [0.6, 0.7], [0.5, 0.5]])

# 1. Accuracy, its interval, and the size of a manual check
p = sum(correct) / 8                                         # 0.75
se = math.sqrt(p * (1 - p) / 8)                              # 0.153
audit = math.ceil(1.96**2 * p * (1 - p) / 0.05**2)           # 289 samples for +/- 0.05

# 2. Diversity: topic entropy and mean cosine distance
shares = [topics.count(t) / 8 for t in ("math", "code", "chat")]
H = -sum(s * math.log(s) for s in shares)                    # 1.040 of a maximum of 1.099
unit = E / np.linalg.norm(E, axis=1, keepdims=True)
sim = unit @ unit.T
distance = np.mean(1 - sim[np.triu_indices(8, k=1)])         # 0.263; sim[0, 2] = 0.9998

# 3. Complexity: the mean hides the distribution
mean, median = np.mean(tokens), np.median(tokens)            # 29 and 13

# 4. Judge against checker
judge = [5, 2, 5, 4, 4, 5, 4, 1]
keep = [j >= 3 for j in judge]
agree = np.mean([k == bool(c) for k, c in zip(keep, correct)])   # 0.75
chance = 0.75 * 0.75 + 0.25 * 0.25                           # 0.625
kappa = (agree - chance) / (1 - chance)                      # 0.333
```

## Comparison

| Property | Question | How to measure it | Toy value | What goes wrong without it |
|---|---|---|---|---|
| Accuracy | Is each answer right? | Checker on a random part, with an interval | 0.75 | The model learns the errors |
| Diversity | Are the samples different? | Topic entropy, embedding distance, near duplicates | 1.040; 0.263 | The model fails on cases that it never saw |
| Complexity | Is there something to learn? | Turns, token distribution, share that a baseline solves | 1.375 turns; 0.5 solved | The dataset is large and teaches little |
| Combined checks | Which samples stay? | Rules, checker, judge, manual review | 5 of 8 stay | One filter breaks what it does not measure |

## Checks and common misconceptions

- **"More data is better."** The toy dataset is better with 5 samples than with 8: the accuracy goes from 0.75 to 1.0 and the entropy from 1.040 to 1.055. Check: measure the three properties before and after you add data.
- **"A judge model checked the dataset, so it is clean."** The toy judge kept a wrong sample and removed a correct one, with a kappa of 0.333. Check: ask how the judge was tested against human labels.
- **"The mean length shows how complex the data is."** The mean was 29 tokens and the median 13. Check: look at the distribution of lengths and of turns, and at the share that a baseline model already solves.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no page on this topic. The three properties and the ways to check them come from [LLM Datasets](https://github.com/mlabonne/llm-datasets) by the same author. The text and the example are new. Each statement about that list was read in it on 11 November 2026.

- [LLM Datasets](https://github.com/mlabonne/llm-datasets), Maxime Labonne (the definitions of accuracy, diversity and complexity; the combination of manual review, rules and judge models; the "Thinking" column of the tables).
