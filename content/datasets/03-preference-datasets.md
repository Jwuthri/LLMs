# 3. Preference datasets

## Core idea

A preference dataset does not say what the correct answer is. It says which of two answers is better.
This is easier to collect than a correct answer. But each pair carries little information, and some labels are wrong.
Thus you must know where the pairs come from and how many you need.

## The example

This page makes 4 pairs from the 8 samples of the toy dataset of this track (`examples/datasets/toy_dataset.py`). The prompts come from the samples. The second answers and all the quality scores are assumed.

| Pair | Prompt (sample) | Chosen | Rejected |
|---|---|---|---|
| 1 | Count the letters: cat (4) | `3` | `4` |
| 2 | Tell me about the cat (8) | `The cat is a small pet. It sat.` | `I cannot help with that.` |
| 3 | Write Python that counts the words of a text (5) | `len(text.split())` | `text.count(' ') + 1` |
| 4 | Count the words: the cat sat (1) | `3 or 4` | `5` |

Sample 4 of the toy dataset has the wrong answer `4`. Sample 8 has a refusal. In an instruction dataset, you delete these two samples (see [What is a good dataset?](01-quality.md)). Here, they become the rejected answers of pairs 1 and 2.

This page is about the data. The training methods that use the pairs (reward model, DPO, PPO, GRPO) are in [Preference alignment](../scientist/05-preference-alignment.md).

## 1. What a pair is

### Setup

A **preference dataset** is a list of pairs. A **pair** has three parts: a **prompt** (the question), a **chosen answer** (the better answer) and a **rejected answer** (the worse answer).

Assume that each answer has a quality score from 0 (no use) to 1 (perfect). The **margin** of a pair is the quality of the chosen answer minus the quality of the rejected answer. A **tie** is a pair with a margin of 0: the two answers are equally good.

A real dataset usually has no quality score. It has only the label "chosen" or "rejected". The scores of this example show what the label hides.

### Steps

1. **Calculate the margin of each pair.** Margin = chosen quality − rejected quality.

    | Pair | Chosen | Rejected | Margin |
    |---|---|---|---|
    | 1 | 1.0 | 0.0 | 1.0 |
    | 2 | 0.8 | 0.1 | 0.7 |
    | 3 | 0.9 | 0.8 | 0.1 |
    | 4 | 0.3 | 0.0 | 0.3 |

    **Why:** the margin says how clear the lesson of the pair is. In pair 1, one answer is correct and one is wrong. In pair 3, the two programs are almost equally good.

2. **Calculate the mean margin.** $(1.0 + 0.7 + 0.1 + 0.3) / 4 = 0.525$.

    **Why:** the mean margin is one number for the full dataset. A low mean margin shows that many pairs are near a tie.

3. **Drop the pairs with a margin below 0.2.** Pair 3 goes out. 3 pairs stay, and the mean margin goes up to 0.667.

    **Why:** a pair with a margin near 0 teaches almost nothing. The model learns "prefer `len(text.split())`", but the other program is also good. And if the label of such a pair is wrong, nobody sees it.

4. **Look at the level of the chosen answers.** The mean quality of the 4 chosen answers is 0.75. The chosen answer of pair 4 (`3 or 4`) has a quality of only 0.3. The rejected answer of pair 3 has a quality of 0.8.

    **Why:** a pair fixes a direction, not a level. The label "chosen" says "better than the other answer". It does not say "good". The margin filter of step 3 keeps pair 4, because its margin of 0.3 is larger than the margin of pair 3. A model that trains on pair 4 moves toward a bad answer.

<div class="key-point">

A pair says which answer is better, and not how good the answers are. Check the two things separately: the margin (is the difference clear?) and the level (is the chosen answer good?).

</div>

### From toy to real

- The [README of LLM Datasets](https://github.com/mlabonne/llm-datasets) defines preference data as chosen and rejected answers, used to align a model with human style and values. Its table has 10 preference datasets, from 10.9k to 77k pairs.
- Some real datasets keep the margin. In the preference part of HelpSteer3 (40,476 samples), each sample has two responses and a whole-number score from −3 to 3. A negative score means that response 1 is better. A positive score means that response 2 is better. 0 means that they are about equal.
- Some real datasets keep the level. In arena-human-preference-140k (135,634 rows), the vote of the user has 4 possible values: `model_a`, `model_b`, `tie` and `both_bad`. The value `both_bad` is the case of pair 4.
- When you change such data into pairs, you must decide what to do with the ties and the "both bad" rows. One simple decision is to drop them.

## 2. Where pairs come from

### Setup

Pairs come from three sources.

- A **human preference** is the vote of a person. Two models answer the same prompt, and the person chooses the better answer.
- A judge model gives a score to several answers to one prompt. **Binarization** changes these scores into a chosen and a rejected answer.
- A **synthetic pair** is made by construction: take a correct answer, and make a second answer with an error added on purpose. No vote and no score are necessary.

<figure class="diagram" data-src="preference-datasets/sources.svg">
<figcaption>The three sources of pairs. Each source gives the same three fields, but not the same cost and not the same errors.</figcaption>
</figure>

For the toy, assume 3 answers (A, B, C) to the prompt "Tell me about the cat". Assume that a judge model gives them the scores 4.5, 4.0 and 2.0 on a scale from 1 to 5. Here, the margin of a pair is the difference between the two scores.

### Steps

1. **List the pairs that the scores permit.** 3 answers give $3 \times 2 / 2 = 3$ pairs: A against B (margin 0.5), A against C (margin 2.5) and B against C (margin 2.0).

    **Why:** scores carry more information than one vote. With $k$ answers, you get $k(k-1)/2$ possible pairs, and each pair has a margin.

2. **Binarize with the rule "best against worst".** The pair is (chosen A, rejected C). The margin is 2.5.

    **Why:** this rule gives the clearest pairs. But the rejected answers are all very bad, so the model does not learn the small differences between good answers.

3. **Binarize with the rule "best against second".** The pair is (chosen A, rejected B). The margin is 0.5. With the rule "best against one random other answer", the mean margin is $(0.5 + 2.5) / 2 = 1.5$. With all 3 pairs, it is $(0.5 + 2.5 + 2.0) / 3 = 1.667$.

    **Why:** the rule is a choice, and it changes the dataset. From the same scores, the mean margin is 2.5, 1.5 or 0.5. Small margins teach small differences, but an error of 0.5 in one score of the judge can swap the pair. Thus read the dataset card to know the rule.

4. **Count the generated answers for one pair.** Human vote: 2 answers and 1 vote. Judge, best against worst: 3 answers and 3 scores. Judge, all pairs: $3 / 3 = 1$ answer for each pair. Construction: 1 new answer, if the correct answer comes from a dataset that exists.

    **Why:** the cost decides the size of the dataset. A human vote is slow and you cannot generate it. A judge score costs one more model call. Construction is the cheapest source, but the error is always of the type that you added, and never of the type that a real model makes.

<div class="key-point">

The source of a pair decides its cost and its errors. Before you use a preference dataset, find three things on its card: who wrote the answers, who chose, and with which rule.

</div>

### From toy to real

The datasets of the README table, sorted by source. The number of pairs and the note come from the README. A longer list is in the [catalogue](../datasets-catalogue.md).

| Source | Dataset | Pairs |
|---|---|---|
| Human votes | chatbot_arena_conversations | 33k |
| Human votes | HelpSteer3 (14 languages) | 40.5k |
| Judge scores | ultrafeedback-binarized-preferences-cleaned | 61.1k |
| Construction | Code-Preference-Pairs | 53k |
| Construction | Human-Like-DPO-Dataset | 10.9k |
| Mix | orpo-dpo-mix-40k | 44k |
| Mix | Skywork-Reward-Preference-80K-v0.2 | 77k |

- **Human votes.** The card of chatbot_arena_conversations describes 33K cleaned conversations with human votes between two models. They come from 20 models and 13K unique IP addresses, from April to June 2023. The card warns that the dataset contains unsafe conversations. The README also lists arena-human-preference-140k (136k rows, July 2025, CC-BY-4.0) in its table of real conversations.
- **Judge scores.** The UltraFeedback paper describes more than 1 million items of GPT-4 feedback for 250k conversations. The Argilla version makes the pairs from the average of the ratings for each answer, and not from the single overall score. It also removes the prompts that the original dataset reused from the TruthfulQA benchmark. If these prompts stay, a model can see test questions during training.
- **Construction.** The card of Code-Preference-Pairs says that a model adds bugs to a code example. The rejected answer is the code with the bug, and the accepted answer is the code without change. The card of Human-Like-DPO-Dataset describes 10,884 samples on 256 topics. The chosen answer is conversational, and the rejected answer is formal and structured. FalseReject (CC-BY-NC-4.0) is about over-refusal: a model that declines safe prompts that only look sensitive. This is the case of pair 2.
- A pair that the model itself generates is a fourth source. It is in part 1 of [Preference alignment](../scientist/05-preference-alignment.md).

## 3. Label noise

### Setup

**Label noise** is the share of the pairs with a wrong label. A **swapped label** is a pair in which the worse answer has the label "chosen".

Noise has two causes. The first cause is random error: two persons do not always agree. The **agreement between annotators** measures this (see [Evaluation](../scientist/06-evaluation.md)). The second cause is a bias, which is an error in one direction. **Position bias** is one example: the judge or the person prefers the first answer shown.

For the toy, assume two types of answer: type A (short and correct) and type B (long and polite). Assume that A is truly better in 80% of the pairs. The true preference rate is $p = 0.8$. Let $e$ be the share of swapped labels.

### Steps

1. **Calculate the measured rate.** A label says "A" in two cases: A is better and the label is correct, or B is better and the label is swapped.

    $$m = p(1 - e) + (1 - p)e$$

    | Swapped share $e$ | Measured rate $m$ | Bits per label |
    |---|---|---|
    | 0 | 0.8 | 1.0 |
    | 0.1 | 0.74 | 0.531 |
    | 0.2 | 0.68 | 0.278 |
    | 0.3 | 0.62 | 0.119 |

    **Why:** swapped labels pull the rate toward 0.5. With $e = 0.5$, the measured rate is 0.5 for any true rate: the labels say nothing.

2. **Calculate the information of one label.** A label without noise is one answer to a yes-or-no question: 1 bit at most. With noise, it carries $1 - H(e)$ bits, where $H$ is the entropy of a coin with probability $e$ (see [Metrics and losses](../fundamentals/05-metrics.md)). With $e = 0.1$: 0.531 bits. With $e = 0.2$: 0.278 bits.

    **Why:** this is why each pair carries little information. With 20% of swapped labels, you need about $1 / 0.278 = 3.6$ pairs to get the information of one clean pair.

3. **Add position bias.** Take only the ties, where the true preference is 0.5. Assume that the judge chooses the first answer shown in 60% of these pairs. If A is always first, the measured preference for A is 0.6. The judge invented a preference of 0.1.

    **Why:** random noise makes a dataset weaker. A bias makes it wrong: the model learns a preference that does not exist.

4. **Show each pair in the two orders.** With A first, A wins 0.6 of the pairs. With B first, A wins 0.4. The mean is $(0.6 + 0.4) / 2 = 0.5$: the bias is gone. The same answer wins in the two orders in $2 \times 0.6 \times 0.4 = 0.48$ of the pairs. In the other 0.52, the order decides the winner.

    **Why:** a real preference stays when you change the order. Thus a common rule is: keep a pair only if the same answer wins in the two orders, and label the others as ties. The cost is 2 judge calls for each pair.

<div class="llm-widget" data-widget="label-noise"></div>

The widget starts with $p = 0.8$, $e = 0.1$ and 100 pairs. The measured rate is 0.74, and its 95% interval is [0.654, 0.826]. The true rate of 0.8 is inside the interval: with 100 pairs, you cannot see this noise. Part 4 explains the interval.

<div class="key-point">

Random noise pulls each preference toward 0.5 and makes each pair worth less. A bias adds a false preference. More pairs repair the first problem, and only a change of the collection (two orders, better raters) repairs the second.

</div>

### From toy to real

- HelpSteer3 uses agreement as a filter. Its card says that 3 to 5 annotators labelled each sample, and that the dataset keeps the three annotations that agree most.
- Votes of the public do not always have such a filter. A user of an arena can vote without reading the two answers. Read the card to know which votes were removed.
- A judge model has the biases of a model, and position bias is one of them. [Evaluation](../scientist/06-evaluation.md) shows how to check a judge against human labels.
- The largest bias of human votes is about content, not position. Persons often prefer an answer that agrees with them. A model trained on such votes learns to agree. This is sycophancy: see [Alignment](../safety/02-alignment.md).

## 4. How many pairs

### Setup

The **sample size** $n$ is the number of pairs. A preference rate that you measure on $n$ pairs has a sampling error. The **interval of a preference rate** is the range that contains the true rate with a probability of about 95%:

$$p \pm 1.96 \sqrt{p(1 - p) / n}$$

**Coverage of the prompts** says which topics and styles the prompts of the pairs contain, and in which shares.

For the toy, assume a measured rate of 0.7: in 70% of the pairs, the chosen answer is of type A.

### Steps

1. **Calculate the interval for 3 sizes.**

    | Pairs | Half width | 95% interval |
    |---|---|---|
    | 100 | 0.090 | [0.610, 0.790] |
    | 1,000 | 0.028 | [0.672, 0.728] |
    | 10,000 | 0.009 | [0.691, 0.709] |

    **Why:** the half width decreases with $\sqrt{n}$. 100 times more pairs make the interval only 10 times narrower.

2. **Calculate the pairs needed to tell 0.55 from 0.5.** The interval must not contain 0.5, so the half width must be 0.05 or less.

    $$n = 1.96^2 \times 0.55 \times 0.45 / 0.05^2 = 380.3$$

    You need 381 pairs. For a rate of 0.7, 21 pairs are sufficient.

    **Why:** a weak preference needs many pairs. And noise makes each preference weaker (part 3): with 30% of swapped labels, a true rate of 0.8 looks like 0.62.

3. **Count the pairs of each topic.** The toy dataset has 3 topics, with the shares 0.5 (math), 0.25 (code) and 0.25 (chat). Assume a set of 1,000 pairs with 800 math pairs, 150 code pairs and 50 chat pairs. The shares are 0.8, 0.15 and 0.05.

    | Topic | Pairs | 95% interval of 0.7 |
    |---|---|---|
    | math | 800 | [0.668, 0.732] |
    | code | 150 | [0.627, 0.773] |
    | chat | 50 | [0.573, 0.827] |

    **Why:** the total of 1,000 pairs hides the small topics. For chat, the dataset has the precision of 50 pairs. The half width is 0.127, and not 0.028.

4. **Compare the coverage with the use of the model.** If users ask chat questions half of the time, 5% of chat pairs is too small. Add chat pairs, or do not expect a change in chat.

    **Why:** a pair teaches a preference for prompts like its prompt. A set of pairs about the style of an answer (short against long) does not teach which program is correct. A set about one style does not teach another style.

<figure class="plot">
<img src="/figures/preference-datasets/interval.svg" alt="The 95% interval of a measured preference rate of 0.7 against the number of pairs, on a log scale. 100 pairs: 0.610 to 0.790. 1,000 pairs: 0.672 to 0.728. 10,000 pairs: 0.691 to 0.709.">
<figcaption>The interval of a rate of 0.7 gets narrower slowly: each factor of 10 in pairs divides the width by about 3.</figcaption>
</figure>

<div class="key-point">

Count the pairs for each topic and each style, not only the total. The precision of a preference comes from the pairs that show it.

</div>

### From toy to real

- The preference datasets of the README have between 10.9k and 77k pairs. At 10,000 pairs, the interval of a rate of 0.7 is ±0.009. Thus the total is rarely the problem. The noise and the coverage are.
- Many preference datasets cover one thing on purpose. Code-Preference-Pairs covers bugs in code. Human-Like-DPO-Dataset covers a conversational style. FalseReject covers over-refusal. Each one moves one behaviour.
- Other datasets say how they cover several things. The card of HelpSteer3 names 4 domains: General, STEM, Code and Multilingual. A mix such as orpo-dpo-mix-40k puts several DPO datasets together.
- Check the licence before you train. In the README, FalseReject is CC-BY-NC-4.0 (no commercial use). The card of chatbot_arena_conversations gives CC-BY-4.0 for the user prompts and CC-BY-NC-4.0 for the model outputs. The card of Code-Preference-Pairs adds its own conditions.
- The tools that generate, score and filter pairs are in [Tools](04-tools.md). The checks of quality for one sample are in [Quality](01-quality.md).

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import math
import numpy as np

# 1. Margins of the 4 pairs: (chosen quality, rejected quality), assumed
quality = np.array([[1.0, 0.0], [0.8, 0.1], [0.9, 0.8], [0.3, 0.0]])
margin = quality[:, 0] - quality[:, 1]                 # [1.0, 0.7, 0.1, 0.3]
mean_margin = margin.mean()                            # 0.525
kept = margin[margin >= 0.2].mean()                    # 0.667 (pair 3 goes out)

# 2. Binarization of the judge scores of 3 answers
a, b, c = 4.5, 4.0, 2.0
best_worst, best_second = a - c, a - b                 # 2.5 and 0.5
all_pairs = np.mean([a - b, a - c, b - c])             # 1.667

# 3. Swapped labels and position bias
p, e = 0.8, 0.1
measured = p * (1 - e) + (1 - p) * e                   # 0.74
bits = 1 + e * math.log2(e) + (1 - e) * math.log2(1 - e)   # 0.531
first = 0.6
both_orders = (first + (1 - first)) / 2                # 0.5
stable = 2 * first * (1 - first)                       # 0.48

# 4. Interval of a rate, and the number of pairs
half = 1.96 * math.sqrt(0.7 * 0.3 / 1000)              # 0.028 -> [0.672, 0.728]
needed = math.ceil(1.96**2 * 0.55 * 0.45 / 0.05**2)    # 381
```

Real-library sketch (the tests do not run this code). It reads one dataset of the README table with the `datasets` library. Read the card first: the names of the columns change from one dataset to the other.

```python
from datasets import load_dataset

pairs = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
print(pairs.column_names)   # find the prompt, the chosen answer and the rejected answer
```

## Comparison

| Source | Cost of 1 pair (toy) | Typical error | Real example |
|---|---|---|---|
| Human votes | 2 answers, 1 vote | Careless votes, a preference for agreement | chatbot_arena_conversations |
| Judge scores | 3 answers, 3 scores | Biases of the judge, margin set by the rule | ultrafeedback-binarized-preferences-cleaned |
| Construction | 1 new answer | Only the errors that you added | Code-Preference-Pairs |

| Problem | Toy number | Repair |
|---|---|---|
| Margin near 0 | Pair 3: 0.1 | Drop the pair, or label it as a tie |
| Chosen answer is bad | Pair 4: quality 0.3 | Check the level of the chosen answers |
| Swapped labels | 0.8 becomes 0.68 at 20% | More raters, an agreement filter |
| Position bias | 0.5 becomes 0.6 | Show the two orders |
| Small topic | chat: ±0.127 with 50 pairs | Add pairs for this topic |

## Checks and common misconceptions

- **"The chosen answer is a good answer."** It is only better than the rejected answer. Check: read 20 chosen answers. If many are bad, the dataset teaches a direction from a bad starting point.
- **"More pairs repair the noise."** More pairs repair random noise: the interval gets narrower. They do not repair a bias such as position bias or a preference for agreement. Check: find on the card whether the order of the answers was random.
- **"A preference dataset teaches facts."** It teaches which of two answers to prefer, for prompts like its prompts. Check: for new knowledge, use an instruction dataset. For a style or a behaviour, count the pairs that show this style.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no page on this topic. The list of datasets comes from [LLM Datasets](https://github.com/mlabonne/llm-datasets) by the same author. The text and the example are new. Each statement about a real dataset was read in the linked source on 11 November 2026.

- [LLM Datasets, README](https://github.com/mlabonne/llm-datasets), Maxime Labonne (the definition of preference data; the name, number of pairs, date and note of each dataset; the licences of FalseReject and arena-human-preference-140k).
- [chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations), LMSYS, dataset card (33K conversations, 20 models, 13K IP addresses, April to June 2023, the warning, the two licences).
- [arena-human-preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k), LMArena, dataset card (135,634 rows, the 4 values of the vote).
- [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3), NVIDIA, dataset card (40,476 preference samples, the score from −3 to 3, 3 to 5 annotators, the 4 domains).
- [ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned), Argilla, dataset card (binarization from the average of the ratings, the removal of the TruthfulQA prompts).
- [UltraFeedback: Boosting Language Models with Scaled AI Feedback](https://arxiv.org/abs/2310.01377), Cui et al., 2023 (abstract only: more than 1 million items of GPT-4 feedback, 250k conversations).
- [Code-Preference-Pairs](https://huggingface.co/datasets/Vezora/Code-Preference-Pairs), Vezora, dataset card (how the bugs are added, the licence conditions).
- [Human-Like-DPO-Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset), HumanLLMs, dataset card (10,884 samples, 256 topics, the two styles of answer).
- [FalseReject](https://huggingface.co/datasets/AmazonScience/FalseReject), Amazon Science, dataset card (the definition of over-refusal).
