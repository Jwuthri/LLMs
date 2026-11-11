# 6. Evaluation

## Core idea

An evaluation gives a number to the question "how good is this model?".
The number can come from three sources: a fixed test with known answers, people, or a second model.
Each number has an error. A good evaluation states this error, and it shows what to fix next.

## The example

This page uses a small test: 10 multiple-choice questions with 4 choices each, and two models. The results are assumed. `1` is a correct answer.

| Question | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Correct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Model A | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 7 |
| Model B | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 6 |

A model that picks a choice at random gets 25%.

## 1. Automated benchmarks

### Setup

A **benchmark** is a fixed set of questions with known answers, plus a rule that gives a score. A program does the check, so the test is cheap and you can repeat it for each new model.

### Steps

1. **Calculate the accuracy.** The **accuracy** is the share of correct answers. Model A: 7 / 10 = 0.70. Model B: 6 / 10 = 0.60.

    **Why:** one number lets you compare models. But it hides how sure you can be.

2. **Calculate the standard error.** The **standard error (SE)** says how much the accuracy $p$ changes if you take a different set of $n$ questions.

    $$SE = \sqrt{\frac{p(1 - p)}{n}}, \qquad SE_A = \sqrt{\frac{0.7 \cdot 0.3}{10}} = 0.145, \qquad SE_B = \sqrt{\frac{0.6 \cdot 0.4}{10}} = 0.155$$

    **Why:** the 10 questions are a sample of all the questions that you could ask. A different sample gives a different accuracy.

3. **Calculate the 95% confidence interval.** A **confidence interval** is a range that probably contains the true accuracy. The usual formula is $p \pm 1.96 \cdot SE$.

    | Model | Accuracy | ± | Interval |
    |---|---|---|---|
    | A | 0.70 | 0.284 | 0.416 to 0.984 |
    | B | 0.60 | 0.304 | 0.296 to 0.904 |

    The difference is $0.10 \pm 0.416$. This range contains 0. This formula treats the two results as independent. The two models answer the same questions, so a paired test (compare the two models question by question) gives a narrower range. Here the paired range also contains 0.

    **Why:** the two intervals overlap almost fully. The models disagree on only 3 questions: A alone is correct on 2, B alone on 1. Ten questions cannot say which model is better.

4. **Do the same with n = 1,000.** Assume the same accuracies on 1,000 questions.

    | Model | Accuracy | ± | Interval |
    |---|---|---|---|
    | A | 0.70 | 0.028 | 0.672 to 0.728 |
    | B | 0.60 | 0.030 | 0.570 to 0.630 |

    The difference is $0.10 \pm 0.042$. It does not contain 0. The smallest test that separates these two models has 173 questions.

    **Why:** the standard error falls with $\sqrt{n}$. 100 times more questions give an interval that is 10 times narrower.

5. **Check for contamination.** **Contamination** means that test questions were in the training data. Assume that questions 1, 2 and 5 were in the training data of model A. Without them, A has 4 / 7 = 0.571 and B has 4 / 7 = 0.571.

    **Why:** a model can memorize an answer. Then the benchmark measures memory, not skill. Here the full lead of model A came from the 3 questions that it saw before.

<figure class="plot">
<img src="/figures/evaluation/interval-width.svg" alt="Half-width of the 95% interval against the number of questions. For the difference of the two models: 0.416 at n = 10, 0.100 at n = 173, 0.042 at n = 1,000.">
<figcaption>The interval of the difference becomes narrower than the measured gap of 0.10 only from 173 questions.</figcaption>
</figure>

<div class="key-point">

A benchmark score without an interval is half of the result. Report the accuracy, the number of questions and the interval.

</div>

### From toy to real

- MMLU is a common benchmark: 57 tasks, 4 choices, 14,042 test questions in the released data (the paper says 14,079). At an accuracy of 0.70, the interval is ± 0.008. A gap of 1 point between two models is near this limit.
- The formula above is rough for 10 questions (the interval of A almost reaches 1). The conclusion does not change.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [Lighteval](https://github.com/huggingface/lighteval) run many benchmarks with one command.
- Benchmarks fit tasks with one correct answer. They are weak for open or creative tasks. Public questions also leak into training data with time.
- A model can also identify that it is in a test and change its behaviour. The page [AI safety](09-ai-safety.md) gives the numbers.

## 2. Human evaluation

### Setup

People read the answers and judge them. There are three levels: a quick look by one person (a "vibe check"), annotation with written guidelines, and public votes in an **arena**. In an arena, two anonymous models answer the same prompt, and the user picks the better answer.

### Steps

1. **Arena: calculate the expected score.** The **Elo rating** is one number for each model. Model A has 1,000 points and model B has 1,100.

    $$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}} = \frac{1}{1 + 10^{0.25}} = 0.360$$

    **Why:** the gap between two ratings predicts the result. A model with 100 points more wins 64% of the battles.

2. **Update the ratings after one battle.** Model A wins, so its score is $S_A = 1$. With $K = 32$:

    $$R_A' = 1000 + 32 \cdot (1 - 0.360) = 1020.5, \qquad R_B' = 1100 + 32 \cdot (0 - 0.640) = 1079.5$$

    If B wins, A loses only 11.5 points (988.5).

    **Why:** a surprise moves the ratings more than an expected result. $K$ sets how fast a rating can change.

3. **Annotation: measure the observed agreement.** An **annotator** is a person who labels answers. Two annotators label the same 50 answers as "good" or "bad".

    | | Annotator 2: good | Annotator 2: bad |
    |---|---|---|
    | Annotator 1: good | 20 | 5 |
    | Annotator 1: bad | 10 | 15 |

    They agree on $20 + 15 = 35$ answers: $p_o = 35 / 50 = 0.70$.

    **Why:** if two people give different labels to the same answer, the label is noise. You must know this before you use the labels.

4. **Remove the agreement by chance.** Annotator 1 says "good" for 25 / 50 = 0.5 of the answers. Annotator 2 says "good" for 30 / 50 = 0.6. **Cohen's kappa** is:

    $$p_e = 0.5 \cdot 0.6 + 0.5 \cdot 0.4 = 0.50, \qquad \kappa = \frac{p_o - p_e}{1 - p_e} = \frac{0.70 - 0.50}{1 - 0.50} = 0.40$$

    **Why:** two people who label at random with these rates agree on 50% of the answers. Kappa counts only the agreement above chance: 0 is chance, 1 is full agreement. An agreement of 70% looks good, but 0.40 says that the guidelines are not clear.

<div class="key-point">

Human judgment is the reference for subjective quality. It is only useful when you measure how much the people agree, or when you have many votes.

</div>

### From toy to real

- Chatbot Arena shows the model names only after the vote. Its paper reports more than 240,000 votes. It first reported Elo scores. It now fits a Bradley-Terry model on all the votes: the model of the [reward model](05-preference-alignment.md) page.
- People are good judges of tone and usefulness. They are less reliable for facts: a wrong answer that sounds sure can win the vote.
- Human evaluation is slow and expensive, so teams use it on a small sample.

## 3. Model-based evaluation

### Setup

A **judge model** is an LLM that reads an answer and grades it. Its prompt contains a **rubric**: a written list of criteria with points.

### Steps

1. **Score one answer with a rubric.** The judge gives 2 / 2 for "correct", 1 / 2 for "complete" and 2 / 2 for "clear". The score is 5 / 6 = 0.833.

    **Why:** separate criteria make the judge say what is wrong, and they make the scores of different answers comparable.

2. **Or compare two answers.** The judge reads the question, then the answer of model A, then the answer of model B. It says which is better. On the 10 questions, A wins 7 and B wins 3.

    **Why:** "which is better?" is an easier question than "how many points?". The verdicts are more stable.

3. **Swap the order and ask again.** Now the answer of B comes first.

    | Question | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A wins |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | A shown first | A | **A** | B | A | **A** | B | A | B | **A** | A | 7 |
    | B shown first | A | **B** | B | A | **B** | B | A | B | **B** | A | 4 |

    **Why:** a fair judge gives the same winner in the two orders. The swap is a test of the judge, not of the models.

4. **Count the flips.** A **flip** is a pair that changes its winner. There are 3 flips in 10 pairs (questions 2, 5 and 9): a flip rate of 30%. In each flip, the judge picked the answer that it read first. The first answer wins 13 of the 20 verdicts (65%). Count a flip as a tie: A wins 4, B wins 3, 3 ties.

    **Why:** this is **position bias**. One order says "A wins 7 to 3". The other order says "B wins 6 to 4". With one order only, you report a clear winner that does not exist.

<div class="key-point">

A judge model is fast and cheap, and it can grade open answers. It is also a model with its own errors, so test the judge before you trust its scores.

</div>

### From toy to real

- Check the judge against people: grade a sample by hand, then calculate the agreement (the kappa of section 2).
- Judges have other known biases: they prefer long answers, and they prefer answers that their own model wrote.
- Scores can change between two runs. Use a temperature of 0, a clear rubric and some graded examples in the prompt.

## 4. Feedback signal

### Setup

An accuracy of 0.70 says how good the model is. It does not say what to fix. For that, read the wrong answers and put each one in a category.
Assume that model A gets 0.70 on 1,000 questions: 300 errors.

<figure class="diagram" data-src="evaluation/loop.svg">
<figcaption>Evaluation is a loop: the errors of one round decide the training data of the next round.</figcaption>
</figure>

### Steps

1. **Make the error table.**

    | Category | Questions | Errors | Error rate | Share of all errors | Gain if the rate is halved |
    |---|---|---|---|---|---|
    | Multi-step instructions | 200 | 120 | 60% | 40.0% | +0.06 |
    | Missing knowledge | 400 | 80 | 20% | 26.7% | +0.04 |
    | Arithmetic | 150 | 60 | 40% | 20.0% | +0.03 |
    | Output format | 250 | 40 | 16% | 13.3% | +0.02 |

    **Why:** one number becomes four smaller numbers, and each one points to a different fix.

2. **Read the error rate, not only the count.** "Missing knowledge" has more errors than "arithmetic" (80 and 60). But its error rate is half (20% and 40%).

    **Why:** the count depends on how many questions of each category the benchmark contains. The rate shows the skill of the model.

3. **Estimate the gain.** If the error rate for multi-step instructions falls from 60% to 30%, 60 errors go away. The accuracy goes from 0.70 to 0.76.

    **Why:** work first where the rate and the count are both high. Here it is the same category.

4. **Feed the result back.** Write or generate new training examples with multi-step instructions (see [post-training datasets](03-datasets.md)). Train again. Then evaluate again on questions that the training never saw.

    **Why:** if you train on the test questions, you contaminate your own benchmark (step 5 of section 1). The score goes up and the model does not get better.

<div class="key-point">

The most useful output of an evaluation is not the score. It is the list of errors, sorted by cause.

</div>

### From toy to real

- Each category has its own fix. Missing knowledge needs new data or [retrieval](../engineer/03-rag.md). A wrong format needs more examples of the format. Overfitting needs other [training parameters](04-fine-tuning.md).
- Start by hand on about 100 errors to find the categories. A judge model can then label the rest.
- Goodhart's law: "When a measure becomes a target, it ceases to be a good measure." Keep one test set private and use it rarely.

## Minimal code

This code gives the same numbers as sections 1 and 2.

```python
import numpy as np

a = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 0])
p, n = a.mean(), len(a)                        # 0.7, 10
se = np.sqrt(p * (1 - p) / n)                  # 0.145
low, high = p - 1.96 * se, p + 1.96 * se       # 0.416, 0.984

expected = 1 / (1 + 10 ** ((1100 - 1000) / 400))   # 0.360
new_rating = 1000 + 32 * (1 - expected)        # 1020.5

table = np.array([[20, 5], [10, 15]])          # annotator 1 x annotator 2
p_o = np.trace(table) / table.sum()            # 0.70
p_e = (table.sum(1) / 50) @ (table.sum(0) / 50)    # 0.50
kappa = (p_o - p_e) / (1 - p_e)                # 0.40
```

Real-library sketch (the tests do not run this code):

```bash
lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct --tasks mmlu --batch_size 8
```

## Comparison

| Method | Who decides | Toy result | Cost | Good for | Risk |
|---|---|---|---|---|---|
| Automated benchmark | A program with an answer key | 0.70 ± 0.284 with 10 questions | Lowest | Tasks with one correct answer | Contamination; a small test proves nothing |
| Human evaluation | People | Elo 1,000 → 1,020.5; kappa = 0.40 | Highest | Tone, usefulness, open tasks | Slow; people disagree; weak on facts |
| Judge model | An LLM with a rubric | 3 flips in 10 pairs | Low | Open answers at scale | Position bias; prefers its own outputs |
| Error analysis | You, then a judge model | Multi-step: 60% errors | Your time | The decision of what to fix | Tuning to the test set |

## Checks and common misconceptions

- **"Model A has 70% and model B has 60%, so A is better."** With 10 questions the difference is $0.10 \pm 0.416$. Check: calculate the interval before you compare. This gap needs 173 questions or more.
- **"A higher score on a public benchmark means a better model."** The model can know the questions. Check: test again on new questions that you wrote after the training data was collected.
- **"A judge model is objective."** In the example, the order of the answers changes the winner. Check: judge each pair in the two orders, and compare the judge with human labels on a sample.

## Sources

This page follows section 6 of "The LLM Scientist" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [LLM evaluation guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook), Hugging Face.
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard), Hugging Face.
- [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), EleutherAI.
- [Lighteval](https://github.com/huggingface/lighteval), Hugging Face.
- [Chatbot Arena](https://lmarena.ai/), LMSYS.
- [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300), Hendrycks et al., 2020 (MMLU: 57 tasks, 14,079 test questions in the paper; the [released test set](https://huggingface.co/datasets/cais/mmlu) has 14,042).
- [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640), Evan Miller, 2024 (standard errors and intervals for benchmark scores).
- [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132), Chiang et al., 2024 (anonymous battles, 240K votes, Bradley-Terry scores).
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685), Zheng et al., 2023 (position, verbosity and self-enhancement biases).
- [A Coefficient of Agreement for Nominal Scales](https://doi.org/10.1177/001316446002000104), Jacob Cohen, 1960 (kappa).
