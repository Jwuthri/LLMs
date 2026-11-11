# 5. Metrics and losses

## Core idea

A metric changes the behaviour of a model into one number, so that you can compare two models or two days of training.
Each metric answers one question and ignores the others. Thus the first step is to ask what you have: token probabilities, two distributions, scores with labels, a written text, or a ranked list.
A **loss** is a metric that training makes smaller. The other metrics are for reading only.

## The example

This page uses the [toy example](../toy-example.md) and four small data sets. Values marked "assume" are made for this page.

| Part | Data |
|---|---|
| 1 | The toy probabilities [0.099, 0.242, 0.659] after "the cat". The correct token is `sat` |
| 2 | A second model with the probabilities [0.186, 0.307, 0.506] (assumed) |
| 3 | A detector gives a score to 10 requests: 4 harmful, 6 harmless (assumed) |
| 4 | The reference text "the cat sat" and the model text "the cat cat" |

<figure class="diagram" data-src="metrics/chooser.svg">
<figcaption>What you have decides which metric you can use. The number is the part of this page.</figcaption>
</figure>

## 1. Loss and perplexity

### Setup

A language model gives a probability to each possible next token. The **cross-entropy loss** of one prediction is the negative logarithm of the probability that the model gave to the correct token. The page [Mathematics](01-math.md) shows where it comes from.
Use it when you have the probabilities of the model and the correct token: in training, and to compare two models on the same text.

### Steps

1. **Calculate the loss of one token.** The model gives `sat` the probability 0.659.

    $$L = -\ln 0.659 = 0.417$$

    **Why:** the logarithm makes the loss 0 for a probability of 1, and very large for a probability near 0. A model that is sure and wrong pays the most.

2. **Change the unit to bits.** The natural logarithm gives **nats**. Divide by $\ln 2$ to get bits: $0.417 / 0.693 = 0.602$ bits.

    **Why:** papers use the two units. A loss of 0.602 bits means that the correct token costs 0.602 yes-or-no questions. Always check the unit before you compare two numbers.

3. **Take the mean over the tokens.** Assume that the model gives the correct token the probabilities 0.659, 0.5 and 0.25 at three positions. The losses are 0.417, 0.693 and 1.386. The mean is 0.832.

    **Why:** the loss of a text is a mean for each token, so texts of different lengths are comparable. One bad token (1.386) moves the mean more than one good token.

4. **Change the loss into a perplexity.** **Perplexity** is $e^{L}$.

    $$e^{0.832} = 2.298$$

    **Why:** the number has a simple meaning: the model is as unsure as if it chose between 2.3 equally probable tokens. A model that knows nothing about a vocabulary of 3 words has a loss of $\ln 3 = 1.099$ and a perplexity of 3.

5. **Compare the loss with the accuracy.** The top token is correct at position 1 (0.659). At position 3 (0.25) it is probably wrong. Accuracy counts only "top token correct or not". The loss also counts how sure the model was.

    **Why:** the loss moves when the model becomes a little more sure, so the gradient has a direction. Accuracy changes in jumps, so you cannot train on it.

<div class="key-point">

Loss and perplexity are the same information: perplexity is $e^{\text{loss}}$. Use them to follow training and to compare models on the same text with the same tokenizer. They do not say if an answer is useful.

</div>

### From toy to real

- In PyTorch, `CrossEntropyLoss` takes the logits, not the probabilities. Its default is the mean over the tokens (`reduction='mean'`), and it ignores the positions with the label -100 (`ignore_index`). This is how padding is removed from the loss.
- Compare the training loss with the loss on held-out text. When the first goes down and the second goes up, the model memorises: see [Neural networks](03-neural-networks.md).
- You cannot compare the perplexity of two models with different tokenizers: a model with shorter tokens has more, easier predictions for the same text.
- A lower loss does not always mean a better assistant. For that, use the metrics of part 4 and the methods of [Evaluation](../scientist/06-evaluation.md).

## 2. Entropy and KL divergence

### Setup

Sometimes there is no single correct token: you have two full distributions and you want their distance. The toy model is $p$ = [0.099, 0.242, 0.659]. A second model is $q$ = [0.186, 0.307, 0.506]: think of a smaller student model, or of the same model after quantization.
The **entropy** $H(p)$ is the mean loss of a model that predicts its own distribution: how unsure $p$ is. The **KL divergence** is the additional loss when you use $q$ where $p$ is the truth.

$$\text{KL}(p \,\|\, q) = \sum_i p_i \ln \frac{p_i}{q_i}$$

### Steps

1. **Calculate the entropy of p.** $H(p) = -\sum_i p_i \ln p_i = 0.847$.

    **Why:** this is the part of the loss that no model can remove, if $p$ is the truth. A distribution with all the probability on one token has an entropy of 0.

2. **Calculate the cross-entropy between p and q.** $H(p, q) = -\sum_i p_i \ln q_i = 0.900$.

    **Why:** this is the mean loss of $q$ when the tokens come from $p$. Part 1 was the special case in which $p$ is 1 for the correct token.

3. **Subtract.** $\text{KL}(p \,\|\, q) = 0.900 - 0.847 = 0.053$.

    **Why:** the KL divergence is only the part that comes from the difference between the two models. It is 0 when $q = p$, and never negative.

4. **Change the order.** $\text{KL}(q \,\|\, p) = 0.058$, not 0.053. The **Jensen-Shannon divergence** is the mean KL of $p$ and of $q$ to their mid point: 0.014, the same in the two orders.

    **Why:** KL is not a distance: the order matters. The first distribution gives the weights of the sum, so it decides which tokens count.

5. **Give q almost no probability for one token.** Assume $q'$ = [0.001, 0.299, 0.7]. Then $\text{KL}(p \,\|\, q') = 0.362$ and $\text{KL}(q' \,\|\, p) = 0.100$. The token `the` alone gives $0.099 \times \ln(0.099 / 0.001) = 0.452$ in the first sum.

    **Why:** $\text{KL}(p \,\|\, q)$ punishes a $q$ that forgets something that $p$ does. In the other order, that same token counts almost nothing, because $q'$ gives it a weight of 0.001. Choose the order from the error that you want to prevent.

<div class="key-point">

Cross-entropy = entropy + KL. Use KL when you compare a model with a reference model, not with a correct answer. It is not symmetric, and it becomes very large when the second distribution has a probability near 0 where the first one does not.

</div>

### From toy to real

- **Distillation** trains a small model to copy the full output distribution of a large model: the loss is a KL divergence between the two.
- **A penalty in training.** [Preference alignment](../scientist/05-preference-alignment.md) keeps the trained policy near a reference model, and PPO uses a KL term for this. The safety lessons use the same idea to say "this model did not change much".
- **A check after a change.** After [quantization](../scientist/07-quantization.md) or steering, the mean KL between the old and the new next-token distributions shows how much the behaviour moved, without labels.
- In PyTorch, `KLDivLoss` has a trap: its documentation says that `reduction='mean'` does not return the true KL value, and that `'batchmean'` agrees with the mathematical definition.

## 3. Scores with labels: precision, recall, AUROC, calibration

### Setup

A detector (a classifier, a probe, a judge model) gives each item a score. You also have the true label. This is the case for a harmful-request filter, a hallucination detector, or a probe in [Interpretability](../safety/05-interpretability.md).
Assume 10 requests. The 4 harmful ones have the scores 0.9, 0.8, 0.6, 0.3. The 6 harmless ones have 0.7, 0.4, 0.35, 0.2, 0.1, 0.05. The detector flags a request at a score of 0.5 or more.

### Steps

1. **Count the four results.** This table is the **confusion matrix**.

    | | Flagged | Not flagged |
    |---|---|---|
    | Harmful (4) | 3 true positives | 1 false negative |
    | Harmless (6) | 1 false positive | 5 true negatives |

    **Why:** the two types of error have different costs. One number that mixes them hides the one that you care about.

2. **Calculate precision, recall and F1.** **Precision** is the share of the flags that are correct: $3 / 4 = 0.75$. **Recall** (or true positive rate) is the share of the harmful requests that are found: $3 / 4 = 0.75$. The **F1 score** is their harmonic mean:

    $$F_1 = \frac{2 \times 0.75 \times 0.75}{0.75 + 0.75} = 0.75$$

    The **accuracy** is $8 / 10 = 0.8$. The **false positive rate** is $1 / 6 = 0.167$.

    **Why:** precision answers "can I trust a flag?". Recall answers "what do I miss?". The harmonic mean is low when one of the two is low, so a detector cannot get a good F1 with only one of them.

3. **Move the threshold.**

    | Threshold | Precision | Recall | F1 | False positive rate |
    |---|---|---|---|---|
    | 0.25 | 0.571 | 1.0 | 0.727 | 0.5 |
    | 0.5 | 0.75 | 0.75 | 0.75 | 0.167 |
    | 0.75 | 1.0 | 0.5 | 0.667 | 0 |

    **Why:** the threshold is a choice of the developer, not a property of the detector. Each row is the same detector.

4. **Measure the detector without a threshold.** The **ROC curve** shows the recall against the false positive rate for all thresholds. The **AUROC** is the area under it. It is equal to the share of (harmful, harmless) pairs in which the harmful request has the higher score. There are $4 \times 6 = 24$ pairs, and 20 are in the correct order.

    $$\text{AUROC} = 20 / 24 = 0.833$$

    **Why:** 0.5 is a random order and 1 is a perfect order. Use the AUROC to compare two detectors before you choose a threshold.

5. **Make the harmful requests rare.** Keep the recall (0.75) and the false positive rate (0.167), with 10 harmful requests in 1,000. The detector finds 7.5 of them and flags 165 harmless ones. The precision is $7.5 / 172.5 = 0.043$. The AUROC is still 0.833.

    **Why:** the AUROC does not see the base rate. When positives are rare, report the precision at a fixed recall, or the **average precision** (the area under the precision-recall curve: 0.830 for the 10 requests).

6. **Check if the scores are probabilities.** A detector is **calibrated** when, of the items with a score of 0.7, 70% are positive. The **Brier score** is the mean squared difference between score and label: 0.152 here (0 is perfect). The **expected calibration error (ECE)** puts the scores in bins and compares the mean score with the share of positives in each bin. With the two bins "below 0.5" and "0.5 or more", it is 0.040.

    **Why:** the AUROC only uses the order of the scores. If a person or a program reads the score as a probability, you also need the calibration.

<figure class="plot">
<img src="/figures/metrics/roc.svg" alt="ROC curve of the toy detector: true positive rate against false positive rate. The area under the curve is 0.833. The threshold 0.5 is the point with a false positive rate of 0.167 and a recall of 0.75. A random detector is the diagonal, with an area of 0.5.">
<figcaption>Each threshold is one point of the curve. The area summarises all of them.</figcaption>
</figure>

<div class="llm-widget" data-widget="threshold-metrics"></div>

<div class="key-point">

Precision, recall and F1 describe one threshold. The AUROC describes the order of the scores for all thresholds, but it hides the base rate. Calibration says if a score can be read as a probability.

</div>

### From toy to real

- With 10 items, one item changes a rate by 0.1 or more. Report an interval: [Evaluation](../scientist/06-evaluation.md) shows how.
- The paper "On Calibration of Modern Neural Networks" (2017) found that modern networks are poorly calibrated. Its fix, temperature scaling, divides the logits by one number: the temperature of the [sampling](../scientist/01-architecture.md) step.
- scikit-learn has these metrics as functions: `confusion_matrix`, `f1_score`, `roc_auc_score`, `average_precision_score`, `brier_score_loss`, `log_loss`.
- The safety lessons use these numbers: the AUC of a monitor in [Control](../safety/06-control.md), and the base rate of a detector in [Scheming](../safety/04-scheming.md).

## 4. Written text and ranked lists

### Setup

The output is now a text, and you have a reference answer or a set of tests. The reference is "the cat sat". The model wrote "the cat cat".
A [retrieval](../engineer/02-vector-storage.md) system gives a ranked list of documents, and you know which documents are relevant.

### Steps

1. **Exact match.** The two texts are not equal: the **exact match** is 0.

    **Why:** it is the strictest metric. Use it when one answer is correct: a number, a name, a choice. It gives 0 to an answer that is almost correct.

2. **Token F1.** Count the words that the two texts share: `the` (1) and `cat` (1, because the reference has it one time). Precision is $2 / 3$, recall is $2 / 3$, and the F1 is 0.667.

    **Why:** it gives a part of the points for a part of the answer. Use it for short answers in question answering.

3. **BLEU: precision on n-grams.** An **n-gram** is a sequence of $n$ words. Of the 3 single words of the model text, 2 are in the reference (0.667). Of its 2 pairs, "the cat" is in the reference and "cat cat" is not (0.5). **BLEU** is the geometric mean of these precisions, with a penalty for a text that is too short (none here):

    $$\text{BLEU}_2 = \sqrt{0.667 \times 0.5} = 0.577$$

    **Why:** pairs and longer n-grams reward the correct word order. BLEU was made for translation, where many words of a good answer are in the reference.

4. **ROUGE-L: the longest common subsequence.** The longest sequence of words that is in the two texts in the same order is "the cat": 2 words. **ROUGE-L** divides it by the length of the reference: a recall of 0.667.

    **Why:** ROUGE looks from the side of the reference: how much of it did the model cover? It was made for summaries.

5. **pass@k for code.** Run the tests on the code that the model wrote. Assume that one sample passes with a probability of 0.2. **pass@k** is the probability that 1 of $k$ samples or more passes: $1 - 0.8^k$. This gives 0.2, 0.672 and 0.893 for $k$ = 1, 5 and 10.

    **Why:** code has a real test of correctness, so no comparison of words is necessary. Always give $k$: pass@10 is 4.5 times pass@1 here, for the same model.

6. **Ranked lists.** Assume 3 queries with the first relevant document at the ranks 1, 3 and 5. **Recall@3** is the share of the queries with a relevant document in the top 3: 0.667. The **mean reciprocal rank (MRR)** is the mean of 1 / rank: $(1 + 1/3 + 1/5) / 3 = 0.511$. **nDCG** also uses grades of relevance: for the grades [0, 2, 1] at the ranks 1 to 3, it is $1.762 / 2.631 = 0.670$.

    **Why:** recall@k asks "is it in the part that the model will read?". MRR and nDCG also reward a higher rank. Use recall@k for the retrieval step of [RAG](../engineer/03-rag.md), and nDCG when documents are more or less relevant.

<div class="key-point">

Word-overlap metrics measure the similarity to a reference, not the truth of the answer. Use a real test when one exists (exact match, unit tests). Use overlap metrics to see a change, and a judge or a human for the quality.

</div>

### From toy to real

- "the cat cat" gets 0.667 and is wrong. A correct answer with other words, for example "a cat was sitting", gets almost 0. This is the limit of all overlap metrics.
- BLEU has parameters. The paper "A Call for Clarity in Reporting BLEU Scores" (2018) found differences of up to 1.8 points between usual configurations, and made the tool SacreBLEU to fix them.
- The HumanEval paper (2021) measures functional correctness. Its abstract reports 70.2% of the problems solved with 100 samples for each problem: a pass@100.
- SQuAD (2016) reports exact match and F1. Its abstract gives an F1 of 51.0% for a logistic regression model, against 20% for a simple baseline.
- For open answers, a judge model gives the score: see [Evaluation](../scientist/06-evaluation.md), also for the agreement between judges (Cohen's kappa) and the Elo rating.
- The Hugging Face `evaluate` library has many of these metrics. Its page now points to LightEval as the more actively maintained tool.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import numpy as np

p = np.array([0.099, 0.242, 0.659])                  # rounded toy probabilities
q = np.array([0.186, 0.307, 0.506])

# 1. Loss and perplexity
loss = -np.log(0.659)                                # 0.417 nats = 0.602 bits
ppl = np.exp(np.mean(-np.log([0.659, 0.5, 0.25])))   # 2.298

# 2. Entropy, cross-entropy, KL
H = -(p * np.log(p)).sum()                           # 0.847
kl_pq = (p * np.log(p / q)).sum()                    # 0.053
kl_qp = (q * np.log(q / p)).sum()                    # 0.058: not symmetric

# 3. Scores with labels
pos, neg = [0.9, 0.8, 0.6, 0.3], [0.7, 0.4, 0.35, 0.2, 0.1, 0.05]
tp, fp = sum(s >= 0.5 for s in pos), sum(s >= 0.5 for s in neg)
precision, recall = tp / (tp + fp), tp / len(pos)    # 0.75, 0.75
f1 = 2 * precision * recall / (precision + recall)   # 0.75
auroc = np.mean([a > b for a in pos for b in neg])   # 0.833

# 4. Text and ranking
pass_at_5 = 1 - (1 - 0.2) ** 5                       # 0.672
mrr = np.mean([1 / r for r in (1, 3, 5)])            # 0.511
```

## Comparison

| Metric | What you need | Question that it answers | Toy value | Main limit |
|---|---|---|---|---|
| Cross-entropy loss | Token probabilities, correct tokens | How much probability on the correct token? | 0.417 | Not a measure of usefulness |
| Perplexity | The same | Between how many tokens does the model hesitate? | 2.298 | Depends on the tokenizer |
| KL divergence | Two distributions | How far is one model from a reference model? | 0.053 | Not symmetric; large near 0 |
| Accuracy | Decisions, labels | Which share is correct? | 0.8 | Hides rare classes |
| Precision and recall | Decisions, labels | Can I trust a flag? What do I miss? | 0.75, 0.75 | Depend on the threshold |
| F1 | The same | One number for the two | 0.75 | Hides which of the two is low |
| AUROC | Scores, labels | Is the order of the scores correct? | 0.833 | Does not see the base rate |
| Brier score, ECE | Scores, labels | Is a score a probability? | 0.152, 0.040 | ECE depends on the bins |
| Exact match | Text, reference | Is the answer exactly the reference? | 0 | No points for a near answer |
| BLEU, ROUGE | Text, reference | How many words are shared? | 0.577, 0.667 | Similar words, not truth |
| pass@k | Code, tests | Does 1 of k samples pass? | 0.672 for k = 5 | Only as good as the tests |
| Recall@k, MRR, nDCG | Ranked list, relevance | Is the correct document near the top? | 0.667, 0.511, 0.670 | Need relevance labels |

## Checks and common misconceptions

- **"The accuracy is 99%, so the detector is good."** With 10 harmful requests in 1,000, a detector that flags nothing has an accuracy of 99% and a recall of 0. Check: read the confusion matrix, and the precision and recall for the rare class.
- **"A lower loss means better answers."** The loss measures the probability of the reference tokens. A model can get a lower loss and write worse answers, and "the cat cat" gets a BLEU of 0.577. Check: use a metric that fits the task, and read some outputs.
- **"KL divergence is a distance."** $\text{KL}(p \,\|\, q) = 0.053$ and $\text{KL}(q \,\|\, p) = 0.058$. Check: say which distribution is the reference, and use the Jensen-Shannon divergence when you need a symmetric number.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. Each statement about a paper or a library was read in the linked source on 11 November 2026.

- [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) and [KLDivLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html), PyTorch documentation (logits as input, the default reduction, `ignore_index`, the note on `batchmean`).
- [Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html), scikit-learn documentation (the names of the functions).
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), Hinton et al., 2015 (distillation).
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599), Guo et al., 2017 (poor calibration, temperature scaling).
- [Bleu: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/), Papineni et al., 2002.
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/), Lin, 2004.
- [A Call for Clarity in Reporting BLEU Scores](https://arxiv.org/abs/1804.08771), Post, 2018 (the 1.8 points, SacreBLEU).
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374), Chen et al., 2021 (HumanEval, 70.2% with 100 samples).
- [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250), Rajpurkar et al., 2016 (the two F1 values).
- [Evaluate](https://huggingface.co/docs/evaluate/index), Hugging Face (what the library is, the pointer to LightEval).
