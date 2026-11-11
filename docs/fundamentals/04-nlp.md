# 4. Natural language processing

## Core idea

Natural language processing (NLP) turns text into numbers, so that a program can calculate with words.
The first methods count words. The later methods learn a vector for each word and read the words in order.
Each method on this page repairs one limit of the method before it, and the last limit leads to the Transformer.

## The example

This page uses a **corpus** (a set of texts) of 3 short documents. It also uses the vectors and the matrix $W$ of the [toy example](../toy-example.md).

| Item | Value |
|---|---|
| Document D1 | "The cat sat." |
| Document D2 | "The dog sat." |
| Document D3 | "The cats sat on the cat." |
| Embeddings | `the` = [1, 0], `cat` = [0, 1], `sat` = [1, 1] |
| Weight matrix $W$ | $\begin{bmatrix} 0.5 & -1.2 \\ 0.8 & 0.3 \end{bmatrix}$ |

## 1. Text preprocessing

### Setup

**Preprocessing** cleans the raw text so that two strings with the same meaning (`The` and `the`, `cat` and `cats`) become the same unit.

### Steps

1. **Make the text lowercase and split it into tokens.** A **token** is one unit of text. D3 becomes `the` `cats` `sat` `on` `the` `cat` `.` (7 tokens).

    **Why:** a program that counts strings sees `The` and `the` as two different words. Lowercase text makes them one.
    { .why }

2. **Remove the punctuation and the stop words.** A **stop word** is a very frequent word with little meaning, here `the` and `on`. The 3 documents have 12 words. Only 7 remain: `cat sat`, `dog sat`, `cat sat cat`.

    **Why:** stop words are in each document, so they do not help to tell two documents apart.
    { .why }

3. **Stem each word.** **Stemming** cuts the end of a word with a fixed rule. Our rule removes a final "s": `cats` → `cat`. The word `sat` does not change.

    **Why:** `cat` and `cats` must add to the same count. A rule is fast and needs no dictionary.
    { .why }

4. **Or lemmatize each word.** **Lemmatization** replaces a word with its dictionary form (its **lemma**). A dictionary knows that `sat` is a form of `sit`. Thus `cats` → `cat` and `sat` → `sit`.

    **Why:** a cut rule cannot find `sit` in `sat`. A dictionary can, but it is slower and you need one for each language.
    { .why }

5. **Or let the data decide the units: subword tokens.** Byte-pair encoding (BPE) starts from characters and merges the most frequent pair, again and again. The [architecture page](../scientist/01-architecture.md#2-tokenization) shows the first 3 merges on the corpus `cat cat sat the`. Here are all the merges:

    | Merges done | New token | Vocabulary size | Corpus length (tokens) |
    |---|---|---|---|
    | 0 | (characters only) | 6 | 12 |
    | 1 | `at` | 7 | 9 |
    | 2 | `cat` | 8 | 7 |
    | 3 | `th` | 9 | 6 |
    | 4 | `the` | 10 | 5 |
    | 5 | `sat` | 11 | 4 |

    **Why:** each merge adds 1 token to the vocabulary and makes the text shorter. You choose where to stop: a small vocabulary with long texts, or a large vocabulary with short texts.
    { .why }

<div class="llm-widget" data-widget="bpe"></div>

!!! success "Key point"
    Classic preprocessing deletes information (capital letters, word endings, stop words) to make counting easier. Subword tokens keep all the information and let the model decide what is important.

### From toy to real

- NLTK and spaCy do these steps. The Porter stemmer, a list of rules for English, can join different words: `universe` and `university` both become `univers`.
- A real lemmatizer uses the role of the word in the sentence: "saw" becomes `see` when it is a verb, and stays `saw` when it is a tool.
- An LLM uses none of steps 1 to 4. Its tokenizer keeps capital letters, punctuation and stop words, because they change the meaning: "not" is a stop word in many lists.

## 2. Feature extraction techniques

### Setup

A **feature** is a number that describes a document. We use the tokens after steps 1 and 3, and we keep the stop words to see what TF-IDF does with them:
D1 = `the cat sat`, D2 = `the dog sat`, D3 = `the cat sat on the cat`.
The vocabulary has 5 terms: `cat`, `dog`, `on`, `sat`, `the`.

### Steps

1. **Count each term in each document.** This is the **bag-of-words** (BoW) representation: one vector of counts for each document.

    | Document | `cat` | `dog` | `on` | `sat` | `the` |
    |---|---|---|---|---|---|
    | D1 | 1 | 0 | 0 | 1 | 1 |
    | D2 | 0 | 1 | 0 | 1 | 1 |
    | D3 | 2 | 0 | 1 | 1 | 2 |

    **Why:** a classifier needs vectors of the same length. The count vector always has one number for each vocabulary term.
    { .why }

2. **Calculate the term frequency (TF).** Divide each count by the number of tokens of the document. For `cat` in D3: 2 / 6 = 0.333. For `cat` in D1: 1 / 3 = 0.333.

    **Why:** a long document has larger counts. The division removes the effect of the length.
    { .why }

3. **Calculate the inverse document frequency (IDF).** $N$ is the number of documents (3). $\text{df}(t)$ is the number of documents that contain the term $t$.

    $$\text{idf}(t) = \ln\frac{N}{\text{df}(t)}$$

    | Term | `cat` | `dog` | `on` | `sat` | `the` |
    |---|---|---|---|---|---|
    | df | 2 | 1 | 1 | 3 | 3 |
    | idf | ln(3/2) = 0.405 | ln(3/1) = 1.099 | 1.099 | ln(3/3) = 0 | 0 |

    **Why:** a term that is in all documents tells you nothing about one document. A rare term tells you much.
    { .why }

4. **Multiply: TF-IDF = tf × idf.** For `cat` in D1: 0.333 × 0.405 = 0.135. For `on` in D3: 0.167 × 1.099 = 0.183.

    | Document | `cat` | `dog` | `on` | `sat` | `the` |
    |---|---|---|---|---|---|
    | D1 | 0.135 | 0 | 0 | 0 | 0 |
    | D2 | 0 | 0.366 | 0 | 0 | 0 |
    | D3 | 0.135 | 0 | 0.183 | 0 | 0 |

    **Why:** the result is high only for a term that is frequent in this document and rare in the others. `the` and `sat` get 0 without a stop-word list.
    { .why }

5. **Compare two documents with the cosine similarity.** The **cosine similarity** is the dot product of two vectors divided by the product of their lengths: 1 for the same direction, 0 for no shared direction. With the counts, D1 and D2 have a similarity of 2 / 3 = 0.667. With TF-IDF, they have 0, and D1 and D3 have 0.594.

    **Why:** D1 and D2 share only `the` and `sat`, the words that all documents have. TF-IDF sees that one is about a cat and the other about a dog.
    { .why }

6. **Count n-grams to keep some word order.** An **n-gram** is a sequence of n adjacent tokens. A bigram has n = 2. D1 has the bigrams `the cat` and `cat sat`. In the corpus, `the cat` occurs 3 times and `the dog` 1 time. This gives a small language model:

    $$\begin{aligned} P(\text{cat} \mid \text{the}) &= \frac{\text{count}(\text{the cat})}{\text{count}(\text{the} \ \cdot)} \\[1mm] &= \frac{3}{4} = 0.75 \end{aligned}$$

    **Why:** bag-of-words gives the same vector for "the cat sat" and "sat the cat". Bigrams are different for these two texts.
    { .why }

!!! success "Key point"
    BoW and TF-IDF describe a document by its words, not by their order or their meaning. N-grams add a short window of order, and the n-gram model is the first next-word predictor.

### From toy to real

- scikit-learn uses a smoothed formula, $\ln\frac{1 + N}{1 + \text{df}} + 1$, and then scales each document vector to length 1.
- A real vocabulary has many thousands of terms, so each vector is almost all zeros (a **sparse** vector).
- N-grams grow fast. 5 terms give 5² = 25 possible bigrams. The 50,257 tokens of GPT-2 give 2,525,766,049. Most never occur in the training text, so the model has no count for them.

## 3. Word embeddings

### Setup

In the count vectors, `cat` and `dog` are two columns with no relation. A **word embedding** is a short, dense vector for each word. Words that occur in similar contexts get similar vectors.

### Steps

1. **Start with one-hot vectors, and see the problem.** A **one-hot** vector has a 1 at the index of the word and 0 elsewhere: `the` = [1, 0, 0], `cat` = [0, 1, 0], `sat` = [0, 0, 1]. The cosine similarity of each pair is 0.

    **Why:** this is what a count method sees. All words are equally different, so nothing learned about `cat` helps with `dog`.
    { .why }

2. **Use dense vectors and measure the similarity.** With the toy embeddings:

    $$\cos(\text{cat}, \text{sat}) = \frac{0 \cdot 1 + 1 \cdot 1}{\sqrt{1} \cdot \sqrt{2}} = 0.707$$

    Also: `the` and `sat` give 0.707, `the` and `cat` give 0.

    **Why:** a dense vector can be near some words and far from others. Similarity becomes a number that you can calculate.
    { .why }

3. **Make training pairs from the text.** **Word2Vec** in its **skip-gram** form takes each word (the center) and tries to predict the words near it (the context). With a window of 1, "the cat sat" gives 4 pairs: (the → cat), (cat → the), (cat → sat), (sat → cat).

    **Why:** no person has to label the data. The text itself says which words occur together.
    { .why }

4. **Score one pair: center `cat`, context `sat`.** Each word has a center vector $v$ and a context vector $u$. Assume that both are the toy embeddings. The score of a context word is the dot product $u_w \cdot v_c$. Softmax turns the scores into probabilities.

    $$P(o \mid c) = \frac{e^{u_o \cdot v_c}}{\sum_w e^{u_w \cdot v_c}}$$

    The scores for `the`, `cat`, `sat` are [0, 1, 1]. The probabilities are [0.155, 0.422, 0.422]. The loss is $L = -\ln P(o \mid c) = -\ln(0.422) = 0.862$.

    **Why:** the loss is small when the model gives a high probability to the word that is really in the context.
    { .why }

5. **Move the vector of `cat` against the gradient.** The gradient is the expected context vector minus the true one.

    $$\begin{aligned} \frac{\partial L}{\partial v_c} &= \sum_w P(w \mid c)\, u_w - u_o \\ &= [0.578,\ 0.845] - [1,\ 1] \\ &= [-0.422,\ -0.155] \end{aligned}$$

    With a learning rate of 0.5 (an assumption), $v_{\text{cat}} = [0, 1] - 0.5 \cdot [-0.422, -0.155] = [0.211, 1.078]$.
    Now $P(\text{sat} \mid \text{cat})$ is 0.465 and the loss is 0.765.

    **Why:** the step moves `cat` toward the context vector of `sat`. After millions of pairs, words with the same neighbours are in the same region.
    { .why }

!!! success "Key point"
    An embedding is not written by a person. It is the result of a prediction task: "which words occur near this word?"

### From toy to real

- The published Word2Vec vectors have 300 dimensions for 3 million words and phrases.
- The softmax of step 4 adds over the full vocabulary, which is too slow for millions of words. **Negative sampling** replaces it: the model must tell the true pair from a few random pairs.
- **GloVe** learns the vectors from counts of how often two words occur together. **FastText** adds the vectors of the character n-grams of a word, so a new word also gets a vector.
- These embeddings are **static**: "bank" has the same vector in "river bank" and "bank account". An LLM starts from such a table, and [attention](../scientist/01-architecture.md#3-attention-mechanisms) then changes each vector with its context.

## 4. Recurrent neural networks

### Setup

A **recurrent neural network** (RNN) reads one token at a time and keeps a memory vector, the **hidden state** $h_t$.

$$h_t = \tanh(W h_{t-1} + U x_t)$$

$x_t$ is the embedding of token $t$ and $W$ is the toy matrix. Assume that $U$ is the identity matrix (so $U x_t = x_t$) and that $h_0 = [0, 0]$. The function tanh puts each number between −1 and 1.

<figure class="diagram">
--8<-- "docs/assets/figures/nlp/rnn.svg"
<figcaption>The same cell reads the 3 tokens. Each arrow between two cells carries the hidden state.</figcaption>
</figure>

### Steps

1. **Read `the`.** $W h_0 = [0, 0]$, so $h_1 = \tanh([1, 0]) = [0.762, 0]$.

    **Why:** the first state holds only the first word.
    { .why }

2. **Read `cat`.** $W h_1 = [0.5 \cdot 0.762,\ 0.8 \cdot 0.762] = [0.381, 0.609]$. Add $x_2 = [0, 1]$: $[0.381, 1.609]$. Then $h_2 = \tanh([0.381, 1.609]) = [0.363, 0.923]$.

    **Why:** the new state mixes the old memory ($W h_1$) with the new word ($x_2$). Word order now changes the result.
    { .why }

3. **Read `sat`.** $W h_2 = [-0.926, 0.568]$. Add $x_3 = [1, 1]$: $[0.074, 1.568]$. Then $h_3 = [0.074, 0.917]$.

    **Why:** $h_3$ is a summary of the full sentence in 2 numbers. A last layer can turn it into a class or into logits for the next word.
    { .why }

4. **Send the error back through each step.** Training must know how $h_t$ changes when an earlier state changes. For one step back, this is the matrix $\text{diag}(1 - h_t^2)\, W$. At $h_3$, the factor $1 - h_3^2$ is [0.995, 0.160]. For $k$ steps back, you multiply $k$ of these matrices. Here is the size of the product for "the cat sat" repeated 20 times (60 tokens):

    | Steps back (k) | 1 | 10 | 30 | 59 |
    |---|---|---|---|---|
    | Size of the gradient | 1.296 | 0.338 | 0.011 | 0.000097 |

    **Why:** this is the **vanishing gradient**. The factor $1 - h^2$ is never more than 1 and is often near 0. It shrinks the signal more than $W$ stretches it. The word 59 steps back gets almost no learning signal.
    { .why }

5. **Add a memory with gates: the LSTM.** A **long short-term memory** (LSTM) network has a second state, the cell $c_t$. A **gate** is a vector of numbers between 0 and 1 that the network calculates from $h_{t-1}$ and $x_t$.

    $$\begin{aligned} c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\ h_t &= o_t \odot \tanh(c_t) \end{aligned}$$

    The forget gate $f_t$ keeps old memory. The input gate $i_t$ writes new content $g_t$. The output gate $o_t$ shows the result. $\odot$ multiplies element by element. Along the cell, one step back multiplies the gradient by $f_t$ only. Assume $f = 0.99$: after 59 steps the factor is $0.99^{59} = 0.553$.

    **Why:** the cell changes by addition, not through a matrix and a tanh. With a forget gate near 1, the gradient passes almost unchanged.
    { .why }

<figure class="plot" markdown>
![Line chart on a log scale. The RNN gradient falls from 1.296 at 1 step to 0.000097 at 59 steps. The LSTM cell path with a forget gate of 0.99 falls only to 0.553.](../assets/figures/nlp/gradient.svg#only-light)
![Line chart on a log scale. The RNN gradient falls from 1.296 at 1 step to 0.000097 at 59 steps. The LSTM cell path with a forget gate of 0.99 falls only to 0.553.](../assets/figures/nlp/gradient-dark.svg#only-dark)
<figcaption>The signal of the toy RNN gets about 10,000 times smaller over 59 tokens. The LSTM cell path loses less than half.</figcaption>
</figure>

!!! success "Key point"
    An RNN compresses the past text into one vector and updates it token by token. Gates (LSTM, GRU) help to keep old information, but the text still goes through one vector, one step at a time.

### From toy to real

- A **gated recurrent unit** (GRU) has 2 gates (update and reset) and no separate cell. With 2 inputs and 2 hidden units, the RNN layer has 10 parameters, the GRU 30 and the LSTM 40. (PyTorch keeps 2 bias vectors and reports 12, 36 and 48.)
- When $W$ stretches more than tanh shrinks, the product grows: the **exploding gradient**. The repair is gradient clipping (see [Pre-training models](../scientist/02-pre-training.md)).
- An RNN cannot read token 60 before token 59, so it cannot process the tokens of a text in parallel on a GPU.
- Attention removes the two limits: each token reads each earlier token in one step, and all positions are calculated together. This is why the Transformer (2017) replaced the LSTM for language tasks.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import numpy as np

# TF-IDF on the 3 documents (columns: cat, dog, on, sat, the)
counts = np.array([[1, 0, 0, 1, 1], [0, 1, 0, 1, 1], [2, 0, 1, 1, 2]])
tf = counts / counts.sum(axis=1, keepdims=True)
idf = np.log(len(counts) / (counts > 0).sum(axis=0))    # [0.405, 1.099, 1.099, 0, 0]
tfidf = tf * idf                                        # D3: [0.135, 0, 0.183, 0, 0]

# One skip-gram step: center "cat", context "sat"
E = np.array([[1., 0.], [0., 1.], [1., 1.]])
probs = np.exp(E @ E[1]) / np.exp(E @ E[1]).sum()       # [0.155, 0.422, 0.422]
v_cat = E[1] - 0.5 * (probs @ E - E[2])                 # [0.211, 1.078]

# RNN over "the cat sat"
W = np.array([[0.5, -1.2], [0.8, 0.3]])
h = np.zeros(2)
for x in E:
    h = np.tanh(W @ h + x)                              # ends at [0.074, 0.917]
```

Real-library sketch (the tests do not run this code):

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

features = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(["The cat sat.", "The dog sat."])
lstm = torch.nn.LSTM(input_size=2, hidden_size=2, batch_first=True)
```

## Comparison

| Method | Unit | Uses word order | Knows similar words | Toy result | Main limit |
|---|---|---|---|---|---|
| Bag-of-words | Document = counts | No | No | D1 = [1, 0, 0, 1, 1] | Frequent words dominate |
| TF-IDF | Document = weights | No | No | `cat` in D1 = 0.135 | `cat` and `dog` have no relation |
| N-grams | n adjacent tokens | n tokens | No | P(cat after the) = 0.75 | Most n-grams are never seen |
| Word2Vec, GloVe, FastText | Word = dense vector | No | Yes | cos(cat, sat) = 0.707 | One vector for each word, no context |
| RNN | State updated at each token | Yes | Yes | h₃ = [0.074, 0.917] | Gradient vanishes: 0.000097 at 59 steps |
| LSTM, GRU | State + gates | Yes | Yes | Cell path: 0.553 at 59 steps | One token at a time, no parallel training |
| Transformer | Each token reads all earlier tokens | Yes | Yes | See the [architecture page](../scientist/01-architecture.md) | Cost grows with the square of the length |

## Checks and common misconceptions

- **"Always remove stop words and stem the text."** This helps methods that count words. Do not do it before an LLM or a modern embedding model. Check: give the raw text to the tokenizer of the model.
- **"TF-IDF understands the topic."** It only matches strings. A document that says "kitten" and a query that says "cat" share no term, so their similarity is 0. Check: if your search must find synonyms, you need embeddings.
- **"An LSTM has no vanishing gradient."** It has a better path, not a perfect one. With a forget gate of 0.99 the signal still falls to 0.553 after 59 steps, and the past must fit in one vector.

## Sources

This page follows section 4 of "LLM Fundamentals" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [Word Embeddings](https://lena-voita.github.io/nlp_course/word_embeddings.html), Lena Voita.
- [NLP with spaCy in Python](https://realpython.com/natural-language-processing-spacy-python/), Real Python.
- [NLP Guide](https://www.kaggle.com/learn-guide/natural-language-processing), Kaggle.
- [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/), Jay Alammar.
- [PyTorch RNN from Scratch](https://jaketae.github.io/study/pytorch-rnn/), Jake Tae.
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), Christopher Olah.
- [Porter stemmer](https://tartarus.org/martin/PorterStemmer/), Martin Porter.
- [TF-IDF term weighting](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting), scikit-learn (smoothed formula).
- [Skip-gram](https://arxiv.org/abs/1301.3781) and [negative sampling](https://arxiv.org/abs/1310.4546), Mikolov et al., 2013.
- [word2vec project page](https://code.google.com/archive/p/word2vec/), Google (300 dimensions, 3 million words and phrases).
- [GloVe](https://nlp.stanford.edu/projects/glove/), Pennington et al., 2014. [FastText](https://arxiv.org/abs/1607.04606), Bojanowski et al., 2016.
- [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), Radford et al., 2019 (vocabulary size).
