# 1. The LLM architecture

## Core idea

A large language model (LLM) does one thing: it reads a text and gives a score to each possible next word.
It picks one word, adds it to the text, and does the same thing again.
All the rest (tokens, attention, sampling) is the machinery that makes this one step work.

## The example

This page uses the full [toy example](../toy-example.md).

| Item | Value |
|---|---|
| Vocabulary | `the` = 0, `cat` = 1, `sat` = 2 |
| Embeddings | `the` = [1, 0], `cat` = [0, 1], `sat` = [1, 1] |
| Logits after "the cat" | [0.1, 1.0, 2.0] |

A **logit** is a raw score for one token. A higher logit means "more likely".

## 1. Architectural overview

### Setup

The model gets the text "the cat". It must write the next word.
The model is **decoder-only**: it has one stack of layers, and each layer can only look at the words on its left.

<figure class="diagram">
--8<-- "docs/assets/figures/architecture/pipeline.svg"
<figcaption>The 7 steps of one generation loop, with the toy numbers.</figcaption>
</figure>

### Steps

1. **Split the text into tokens.** "the cat" becomes the ids `[0, 1]`. A **token** is one unit of text with a number (an id).

    **Why:** a neural network can only calculate with numbers, not with letters.
    { .why }

2. **Replace each id with its embedding.** An **embedding** is a vector (a list of numbers) that the model learns for each token. We get `[1, 0]` and `[0, 1]`.

    **Why:** an id is only a label. Id 2 is not "twice" id 1. A vector can put similar words near each other.
    { .why }

3. **Send the vectors through N blocks.** Each **block** has an attention layer (each token reads the other tokens) and a feed-forward layer (a small network that changes each vector alone).

    **Why:** the vector of "cat" must change to mean "cat, after the". The blocks mix this context into each vector.
    { .why }

4. **Turn the last vector into logits.** A last matrix gives one score for each vocabulary entry: `[0.1, 1.0, 2.0]`.

    **Why:** the model must compare all possible next tokens on the same scale.
    { .why }

5. **Turn logits into probabilities with softmax.** **Softmax** makes each score positive and makes the sum equal to 1.

    $$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

    $$e^{0.1} = 1.105,\quad e^{1.0} = 2.718,\quad e^{2.0} = 7.389,\quad \text{sum} = 11.212$$

    $$p = \left[\tfrac{1.105}{11.212},\ \tfrac{2.718}{11.212},\ \tfrac{7.389}{11.212}\right] = [0.099,\ 0.242,\ 0.659]$$

    **Why:** you can only draw a random token from numbers that are probabilities.
    { .why }

6. **Pick one token.** Here `sat` has the highest probability (0.659).

    **Why:** the output must be one word, not a list of scores.
    { .why }

7. **Add the token to the text and start again.** The input is now "the cat sat". This loop is called **autoregressive** generation.

    **Why:** the model only knows how to write one token. A long text is this one step, done many times.
    { .why }

!!! success "Key point"
    A decoder-only LLM is a next-token predictor in a loop. In steps 1, 2 and 4 the model looks up or multiplies fixed tables. Only step 3 mixes information between tokens.

### From toy to real

The first Transformer (2017) had two stacks. An **encoder** read the full input text. A **decoder** wrote the output text. That design fits translation. GPT kept only the decoder, because "continue this text" is enough to do almost any text task.

| Family | Each token can see | Trained to | Example | Good for |
|---|---|---|---|---|
| Encoder-only | Left and right | Fill in hidden words | BERT | Classification, embeddings |
| Encoder-decoder | Encoder: all. Decoder: left + encoder output | Write an output for an input | T5, the 2017 Transformer | Translation |
| Decoder-only | Left only | Write the next token | GPT, Llama | Text generation, chat |

Scale of each part:

| Part | Toy | GPT-2 small | Llama 3 8B |
|---|---|---|---|
| Vocabulary size | 3 | 50,257 | 128,256 |
| Embedding size | 2 | 768 | 4,096 |
| Blocks | 1 | 12 | 32 |
| Context (max tokens) | 3 | 1,024 | 8,192 |
| Parameters | 6 in the embedding table | 124 million | 8 billion |

The embedding table of GPT-2 small alone has 50,257 × 768 = 38,597,376 numbers.

## 2. Tokenization

### Setup

Our toy tokenizer has one token for each word. This breaks on a new word: "hat" has no id.
Real models use **byte-pair encoding (BPE)**: start from single characters, then merge the pair that occurs most often, again and again.
We train BPE on a small corpus (a set of texts): `cat` 2 times, `sat` 1 time, `the` 1 time.

### Steps

1. **Start with one token for each character.** The words are `c a t`, `s a t`, `t h e`. The vocabulary is `{a, c, e, h, s, t}`: 6 tokens.

    **Why:** with all characters in the vocabulary, the tokenizer can write any word. No word is "unknown".
    { .why }

2. **Count each pair of adjacent tokens.** Multiply by the word count.

    | Pair | `c a` | `a t` | `s a` | `t h` | `h e` |
    |---|---|---|---|---|---|
    | Count | 2 | **3** | 1 | 1 | 1 |

    **Why:** a frequent pair is a good candidate for one token. It saves the most tokens.
    { .why }

3. **Merge the most frequent pair.** `a t` becomes the new token `at`. The words are now `c at`, `s at`, `t h e`.

    **Why:** each merge makes frequent texts shorter. Shorter inputs cost less to process.
    { .why }

4. **Do it again.** New counts: `c at` = 2, `s at` = 1, `t h` = 1, `h e` = 1. Merge `c at` into `cat`. On the third round all pairs have a count of 1, and we merge `t h` into `th`.

    **Why:** you stop when the vocabulary has the size you want. The vocabulary size is a design choice.
    { .why }

5. **Encode new text with the merges, in the same order.** The vocabulary has 9 tokens: `a c e h s t at cat th`.

    | Word | Tokens | Count |
    |---|---|---|
    | `cat` | `cat` | 1 |
    | `sat` | `s` `at` | 2 |
    | `hat` (never seen) | `h` `at` | 2 |

    **Why:** frequent words get one token. Rare words fall back to smaller pieces, so nothing is lost.
    { .why }

!!! success "Key point"
    A tokenizer is a compression table that is learned from data before the model is trained. Frequent strings cost 1 token. Rare strings cost many.

### From toy to real

- GPT-2 does 50,000 merges on top of 256 byte values, plus 1 special token: 50,257 tokens.
- Real tokenizers work on **bytes**, not characters. Thus they can encode any language and any emoji.
- For English text, 1 token is about 4 characters or 0.75 words. This is a rule of thumb, not a law.
- The model sees tokens, not letters. This is why an LLM can fail to count the letters in a word.
- Other algorithms: **WordPiece** (BERT) and **Unigram** (SentencePiece). They build a subword vocabulary with a different rule, but the result has the same shape.

## 3. Attention mechanisms

### Setup

The input is "the cat sat": 3 embedding vectors. We want a new vector for each token that contains information from the other tokens.
**Self-attention** does this: each token calculates how much it must read from each token, then takes a weighted average.

To keep the numbers small, the query, key and value of a token are all equal to its embedding. A real model first multiplies the embedding by three learned matrices, $W_Q$, $W_K$ and $W_V$.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

- $Q$ (query): what a token looks for. $K$ (key): what a token offers. $V$ (value): what a token gives when it is read.
- $d_k$: the size of the key vectors. Here $d_k = 2$.
- $M$: the **causal mask**. It is $-\infty$ for each later token and 0 for the others.

### Steps

1. **Calculate the dot product of each query with each key.** The **dot product** multiplies two vectors element by element and adds the results.

    $$QK^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$$

    Row 3 is the token `sat`: `sat·the` = 1, `sat·cat` = 1, `sat·sat` = 2.

    **Why:** the dot product is large when two vectors point in the same direction. It measures "this key matches my query".
    { .why }

2. **Divide by $\sqrt{d_k} = \sqrt{2} = 1.414$.** Row 3 becomes `[0.707, 0.707, 1.414]`.

    **Why:** dot products grow with the vector size. Large scores make softmax give almost all the weight to one token, and then the gradients become very small. The division keeps the scores in a stable range.
    { .why }

3. **Apply the causal mask.** Each token can only see itself and the tokens before it.

    $$\begin{bmatrix} 0.707 & -\infty & -\infty \\ 0 & 0.707 & -\infty \\ 0.707 & 0.707 & 1.414 \end{bmatrix}$$

    **Why:** during generation the later tokens do not exist. During training they are the answer. The mask stops the model from reading the answer.
    { .why }

4. **Apply softmax to each row.** For row 3: $e^{0.707} = 2.028$, $e^{1.414} = 4.113$, sum = 8.169.

    $$\text{weights} = \begin{bmatrix} 1 & 0 & 0 \\ 0.330 & 0.670 & 0 \\ 0.248 & 0.248 & 0.503 \end{bmatrix}$$

    **Why:** the weights of one token must be positive and must add up to 1, so that the result is a weighted average.
    { .why }

5. **Multiply the weights by the values.** For `sat`:

    $$0.248 \cdot [1, 0] + 0.248 \cdot [0, 1] + 0.503 \cdot [1, 1] = [0.752,\ 0.752]$$

    **Why:** this is the mix. The new vector of `sat` now contains 25% of `the`, 25% of `cat` and 50% of itself.
    { .why }

<figure class="diagram">
--8<-- "docs/assets/figures/architecture/attention.svg"
<figcaption>Step 5 for the token "sat". A thicker arrow is a larger weight.</figcaption>
</figure>

<div class="llm-widget" data-widget="attention"></div>

!!! success "Key point"
    Attention is a weighted average. The weights come from dot products between tokens, so the model learns *where to look* through the matrices $W_Q$ and $W_K$. A token at position 1 and a token at position 1,000 are read in one step: distance has no cost.

### From toy to real

- **Learned projections.** $Q = XW_Q$, $K = XW_K$, $V = XW_V$. Training changes these matrices so that useful tokens get high scores.
- **Multi-head attention.** The model runs many attention calculations in parallel, each with its own matrices, and joins the results. GPT-2 small has 12 heads of size 64 (12 × 64 = 768). One head can follow grammar while another follows names.
- **Position.** Our toy attention gives the same result if you change the word order of the earlier tokens. Real models add position information, for example **RoPE** (rotary position embedding), which rotates $Q$ and $K$ by an angle that depends on the position.
- **Cost.** Attention compares each pair of tokens. The toy has 3 × 3 = 9 pairs. A context of 8,192 tokens has 8,192² = 67,108,864 pairs, for each head and each layer. This quadratic cost is why long contexts are expensive.

## 4. Sampling techniques

### Setup

The model gave the probabilities `[0.099, 0.242, 0.659]` for `the`, `cat`, `sat`.
A **sampling** (or **decoding**) method is the rule that picks one token from these probabilities.

### Steps

1. **Greedy search: take the highest probability.** We take `sat` (0.659). The result is always the same.

    **Why:** it is the simplest rule, and it is correct when there is one good answer (for example, code or a calculation).
    { .why }

2. **Beam search: keep the best B sequences, not the best token.** With B = 2 we keep `sat` and `cat`. Assume these probabilities for the second token:

    | First token | Best second token | Sequence probability |
    |---|---|---|
    | `sat` (0.659) | `the` (0.34) | 0.659 × 0.34 = 0.224 |
    | `cat` (0.242) | `sat` (0.95) | 0.242 × 0.95 = **0.230** |

    Greedy search gives "sat the" (0.224). Beam search finds "cat sat" (0.230).

    **Why:** the best first token does not always start the best sentence. Beam search pays B times the compute to look one step further.
    { .why }

3. **Temperature: divide the logits by T before softmax.**

    $$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

    | T | Logits / T | Probabilities (`the`, `cat`, `sat`) |
    |---|---|---|
    | 0.5 | [0.2, 2.0, 4.0] | [0.019, 0.117, 0.864] |
    | 1.0 | [0.1, 1.0, 2.0] | [0.099, 0.242, 0.659] |
    | 2.0 | [0.05, 0.5, 1.0] | [0.194, 0.304, 0.502] |

    **Why:** T controls the risk. T < 1 makes the gaps between logits larger, so the model becomes more sure. T > 1 makes the gaps smaller, so rare tokens get more chances. T → 0 is greedy search.
    { .why }

4. **Top-k: keep the k most probable tokens.** With k = 2 we remove `the`. We divide the others by their sum (0.242 + 0.659 = 0.901): `cat` = 0.269, `sat` = 0.731.

    **Why:** the vocabulary has thousands of bad tokens with a very small probability each. Together they are picked too often. Top-k removes this tail.
    { .why }

5. **Top-p (nucleus): keep the smallest set of tokens with a total probability ≥ p.** With p = 0.9: `sat` alone is 0.659 (too small), `sat` + `cat` is 0.901 (enough). We keep 2 tokens, the same result as top-k = 2 here. With p = 0.5, `sat` alone is enough.

    **Why:** a fixed k is wrong in two cases. When the model is sure, k = 50 keeps 49 bad tokens. When the model is not sure, k = 50 can cut good tokens. Top-p adapts the number of tokens to the shape of the distribution.
    { .why }

<figure class="plot" markdown>
![Bar chart of the probability of each token for T = 0.5, 1.0 and 2.0. For "sat": 0.86, 0.66, 0.50.](../assets/figures/architecture/temperature.svg#only-light)
![Bar chart of the probability of each token for T = 0.5, 1.0 and 2.0. For "sat": 0.86, 0.66, 0.50.](../assets/figures/architecture/temperature-dark.svg#only-dark)
<figcaption>A low temperature moves probability to the best token. A high temperature makes the three tokens more equal.</figcaption>
</figure>

<div class="llm-widget" data-widget="sampling"></div>

!!! success "Key point"
    Sampling does not change the model. It changes only how you read the output distribution. The usual order is: temperature first, then top-k or top-p, then one random draw.

### From toy to real

- Common chat settings are a temperature between 0.7 and 1.0 with top-p between 0.9 and 0.95. They are starting values. Tune them for your task.
- With a real vocabulary, top-p = 0.9 can keep 1 token for one step and 500 tokens for the next step.
- Beam search is common in translation. It is rare in chat, because the most probable long text is often dull and repeats itself.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import numpy as np

def softmax(z, T=1.0):
    e = np.exp(z / T - np.max(z / T, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# Attention on "the cat sat" (query = key = value = embedding)
X = np.array([[1., 0.], [0., 1.], [1., 1.]])
scores = X @ X.T / np.sqrt(X.shape[1])
scores[np.triu_indices(3, k=1)] = -np.inf      # causal mask
weights = softmax(scores)                      # row 3: [0.248, 0.248, 0.503]
output = weights @ X                           # row 3: [0.752, 0.752]

# Sampling from the logits after "the cat"
logits = np.array([0.1, 1.0, 2.0])
probs = softmax(logits, T=1.0)                 # [0.099, 0.242, 0.659]
keep = np.argsort(probs)[::-1][:2]             # top-k = 2
p = np.zeros(3); p[keep] = probs[keep] / probs[keep].sum()   # [0, 0.269, 0.731]
next_id = np.random.default_rng(0).choice(3, p=p)
```

## Comparison

| Method | Random? | Parameter | Toy result | Good for | Risk |
|---|---|---|---|---|---|
| Greedy search | No | None | `sat` | One correct answer | Repeats itself in long texts |
| Beam search | No | Beams B | "cat sat" (0.230) | Translation, short exact outputs | B times the compute; dull text |
| Temperature | Yes | T > 0 | T = 2: `sat` at 0.502 | Control of variety | High T gives nonsense |
| Top-k | Yes | k | k = 2: [0, 0.269, 0.731] | Simple cut of the tail | Same k for all steps |
| Top-p | Yes | p in (0, 1] | p = 0.9: [0, 0.269, 0.731] | Default for chat | A flat distribution keeps many tokens |

## Checks and common misconceptions

- **"The model writes words."** It writes tokens. Check: count the tokens of your prompt with the tokenizer of your model, not the words. Limits and prices use tokens.
- **"Temperature 0 makes the model correct."** It makes the model *repeatable*, not correct. The model gives the same most-probable answer each time, also when that answer is wrong.
- **"Attention means the model understands what is important."** An attention weight is a learned mixing number in one head of one layer. It is not an explanation of the answer. A real model has hundreds of heads, and the layers after them change the result.

## Sources

This page follows section 1 of "The LLM Scientist" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [Visual intro to Transformers](https://www.youtube.com/watch?v=wjZofJX0v4M), 3Blue1Brown.
- [LLM Visualization](https://bbycroft.net/llm), Brendan Bycroft.
- [nanoGPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [tokenization video](https://www.youtube.com/watch?v=zduSFxRajkE), Andrej Karpathy.
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/), Lilian Weng.
- [Decoding Strategies in LLMs](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html), Maxime Labonne.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017 (the encoder-decoder Transformer).
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), Radford et al., 2019 (GPT-2 sizes).
