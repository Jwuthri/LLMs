# 6. Inference optimization

## Core idea

Text generation is slow for two reasons. The model does the same work again for each new token, and attention needs much memory.
The three methods on this page remove this waste. Flash Attention stores less, the key-value cache calculates less, and speculative decoding runs the large model fewer times.
None of them changes the result of the model.

## The example

This page uses attention on "the cat sat" from the [toy example](../toy-example.md). The page [The LLM architecture](../scientist/01-architecture.md) calculates these values.

| Item | Value |
|---|---|
| Embeddings (also used as query, key and value) | `the` = [1, 0], `cat` = [0, 1], `sat` = [1, 1] |
| Attention scores of the query `sat` | [0.707, 0.707, 1.414] |
| Attention weights of `sat` | [0.248, 0.248, 0.503] |
| Attention output of `sat` | [0.752, 0.752] |
| Probabilities of the next token after "the cat" | [0.099, 0.242, 0.659] |

## 1. Flash Attention

### Setup

Standard attention on $n$ tokens calculates $n \times n$ scores, stores this full matrix, applies softmax to each row, and multiplies by the values. The toy matrix has 3 × 3 = 9 scores.

A GPU has two types of memory. The main memory (**HBM**, high bandwidth memory) is large but slow to read and write. The memory on the chip (**SRAM**) is fast but very small. The full score matrix does not fit in the fast memory, so standard attention moves it to the slow memory and back.

**Flash Attention** cuts the keys and values into blocks that fit in the fast memory. It never stores the full matrix. But softmax needs the sum of a full row. The solution is an **online softmax**: a softmax that reads one block at a time and keeps a small state. The state of one query is a maximum $m$, a sum $l$ and an output $o$. For a new block with the scores $s_j$ and the values $v_j$:

$$m' = \max(m, \max_j s_j)$$

$$l' = e^{m - m'}\, l + \sum_j e^{s_j - m'}$$

$$o' = e^{m - m'}\, o + \sum_j e^{s_j - m'}\, v_j$$

The attention output is $o / l$ after the last block.

### Steps

The query is `sat`. Block 1 has the scores of `the` and `cat`. Block 2 has the score of `sat`.

1. **Read block 1.** The maximum is $m = 0.707$. Each score minus $m$ is 0, so $l = e^0 + e^0 = 2$. The output is $o = 1 \cdot [1, 0] + 1 \cdot [0, 1] = [1, 1]$.

    **Why:** only 2 scores are in memory. The subtraction of the maximum keeps each exponent at or below 0, so $e^x$ cannot overflow.
    { .why }

2. **Read block 2 and update the maximum.** $m' = \max(0.707, 1.414) = 1.414$. The correction factor is $e^{0.707 - 1.414} = 0.493$.

    **Why:** the old sums used the old maximum. One multiplication corrects them. The algorithm does not read block 1 again.
    { .why }

3. **Update the sum and the output.** $l = 0.493 \times 2 + e^0 = 1.986$. $o = 0.493 \cdot [1, 1] + 1 \cdot [1, 1] = [1.493, 1.493]$.

    **Why:** the state has a fixed size ($m$, $l$, $o$) for any number of blocks.
    { .why }

4. **Divide at the end.** $o / l = [0.752, 0.752]$. The weights are $[0.493, 0.493, 1] / 1.986 = [0.248, 0.248, 0.503]$. These are the values of standard attention.

    **Why:** Flash Attention is exact. It changes the order of the calculation, not the result.
    { .why }

5. **Count the memory.** Standard attention stores $n^2$ scores for each head. Flash Attention stores one block of scores at a time, plus $m$ and $l$ for each query: $2n$ numbers. For $n = 8{,}192$ tokens: 67,108,864 scores, or 16,384 state numbers.

    **Why:** this is the meaning of "from quadratic to linear". The *memory* becomes linear in $n$. The *compute* does not: the algorithm still calculates each of the $n^2$ scores one time.
    { .why }

<figure class="diagram">
--8<-- "docs/assets/figures/inference-optimization/online-softmax.svg"
<figcaption>The online softmax keeps 3 values between blocks and gives the exact output [0.752, 0.752].</figcaption>
</figure>

!!! success "Key point"
    Flash Attention gives the same output as standard attention. It lowers the memory of the score matrix from $O(n^2)$ to $O(n)$. It is faster because the data stays in the fast memory, not because it does fewer calculations.

### From toy to real

- Llama 3 8B has 32 heads. At 8,192 tokens, the score matrix of one layer has 67,108,864 × 32 scores. In 16-bit numbers (FP16) that is 4 GiB. Standard attention writes it to the slow memory.
- The Flash Attention paper reports a 3× speed-up for GPT-2 at 1,000 tokens, and a 15% faster training of BERT-large at 512 tokens.
- In training, the method also does not store the matrix for the backward pass. It calculates the blocks again when it needs them.

Real-library sketch (the tests do not run this code):

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", attn_implementation="flash_attention_2")
```

## 2. Key-value cache

### Setup

Generation runs the model one time for each new token. We call one run a **pass**. Pass $t$ reads a context of $t$ tokens.

The causal mask stops a token from reading later tokens. Thus the key and the value of an earlier token do not change when you add a token. The **key-value cache (KV cache)** stores them, for each layer and each head.

We count two types of work for one layer and one head: the key-value pairs that are calculated, and the query-key dot products.

### Steps

1. **Without a cache, each pass calculates all again.** Pass $t$ calculates $t$ key-value pairs and $t(t+1)/2$ dot products, because query $i$ reads $i$ keys.

    | Pass (context) | 1 (`the`) | 2 (`the cat`) | 3 (`the cat sat`) | Total |
    |---|---|---|---|---|
    | Key-value pairs | 1 | 2 | 3 | **6** |
    | Dot products | 1 | 3 | 6 | **10** |

    **Why:** the model is a function of its input. If you save nothing, each call starts from zero.
    { .why }

2. **With a cache, each pass calculates only the new token.** Pass $t$ calculates 1 key-value pair and $t$ dot products: the new query with the $t$ keys.

    | Pass (context) | 1 | 2 | 3 | Total |
    |---|---|---|---|---|
    | Key-value pairs | 1 | 1 | 1 | **3** |
    | Dot products | 1 | 2 | 3 | **6** |

    **Why:** the attention rows of the earlier tokens cannot change. Only the last row is new, and only the last row gives the next token.
    { .why }

3. **Write the totals for $n$ passes.**

    | Total of $n$ passes | No cache | Cache |
    |---|---|---|
    | Key-value pairs | $\frac{n(n+1)}{2}$ | $n$ |
    | Dot products | $\frac{n(n+1)(n+2)}{6}$ | $\frac{n(n+1)}{2}$ |
    | Key-value pairs, 3 passes | 6 | 3 |
    | Dot products, 3 passes | 10 | 6 |
    | Key-value pairs, 1,000 passes | 500,500 | 1,000 |
    | Dot products, 1,000 passes | 167,167,000 | 500,500 |

    **Why:** the cache changes the total attention work from the order of $n^3$ to the order of $n^2$. At 1,000 tokens it does 334 times fewer dot products.
    { .why }

4. **Calculate the memory of the cache.**

    $$\begin{aligned} \text{bytes} = {} & 2 \times \text{layers} \times \text{heads} \times \text{head size} \\ & \times \text{tokens} \times \text{bytes for each number} \end{aligned}$$

    The 2 is for the key and the value. Toy: 2 × 1 × 1 × 2 × 3 = 12 numbers. Llama 3 8B has 32 layers and 32 heads of size 128. With normal attention heads and FP16, one token needs 2 × 32 × 32 × 128 × 2 = 524,288 bytes. A full context of 8,192 tokens needs 4 GiB.

    **Why:** the cache exchanges compute for memory. Each token of each request has its own entry. On a server, the cache often sets the limit for the number of parallel requests.
    { .why }

5. **Share the keys and values between heads.** In **multi-query attention (MQA)**, all query heads use one key-value head. In **grouped-query attention (GQA)**, each group of query heads uses one key-value head. Llama 3 8B uses GQA with 8 key-value heads for 32 query heads.

    | Design | Key-value heads | Bytes for each token | Cache at 8,192 tokens |
    |---|---|---|---|
    | Multi-head attention | 32 | 524,288 | 4 GiB |
    | GQA (the real Llama 3 8B) | 8 | 131,072 | 1 GiB |
    | MQA | 1 | 16,384 | 0.125 GiB |

    **Why:** the cache size is proportional to the number of key-value heads. MQA saves the most memory but can lower the quality. GQA is the middle: 4 times less memory here, with a quality near that of multi-head attention.
    { .why }

<figure class="diagram">
--8<-- "docs/assets/figures/inference-optimization/kv-cache.svg"
<figcaption>Pass 3 of the toy. The cache replaces 2 of the 3 key-value calculations and 3 of the 6 dot products.</figcaption>
</figure>

<div class="llm-widget" data-widget="kv-cache"></div>

!!! success "Key point"
    The KV cache keeps the keys and values of all earlier tokens, so each new token costs one new row of attention. The price is memory: 2 × layers × heads × head size numbers for each token.

### From toy to real

- **Two phases.** The first pass reads the full prompt in parallel and fills the cache. This is the **prefill**. Each later pass reads one token. This is the **decode** phase. See [Deploying LLMs](07-deploying.md).
- The counts on this page are for one layer and one head. A real model multiplies them by its layers and heads.
- MQA and GQA are part of the architecture. The model must be trained with them. You cannot switch them on later.
- Servers such as vLLM also keep the cache of a shared prompt start between requests (prefix caching).

## 3. Speculative decoding

### Setup

A pass of a large model is slow. But one pass gives a probability distribution for *each* position of its input, in parallel.
**Speculative decoding** uses this fact. A small **draft model** writes $k$ tokens fast. Then the large **target model** checks all $k$ tokens in one pass.

The target probabilities after "the cat" are $p = [0.099, 0.242, 0.659]$. Assume that the draft model gives $q = [0.2, 0.3, 0.5]$.

### Steps

1. **The draft model samples a token from $q$.** For example, it writes `cat` with a probability of 0.3.

    **Why:** the draft model is cheap. We write $c$ for its cost relative to the target model. Assume $c = 0.05$.
    { .why }

2. **Keep the draft token $x$ with the probability $\min\!\left(1, \frac{p(x)}{q(x)}\right)$.**

    | Token | `the` | `cat` | `sat` |
    |---|---|---|---|
    | $p / q$ | 0.0986 / 0.2 | 0.2424 / 0.3 | 0.6590 / 0.5 |
    | Probability to keep | 0.493 | 0.808 | 1 |

    The table shows $p$ with 4 decimals, so that you can check the division.

    **Why:** the draft model writes `the` two times too often (0.2 against 0.099), so about half of these must go. It writes `sat` too rarely, so `sat` always stays.
    { .why }

3. **Calculate the acceptance rate.** $a = \sum_x \min(p(x), q(x)) = 0.099 + 0.242 + 0.5 = 0.841$.

    **Why:** $a$ is the overlap of the two distributions. Equal distributions give $a = 1$.
    { .why }

4. **After a rejection, sample a new token from the remainder.** The remainder is $\max(0, p - q) = [0, 0, 0.159]$, scaled to a sum of 1: $[0, 0, 1]$. The new token is always `sat`. Check the result: $P(\text{the}) = 0.2 \times 0.493 = 0.099$, $P(\text{cat}) = 0.3 \times 0.808 = 0.242$, $P(\text{sat}) = 0.5 \times 1 + (1 - 0.841) \times 1 = 0.659$.

    **Why:** the output distribution is exactly $p$. Speculative decoding changes the speed, not the quality.
    { .why }

5. **Count the tokens that one target pass gives.** Draft token $i$ stays only if all the tokens before it stay: probability $a^i$. The pass stops at the first rejection. The target model always adds 1 token of its own.

    $$E[\text{tokens}] = 1 + a + a^2 + \dots + a^k = \frac{1 - a^{k+1}}{1 - a}$$

    For $a = 0.841$ and $k = 4$: 2.643 draft tokens stay, so one pass gives 3.643 tokens.

    **Why:** the tokens after a rejected token continue a wrong text, so they are of no use.
    { .why }

6. **Calculate the speed-up.** One round costs $k$ draft passes and 1 target pass: $kc + 1$.

    $$\text{speed-up} = \frac{1 - a^{k+1}}{(1 - a)(kc + 1)}$$

    $$\frac{3.643}{4 \times 0.05 + 1} = 3.036$$

    **Why:** the gain needs a high $a$ and a low $c$. With $a = 0.5$, $c = 0.3$ and $k = 4$, the result is 0.881: slower than normal decoding.
    { .why }

<figure class="plot" markdown>
![Line chart of the speed-up against the draft length k for acceptance rates 0.9, 0.8 and 0.6, with a draft cost of 0.05. The best draft lengths are 13, 8 and 4 tokens, with speed-ups of 4.67, 3.09 and 1.92.](../assets/figures/inference-optimization/speculative-speedup.svg#only-light)
![Line chart of the speed-up against the draft length k for acceptance rates 0.9, 0.8 and 0.6, with a draft cost of 0.05. The best draft lengths are 13, 8 and 4 tokens, with speed-ups of 4.67, 3.09 and 1.92.](../assets/figures/inference-optimization/speculative-speedup-dark.svg#only-dark)
<figcaption>Each acceptance rate has a best draft length (the labelled point). A longer draft wastes more work after the first rejection.</figcaption>
</figure>

!!! success "Key point"
    A small model writes a draft, and the large model checks $k$ tokens in one pass. The accept rule $\min(1, p/q)$ keeps the exact output distribution of the large model.

### From toy to real

- With greedy decoding the rule is simpler: keep a draft token if it is the top token of the target model.
- The first paper reports a speed-up of 2 to 3 times for T5-XXL. Hugging Face calls the method **assisted generation**.
- **EAGLE-3** does not use a separate small model. Its draft part reads the internal features of the target model. The paper reports a speed-up of up to 6.5 times.
- The formula assumes that the GPU has spare capacity to check $k$ positions in parallel. On a busy server the gain is smaller: the EAGLE-3 paper reports 1.38 times more throughput at a batch size of 64.

Real-library sketch (the tests do not run this code):

```python
outputs = target_model.generate(**inputs, assistant_model=draft_model)
```

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import numpy as np

# Online softmax: the query "sat", keys read in blocks of 2
scores = np.array([0.707, 0.707, 1.414]); values = np.array([[1., 0.], [0., 1.], [1., 1.]])
m, l, o = -np.inf, 0.0, np.zeros(2)
for start in range(0, 3, 2):
    s, v = scores[start:start + 2], values[start:start + 2]
    m_new = max(m, s.max())
    l = np.exp(m - m_new) * l + np.exp(s - m_new).sum()
    o = np.exp(m - m_new) * o + np.exp(s - m_new) @ v
    m = m_new
output = o / l                                    # [0.752, 0.752]

# KV cache: totals of n passes
n = 3
dots_no_cache = sum(t * (t + 1) // 2 for t in range(1, n + 1))   # 10
dots_cache = sum(range(1, n + 1))                                # 6

# Speculative decoding
p = np.exp([0.1, 1.0, 2.0]) / np.exp([0.1, 1.0, 2.0]).sum()   # [0.099, 0.242, 0.659]
q = np.array([0.2, 0.3, 0.5])
keep = np.minimum(1, p / q)                       # [0.493, 0.808, 1]
a = np.minimum(p, q).sum()                        # 0.841
speedup = (1 - a ** 5) / ((1 - a) * (4 * 0.05 + 1))   # 3.036
```

## Comparison

| Method | What it removes | Output | Toy result | Price |
|---|---|---|---|---|
| Flash Attention | The stored $n \times n$ score matrix | Exactly the same | State of 3 values, output [0.752, 0.752] | Needs a special GPU kernel |
| KV cache | The work on earlier tokens | Exactly the same | 6 dot products, not 10 | Memory for each token |
| MQA and GQA | Key-value heads | Different model (trained that way) | Llama 3 8B: 1 GiB, not 4 GiB | Small quality loss possible |
| Speculative decoding | Passes of the large model | Same distribution | 3.643 tokens for each pass | A second model; wasted drafts |

## Checks and common misconceptions

- **"Flash Attention makes attention linear."** Only the memory is linear. Check: make the context 2 times longer and measure the attention time. It becomes about 4 times longer.
- **"The KV cache is free."** It uses memory for each token of each request. Check: calculate 2 × layers × key-value heads × head size × tokens × bytes × batch size before you increase the context or the batch.
- **"Speculative decoding always helps."** It helps when the draft model agrees with the target model. Check: measure the acceptance rate on your own prompts. A low rate makes generation slower.

## Sources

This page follows section 6 of "The LLM Engineer" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [GPU inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one), Hugging Face.
- [LLM inference performance engineering: best practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices), Databricks.
- [Optimizing LLMs for speed and memory](https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization), Hugging Face.
- [Assisted generation](https://huggingface.co/blog/assisted-generation), Hugging Face.
- [EAGLE-3](https://arxiv.org/abs/2503.01840), Li et al., 2025 (speed-up of up to 6.5 times; 1.38 times at batch size 64).
- [Speculators](https://github.com/vllm-project/speculators), vLLM project.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), Dao et al., 2022 (exact attention, linear memory, speed-up numbers).
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), Shazeer, 2019 (MQA).
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245), Ainslie et al., 2023 (GQA).
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), Leviathan et al., 2023 (accept rule, expected tokens, speed-up formula).
- [Llama 3 8B configuration](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json), Meta (32 layers, 32 heads, 8 key-value heads, hidden size 4,096, context 8,192).
- [vLLM](https://github.com/vllm-project/vllm) (prefix caching).
