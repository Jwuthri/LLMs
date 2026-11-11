# 3. Retrieval augmented generation

## Core idea

An LLM answers from what it saw in training. It does not know your documents, and it can invent an answer.
**Retrieval augmented generation (RAG)** first searches your documents for text that is relevant to the question. Then it puts this text into the prompt.
The LLM reads the answer in the prompt. You do not train the model.

## The example

This page uses the vector store of [page 2](02-vector-storage.md): 3 chunks from the file `cat.md`. The embedding of a text is the mean of its word vectors.

| Id | Chunk | Section | Vector |
|---|---|---|---|
| c1 | The cat sat on the mat. | Habits | [1, 0.667] |
| c2 | The cat sleeps all day. | Habits | [0.4, 1] |
| c3 | The cat eats fish. | Food | [0.25, 1.5] |

The question is "What does the cat eat?". The questions use these new word vectors, fixed by hand: `what` = [1, 2], `does` = [1, 0], `eat` = [0, 3], `where` = [2, 0], `sit` = [1, 1], `it` = [0, 1], `mice` = [0, 3].
The scripts of this course do not call an LLM. When a step needs the output of an LLM, the text says "assume".

## 1. Orchestrators

### Setup

An **orchestrator** is a library that connects the parts of an LLM application: loaders, splitters, the embedding model, the vector store, the prompt and the LLM. LangChain and LlamaIndex are the 2 common ones.
The orchestrator runs 2 phases. The indexing phase is page 2. The query phase is the 4 steps below.

<figure class="diagram">
--8<-- "docs/assets/figures/rag/loop.svg"
<figcaption>The 2 phases of RAG. The query phase runs for each question, with the numbers of this page.</figcaption>
</figure>

### Steps

1. **Embed the question.** "what does the cat eat" gives:

    $$\frac{[1, 2] + [1, 0] + [1, 0] + [0, 1] + [0, 3]}{5} = \frac{[3, 6]}{5} = [0.6,\ 1.2]$$

    **Why:** the search compares vectors. The question must be in the same space as the chunks, so it goes through the same embedding model.
    { .why }

2. **Retrieve the top-k chunks.** The cosine similarity of the question with each chunk is 0.997 for c2, 0.956 for c3 and 0.868 for c1. With k = 2, the retriever returns c2 and c3. The correct chunk, c3, is only second: the words `what` and `does` pull the question vector away from the food words.

    **Why:** the search makes errors, and k > 1 gives it a margin. But each added chunk costs prompt tokens and can distract the LLM.
    { .why }

3. **Build the prompt.** Put the chunks and the question into a fixed template.

    ```text
    Use only the context to answer the question.

    Context:
    [1] The cat sleeps all day.
    [2] The cat eats fish.

    Question: What does the cat eat?
    Answer:
    ```

    **Why:** "Use only the context" tells the LLM to read, not to remember. The numbers let the LLM say which chunk it used.
    { .why }

4. **Call the LLM.** Assume that it answers "The cat eats fish."

    **Why:** a search only finds text. The LLM reads the 2 chunks, ignores the one about sleep, and writes a direct answer.
    { .why }

Without a library, these 4 steps are 4 function calls: `embed`, `search`, `format`, `generate`. An orchestrator adds 3 things: one interface for many products (you change the vector store or the LLM in one line), ready parts (loaders, splitters, retrievers, memory), and support code (retries, streaming, a trace of each step).

!!! success "Key point"
    RAG is a search and a prompt template. The LLM does not change. A wrong answer has 2 possible causes: the retriever did not bring the correct chunk, or the LLM did not use it.

### From toy to real

- The **Model Context Protocol (MCP)** is a standard to connect an LLM application (the client) to tools and data sources (the servers). You write one connector for a data source, and each MCP client can use it.
- The indexing settings change the result of the search. The LangChain RAG tutorial uses chunks of 1,000 characters with an overlap of 200 characters, and it retrieves k = 4 chunks. These are starting values.

The widget below cuts the 15 words of `cat.md` into fixed-size chunks. It starts with the values of page 2: a size of 6 words and an overlap of 2 words give 4 chunks, which hold 21 words in total.

<div class="llm-widget" data-widget="chunking"></div>

## 2. Retrievers

### Setup

A **retriever** takes a question and returns chunks. Step 2 above is the simplest retriever: one vector search. It put the correct chunk second. The 5 steps below are methods to do better.

### Steps

1. **Rewrite the query.** In a chat, the second question is "Where does it sit?". An LLM first rewrites it as "where does the cat sit" (assume). Its vector is [1, 0.4], and the top chunk is c1 with 0.979.

    **Why:** the retriever sees only the last message, and `it` has no meaning without the messages before. In a store with texts about 100 animals, the word `cat` is necessary.
    { .why }

2. **Embed a draft answer, not the question.** This method is **HyDE** (hypothetical document embeddings). The LLM writes an answer without context. Assume "The cat eats mice." The fact is wrong, but its vector is [0.25, 1.75]. The cosine is 1.000 for c3 (0.9997 before rounding), 0.972 for c2 and 0.667 for c1. The correct chunk is now first.

    **Why:** a question does not look like its answer. A draft answer looks like the correct chunk, also when its facts are wrong. The cost is one more LLM call for each question.
    { .why }

3. **Keep more than one vector for each chunk.** A **multi-vector** retriever such as ColBERT keeps one vector for each token. It compares each token of the question with the best token of the chunk. Our store then holds 15 vectors, not 3.

    **Why:** one vector for a chunk is a mean, and a mean loses details. More vectors keep the details, but the store is 5 times larger here.
    { .why }

4. **Calculate a keyword score with BM25.** **BM25** gives a score to a chunk $d$ for the words $t$ of the question $q$ that it contains.

    $$\text{score}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f \cdot (k_1 + 1)}{f + k_1 \left(1 - b + b \cdot \frac{L}{L_\text{mean}}\right)}$$

    $$\text{IDF}(t) = \ln\!\left(1 + \frac{N - n_t + 0.5}{n_t + 0.5}\right)$$

    - $f$: how many times the chunk contains $t$. $L$: the number of words of the chunk. $L_\text{mean}$ = (6 + 5 + 4) / 3 = 5.
    - $N$ = 3 chunks. $n_t$: the number of chunks that contain $t$. Usual settings: $k_1 = 1.2$ and $b = 0.75$.

    First, **stemming** cuts each word to its base form: `eats` becomes `eat`. For the word `eat` and c3 ($n_t = 1$, $f = 1$, $L = 4$):

    $$\text{IDF} = \ln\!\left(1 + \frac{2.5}{1.5}\right) = 0.981$$

    $$\frac{1 \times 2.2}{1 + 1.2 \times (0.25 + 0.75 \times 0.8)} = \frac{2.2}{2.02} = 1.089$$

    The score of `eat` for c3 is 0.981 × 1.089 = 1.068. The words `the` and `cat` are in all 3 chunks, so their IDF is only 0.134. They add 0.145 each to c3. The totals are 1.359 for c3, 0.297 for c1 and 0.267 for c2.

    **Why:** a rare word that matches exactly is strong evidence. A word that is in each chunk is almost no evidence. BM25 finds exact names, codes and numbers, which an embedding makes unclear.
    { .why }

5. **Hybrid search: add the 2 scores.** First scale each list of scores so that its lowest value is 0 and its highest value is 1. Then calculate $\alpha \times \text{vector} + (1 - \alpha) \times \text{keyword}$, with $\alpha = 0.5$.

    | Chunk | Vector score, scaled | Keyword score, scaled | Hybrid |
    |---|---|---|---|
    | c1 | 0 | 0.028 | 0.014 |
    | c2 | 1 | 0 | 0.500 |
    | c3 | 0.682 | 1 | **0.841** |

    The order is now c3, c2, c1.

    **Why:** the 2 methods fail on different questions. The vectors know that `sit` and `sat` are near. The keywords know that `eat` is in one chunk only.
    { .why }

<figure class="plot" markdown>
![Bar chart of the scaled scores of the 3 chunks. Vector: 0, 1, 0.68. Keyword: 0.03, 0, 1. Hybrid: 0.01, 0.50, 0.84.](../assets/figures/rag/hybrid.svg#only-light)
![Bar chart of the scaled scores of the 3 chunks. Vector: 0, 1, 0.68. Keyword: 0.03, 0, 1. Hybrid: 0.01, 0.50, 0.84.](../assets/figures/rag/hybrid-dark.svg#only-dark)
<figcaption>The vector search prefers c2. The keyword search prefers c3. The hybrid score puts c3 first.</figcaption>
</figure>

!!! success "Key point"
    You can change the question (rewrite, HyDE), the index (multi-vector) or the score (BM25, hybrid). Each method repairs a different failure, so look at the failed questions before you select one.

### From toy to real

- Elasticsearch uses $k_1 = 1.2$ and $b = 0.75$ as default values.
- Generative retrievers such as CoRAG let the model search in several rounds. The model writes a new query after each result.
- A second way to combine 2 result lists uses the ranks, not the scores. See reciprocal rank fusion on [page 4](04-advanced-rag.md).

## 3. Memory

### Setup

An LLM keeps nothing between 2 calls. To continue a chat, the application must send the old turns again in the prompt. **Memory** is the rule that selects what to send.
The chat has 2 turns, and the user asks a third question. In the toy model, one word is one token.

| Turn | User | Assistant | Tokens |
|---|---|---|---|
| 1 | What does the cat eat? | The cat eats fish. | 5 + 4 = 9 |
| 2 | Where does it sit? | The cat sat on the mat. | 4 + 6 = 10 |
| 3 | Does it eat mice? | | 4 |

### Steps

1. **Buffer: send all the old turns.** The history is 9 + 10 = 19 tokens.

    **Why:** nothing is lost. But the history grows with each turn. It costs more for each call, and it stops at the context limit of the model.
    { .why }

2. **Window: send only the last n turns.** With n = 1, the history is turn 2: 10 tokens.

    **Why:** the cost is constant. But the model forgets: turn 1, about food, is not sent, and the new question is about food.
    { .why }

3. **Summary: an LLM replaces the old turns with a short text.** Assume the summary "Cat eats fish, sat on the mat.": 7 tokens.

    **Why:** the main facts stay, at a small constant size. The cost is an LLM call, and the summary can drop a detail that you need later.
    { .why }

4. **Vector memory: store each turn as a vector, and retrieve the most similar turn.** The cosine similarity of "does it eat mice" is 0.984 with turn 1 and 0.631 with turn 2. The memory sends turn 1: 9 tokens.

    **Why:** this is RAG on the chat history. It finds the relevant turn, also when the turn is very old. But it can miss a turn that is necessary and not similar.
    { .why }

!!! success "Key point"
    Memory is a selection of old text to send again. The context window is limited, so each type of memory makes a compromise between cost and what the model can remember.

### From toy to real

- Real turns have hundreds of tokens, and the limit is the context window of the model.
- Products mix the types: a summary of the old turns, a window of the last turns, and a vector store for facts.

## 4. Evaluation

### Setup

RAG has 2 parts that can fail, so you measure each part: 2 metrics for the retrieval and 2 metrics for the generation.
The test case is our question, with a reference made by a person: the correct answer is "The cat eats fish." and the relevant chunk is c3.

### Steps

1. **Context precision: are the relevant chunks at the top?** Let $v_k$ = 1 if the chunk at rank $k$ is relevant, and 0 if not.

    $$\text{context precision} = \frac{\sum_{k=1}^{K} \text{precision@}k \times v_k}{\text{number of relevant chunks in the top } K}$$

    The plain retriever returned c2 (not relevant), then c3 (relevant). Precision@2 is 1/2, so the score is 0.5. The hybrid retriever returned c3, then c2: precision@1 is 1/1, so the score is 1.0.

    **Why:** this metric measures the ranking. A chunk that is not relevant and is before the relevant one is noise in the prompt.
    { .why }

2. **Context recall: is the necessary information in the context?** Divide the number of claims of the reference answer that the context supports by the number of claims of the reference answer. The reference has 1 claim. With k = 2, c3 is in the context: 1/1 = 1.0. With k = 1, the context is only c2: 0/1 = 0.

    **Why:** if the fact is not in the context, the LLM cannot give a correct answer that comes from your documents. A low recall is a retrieval problem. A better prompt does not repair it.
    { .why }

3. **Faithfulness: does the answer come from the context?** Divide the number of claims of the answer that the context supports by the number of claims of the answer. Assume a weaker answer: "The cat eats fish. The cat sleeps all day." The context c2 + c3 supports the 2 claims: 2/2 = 1.0. If the answer adds "The cat drinks milk.", the score is 2/3 = 0.667.

    **Why:** a claim that is not in the context comes from the memory of the model. It can be an invention.
    { .why }

4. **Answer relevancy: does the answer address the question?** Divide the number of statements that address the question by the number of statements. In the weaker answer, "The cat sleeps all day." does not address the question: 1/2 = 0.5.

    **Why:** an answer can be fully faithful and not useful. Here the chunk c2 went from the context into the answer. The low context precision of step 1 caused it.
    { .why }

!!! success "Key point"
    Measure the retrieval and the generation separately. The 4 metrics tell you which part to repair: the ranking, the content of the context, the invention, or the focus of the answer.

### From toy to real

- In a real test, an LLM (a "judge") divides the answer into claims and checks each claim. Ragas and DeepEval are libraries that do this.
- The definitions change between libraries. The ratio of step 4 is the one of DeepEval. Ragas generates questions from the answer and compares them with the original question.
- The judge is also an LLM. Check a sample of its decisions manually.

## Minimal code

This code gives the numbers of sub-part 1 and the context precision.

```python
import numpy as np

V = {"the": [1, 0], "cat": [0, 1], "sat": [1, 1], "on": [1, 0], "mat": [2, 2], "sleeps": [0, 2],
     "all": [1, 0], "day": [0, 2], "eats": [0, 3], "fish": [0, 2], "what": [1, 2], "does": [1, 0], "eat": [0, 3]}

def embed(text):
    words = text.lower().replace(".", "").replace("?", "").split()
    v = np.mean([V[w] for w in words], axis=0)
    return v / np.linalg.norm(v)

chunks = ["The cat sat on the mat.", "The cat sleeps all day.", "The cat eats fish."]
store = np.array([embed(c) for c in chunks])                 # indexing phase

question = "What does the cat eat?"
scores = store @ embed(question)                             # [0.868, 0.997, 0.956]
top = np.argsort(-scores)[:2]                                # [1, 2] -> c2, c3
context = "\n".join(f"[{i}] {chunks[j]}" for i, j in enumerate(top, start=1))
prompt = f"Use only the context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

def context_precision(relevant):                             # relevant: 1 or 0 for each rank
    hits = np.cumsum(relevant)
    return sum(h / k for k, (h, v) in enumerate(zip(hits, relevant), start=1) if v) / max(sum(relevant), 1)

context_precision([0, 1])                                    # 0.5 (c2, then c3)
context_precision([1, 0])                                    # 1.0 (c3, then c2)
```

Real-library sketch (the tests do not run this code). LlamaIndex reads the key of the LLM provider from the environment.

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

documents = SimpleDirectoryReader("data").load_data()        # load
index = VectorStoreIndex.from_documents(documents)           # split, embed, store
engine = index.as_query_engine(similarity_top_k=2)           # retrieve, build the prompt, call the LLM
print(engine.query("What does the cat eat?"))
```

## Comparison

| Retriever | What it changes | Toy result | Added cost | Good for |
|---|---|---|---|---|
| Vector search | Nothing | c2, c3, c1 | None | A first version |
| Query rewriting | The question | c1 at 0.979 for the second question | 1 LLM call | Chats, unclear questions |
| HyDE | The text that is embedded | c3 first (1.000) | 1 LLM call | Questions that do not look like the documents |
| Multi-vector | The index | 15 vectors, not 3 | Storage, compute | Details in long chunks |
| BM25 | The score | c3 first (1.359) | A keyword index | Names, codes, rare words |
| Hybrid | The score | c3 first (0.841) | 2 searches | Mixed questions |

## Checks and common misconceptions

- **"RAG stops hallucinations."** RAG makes them less frequent. The LLM can ignore the context or mix it with its memory. Check: measure the faithfulness on a test set.
- **"More chunks give a better answer."** A larger k increases the context recall, but it adds noise and tokens. In our example, k = 2 was necessary to get c3, and it also brought c2 into the answer. Check: measure the 2 context metrics for 2 or 3 values of k.
- **"The answer is wrong, so I must change the prompt or the model."** Look at the retrieved chunks first. Check: for 20 failed questions, read the context. If the fact is not there, the problem is the retrieval.

## Sources

This page follows section 3 of "The LLM Engineer" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [LlamaIndex - High-level concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html).
- [Model Context Protocol](https://modelcontextprotocol.io/introduction).
- [Pinecone - Retrieval Augmentation](https://www.pinecone.io/learn/series/langchain/langchain-retrieval-augmentation/).
- [LangChain - Q&A with RAG](https://python.langchain.com/docs/tutorials/rag/) (chunks of 1,000 characters, overlap of 200, k = 4).
- [LangChain - Memory types](https://python.langchain.com/docs/how_to/chatbots_memory/).
- [RAG pipeline - Metrics](https://docs.ragas.io/en/stable/concepts/metrics/index.html), Ragas (context precision, context recall, faithfulness), and [DeepEval - Answer relevancy](https://deepeval.com/docs/metrics-answer-relevancy).
- [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496), Gao et al., 2022 (HyDE).
- [ColBERT](https://arxiv.org/abs/2004.12832), Khattab and Zaharia, 2020 (one vector for each token).
- [Chain-of-Retrieval Augmented Generation](https://arxiv.org/abs/2501.14342), Wang et al., 2025 (CoRAG).
- [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019), Robertson and Zaragoza, 2009, and [Elasticsearch - Similarity settings](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html) (default values of BM25).
