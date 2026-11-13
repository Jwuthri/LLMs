# 4. Advanced RAG

## Core idea

The RAG loop of [page 3](03-rag.md) has one tool: a vector search in a set of texts. Real questions need more: a number from a database, a calculation, a better ranking.
Advanced RAG adds a step before the search (select and write the correct query) and a step after the search (clean the results).
It also replaces manual prompt writing with a measurement: you select each prompt with a metric.

## The example

This page uses the 3 chunks of [page 2](02-vector-storage.md) and the question of page 3, "What does the cat eat?". The correct chunk is c3. Page 3 gave 2 ranked lists for this question:

| Search | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| Vector (cosine) | c2 "The cat sleeps all day." | c3 "The cat eats fish." | c1 "The cat sat on the mat." |
| Keyword (BM25) | c3 | c1 | c2 |

The new part is a database table, `meals`, with 3 rows.

| day | food | grams |
|---|---|---|
| Monday | fish | 60 |
| Tuesday | fish | 80 |
| Wednesday | chicken | 50 |

<figure class="diagram" data-src="advanced-rag/pipeline.svg">
<figcaption>The 4 sub-parts of this page, as steps around the basic RAG loop.</figcaption>
</figure>

## 1. Query construction

### Setup

The question is "How many grams of fish did the cat eat?". A vector search cannot answer it: no chunk contains the number, and a sum needs all the rows.
**Query construction** means that the LLM translates the question into the query language of the data source: SQL for a table, Cypher for a graph database, a metadata filter for a vector store.

### Steps

1. **Give the schema to the LLM.** The prompt contains the question and the definition of the table: `CREATE TABLE meals (day TEXT, food TEXT, grams INTEGER)`.

    **Why:** the LLM must know the exact names of the tables and columns. If it does not, it invents them.

2. **The LLM writes the query.** This is **text-to-SQL**. Assume this output:

    ```sql
    SELECT SUM(grams) FROM meals WHERE food = 'fish';
    ```

    **Why:** the database filters and adds exactly. An LLM that reads the rows does the same work approximately.

3. **Run the query with read-only access.** The database returns 60 + 80 = 140. Our script sets `PRAGMA query_only = ON` first. A `DELETE` statement then gives an error.

    **Why:** the SQL is model output, so it is input that you cannot trust (see [Securing LLMs](08-securing.md)). Read-only access limits the damage of a bad query.

4. **Give the result to the LLM.** The prompt contains the question, the SQL and the result 140. Assume the answer "The cat ate 140 grams of fish."

    **Why:** the user wants a sentence, not a row of a table.

5. **Use a metadata filter for the text search.** For "What does the cat eat?", assume that the LLM writes a filter with the query: `section = "Food"`. The store keeps 1 chunk of 3 before the ranking. The result is c3 (0.956). The chunk c2 (0.997), which was first on page 3, is not a candidate.

    **Why:** a filter is exact, and a similarity is not. The filter removes wrong candidates before the ranking can prefer them.

<div class="key-point">

Structured data needs a structured query. Let the LLM write the query, and let the database calculate.

</div>

### From toy to real

- A real database has hundreds of tables. First retrieve the definitions of the relevant tables, then put only those into the prompt. Some example rows and some pairs of question and SQL help the LLM.
- If the query fails, send the error message back to the LLM and let it correct the query.
- A **self-query retriever** lets the LLM write the search text and the metadata filter in one step.

## 2. Tools

### Setup

A **tool** is a function that the application can run for the LLM. Each tool has a name and a description. An **agent** is an LLM that selects tools in a loop (see [5. Agents](05-agents.md)).
We have 3 tools. The first task is a retrieval task: which tool answers this question?

| Tool | Description |
|---|---|
| `search` | Find a text about the cat in the documents. |
| `sql` | Count or sum the grams of food in the meals table. |
| `calculator` | Add or multiply two numbers. |

### Steps

1. **Give a score to each tool description.** We use BM25 from page 3. The 3 descriptions are the documents.

    | Question | `search` | `sql` | `calculator` | Best tool |
    |---|---|---|---|---|
    | What does the cat eat? | **1.582** | 0.593 | 0 | `search` |
    | How many grams of fish did the cat eat? | 1.582 | **2.327** | 0 | `sql` |
    | Add 60 and 80. | 0 | 0 | **1.173** | `calculator` |

    **Why:** to select a tool is to rank texts by their relevance to the question. This is the same operation as the retrieval of a chunk. A real agent lets the LLM read the descriptions and select. With hundreds of tools, a retrieval step such as this one first keeps the best descriptions, so that they fit in the prompt.

2. **Run the tool.** `search` returns c2 and c3. `sql` returns 140. `calculator` returns 60 + 80 = 140.

    **Why:** each tool does a thing that an LLM does badly: find private facts, add many rows, calculate exactly.

3. **Put the result into the prompt.** The LLM writes the answer from it.

    **Why:** a tool result is context, the same as a retrieved chunk. Basic RAG is the special case with one tool that always runs.

<div class="key-point">

A tool description is a small document. The quality of the tool selection depends on the descriptions, as the quality of the retrieval depends on the chunks.

</div>

### From toy to real

- With **function calling**, the LLM returns the name of a tool and its arguments as JSON. The application runs the function.
- Common tools are a web search, Wikipedia, a Python interpreter, or a product such as Jira.
- One question can need many tool calls, one after the other. That loop is the subject of [page 5](05-agents.md).

## 3. Post-processing

### Setup

**Post-processing** is the work between the retrieval and the prompt. Its input is the 2 ranked lists of the example. Its aim is to put c3 first and to remove text that does not help.

### Steps

1. **Re-rank with a cross-encoder.** A **cross-encoder** is a model that reads the question and one chunk together and gives one relevance score. Assume the scores 0.95 for c3, 0.30 for c2 and 0.10 for c1. The top-2 of the vector search, c2 then c3, becomes c3 then c2. If you keep only the first chunk, the context goes from 9 tokens to 4 tokens.

    **Why:** the embedding model made the vector of each chunk before it saw the question. The cross-encoder sees the 2 texts, so its attention can compare each word of the question with each word of the chunk. It is more accurate, but it needs one model call for each chunk. Thus you use it only on the retrieved chunks, not on the full store.

2. **Fuse the 2 lists with reciprocal rank fusion (RRF).** Each list gives $1 / (k + \text{rank})$ to a chunk, with $k = 60$. Add the values.

    $$\text{RRF}(d) = \sum_{\text{lists}} \frac{1}{k + \text{rank}(d)}$$

    | Chunk | Vector rank | Keyword rank | Calculation | RRF score |
    |---|---|---|---|---|
    | c1 | 3 | 2 | 1/63 + 1/62 = 0.01587 + 0.01613 | 0.03200 |
    | c2 | 1 | 3 | 1/61 + 1/63 = 0.01639 + 0.01587 | 0.03227 |
    | c3 | 2 | 1 | 1/62 + 1/61 = 0.01613 + 0.01639 | **0.03252** |

    The order is c3, c2, c1. The scores are near, so this table shows 5 decimals.

    **Why:** RRF uses only the ranks. You do not have to put a cosine and a BM25 score on the same scale, as page 3 did. A chunk that is high in the 2 lists gets a better score than a chunk that is first in one list and far down in the other. The constant $k = 60$ makes the gap between rank 1 and rank 2 small, so one list cannot control the result. **RAG-fusion** uses the same formula on the lists of many rewritten questions.

3. **Add diversity with maximal marginal relevance (MMR).** MMR selects the chunks one at a time. The score of a candidate $d$ is its relevance minus its similarity to the chunks $S$ that are already selected.

    $$\text{MMR}(d) = \lambda \cdot \text{rel}(d) - (1 - \lambda) \cdot \max_{s \in S} \text{sim}(d, s)$$

    c3 is selected. The cosine similarity to c3 is 0.977 for c2 and 0.684 for c1. With $\lambda = 0.5$ and the relevance scores of step 1:

    - c2: 0.5 × 0.30 − 0.5 × 0.977 = −0.338
    - c1: 0.5 × 0.10 − 0.5 × 0.684 = −0.292

    MMR selects c1. With $\lambda = 0.7$, the scores are −0.083 for c2 and −0.135 for c1, and MMR selects c2.

    **Why:** 2 chunks that say the same thing use the context 2 times for one fact. MMR gives points for relevance and takes points for repetition. $\lambda$ sets the balance.

<div class="key-point">

Tune the retrieval for recall: the fact must be somewhere in the candidates. Tune the post-processing for precision: put the fact first and remove the other text.

</div>

### From toy to real

- A usual design retrieves many chunks with a fast search, re-ranks them, and keeps a small number for the prompt.
- The Sentence Transformers library has cross-encoder models, for example `cross-encoder/ms-marco-MiniLM-L6-v2`.
- **Classification** is one more filter: a small model gives a label to each chunk (the subject, or "relevant" and "not relevant"), and the pipeline removes chunks by label.
- The value $k = 60$ comes from the paper that introduced RRF.

## 4. Program LLMs

### Setup

The pipeline now has many prompts: rewrite the question, write the SQL, select a tool, write the answer. It is slow to tune each prompt manually, and a change in one step can break the next step.
**DSPy** is a framework in which you write the program (the steps, with their inputs and outputs) and a metric. An **optimizer** then searches for the prompts that give the highest metric. A prompt becomes a parameter that you optimize, as a weight is.
We do this search manually for the answer step: 3 candidate prompts, 5 examples.

### Steps

1. **Write the examples and the metric.** Each example is a question and its correct ("gold") answer. The metric is **exact match**: 1 if the answer has the same words as the gold answer, 0 if not. Capital letters and punctuation do not count.

    | # | Question | Gold answer |
    |---|---|---|
    | 1 | What does the cat eat? | fish |
    | 2 | Where did the cat sit? | on the mat |
    | 3 | How long does the cat sleep? | all day |
    | 4 | What sat on the mat? | the cat |
    | 5 | Does the cat eat fish? | yes |

    **Why:** an optimizer needs a number to increase. Without a metric, "this prompt is better" is an opinion.

2. **Run each candidate prompt on each example.** This is 3 × 5 = 15 LLM calls. Assume the answers. For example 1, P1 gives "The cat eats fish and meat." (0), P2 gives "The cat eats fish." (0) and P3 gives "Fish." (1).

    | Prompt | Text | Result for examples 1 to 5 | Score |
    |---|---|---|---|
    | P1 | Answer the question. | 0, 1, 0, 0, 1 | 2/5 = 0.4 |
    | P2 | Answer the question. Use only the context. | 0, 1, 1, 0, 1 | 3/5 = 0.6 |
    | P3 | Answer in a few words. Use only the context. | 1, 1, 1, 1, 0 | 4/5 = **0.8** |

    **Why:** you cannot know the effect of a prompt before you run it. Only a measurement on examples shows it.

3. **Keep the best prompt.** P3 has the highest score, 0.8.

    **Why:** propose, measure, keep. This is the loop that an optimizer does for you. A DSPy optimizer also writes the candidates, and it adds good examples to the prompt.

4. **Check the uncertainty.** With 5 examples, the standard error of the score 0.8 is $\sqrt{0.8 \times 0.2 / 5} = 0.179$. The difference between P3 and P2 is one example.

    **Why:** on a small set, you cannot tell an improvement from luck. Use more examples, and keep a test set that the optimizer does not see.

<figure class="plot">
<img src="/figures/advanced-rag/prompts.svg" alt="Bar chart of the exact-match score of 3 prompts on 5 examples: P1 0.4, P2 0.6, P3 0.8.">
<figcaption>Each bar is one candidate prompt. The optimizer keeps the highest bar.</figcaption>
</figure>

<div class="key-point">

A prompt is a parameter. Define a metric, measure the candidates on examples, keep the best. This is the training loop, with text in the place of weights.

</div>

### From toy to real

- DSPy has 3 building blocks. A **signature** names the inputs and outputs of a step. A **module** is a method to do the step (for example, think step by step). An **optimizer** tunes the prompts, and it can also fine-tune the weights of the model.
- The metric can be an LLM judge, for example the faithfulness of [page 3](03-rag.md).
- The cost is candidates × examples LLM calls for each step of the pipeline.

## Minimal code

This code gives the numbers of the SQL query, the RRF table and the prompt selection.

```python
import sqlite3

db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE meals (day TEXT, food TEXT, grams INTEGER)")
db.executemany("INSERT INTO meals VALUES (?, ?, ?)",
               [("Monday", "fish", 60), ("Tuesday", "fish", 80), ("Wednesday", "chicken", 50)])
db.execute("PRAGMA query_only = ON")                         # the query of the LLM cannot change the data
sql = "SELECT SUM(grams) FROM meals WHERE food = 'fish';"    # assume: the LLM wrote this
total = db.execute(sql).fetchall()                           # [(140,)]

def rrf(ranked_lists, k=60):
    scores = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0) + 1 / (k + rank)
    return scores

fused = rrf([["c2", "c3", "c1"], ["c3", "c1", "c2"]])        # c3 0.03252, c2 0.03227, c1 0.03200

results = {"P1": [0, 1, 0, 0, 1], "P2": [0, 1, 1, 0, 1], "P3": [1, 1, 1, 1, 0]}   # 1 = exact match
scores = {name: sum(r) / len(r) for name, r in results.items()}                  # 0.4, 0.6, 0.8
best = max(scores, key=scores.get)                           # "P3"
```

Real-library sketch (the tests do not run this code). DSPy reads the API key from the environment.

```python
import dspy

dspy.configure(lm=dspy.LM("provider/model-name"))
answer = dspy.ChainOfThought("context, question -> answer")  # signature and module

def exact_match(example, prediction, trace=None):            # metric
    return example.answer.lower() == prediction.answer.lower()

optimizer = dspy.BootstrapFewShot(metric=exact_match)        # optimizer
best_answer = optimizer.compile(answer, trainset=examples)   # examples: a list of dspy.Example
```

## Comparison

| Technique | Where | What it does | Toy result | Cost or risk |
|---|---|---|---|---|
| Text-to-SQL | Before the search | The LLM writes a database query | 140 grams | The query can be wrong or dangerous |
| Metadata filter | Before the search | Removes candidates exactly | 3 chunks → 1 | Needs clean metadata |
| Tool selection | Before the search | Selects the data source | 3 questions, 3 different tools | Needs good descriptions |
| Cross-encoder | After the search | Gives a new score to each chunk | c3, then c2 | One model call for each chunk |
| RRF | After the search | Fuses ranked lists | c3: 0.03252 | Ignores the size of the scores |
| MMR | After the search | Removes repetition | c1 second, with λ = 0.5 | Can remove a relevant chunk |
| Prompt optimization | All prompts | Selects a prompt with a metric | P3: 0.8 | 15 LLM calls here |

## Checks and common misconceptions

- **"A vector store can answer each question about my data."** It finds text. It does not count, add or sort. Check: put 50 real questions into 3 groups (find a text, calculate on rows, other). The second group needs a database query.
- **"A re-ranker repairs a bad retriever."** A re-ranker only changes the order of the chunks that it gets. If the correct chunk is not in the candidates, it stays lost. Check: measure the context recall before the re-ranker.
- **"The optimized prompt is the best prompt."** It is the best of the candidates, on your examples, for one model. Check: measure it on a test set that the optimizer did not see, and run the optimizer again when you change the model.

## Sources

This page follows section 4 of "The LLM Engineer" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [LangChain - Query Construction](https://blog.langchain.dev/query-construction/).
- [LangChain - Build a SQL agent](https://docs.langchain.com/oss/python/langchain/sql-agent).
- [Pinecone - LLM agents](https://www.pinecone.io/learn/series/langchain/langchain-agents/).
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/), Lilian Weng.
- [LangChain - OpenAI's RAG](https://blog.langchain.dev/applying-openai-rag/).
- [DSPy in 8 Steps](https://web.archive.org/web/20241016053304/https://dspy-docs.vercel.app/docs/building-blocks/solving_your_task) (archived copy of 2024; the page is no longer in the DSPy documentation) and [DSPy](https://arxiv.org/abs/2310.03714), Khattab et al., 2023.
- [RAG-fusion](https://github.com/Raudaschl/rag-fusion), Adrian Raudaschl.
- [Reciprocal rank fusion outperforms Condorcet and individual rank learning methods](https://doi.org/10.1145/1571941.1572114), Cormack, Clarke and Büttcher, 2009 (RRF, k = 60).
- [The use of MMR, diversity-based reranking for reordering documents and producing summaries](https://doi.org/10.1145/290941.291025), Carbonell and Goldstein, 1998 (MMR).
- [Sentence Transformers - Cross-encoders](https://www.sbert.net/docs/cross_encoder/usage/usage.html).
