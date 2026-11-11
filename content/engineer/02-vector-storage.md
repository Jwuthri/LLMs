# 2. Building a vector storage

## Core idea

An LLM knows only its training data. To let it use your documents, you first make them searchable by meaning.
You cut each document into small pieces, turn each piece into a vector (a list of numbers), and keep the vectors in a store.
A search is then geometry: the result is the pieces whose vectors point in the same direction as the query vector.

## The example

This page uses one small file, `cat.md`, and one query, "the cat sat". The pages [3. RAG](03-rag.md) and [4. Advanced RAG](04-advanced-rag.md) use the same file.

```text
# The cat

## Habits
The cat sat   on the mat. The cat sleeps all day.

## Food
The cat eats fish.

Page 1 of 1
```

The embeddings extend the [toy example](../toy-example.md). Each word has a vector of 2 numbers. The first 3 rows are the toy vectors. We fix the other rows by hand.

| Words | Vector |
|---|---|
| `the` (toy), `on`, `all` | [1, 0] |
| `cat` (toy) | [0, 1] |
| `sat` (toy) | [1, 1] |
| `mat` | [2, 2] |
| `sleeps`, `day`, `fish` | [0, 2] |
| `eats` | [0, 3] |

## 1. Ingesting documents

### Setup

**Ingestion** is the work to get text out of a source (a file, a web page, a database) and into your pipeline.
A **document loader** is a function that reads one format. It returns the text and the **metadata** (data about the text, for example the file name or the page number).

<figure class="diagram" data-src="vector-storage/pipeline.svg">
<figcaption>The 5 steps that build the store, and the search that uses it, with the numbers of this page.</figcaption>
</figure>

### Steps

1. **Load the file.** The loader reads `cat.md`. It returns a text of 112 characters and the metadata `{"source": "cat.md"}`.

    **Why:** the pipeline works on plain text, but each format (PDF, HTML, JSON) hides the text in a different way. The metadata lets you show the source of an answer later.

2. **Clean the text.** Remove the footer line `Page 1 of 1` and the run of 3 spaces. 96 characters stay.

    **Why:** each character that you keep goes into a chunk, and then into a vector. A footer that is on each page makes all chunks look alike.

<div class="key-point">

The store can only find text that the loader extracted correctly. Bad ingestion sets an upper limit on the quality of each later step.

</div>

### From toy to real

- LangChain and LlamaIndex have loaders for PDF, HTML, Markdown and JSON files, and for services such as GitHub or Google Drive.
- The hard cases are tables, pages with 2 columns, scanned pages and repeated headers and footers.
- Keep the metadata: source, page, date. [Page 4](04-advanced-rag.md) uses it to filter the search.

## 2. Splitting documents

### Setup

A **chunk** is a piece of a document that you embed and store as one unit. A splitter has 2 settings: the **chunk size** (the maximum length) and the **chunk overlap** (the length that 2 neighbour chunks share).
The body of our file has 15 words. We count in words, and the chunk size is 6 words.

### Steps

1. **Fixed size: cut after each 6 words, with an overlap of 2.** Each chunk starts 6 − 2 = 4 words after the chunk before.

    | Chunk | Words | Text |
    |---|---|---|
    | 1 | 1 to 6 | `The cat sat on the mat.` |
    | 2 | 5 to 10 | `the mat. The cat sleeps all` |
    | 3 | 9 to 14 | `sleeps all day. The cat eats` |
    | 4 | 13 to 15 | `cat eats fish.` |

    For a text of $N$ words, a size $s$ and an overlap $o$, the number of chunks is:

    $$\left\lceil \frac{N - o}{s - o} \right\rceil = \left\lceil \frac{15 - 2}{6 - 2} \right\rceil = \lceil 3.25 \rceil = 4$$

    **Why:** this is the simplest rule, and each chunk fits the limit of the embedding model. But the rule does not see the meaning. No chunk contains the full sentence "The cat sleeps all day.", and chunk 3 mixes sleep with food.

2. **Recursive: cut at the largest natural border first.** The separators are, in order: paragraph, sentence, word. Keep a piece that has 6 words or less. Cut a longer piece with the next separator.

    - Paragraph 1 has 11 words. It is too long. Its sentences have 6 words and 5 words. Keep them.
    - Paragraph 2 has 4 words. Keep it.

    The result is 3 chunks: the 3 sentences.

    **Why:** a paragraph or a sentence is a unit of meaning. A chunk with one complete idea gives a vector with one clear direction.

3. **By header: cut at each `##` line, and keep the header as metadata.** The section "Habits" has 11 words and the section "Food" has 4 words. "Habits" is too long, thus we apply step 2 in it.

    | Id | Chunk | Words | Metadata |
    |---|---|---|---|
    | c1 | `The cat sat on the mat.` | 6 | section = Habits |
    | c2 | `The cat sleeps all day.` | 5 | section = Habits |
    | c3 | `The cat eats fish.` | 4 | section = Food |

    **Why:** the author already divided the text by subject. The header says what a chunk is about, also when the chunk text does not say it.

<div class="key-point">

The chunk is the unit that the search returns. Split where the meaning changes, not where a counter stops, and keep the metadata with each chunk.

</div>

### From toy to real

- Real splitters count characters or tokens, not words. The recursive splitter of LangChain tries the separators `"\n\n"`, `"\n"`, `" "` and `""` in this order. It also joins small neighbour pieces until the chunk size is full.
- A small chunk gives an accurate vector but little context for the LLM. A large chunk gives more context, but its vector is the mean of many ideas.
- The overlap repeats text, so it costs storage. The [chunking widget on page 3](03-rag.md) lets you try both settings.

## 3. Embedding models

### Setup

An **embedding model** turns a text into one vector. Texts with a similar meaning get vectors that point in a similar direction.
Our toy model takes the vector of each word and calculates the mean. This is **mean pooling**. The model changes the text to lower case and ignores the punctuation.

### Steps

1. **Embed each chunk.** For c3, "the cat eats fish":

    $$\frac{[1, 0] + [0, 1] + [0, 3] + [0, 2]}{4} = \frac{[1, 6]}{4} = [0.25,\ 1.5]$$

    | Chunk | Sum of the word vectors / number of words | Vector | Length |
    |---|---|---|---|
    | c1 | [6, 4] / 6 | [1, 0.667] | 1.202 |
    | c2 | [2, 5] / 5 | [0.4, 1] | 1.077 |
    | c3 | [1, 6] / 4 | [0.25, 1.5] | 1.521 |

    The **length** of a vector $[x, y]$ is $\sqrt{x^2 + y^2}$.

    **Why:** the store needs one vector of a fixed size for each chunk, for a text of any length.

2. **Embed the query with the same model.** "the cat sat" gives $([1, 0] + [0, 1] + [1, 1]) / 3 = [0.667,\ 0.667]$. Its length is 0.943.

    **Why:** vectors from 2 different models are in 2 different spaces. You cannot compare them.

3. **Calculate the dot product** $q \cdot c = q_1 c_1 + q_2 c_2$. For c1: $0.667 \times 1 + 0.667 \times 0.667 = 1.111$. For c2: 0.933. For c3: 1.167. The ranking is c3, c1, c2.

    **Why:** the dot product is the cheapest measure. It is large when 2 vectors point in the same direction, but also when a vector is long. Here c3 is first only because it is the longest vector.

4. **Calculate the cosine similarity**: divide the dot product by the 2 lengths.

    $$\cos(q, c) = \frac{q \cdot c}{\lVert q \rVert \, \lVert c \rVert} \qquad \cos(q, c_1) = \frac{1.111}{0.943 \times 1.202} = 0.981$$

    For c2: 0.919. For c3: 0.814. The ranking is c1, c2, c3.

    **Why:** the cosine keeps only the angle between the vectors. The length of a vector frequently shows the length of the text or the frequency of its words, not its meaning.

5. **Calculate the Euclidean distance**: the length of the straight line between 2 points. Smaller is nearer.

    $$d(q, c_1) = \sqrt{(0.667 - 1)^2 + (0.667 - 0.667)^2} = 0.333$$

    For c2: 0.427. For c3: 0.932. The ranking is c1, c2, c3.

    **Why:** some indexes only work with a distance. Here the distance agrees with the cosine, but a very long vector can change this ranking.

6. **Normalize: divide each vector by its length.** Each vector now has a length of 1.

    | Vector | c1 | c2 | c3 | Query |
    |---|---|---|---|---|
    | Normalized | [0.832, 0.555] | [0.371, 0.928] | [0.164, 0.986] | [0.707, 0.707] |
    | Dot product with the query | 0.981 | 0.919 | 0.814 | |
    | Euclidean distance to the query | 0.197 | 0.402 | 0.610 | |

    For c1: $0.707 \times 0.832 + 0.707 \times 0.555 = 0.981$. This is the cosine of step 4.

    **Why:** when the 2 lengths are 1, the cosine formula divides by 1, so it *is* the dot product. The distance follows the cosine also: $d^2 = 2 - 2\cos$. You normalize one time, when you build the store. Then the cheapest measure gives the correct ranking.

<figure class="plot">
<img src="/figures/vector-storage/plane.svg" alt="The 3 chunk vectors and the query vector in a plane. The query at 45 degrees is nearest in angle to c1. The vector c3 is the longest.">
<figcaption>The cosine measures the angle between 2 lines. The empty circles are the normalized vectors: they all have a length of 1.</figcaption>
</figure>

<div class="llm-widget" data-widget="cosine"></div>

<div class="key-point">

An embedding model turns "similar meaning" into "small angle". Normalize your vectors. Then the dot product, the cosine similarity and the Euclidean distance give the same ranking.

</div>

### From toy to real

- A real embedding model is a Transformer, frequently an encoder (see [the LLM architecture](../scientist/01-architecture.md)). Its attention layers mix the context into each token vector before the pooling. Our mean gives the same vector to "the cat eats fish" and "the fish eats cat".
- Real vectors are larger: 384 numbers for `all-MiniLM-L6-v2` and 1,024 numbers for `bge-large-en-v1.5`. Two numbers cannot keep many meanings apart: `sleeps`, `day` and `fish` share one vector in our table.
- Each model has a maximum input: 256 word pieces for `all-MiniLM-L6-v2` and 512 tokens for `bge-large-en-v1.5`. The model ignores the text after this limit, so the chunk size must be below it.
- Use the MTEB leaderboard to make a short list of models. Then test them on your own documents.
- If you change the model, you must embed all the chunks again.

## 4. Vector databases

### Setup

A **vector database** stores the vectors with their text and metadata. For a query vector, it returns the **top-k**: the k stored vectors that are most similar.
Our store is a matrix of 3 rows and 2 columns (the normalized vectors of step 6), with the 3 texts and their metadata.

### Steps

1. **Exact search: compare the query with each row.** This is 3 dot products of 2 numbers: $N \times d = 3 \times 2 = 6$ multiplications. Then sort. The top-2 is c1 (0.981) and c2 (0.919).

    **Why:** if you compare with each vector, you are sure to find the best one. This is a **flat** index. Its cost grows in proportion to the number of vectors.

2. **IVF: put the vectors into clusters, then search only the nearest cluster.** An **inverted file (IVF)** index runs k-means on the vectors. With 2 clusters we get A = {c1} and B = {c2, c3}. The **centroid** (the mean direction) of A is [0.832, 0.555], and the centroid of B is [0.269, 0.963]. For the query, the centroid scores are 0.981 for A and 0.871 for B. We search only A, and we find c1.

    **Why:** most vectors are far from the query. One comparison with a centroid lets you ignore a full group of vectors.

3. **See the index fail.** A second query, "the cat sat all day", has the vector [0.6, 0.8]. The exact search gives c2 (0.966), then c1 (0.943). The centroid scores are 0.943 for A and 0.932 for B. Thus the index searches only A and returns c1. The best chunk, c2, is in cluster B.

    **Recall** is the part of the true top-k that the index returns. For our 2 queries and k = 1, the recall is 1 / 2 = 0.5. If the index searches 2 clusters (`nprobe = 2`), the recall is 1.0, but the index reads all the vectors again.

    **Why:** an **approximate nearest neighbour (ANN)** index is fast because it reads fewer vectors, so it can miss the best one. The setting `nprobe` (the number of clusters to read) moves the index between "fast" and "exact".

4. **HNSW: walk a graph.** A **hierarchical navigable small world (HNSW)** index links each vector to its near neighbours. Our graph is c1 – c2 – c3. For the query "the cat sat", start at c3 (0.814). Its neighbour c2 is better (0.919): move. The neighbour c1 is better (0.981): move. No neighbour is better: stop and return c1.

    **Why:** each step moves nearer to the query, and the search never reads the vectors that are far from the path. A real index adds upper layers with few vectors and long links, so the first steps are long.

<figure class="diagram" data-src="vector-storage/ann.svg">
<figcaption>Left: IVF reads cluster A only and does not see c2, the true best chunk. Right: HNSW moves from neighbour to neighbour until no neighbour is nearer.</figcaption>
</figure>

<div class="key-point">

Exact search is simple and always correct, but its cost is N × d for each query. An ANN index gives up a small part of the recall to read far fewer vectors.

</div>

### From toy to real

Take 1 million vectors of 1,024 numbers, each stored as a 32-bit float (4 bytes).

- **Memory.** 1,000,000 × 1,024 × 4 bytes = 4,096,000,000 bytes = 4.096 GB.
- **Exact search.** 1,000,000 × 1,024 = 1.024 billion multiplications for each query.
- **IVF.** For a store of less than 1 million vectors, FAISS recommends $4\sqrt{N}$ to $16\sqrt{N}$ clusters. At this limit, that is 4,000 to 16,000 clusters. Assume 4,000 clusters of the same size (250 vectors) and `nprobe = 16`. One query reads 4,000 centroids + 16 × 250 vectors = 8,000 vectors. This is 0.8% of the store.
- **HNSW.** With `M = 16` links for each vector, FAISS gives a memory use of $d \times 4 + M \times 2 \times 4$ bytes for each vector. The links add 128 bytes for each vector, 0.128 GB in total: 4.224 GB with the vectors.
- **Tools.** FAISS and Annoy are libraries that hold an index in memory. Chroma, Milvus and Pinecone are databases: they add storage on disk, metadata filters, updates and an API.

## Minimal code

This code gives the same numbers as the steps of sub-part 3 and the exact search.

```python
import numpy as np

V = {"the": [1, 0], "cat": [0, 1], "sat": [1, 1], "on": [1, 0], "mat": [2, 2],
     "sleeps": [0, 2], "all": [1, 0], "day": [0, 2], "eats": [0, 3], "fish": [0, 2]}

def embed(text):                                    # mean pooling
    return np.mean([V[w] for w in text.lower().replace(".", "").split()], axis=0)

chunks = ["The cat sat on the mat.", "The cat sleeps all day.", "The cat eats fish."]
C = np.array([embed(c) for c in chunks])            # [[1, 0.667], [0.4, 1], [0.25, 1.5]]
q = embed("the cat sat")                            # [0.667, 0.667]

dot = C @ q                                         # [1.111, 0.933, 1.167] -> c3 first
cosine = dot / (np.linalg.norm(C, axis=1) * np.linalg.norm(q))   # [0.981, 0.919, 0.814]
distance = np.linalg.norm(C - q, axis=1)            # [0.333, 0.427, 0.932]

C_unit = C / np.linalg.norm(C, axis=1, keepdims=True)   # normalize one time, at indexing
q_unit = q / np.linalg.norm(q)
scores = C_unit @ q_unit                            # dot product = cosine
top_2 = np.argsort(-scores)[:2]                     # [0, 1] -> c1, c2
```

Real-library sketch (the tests do not run this code):

```python
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")              # 384 numbers for each text
vectors = model.encode(chunks, normalize_embeddings=True)
index = faiss.IndexFlatIP(384)                               # exact search, inner (dot) product
# index = faiss.IndexHNSWFlat(384, 16, faiss.METRIC_INNER_PRODUCT)   # ANN, M = 16
index.add(vectors)
scores, ids = index.search(model.encode(["the cat sat"], normalize_embeddings=True), 2)
```

## Comparison

| Index | How it searches | Toy result | Vectors read at N = 1 million | Recall | Extra memory |
|---|---|---|---|---|---|
| Flat (exact) | Reads each vector | c1 for query 1, c2 for query 2 | 1,000,000 | 1.0 | None |
| IVF | Reads the `nprobe` nearest clusters | Misses c2 for query 2 | 8,000 (`nprobe = 16`) | Below 1.0. A larger `nprobe` increases it | The centroids |
| HNSW | Walks a graph of neighbours | c3 → c2 → c1 for query 1 | Grows slowly with N | Below 1.0. A wider search increases it | 0.128 GB of links (`M = 16`) |

## Checks and common misconceptions

- **"A higher dot product means a more similar text."** Only if the vectors have the same length. In step 3, c3 was first with 1.167 because it was long. Check: read the model card to see if the model gives normalized vectors. If not, normalize them.
- **"The index returns the best matches."** An ANN index returns matches that are *probably* the best. Check: run 100 real queries on a flat index and on your ANN index, and calculate the recall.
- **"A larger chunk is always better, because the LLM gets more context."** The vector of a large chunk is the mean of many ideas, so it matches no question well. Check: read 20 chunks without their neighbours. Can you say what each chunk is about?

## Sources

This page follows section 2 of "The LLM Engineer" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [LangChain - Text splitters](https://docs.langchain.com/oss/python/integrations/splitters).
- [Sentence Transformers library](https://www.sbert.net/).
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
- [The Top 7 Vector Databases](https://www.datacamp.com/blog/the-top-5-vector-databases), Moez Ali.
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384 dimensions, 256 word pieces) and [bge-large-en-v1.5 model card](https://huggingface.co/BAAI/bge-large-en-v1.5) (1,024 dimensions, 512 tokens).
- [FAISS - Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) (flat index, number of IVF clusters, memory of HNSW).
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320), Malkov and Yashunin, 2016 (HNSW).
