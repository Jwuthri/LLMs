"""Numbers and plots for docs/engineer/02-vector-storage.md.

Run: uv run python -m examples.engineer.vector_storage
"""

import math
import re

import numpy as np

from examples.engineer.corpus import CHUNK_IDS, CHUNKS, Chunk, chunk_vectors, cosine, embed, normalize
from examples.plotstyle import save_light_dark

PAGE = "vector-storage"

# --- 1. Ingest: load one raw file, then clean it --------------------------------------------

RAW_DOCUMENT = (
    "# The cat\n"
    "\n"
    "## Habits\n"
    "The cat sat   on the mat. The cat sleeps all day.\n"
    "\n"
    "## Food\n"
    "The cat eats fish.\n"
    "\n"
    "Page 1 of 1\n"
)


def load() -> dict:
    """A loader returns the text of one file and its metadata."""
    return {"text": RAW_DOCUMENT, "metadata": {"source": "cat.md"}}


def clean(text: str) -> str:
    """Remove the page footer and each run of 2 or more spaces."""
    lines = [line for line in text.splitlines() if not re.fullmatch(r"Page \d+ of \d+", line)]
    return re.sub(r" {2,}", " ", "\n".join(lines)).strip()


# --- 2. Split ------------------------------------------------------------------------------

def split_by_header(text: str) -> list[tuple[str, str]]:
    """(header, body) for each `## header` section of a Markdown text."""
    sections = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    return [(header.strip(), body.strip()) for header, body in (s.split("\n", 1) for s in sections)]


def body_text(text: str) -> str:
    """The text without its header lines. Sections are separated by one empty line."""
    return "\n\n".join(body for _, body in split_by_header(text))


def split_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Chunks of `size` words. Each chunk starts `size - overlap` words after the one before."""
    words = text.split()
    chunks = []
    for start in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def fixed_chunk_count(words: int, size: int, overlap: int) -> int:
    """Number of chunks that `split_fixed` gives for a text of `words` words."""
    return max(1, math.ceil((words - overlap) / (size - overlap)))


# Separators of the recursive splitter, largest unit first: paragraph, sentence, word.
SEPARATORS = [r"\n\n", r"(?<=\.)\s", r"\s"]


def split_recursive(text: str, size: int, separators: list[str] = SEPARATORS) -> list[str]:
    """Keep a piece if it has `size` words or less. If not, cut it with the next separator."""
    if len(text.split()) <= size:
        return [text]
    pieces = re.split(separators[0], text)
    return [chunk for piece in pieces for chunk in split_recursive(piece, size, separators[1:])]


def build_chunks(size: int = 6) -> list[Chunk]:
    """Split by header, then recursively inside each section. Each chunk keeps its header."""
    document = load()
    sections = split_by_header(clean(document["text"]))
    pieces = [(text, header) for header, body in sections for text in split_recursive(body, size)]
    return [
        Chunk(f"c{i}", text, header, document["metadata"]["source"])
        for i, (text, header) in enumerate(pieces, start=1)
    ]


# --- 3. Embed and compare -------------------------------------------------------------------

QUERY = "the cat sat"
QUERY_2 = "the cat sat all day"  # the query that the IVF index gets wrong


def similarity_table(query_vector: np.ndarray, vectors: np.ndarray) -> dict[str, np.ndarray]:
    """Dot product, cosine similarity and Euclidean distance between the query and each row."""
    return {
        "dot": vectors @ query_vector,
        "cosine": np.array([cosine(query_vector, v) for v in vectors]),
        "euclidean": np.linalg.norm(vectors - query_vector, axis=1),
    }


def ranking(scores: np.ndarray, largest_first: bool = True) -> list[str]:
    """Chunk ids in the order of the scores."""
    order = np.argsort(-scores if largest_first else scores, kind="stable")
    return [CHUNK_IDS[i] for i in order]


def top_k(query_vector: np.ndarray, vectors: np.ndarray, k: int) -> list[tuple[str, float]]:
    """Exact search: compare the query with each stored vector, keep the k best (cosine)."""
    scores = normalize(vectors) @ normalize(query_vector)
    return [(CHUNK_IDS[i], float(scores[i])) for i in np.argsort(-scores, kind="stable")[:k]]


# --- 4. Approximate nearest neighbour (ANN) indexes ------------------------------------------

def kmeans(vectors: np.ndarray, start: list[int], rounds: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """K-means on unit vectors. Returns the unit centroids and the cluster of each vector."""
    centroids = vectors[start].copy()
    for _ in range(rounds):
        assignment = np.argmax(vectors @ centroids.T, axis=1)
        centroids = normalize(np.array([vectors[assignment == c].mean(axis=0) for c in range(len(start))]))
    return centroids, assignment


def ivf_search(query_vector: np.ndarray, vectors: np.ndarray, centroids: np.ndarray,
               assignment: np.ndarray, nprobe: int) -> tuple[str, int]:
    """Search only the `nprobe` clusters with the nearest centroids. Returns (best id, comparisons)."""
    q = normalize(query_vector)
    probed = np.argsort(-(centroids @ q), kind="stable")[:nprobe]
    candidates = [i for i in range(len(vectors)) if assignment[i] in probed]
    best = max(candidates, key=lambda i: float(vectors[i] @ q))
    return CHUNK_IDS[best], len(centroids) + len(candidates)


def ivf_recall(queries: list[str], vectors: np.ndarray, nprobe: int) -> float:
    """Recall at k = 1: the share of the queries for which IVF returns the same chunk as the exact search."""
    unit = normalize(vectors)
    centroids, assignment = kmeans(unit, start=[0, len(vectors) - 1])
    hits = [
        ivf_search(embed(text), unit, centroids, assignment, nprobe)[0] == top_k(embed(text), vectors, 1)[0][0]
        for text in queries
    ]
    return sum(hits) / len(hits)


# A hand-made graph for the HNSW walk: each chunk is linked to its nearest neighbours.
GRAPH = {"c1": ["c2"], "c2": ["c1", "c3"], "c3": ["c2"]}


def greedy_walk(query_vector: np.ndarray, vectors: np.ndarray, entry: str,
                graph: dict[str, list[str]] = GRAPH) -> list[tuple[str, float]]:
    """Start at `entry`. Move to the neighbour that is nearest to the query. Stop when none is nearer."""
    q = normalize(query_vector)
    score = {chunk_id: float(v @ q) for chunk_id, v in zip(CHUNK_IDS, vectors)}
    path = [entry]
    while True:
        best = max(graph[path[-1]], key=score.get)
        if score[best] <= score[path[-1]]:
            return [(node, score[node]) for node in path]
        path.append(best)


def scale_numbers(n: int = 1_000_000, d: int = 1_024, hnsw_m: int = 16, clusters: int = 4_000,
                  nprobe: int = 16) -> dict[str, float]:
    """Cost of exact search, memory and IVF cost for `n` float32 vectors of `d` dimensions."""
    vector_bytes = n * d * 4
    hnsw_link_bytes = n * hnsw_m * 2 * 4  # FAISS: (d * 4 + M * 2 * 4) bytes per vector
    ivf_comparisons = clusters + nprobe * n // clusters
    return {
        "exact_multiplications": n * d,
        "vector_gb": vector_bytes / 1e9,
        "vector_gib": vector_bytes / 2**30,
        "hnsw_link_gb": hnsw_link_bytes / 1e9,
        "hnsw_total_gb": (vector_bytes + hnsw_link_bytes) / 1e9,
        "ivf_comparisons": ivf_comparisons,
        "ivf_fraction": ivf_comparisons / n,
    }


# --- Plot ----------------------------------------------------------------------------------

def make_figures() -> None:
    vectors = chunk_vectors()
    query_vector = embed(QUERY)

    def draw(ax, theme):
        arc = np.linspace(0, np.pi / 2, 100)
        ax.plot(np.cos(arc), np.sin(arc), color=theme["axis"], linewidth=1, linestyle="--")
        ax.text(1.02, 0.08, "length 1", color=theme["muted"], fontsize=8)
        items = [(chunk.id, v, theme["series"][i], (7, 3), "left")
                 for i, (chunk, v) in enumerate(zip(CHUNKS, vectors))]
        items.append(("query", query_vector, theme["series"][3], (-9, 6), "right"))
        for label, v, colour, offset, align in items:
            unit = normalize(v)
            ax.plot([0, v[0]], [0, v[1]], color=colour, linewidth=2)
            ax.plot(*v, "o", color=colour, markersize=6)
            ax.plot(*unit, "o", markerfacecolor=theme["surface"], markeredgecolor=colour, markersize=6)
            ax.annotate(f"{label} [{v[0]:.3g}, {v[1]:.3g}]", v, xytext=offset, ha=align,
                        textcoords="offset points", color=theme["ink"], fontsize=9)
        ax.set_xlim(0, 2.1)
        ax.set_ylim(0, 1.85)
        ax.set_aspect("equal")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

    save_light_dark(draw, PAGE, "plane", size=(6.4, 5.2))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    cleaned = clean(load()["text"])
    body = body_text(cleaned)
    print("raw characters:", len(RAW_DOCUMENT), "| clean characters:", len(cleaned))
    print("body words:", len(body.split()))
    print("by header:", split_by_header(cleaned))
    print("fixed 6/2:", split_fixed(body, 6, 2), "count formula:", fixed_chunk_count(15, 6, 2))
    print("recursive 6:", split_recursive(body, 6))
    print("chunks:", build_chunks())

    vectors = chunk_vectors()
    q = embed(QUERY)
    print("chunk vectors:", vectors, "norms:", np.linalg.norm(vectors, axis=1), sep="\n")
    print("query:", q, "norm:", np.linalg.norm(q))
    table = similarity_table(q, vectors)
    for name, values in table.items():
        print(name, values, "->", ranking(values, largest_first=name != "euclidean"))
    unit, unit_q = normalize(vectors), normalize(q)
    print("unit vectors:", unit, "unit query:", unit_q, sep="\n")
    for name, values in similarity_table(unit_q, unit).items():
        print("unit", name, values, "->", ranking(values, largest_first=name != "euclidean"))
    print("top-2:", top_k(q, vectors, 2))

    centroids, assignment = kmeans(unit, start=[0, 2])
    print("centroids:", centroids, "assignment:", assignment, sep="\n")
    for text in (QUERY, QUERY_2):
        qv = embed(text)
        print(f"'{text}' {qv}: exact {top_k(qv, vectors, 3)}")
        print("  centroid scores:", centroids @ normalize(qv))
        print("  ivf nprobe=1:", ivf_search(qv, unit, centroids, assignment, 1),
              "nprobe=2:", ivf_search(qv, unit, centroids, assignment, 2))
        print("  graph walk from c3:", greedy_walk(qv, unit, "c3"))
    print("ivf recall@1, nprobe=1:", ivf_recall([QUERY, QUERY_2], vectors, 1),
          "nprobe=2:", ivf_recall([QUERY, QUERY_2], vectors, 2))
    for name, value in scale_numbers().items():
        print(f"{name}: {value:,.4f}")
    make_figures()
