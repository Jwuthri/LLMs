"""The shared corpus of the 3 retrieval pages (content/engineer/02, 03 and 04).

Three chunks about the cat, and a 2-dimensional word table that extends the toy embeddings
(`the`, `cat`, `sat` come from examples/toy.py; the other vectors are fixed by hand).
The toy embedding model is: lower-case the text, look up each word, take the mean of the vectors.
"""

import re
from dataclasses import dataclass

import numpy as np

from examples.toy import VOCAB, E

# One 2-dimensional vector per word. The first 3 rows are the toy embeddings.
WORD_VECTORS: dict[str, np.ndarray] = {
    **{word: E[i] for i, word in enumerate(VOCAB)},  # the = [1, 0], cat = [0, 1], sat = [1, 1]
    # words of the 3 chunks
    "on": np.array([1.0, 0.0]),
    "mat": np.array([2.0, 2.0]),
    "sleeps": np.array([0.0, 2.0]),
    "all": np.array([1.0, 0.0]),
    "day": np.array([0.0, 2.0]),
    "eats": np.array([0.0, 3.0]),
    "fish": np.array([0.0, 2.0]),
    # words of the questions
    "what": np.array([1.0, 2.0]),
    "does": np.array([1.0, 0.0]),
    "eat": np.array([0.0, 3.0]),
    "where": np.array([2.0, 0.0]),
    "sit": np.array([1.0, 1.0]),
    "it": np.array([0.0, 1.0]),
    "mice": np.array([0.0, 3.0]),
}


@dataclass(frozen=True)
class Chunk:
    """One piece of a document, as it is stored in the vector store."""

    id: str
    text: str
    section: str  # metadata: the header of the source file that the chunk comes from
    source: str = "cat.md"


# The 3 chunks. content/engineer/02-vector-storage.md builds them from one raw file.
CHUNKS = [
    Chunk("c1", "The cat sat on the mat.", "Habits"),
    Chunk("c2", "The cat sleeps all day.", "Habits"),
    Chunk("c3", "The cat eats fish.", "Food"),
]
CHUNK_IDS = [chunk.id for chunk in CHUNKS]


def tokenize(text: str) -> list[str]:
    """Lower-case words, without punctuation. One word is one token in the toy model."""
    return re.findall(r"[a-z]+", text.lower())


def embed(text: str) -> np.ndarray:
    """Toy embedding model: the mean of the word vectors (mean pooling). Unknown word: KeyError."""
    return np.mean([WORD_VECTORS[word] for word in tokenize(text)], axis=0)


def normalize(v: np.ndarray) -> np.ndarray:
    """Scale each vector to length 1."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between 2 vectors."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def chunk_vectors() -> np.ndarray:
    """One row per chunk: the embedding of its text (not normalized)."""
    return np.array([embed(chunk.text) for chunk in CHUNKS])


def rank_by_cosine(query_vector: np.ndarray, chunks: list[Chunk] = CHUNKS) -> list[tuple[str, float]]:
    """(chunk id, cosine similarity) for each chunk, best first."""
    scored = [(chunk.id, cosine(query_vector, embed(chunk.text))) for chunk in chunks]
    return sorted(scored, key=lambda item: -item[1])


def get_chunk(chunk_id: str) -> Chunk:
    return CHUNKS[CHUNK_IDS.index(chunk_id)]
