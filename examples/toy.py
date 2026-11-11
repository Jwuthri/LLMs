"""The shared toy example. Every page reuses these numbers (see docs/toy-example.md)."""

import numpy as np

VOCAB = ["the", "cat", "sat"]

# One 2-dimensional embedding vector per token id.
E = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

# Scores of the model for the next token after "the cat", one per vocabulary entry.
LOGITS = np.array([0.1, 1.0, 2.0])

# A 2x2 weight matrix of one layer.
W = np.array([[0.5, -1.2], [0.8, 0.3]])


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Turn scores into probabilities that sum to 1."""
    scaled = z / temperature
    exps = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    return exps / exps.sum(axis=-1, keepdims=True)


def encode(text: str) -> list[int]:
    """Word-level tokenizer of the toy vocabulary."""
    return [VOCAB.index(word) for word in text.split()]
