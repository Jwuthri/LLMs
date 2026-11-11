"""The toy instruction dataset of the LLM Datasets track. All four lessons reuse these 8 samples.

Each sample is one conversation about the toy sentence "the cat sat". All the values are assumed:
the texts are written by hand, and no model is called.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sample:
    id: int
    topic: str  # "math", "code" or "chat"
    instruction: str
    answer: str
    turns: int  # number of user messages in the conversation
    tokens: int  # length of the full conversation
    correct: bool  # the result of a checker (a solver, a unit test or a human)
    embedding: tuple[float, float]  # a 2-number embedding of the instruction


SAMPLES = [
    Sample(1, "math", "Count the words: the cat sat", "3", 1, 12, True, (1.0, 0.1)),
    Sample(2, "math", "Count the words: the cat", "2", 1, 10, True, (0.9, 0.2)),
    Sample(3, "math", "Count the words: the cat sat.", "3", 1, 12, True, (0.98, 0.12)),
    Sample(4, "math", "Count the letters: cat", "4", 1, 10, False, (0.8, 0.4)),
    Sample(5, "code", "Write Python that counts the words of a text", "len(text.split())", 1, 24, True, (0.1, 1.0)),
    Sample(6, "code", "This function counts words wrong. Fix it, then add a test", "(two answers)", 2, 60, True, (0.2, 0.9)),
    Sample(7, "chat", "Explain why the cat sat, then make it shorter, then translate it", "(three answers)", 3, 90, True, (0.6, 0.7)),
    Sample(8, "chat", "Tell me about the cat", "I cannot help with that.", 1, 14, False, (0.5, 0.5)),
]

TOPICS = ["math", "code", "chat"]


def embeddings() -> np.ndarray:
    return np.array([sample.embedding for sample in SAMPLES])


def topic_counts() -> dict[str, int]:
    return {topic: sum(sample.topic == topic for sample in SAMPLES) for topic in TOPICS}
