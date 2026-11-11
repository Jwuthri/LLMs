"""Each number shown in content/engineer/03-rag.md."""

import numpy as np

from examples.engineer.corpus import CHUNKS, embed, rank_by_cosine
from examples.engineer.rag import (
    ANSWER_CLAIMS, ASSUMED_SUMMARY, CHAT, FOLLOW_UP, HYDE_ANSWER, NEW_QUESTION, QUESTION, answer_relevancy,
    bm25, bm25_totals, build_prompt, context_precision, context_recall, count_tokens, faithfulness,
    hybrid, keywords, memory_similarities, memory_tokens, order, retrieve, rewrite,
)
from examples.engineer.vector_storage import body_text, clean, fixed_chunk_count, load, split_fixed


def rounded(ranked):
    return [(chunk_id, round(score, 3)) for chunk_id, score in ranked]


def test_retrieve():
    assert np.allclose(embed(QUESTION), [0.6, 1.2])
    assert rounded(rank_by_cosine(embed(QUESTION))) == [("c2", 0.997), ("c3", 0.956), ("c1", 0.868)]
    assert [chunk_id for chunk_id, _ in retrieve(QUESTION, k=2)] == ["c2", "c3"]


def test_prompt_string():
    assert build_prompt(QUESTION, ["c2", "c3"]) == (
        "Use only the context to answer the question.\n"
        "\n"
        "Context:\n"
        "[1] The cat sleeps all day.\n"
        "[2] The cat eats fish.\n"
        "\n"
        "Question: What does the cat eat?\n"
        "Answer:"
    )


def test_chunking_widget_start_values():
    body = body_text(clean(load()["text"]))
    assert len(body.split()) == 15
    chunks = split_fixed(body, size=6, overlap=2)
    assert len(chunks) == fixed_chunk_count(15, 6, 2) == 4
    assert sum(len(chunk.split()) for chunk in chunks) == 21  # words in the store, with the overlap


def test_query_rewriting():
    assert rewrite(FOLLOW_UP) == "where does the cat sit"
    assert np.allclose(embed(rewrite(FOLLOW_UP)), [1.0, 0.4])
    assert rounded(rank_by_cosine(embed(rewrite(FOLLOW_UP))))[0] == ("c1", 0.979)


def test_hyde():
    assert np.allclose(embed(HYDE_ANSWER), [0.25, 1.75])
    assert rounded(rank_by_cosine(embed(HYDE_ANSWER))) == [("c3", 1.0), ("c2", 0.972), ("c1", 0.667)]
    assert round(rank_by_cosine(embed(HYDE_ANSWER))[0][1], 4) == 0.9997


def test_multi_vector_storage():
    assert len(CHUNKS) == 3
    assert sum(count_tokens(chunk.text) for chunk in CHUNKS) == 15


def test_bm25():
    assert keywords("The cat eats fish.") == ["the", "cat", "eat", "fish"]
    assert "eat" in keywords(QUESTION)
    scores = bm25(QUESTION)
    assert round(scores["c3"]["eat"], 3) == 1.068
    assert round(np.log(1 + 2.5 / 1.5), 3) == 0.981  # IDF of `eat`: in 1 of 3 chunks
    assert round(2.2 / 2.02, 3) == 1.089  # term-frequency part for c3 (4 words, mean 5)
    assert round(np.log(1 + 0.5 / 3.5), 3) == 0.134  # IDF of `the` and `cat`: in 3 of 3 chunks
    assert round(scores["c3"]["the"], 3) == round(scores["c3"]["cat"], 3) == 0.145
    totals = bm25_totals(QUESTION)
    assert {key: round(value, 3) for key, value in totals.items()} == {"c1": 0.297, "c2": 0.267, "c3": 1.359}
    assert order(totals) == ["c3", "c1", "c2"]


def test_hybrid_search():
    scores = hybrid(QUESTION, alpha=0.5)
    assert {key: round(row["vector"], 3) for key, row in scores.items()} == {"c1": 0.0, "c2": 1.0, "c3": 0.682}
    assert {key: round(row["keyword"], 3) for key, row in scores.items()} == {"c1": 0.028, "c2": 0.0, "c3": 1.0}
    assert {key: round(row["hybrid"], 3) for key, row in scores.items()} == {"c1": 0.014, "c2": 0.5, "c3": 0.841}
    assert order({key: row["hybrid"] for key, row in scores.items()}) == ["c3", "c2", "c1"]


def test_memory_tokens():
    assert [count_tokens(f"{q} {a}") for q, a in CHAT] == [9, 10]
    assert count_tokens(NEW_QUESTION) == 4
    assert count_tokens(ASSUMED_SUMMARY) == 7
    assert memory_tokens() == {"buffer": 19, "window": 10, "summary": 7, "vector": 9}
    assert [round(s, 3) for s in memory_similarities()] == [0.984, 0.631]


def test_retrieval_metrics():
    assert context_precision(["c2"]) == 0.0 and context_recall(["c2"]) == 0.0
    assert context_precision(["c2", "c3"]) == 0.5 and context_recall(["c2", "c3"]) == 1.0
    assert context_precision(["c3", "c2"]) == 1.0 and context_recall(["c3", "c2"]) == 1.0


def test_generation_metrics():
    claims = [item["claim"] for item in ANSWER_CLAIMS]
    assert faithfulness(claims, ["c2", "c3"]) == 1.0
    assert round(faithfulness(claims + ["The cat drinks milk."], ["c2", "c3"]), 3) == 0.667
    assert answer_relevancy([item["relevant"] for item in ANSWER_CLAIMS]) == 0.5
