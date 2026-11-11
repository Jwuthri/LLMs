"""Each number shown in docs/engineer/04-advanced-rag.md."""

import sqlite3

import pytest

from examples.engineer.advanced_rag import (
    ASSUMED_FILTER, ASSUMED_SQL, CALCULATOR_QUESTION, EXAMPLES, MEALS, PROMPTS, SQL_QUESTION, answer_with_tool,
    calculator, context_tokens, exact_match, filtered_search, keyword_list, mmr_scores, prompt_results,
    prompt_scores, rerank, rrf, run_sql, standard_error, tool_scores, vector_list,
)
from examples.engineer.corpus import CHUNKS, cosine, embed
from examples.engineer.rag import QUESTION, order


def test_text_to_sql():
    assert len(MEALS) == 3
    assert ASSUMED_SQL == "SELECT SUM(grams) FROM meals WHERE food = 'fish';"
    assert run_sql(ASSUMED_SQL) == [(140,)]
    assert run_sql("SELECT grams FROM meals WHERE food = 'fish'") == [(60,), (80,)]


def test_sql_tool_is_read_only():
    with pytest.raises(sqlite3.OperationalError):
        run_sql("DELETE FROM meals")


def test_metadata_filter():
    assert ASSUMED_FILTER == {"section": "Food"}
    result = filtered_search(QUESTION, ASSUMED_FILTER)
    assert [(chunk_id, round(score, 3)) for chunk_id, score in result] == [("c3", 0.956)]
    assert len(filtered_search(QUESTION, {"section": "Habits"})) == 2


def test_tool_choice():
    def rounded(question):
        return {tool: round(score, 3) for tool, score in tool_scores(question).items()}

    assert rounded(QUESTION) == {"search": 1.582, "sql": 0.593, "calculator": 0}
    assert rounded(SQL_QUESTION) == {"search": 1.582, "sql": 2.327, "calculator": 0}
    assert rounded(CALCULATOR_QUESTION) == {"search": 0, "sql": 0, "calculator": 1.173}
    assert answer_with_tool(QUESTION) == ("search", ["c2", "c3"])
    assert answer_with_tool(SQL_QUESTION) == ("sql", 140)
    assert answer_with_tool(CALCULATOR_QUESTION) == ("calculator", 140)
    assert calculator("Add 60 and 80.") == 140


def test_rerank_with_cross_encoder():
    assert vector_list() == ["c2", "c3", "c1"]
    assert rerank(["c2", "c3"]) == ["c3", "c2"]
    assert context_tokens(["c2", "c3"]) == 9
    assert context_tokens(rerank(["c2", "c3"])[:1]) == 4


def test_reciprocal_rank_fusion():
    assert vector_list() == ["c2", "c3", "c1"]
    assert keyword_list() == ["c3", "c1", "c2"]
    assert [round(1 / (60 + rank), 5) for rank in (1, 2, 3)] == [0.01639, 0.01613, 0.01587]
    fused = rrf([vector_list(), keyword_list()], k=60)
    assert {key: round(value, 5) for key, value in fused.items()} == {"c1": 0.03200, "c2": 0.03227, "c3": 0.03252}
    assert order(fused) == ["c3", "c2", "c1"]


def test_diversity_with_mmr():
    c1, c2, c3 = (embed(chunk.text) for chunk in CHUNKS)
    assert round(cosine(c1, c3), 3) == 0.684 and round(cosine(c2, c3), 3) == 0.977
    half = mmr_scores("c3", ["c1", "c2"], weight=0.5)
    assert {key: round(value, 3) for key, value in half.items()} == {"c1": -0.292, "c2": -0.338}
    assert order(half)[0] == "c1"
    more_relevance = mmr_scores("c3", ["c1", "c2"], weight=0.7)
    assert {key: round(value, 3) for key, value in more_relevance.items()} == {"c1": -0.135, "c2": -0.083}
    assert order(more_relevance)[0] == "c2"


def test_prompt_optimization():
    assert len(PROMPTS) * len(EXAMPLES) == 15  # LLM calls: candidates x examples
    assert exact_match("Fish.", "fish") == 1 and exact_match("The cat eats fish.", "fish") == 0
    assert prompt_results() == {"P1": [0, 1, 0, 0, 1], "P2": [0, 1, 1, 0, 1], "P3": [1, 1, 1, 1, 0]}
    assert prompt_scores() == {"P1": 0.4, "P2": 0.6, "P3": 0.8}
    assert order(prompt_scores())[0] == "P3"
    assert round(standard_error(0.8, 5), 3) == 0.179
