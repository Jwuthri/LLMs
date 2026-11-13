"""Numbers and plots for content/engineer/04-advanced-rag.md.

Run: uv run python -m examples.engineer.advanced_rag
"""

import math
import re
import sqlite3
import textwrap

import numpy as np

from examples.engineer.corpus import CHUNKS, cosine, embed, get_chunk, rank_by_cosine, tokenize
from examples.engineer.rag import QUESTION, bm25_totals, count_tokens, order, retrieve
from examples.plotstyle import save_figure

PAGE = "advanced-rag"

# --- 1. Query construction -------------------------------------------------------------------

MEALS = [("Monday", "fish", 60), ("Tuesday", "fish", 80), ("Wednesday", "chicken", 50)]

SQL_QUESTION = "How many grams of fish did the cat eat?"
# The script does not call an LLM. This is the SQL that we assume the LLM writes for SQL_QUESTION.
ASSUMED_SQL = "SELECT SUM(grams) FROM meals WHERE food = 'fish';"
# The metadata filter that we assume the LLM writes for QUESTION ("What does the cat eat?").
ASSUMED_FILTER = {"section": "Food"}


def run_sql(sql: str) -> list[tuple]:
    """Run one SQL statement on the `meals` table. The connection is read-only for that statement."""
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE meals (day TEXT, food TEXT, grams INTEGER)")
    connection.executemany("INSERT INTO meals VALUES (?, ?, ?)", MEALS)
    connection.execute("PRAGMA query_only = ON")  # a write statement now raises an error
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def filtered_search(question: str, metadata_filter: dict[str, str]) -> list[tuple[str, float]]:
    """Keep the chunks whose metadata matches the filter, then rank them by cosine similarity."""
    kept = [c for c in CHUNKS if all(getattr(c, key) == value for key, value in metadata_filter.items())]
    return rank_by_cosine(embed(question), kept)


# --- 2. Tools --------------------------------------------------------------------------------

TOOLS = {
    "search": "Find a text about the cat in the documents.",
    "sql": "Count or sum the grams of food in the meals table.",
    "calculator": "Add or multiply two numbers.",
}
CALCULATOR_QUESTION = "Add 60 and 80."


def tool_scores(question: str) -> dict[str, float]:
    """Tool choice as retrieval: the BM25 score of each tool description for the question."""
    return bm25_totals(question, TOOLS)


def calculator(question: str) -> int:
    """Toy calculator tool: add the whole numbers of the question."""
    return sum(int(number) for number in re.findall(r"\d+", question))


def answer_with_tool(question: str) -> tuple[str, object]:
    """Choose the tool with the best score, run it, and return (tool name, result)."""
    tool = order(tool_scores(question))[0]
    if tool == "search":
        return tool, [chunk_id for chunk_id, _ in retrieve(question)]
    if tool == "sql":
        return tool, run_sql(ASSUMED_SQL)[0][0]
    return tool, calculator(question)


# --- 3. Post-processing ----------------------------------------------------------------------

# Scores that we assume a cross-encoder gives to (QUESTION, chunk). A real model reads both texts.
CROSS_ENCODER_SCORES = {"c1": 0.10, "c2": 0.30, "c3": 0.95}


def vector_list(question: str = QUESTION) -> list[str]:
    """Chunk ids ranked by cosine similarity."""
    return [chunk_id for chunk_id, _ in rank_by_cosine(embed(question))]


def keyword_list(question: str = QUESTION) -> list[str]:
    """Chunk ids ranked by BM25."""
    return order(bm25_totals(question))


def rerank(chunk_ids: list[str], scores: dict[str, float] = CROSS_ENCODER_SCORES) -> list[str]:
    """Sort the retrieved chunks by the cross-encoder score."""
    return sorted(chunk_ids, key=lambda chunk_id: -scores[chunk_id])


def rrf(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion: the sum of 1 / (k + rank) over each list that holds the item."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return scores


def mmr_scores(selected: str, candidates: list[str], weight: float,
               relevance: dict[str, float] = CROSS_ENCODER_SCORES) -> dict[str, float]:
    """Maximal marginal relevance: weight x relevance - (1 - weight) x similarity to the selected chunk."""
    selected_vector = embed(get_chunk(selected).text)
    return {
        c: weight * relevance[c] - (1 - weight) * cosine(embed(get_chunk(c).text), selected_vector)
        for c in candidates
    }


def context_tokens(chunk_ids: list[str]) -> int:
    return sum(count_tokens(get_chunk(c).text) for c in chunk_ids)


# --- 4. Program LLMs: a prompt is a parameter ---------------------------------------------------

EXAMPLES = [
    ("What does the cat eat?", "fish"),
    ("Where did the cat sit?", "on the mat"),
    ("How long does the cat sleep?", "all day"),
    ("What sat on the mat?", "the cat"),
    ("Does the cat eat fish?", "yes"),
]

PROMPTS = {
    "P1": "Answer the question.",
    "P2": "Answer the question. Use only the context.",
    "P3": "Answer in a few words. Use only the context.",
}

# The answers that we assume the LLM gives with each prompt, one per example.
ASSUMED_OUTPUTS = {
    "P1": ["The cat eats fish and meat.", "On the mat.", "About 13 hours.", "The cat sat on the mat.", "Yes."],
    "P2": ["The cat eats fish.", "On the mat.", "All day.", "The cat sat on the mat.", "Yes."],
    "P3": ["Fish.", "On the mat.", "All day.", "The cat.", "Yes, it does."],
}


def exact_match(answer: str, gold: str) -> int:
    """1 if the answer has the same words as the gold answer (case and punctuation ignored)."""
    return int(tokenize(answer) == tokenize(gold))


def prompt_results() -> dict[str, list[int]]:
    """For each prompt: 1 or 0 for each example."""
    golds = [gold for _, gold in EXAMPLES]
    return {
        name: [exact_match(answer, gold) for answer, gold in zip(outputs, golds)]
        for name, outputs in ASSUMED_OUTPUTS.items()
    }


def prompt_scores() -> dict[str, float]:
    """The metric of each prompt: the share of examples with an exact match."""
    return {name: sum(results) / len(results) for name, results in prompt_results().items()}


def standard_error(score: float, n: int) -> float:
    """Standard error of a share measured on n examples."""
    return math.sqrt(score * (1 - score) / n)


# --- Plot ------------------------------------------------------------------------------------

def make_figures() -> None:
    scores = prompt_scores()

    def draw(ax, theme):
        names = list(scores)
        bars = ax.bar(names, [scores[n] for n in names], 0.5, color=theme["series"][0])
        for rect, name in zip(bars, names):
            correct = sum(prompt_results()[name])
            ax.text(rect.get_x() + rect.get_width() / 2, scores[name] + 0.02,
                    f"{scores[name]:.1f} ({correct} of 5)",
                    ha="center", va="bottom", fontsize=9, color=theme["ink2"])
        ax.set_xticks(range(len(names)), [f"{n}\n" + textwrap.fill(PROMPTS[n], 24) for n in names], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Exact match on 5 examples")
        ax.grid(axis="x", visible=False)

    save_figure(draw, PAGE, "prompts")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("sql:", ASSUMED_SQL, "->", run_sql(ASSUMED_SQL))
    print("all rows:", run_sql("SELECT * FROM meals"))
    print("no filter:", rank_by_cosine(embed(QUESTION)))
    print("filter", ASSUMED_FILTER, "->", filtered_search(QUESTION, ASSUMED_FILTER))

    for question in (QUESTION, SQL_QUESTION, CALCULATOR_QUESTION):
        scores = {tool: round(score, 3) for tool, score in tool_scores(question).items()}
        print(f"'{question}' tool scores {scores} ->", answer_with_tool(question))

    vector, keyword = vector_list(), keyword_list()
    print("vector list:", vector, "| keyword list:", keyword)
    print("cross-encoder order:", rerank(vector[:2]), "| context tokens:", context_tokens(vector[:2]),
          "->", context_tokens(rerank(vector[:2])[:1]))
    fused = rrf([vector, keyword])
    print("rrf:", {c: round(s, 5) for c, s in fused.items()}, "->", order(fused))
    print("1/61, 1/62, 1/63:", round(1 / 61, 5), round(1 / 62, 5), round(1 / 63, 5))
    for weight in (0.5, 0.7):
        scores = mmr_scores("c3", ["c1", "c2"], weight)
        print(f"mmr weight={weight}:", {c: round(s, 3) for c, s in scores.items()}, "->", order(scores)[0])
    print("cosine to c3: c1", round(cosine(embed(CHUNKS[0].text), embed(CHUNKS[2].text)), 3),
          "c2", round(cosine(embed(CHUNKS[1].text), embed(CHUNKS[2].text)), 3))

    print("prompt results:", prompt_results())
    print("prompt scores:", prompt_scores(), "-> best", order(prompt_scores())[0])
    print("standard error of 0.8 on 5 examples:", round(standard_error(0.8, 5), 3))
    make_figures()
