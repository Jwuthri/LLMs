"""Numbers and plots for docs/engineer/03-rag.md.

Run: uv run python -m examples.engineer.rag
"""

import math

import numpy as np

from examples.engineer.corpus import CHUNK_IDS, CHUNKS, cosine, embed, get_chunk, rank_by_cosine, tokenize
from examples.plotstyle import save_light_dark

PAGE = "rag"

# --- 1. The RAG loop -------------------------------------------------------------------------

QUESTION = "What does the cat eat?"
TOP_K = 2

PROMPT_TEMPLATE = (
    "Use only the context to answer the question.\n"
    "\n"
    "Context:\n"
    "{context}\n"
    "\n"
    "Question: {question}\n"
    "Answer:"
)

# The script does not call an LLM. This is the answer that we assume the LLM gives for the prompt.
ASSUMED_ANSWER = "The cat eats fish."


def retrieve(question: str, k: int = TOP_K) -> list[tuple[str, float]]:
    """Embed the question, compare it with each chunk, keep the k best (cosine similarity)."""
    return rank_by_cosine(embed(question))[:k]


def build_prompt(question: str, chunk_ids: list[str]) -> str:
    """Put the retrieved chunks and the question into the prompt template."""
    context = "\n".join(f"[{i}] {get_chunk(c).text}" for i, c in enumerate(chunk_ids, start=1))
    return PROMPT_TEMPLATE.format(context=context, question=question)


# --- 2. Retrievers ---------------------------------------------------------------------------

FOLLOW_UP = "Where does it sit?"
HYDE_ANSWER = "The cat eats mice."  # the draft answer that we assume the LLM writes for QUESTION


def rewrite(question: str, subject: str = "the cat") -> str:
    """Toy query rewriter: replace the pronoun `it` with the subject of the chat."""
    return " ".join(subject if word == "it" else word for word in tokenize(question))


def stem(word: str) -> str:
    """Toy stemmer: remove a final `s` (eats -> eat, sleeps -> sleep)."""
    return word.removesuffix("s")


def keywords(text: str) -> list[str]:
    return [stem(word) for word in tokenize(text)]


def bm25(query: str, texts: dict[str, str] | None = None, k1: float = 1.2,
         b: float = 0.75) -> dict[str, dict[str, float]]:
    """BM25 score of each text for the query, one value per query term that is in the text.

    `texts` is {id: text}. The default is the 3 chunks of the corpus.
    """
    if texts is None:
        texts = {chunk.id: chunk.text for chunk in CHUNKS}
    docs = {doc_id: keywords(text) for doc_id, text in texts.items()}
    n_docs = len(docs)
    mean_length = sum(len(words) for words in docs.values()) / n_docs
    scores: dict[str, dict[str, float]] = {}
    for doc_id, words in docs.items():
        scores[doc_id] = {}
        for term in dict.fromkeys(keywords(query)):  # each query term one time, in order
            frequency = words.count(term)
            if frequency == 0:
                continue
            containing = sum(term in other for other in docs.values())
            idf = math.log(1 + (n_docs - containing + 0.5) / (containing + 0.5))
            length_norm = 1 - b + b * len(words) / mean_length
            scores[doc_id][term] = idf * frequency * (k1 + 1) / (frequency + k1 * length_norm)
    return scores


def bm25_totals(query: str, texts: dict[str, str] | None = None) -> dict[str, float]:
    return {doc_id: sum(terms.values()) for doc_id, terms in bm25(query, texts).items()}


def min_max(scores: dict[str, float]) -> dict[str, float]:
    """Scale the scores so that the lowest is 0 and the highest is 1."""
    low, high = min(scores.values()), max(scores.values())
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def hybrid(query: str, alpha: float = 0.5) -> dict[str, dict[str, float]]:
    """alpha x vector score + (1 - alpha) x keyword score, each scaled to 0..1 first."""
    vector = min_max(dict(rank_by_cosine(embed(query))))
    keyword = min_max(bm25_totals(query))
    return {
        chunk_id: {
            "vector": vector[chunk_id],
            "keyword": keyword[chunk_id],
            "hybrid": alpha * vector[chunk_id] + (1 - alpha) * keyword[chunk_id],
        }
        for chunk_id in CHUNK_IDS
    }


def order(scores: dict[str, float]) -> list[str]:
    """Ids, best score first."""
    return sorted(scores, key=lambda key: -scores[key])


# --- 3. Memory -------------------------------------------------------------------------------

CHAT = [
    ("What does the cat eat?", "The cat eats fish."),
    ("Where does it sit?", "The cat sat on the mat."),
]
NEW_QUESTION = "Does it eat mice?"
# The summary that we assume a small LLM writes for the 2 turns of CHAT.
ASSUMED_SUMMARY = "Cat eats fish, sat on the mat."


def count_tokens(text: str) -> int:
    """One word is one token in the toy model."""
    return len(tokenize(text))


def memory_tokens() -> dict[str, int]:
    """History tokens that each memory type adds to the prompt of the third question."""
    turns = [f"{question} {answer}" for question, answer in CHAT]
    new_vector = embed(NEW_QUESTION)
    most_similar = max(turns, key=lambda turn: cosine(new_vector, embed(turn)))
    return {
        "buffer": sum(count_tokens(turn) for turn in turns),
        "window": count_tokens(turns[-1]),
        "summary": count_tokens(ASSUMED_SUMMARY),
        "vector": count_tokens(most_similar),
    }


def memory_similarities() -> list[float]:
    """Cosine similarity between the new question and each past turn."""
    new_vector = embed(NEW_QUESTION)
    return [cosine(new_vector, embed(f"{question} {answer}")) for question, answer in CHAT]


# --- 4. Evaluation ---------------------------------------------------------------------------

RELEVANT = {"c3"}  # the chunks that a person marked as relevant for QUESTION
REFERENCE_CLAIMS = ["The cat eats fish."]  # the claims of the correct answer
# A weaker answer that we assume, split into claims. `relevant` is the label of a judge.
ANSWER_CLAIMS = [
    {"claim": "The cat eats fish.", "relevant": True},
    {"claim": "The cat sleeps all day.", "relevant": False},
]


def context_precision(retrieved: list[str], relevant: set[str] = RELEVANT) -> float:
    """Mean of precision@k over each rank k that holds a relevant chunk."""
    hits, total = 0, 0.0
    for k, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def supported(claim: str, context: list[str]) -> bool:
    """Toy judge: a claim is supported if one chunk of the context says the same words."""
    return any(tokenize(claim) == tokenize(get_chunk(c).text) for c in context)


def context_recall(retrieved: list[str], reference_claims: list[str] = REFERENCE_CLAIMS) -> float:
    """Claims of the correct answer that the context supports / claims of the correct answer."""
    return sum(supported(claim, retrieved) for claim in reference_claims) / len(reference_claims)


def faithfulness(claims: list[str], context: list[str]) -> float:
    """Claims of the answer that the context supports / claims of the answer."""
    return sum(supported(claim, context) for claim in claims) / len(claims)


def answer_relevancy(labels: list[bool]) -> float:
    """Statements of the answer that address the question / statements of the answer."""
    return sum(labels) / len(labels)


# --- Plot ------------------------------------------------------------------------------------

def make_figures() -> None:
    scores = hybrid(QUESTION)

    def draw(ax, theme):
        width = 0.26
        labels = {"vector": "Vector (cosine)", "keyword": "Keyword (BM25)", "hybrid": "Hybrid, alpha = 0.5"}
        for i, (name, label) in enumerate(labels.items()):
            values = [scores[chunk_id][name] for chunk_id in CHUNK_IDS]
            xs = np.arange(len(CHUNK_IDS)) + (i - 1) * (width + 0.02)
            bars = ax.bar(xs, values, width, color=theme["series"][i], label=label)
            for rect, value in zip(bars, values):
                ax.text(rect.get_x() + rect.get_width() / 2, value + 0.015, f"{value:.2f}",
                        ha="center", va="bottom", fontsize=8, color=theme["ink2"])
        ax.set_xticks(np.arange(len(CHUNK_IDS)), [f"{c.id}: {c.text}" for c in CHUNKS], fontsize=8)
        ax.set_ylim(0, 1.25)
        ax.set_ylabel("Score, scaled to 0-1")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", ncols=3, fontsize=8)

    save_light_dark(draw, PAGE, "hybrid")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("question vector:", embed(QUESTION))
    print("ranking:", rank_by_cosine(embed(QUESTION)))
    retrieved = [chunk_id for chunk_id, _ in retrieve(QUESTION)]
    print("retrieved:", retrieved)
    print("--- prompt ---", build_prompt(QUESTION, retrieved), "--- end ---", sep="\n")
    print("prompt tokens:", count_tokens(build_prompt(QUESTION, retrieved)))

    rewritten = rewrite(FOLLOW_UP)
    print("follow-up:", embed(FOLLOW_UP), rank_by_cosine(embed(FOLLOW_UP)))
    print("rewritten:", rewritten, embed(rewritten), rank_by_cosine(embed(rewritten)))
    print("HyDE:", embed(HYDE_ANSWER), rank_by_cosine(embed(HYDE_ANSWER)))
    print("vectors, one per chunk:", len(CHUNKS), "| one per token:", sum(count_tokens(c.text) for c in CHUNKS))

    print("keywords:", keywords(QUESTION), {c.id: keywords(c.text) for c in CHUNKS})
    for chunk_id, terms in bm25(QUESTION).items():
        print("bm25", chunk_id, {t: round(s, 4) for t, s in terms.items()}, "total",
              round(sum(terms.values()), 4))
    for chunk_id, row in hybrid(QUESTION).items():
        print("hybrid", chunk_id, {k: round(v, 4) for k, v in row.items()})
    print("hybrid order:", order({c: row["hybrid"] for c, row in hybrid(QUESTION).items()}))

    print("memory tokens:", memory_tokens(), "new question tokens:", count_tokens(NEW_QUESTION))
    print("memory similarities:", memory_similarities())

    contexts = {"k=1": ["c2"], "k=2": ["c2", "c3"], "k=3": ["c2", "c3", "c1"], "hybrid k=2": ["c3", "c2"]}
    for label, context in contexts.items():
        print(label, "context precision:", context_precision(context),
              "context recall:", context_recall(context))
    claims = [item["claim"] for item in ANSWER_CLAIMS]
    print("faithfulness:", faithfulness(claims, ["c2", "c3"]))
    print("faithfulness with a milk claim:", faithfulness(claims + ["The cat drinks milk."], ["c2", "c3"]))
    print("answer relevancy:", answer_relevancy([item["relevant"] for item in ANSWER_CLAIMS]))
    make_figures()
