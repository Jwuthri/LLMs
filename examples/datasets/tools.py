"""Numbers and plots for content/datasets/04-tools.md.

Run: uv run python -m examples.datasets.tools

The page follows the four steps of the "Tools" section of LLM Datasets: scrape, filter, generate, explore.
Each value below is an assumption of the page or a result calculated from those assumptions.
No real tool is called and nothing is measured.
"""

import math

import numpy as np

from examples.datasets.toy_dataset import SAMPLES, TOPICS, Sample, embeddings
from examples.plotstyle import save_figure

PAGE = "dataset-tools"

# --- 1. Scraping ---------------------------------------------------------------------------

# Assumed pages: (name, raw characters of the HTML file, characters of the main text after extraction).
PAGES = [
    ("Long article", 40_000, 12_000),
    ("Short article", 25_000, 5_000),
    ("Forum thread", 60_000, 3_000),
    ("Login page", 15_000, 300),
    ("Documentation page", 10_000, 4_700),
]
MIN_CLEAN_CHARACTERS = 500          # the rule: drop a page with less clean text than this
CHARACTERS_PER_TOKEN = 4            # rule of thumb for English text (Scientist lesson 1)
CRAWL_PAGES = 1_000_000             # assumed size of a real crawl


def page_yield(raw: int, clean: int) -> float:
    """Share of the raw characters that stay after extraction."""
    return clean / raw


def total_yield(pages=PAGES) -> float:
    return sum(clean for _, _, clean in pages) / sum(raw for _, raw, _ in pages)


def kept_pages(minimum: int = MIN_CLEAN_CHARACTERS) -> list[tuple[str, int, int]]:
    return [page for page in PAGES if page[2] >= minimum]


def crawl_estimate(pages: int = CRAWL_PAGES) -> dict[str, float]:
    """Tokens of a crawl, if its pages are like the 5 toy pages."""
    kept = kept_pages()
    kept_share = len(kept) / len(PAGES)
    characters_per_page = sum(clean for _, _, clean in kept) / len(kept)
    tokens_per_page = characters_per_page / CHARACTERS_PER_TOKEN
    return {
        "kept_share": kept_share,
        "kept_pages": pages * kept_share,
        "characters_per_page": characters_per_page,
        "tokens_per_page": tokens_per_page,
        "tokens": pages * kept_share * tokens_per_page,
    }


# --- 2. Filtering --------------------------------------------------------------------------

REFUSAL_WORDS = "I cannot"
# An assumed 9th sample for the false positive: a correct answer that quotes the same words.
QUOTE_SAMPLE = Sample(9, "chat", "What does the cat say in the story?", 'The cat says: "I cannot sleep."',
                      1, 16, True, (0.55, 0.6))
THRESHOLDS = (0.999, 0.99, 0.95)
COMPARISONS_PER_SECOND = 10_000_000  # assumed speed of one machine


def refusal_flags(samples=SAMPLES, words: str = REFUSAL_WORDS) -> list[int]:
    """Ids of the samples whose answer contains the refusal words."""
    return [sample.id for sample in samples if words.lower() in sample.answer.lower()]


def rule_precision() -> float:
    """Share of the flagged samples that are real refusals, when the quotation sample is in the data."""
    flagged = refusal_flags(SAMPLES + [QUOTE_SAMPLE])
    return sum(sample_id != QUOTE_SAMPLE.id for sample_id in flagged) / len(flagged)


def after_rules() -> list[Sample]:
    flagged = set(refusal_flags())
    return [sample for sample in SAMPLES if sample.id not in flagged]


def exact_duplicates(samples=SAMPLES) -> list[tuple[int, int]]:
    """Pairs of samples with the same instruction, character for character."""
    return [(a.id, b.id) for i, a in enumerate(samples) for b in samples[i + 1:] if a.instruction == b.instruction]


def cosine_matrix(vectors: np.ndarray | None = None) -> np.ndarray:
    vectors = embeddings() if vectors is None else vectors
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return unit @ unit.T


def pairs_above(threshold: float) -> list[tuple[int, int, float]]:
    """Pairs (id, id, cosine similarity) of the 8 samples at or above the threshold."""
    sim = cosine_matrix()
    n = len(SAMPLES)
    return [(SAMPLES[i].id, SAMPLES[j].id, float(sim[i, j]))
            for i in range(n) for j in range(i + 1, n) if sim[i, j] >= threshold]


def fuzzy_deduplicate(threshold: float, samples=SAMPLES) -> tuple[list[int], dict[int, int]]:
    """Read the samples in order. Keep a sample if no kept sample is at or above the threshold.

    Returns the kept ids and, for each removed id, the kept id that it duplicates.
    """
    vectors = np.array([sample.embedding for sample in samples])
    sim = cosine_matrix(vectors)
    kept: list[int] = []
    removed: dict[int, int] = {}
    for i, sample in enumerate(samples):
        match = next((j for j in kept if sim[i, j] >= threshold), None)
        if match is None:
            kept.append(i)
        else:
            removed[sample.id] = samples[match].id
    return [samples[i].id for i in kept], removed


def pair_count(n: int) -> int:
    """Comparisons to compare each pair of n samples one time."""
    return n * (n - 1) // 2


def after_deduplication(threshold: float = THRESHOLDS[0]) -> list[Sample]:
    survivors = after_rules()
    kept, _ = fuzzy_deduplicate(threshold, survivors)
    return [sample for sample in survivors if sample.id in kept]


# --- 3. Generation -------------------------------------------------------------------------

KEEP_RATE = 0.6                     # assumed share of the generated samples that pass the checks
SCALE_NEW_SAMPLES = 10_000          # assumed need of a real project
PROMPT_TOKENS = 300                 # assumed tokens of one generation prompt
OUTPUT_TOKENS = 500                 # assumed tokens of one generated sample
BATCH = 64
# One decode step for a batch of B requests: the model of Scientist lesson 10 (training libraries).
DECODE_BASE_MS = 19.8
DECODE_PER_REQUEST_MS = 0.2
PRICE_INPUT = 0.5                   # assumed price, dollars for each million input tokens
PRICE_OUTPUT = 2.0                  # assumed price, dollars for each million output tokens
BATCH_DISCOUNT = 0.5                # assumed discount of a batch API


def counts_by_topic(samples) -> dict[str, int]:
    return {topic: sum(sample.topic == topic for sample in samples) for topic in TOPICS}


def missing_by_topic(samples) -> dict[str, int]:
    """New samples that each topic needs to be as large as the largest topic."""
    counts = counts_by_topic(samples)
    target = max(counts.values())
    return {topic: target - count for topic, count in counts.items()}


def to_generate(needed: int, keep_rate: float = KEEP_RATE) -> int:
    return math.ceil(needed / keep_rate)


def decode_tokens_per_second(batch: int) -> float:
    return batch / ((DECODE_BASE_MS + DECODE_PER_REQUEST_MS * batch) / 1000)


def generation_plan(needed: int = SCALE_NEW_SAMPLES) -> dict[str, float]:
    generated = to_generate(needed)
    output_tokens = generated * OUTPUT_TOKENS
    input_tokens = generated * PROMPT_TOKENS
    dollars = (input_tokens * PRICE_INPUT + output_tokens * PRICE_OUTPUT) / 1e6
    return {
        "generated": generated,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "one_by_one_hours": output_tokens / decode_tokens_per_second(1) / 3600,
        "batched_tokens_per_second": decode_tokens_per_second(BATCH),
        "batched_minutes": output_tokens / decode_tokens_per_second(BATCH) / 60,
        "dollars": dollars,
        "dollars_per_kept_sample": dollars / needed,
        "dollars_with_batch_discount": dollars * (1 - BATCH_DISCOUNT),
        "output_tokens_per_kept_sample": OUTPUT_TOKENS / KEEP_RATE,
    }


# --- 4. Exploration ------------------------------------------------------------------------

START_CENTRES = (0, 4)              # 2-means starts at the embeddings of samples 1 and 5


def two_means(vectors: np.ndarray | None = None, start=START_CENTRES) -> dict:
    """Plain 2-means: give each point to its nearest centre, move each centre to the mean, repeat."""
    vectors = embeddings() if vectors is None else vectors
    centres = vectors[list(start)].copy()
    labels = np.full(len(vectors), -1)
    rounds = 0
    while True:
        distances = np.linalg.norm(vectors[:, None, :] - centres[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centres = np.array([vectors[labels == k].mean(axis=0) for k in range(len(centres))])
        rounds += 1
    own_distance = distances[np.arange(len(vectors)), labels]
    return {
        "clusters": [[SAMPLES[i].id for i in np.flatnonzero(labels == k)] for k in range(len(centres))],
        "centres": centres,
        "distances": own_distance,
        "outlier": SAMPLES[int(own_distance.argmax())].id,
        "rounds": rounds,
    }


def funnel() -> list[tuple[str, int, dict[str, int]]]:
    """Count of samples, and count for each topic, after each step of the lesson."""
    collected = list(SAMPLES)
    ruled = after_rules()
    deduplicated = after_deduplication()
    final = counts_by_topic(deduplicated)
    for topic, count in missing_by_topic(deduplicated).items():
        final[topic] += count
    return [
        ("collected", len(collected), counts_by_topic(collected)),
        ("after rules", len(ruled), counts_by_topic(ruled)),
        ("after deduplication", len(deduplicated), counts_by_topic(deduplicated)),
        ("after generation", sum(final.values()), final),
    ]


# --- Plot ----------------------------------------------------------------------------------

def make_figures() -> None:
    def draw(ax, theme):
        thresholds = np.round(np.arange(0.900, 1.0001, 0.001), 3)
        kept = [len(fuzzy_deduplicate(t)[0]) for t in thresholds]
        ax.step(thresholds, kept, where="post", color=theme["series"][0], label="Samples kept")
        marks = sorted(THRESHOLDS)
        ax.scatter(marks, [len(fuzzy_deduplicate(t)[0]) for t in marks], color=theme["series"][1], zorder=3,
                   label="Thresholds of the page")
        for t in marks:
            count = len(fuzzy_deduplicate(t)[0])
            ax.annotate(f"{t}: {count} kept", (t, count), textcoords="offset points", xytext=(-6, 8),
                        ha="right", fontsize=9, color=theme["ink2"])
        ax.set_xlabel("Similarity threshold (a pair at or above it is a duplicate)")
        ax.set_ylabel("Samples kept (of 8)")
        ax.set_ylim(0, 8.8)
        ax.set_yticks(range(0, 9, 2))
        ax.set_xlim(0.9, 1.002)
        ax.legend(loc="lower right")

    save_figure(draw, PAGE, "kept-against-threshold")


if __name__ == "__main__":
    for name, raw, clean in PAGES:
        print(f"{name}: raw {raw}, clean {clean}, yield {page_yield(raw, clean):.3f}")
    print("total raw:", sum(p[1] for p in PAGES), "| clean:", sum(p[2] for p in PAGES), f"| yield {total_yield():.3f}")
    print("kept pages:", [p[0] for p in kept_pages()], "| clean characters:", sum(p[2] for p in kept_pages()))
    print("crawl:", crawl_estimate())

    print("refusal flags:", refusal_flags(), "| with the quotation sample:", refusal_flags(SAMPLES + [QUOTE_SAMPLE]),
          "| precision:", rule_precision())
    print("after rules:", [s.id for s in after_rules()])
    print("exact duplicates:", exact_duplicates())
    print("cosine matrix:\n", np.round(cosine_matrix(), 3))
    for threshold in THRESHOLDS + (0.9,):
        kept, removed = fuzzy_deduplicate(threshold)
        pairs = [(a, b, round(s, 4)) for a, b, s in pairs_above(threshold)]
        print(f"threshold {threshold}: pairs {pairs} | kept {kept} ({len(kept)}) | removed {removed}")
    for n in (8, 10_000, 1_000_000):
        print(f"pairs of {n}: {pair_count(n)} | seconds: {pair_count(n) / COMPARISONS_PER_SECOND}",
              f"| hours: {pair_count(n) / COMPARISONS_PER_SECOND / 3600:.3f}")
    print("after deduplication:", [s.id for s in after_deduplication()], counts_by_topic(after_deduplication()))

    missing = missing_by_topic(after_deduplication())
    print("missing:", missing, "| k:", sum(missing.values()), "| to generate:", to_generate(sum(missing.values())),
          "| expected kept:", to_generate(sum(missing.values())) * KEEP_RATE)
    print("generation plan:", {k: round(v, 3) for k, v in generation_plan().items()})

    result = two_means()
    print("2-means:", result["clusters"], "| centres:", np.round(result["centres"], 3),
          "| distances:", np.round(result["distances"], 3), "| outlier:", result["outlier"], "| rounds:", result["rounds"])
    for step, count, topics in funnel():
        print(f"{step}: {count} {topics}", {t: round(c / count, 3) for t, c in topics.items()})
    make_figures()
