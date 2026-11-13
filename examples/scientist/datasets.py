"""Numbers and plots for content/scientist/03-datasets.md.

Run: uv run python -m examples.scientist.datasets

No model is called. Each "generated" text in this file is written by hand and plays the role of a
model output.
"""

import itertools
import json

import numpy as np

from examples.plotstyle import save_figure

PAGE = "datasets"

# --- Storage formats and chat templates ----------------------------------------------------

SHAREGPT = {"conversations": [
    {"from": "system", "value": "You are a concise assistant."},
    {"from": "human", "value": "Count the words: the cat sat"},
    {"from": "gpt", "value": "3"},
]}
MESSAGES = {"messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Count the words: the cat sat"},
    {"role": "assistant", "content": "3"},
]}
SHAREGPT_ROLES = {"system": "system", "human": "user", "gpt": "assistant"}
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def sharegpt_to_messages(sample: dict) -> dict:
    return {"messages": [{"role": SHAREGPT_ROLES[turn["from"]], "content": turn["value"]}
                         for turn in sample["conversations"]]}


def chatml_pieces(message: dict) -> list[str]:
    """The 6 pieces of one ChatML message, in order."""
    return [IM_START, message["role"], "\n", message["content"], IM_END, "\n"]


def render_chatml(messages: list[dict], add_generation_prompt: bool = False) -> str:
    text = "".join("".join(chatml_pieces(message)) for message in messages)
    return text + (f"{IM_START}assistant\n" if add_generation_prompt else "")


ALPACA_HEADER = ("Below is an instruction that describes a task. "
                 "Write a response that appropriately completes the request.")


def render_alpaca(instruction: str, output: str) -> str:
    return f"{ALPACA_HEADER}\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"


# --- Synthetic data generation (the model outputs are written by hand) ---------------------

SEED_TASKS = ["count the words in a text", "reverse the order of the words in a text", "find the longest word in a text"]
PERSONAS = ["a beginner", "a senior developer"]
STYLES = ["a short question", "a detailed specification"]
GENERATOR_SYSTEM_PROMPT = ("You write training data for a Python tutor. "
                           "Return only JSON with the keys \"instruction\" and \"answer\".")


def generator_prompt(seed: str, persona: str = PERSONAS[0], style: str = STYLES[0]) -> list[dict]:
    """The request that a pipeline sends to the generator model for one seed task."""
    user = f"Seed task: {seed}.\nWrite the instruction as {style} from {persona}. Then write the answer."
    return [{"role": "system", "content": GENERATOR_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def all_prompts() -> list[list[dict]]:
    return [generator_prompt(*combo) for combo in itertools.product(SEED_TASKS, PERSONAS, STYLES)]


# Hand-written. In a real pipeline, these 3 strings come back from the generator model.
GENERATED = [
    json.dumps({"instruction": "Write a Python function count_words(text) that returns the number of words in text.",
                "answer": "def count_words(text):\n    return len(text.split())"}),
    json.dumps({"instruction": "Write a Python function reverse_words(text) that returns the words of text in reverse order.",
                "answer": "def reverse_words(text):\n    return \" \".join(reversed(text.split()))"}),
    json.dumps({"instruction": "Write a Python function longest_word(text) that returns the longest word of text.",
                "answer": "def longest_word(text):\n    return max(text.split(), key=len)"}),
]
BAD_OUTPUT = "Sure! Here is the JSON: " + GENERATED[0]  # a frequent failure: text around the JSON


def parse_sample(raw: str) -> dict | None:
    """Keep an output only if it is valid JSON with exactly the two keys, both not empty."""
    try:
        sample = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(sample, dict) or set(sample) != {"instruction", "answer"} or not all(sample.values()):
        return None
    return sample


def to_messages(sample: dict) -> dict:
    return {"messages": [{"role": "user", "content": sample["instruction"]},
                         {"role": "assistant", "content": sample["answer"]}]}


# --- Data enhancement: verified outputs and rejection sampling -----------------------------

# One list of (arguments, expected result) for each function.
UNIT_TESTS = {
    "count_words": [(("the cat sat",), 3), (("",), 0)],
    "reverse_words": [(("the cat sat",), "sat cat the")],
    "longest_word": [(("the cats sat",), "cats")],
}
FUNCTION_NAMES = list(UNIT_TESTS)


def run_tests(code: str, name: str, tests: list[tuple]) -> list[bool]:
    """Define the function from its source text, then run each test. An error counts as a failure."""
    namespace: dict = {}
    try:
        exec(code, namespace)  # the source text is a constant of this file, never a model output
    except Exception:
        return [False] * len(tests)
    results = []
    for arguments, expected in tests:
        try:
            results.append(namespace[name](*arguments) == expected)
        except Exception:
            results.append(False)
    return results


# Hand-written. In a real pipeline, these are 4 answers of the model to the same instruction.
CANDIDATES = {
    "A": "def count_words(text):\n    return len(text)",
    "B": "def count_words(text):\n    return text.count(\" \") + 1",
    "C": "def count_words(text):\n    return len(text.split(\" \"))",
    "D": "def count_words(text):\n    return len(text.split())",
}


def rejection_sampling(candidates: dict[str, str], tests: list[tuple]) -> dict[str, list[bool]]:
    return {label: run_tests(code, "count_words", tests) for label, code in candidates.items()}


def chance_of_one_good(p: float, n: int) -> float:
    """Probability that at least 1 of n independent answers is correct."""
    return 1 - (1 - p) ** n


# --- Quality filtering ---------------------------------------------------------------------

# Hand-written pool of (instruction, answer) pairs, as a generator could return them.
POOL = [
    ("count the words in the text", "def count_words(text):\n    return len(text.split())"),
    ("count the words in this text", "def count_words(text):\n    return len(text.split())"),
    ("reverse the words in the text", "def reverse_words(text):\n    return \" \".join(reversed(text.split()))"),
    ("find the longest word", "def longest_word(text):\n    return max(len(word) for word in text.split())"),
    ("count the letters in the text", "I'm sorry, but I cannot help with that."),
    ("sort the words", "ok"),
]
MIN_ANSWER_CHARS = 10  # assumed limit
REFUSALS = ("i'm sorry", "i cannot", "as an ai")
JACCARD_LIMIT = 0.8  # assumed limit
JUDGE_SCORES = {"count the words in the text": 5, "reverse the words in the text": 5, "find the longest word": 2}
JUDGE_LIMIT = 4  # assumed limit; the scores above are written by hand, on a scale from 1 to 5


def long_enough(answer: str) -> bool:
    return len(answer) >= MIN_ANSWER_CHARS


def is_refusal(answer: str) -> bool:
    return any(pattern in answer.lower() for pattern in REFUSALS)


def jaccard(a: str, b: str) -> float:
    """Size of the intersection / size of the union of the two word sets."""
    set_a, set_b = set(a.split()), set(b.split())
    return len(set_a & set_b) / len(set_a | set_b)


def drop_near_duplicates(samples: list[tuple[str, str]], limit: float) -> list[tuple[str, str]]:
    """Keep a sample only if its instruction is not too similar to an instruction already kept."""
    kept: list[tuple[str, str]] = []
    for sample in samples:
        if all(jaccard(sample[0], other[0]) < limit for other in kept):
            kept.append(sample)
    return kept


def filter_pool() -> dict[str, list[tuple[str, str]]]:
    """Apply the 4 filters in order. Returns the pool after each filter."""
    after_length = [s for s in POOL if long_enough(s[1])]
    after_refusal = [s for s in after_length if not is_refusal(s[1])]
    after_duplicates = drop_near_duplicates(after_refusal, JACCARD_LIMIT)
    after_judge = [s for s in after_duplicates if JUDGE_SCORES[s[0]] >= JUDGE_LIMIT]
    return {"start": POOL, "length": after_length, "refusal": after_refusal,
            "duplicates": after_duplicates, "judge": after_judge}


A, B, C, D = (POOL[i][0] for i in range(4))
WORDS = sorted(set(" ".join([A, B, C, D]).split()))  # the 10 different words of the 4 instructions


def minhash_ranks(k: int, seed: int = 0) -> np.ndarray:
    """k random orders of the words. ranks[i, j] = position of word j in order i."""
    rng = np.random.default_rng(seed)
    return np.array([rng.permutation(len(WORDS)) for _ in range(k)])


def minhash_signature(text: str, ranks: np.ndarray) -> np.ndarray:
    """For each order, the smallest rank among the words of the text."""
    columns = [WORDS.index(word) for word in set(text.split())]
    return ranks[:, columns].min(axis=1)


def minhash_estimate(a: str, b: str, k: int, seed: int = 0) -> float:
    """Part of the k signature positions that are equal: an estimate of the Jaccard similarity."""
    ranks = minhash_ranks(k, seed)
    return float(np.mean(minhash_signature(a, ranks) == minhash_signature(b, ranks)))


def standard_error(j: float, k: int) -> float:
    return float(np.sqrt(j * (1 - j) / k))


# --- Plot ----------------------------------------------------------------------------------

def make_figures() -> None:
    sizes = np.arange(1, 257)
    ranks = minhash_ranks(256)
    pairs = [("samples 1 and 2", A, B), ("samples 1 and 3", A, C)]

    def draw(ax, theme):
        for i, (label, first, second) in enumerate(pairs):
            equal = minhash_signature(first, ranks) == minhash_signature(second, ranks)
            ax.plot(sizes, np.cumsum(equal) / sizes, color=theme["series"][i], label=f"MinHash estimate, {label}")
            ax.axhline(jaccard(first, second), color=theme["series"][i], linestyle="--", linewidth=1,
                       label=f"True Jaccard, {label}: {jaccard(first, second):.3f}")
        ax.set_xlabel("Number of hash functions k")
        ax.set_ylabel("Similarity")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=8)

    save_figure(draw, PAGE, "minhash")


if __name__ == "__main__":
    print("sharegpt -> messages equal:", sharegpt_to_messages(SHAREGPT) == MESSAGES)
    for message in MESSAGES["messages"]:
        pieces = chatml_pieces(message)
        print([len(piece) for piece in pieces], "=", sum(map(len, pieces)), pieces)
    rendered = render_chatml(MESSAGES["messages"])
    print(repr(rendered), len(rendered), "characters;", sum(len(m["content"]) for m in MESSAGES["messages"]), "of content")
    print(repr(render_chatml(MESSAGES["messages"][:2], add_generation_prompt=True)))
    print(repr(render_alpaca("Count the words: the cat sat", "3")))

    print("prompts:", len(all_prompts()), "=", len(SEED_TASKS), "x", len(PERSONAS), "x", len(STYLES))
    print(generator_prompt(SEED_TASKS[0]))
    samples = [parse_sample(raw) for raw in GENERATED]
    print("valid outputs:", sum(s is not None for s in samples), "of", len(GENERATED), "| bad output:", parse_sample(BAD_OUTPUT))
    print(repr(render_chatml(to_messages(samples[0])["messages"])))

    for name, sample in zip(FUNCTION_NAMES, samples):
        print("verified", name, run_tests(sample["answer"], name, UNIT_TESTS[name]))
    print("rejection sampling, 2 tests:", rejection_sampling(CANDIDATES, UNIT_TESTS["count_words"]))
    print("rejection sampling, 1 test:", rejection_sampling(CANDIDATES, UNIT_TESTS["count_words"][:1]))
    print("chance of one good answer:", {n: round(chance_of_one_good(0.25, n), 3) for n in (1, 4, 10)})

    for name, pool in filter_pool().items():
        print(name, len(pool), [s[0] for s in pool])
    print("wrong answer passes the rules but fails the unit test:",
          run_tests(POOL[3][1], "longest_word", UNIT_TESTS["longest_word"]))
    print("jaccard A-B:", jaccard(A, B), "A-C:", jaccard(A, C), "A-D:", jaccard(A, D))
    for other in (B, C, D):
        print("shared words:", len(set(A.split()) & set(other.split())), "all words:", len(set(A.split()) | set(other.split())))
    print("words:", WORDS)
    ranks = minhash_ranks(4)
    print("ranks (4 orders):", ranks, sep="\n")
    for label, text in (("A", A), ("B", B), ("C", C), ("D", D)):
        print("signature", label, minhash_signature(text, ranks))
    for k in (4, 16, 128, 256):
        print(f"k={k}: A-B {minhash_estimate(A, B, k):.3f}, A-C {minhash_estimate(A, C, k):.3f}, "
              f"A-D {minhash_estimate(A, D, k):.3f}, standard error A-B {standard_error(jaccard(A, B), k):.3f}")
    make_figures()
