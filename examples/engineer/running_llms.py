"""Numbers and plots for content/engineer/01-running-llms.md.

Run: uv run python -m examples.engineer.running_llms
"""

import json
from math import ceil

import numpy as np

from examples.plotstyle import save_figure
from examples.toy import LOGITS, VOCAB, softmax

PAGE = "running-llms"


# --- LLM APIs: the chat "messages" shape and the cost of one request -----------------------

REQUEST = {
    "model": "example-model",
    "messages": [
        {"role": "system", "content": "Complete the sentence with one word."},
        {"role": "user", "content": "the cat"},
    ],
    "max_tokens": 5,
    "temperature": 0,
}

def approx_tokens(text: str) -> int:
    """Rule of thumb for English: 1 token is about 4 characters."""
    return ceil(len(text) / 4)


RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": "sat"}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": sum(approx_tokens(m["content"]) for m in REQUEST["messages"]),
        "completion_tokens": approx_tokens("sat"),
    },
}

# A larger request for the cost example (assumed sizes).
INPUT_TOKENS = 1000
OUTPUT_TOKENS = 200
REQUESTS_PER_DAY = 10_000

# ASSUMED prices for the example, in dollars for each million tokens. They are not the prices of a real provider.
PRICE_INPUT_PER_MILLION = 3.00
PRICE_OUTPUT_PER_MILLION = 15.00


def request_cost(input_tokens: int, output_tokens: int,
                 price_input: float = PRICE_INPUT_PER_MILLION,
                 price_output: float = PRICE_OUTPUT_PER_MILLION) -> dict:
    """Cost in dollars: tokens x price for each million tokens / 1,000,000."""
    cost_input = input_tokens * price_input / 1e6
    cost_output = output_tokens * price_output / 1e6
    return {"input": cost_input, "output": cost_output, "total": cost_input + cost_output}


# --- Open-source LLMs: memory to run a model on your machine -------------------------------

PARAMS = 8_000_000_000

# Llama 3 8B (see Sources on the page): blocks, context, and the size of the key and value vectors
# of one token in one block (8 key-value heads x 128 numbers).
LAYERS = 32
CONTEXT = 8192
KV_SIZE = 8 * 128


def weights_gb(params: int, bits_per_weight: float) -> float:
    return params * bits_per_weight / 8 / 1e9


def kv_cache_gb(layers: int, context: int, kv_size: int, bytes_per_number: int = 2) -> float:
    """Key-value cache: 2 vectors (key, value) for each token of the context in each block."""
    return 2 * layers * context * kv_size * bytes_per_number / 1e9


# --- Prompt engineering: 4 prompts for the same task ---------------------------------------

QUESTION = "A cat sleeps 14 hours each day. How many hours does it sleep in 3 days?"

PROMPTS = {
    "zero-shot": f"Q: {QUESTION}\nA:",
    "few-shot": (
        "Q: A dog eats 2 meals each day. How many meals does it eat in 5 days?\nA: 10\n"
        "Q: A bird sings 6 songs each hour. How many songs does it sing in 4 hours?\nA: 24\n"
        f"Q: {QUESTION}\nA:"
    ),
    "chain of thought": f"Q: {QUESTION}\nA: Let's think step by step.",
    "ReAct": (
        "Answer the question. You can use one tool: calculator[expression].\n"
        "Use this format:\nThought: what you must do\nAction: calculator[expression]\n"
        "Observation: the result of the tool\nAnswer: the final answer\n\n"
        f"Question: {QUESTION}\n"
    ),
}


def calculator(expression: str) -> int:
    """The tool of the ReAct example: multiply two whole numbers written as 'a * b'."""
    left, right = expression.split("*")
    return int(left) * int(right)


# --- Structuring outputs: constrained decoding ---------------------------------------------

ALLOWED = ["the", "cat"]  # the grammar allows only these tokens at this step


def constrained_probs(logits: np.ndarray, vocab: list[str], allowed: list[str]) -> dict:
    """Set the logit of each token that the grammar does not allow to -infinity, then softmax."""
    mask = np.array([token in allowed for token in vocab])
    masked = np.where(mask, logits, -np.inf)
    return {"mask": mask, "masked": masked, "exp": np.where(mask, np.exp(logits), 0.0),
            "probs": softmax(masked)}


SCHEMA_OUTPUTS = ['{"word": "the"}', '{"word": "cat"}']  # all outputs that the toy JSON schema allows


def free_steps(valid: list[str], chosen: str) -> list[int]:
    """Positions of `chosen` where the grammar allows more than one next character."""
    positions = []
    for i in range(len(chosen)):
        same_prefix = [s for s in valid if s[:i] == chosen[:i]]
        if len({s[i] for s in same_prefix}) > 1:
            positions.append(i)
    return positions


# --- Plot ----------------------------------------------------------------------------------

def make_figures() -> None:
    before = softmax(LOGITS)
    after = constrained_probs(LOGITS, VOCAB, ALLOWED)["probs"]

    def draw(ax, theme):
        width = 0.36
        for i, (probs, label) in enumerate(((before, "No constraint"), (after, "Grammar: only the, cat"))):
            xs = np.arange(len(VOCAB)) + (i - 0.5) * (width + 0.03)
            bars = ax.bar(xs, probs, width, color=theme["series"][i], label=label)
            for rect, p in zip(bars, probs):
                ax.text(rect.get_x() + rect.get_width() / 2, p + 0.015, f"{p:.3f}",
                        ha="center", va="bottom", fontsize=8, color=theme["ink2"])
        ax.set_xticks(np.arange(len(VOCAB)), VOCAB)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability of the next token")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")

    save_figure(draw, PAGE, "constrained")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print(json.dumps(REQUEST, indent=2))
    print(json.dumps(RESPONSE, indent=2))
    cost = request_cost(INPUT_TOKENS, OUTPUT_TOKENS)
    print("cost of one request:", cost, "per day:", REQUESTS_PER_DAY * cost["total"],
          "per 30 days:", 30 * REQUESTS_PER_DAY * cost["total"])

    cache = kv_cache_gb(LAYERS, CONTEXT, KV_SIZE)
    print("kv cache GB:", cache, "at 2,048 tokens:", kv_cache_gb(LAYERS, 2048, KV_SIZE))
    for name, bits in (("FP16", 16), ("4-bit", 4)):
        print(f"{name}: weights {weights_gb(PARAMS, bits)} GB, total {weights_gb(PARAMS, bits) + cache:.3f} GB")

    for name, prompt in PROMPTS.items():
        tokens = approx_tokens(prompt)
        print(f"{name}: {len(prompt)} characters, about {tokens} tokens,",
              f"input cost for {REQUESTS_PER_DAY} calls ${REQUESTS_PER_DAY * request_cost(tokens, 0)['total']:.3f}")
    print("calculator[14 * 3] =", calculator("14 * 3"))

    c = constrained_probs(LOGITS, VOCAB, ALLOWED)
    print("probs:", softmax(LOGITS), "masked logits:", c["masked"], "exp:", c["exp"], "sum:", c["exp"].sum())
    print("constrained probs:", c["probs"], "renormalized:", softmax(LOGITS)[:2] / softmax(LOGITS)[:2].sum())
    for output in SCHEMA_OUTPUTS:
        free = free_steps(SCHEMA_OUTPUTS, output)
        print(output, "length", len(output), "free steps at", free, "forced steps", len(output) - len(free))
    make_figures()
