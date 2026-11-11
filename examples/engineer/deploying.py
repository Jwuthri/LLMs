"""Numbers and plots for docs/engineer/07-deploying.md.

Run: uv run python -m examples.engineer.deploying

Each speed, price and memory budget in this file is an assumption for the example.
"""

import math

from examples.plotstyle import save_light_dark

PAGE = "deploying"
GB = 1e9  # bytes

# --- The sizing problem (assumed values) ----------------------------------------------------

PARAMS = 8e9          # parameters of the model
INPUT_TOKENS = 500    # tokens of one request (the prompt)
OUTPUT_TOKENS = 200   # tokens of one answer

# Shape of the key-value cache: the same as Llama 3 8B (32 layers, 8 key-value heads of size 128).
KV_BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2  # key and value, FP16

# Server GPU.
GPU_MEMORY_GB = 24.0
GPU_PRICE_USD_PER_HOUR = 2.0
PREFILL_TOKENS_PER_S = 5000.0   # speed to read the prompt
STEP_FIXED_MS = 19.8            # time of one decode step that does not depend on the batch
STEP_PER_REQUEST_MS = 0.2       # extra time of one decode step for each request in the batch

# Laptop.
LAPTOP_BUDGET_GB = 8.0
LAPTOP_PREFILL_TOKENS_PER_S = 250.0
LAPTOP_DECODE_TOKENS_PER_S = 20.0

# Phone or browser tab.
PHONE_BUDGET_GB = 3.0
RUNTIME_RESERVE_GB = 0.5        # key-value cache and runtime of a small device


# --- Memory -----------------------------------------------------------------------------------

def weights_gb(params: float, bits: int) -> float:
    """Memory of the weights: parameters x bits / 8."""
    return params * bits / 8 / GB


def kv_cache_gb(tokens: int, batch: int = 1) -> float:
    """Memory of the key-value cache for `batch` requests of `tokens` tokens each."""
    return KV_BYTES_PER_TOKEN * tokens * batch / GB


def memory_gb(bits: int, batch: int = 1) -> float:
    """Weights plus the key-value cache of the full requests (input + output tokens)."""
    return weights_gb(PARAMS, bits) + kv_cache_gb(INPUT_TOKENS + OUTPUT_TOKENS, batch)


def max_batch(memory_limit_gb: float, bits: int) -> int:
    """Largest batch for which the weights and the caches fit in the memory limit."""
    free = memory_limit_gb - weights_gb(PARAMS, bits)
    return math.floor(free / kv_cache_gb(INPUT_TOKENS + OUTPUT_TOKENS))


def largest_model(budget_gb: float, bits: int, reserve_gb: float = RUNTIME_RESERVE_GB) -> float:
    """Largest parameter count that fits: (budget - reserve) x 8 / bits."""
    return (budget_gb - reserve_gb) * GB * 8 / bits


def device_memory_gb(params: float, bits: int, reserve_gb: float = RUNTIME_RESERVE_GB) -> float:
    """Memory that a small device needs: weights plus a fixed reserve."""
    return weights_gb(params, bits) + reserve_gb


def fits(params: float, bits: int, budget_gb: float) -> bool:
    return device_memory_gb(params, bits) <= budget_gb


# --- Latency ----------------------------------------------------------------------------------

def latency_s(prefill_speed: float, decode_speed: float) -> dict[str, float]:
    """Time to first token (TTFT) and total time of one request."""
    prefill = INPUT_TOKENS / prefill_speed
    return {"prefill": prefill, "ttft": prefill + 1 / decode_speed,
            "total": prefill + OUTPUT_TOKENS / decode_speed}


def step_ms(batch: int) -> float:
    """Time of one decode step for a batch: each request gets 1 token."""
    return STEP_FIXED_MS + STEP_PER_REQUEST_MS * batch


def decode_speed(batch: int) -> float:
    """Tokens per second that ONE request of the batch gets."""
    return 1000 / step_ms(batch)


# --- Throughput and cost ----------------------------------------------------------------------

def requests_per_s(batch: int) -> float:
    """Continuous batching keeps `batch` requests in the GPU: batch / time of one request."""
    return batch / latency_s(PREFILL_TOKENS_PER_S, decode_speed(batch))["total"]


def tokens_per_hour(batch: int) -> float:
    """Input and output tokens that one GPU processes in one hour."""
    return requests_per_s(batch) * 3600 * (INPUT_TOKENS + OUTPUT_TOKENS)


def cost_per_million_tokens(batch: int, utilization: float = 1.0) -> float:
    """GPU price / millions of tokens for each hour. `utilization`: part of the hour with work."""
    return GPU_PRICE_USD_PER_HOUR / (tokens_per_hour(batch) * utilization / 1e6)


def gpu_hours(requests: float, batch: int) -> float:
    """GPU-hours to serve `requests` requests."""
    return requests / (requests_per_s(batch) * 3600)


# --- Plot ---------------------------------------------------------------------------------------

def make_figures() -> None:
    limit = max_batch(GPU_MEMORY_GB, 16)
    batches = list(range(1, limit + 1))
    marked = [1, 8, 32, limit]

    def draw(ax, theme):
        ax.plot(batches, [cost_per_million_tokens(b) for b in batches], color=theme["series"][0])
        ax.scatter(marked, [cost_per_million_tokens(b) for b in marked], color=theme["series"][0], zorder=3)
        for b in marked:
            ax.annotate(f"B = {b}: {cost_per_million_tokens(b):.2f} USD", (b, cost_per_million_tokens(b)),
                        textcoords="offset points", xytext=(6, 6), fontsize=8, color=theme["ink2"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
        ax.set_yticks([0.1, 0.3, 1, 3], ["0.10", "0.30", "1.00", "3.00"])
        ax.minorticks_off()
        ax.set_xlim(0.9, 260)
        ax.set_ylim(0.05, 6)
        ax.set_xlabel("Batch size B (requests in the GPU at the same time)")
        ax.set_ylabel("Cost for 1 million tokens (USD)")

    save_light_dark(draw, PAGE, "cost-vs-batch")


if __name__ == "__main__":
    print("weights GB: FP16", weights_gb(PARAMS, 16), "| 8-bit", weights_gb(PARAMS, 8),
          "| 4-bit", weights_gb(PARAMS, 4))
    print("KV cache bytes per token:", KV_BYTES_PER_TOKEN,
          "| one request (700 tokens) GB:", round(kv_cache_gb(700), 4))
    print("memory GB, 1 request: FP16", round(memory_gb(16), 3), "| 4-bit", round(memory_gb(4), 3))
    print("memory GB, FP16, batch 32:", round(memory_gb(16, 32), 3), "| KV only:", round(kv_cache_gb(700, 32), 3))
    print("max batch on the 24 GB GPU at FP16:", max_batch(GPU_MEMORY_GB, 16))

    laptop = latency_s(LAPTOP_PREFILL_TOKENS_PER_S, LAPTOP_DECODE_TOKENS_PER_S)
    print("laptop:", {k: round(v, 3) for k, v in laptop.items()})
    for batch in (1, 32):
        server = latency_s(PREFILL_TOKENS_PER_S, decode_speed(batch))
        print(f"server B = {batch}: step {step_ms(batch):.1f} ms, {decode_speed(batch):.2f} tokens/s per request,",
              {k: round(v, 3) for k, v in server.items()})
        print(f"  requests/s {requests_per_s(batch):.3f} | output tokens/s {requests_per_s(batch) * OUTPUT_TOKENS:.1f}"
              f" | tokens/hour {tokens_per_hour(batch):,.0f} | USD per 1M tokens {cost_per_million_tokens(batch):.3f}")
    print("demo queue: 10 requests one by one, the last one ends after",
          round(10 * latency_s(PREFILL_TOKENS_PER_S, decode_speed(1))["total"], 1), "s")
    print("requests per minute at B = 1:", round(requests_per_s(1) * 60, 1))
    print("cost ratio B = 1 / B = 32:", round(cost_per_million_tokens(1) / cost_per_million_tokens(32), 1))
    print("USD per 1M tokens at B = 32 and 30% utilization:", round(cost_per_million_tokens(32, 0.3), 3))
    hours = gpu_hours(1e6, 32)
    print("1M requests per day: GPU-hours", round(hours, 1), "| GPUs", round(hours / 24, 2),
          "->", math.ceil(hours / 24), "| USD per day for 2 GPUs:", 2 * 24 * GPU_PRICE_USD_PER_HOUR)

    print("largest model at 4 bits: laptop", largest_model(LAPTOP_BUDGET_GB, 4) / 1e9,
          "B | phone", largest_model(PHONE_BUDGET_GB, 4) / 1e9, "B")
    for params, bits in ((8e9, 16), (8e9, 4), (3e9, 4), (1e9, 4)):
        print(f"{params / 1e9:.0f}B at {bits} bits: weights {weights_gb(params, bits)} GB,"
              f" with reserve {device_memory_gb(params, bits)} GB,"
              f" phone {fits(params, bits, PHONE_BUDGET_GB)}, laptop {fits(params, bits, LAPTOP_BUDGET_GB)}")
    make_figures()
