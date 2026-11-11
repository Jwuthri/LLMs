"""Numbers and plots for content/scientist/10-training-libraries.md.

Run: uv run python -m examples.scientist.training_libraries

Each value below is an assumption of the page or an estimate from those assumptions. Nothing is measured.
"""

from examples.plotstyle import save_figure

PAGE = "training-libraries"

# --- The job -------------------------------------------------------------------------------

PARAMS = 8_000_000_000
EXAMPLES = 10_000
TOKENS_PER_EXAMPLE = 500
GPU_GB = 24.0

# --- Assumptions for the memory estimate ---------------------------------------------------

USABLE_SHARE = 0.9                 # keep 10% of the GPU memory free
FOUR_BIT_EFFECTIVE_BITS = 5.5      # 4 bits + 1.5 bits for the parts that stay in 16 bits and the constants
STATE_BYTES_PER_PARAM = 16         # one trained parameter: weight + gradient + optimizer states (ZeRO count)
ACTIVATION_GB_PER_BILLION = 0.5    # micro-batch of 4 examples of 500 tokens, gradient checkpointing on

LORA_RANK = 16
BLOCKS = 32
# (inputs, outputs) of the 7 linear layers of one block: the sizes of Llama 3.1 8B.
LORA_LAYERS = {
    "q_proj": (4096, 4096),
    "k_proj": (4096, 1024),
    "v_proj": (4096, 1024),
    "o_proj": (4096, 4096),
    "gate_proj": (4096, 14336),
    "up_proj": (4096, 14336),
    "down_proj": (14336, 4096),
}

METHODS = ("full", "lora", "qlora")
METHOD_LABELS = {"full": "Full fine-tuning", "lora": "LoRA, 16-bit base", "qlora": "QLoRA, 4-bit base"}

# --- Assumptions for the steps and the speeds ----------------------------------------------

EPOCHS = 3
MICRO_BATCH = 4
ACCUMULATION = 4
TRAIN_TOKENS_PER_SECOND = 2_500    # forward and backward pass

# One decode step of the engine for a batch of B requests: the server GPU of the Deploying lesson.
DECODE_BASE_MS = 19.8
DECODE_PER_REQUEST_MS = 0.2

RL_PROMPTS = 16
RL_SAMPLES_PER_PROMPT = 8
RL_PROMPT_TOKENS = 200
RL_SAMPLE_TOKENS = 300
RL_STEPS = 500

EVAL_PROMPTS = 1_000
EVAL_OUTPUT_TOKENS = 200
EVAL_BATCH = 64
EVAL_RUNS = 5


# --- Memory --------------------------------------------------------------------------------

def gigabytes(parameters: float, bits: float) -> float:
    """Memory of `parameters` numbers stored with `bits` bits each."""
    return parameters * bits / 8 / 1e9


def lora_parameters_per_block(rank: int = LORA_RANK) -> int:
    """Trained numbers of one block: rank * (inputs + outputs) for each adapted layer."""
    return sum(rank * (d_in + d_out) for d_in, d_out in LORA_LAYERS.values())


def lora_parameters(rank: int = LORA_RANK, blocks: int = BLOCKS) -> int:
    return blocks * lora_parameters_per_block(rank)


def memory_parts(method: str, parameters: float = PARAMS) -> dict[str, float]:
    """Estimate of the GPU memory in GB: the weights, the training states and the activations."""
    activations = ACTIVATION_GB_PER_BILLION * parameters / 1e9
    if method == "full":
        # 16 bytes for each parameter: 2 for the 16-bit weight, 14 for its gradient and optimizer states.
        return {
            "weights": gigabytes(parameters, 16),
            "states": gigabytes(parameters, 8 * STATE_BYTES_PER_PARAM - 16),
            "activations": activations,
        }
    adapter = lora_parameters() * parameters / PARAMS  # the adapter grows with the model
    base_bits = 16 if method == "lora" else FOUR_BIT_EFFECTIVE_BITS
    return {
        "weights": gigabytes(parameters, base_bits),
        "states": gigabytes(adapter, 8 * STATE_BYTES_PER_PARAM),
        "activations": activations,
    }


def memory_total(method: str, parameters: float = PARAMS) -> float:
    return sum(memory_parts(method, parameters).values())


def usable_gb(gpu_gb: float = GPU_GB) -> float:
    return USABLE_SHARE * gpu_gb


def fits(method: str, gpu_gb: float = GPU_GB, parameters: float = PARAMS) -> bool:
    return memory_total(method, parameters) <= usable_gb(gpu_gb)


def full_memory_per_gpu(gpus: int, sharded: bool, parameters: float = PARAMS) -> float:
    """Full fine-tuning on several GPUs. Sharded (FSDP, ZeRO stage 3): each GPU holds 1 / gpus of the
    weights and the states. Not sharded: each GPU holds a full copy."""
    parts = memory_parts("full", parameters)
    held = parts["weights"] + parts["states"]
    return (held / gpus if sharded else held) + parts["activations"]


# --- Steps, tokens and time ----------------------------------------------------------------

def effective_batch() -> int:
    return MICRO_BATCH * ACCUMULATION


def optimizer_steps() -> float:
    return EXAMPLES * EPOCHS / effective_batch()


def sft_tokens() -> int:
    return EXAMPLES * TOKENS_PER_EXAMPLE * EPOCHS


def training_seconds(tokens: float) -> float:
    return tokens / TRAIN_TOKENS_PER_SECOND


def decode_step_ms(batch: int) -> float:
    """Time of one decode step when the engine decodes `batch` requests together."""
    return DECODE_BASE_MS + DECODE_PER_REQUEST_MS * batch


def decode_tokens_per_second(batch: int) -> float:
    """Output tokens for each second: each decode step gives one token to each request of the batch."""
    return batch / (decode_step_ms(batch) / 1000)


def generation_seconds(tokens: float, batch: int) -> float:
    return tokens / decode_tokens_per_second(batch)


def rl_step() -> dict[str, float]:
    """Tokens and seconds of one RL step, with generation one by one and in one batch."""
    samples = RL_PROMPTS * RL_SAMPLES_PER_PROMPT
    generated = samples * RL_SAMPLE_TOKENS
    trained = samples * (RL_PROMPT_TOKENS + RL_SAMPLE_TOKENS)
    train_s = training_seconds(trained)
    one_by_one_s = generation_seconds(generated, 1)
    batched_s = generation_seconds(generated, samples)
    return {
        "samples": samples,
        "generated_tokens": generated,
        "trained_tokens": trained,
        "train_s": train_s,
        "one_by_one_s": one_by_one_s,
        "batched_s": batched_s,
        "generation_speed_up": one_by_one_s / batched_s,
        "one_by_one_share": one_by_one_s / (one_by_one_s + train_s),
        "batched_share": batched_s / (batched_s + train_s),
        "step_one_by_one_s": one_by_one_s + train_s,
        "step_batched_s": batched_s + train_s,
    }


def evaluation() -> dict[str, float]:
    """Time to generate the answers of one evaluation run."""
    tokens = EVAL_PROMPTS * EVAL_OUTPUT_TOKENS
    one_by_one_s = generation_seconds(tokens, 1)
    batched_s = generation_seconds(tokens, EVAL_BATCH)
    return {
        "tokens": tokens,
        "one_by_one_s": one_by_one_s,
        "batched_tokens_per_s": decode_tokens_per_second(EVAL_BATCH),
        "batched_s": batched_s,
        "speed_up": one_by_one_s / batched_s,
    }


# --- Plot ----------------------------------------------------------------------------------

def make_figures() -> None:
    part_labels = {
        "weights": "Weights",
        "states": "Gradients and optimizer states",
        "activations": "Activations",
    }

    def draw(ax, theme):
        rows = list(reversed(METHODS))  # full fine-tuning at the top
        left = [0.0] * len(rows)
        for i, (part, label) in enumerate(part_labels.items()):
            values = [memory_parts(method)[part] for method in rows]
            ax.barh(range(len(rows)), values, left=left, height=0.55, color=theme["series"][i], label=label)
            left = [a + b for a, b in zip(left, values)]
        for y, total in enumerate(left):
            # A short bar gets its label on the right of the GPU line, so that the line does not cross the text.
            ax.text(max(total, GPU_GB) + 2, y, f"{total:.1f} GB", va="center", fontsize=9, color=theme["ink2"])
        ax.axvline(GPU_GB, color=theme["ink2"], linestyle="--", linewidth=1.2)
        ax.text(GPU_GB + 2, len(rows) - 0.45, f"GPU: {GPU_GB:.0f} GB", fontsize=9, color=theme["ink2"])
        ax.set_yticks(range(len(rows)), [METHOD_LABELS[method] for method in rows])
        ax.set_ylim(-0.5, len(rows) - 0.25)
        ax.set_xlim(0, 150)
        ax.set_xlabel("Estimated GPU memory (GB)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")

    save_figure(draw, PAGE, "memory")


if __name__ == "__main__":
    print("weights 16-bit GB:", gigabytes(PARAMS, 16), "| 4-bit GB:", gigabytes(PARAMS, 4),
          "| 4-bit with overhead GB:", gigabytes(PARAMS, FOUR_BIT_EFFECTIVE_BITS))
    print("LoRA parameters per block:", lora_parameters_per_block(), "| total:", lora_parameters(),
          f"| share: {100 * lora_parameters() / PARAMS:.3f}%",
          f"| adapter states GB: {gigabytes(lora_parameters(), 8 * STATE_BYTES_PER_PARAM):.3f}")
    for method in METHODS:
        parts = {name: round(value, 3) for name, value in memory_parts(method).items()}
        print(f"{method}: {parts} total {memory_total(method):.3f} GB, fits in {usable_gb():.1f} GB: {fits(method)}")
    print("full / 16-bit weights:", memory_parts("full")["weights"] + memory_parts("full")["states"],
          "GB =", (memory_parts("full")["weights"] + memory_parts("full")["states"]) / gigabytes(PARAMS, 16), "times")
    print("16-bit weights / GPU:", gigabytes(PARAMS, 16) / GPU_GB, "| full / GPU:", memory_total("full") / GPU_GB,
          "| free with LoRA:", usable_gb() - memory_total("lora"), "| free with QLoRA:", usable_gb() - memory_total("qlora"))
    print("decode step ms, batch 128:", decode_step_ms(RL_PROMPTS * RL_SAMPLES_PER_PROMPT), "| batch 64:", decode_step_ms(EVAL_BATCH))
    print("4 GPUs of 80 GB, usable", usable_gb(80), "GB | full copy on each:", full_memory_per_gpu(4, False),
          "| sharded:", full_memory_per_gpu(4, True))
    print("optimizer steps:", optimizer_steps(), "| SFT tokens:", sft_tokens(),
          "| SFT step tokens:", effective_batch() * TOKENS_PER_EXAMPLE,
          "| SFT step s:", training_seconds(effective_batch() * TOKENS_PER_EXAMPLE),
          "| SFT job s:", training_seconds(sft_tokens()), "= min:", training_seconds(sft_tokens()) / 60)
    print("DPO policy tokens:", 2 * sft_tokens())
    step = rl_step()
    print("RL step:", {name: round(value, 3) for name, value in step.items()})
    print("RL generated / SFT step tokens:", step["generated_tokens"] / (effective_batch() * TOKENS_PER_EXAMPLE))
    print(f"RL {RL_STEPS} steps: generated", RL_STEPS * step["generated_tokens"],
          "| batched h:", RL_STEPS * step["step_batched_s"] / 3600,
          "| one by one h:", RL_STEPS * step["step_one_by_one_s"] / 3600)
    result = evaluation()
    print("evaluation:", {name: round(value, 3) for name, value in result.items()},
          "| one by one min:", result["one_by_one_s"] / 60)
    print(f"{EVAL_RUNS} evaluations: one by one min:", EVAL_RUNS * result["one_by_one_s"] / 60,
          "| batched min:", EVAL_RUNS * result["batched_s"] / 60)
    make_figures()
