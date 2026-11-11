# 10. Training libraries

## Core idea

All the training libraries do the same loop: read a batch, calculate a loss, move the weights.
They differ in three things: how much they do for you, how much memory and speed they give on your hardware, and how easy it is to change the loop.
Choose with three questions: does the job fit in your memory, what data do you have, and how much of the loop must you change.

## The example

This page uses one job.

| Item | Value |
|---|---|
| Model | 8 billion parameters |
| Data | 10,000 examples, about 500 tokens each |
| Hardware | One GPU with 24 GB of memory |
| Training (assumed) | 3 epochs, micro-batch of 4, gradient accumulation of 4 |
| Speed (assumed) | Training: 2,500 tokens/s. One decode step for B requests: 19.8 ms + 0.2 ms × B |

The decode speed is the server GPU of the [Deploying](../engineer/07-deploying.md) lesson.
All the numbers are estimates from stated assumptions. They are not measurements.
This project did not run these libraries: each statement about a library comes from its documentation or its repository.

A **training library** is code that runs the training loop for you. A **trainer** is the class or the command that you call.

## 1. Supervised fine-tuning (SFT) libraries

### Setup

The 10,000 examples have reference answers, so the method is [supervised fine-tuning](04-fine-tuning.md). That lesson explains full fine-tuning, LoRA and QLoRA.
Which of the three fits in 24 GB? The answer decides which libraries you can use.

### Steps

1. **Calculate the memory of the weights.** At 16 bits, 8 billion parameters need 16 GB. At 4 bits they need 4 GB. Assume 1.5 bits more for each parameter, for the parts that are not stored in 4 bits: 5.5 GB.

    **Why:** the weights must be in GPU memory before the first step. At 16 bits they take two thirds of the GPU.

2. **Add the training states of full fine-tuning.** Assume 16 bytes for each trained parameter: 2 for the weight, 14 for its gradient and its optimizer states. This is the count of [lesson 4](04-fine-tuning.md).

    $$8 \times 10^9 \times 16 \text{ bytes} = 128 \text{ GB} = 8 \times \text{the 16-bit weights}$$

    **Why:** the optimizer keeps numbers for each weight that it moves. This multiple, not the size of the model file, decides if full fine-tuning fits.

3. **Count the LoRA parameters.** Assume rank $r = 16$ on the 7 linear layers of each of the 32 blocks, with the layer sizes of Llama 3.1 8B. One layer adds $r \times (d_{in} + d_{out})$ numbers. One block adds 1,310,720.

    $$32 \times 1{,}310{,}720 = 41{,}943{,}040 \text{ parameters} = 0.524\% \text{ of the model}$$

    At 16 bytes each, the adapter and its training states need 0.671 GB.

    **Why:** LoRA cuts the training states from 112 GB to less than 1 GB. The frozen weights become the largest part.

4. **Add the activations and compare with the GPU.** Assume 4 GB of activations for a micro-batch of 4. Assume that 10% of the GPU stays free: the limit is 21.6 GB.

    | Method | Weights | Training states | Total with activations | Fits? |
    |---|---|---|---|---|
    | Full fine-tuning | 16 GB | 112 GB | 132 GB | No |
    | LoRA, 16-bit base | 16 GB | 0.7 GB | 20.7 GB | Yes, 0.9 GB free |
    | QLoRA, 4-bit base | 5.5 GB | 0.7 GB | 10.2 GB | Yes, 11.4 GB free |

    **Why:** one long example can use a margin of 0.9 GB. QLoRA leaves room for longer texts or a larger batch.

5. **Count the optimizer steps and the tokens.** The effective batch size is 4 × 4 = 16.

    $$\text{steps} = \frac{10{,}000 \times 3}{16} = 1{,}875 \qquad \text{tokens} = 10{,}000 \times 500 \times 3 = 15{,}000{,}000$$

    At 2,500 tokens/s, the job takes 6,000 s: 100 minutes.

    **Why:** steps and tokens give the time before you start. A job of 100 minutes does not need a cluster tool.

<figure class="plot">
<img src="/figures/training-libraries/memory.svg" alt="Estimated GPU memory. Full fine-tuning: 132.0 GB. LoRA: 20.7 GB. QLoRA: 10.2 GB. The GPU has 24 GB.">
<figcaption>Estimates, not measurements. The gradients and the optimizer states make full fine-tuning 5.5 times larger than the GPU.</figcaption>
</figure>

<div class="llm-widget" data-widget="finetune-memory"></div>

<div class="key-point">

Memory chooses the method, and the method chooses the library. For this job, full fine-tuning does not fit (132 GB) and QLoRA fits with room (10.2 GB). The libraries below can all do QLoRA. They differ in the work that you do and in the hardware that they support.

</div>

### From toy to real

The LlamaFactory README has its own estimates for 7 billion parameters: 6 GB for 4-bit QLoRA, 16 GB for LoRA, and 60 to 120 GB for full fine-tuning. Our estimates are in the same range, and higher for QLoRA.

- **Transformers `Trainer` with PEFT and bitsandbytes.** The layer below TRL: `Trainer` is a complete training and evaluation loop, PEFT adds the LoRA adapter, bitsandbytes stores the base model in 4 bits. Take it for full control. Limit: you write the most code.
- **TRL `SFTTrainer`.** A light wrapper around `Trainer` that takes the LoRA and the 4-bit configurations as arguments. Take it as the default in Python. Limit: on several GPUs it puts one full model on each GPU, and its documentation points to DeepSpeed when that does not fit.
- **Unsloth.** A fine-tuning tool with its own optimized kernels. You use it in Python (it works with `SFTTrainer`), in a web UI or in a desktop app. Its README claims 2 times the speed and 70% less memory. Take it for one GPU with little memory. Limits: its documentation describes multi-GPU training as a manual set-up with Accelerate or DeepSpeed. The web UI has an AGPL-3.0 licence (the core package: Apache-2.0).
- **Axolotl.** One YAML file describes the data, the training and the evaluation. Take it for runs that you repeat and for several GPUs: its multi-GPU guide covers DeepSpeed and FSDP. Limit: the README asks for an NVIDIA or AMD GPU.
- **LlamaFactory.** YAML files with a command line, or a web UI. Take it when the team does not write Python. Limits: its last release is from May 2026, and its README points to a different project for GRPO.
- **MLX-LM.** Fine-tuning on Apple silicon: LoRA, DoRA or full, and QLoRA when the base model is quantized. Take it on a Mac. Limits: Apple silicon only, and a GGUF export for only a few model families.

## 2. Preference tuning and reinforcement learning (RL) libraries

### Setup

[Lesson 5](05-preference-alignment.md) explains DPO, PPO and GRPO. Your data chooses between them:

- You have reference answers: use SFT.
- You have pairs of a good and a bad answer: use DPO.
- You have only a **checker** (a function that gives a score to an answer) or a simulator: use RL. RL calls the checker the **reward function**.

In RL, the model writes its own answers at each step. Assume 16 prompts for each step, 8 samples for each prompt (the default of TRL), 200 prompt tokens and 300 generated tokens for each sample.

### Steps

1. **Count the tokens that one RL step generates.**

    $$16 \times 8 \times 300 = 38{,}400 \text{ tokens}$$

    One SFT step reads 16 × 500 = 8,000 tokens. One RL step generates 4.8 times that number before it trains.

    **Why:** SFT reads text that exists. RL must first write its text, token by token.

2. **Calculate the generation time, one by one and in one batch.** One by one, the engine gives 50 tokens/s: 38,400 / 50 = 768 s. With the 128 samples in one batch, a decode step takes 19.8 + 0.2 × 128 = 45.4 ms. The 300 steps take 13.62 s.

    **Why:** the GPU and the model are the same, and the time is 56.4 times shorter. The only difference is how the **inference engine** (the program that generates) fills its batch.

3. **Add the training part of the step.** The step trains on 128 × (200 + 300) = 64,000 tokens: 25.6 s at 2,500 tokens/s.

    | Engine use | Generate | Train | Step | Share of generation |
    |---|---|---|---|---|
    | One by one | 768 s | 25.6 s | 793.6 s | 96.8% |
    | One batch of 128 | 13.62 s | 25.6 s | 39.22 s | 34.7% |

    **Why:** generation is the part that can change by a factor of 56. Thus the inference engine, more than the algorithm, decides the cost of RL.

4. **Compare a run with the SFT job.** Assume 500 RL steps. They generate 19.2 million tokens: more than the 15 million tokens that the full SFT job reads. The run takes 5.4 hours with batched generation and 110 hours one by one. The SFT job takes 100 minutes.

    **Why:** RL costs several times more than SFT on the same model. You need a reason to pay this.

5. **Count the tokens of DPO for comparison.** The model reads 2 answers for each of 10,000 pairs: 30 million tokens in 3 epochs, 2 times the SFT job. Nothing is generated.

    **Why:** DPO is the cheap middle level. It needs pairs, but no inference engine and no reward function.

<div class="key-point">

In RL the cost is mostly generation. Look first at how a library generates, then at its algorithms. Write the reward function first: no library gives it to you.

</div>

### From toy to real

The TRL documentation says: "Generation is often the main bottleneck when training with online methods." The OpenRLHF README gives 80% of the training time for the generation of samples.

- **TRL `DPOTrainer` and `GRPOTrainer`.** The same interface as `SFTTrainer`. `GRPOTrainer` takes your reward functions and can generate with vLLM, in the same process or from a server. Take them first on one machine. Limit: the documentation lists 7 stable trainers. The other methods are in `trl.experimental`, which can change in any release.
- **Unsloth (GRPO notebooks).** GRPO on one GPU. Take it to learn RL with a small budget. Limit: the multi-GPU limit of part 1.
- **verl.** An RL library for clusters: Ray starts the workers, FSDP or Megatron-LM trains, vLLM or SGLang generates. Take it for large models, many GPUs or a new algorithm. Limit: its quickstart asks for a GPU of 24 GB for a model of 0.5 billion parameters.
- **OpenRLHF.** Ray, vLLM and DeepSpeed, with PPO, GRPO and other algorithms. Take it for the same cases as verl. Limit: the first lines of its README now recommend a newer framework (Molt).
- **The `verifiers` library.** It is not a trainer. You write an **environment** (the tasks, the tools and the rewards) one time, and use it for evaluation and for RL. Take it when the task is an agent with tools. Limit: its README describes a tight integration with the hub and the trainer (prime-rl) of Prime Intellect.
- **ART (OpenPipe).** A client in your application records what the agent does. A server trains a LoRA with GRPO and serves it with vLLM. Take it to train an agent that exists. Limit: LoRA with GRPO only.
- **Tinker (Thinking Machines).** A hosted API. You write the loop on a CPU machine with calls such as `forward_backward`, `optim_step` and `sample`, and the service runs the GPUs. Take it when you have no GPU, or to write your own loop. Limits: LoRA only, only the models on its list, and a price for each million tokens.

## 3. The tools around the trainer

### Setup

Around the trainer you need tools for the data, the tracking, the generation, the [evaluation](06-evaluation.md), several GPUs and the export.
The arithmetic is for the evaluation: 1,000 prompts with 200 output tokens each, after each of 5 checkpoints.

### Steps

1. **Count the output tokens of one evaluation.** 1,000 × 200 = 200,000 tokens.

    **Why:** an evaluation is generation. The speed of the trainer does not help here.

2. **Calculate the time one by one.** 200,000 / 50 = 4,000 s: 66.7 minutes. The 5 evaluations take 333 minutes. The training takes 100 minutes.

    **Why:** a slow evaluation costs more than the training. Then you evaluate less often, and you see a problem late.

3. **Calculate the time with a batched engine.** With 64 requests in the batch, one decode step takes 19.8 + 0.2 × 64 = 32.6 ms. The engine gives 64 / 0.0326 = 1,963 tokens/s. One evaluation takes 101.9 s: 39.3 times faster. The 5 evaluations take 8.5 minutes.

    **Why:** the same engine serves the evaluation and the RL samples. It is the second tool to install, after the trainer.

<div class="key-point">

Choose these tools with the same rule as the trainer: the simplest one that does the job. A batched inference engine pays for itself at the first evaluation.

</div>

### From toy to real

| Job | Tool | Note |
|---|---|---|
| Load the data | `datasets` | CSV, JSON, Parquet; memory-mapped files; streaming |
| Make synthetic data | distilabel | An LLM generates and judges the data. Last release: January 2025. Community members now maintain it |
| Track experiments | Weights & Biases, Trackio | W&B is a hosted service with an account. Trackio is free and local, with the same `init`, `log` and `finish` calls |
| Generate fast | vLLM | Continuous batching, a server in the OpenAI format, LoRA adapters |
| Evaluate | lm-evaluation-harness, Lighteval, Inspect | Benchmarks from the command line. Inspect: a dataset, a solver and a scorer. Lighteval: last release in November 2025 |
| Use several GPUs | Accelerate | Built on `torch.distributed`. It also drives FSDP and DeepSpeed |
| Export | GGUF | See [Quantization](07-quantization.md). Axolotl and Unsloth can export to GGUF |

## 4. How to choose

### Setup

Three cases use the model of 8 billion parameters.

| Case | Data | Hardware |
|---|---|---|
| A | 10,000 examples with answers | One GPU with 24 GB |
| B | The same, with full fine-tuning | 4 GPUs with 80 GB each |
| C | Prompts and a checker, no answers | One GPU with 24 GB |

<figure class="diagram" data-src="training-libraries/decision-path.svg">
<figcaption>The three questions for case A. Each answer removes libraries from the list.</figcaption>
</figure>

### Steps

1. **Question 1: does the job fit in memory?** A: QLoRA needs 10.2 GB of the 21.6 GB. B: the limit for each GPU is 72 GB. One full copy on each GPU needs 132 GB: no. With the weights and the states split over the 4 GPUs (FSDP, or ZeRO stage 3), each GPU holds 128 / 4 + 4 = 36 GB: yes. C: QLoRA fits, and the inference engine needs a part of the same GPU.

    **Why:** a library that cannot do the method that fits is out. For B, the library must split the model.

2. **Question 2: what data do you have?** A and B have answers: SFT. C has only a checker: RL, with 38,400 generated tokens for each step.

    **Why:** the data sets the method. The method sets the group of libraries: part 1 or part 2.

3. **Question 3: how much of the loop must you change?** Nothing: a YAML file or a web UI is enough. One function (a loss or a reward): a trainer class in Python. All of it: your own loop.

    **Why:** each level of help hides a part of the loop. Take the highest level that still shows the part that you must change.

4. **Read the answer of each case.**

    | Case | Fits with | Method | Take first |
    |---|---|---|---|
    | A | QLoRA, 10.2 GB | SFT | Unsloth or TRL `SFTTrainer`. No code: Axolotl or LlamaFactory |
    | B | Split, 36 GB for each GPU | SFT | Axolotl with FSDP, or TRL with DeepSpeed |
    | C | QLoRA and the engine | RL | TRL `GRPOTrainer` with vLLM. On a cluster: verl |

    **Why:** three answers give a short list. You compare 2 tools, not 13.

<div class="key-point">

Ask in this order: memory, data, loop. Then take the tool with the most help that still passes the three questions.

</div>

### From toy to real

| Your situation | Take first |
|---|---|
| One NVIDIA GPU, answers, Python | TRL `SFTTrainer` or Unsloth |
| The same, no code | Axolotl (YAML) or LlamaFactory (web UI) |
| A Mac | MLX-LM |
| Several GPUs, full fine-tuning | Axolotl, or TRL with DeepSpeed |
| Pairs of good and bad answers | TRL `DPOTrainer` |
| A checker, one machine | TRL `GRPOTrainer` with vLLM, or an Unsloth notebook |
| A checker, a cluster | verl |
| An agent with tools | ART, or a `verifiers` environment |
| No GPU | Tinker |

The usual order of work starts with the simplest method:

- [Prompting](../engineer/01-running-llms.md): no training.
- SFT with QLoRA: 100 minutes here.
- DPO, when you have pairs: 2 times the tokens of SFT.
- RL, when you have only a checker: 5.4 hours for 500 steps here.

Move up one level only when a measured result says that the level below is not enough.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
PARAMS = 8e9

def gb(numbers, bits):
    return numbers * bits / 8 / 1e9

sizes = [(4096, 4096)] * 2 + [(4096, 1024)] * 2 + [(4096, 14336)] * 2 + [(14336, 4096)]
lora = 32 * sum(16 * (d_in + d_out) for d_in, d_out in sizes)   # 41,943,040 parameters
states = gb(lora, 8 * 16)                                       # 0.671 GB
full = gb(PARAMS, 8 * 16) + 4                                   # 132 GB
lora_16bit = gb(PARAMS, 16) + states + 4                        # 20.7 GB
qlora = gb(PARAMS, 5.5) + states + 4                            # 10.2 GB
steps = 10_000 * 3 / (4 * 4)                                    # 1,875

generated = 16 * 8 * 300                                        # 38,400 tokens for each RL step
decode_step = (19.8 + 0.2 * 128) / 1000                         # 0.0454 s for a batch of 128
batched = generated / (128 / decode_step)                       # 13.62 s
```

Real-library sketch (the tests do not run this code):

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    train_dataset=load_dataset("json", data_files="train.jsonl", split="train"),  # rows with "messages"
    quantization_config=BitsAndBytesConfig(                                       # QLoRA: 4-bit base
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    peft_config=LoraConfig(
        r=16, lora_alpha=16, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    args=SFTConfig(
        output_dir="out", num_train_epochs=3, learning_rate=2e-4,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,             # effective batch: 16
    ),
)
trainer.train()
```

## Comparison

SFT libraries:

| Library | Best for | Strong points | Limits | How you configure it |
|---|---|---|---|---|
| `Trainer`, PEFT, bitsandbytes | Full control | TRL builds on it | The most code | Python |
| TRL `SFTTrainer` | Default in Python | LoRA and 4-bit as arguments | One model on each GPU | Python or `trl` CLI |
| Unsloth | One small GPU | Notebooks, GGUF export, web UI | Manual multi-GPU; AGPL web UI | Python, web UI or app |
| Axolotl | Repeated runs, several GPUs | DeepSpeed, FSDP; DPO and GRPO too | NVIDIA or AMD GPU | YAML |
| LlamaFactory | No code | Web UI, many methods | Release of May 2026; GRPO in a different project | YAML or web UI |
| MLX-LM | A Mac | LoRA, DoRA, full, QLoRA | Apple silicon only | CLI or YAML |

RL libraries and services:

| Library | Best for | Strong points | Limits | How you configure it |
|---|---|---|---|---|
| TRL `DPOTrainer`, `GRPOTrainer` | First try, one machine | Interface of `SFTTrainer`; vLLM | 7 stable trainers; the others experimental | Python |
| Unsloth GRPO | Small budget | Notebooks for one GPU | Manual multi-GPU | Python |
| verl | Clusters, new algorithms | Ray; FSDP or Megatron-LM; vLLM or SGLang | 24 GB for its smallest example | Hydra configuration |
| OpenRLHF | Clusters | Ray, vLLM, DeepSpeed | README recommends Molt | Command-line flags |
| `verifiers` | Agent tasks | One environment for evaluation and RL | Not a trainer | Python package |
| ART | An agent that exists | Client and server; hosted backend | LoRA with GRPO only | Python client |
| Tinker | No GPU, own loop | Loop primitives, hosted GPUs | LoRA only; price for each token | Python API |

## Checks and common misconceptions

- **"The estimate says that it fits, so it will run."** An estimate is not a measurement. Check: run 10 steps with your longest examples and read the peak memory.
- **"RL is the advanced method, so it gives the best model."** RL costs the most (5.4 hours against 100 minutes here), and a wrong reward function teaches the wrong thing. Check: if you have answers or pairs, do SFT or DPO first and measure.
- **"The tutorial that I found is still correct."** These libraries change fast. The torchtune README says that the library is no longer actively maintained. Check: read the README and the date of the last release, and pin the versions of each run.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. This project did not run these libraries: each statement about a library was read in its linked documentation or repository on 11 November 2026.

- [TRL documentation](https://huggingface.co/docs/trl/index) and [repository](https://github.com/huggingface/trl) (the trainers, the QLoRA arguments, multi-GPU, the 8 samples, vLLM, the quote).
- [Transformers `Trainer`](https://huggingface.co/docs/transformers/main/en/trainer), [PEFT](https://github.com/huggingface/peft) and [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (what each part does).
- [Unsloth repository](https://github.com/unslothai/unsloth) and [multi-GPU page](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) (the three ways to use it, the speed claim, the licences, multi-GPU).
- [Axolotl repository](https://github.com/axolotl-ai-cloud/axolotl) and [multi-GPU guide](https://docs.axolotl.ai/docs/multi-gpu.html) (the YAML file, the methods, the GPUs, the GGUF export).
- [LlamaFactory repository](https://github.com/hiyouga/LlamaFactory) (the web UI, the methods, the memory estimates, the release of 30 May 2026).
- [MLX LM repository](https://github.com/ml-explore/mlx-lm) and [LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) (the fine-tuning types, the configuration, the GGUF limit).
- [verl repository](https://github.com/verl-project/verl) and [quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html) (Ray, the backends, the 24 GB).
- [OpenRLHF repository](https://github.com/OpenRLHF/OpenRLHF) (the parts, the algorithms, the 80%, Molt).
- [verifiers repository](https://github.com/PrimeIntellect-ai/verifiers) (environments, the link with prime-rl).
- [ART repository](https://github.com/OpenPipe/ART) (client and server, LoRA with GRPO, vLLM, the hosted backend).
- [Tinker documentation](https://tinker-docs.thinkingmachines.ai/) and [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) (the calls, LoRA only, the price unit).
- [vLLM documentation](https://docs.vllm.ai/en/latest/) (continuous batching, the server, LoRA adapters).
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [Lighteval](https://github.com/huggingface/lighteval) and [Inspect](https://inspect.aisi.org.uk/) (what each tool does, the Lighteval release of 24 November 2025).
- [Accelerate](https://huggingface.co/docs/accelerate/index), [Datasets](https://github.com/huggingface/datasets), [distilabel](https://github.com/argilla-io/distilabel), [Trackio](https://github.com/gradio-app/trackio) and [W&B](https://github.com/wandb/wandb) (what each tool does; distilabel: the release of 28 January 2025, the maintenance note).
- [torchtune repository](https://github.com/meta-pytorch/torchtune) (the note that it is no longer actively maintained).
- [Fine-tune Llama 3.1 Ultra-Efficiently with Unsloth](https://huggingface.co/blog/mlabonne/sft-llama3), Maxime Labonne (42 million trained parameters for rank 16), and [ZeRO](https://arxiv.org/abs/1910.02054), Rajbhandari et al., 2019 (16 bytes for each parameter, as in lesson 4).
