# 7. Deploying LLMs

## Core idea

To **deploy** a model is to make it available to users.
Three numbers decide where a model can run: the memory that it needs, the time that a user waits, and the cost of one answer.
The same model can run on a laptop, in a demo page, on a GPU server or on a phone. Only these three numbers change.

## The example

The [toy example](../toy-example.md) is too small for deployment questions. This page uses one sizing problem in all 4 sub-parts.
**Each value in this table is an assumption for the example.** It is not a measurement. Measure your own system.

| Item | Assumed value |
|---|---|
| Model | 8 billion parameters. The key-value cache has the shape of Llama 3 8B: 131,072 bytes for each token in 16-bit numbers (FP16). |
| Request | 500 input tokens, 200 output tokens |
| Laptop | 8 GB of memory for the model. Prefill: 250 tokens/s. Decode: 20 tokens/s. |
| Server GPU | 24 GB of memory. Price: 2.00 USD for each hour. Prefill: 5,000 tokens/s. One decode step: 19.8 ms + 0.2 ms × B. |
| Phone or browser | 3 GB of memory for the model, with 0.5 GB reserved for the cache and the runtime |

On this page, 1 GB = 10⁹ bytes. $B$ is the **batch size**: the number of requests that the GPU processes at the same time.
The steps show rounded values, but each calculation uses the values before rounding. Thus the last digit of a result can differ.
The page [Inference optimization](06-inference-optimization.md) explains the key-value cache (KV cache), prefill and decode.

## 1. Local deployment

### Setup

In a **local deployment**, the model runs on the machine of the user. No text leaves the machine, which is good for privacy.
The first question is: does the model fit in the 8 GB of the laptop? The second question is: how long is the wait?

### Steps

1. **Calculate the memory of the weights.**

    $$\text{weights (bytes)} = \text{parameters} \times \frac{\text{bits for each parameter}}{8}$$

    FP16: 8 × 10⁹ × 16 / 8 = 16 GB. With 4 bits: 8 × 10⁹ × 4 / 8 = 4 GB.

    **Why:** all the weights must be in memory before the first token. The FP16 model does not fit in 8 GB. [Quantization](../scientist/07-quantization.md) to 4 bits makes it 4 times smaller.
    { .why }

2. **Add the KV cache.** One request has 500 + 200 = 700 tokens: 700 × 131,072 bytes = 0.0918 GB, or 0.092 GB. The totals are 16.092 GB (FP16) and 4.092 GB (4 bits).

    **Why:** the weights are not the only memory. The cache grows with each token of the context. Here 4.092 GB is less than 8 GB, so the 4-bit model fits.
    { .why }

3. **Calculate the time to first token.** The **time to first token (TTFT)** is the wait before the first word shows.

    $$\text{TTFT} = \frac{\text{input tokens}}{\text{prefill speed}} + \frac{1}{\text{decode speed}}$$

    $$\text{TTFT} = \frac{500}{250} + \frac{1}{20} = 2.05\ \text{s}$$

    **Why:** the model must read the full prompt before it can write. A longer prompt gives a longer TTFT.
    { .why }

4. **Calculate the total latency.** The **latency** is the time until the answer is complete.

    $$\text{latency} = \frac{\text{input tokens}}{\text{prefill speed}} + \frac{\text{output tokens}}{\text{decode speed}}$$

    $$\text{latency} = \frac{500}{250} + \frac{200}{20} = 12.0\ \text{s}$$

    **Why:** the model writes one token for each pass, so the output length controls the total time. Here the 200 output tokens take 10 of the 12 seconds.
    { .why }

!!! success "Key point"
    Local deployment is a memory question first. Weights = parameters × bits / 8. Then two speeds give the wait: the prefill speed for the prompt and the decode speed for the answer.

### From toy to real

Local tools download a quantized model file and run it with one command or one click. LM Studio, Ollama, TextGen (the new name of oobabooga's text-generation-webui) and KoboldCpp all do this. The table in **Comparison** shows how they differ.

Real-library sketch (the tests do not run this code): Ollama starts a chat and also serves a local API at `http://localhost:11434`.

```bash
ollama run <model-name>
```

## 2. Demo deployment

### Setup

A **demo** is a small web page that lets a few persons try your model. A Python library builds the page, and one process answers one request at a time.
We now use the server GPU with $B = 1$: one decode step takes 19.8 + 0.2 = 20 ms, which is 50 tokens/s.

<figure class="diagram">
--8<-- "docs/assets/figures/deploying/timeline.svg"
<figcaption>One request on the server GPU, alone. The user sees the first token after 0.12 s and the last token after 4.10 s.</figcaption>
</figure>

### Steps

1. **Calculate what one user sees.** TTFT = 500 / 5,000 + 0.02 = 0.12 s. Latency = 0.1 + 200 × 0.02 = 4.1 s.

    **Why:** these are the same two formulas as on the laptop. Only the speeds changed.
    { .why }

2. **Stream the tokens.** To **stream** is to show each token when the model writes it. The user reads the first words after 0.12 s, not after 4.1 s.

    **Why:** streaming does not make the model faster. It makes the wait shorter for the eye. This is why TTFT is a separate number from latency.
    { .why }

3. **Count the requests that the demo can serve.** One request at a time gives 1 / 4.1 = 0.244 requests/s, or 14.6 requests for each minute. If 10 requests come at the same time, the last answer is complete after 10 × 4.1 = 41 s.

    **Why:** without batching, requests wait in a queue. This is acceptable for a demo. It is not acceptable for a product.
    { .why }

!!! success "Key point"
    A demo needs little code and serves few users. Stream the output: the user then waits for the TTFT (0.12 s), not for the full latency (4.1 s).

### From toy to real

Gradio and Streamlit build a chat page in a few lines of Python. Hugging Face Spaces hosts such a page online.

Real-library sketch (the tests do not run this code):

```python
import gradio as gr

def reply(message, history):
    return my_model(message)      # your function that calls the model

gr.ChatInterface(reply).launch()
```

## 3. Server deployment

### Setup

A server has many users at the same time, and a GPU-hour costs money when it is empty too. The aim is to get the most tokens from each GPU-hour.

The method is **batching**: the GPU processes $B$ requests in one pass. One decode step then gives 1 token to each request. We assume that a step takes 19.8 ms + 0.2 ms × $B$: a fixed part to read the weights, and a small part for each request.

With **continuous batching**, a new request enters the batch at the next decode step, as soon as another request ends.

<figure class="diagram">
--8<-- "docs/assets/figures/deploying/batching.svg"
<figcaption>Static batching leaves slots idle until the longest answer ends. Continuous batching fills a free slot at the next step.</figcaption>
</figure>

### Steps

1. **Calculate the speed of one request in a batch of 32.** One step takes 19.8 + 0.2 × 32 = 26.2 ms. Each request gets 1,000 / 26.2 = 38.17 tokens/s.

    **Why:** each user is a little slower than alone (38.17 against 50 tokens/s). That is the price of sharing the GPU.
    { .why }

2. **Calculate the latency of one request.** 0.1 + 200 × 0.0262 = 5.34 s. The TTFT is 0.1 + 0.0262 = 0.126 s.

    **Why:** you must check that the slower answer is still acceptable for the user before you increase $B$.
    { .why }

3. **Calculate the throughput.** The **throughput** is the work that the system completes in each second.

    $$\text{requests/s} = \frac{B}{\text{latency}(B)} = \frac{32}{5.34} = 5.993$$

    Alone, the result was 0.244 requests/s. The batch gives 24.6 times more, or 1,198.5 output tokens/s.

    **Why:** continuous batching keeps all 32 slots full. So 32 requests end in each period of 5.34 s.
    { .why }

4. **Check the memory.** The cache of 32 requests is 32 × 700 × 131,072 bytes = 2.936 GB. With 16 GB of weights, the total is 18.936 GB. This fits in 24 GB. The free 8 GB hold 8 / 0.0918 = 87.2 caches, so the largest batch is 87 requests.

    **Why:** each request in the batch has its own KV cache. The memory, not the speed, gives the upper limit of $B$.
    { .why }

5. **Calculate the cost for 1 million tokens.** One request is 700 tokens. In one hour the GPU processes 5.993 × 3,600 × 700 = 15,101,124 tokens.

    $$\text{cost for 1M tokens} = \frac{\text{GPU price for each hour}}{\text{million tokens for each hour}}$$

    $$\frac{2.00}{15.10} = 0.132\ \text{USD}$$

    With $B = 1$, the GPU processes 614,634 tokens in one hour: 3.254 USD for 1 million tokens.

    **Why:** you pay for GPU time, not for tokens. A GPU that has work only 30% of the time costs 0.132 / 0.3 = 0.441 USD for 1 million tokens.
    { .why }

6. **Calculate the GPU-hours for a workload.** Assume 1 million requests each day. One GPU serves 5.993 × 3,600 = 21,573 requests in one hour. The day needs 46.4 GPU-hours, which is 1.93 GPUs. You need 2 GPUs: 2 × 24 × 2.00 = 96 USD each day.

    **Why:** a **GPU-hour** (one GPU for one hour) is the unit that you buy. This step changes a traffic forecast into a number of machines. Real traffic has peaks, so plan for the busiest hour, not for the average.
    { .why }

<figure class="plot" markdown>
![Line chart of the cost for 1 million tokens against the batch size, on logarithmic axes. The cost falls from 3.25 USD at B = 1 to 0.43 USD at B = 8, 0.13 USD at B = 32 and 0.07 USD at B = 87.](../assets/figures/deploying/cost-vs-batch.svg#only-light)
![Line chart of the cost for 1 million tokens against the batch size, on logarithmic axes. The cost falls from 3.25 USD at B = 1 to 0.43 USD at B = 8, 0.13 USD at B = 32 and 0.07 USD at B = 87.](../assets/figures/deploying/cost-vs-batch-dark.svg#only-dark)
<figcaption>The same GPU at the same price. A larger batch divides the cost of each token. The curve ends at B = 87, the memory limit.</figcaption>
</figure>

!!! success "Key point"
    On a server, batching is the main lever. In this example a batch of 32 makes each user 1.3 times slower and each token 24.6 times cheaper.

### From toy to real

- **vLLM** is an inference server with continuous batching, an API in the OpenAI format, and PagedAttention, which manages the KV-cache memory.
- **TGI** (Text Generation Inference) is the server of Hugging Face. Its repository became read-only (archived) in March 2026. Its README recommends vLLM or SGLang for new projects.
- **SkyPilot** starts your server on a cloud or on your own cluster from one configuration file.
- API providers usually give one price for input tokens and a higher price for output tokens. An output token needs one pass each. Input tokens are read in parallel.

Real-library sketch (the tests do not run this code):

```bash
vllm serve <model-name>      # serves http://localhost:8000/v1/chat/completions
```

## 4. Edge deployment

### Setup

In an **edge deployment**, the model runs on a small device of the user: a phone, or a browser tab. There is no server cost, and the text stays on the device.
The limit is memory. We assume 3 GB for the model, and we reserve 0.5 GB of it for the KV cache and the runtime.

### Steps

1. **Try the server model.** 16 GB of FP16 weights + 0.5 GB = 16.5 GB. The limit is 3 GB.

    **Why:** this shows the size of the gap: the model is more than 5 times too large.
    { .why }

2. **Quantize to 4 bits.** 4 GB + 0.5 GB = 4.5 GB. This is still more than 3 GB.

    **Why:** quantization divides the memory by 4. For this device that is not sufficient.
    { .why }

3. **Calculate the largest model that fits.**

    $$\text{parameters} \le \frac{(\text{budget} - \text{reserve}) \times 8}{\text{bits}}$$

    $$\frac{(3 - 0.5) \times 10^9 \times 8}{4} = 5 \times 10^9$$

    **Why:** the formula of step 1 in sub-part 1, turned around. It gives the limit before you download a model: 5 billion parameters at 4 bits.
    { .why }

4. **Select a smaller model and quantize it.**

    | Model | Weights | With the 0.5 GB reserve | Phone (3 GB) | Laptop (8 GB) |
    |---|---|---|---|---|
    | 8 billion, FP16 | 16 GB | 16.5 GB | No | No |
    | 8 billion, 4 bits | 4 GB | 4.5 GB | No | Yes |
    | 3 billion, 4 bits | 1.5 GB | 2.0 GB | Yes | Yes |
    | 1 billion, 4 bits | 0.5 GB | 1.0 GB | Yes | Yes |

    **Why:** on the edge you use two levers together: fewer parameters and fewer bits. A smaller model is less capable, so give it a narrower task.
    { .why }

!!! success "Key point"
    The device memory sets the model size: parameters ≤ (budget − reserve) × 8 / bits. For a phone that means a small model and 4-bit weights.

### From toy to real

- **MLC LLM** compiles a model for the hardware of the device. It lists iOS, Android and web browsers (through WebGPU) as targets.
- **mnn-llm** runs models with the MNN inference engine on Android and iOS. Its README says that the project is now a part of MNN.
- A phone also has limits for heat and battery. Measure the speed on the real device.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
PARAMS, N_IN, N_OUT = 8e9, 500, 200
KV_BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2                  # 131,072 (key and value, FP16)

def weights_gb(params, bits):  return params * bits / 8 / 1e9            # 16.0 at FP16, 4.0 at 4 bits
def kv_cache_gb(tokens, batch=1):  return KV_BYTES_PER_TOKEN * tokens * batch / 1e9   # 0.092

def step_s(batch):  return (19.8 + 0.2 * batch) / 1000      # assumed time of one decode step
def ttft_s(batch):  return N_IN / 5000 + step_s(batch)                   # 0.12 for B = 1
def latency_s(batch):  return N_IN / 5000 + N_OUT * step_s(batch)        # 4.1 and 5.34

def requests_per_s(batch):  return batch / latency_s(batch)              # 0.244 and 5.993
def usd_per_million_tokens(batch, price=2.0, utilization=1.0):
    tokens_per_hour = requests_per_s(batch) * 3600 * (N_IN + N_OUT)
    return price / (tokens_per_hour * utilization / 1e6)                 # 3.254 and 0.132

def largest_model(budget_gb, bits, reserve_gb=0.5):
    return (budget_gb - reserve_gb) * 1e9 * 8 / bits                     # 5e9 for a 3 GB phone
```

## Comparison

The four places, with the results of the example:

| Place | Users | Limit in the example | Result in the example |
|---|---|---|---|
| Local (laptop) | 1 | 8 GB of memory | The 4-bit model fits (4.092 GB). One answer takes 12.0 s. |
| Demo | A few | One request at a time | 14.6 requests for each minute. TTFT 0.12 s with streaming. |
| Server | Many | 24 GB: a batch of 87 at most | Batch of 32: 5.993 requests/s, 0.132 USD for 1 million tokens |
| Edge (phone, browser) | 1 | 3 GB of memory | A model of 3 billion parameters at 4 bits (2.0 GB) |

The tools of the course section:

| Tool | Place | What it is |
|---|---|---|
| LM Studio | Local | Desktop application that downloads and runs local models (llama.cpp and MLX engines) |
| Ollama | Local | Command-line tool and local API to run open models |
| TextGen (oobabooga) | Local | Desktop application with several engines (llama.cpp, Transformers, ExLlamaV3) |
| KoboldCpp | Local | One executable file, built on llama.cpp, with a web page and an API |
| Gradio | Demo | Python library that builds a web page for a model |
| Streamlit | Demo | Python library that builds data and chat applications |
| Hugging Face Spaces | Demo | Hosting for demo applications |
| vLLM | Server | Inference server with continuous batching and PagedAttention |
| TGI | Server | Inference server of Hugging Face, archived (read-only) in March 2026 |
| SkyPilot | Server | Tool that starts jobs and servers on clouds or your own cluster |
| MLC LLM | Edge | Compiler and engine for iOS, Android and browsers |
| mnn-llm | Edge | LLM runtime on the MNN engine for Android and iOS |

## Checks and common misconceptions

- **"If the weights fit, the model fits."** Add the KV cache: bytes for each token × tokens × batch size. Check: in the example, 16 GB of weights fit in 24 GB, but a batch of 88 does not.
- **"A larger batch makes each answer faster."** It makes each answer slower (38.17 against 50 tokens/s) and each token cheaper. Check: measure the latency of one request and the throughput of the server as two separate numbers.
- **"The cost of a token is a property of the model."** It depends on the batch size and on how busy the GPU is: 3.254, 0.132 or 0.441 USD in this example, for the same model on the same GPU.

## Sources

This page follows section 7 of "The LLM Engineer" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [Streamlit: build a basic LLM chat app](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps).
- [Hugging Face LLM inference container for Amazon SageMaker](https://huggingface.co/blog/sagemaker-huggingface-llm).
- [Blog of Philipp Schmid](https://www.philschmid.de/) (deployment articles).
- [Optimizing latency](https://hamel.dev/notes/llm/inference/03_inference.html), Hamel Husain (comparison of TGI, vLLM, CTranslate2 and MLC).
- Local tools: [LM Studio](https://lmstudio.ai/), [Ollama](https://github.com/ollama/ollama), [TextGen / text-generation-webui](https://github.com/oobabooga/textgen), [KoboldCpp](https://github.com/LostRuins/koboldcpp).
- Demo tools: [Gradio](https://www.gradio.app/), [Streamlit](https://docs.streamlit.io/), [Hugging Face Spaces](https://huggingface.co/spaces).
- Server tools: [vLLM](https://github.com/vllm-project/vllm), [TGI](https://github.com/huggingface/text-generation-inference) (archived; the README names the servers to use now), [SkyPilot](https://skypilot.readthedocs.io/en/latest/).
- Edge tools: [MLC LLM](https://github.com/mlc-ai/mlc-llm), [mnn-llm](https://github.com/wangzhaode/mnn-llm/blob/master/README_en.md).
- [Llama 3 8B configuration](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json), Meta (32 layers, 8 key-value heads of size 128: the shape of the KV cache in the example).
