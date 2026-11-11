# 1. Running LLMs

## Core idea

To use an LLM, you send it a text and you get a text back. The model can run on the server of a company (an API) or on your machine.
In the two cases, you control only two things: the text that goes in (the prompt), and the rule that picks each next token (the decoding).
This page shows how to do the two things well, and what they cost.

## The example

| Item | Value |
|---|---|
| Request | The sentence "the cat" of the [toy example](../toy-example.md). The model answers `sat` |
| Cost example | 1,000 input tokens and 200 output tokens. **Assumed** prices: 3 dollars for 1 million input tokens, 15 dollars for 1 million output tokens |
| Local model | 8 billion parameters, 32 blocks, context of 8,192 tokens |
| Logits after "the cat" | `[0.1, 1.0, 2.0]` for `the`, `cat`, `sat` |
| Prompt task | "A cat sleeps 14 hours each day. How many hours does it sleep in 3 days?" (answer: 42) |

The prices are not the prices of a real provider. Real prices change often: read the price page of the provider.

## 1. LLM APIs

### Setup

An **API** (application programming interface) is a server that runs the model for you. Your program sends an HTTP request and gets a response.
Most providers use the same shape for the data: a list of **messages**, each with a role and a content.

<figure class="diagram">
--8<-- "docs/assets/figures/running-llms/api-call.svg"
<figcaption>One API call. You pay for the tokens that go in and for the tokens that come back.</figcaption>
</figure>

### Steps

1. **Build the request.** The role `system` gives the rules. The role `user` gives the question. The role `assistant` holds the earlier answers of the model.

    ```json
    {
      "model": "example-model",
      "messages": [
        {"role": "system", "content": "Complete the sentence with one word."},
        {"role": "user", "content": "the cat"}
      ],
      "max_tokens": 5,
      "temperature": 0
    }
    ```

    **Why:** the model has no memory between two calls. Each request must contain the full conversation. The server joins the messages into one token sequence, and the model continues it.
    { .why }

2. **Read the response.** It holds the new message and the token counts.

    ```json
    {
      "choices": [{"message": {"role": "assistant", "content": "sat"}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 11, "completion_tokens": 1}
    }
    ```

    The counts come from the rule "1 token is about 4 characters": 36 and 7 characters give 9 + 2 = 11 input tokens.

    **Why:** `usage` is what you pay for. `finish_reason` tells you if the model stopped by itself (`stop`) or was cut at `max_tokens` (`length`). A cut answer is not complete.
    { .why }

3. **Calculate the cost of one request.** Use the larger request of the example.

    $$\text{cost} = \frac{\text{input tokens} \times \text{input price} + \text{output tokens} \times \text{output price}}{1{,}000{,}000}$$

    $$\frac{1{,}000 \times 3}{1{,}000{,}000} + \frac{200 \times 15}{1{,}000{,}000} = 0.003 + 0.003 = 0.006 \text{ dollars}$$

    **Why:** the two prices are different. Here an output token costs 5 times more, so 200 output tokens cost as much as 1,000 input tokens. A common explanation: the model reads the input tokens in parallel, but it writes the output tokens one at a time.
    { .why }

4. **Multiply by the number of requests.** With 10,000 requests each day: 10,000 × 0.006 = 60 dollars each day, or 1,800 dollars in 30 days.

    **Why:** one call is almost free. The volume makes the bill. Calculate it before you choose a model, a prompt length and an output length.
    { .why }

!!! success "Key point"
    An API call is a JSON request with the full conversation and a JSON response with the answer and the token counts. Cost = tokens × price, counted for input and output.

### From toy to real

- **Private models**: OpenAI, Google and Anthropic run their own models. You cannot download the weights.
- **Open-weight models through an API**: OpenRouter, Hugging Face and Together AI run models that you can also download.
- The field names change a little between providers. For example, the Anthropic API takes the system prompt in a separate `system` field and names the counts `input_tokens` and `output_tokens`.

Real-library sketch (the tests do not run this code). It works with a server that uses the request shape of step 1. It reads the address and the key from the environment.

```python
import json, os, urllib.request

request = urllib.request.Request(
    os.environ["LLM_BASE_URL"] + "/chat/completions",
    data=json.dumps({"model": os.environ["LLM_MODEL"], "messages": messages, "max_tokens": 5}).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["LLM_API_KEY"]},
)
reply = json.load(urllib.request.urlopen(request))
print(reply["choices"][0]["message"]["content"], reply["usage"])
```

## 2. Open-source LLMs

### Setup

You can download the weights of an open model from the Hugging Face Hub and run it on your machine. Then no text leaves your machine, and you pay for hardware, not for tokens.
The first question is: does the model fit in the memory? The rule is:

$$\text{memory} = \text{parameters} \times \text{bytes for each weight} + \text{context overhead}$$

### Steps

1. **Calculate the size of the weights at FP16.** FP16 uses 2 bytes for each weight: $8 \times 10^9 \times 2 = 16$ GB.

    **Why:** the weights are the largest part. 16 GB does not fit in a GPU with 8 GB or 12 GB of memory.
    { .why }

2. **Calculate the size at 4 bits.** 4 bits are 0.5 byte: $8 \times 10^9 \times 0.5 = 4$ GB.

    **Why:** [quantization](../scientist/07-quantization.md) stores each weight with fewer bits. The model is 4 times smaller with a small loss of quality. A real 4-bit file is a little larger (about 4.9 GB): it also holds scales, and some parts of the model keep more than 4 bits.
    { .why }

3. **Add the context overhead.** The model keeps 2 vectors (a key and a value) for each token of the context in each block. This is the **key-value cache**. For Llama 3 8B, each of these vectors has 1,024 numbers of 2 bytes.

    $$2 \times 32 \text{ blocks} \times 8{,}192 \text{ tokens} \times 1{,}024 \times 2 \text{ bytes} = 1.07 \text{ GB}$$

    **Why:** without this cache, the model must process all the earlier tokens again for each new token. The cache grows with the context length: a context of 2,048 tokens needs 0.27 GB.
    { .why }

4. **Add the two parts and compare with your memory.**

    | Precision | Weights | Context (8,192 tokens) | Total |
    |---|---|---|---|
    | FP16 | 16 GB | 1.07 GB | 17.07 GB |
    | 4-bit | 4 GB | 1.07 GB | 5.07 GB |

    **Why:** the 4-bit model fits in a GPU with 8 GB or in the memory of a laptop. The FP16 model does not. The program itself also needs some memory, so keep a margin.
    { .why }

!!! success "Key point"
    Memory = parameters × bytes for each weight + context. Quantization makes the first part smaller. A shorter context makes the second part smaller.

### From toy to real

| Tool | What it is |
|---|---|
| Hugging Face Hub | The site where you find and download models |
| Hugging Face Spaces | Hosted web apps where you can try a model in the browser |
| LM Studio | A desktop app: download a model, chat with it, start a local server |
| llama.cpp | A command-line program for GGUF files, on CPU and GPU |
| Ollama | A command-line tool that downloads and runs models with one command |

Real-library sketch (the tests do not run this code). The two tools start a local server that accepts the request of part 1, so your program does not change.

```bash
llama-server -m model-Q4_K_M.gguf -c 8192 -ngl 99   # llama.cpp: context 8,192, all layers on the GPU
ollama run llama3.1                                 # Ollama: download the model and chat
```

## 3. Prompt engineering

### Setup

A **prompt** is the text that you send to the model. **Prompt engineering** is the work of writing it so that the correct answer becomes more probable.
The weights do not change. The prompt changes only the context, and the context changes the probability of each next token: $P(\text{next token} \mid \text{prompt})$.
The four prompts below ask the same question. The correct answer is 14 × 3 = 42.

### Steps

1. **Zero-shot: ask the question with no example.**

    ```text
    Q: A cat sleeps 14 hours each day. How many hours does it sleep in 3 days?
    A:
    ```

    **Why:** the model saw many questions with answers during training. After `A:`, an answer is the most probable continuation. This works when the task is common and the format of the answer is not important.
    { .why }

2. **Few-shot: show some solved examples first.**

    ```text
    Q: A dog eats 2 meals each day. How many meals does it eat in 5 days?
    A: 10
    Q: A bird sings 6 songs each hour. How many songs does it sing in 4 hours?
    A: 24
    Q: A cat sleeps 14 hours each day. How many hours does it sleep in 3 days?
    A:
    ```

    **Why:** the examples are part of the context. Attention reads them, and the pattern "after `A:` comes one number" becomes very probable. The examples teach the task and the format without any training.
    { .why }

3. **Chain of thought: ask for the steps before the answer.**

    ```text
    Q: A cat sleeps 14 hours each day. How many hours does it sleep in 3 days?
    A: Let's think step by step.
    ```

    The model writes, for example: "One day is 14 hours. 3 days are 14 × 3 = 42 hours. The answer is 42."

    **Why:** the model calculates each token from the tokens before it. When "14 × 3 =" is in the text, the token `42` is an easy next token. Without the steps, the model must do the full calculation inside one token prediction. The written steps are its working memory.
    { .why }

4. **ReAct: let the model use a tool between its thoughts.** **ReAct** means "reason and act". The prompt gives a format. Your program runs the tool and writes the `Observation` line.

    ```text
    Thought: I must multiply 14 by 3.
    Action: calculator[14 * 3]
    Observation: 42
    Answer: 42
    ```

    **Why:** the observation does not come from the model. It comes from a real tool, so it is correct. The next tokens are then predicted from a true fact in the context, not from a guess.
    { .why }

The four prompts do not have the same size. The token counts use the rule "1 token is about 4 characters".

| Technique | Characters of the prompt | Input tokens (about) | Calls to the model |
|---|---|---|---|
| Zero-shot | 77 | 20 | 1 |
| Few-shot | 234 | 59 | 1 |
| Chain of thought | 103 | 26 | 1 (with more output tokens) |
| ReAct (prompt with the format rules) | 285 | 72 | 2 or more |

!!! success "Key point"
    A prompt is the only input of the model. Each technique puts useful text in the context before the answer: examples (few-shot), intermediate results (chain of thought) or tool results (ReAct).

### From toy to real

- Better prompts cost tokens. The few-shot prompt adds 39 input tokens to each call. With the assumed price and 10,000 calls each day, this is 1.17 dollars each day.
- These techniques work better with large models. A small model can fail to follow the ReAct format.
- Do not trust one example. Test a prompt on many inputs and count the correct answers.
- ReAct is the base of agents: a loop of thought, tool call and observation.

## 4. Structuring outputs

### Setup

A program must often parse the answer, for example as JSON (a text format for data). One wrong character breaks the parser.
**Constrained decoding** makes a wrong format impossible. At each step, a **grammar** (a set of rules for valid text) gives the list of allowed tokens. All the other tokens are removed before the sampling.
Toy: after "the cat", the logits are `[0.1, 1.0, 2.0]`. Assume that the grammar allows only `the` and `cat`.

### Steps

1. **Start with the normal probabilities.** Softmax gives `[0.099, 0.242, 0.659]`. The model prefers `sat`.

    **Why:** this is what happens without a constraint. The most probable token is not valid here.
    { .why }

2. **Mask the logits.** Set the logit of each token that is not allowed to $-\infty$: `[0.1, 1.0, -inf]`.

    **Why:** $e^{-\infty} = 0$. After softmax, this token has a probability of exactly 0, so no sampling method can pick it.
    { .why }

3. **Apply softmax again.**

    $$e^{0.1} = 1.105,\quad e^{1.0} = 2.718,\quad \text{sum} = 3.823 \qquad p = \left[\tfrac{1.105}{3.823},\ \tfrac{2.718}{3.823},\ 0\right] = [0.289,\ 0.711,\ 0]$$

    **Why:** the probabilities must add up to 1 again. The ratio between `the` and `cat` does not change: 0.099 ÷ 0.341 = 0.289. The model still chooses, but only among valid tokens.
    { .why }

4. **Do this at each step with a JSON schema.** A **JSON schema** describes the allowed fields and values. Assume a schema that allows a field `word` with the value `the` or `cat`. Only 2 outputs are valid: `{"word": "the"}` and `{"word": "cat"}`. Each has 15 characters. At 14 of the 15 positions, the grammar allows one character only. The model chooses at 1 position: `t` or `c`.

    **Why:** the grammar writes the structure, and the model writes the content. The output always parses.
    { .why }

<figure class="plot" markdown>
![Bar chart of the probability of each token with and without the grammar. Without: the 0.099, cat 0.242, sat 0.659. With the grammar: the 0.289, cat 0.711, sat 0.](../assets/figures/running-llms/constrained.svg#only-light)
![Bar chart of the probability of each token with and without the grammar. Without: the 0.099, cat 0.242, sat 0.659. With the grammar: the 0.289, cat 0.711, sat 0.](../assets/figures/running-llms/constrained-dark.svg#only-dark)
<figcaption>The grammar removes "sat". Its probability goes to the two allowed tokens, in the same ratio as before.</figcaption>
</figure>

!!! success "Key point"
    Constrained decoding changes the logits, not the model. It guarantees the format. It does not guarantee that the content is correct.

### From toy to real

- **JSON mode** of an API guarantees valid JSON. A **JSON schema** mode (structured outputs) also guarantees your fields and types.
- **Outlines** changes a schema or a regular expression into a state machine. For each state, it calculates one time the list of allowed tokens of the vocabulary. The mask of each step is then a fast table lookup.
- llama.cpp has the same function with [grammar files](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md). **LMQL** is a query language in which you write the prompt and the constraints together.
- Set `max_tokens` high enough. A cut output is not valid JSON, also with a grammar.

Real-library sketch (the tests do not run this code), with Outlines:

```python
import outlines
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "microsoft/Phi-3-mini-4k-instruct"
model = outlines.from_transformers(AutoModelForCausalLM.from_pretrained(name), AutoTokenizer.from_pretrained(name))
word = model("Complete the sentence with one word: the cat", Literal["the", "cat"])   # "the" or "cat", never "sat"
```

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import numpy as np

def cost(input_tokens, output_tokens, price_in=3.0, price_out=15.0):   # assumed prices per million tokens
    return (input_tokens * price_in + output_tokens * price_out) / 1e6

print(cost(1000, 200), 10_000 * cost(1000, 200))        # 0.006 dollars, 60 dollars each day

weights_fp16 = 8e9 * 2 / 1e9                            # 16 GB
weights_4bit = 8e9 * 0.5 / 1e9                          # 4 GB
kv_cache = 2 * 32 * 8192 * 1024 * 2 / 1e9               # 1.07 GB
print(weights_fp16 + kv_cache, weights_4bit + kv_cache) # 17.07 GB, 5.07 GB

logits = np.array([0.1, 1.0, 2.0])                      # the, cat, sat
allowed = np.array([True, True, False])                 # the grammar allows "the" and "cat"
masked = np.where(allowed, logits, -np.inf)
probs = np.exp(masked) / np.exp(masked).sum()           # [0.289, 0.711, 0]
```

## Comparison

| | API | Local model |
|---|---|---|
| You pay for | Tokens (toy: 0.006 dollars for each request) | Hardware and electricity |
| Limit | Price, rate limits | Memory (toy: 5.07 GB at 4 bits, 17.07 GB at FP16) |
| Your data | Goes to the provider | Stays on your machine |
| Model choice | The models of the provider | Any open model that fits |

| Technique | What you add to the context | Toy size | Good for | Risk |
|---|---|---|---|---|
| Zero-shot | Nothing | 20 tokens | Common, simple tasks | Free format, wrong on hard tasks |
| Few-shot | Solved examples | 59 tokens | A fixed format, a special task | The model copies the examples too much |
| Chain of thought | The steps, written by the model | 26 tokens + longer output | Calculation and logic | A wrong step gives a wrong answer |
| ReAct | Tool results | 72 tokens, 2 or more calls | Facts and exact calculation | More calls, the format can break |
| Constrained decoding | Nothing (it masks logits) | `[0.289, 0.711, 0]` | Output that a program parses | Valid format with wrong content |

## Checks and common misconceptions

- **"The API remembers my conversation."** It does not. Check the `messages` list that your program sends: each earlier turn is there, and you pay for its tokens again on each call.
- **"A 4-bit model of 4 GB needs 4 GB."** Add the context. Check the memory with the longest context that you will use, not with an empty one.
- **"JSON mode makes the answer correct."** It makes the answer valid JSON. The values can be wrong. Check the values in your program, for example with a schema validator and with tests on known inputs.

## Sources

This page follows section 1 of "The LLM Engineer" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [Run an LLM locally with LM Studio](https://www.kdnuggets.com/run-an-llm-locally-with-lm-studio), Nisha Arya.
- [Prompt engineering guide](https://www.promptingguide.ai/), DAIR.AI (zero-shot, few-shot, chain of thought, ReAct).
- [Outlines](https://github.com/dottxt-ai/outlines) and its [documentation](https://dottxt-ai.github.io/outlines/latest/), dottxt (the code of the sketch follows the README), and the paper [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702), Willard and Louf, 2023 (the state machine and the token index).
- [LMQL overview](https://lmql.ai/docs/language/overview.html).
- API documentation: [OpenAI](https://platform.openai.com/), [Google](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview), [Anthropic](https://docs.anthropic.com/claude/reference/getting-started-with-the-api), [OpenRouter](https://openrouter.ai/), [Hugging Face](https://huggingface.co/inference-api), [Together AI](https://www.together.ai/).
- Local tools: [Hugging Face Hub](https://huggingface.co/models), [Hugging Face Spaces](https://huggingface.co/spaces), [LM Studio](https://lmstudio.ai/), [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.ai/).
- [Llama 3 8B configuration](https://huggingface.co/NousResearch/Meta-Llama-3-8B/blob/main/config.json), a public copy of the model files (32 blocks, 8,192 tokens, hidden size 4,096 with 32 heads, 8 key-value heads: 8 × 128 = 1,024 numbers).
- [Structured outputs](https://platform.openai.com/docs/guides/structured-outputs), OpenAI (JSON mode and JSON schema).
- [ReAct](https://arxiv.org/abs/2210.03629), Yao et al., 2022, and [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916), Kojima et al., 2022 ("Let's think step by step").
