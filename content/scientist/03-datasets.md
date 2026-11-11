# 3. Post-training datasets

## Core idea

Pre-training uses raw text. Post-training uses conversations: an instruction and a good answer.
There is much less data of this type, so you often make it: a strong model writes samples, a checker verifies them, and filters remove the bad ones.
A dataset is good when its samples are correct, different from each other, and hard enough.

## The example

This page uses one small task: count the words of a text. The first sample is a conversation of 3 messages.

| Role | Content |
|---|---|
| system | `You are a concise assistant.` |
| user | `Count the words: the cat sat` |
| assistant | `3` |

No model is called on this page. Each "generated" text is written by hand, and plays the role of a model output.

<figure class="diagram" data-src="datasets/pipeline.svg">
<figcaption>The pipeline of this page, with the toy numbers of each stage.</figcaption>
</figure>

## 1. Storage and chat templates

### Setup

A model reads one sequence of tokens, not a list of messages. Thus a conversation goes through two formats.
The **storage format** is how the dataset file keeps the messages. The **chat template** is the rule that joins the messages into one string.

### Steps

1. **Store the conversation in the ShareGPT format.** Each message has a `from` key (who speaks) and a `value` key (the text).

    ```json
    {"conversations": [
      {"from": "system", "value": "You are a concise assistant."},
      {"from": "human", "value": "Count the words: the cat sat"},
      {"from": "gpt", "value": "3"}
    ]}
    ```

    **Why:** a JSON list keeps the order of the messages and who wrote each one. Many open datasets use this format.

2. **Or store it in the "messages" format of OpenAI and Hugging Face.** The keys are `role` and `content`. `human` becomes `user`, and `gpt` becomes `assistant`.

    ```json
    {"messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "Count the words: the cat sat"},
      {"role": "assistant", "content": "3"}
    ]}
    ```

    **Why:** the two formats hold the same information. Most APIs and training libraries read this one, so a conversion only changes the names of the keys.

3. **Render each message with the ChatML template.** ChatML writes 6 pieces for each message: the start marker `<|im_start|>`, the role, a newline, the content, the end marker `<|im_end|>`, and a newline.

    | Message | Start marker | Role | Newline | Content | End marker | Newline | Total characters |
    |---|---|---|---|---|---|---|---|
    | system | 12 | 6 | 1 | 28 | 10 | 1 | 58 |
    | user | 12 | 4 | 1 | 28 | 10 | 1 | 56 |
    | assistant | 12 | 9 | 1 | 1 | 10 | 1 | 34 |

    **Why:** the model must know where each message starts, where it stops, and who speaks. The markers are strings that do not occur in normal text.

4. **Join the messages.** This is the final training sample. It is one string: `\n` shows each newline character, and the line breaks below are only for reading.

    ```text
    <|im_start|>system\nYou are a concise assistant.<|im_end|>\n
    <|im_start|>user\nCount the words: the cat sat<|im_end|>\n
    <|im_start|>assistant\n3<|im_end|>\n
    ```

    The string has 58 + 56 + 34 = 148 characters: 57 of content and 91 of template.

    **Why:** the model learns this exact string. One missing newline gives different tokens. The template at training time and at use time must be the same.

5. **To get an answer, stop after the assistant header.** At use time, the string ends with `<|im_start|>assistant\n`. The model continues from there and writes `<|im_end|>` when it is done.

    **Why:** the model only continues text. The open header tells it that the next text is an answer. The end marker tells the program when to stop.

<div class="key-point">

The storage format is for people and tools. The chat template is for the model. A model fine-tuned with one template answers badly with another.

</div>

### From toy to real

- **Alpaca** is an older template for one instruction and one answer. It uses plain text headers: `### Instruction:` and `### Response:`.
- Each model family has its own template. The tokenizer of the model stores it, and `apply_chat_template` applies it for you.
- In a ChatML model, each marker is one special token in the vocabulary, not a group of characters.
- A dataset for preference alignment has one more field: for each instruction it stores a chosen answer and a rejected answer.

## 2. Synthetic data generation

### Setup

**Synthetic data** are samples that a model writes. A **seed** is a small input, written by a person, that starts the generation.
The generator is a strong model. Here, its outputs are written by hand.

### Steps

1. **Write the seed tasks.** We use 3: "count the words in a text", "reverse the order of the words in a text", "find the longest word in a text".

    **Why:** the seeds decide the topics of the dataset. The generator cannot add a topic that no seed asks for.

2. **Build one prompt for each seed.** The system prompt is the same for all: `You write training data for a Python tutor. Return only JSON with the keys "instruction" and "answer".` The user message contains the seed: `Seed task: count the words in a text.`

    **Why:** the system prompt sets the quality and the output format one time, for all the samples.

3. **Change the prompts to get different samples.** Add 2 personas ("a beginner", "a senior developer") and 2 styles ("a short question", "a detailed specification"). This gives 3 × 2 × 2 = 12 different prompts.

    **Why:** the same prompt gives almost the same sample each time. The variety of the outputs comes from the variety of the inputs.

4. **Collect the outputs.** Three outputs (written by hand, as a generator could return them):

    | Seed | Instruction | Answer (body of the function) |
    |---|---|---|
    | 1 | Write a Python function count_words(text) that returns the number of words in text. | `return len(text.split())` |
    | 2 | Write a Python function reverse_words(text) that returns the words of text in reverse order. | `return " ".join(reversed(text.split()))` |
    | 3 | Write a Python function longest_word(text) that returns the longest word of text. | `return max(text.split(), key=len)` |

    **Why:** each row becomes a conversation with a `user` message (the instruction) and an `assistant` message (the answer), stored as in section 1.

5. **Parse and validate each output.** Keep an output only if it is valid JSON with the 2 keys and no empty value. Our 3 outputs pass. An output that starts with "Sure! Here is the JSON:" fails.

    **Why:** a model does not always obey the format. One broken row can stop a training job, or teach the model to write "Sure!" in each answer.

<div class="key-point">

Generation is a prompt design task. The system prompt gives the quality, the seeds and the variations give the variety, and a validation step keeps the format clean.

</div>

### From toy to real

- Tools such as distilabel and the Synthetic Data Generator of Argilla run these steps for thousands of seeds.
- The seed can also be a document. The model then writes a question and an answer about this document.
- The generator writes its errors into the dataset. This is why the next two sections check each sample.

## 3. Data enhancement

### Setup

**Data enhancement** makes existing samples better. The most reliable method is a **checker**: a program that decides if an answer is correct.
For code, the checker is a **unit test** (a small program that calls the function and compares the result with the expected value).

### Steps

1. **Write the checker.** For `count_words` we write 2 tests: `count_words("the cat sat") == 3` and `count_words("") == 0`.

    **Why:** a program gives the same decision each time, and it costs almost nothing to run.

2. **Verify the generated answers.** The 3 answers of section 2 pass their tests. They are now **verified outputs**.

    **Why:** a model that learns from wrong answers gives wrong answers. A verified sample is correct, not only plausible.

3. **Ask for more than one answer.** Assume that the generator returns 4 answers for the first instruction (written by hand):

    | Answer | Body of the function | Test 1: "the cat sat" → 3 | Test 2: "" → 0 | Keep |
    |---|---|---|---|---|
    | A | `return len(text)` | Fail | Pass | No |
    | B | `return text.count(" ") + 1` | Pass | Fail | No |
    | C | `return len(text.split(" "))` | Pass | Fail | No |
    | D | `return len(text.split())` | Pass | Pass | Yes |

    **Why:** with a temperature above 0, the model gives a different answer each time. More answers give more chances to get a correct one.

4. **Reject each answer that fails.** This is **rejection sampling**. Only answer D remains: 1 of 4, an acceptance rate of 25%.

    **Why:** you do not need a model that is always correct. You need a model that is sometimes correct and a checker that knows the difference.

5. **Make the checker strict.** With test 1 only, answers B, C and D pass: 3 of 4. Two wrong answers go into the dataset.

    **Why:** the dataset is only as good as the checker. A weak test accepts code that fails on an empty text.

6. **Choose the number of answers.** If each answer is correct with a probability $p$, the probability that at least 1 of $n$ answers is correct is:

    $$P = 1 - (1 - p)^n$$

    With $p = 0.25$: 0.684 for $n = 4$, and 0.944 for $n = 10$.

    **Why:** each answer costs money. The formula tells you how many answers you must buy for a hard task.

<div class="key-point">

Generate many, verify, keep the best. A cheap, strict checker turns an unreliable generator into a reliable source of data.

</div>

### From toy to real

- For mathematics, the checker can be a solver that calculates the true result.
- **Chain-of-Thought**: the answer includes the reasoning steps before the result, so the model learns the method.
- **Personas**: the generator writes as different types of users, as in step 3 of section 2.
- **Auto-Evol**: a model rewrites instructions to make them harder, and also improves its own rewrite rules.
- **Branch-Solve-Merge**: a model splits a task into parts, solves each part, and joins the results.

## 4. Quality filtering

### Setup

We have a pool of 6 generated samples (written by hand). **Quality filtering** removes the bad ones, from the cheapest test to the most expensive.

| # | Instruction | Answer (function body, or full text) |
|---|---|---|
| 1 | count the words in the text | `return len(text.split())` |
| 2 | count the words in this text | `return len(text.split())` |
| 3 | reverse the words in the text | `return " ".join(reversed(text.split()))` |
| 4 | find the longest word | `return max(len(word) for word in text.split())` |
| 5 | count the letters in the text | `I'm sorry, but I cannot help with that.` |
| 6 | sort the words | `ok` |

### Steps

1. **Apply the rule filters.** Rule 1: the answer has 10 characters or more (an assumed limit). Sample 6 goes out (6 → 5). Rule 2: the answer does not contain a refusal such as "I'm sorry" or "I cannot". Sample 5 goes out (5 → 4).

    **Why:** rules are free and remove the obvious failures. A model trained on refusals learns to refuse.

2. **Measure the overlap of two instructions with the Jaccard similarity.** $A$ and $B$ are the sets of words of two texts.

    $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

    Sample 1 has 5 different words: {count, the, words, in, text}. Sample 2 has the same words plus "this".

    | Pair | Shared words | All words | Jaccard |
    |---|---|---|---|
    | 1 and 2 | 5 | 6 | 0.833 |
    | 1 and 3 | 4 | 6 | 0.667 |
    | 1 and 4 | 1 | 8 | 0.125 |

    Assume a limit of 0.8. Sample 2 is a **near-duplicate** of sample 1, and it goes out (4 → 3).

    **Why:** near-duplicates add cost and no new information. They also make the model repeat one style.

3. **Replace each set with a short MinHash signature.** Put all the words in a random order and give each word its rank. The signature value is the smallest rank among the words of the text. Do this 4 times, with 4 different orders:

    | Order | count | in | text | the | words | this | Minimum for sample 1 | Minimum for sample 2 |
    |---|---|---|---|---|---|---|---|---|
    | 1 | 4 | 2 | 5 | 9 | 1 | 0 | 1 | 0 |
    | 2 | 2 | 3 | 4 | 8 | 1 | 7 | 1 | 1 |
    | 3 | 5 | 9 | 2 | 1 | 3 | 6 | 1 | 1 |
    | 4 | 4 | 7 | 3 | 1 | 8 | 9 | 1 | 1 |

    The signatures are [1, 1, 1, 1] and [0, 1, 1, 1]. 3 of 4 values are equal: the estimate of the Jaccard similarity is 0.75.

    **Why:** the word with the smallest rank is a random word of the union. The two minimums are equal when this word is in the two sets, and the probability of this is exactly the Jaccard similarity (5 / 6). So you compare 4 numbers, not 2 full texts.

4. **Use more orders for a better estimate.** With 128 orders, the estimate for samples 1 and 2 is 0.844 (true value: 0.833). The standard error falls from 0.186 with 4 orders to 0.033 with 128.

    **Why:** each order is one random trial. More trials give less noise, as with a coin.

5. **Ask a judge LLM for the rest.** A **judge LLM** is a model that reads a sample and gives a score for criteria that you write. Assume the scores 5, 5 and 2 (written by hand, scale 1 to 5) for samples 1, 3 and 4, and a limit of 4. Sample 4 goes out (3 → 2): its answer returns the length of the longest word, not the word.

    **Why:** sample 4 passes all the rules. Only a reader that understands the task sees the error. A judge does this for samples that have no unit test.

<figure class="plot">
<img src="/figures/datasets/minhash.svg" alt="Line chart of the MinHash estimate against the number of hash functions, from 1 to 256. The estimate for samples 1 and 2 moves toward 0.833. The estimate for samples 1 and 3 moves toward 0.667.">
<figcaption>With few hash functions the estimate is noisy. It gets near the true Jaccard similarity as the number grows.</figcaption>
</figure>

<div class="key-point">

Filter in the order of cost: rules, then duplicates, then a judge. The pool goes from 6 samples to 2, and each filter removes a different type of error.

</div>

### From toy to real

- Real systems use hash functions in place of random orders, and groups of adjacent words in place of single words. FineWeb uses 5-grams and 14 × 8 = 112 hash functions.
- Two texts with the same meaning and different words have a low Jaccard similarity. Tools such as SemHash compare **embeddings** to find them.
- **N-gram decontamination** removes each training sample that shares a long sequence of words with a test set. Without it, the evaluation scores are too high.
- A **reward model** is a small model trained to give a quality score. It is faster than a judge LLM, but you cannot change its criteria with a prompt.

## Minimal code

This code gives the same strings and numbers as the steps on this page.

```python
def render_chatml(messages, add_generation_prompt=False):
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

def jaccard(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b)

def passes(code, name, tests):                    # rejection sampling keeps the answers that pass
    namespace = {}
    exec(code, namespace)                         # only for code that you wrote or run in a sandbox
    return all(namespace[name](*args) == expected for args, expected in tests)

jaccard("count the words in the text", "count the words in this text")   # 0.833
```

Real-library sketch (the tests do not run this code):

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)          # a model with a chat template
text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
prompt = tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
```

## Comparison

| Method | Finds | Toy result | Cost | Limit |
|---|---|---|---|---|
| Format validation | Broken outputs | 3 of 3 valid | Almost none | Says nothing about the content |
| Unit test or solver | Wrong answers | 1 of 4 answers kept | Low | Only for tasks that a program can check |
| Rule filters | Short answers, refusals | 6 → 4 samples | Almost none | Only errors that you can write as a rule |
| Jaccard similarity | Near-duplicates | 0.833 for samples 1 and 2 | Each pair of full texts | Too slow for millions of samples |
| MinHash | Near-duplicates, at scale | 0.844 with 128 hash functions | Small fixed signature | An estimate; no meaning, only shared words |
| Embeddings | Same meaning, other words | Not calculated here | One model call for each sample | Needs a good embedding model |
| Judge LLM | Errors of content and style | Sample 4 gets 2 of 5 | One large model call for each sample | The judge also makes errors |

## Checks and common misconceptions

- **"The chat template is a detail."** It is part of the input. Check: print one rendered sample and compare it, character by character, with the template of your model.
- **"More synthetic samples make a better dataset."** 1,000 copies of one idea teach one idea. Check: measure the near-duplicate rate, and look at the samples that the filters remove.
- **"A judge LLM can replace the other filters."** It is the slowest and most expensive filter, and it makes errors. Check: compare the judge with your own scores on a small set before you trust it.

## Sources

This page follows section 3 of "The LLM Scientist" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [Synthetic Data Generator](https://github.com/argilla-io/synthetic-data-generator), Argilla.
- [LLM Datasets](https://github.com/mlabonne/llm-datasets), Maxime Labonne.
- [NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator), Nvidia.
- [Distilabel](https://distilabel.argilla.io/dev/sections/pipeline_samples/), Argilla.
- [SemHash](https://github.com/MinishLab/semhash), MinishLab.
- [Chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating), Hugging Face (ChatML markers, generation prompt).
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), Taori et al. (Alpaca template).
- [Auto Evol-Instruct](https://arxiv.org/abs/2406.00770), Zeng et al., 2024.
- [Branch-Solve-Merge](https://arxiv.org/abs/2310.15123), Saha et al., 2023.
- [FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb), Hugging Face (MinHash settings).
