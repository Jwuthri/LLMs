# 2. Instruction datasets

## Core idea

An instruction dataset is a list of conversations: an instruction and a good answer.
No single public dataset has all that a model needs.
Thus you choose several datasets, check what each one is (size, licence, date, how it was made), and mix them in proportions that fit your goal.

## The example

This page uses the 8 samples of the toy dataset of this track (`examples/datasets/toy_dataset.py`). Each sample is one conversation about the sentence "the cat sat" of the [toy example](../toy-example.md).

| Item | Value |
|---|---|
| Toy dataset | 8 samples: 4 `math`, 2 `code`, 2 `chat` (232 tokens) |
| Training (assumed) | Batch of 16 samples, 1,000 steps |
| Candidate list | 5 public datasets, 9.679 million samples |
| Three sources (assumed) | 4,000,000, 1,000,000 and 100,000 samples |
| Cap (assumed) | 500,000 samples for each source |
| Real users (assumed) | `math` 0.2, `code` 0.3, `chat` 0.5 |

<figure class="diagram" data-src="instruction-datasets/pipeline.svg">
<figcaption>The four parts of this page, with the main number of each part.</figcaption>
</figure>

No dataset is downloaded on this page. The sizes of the real datasets come from the list [LLM Datasets](https://github.com/mlabonne/llm-datasets). The full list is in the [dataset catalogue](../datasets-catalogue.md).

## 1. The families and what each one teaches

### Setup

An **instruction dataset** is a set of conversations that you use to fine-tune a model. A **sample** is one conversation: one or more instructions of a user, each with a good answer. The lesson [Post-training datasets](../scientist/03-datasets.md) shows how to store, make and filter the samples. This page is about the datasets that other persons published.

**Synthetic data** are samples that a model wrote. **Distilled data** are synthetic data that a strong model wrote to train a different, usually smaller, model.

The list LLM Datasets puts the public instruction datasets in 8 families. A **family** is a group of datasets that teach the same type of task.

| Family | What it teaches | One example (samples) |
|---|---|---|
| General | A balanced mix of chat, code and math | open-perfectblend (1.42M) |
| Math | Reasoning in steps to an exact result | NuminaMath-CoT (859k) |
| Science | Physics, chemistry and biology questions | MegaScience (1.25M) |
| Code | To write and correct programs | rStar-Coder (1M) |
| Instruction following | To obey constraints in the prompt, such as a number of paragraphs | ifeval-like-data (56.3k) |
| Multilingual | Answers in other languages than English | aya dataset (204k) |
| Agent and function calling | To call tools with correct arguments | xlam-function-calling-60k (60k) |
| Real conversations | What users ask | lmsys-chat-1m (1M) |

**Function calling** means that the model does not answer with text. It writes the name of a function and its arguments, and a program runs the function: see [Agents](../engineer/05-agents.md).

In the toy dataset, the three topics `math`, `code` and `chat` play the role of three families.

### Steps

1. **Calculate the share of each topic.** The counts are 4, 2 and 2 of 8 samples.

    $$\text{math} = \tfrac{4}{8} = 0.5 \qquad \text{code} = \tfrac{2}{8} = 0.25 \qquad \text{chat} = \tfrac{2}{8} = 0.25$$

    **Why:** the shares, not the names of the datasets, say what the model will learn. A "general" dataset that is 50% math is a math dataset.

2. **Calculate what the model sees in training.** Assume 1,000 steps with a batch of 16 random samples. The model sees 1,000 × 16 = 16,000 samples: 2,000 passes over the 8 samples.

    | Topic | Share | Expected samples |
    |---|---|---|
    | `math` | 0.5 | 8,000 |
    | `code` | 0.25 | 4,000 |
    | `chat` | 0.25 | 4,000 |

    **Why:** each sample moves the weights a small distance. A model learns what it sees often, so `math` gets two times the training of `chat`.

3. **Calculate the number of steps before the model sees 100 samples of the rarest topic.** The rarest topics are `code` and `chat`, with a share of 0.25.

    $$\text{steps} = \frac{100}{16 \times 0.25} = 25$$

    For `math`, the result is 13 steps.

    **Why:** a short training job can end before the model sees much of a rare family. Part 3 does this calculation for a share of 2%.

4. **Calculate how often a batch has no sample of a topic.** The probability that 16 random samples contain no `chat` sample is $0.75^{16} = 0.010$. Thus about 10 of the 1,000 steps have no `chat` sample. For `math`, it is $0.5^{16} = 0.000015$.

    **Why:** a step with no sample of a topic can move the weights away from that topic. The smaller the share, the more often this occurs.

<div class="key-point">

A family is a skill. The share of a family in the training data decides how much training that skill gets, so count the shares before you train.

</div>

### From toy to real

- The list has 6 general datasets, 6 for math, 2 for science, 11 for code, 4 for instruction following, 4 multilingual and 8 for agents and function calling. The largest one is Nemotron-Cascade-2-SFT-Data, with 15.87 million samples.
- The card of open-perfectblend says that it is an open reproduction of the instruction dataset of the paper "The Perfect Blend". It joins 8 public datasets. The largest is MetaMathQA (395,000 samples, math) and the smallest is a set of 57,362 arena conversations. A step that removes duplicates took out 88.1k samples.
- NuminaMath-CoT has about 860k math problems, from Chinese high school exercises to international olympiad problems. Its largest source is `cn_k12`, with 276,591 samples. Each solution is written as a chain of thought: the reasoning steps, then the answer.
- The aya dataset is not synthetic. Its card gives 204k prompts and answers in 65 languages, written by volunteers between May and December 2023.

## 2. Reading a dataset card

### Setup

A **dataset card** is the page that describes a dataset: on the Hugging Face Hub, it is the README file of the dataset. Before you download 4 million samples, read five fields.

| Field | What it is |
|---|---|
| Size | The number of samples |
| **Licence** | What the law permits you to do with the data |
| **Thinking traces** | Reasoning text that the answer contains before the final result |
| Date | When the dataset was published |
| **Provenance** | Where the data come from: which persons or which model wrote them |

A **permissive licence** permits commercial use. The list says that its datasets have permissive licences (Apache 2.0, MIT, CC-BY-4.0) unless a row says otherwise. A **non-commercial licence**, such as CC-BY-NC-4.0, does not permit you to sell a product that uses the data. The list has a "Thinking" column with the values Yes, No or Mixed.

Our candidate list has 5 real datasets. The mean number of tokens for each sample is assumed, because no card gives it.

| Dataset | Samples | Licence | Date |
|---|---|---|---|
| open-perfectblend | 1.42M | Apache-2.0 | Oct 2024 |
| Dolci-Instruct-SFT | 2.15M | CC-BY-NC-4.0 | Dec 2025 |
| SYNTHETIC-2-SFT-verified | 4M | No note in the list | Jun 2025 |
| MegaScience | 1.25M | CC-BY-NC-SA-4.0 | Jul 2025 |
| NuminaMath-CoT | 859k | Apache-2.0 | Jul 2024 |

### Steps

1. **Add the sizes.** 1.42M + 2.15M + 4M + 1.25M + 0.859M = 9.679 million samples.

    **Why:** the size gives the cost of the download, of the storage and of the training. It says nothing about the quality: see [Dataset quality](01-quality.md).

2. **Apply the licence filter for a commercial product.** Two datasets have a non-commercial licence: Dolci-Instruct-SFT and MegaScience. They have 2.15M + 1.25M = 3.4 million samples.

    $$9.679\text{M} - 3.4\text{M} = 6.279\text{M} \qquad \frac{3.4}{9.679} = 0.351$$

    **Why:** the licence is the only field that you cannot repair with more work. If you train a product on non-commercial data, you must train the model again without them.

3. **Calculate the token total.** Assume a mean of 500 tokens for a sample of open-perfectblend, 2,000 for SYNTHETIC-2-SFT-verified and 600 for NuminaMath-CoT.

    | Dataset | Samples × tokens | Tokens |
    |---|---|---|
    | open-perfectblend | 1.42M × 500 | 0.710 billion |
    | SYNTHETIC-2-SFT-verified | 4M × 2,000 | 8.000 billion |
    | NuminaMath-CoT | 859k × 600 | 0.515 billion |
    | Total | | 9.225 billion |

    **Why:** the training cost follows the tokens, not the samples. Two datasets with the same number of samples can have very different costs.

4. **Read the "Thinking" field.** SYNTHETIC-2-SFT-verified has the value Yes. It is 4 / 6.279 = 0.637 of the samples, and 8.000 / 9.225 = 0.867 of the tokens.

    **Why:** a model that trains on thinking traces learns to write its reasoning before each answer. This is good for hard problems, and slow and expensive for a simple chat. Decide which behaviour you want, then keep one format.

5. **Read the date and the provenance.** In November 2026, the three datasets are 25, 17 and 28 months old. The list says that the traces of SYNTHETIC-2-SFT-verified come from the model DeepSeek-R1-0528.

    **Why:** distilled data cannot be better than the model that wrote them, and they copy its style and its errors. An old synthetic dataset comes from an old model.

<div class="key-point">

The size is the first number on a card and the least important one. The licence decides if you can use the dataset. The thinking traces, the date and the provenance decide what the model will copy.

</div>

### From toy to real

- Read the card, not only the list. The list gives no licence note for ifeval-like-data. Its card has the licence name `qwen`, with a link to the licence of the model that wrote the data (Qwen2.5-72B-Instruct).
- The same card has two subsets: 550k rows that are not filtered, and 56,339 rows that passed the IFEval check. The card says that the second one is the one for fine-tuning. The other one has more samples, with errors.
- The card of xlam-function-calling-60k says how the 60,000 samples were made. Two models wrote them, DeepSeek-V2-Chat and Mixtral-8x22B-Inst. Each sample then passed three checks: of the format, of the real execution of the function, and of the meaning. Persons reviewed 600 samples and found a correct rate above 95%.
- Some datasets are **gated**: you must accept conditions before the download. xlam-function-calling-60k (CC-BY-4.0) and lmsys-chat-1m (its own licence agreement) are gated.
- The card of NuminaMath-CoT gives the processing: text recognition from PDF files, a split into problems and solutions, a translation into English, then a rewrite as a chain of thought. Each stage can add errors.

## 3. Mixing datasets

### Setup

A **mix** (or **blend**) is one training set that you build from several sources. A **cap** is the maximum number of samples that you take from one source. **Downsampling** takes a random part of a source. **Upsampling** repeats the samples of a source.
An **epoch** is one pass over the full training set. The **token budget** is the number of tokens that you can pay for: tokens of one epoch × epochs.

Assume three sources of three families, with their mean tokens for each sample.

| Source | Samples | Mean tokens |
|---|---|---|
| General | 4,000,000 | 600 |
| Code | 1,000,000 | 800 |
| Function calling | 100,000 | 400 |

### Steps

1. **Join the sources with no cap.** The mix has 5,100,000 samples.

    $$\frac{4}{5.1} = 0.784 \qquad \frac{1}{5.1} = 0.196 \qquad \frac{0.1}{5.1} = 0.020$$

    With a batch of 16, the model sees 100 function-calling samples only after 319 steps.

    **Why:** the size of a public dataset comes from the work of its authors, not from your goal. Without a cap, the largest source decides what the model becomes.

2. **Put a cap of 500,000 on each source.** The mix is 500,000 + 500,000 + 100,000 = 1,100,000 samples.

    $$\frac{0.5}{1.1} = 0.455 \qquad \frac{0.5}{1.1} = 0.455 \qquad \frac{0.1}{1.1} = 0.091$$

    The model now sees 100 function-calling samples after 69 steps.

    **Why:** the cap downsamples only the large sources. The small family goes from 0.020 to 0.091 of the mix, and you add no data.

3. **Upsample the small source, if you want equal shares.** Repeat each function-calling sample 5 times: 500,000 × 3 = 1,500,000 samples, and each share is 0.333.

    **Why:** upsampling adds no new information. The model sees the same 100,000 samples 5 times in each epoch, so it can memorise them. Prefer more data of that family, if a dataset exists.

4. **Calculate the tokens of one epoch.**

    $$500{,}000 \times 600 + 500{,}000 \times 800 + 100{,}000 \times 400 = 740{,}000{,}000$$

    With no cap, the total is 3.24 billion tokens.

    **Why:** the cap is also a budget tool. It removes 0.772 of the tokens here, from the families that the mix has too much of.

5. **Calculate the training time.** Assume the speed of the lesson [Training libraries](../scientist/10-training-libraries.md): 2,500 tokens/s.

    $$\frac{740{,}000{,}000}{2{,}500} = 296{,}000 \text{ s} = 82.2 \text{ hours} = 3.43 \text{ days}$$

    With no cap, one epoch takes 360 hours: 15 days. Three epochs of the capped mix take 246.7 hours.

    **Why:** calculate the time before you start. If the result does not fit your budget, change the cap, not the number of families.

<figure class="plot">
<img src="/figures/instruction-datasets/cap.svg" alt="Bar chart of the share of each source in the mix, with no cap and with a cap of 500k. General: 0.784 and 0.455. Code: 0.196 and 0.455. Function calling: 0.020 and 0.091.">
<figcaption>The cap cuts the large source and moves the small family from 0.020 to 0.091 of the mix.</figcaption>
</figure>

<div class="llm-widget" data-widget="dataset-mix"></div>

<div class="key-point">

You choose the proportions of a mix. The sizes of the public datasets do not choose them for you. A cap for each source sets the shares and the token budget in one operation.

</div>

### From toy to real

- open-perfectblend is a published mix. Its card lists the number of samples that it takes from each of its 8 sources, from 395,000 down to 57,362. This list is its recipe.
- The authors of that card could not get all the sources of the paper. The instruction-following data come from a different dataset, and the card says that one category of the paper is missing.
- A mix also joins formats. One source has thinking traces and the next has none. One has a system prompt and the next has none. Convert all sources to one chat template before you join them: see [Post-training datasets](../scientist/03-datasets.md).
- The same sample can be in two sources, because public mixes share their sources. Remove duplicates after you join. The tools for this are in [Tools](04-tools.md).

## 4. Real conversations as a check

### Setup

**Real conversations** are logs of what persons typed into a chatbot, with the answers of the model. The **distribution of use cases** is the share of each type of request in these logs.
The **coverage** of a mix is the share of the real requests that are of a type that the mix contains. A request is **out of distribution** when the training data had nothing like it.

Assume that real users ask for `math` 0.2, `code` 0.3 and `chat` 0.5. The toy mix has 0.5, 0.25 and 0.25. To compare them, we use the KL divergence of the lesson [Metrics and losses](../fundamentals/05-metrics.md).

$$\text{KL}(p \,\|\, q) = \sum_i p_i \ln \frac{p_i}{q_i}$$

### Steps

1. **Calculate the gap for each topic.** Mix − real: `math` +0.3, `code` −0.05, `chat` −0.25. The largest gap is 0.3, for `math`.

    **Why:** the gap shows where to act. The mix has too much `math` and half of the `chat` that users ask for.

2. **Calculate the KL divergence in the two directions.**

    $$\text{KL}(\text{real} \,\|\, \text{mix}) = 0.2 \ln\tfrac{0.2}{0.5} + 0.3 \ln\tfrac{0.3}{0.25} + 0.5 \ln\tfrac{0.5}{0.25} = 0.218$$

    $$\text{KL}(\text{mix} \,\|\, \text{real}) = 0.239$$

    **Why:** one number lets you compare two mixes. Use the first direction: it weights each topic by how often users ask for it, so it punishes a mix that lacks what users want.

3. **Re-weight the mix.** The weight of a topic is real ÷ mix: `math` 0.4, `code` 1.2, `chat` 2.0. For a mix of 1,000 samples:

    | Topic | Before | With repeats | No repeats |
    |---|---|---|---|
    | `math` | 500 | 200 | 100 |
    | `code` | 250 | 300 | 150 |
    | `chat` | 250 | 500 | 250 |
    | Total | 1,000 | 1,000 | 500 |

    **Why:** the two results have the real shares, so their KL divergence is 0. The first one repeats each `chat` sample 2 times. The second one repeats nothing, but it keeps only 500 samples, because `chat` is the limit.

4. **Add a topic that the mix does not have.** Assume that 0.1 of the real requests are translations. The real shares become 0.18, 0.27, 0.45 and 0.1, and the mix has 0 for the new topic. The coverage is 0.9. $\text{KL}(\text{real} \,\|\, \text{mix})$ is infinite, and $\text{KL}(\text{mix} \,\|\, \text{real})$ is only 0.345.

    **Why:** no weight can repair a share of 0. These requests are out of distribution, and you need a new source (here, a multilingual dataset). The first direction shows the problem and the second one hides it.

<div class="key-point">

Public datasets show what researchers built. Real conversations show what users ask. Compare the two distributions, and add data where the mix has a hole.

</div>

### From toy to real

- lmsys-chat-1m has 1,000,000 conversations with 25 models, from 210,479 users in 154 languages. They were collected from April to August 2023 on the Vicuna demo and the Chatbot Arena website.
- WildChat comes from a chatbot that gave free access to ChatGPT to users who agreed to share their conversations. The first release had 1 million conversations. The card of WildChat-4.8M says that the public release has 3,199,860 conversations: the authors removed 1,543,476 toxic ones from the full set. This is why the list gives a size of 3.2M for a dataset with "4.8M" in its name.
- The users of a free demo are not your users. The best distribution of use cases is the one of your own product.
- Use these datasets as a measure first, not as training data. The answers come from older models: the card of lmsys-chat-1m says that unsafe conversations were kept on purpose. The same card says that the data can contain test questions of popular benchmarks, which is a risk for [evaluation](../scientist/06-evaluation.md).
- arena-human-preference-140k has 136k conversations in which a user chose the better of two answers. This is preference data: see [Preference datasets](03-preference-datasets.md).

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import numpy as np

# 1. Shares of the toy topics, and the steps to see 100 samples of the rarest one
counts = np.array([4, 2, 2])                           # math, code, chat
mix = counts / counts.sum()                            # [0.5, 0.25, 0.25]
steps = int(np.ceil(100 / (16 * mix.min())))           # 25

# 2. Licence filter on the candidate list (samples, commercial use permitted?)
cards = [(1_420_000, True), (2_150_000, False), (4_000_000, True), (1_250_000, False), (859_000, True)]
kept = sum(size for size, commercial in cards if commercial)   # 6,279,000 of 9,679,000

# 3. Cap each source, then count the tokens and the time
sizes = np.array([4_000_000, 1_000_000, 100_000])
capped = np.minimum(sizes, 500_000)                    # [500000, 500000, 100000]
shares = capped / capped.sum()                         # [0.455, 0.455, 0.091]
tokens = (capped * np.array([600, 800, 400])).sum()    # 740,000,000
hours = tokens / 2_500 / 3600                          # 82.2

# 4. Compare with real conversations, then re-weight
real = np.array([0.2, 0.3, 0.5])
kl = np.sum(real * np.log(real / mix))                 # 0.218
weights = real / mix                                   # [0.4, 1.2, 2.0]
```

Real-library sketch (the tests do not run this code). It reads the first samples of a dataset without a full download.

```python
from datasets import load_dataset

rows = load_dataset("mlabonne/open-perfectblend", split="train", streaming=True)
for row in rows.take(5):
    print(row)
```

## Comparison

| Choice | Toy result | Good for | Risk |
|---|---|---|---|
| Join all sources, no cap | Shares 0.784, 0.196, 0.020; 360 hours | The most data | The largest source decides the model |
| Cap for each source | Shares 0.455, 0.455, 0.091; 82.2 hours | Balance and a smaller budget | You discard good samples of the large source |
| Cap and upsampling | Shares 0.333 each; 100 hours | A family that must not be rare | The model memorises the repeated samples |
| Re-weight to the real use | KL from 0.218 to 0 | A product with known users | You copy the habits of the users of one demo |
| Licence filter | 6.279M of 9.679M samples kept | A commercial product | You lose a full family (here, science) |

## Checks and common misconceptions

- **"The largest dataset is the best one."** The size is the number of samples, not their quality or their fit. Check: read 20 random samples, and calculate the share of each family in your mix.
- **"It is on the Hub, so I can use it."** A public dataset can have a non-commercial licence, a gate or the licence of a model. Check: read the licence field of the card itself, and the licence of each source of a mix.
- **"A general dataset covers all cases."** A general mix has the families that its authors chose. Check: compare its shares with real conversations, and look for a share of 0.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no page on this topic. The list of datasets and their families come from [LLM Datasets](https://github.com/mlabonne/llm-datasets) by the same author. The text and the example are new. Each statement about a real dataset was read in the linked source on 11 November 2026.

- [LLM Datasets](https://github.com/mlabonne/llm-datasets), Maxime Labonne (the 8 families, the number of datasets in each family, the sizes, dates, licence notes and "Thinking" values of the rows, the description of instruction following, the model behind SYNTHETIC-2-SFT-verified).
- [open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend), dataset card (Apache-2.0, reproduction of "The Perfect Blend", the 8 sources and their sizes, 88.1k duplicates removed, the missing parts).
- [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), dataset card (Apache-2.0, about 860k problems, the sources, the processing stages).
- [aya dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset), dataset card (204k pairs, 65 languages, volunteers, May to December 2023).
- [ifeval-like-data](https://huggingface.co/datasets/argilla/ifeval-like-data), dataset card (the licence name, the two subsets and their sizes, the model that wrote the data).
- [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), dataset card (CC-BY-4.0, the gate, the two models, the three checks, the human review).
- [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m), dataset card (the gate and the licence agreement, 25 models, 210,479 users, 154 languages, the period, the unsafe conversations, the benchmark questions).
- [WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M), dataset card (3,199,860 conversations, 1,543,476 toxic conversations removed).
- [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://arxiv.org/abs/2405.01470), Zhao et al., 2024 (abstract only: free access for users who agreed, 1 million conversations).
