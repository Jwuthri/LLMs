# 4. From raw text to a dataset: the tools

## Core idea

You build a dataset in a loop of four steps: collect raw text, filter it, generate what is missing, and look at the result.
Each step has tools, and the steps are simple.
The work is in the numbers: how much each step removes, what it costs, and what it removes by error.

## The example

This page uses the toy dataset of the track: 8 samples about the sentence "the cat sat". Lesson 1, [What is a good dataset?](01-quality.md), shows them in full.

| Samples | Topic | Note |
|---|---|---|
| 1, 2, 3, 4 | math | Sample 3 is sample 1 with a full stop: a near duplicate. Sample 4 has a wrong answer |
| 5, 6 | code | |
| 7, 8 | chat | Sample 8 is a refusal: "I cannot help with that." |

Each sample has an assumed embedding of 2 numbers. All the values are assumed, and no model is called.
This project did not run the tools of this page: each statement about a tool comes from its documentation.

<figure class="diagram" data-src="dataset-tools/loop.svg">
<figcaption>The four steps with the toy counts. The last step sends you back to the first.</figcaption>
</figure>

## 1. Scraping: from pages and PDFs to text

### Setup

**Scraping** is the automatic collection of documents, for example web pages. A page is not clean text. It contains **boilerplate**: menus, footers and advertisements.
**Extraction** keeps the main text and removes the rest. The **yield** is the share of the raw characters that stay.

Assume 5 web pages with these character counts.

| Page | Raw characters | Clean characters |
|---|---|---|
| Long article | 40,000 | 12,000 |
| Short article | 25,000 | 5,000 |
| Forum thread | 60,000 | 3,000 |
| Login page | 15,000 | 300 |
| Documentation page | 10,000 | 4,700 |

### Steps

1. **Calculate the yield of each page.** Divide the clean characters by the raw characters.

    $$\text{yield} = \frac{\text{clean characters}}{\text{raw characters}} = \frac{12{,}000}{40{,}000} = 0.3$$

    The 5 yields are 0.3, 0.2, 0.05, 0.02 and 0.47.

    **Why:** the yield tells you what a page is. A page with a yield of 0.02 is almost only boilerplate.

2. **Calculate the total yield.** The 5 pages have 150,000 raw characters and 25,000 clean characters: a yield of 0.167.

    **Why:** you pay for the download and the storage of the raw characters, but you train on the clean ones. Here, 5 of 6 characters are lost.

3. **Drop the pages that are too short.** Assume the rule "drop a page with less than 500 clean characters". The login page goes out. 4 pages and 24,700 characters stay.

    **Why:** the rule removes 1 page of 5, but only 1.2% of the clean text.

4. **Estimate the size of a real crawl.** Assume 1,000,000 pages that are like these 5. Then 80% stay: 800,000 pages with 6,175 clean characters each. At 4 characters for each token (the rule of thumb of [the architecture lesson](../scientist/01-architecture.md)), one page gives 1,543.75 tokens.

    $$800{,}000 \times 1{,}543.75 = 1{,}235{,}000{,}000 \text{ tokens}$$

    **Why:** the tokens that you get decide how many pages you must collect.

<div class="key-point">

Scraping is a problem of yield. Measure the yield on a small set of pages, then calculate the number of pages that you need.

</div>

### From toy to real

- **Trafilatura.** A Python and command-line tool that gets the text and the metadata of web pages. Its extractor uses rules. It writes plain text, Markdown, CSV, JSON or XML. Take it for web pages. Limit: its input is HTML, from a URL or from a file that you downloaded. The licence is Apache-2.0 from version 1.8.0, and GPLv3+ before. Last release: October 2026.
- **Marker.** It changes PDF files into Markdown, JSON or HTML. It uses models, and it runs on a GPU, a CPU or Apple silicon. An option calls an LLM to increase the accuracy. Take it for papers and books. Limits: its README says that very complex layouts can fail. The code is Apache-2.0, but the model weights are free only for research, personal use and companies below 5 million dollars of funding or revenue. The link now opens `datalab-to/marker`. Last release: October 2026.

## 2. Filtering: rules and duplicates

### Setup

A **rule-based filter** removes a sample when a simple test is true, for example "the answer contains these words". A **refusal** is an answer in which the model declines the task.
An **exact duplicate** is a sample with the same text as another sample, character for character. A **near duplicate** (or **fuzzy duplicate**) has almost the same text.
To find near duplicates, we compare embeddings with the **cosine similarity**: 1 means the same direction, 0 means no relation.

$$\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}$$

A pair is a duplicate when its similarity is at or above a **threshold**. A **false positive** of a filter is a good sample that the filter removes.
[Post-training datasets](../scientist/03-datasets.md) shows MinHash, a method that compares words and not embeddings.

### Steps

1. **Remove the refusals with a key-word rule.** The rule: remove a sample if its answer contains "I cannot". Sample 8 goes out. 7 samples stay.

    **Why:** a model that trains on refusals learns to refuse. One line of code finds the usual ones.

2. **Find a false positive of the rule.** Assume a ninth sample with the answer `The cat says: "I cannot sleep."` The rule removes it too. Of the 2 samples that the rule removes, 1 is a real refusal: a precision of 0.5.

    **Why:** a rule reads words, not meaning. The rule also does not see the wrong answer of sample 4: that needs a checker.

3. **Search for exact duplicates.** No two instructions are equal. Samples 1 and 3 differ by one character, the full stop.

    **Why:** exact deduplication is cheap (one hash for each sample), but one character defeats it.

4. **Calculate the cosine similarity of each pair.** For samples 1 and 3: the dot product is 0.992, and the lengths are 1.005 and 0.987.

    $$\cos = \frac{0.992}{1.005 \times 0.987} = 0.9998$$

    This step uses 4 decimals, because the values are near 1. The highest of the 28 pairs:

    | Pair | 1 and 3 | 7 and 8 | 2 and 3 | 1 and 2 | 5 and 6 | 2 and 4 | 1 and 4 |
    |---|---|---|---|---|---|---|---|
    | Similarity | 0.9998 | 0.9971 | 0.9953 | 0.9929 | 0.9929 | 0.9701 | 0.9345 |

    Samples 1 (math) and 5 (code) have 0.198.

    **Why:** an embedding puts texts with the same meaning near each other. The full stop of sample 3 changes almost nothing.

5. **Remove the duplicates at three thresholds.** Read the samples in order. Keep a sample only if no kept sample is at or above the threshold.

    | Threshold | Pairs at or above | Removed | Kept |
    |---|---|---|---|
    | 0.999 | 1 | Sample 3 | 7 |
    | 0.99 | 5 | Samples 2, 3, 6, 8 | 4 |
    | 0.95 | 6 | Samples 2, 3, 6, 8 | 4 |

    At 0.9, sample 4 also goes out and 3 samples stay: one for each topic.

    **Why:** at 0.99, 3 of the 4 removed samples are false positives. Samples 2 and 6 are different tasks on the same topic. A threshold that is too low removes variety.

6. **Count the comparisons.** To compare all the pairs of $N$ samples, you need:

    $$\frac{N(N-1)}{2} \text{ comparisons}$$

    | Samples | Comparisons | Time at 10 million comparisons for each second (assumed) |
    |---|---|---|
    | 8 | 28 | Almost zero |
    | 10,000 | 49,995,000 | 5.0 s |
    | 1,000,000 | 499,999,500,000 | 13.9 hours |

    **Why:** 100 times more samples need about 10,000 times more comparisons. Thus the tools use an **index**: a structure that gives, for each sample, a few candidates that are probably near. With 10 candidates for each of 1,000,000 samples, you make 10,000,000 comparisons: about 50,000 times fewer. See the [vector storage lesson](../engineer/02-vector-storage.md).

<figure class="plot">
<img src="/figures/dataset-tools/kept-against-threshold.svg" alt="Step curve of the samples kept against the similarity threshold. 0.999: 7 kept. 0.99: 4 kept. 0.95: 4 kept. 0.9: 3 kept.">
<figcaption>Between 0.999 and 0.99, the count falls from 7 to 4. A small change of the threshold removes half of the dataset.</figcaption>
</figure>

<div class="llm-widget" data-widget="dedup-threshold"></div>

<div class="key-point">

Each filter makes errors of two types: it keeps bad samples, and it removes good ones. The threshold moves you between the two. Read what a filter removes before you trust its count.

</div>

### From toy to real

The lesson continues with the rule and a threshold of 0.999: samples 8 and 3 go out, and 6 stay.

- **Rule-based filtering.** A technique, not a tool: a few lines of Python. Take it first. Limit: the false positives of step 2.
- **NeMo Curator** (NVIDIA). A Python toolkit that filters and deduplicates text, images, video and audio on GPUs, with RAPIDS and Ray. Take it for a very large corpus on NVIDIA GPUs. Its README reports a fuzzy deduplication in 0.65 hours on 3 H100 nodes, against 10.7 hours on CPUs. Limits: the GPU path needs Linux, CUDA 12 and about 16 GB of GPU memory. The link now opens `NVIDIA-NeMo/Curator`. Licence: Apache-2.0. Last release: October 2026.
- **DataTrove** (Hugging Face). A Python library of pipeline blocks: readers, extractors (Trafilatura is one), filters, deduplication (MinHash, sentences, exact substrings) and writers. A pipeline runs on one machine, on Slurm or on Ray. Take it for a large text corpus on CPUs. Limit: it does not split a file, so one large file is one task. Licence: Apache-2.0. Last release: September 2026.
- **Dolma** (Allen Institute for AI). The toolkit of the Dolma pre-training dataset (3 trillion tokens). It has taggers (functions that put labels on documents) and a fast deduplication in Rust. You use commands such as `dolma tag` and `dolma dedupe`. Take it for pre-training data. Limit: its last release is from July 2025.
- **SemHash.** A small Python library for fuzzy deduplication. It uses fast embeddings (Model2Vec) and an approximate index (USearch). Its README reports 1.8 million records in about 74 seconds on a CPU. It also shows the removed pairs that are the least similar. Take it for one dataset on one machine. Limit: other data types than text need your own encoder. Licence: MIT. Last release: September 2026.
- **Argilla.** A web interface, with a Python client, in which people read, label and correct samples. Take it for manual review. Limit: a notice in its repository says that the original authors left and that the project gets bug fixes but no new features. Its last release is from March 2025.

## 3. Generation: make what is missing

### Setup

After the filters, the 6 samples are 3 math, 2 code and 1 chat. The dataset has too little chat.
**Synthetic data generation** uses an LLM to write new samples. A **seed** is a short input, for example a topic, that makes one sample different from the others. A **prompt template** is a text with holes for the seed.
The tool sends many prompts together, in a **batch**. [Post-training datasets](../scientist/03-datasets.md) shows how to build the prompt and how to validate the output.

### Steps

1. **Count what is missing.** The largest topic has 3 samples. To make the 3 topics equal, code needs 1 sample and chat needs 2.

    $$k = (3 - 3) + (3 - 2) + (3 - 1) = 3 \text{ new samples}$$

    **Why:** generation has a target. Without a count, you generate more of what you have.

2. **Add the samples that the checks will reject.** Assume a keep rate of 60%: 6 of 10 generated samples pass the filters of part 2 and a checker.

    $$\text{to generate} = \left\lceil \frac{3}{0.6} \right\rceil = 5$$

    **Why:** a generated sample is raw data. It goes through the same filters as scraped text, and some go out.

3. **Calculate the tokens at a real size.** Assume that a project needs 10,000 new samples, with 300 prompt tokens and 500 output tokens for each sample. You generate 16,667 samples: 5,000,100 input tokens and 8,333,500 output tokens.

    **Why:** the tokens give the time and the price before the first call. Each kept sample costs 833.3 output tokens, not 500.

4. **Calculate the time, one by one and in batches.** Use the speed model of the [training libraries lesson](../scientist/10-training-libraries.md): one decode step for $B$ requests takes 19.8 ms + 0.2 ms × $B$. One by one, the engine gives 50 tokens for each second: 46.3 hours. With 64 requests in the batch, it gives 1,963 tokens for each second: 70.7 minutes.

    **Why:** the batch makes the same job 39.3 times shorter. A generation tool fills the batches, retries the failures and keeps the results.

5. **Calculate the cost for each sample.** Assume an API price of 0.5 dollars for each million input tokens and 2 dollars for each million output tokens. The job costs 19.17 dollars: 0.002 dollars for each kept sample. Assume a batch API with a 50% discount: 9.58 dollars.

    **Why:** the **cost for each sample** includes the rejected samples. A low keep rate makes each kept sample more expensive.

<div class="key-point">

Generate to fill a gap that you counted, and divide by the keep rate. The batch decides the time: 70.7 minutes or 46.3 hours for the same tokens.

</div>

### From toy to real

- **Curator** (Bespoke Labs). A Python library: you write a class with a `prompt` function and a `parse` function, and you call it on a dataset. It gives structured outputs, retries and a cache, and it can use the batch APIs of the providers. Its README says that several providers give about 50% discount in batch mode. Take it for a pipeline of API calls in Python. Limit: it collects anonymous usage telemetry unless you set a variable. Licence: Apache-2.0. Last release: October 2026.
- **Distilabel.** A Python framework: a pipeline of steps and tasks, with many LLM providers, for SFT and preference data. Take it for pipelines that follow research papers. Limit: a notice in its repository says: "The original authors have moved on to other projects." Community members now maintain it. Its last release is from January 2025.
- **Augmentoolkit.** It changes your documents into a domain dataset, and it can also train a model. You use a web interface or a command line. It can generate with a local model. Take it to make a domain expert from documents. Limits: it asks for Python 3.11, and its README says that version 3.0 probably has new bugs. Licence: MIT.
- **Data Prep Kit** (IBM). A set of modules that change files into Parquet, deduplicate, filter, cut into chunks and remove personal data, on Python or Ray. Take it to prepare documents before the generation. Limit: its README lists no module that writes new samples with an LLM. The link now opens `data-prep-kit/data-prep-kit`. Licence: Apache-2.0. Last release: June 2026.

## 4. Exploration: look at the result, then do it again

### Setup

**Exploration** is to look at the dataset as a whole: its groups, its gaps and its strange samples.
An **embedding map** is a picture with one point for each sample, where similar samples are near each other. A **cluster** is a group of points that are near. An **outlier** is a point that is far from the centre of its group.
We put the 8 embeddings in 2 clusters with **k-means** for k = 2: give each point to the nearest centre, move each centre to the mean of its points, and repeat. The start centres are samples 1 and 5.

### Steps

1. **Run 2-means.** After one move of the centres, no point changes its cluster.

    | Cluster | Centre | Samples | Size |
    |---|---|---|---|
    | A | [0.836, 0.264] | 1, 2, 3, 4, 8 | 5 |
    | B | [0.3, 0.867] | 5, 6, 7 | 3 |

    **Why:** the clusters show what the dataset contains, without labels. Here the 4 math samples are together, and chat has no cluster of its own: its 2 samples are in different clusters.

2. **Find the sample that is farthest from its centre.** Sample 8 is at a distance of 0.411 from the centre of A. The next one is sample 7, at 0.343 from the centre of B.

    **Why:** an outlier is where you read first. Sample 8 is the refusal. The map found it without a rule.

3. **Write the funnel of the lesson.** A **funnel** is the count after each step.

    | Step | Samples | Math | Code | Chat |
    |---|---|---|---|---|
    | Collected | 8 | 4 | 2 | 2 |
    | After the rules | 7 | 4 | 2 | 1 |
    | After deduplication | 6 | 3 | 2 | 1 |
    | After generation | 9 | 3 | 3 | 3 |

    **Why:** the funnel shows where the samples go. The share of math falls from 0.5 to 0.333. The share of chat rises from 0.25 to 0.333.

4. **Start the loop again.** Run the same exploration on the 9 samples. If the 3 new samples are in one tight group, the generator repeats itself: change the seeds.

    **Why:** each step changes the dataset, and you see the effect only when you look.

<div class="key-point">

Exploration closes the loop. Counts tell you how many samples each step removed. Only reading tells you if they were the right ones.

</div>

### From toy to real

- **Nomic Atlas.** A hosted service with a Python client (the `nomic` package). You send your data or your embeddings, and you get a map in the browser that you can search, label and share. Atlas groups the points into topics and finds duplicates. Its README gives a range from hundreds to tens of millions of points. Take it for a map of a large dataset. Limits: you need a Nomic account and an access token, and the map is a project in the service, not on your machine. The last release of the client is from November 2025.
- **Reading samples by hand.** Read 50 random samples, then the outliers, then the samples that each filter removed. Argilla (part 2) gives an interface for it.

[The datasets catalogue](../datasets-catalogue.md) lists public datasets that went through these steps.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import math
import numpy as np

from examples.datasets.toy_dataset import SAMPLES, embeddings

# 1. Yield
raw, clean = np.array([40_000, 25_000, 60_000, 15_000, 10_000]), np.array([12_000, 5_000, 3_000, 300, 4_700])
total_yield = clean.sum() / raw.sum()                           # 0.167
tokens = 1_000_000 * 0.8 * clean[clean >= 500].mean() / 4       # 1,235,000,000

# 2. Rule, then fuzzy deduplication
refusals = [s.id for s in SAMPLES if "i cannot" in s.answer.lower()]   # [8]
E = embeddings()
unit = E / np.linalg.norm(E, axis=1, keepdims=True)
sim = unit @ unit.T                                             # sim[0, 2] = 0.9998

def deduplicate(threshold):
    kept = []
    for i in range(len(sim)):
        if all(sim[i, j] < threshold for j in kept):
            kept.append(i)
    return [SAMPLES[i].id for i in kept]

deduplicate(0.999)                                              # [1, 2, 4, 5, 6, 7, 8]
deduplicate(0.99)                                               # [1, 4, 5, 7]
pairs = 1_000_000 * 999_999 // 2                                # 499,999,500,000

# 3. Generation
to_generate = math.ceil(10_000 / 0.6)                           # 16,667
speed = 64 / ((19.8 + 0.2 * 64) / 1000)                         # 1,963 tokens/s
minutes = to_generate * 500 / speed / 60                        # 70.7

# 4. Two clusters (one move of the centres is enough here)
centres = E[[0, 4]]
for _ in range(2):
    labels = np.linalg.norm(E[:, None] - centres[None], axis=2).argmin(axis=1)
    centres = np.array([E[labels == k].mean(axis=0) for k in range(2)])
```

Real-library sketch (the tests do not run this code):

```python
from semhash import SemHash

texts = [sample.instruction for sample in SAMPLES]
result = SemHash.from_records(records=texts).self_deduplicate()
kept = result.selected                                          # the samples that stay
edge_cases = result.get_least_similar_from_duplicates(n=5)      # read these before you trust the threshold
result.rethreshold(0.95)                                        # change the threshold without a new search
```

## Comparison

Steps 1 and 2:

| Step | Tool | Best for | Limits | How you use it |
|---|---|---|---|---|
| Scrape | Trafilatura | Text of web pages | HTML input only | Python, CLI |
| Scrape | Marker | PDF to Markdown | Complex layouts; licence of the weights | CLI, Python |
| Filter | Rules | The first filter | False positives | Your own Python |
| Filter | NeMo Curator | Very large corpus on GPUs | Linux, CUDA 12, 16 GB GPU | Python |
| Filter | DataTrove | Large text corpus on CPUs | One file is one task | Python |
| Filter | Dolma | Pre-training data | Release of July 2025 | CLI, config file |
| Filter | SemHash | Fuzzy duplicates, one machine | Text by default | Python |
| Filter | Argilla | Manual review | Bug fixes only | Web UI, Python |

Steps 3 and 4:

| Step | Tool | Best for | Limits | How you use it |
|---|---|---|---|---|
| Generate | Curator | API calls in batches | Telemetry by default | Python |
| Generate | Distilabel | Pipelines for SFT and preference data | Community maintenance; release of January 2025 | Python |
| Generate | Augmentoolkit | Documents to a domain dataset | Python 3.11; new bugs | Web UI, CLI, YAML |
| Generate | Data Prep Kit | Preparing documents | No generation with an LLM | Python, Ray |
| Explore | Nomic Atlas | Map of a large dataset | Hosted; account | Python, web UI |

## Checks and common misconceptions

- **"Deduplication only removes copies."** With a threshold of 0.99, it removes 4 of the 8 toy samples, and 3 of them are not copies. Check: read the removed pairs with the lowest similarity. If they are different tasks, increase the threshold.
- **"A filter with no errors in my test is safe."** The refusal rule is correct on the 8 samples and wrong on the ninth. Check: for each filter, read 50 samples that it removed and 50 that it kept.
- **"The tool that the tutorial uses is the tool to take."** Two tools of this page have a notice that their authors left, and two repositories moved. Check: read the README, the last release date and the licence.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no page on this topic. The four steps and the list of tools come from [LLM Datasets](https://github.com/mlabonne/llm-datasets) by the same author. The text and the example are new. This project did not run these tools: each statement about a tool was read in its linked documentation or repository on 11 November 2026.

- [Trafilatura repository](https://github.com/adbar/trafilatura) (the formats, the inputs, the licences, release 2.3.1 of 6 October 2026).
- [Marker repository](https://github.com/VikParuchuri/marker), which opens `datalab-to/marker`, (the formats, the hardware, the limits, the two licences, release 2.1.0 of 29 October 2026).
- [NeMo Curator repository](https://github.com/NVIDIA/NeMo-Curator), which opens `NVIDIA-NeMo/Curator`, (the engines, the GPU requirements, the 10.7 and 0.65 hours, release 1.4.0 of 9 October 2026).
- [DataTrove repository](https://github.com/huggingface/datatrove) (the blocks, the executors, the file limit, release 0.10.1 of 30 September 2026).
- [Dolma repository](https://github.com/allenai/dolma) and its [getting-started guide](https://github.com/allenai/dolma/blob/main/docs/getting-started.md) (the 3 trillion tokens, the taggers, the commands, release 1.2.1 of 7 July 2025).
- [SemHash repository](https://github.com/MinishLab/semhash) (Model2Vec, USearch, the 74 seconds, the functions of the sketch, release 0.5.0 of 30 September 2026).
- [Argilla website](https://argilla.io/) and [repository](https://github.com/argilla-io/argilla) (the notice about the authors, release 2.8.0 of March 2025).
- [Curator repository](https://github.com/bespokelabsai/curator/) (the two functions, the batch mode and the 50%, the telemetry, release 0.1.31 of 25 October 2026).
- [Distilabel repository](https://github.com/argilla-io/distilabel) (the quote, the community maintenance, release 1.5.3 of 28 January 2025).
- [Augmentoolkit repository](https://github.com/e-p-armstrong/augmentoolkit) (the interfaces, the local model, Python 3.11, the note on bugs).
- [Data Prep Kit repository](https://github.com/IBM/data-prep-kit), which opens `data-prep-kit/data-prep-kit` (the modules, the runtimes, release 1.1.8 of 22 June 2026).
- [Nomic repository](https://github.com/nomic-ai/nomic) (the Atlas functions, the account, release 3.9.0 of 11 November 2025).
- The [PyPI](https://pypi.org/) pages of `marker-pdf`, `nemo-curator`, `datatrove`, `dolma`, `semhash`, `argilla`, `distilabel`, `bespokelabs-curator` and `data-prep-toolkit` (the release dates).
