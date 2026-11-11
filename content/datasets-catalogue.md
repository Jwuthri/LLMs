# Dataset catalogue

This page lists the public datasets and the tools of [LLM Datasets](https://github.com/mlabonne/llm-datasets), a list that Maxime Labonne keeps up to date. It is a copy of the facts of that list as it was on 11 November 2026: names, sizes, dates and licence notes. The list changes often, so read the original before you choose a dataset.

The lessons of the track explain how to use this page: [What is a good dataset?](datasets/01-quality.md), [Instruction datasets](datasets/02-instruction-datasets.md), [Preference datasets](datasets/03-preference-datasets.md) and [From raw text to a dataset: the tools](datasets/04-tools.md).

## How to read the tables

- **Samples** is the number of samples (conversations or pairs) that the list gives. This site did not download or count any dataset. The number on a dataset card can differ from the number of the list.
- **Thinking** says if the answers contain reasoning text before the final answer: Yes, No or Mixed.
- **Licence note** is empty when the list flags nothing. The list says that, unless it says otherwise, the datasets have permissive licences (Apache 2.0, MIT, CC-BY-4.0 and similar). Always read the licence on the dataset card before you train a model for commercial use: for example, the list has no note for ifeval-like-data, and its card names the licence of the model that wrote the data.
- Each name is a link to the dataset on Hugging Face.

## General

Mixtures of chat, code and mathematics for a general model. See the lesson [Instruction datasets](datasets/02-instruction-datasets.md).

| Dataset | Samples | Thinking | Date | Licence note |
|---|---|---|---|---|
| [**Nemotron-Cascade-2-SFT-Data**](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) | 15.87M | Mixed | Mar 2026 | NVIDIA licence |
| [**SYNTHETIC-2-SFT-verified**](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-SFT-verified) | 4M | Yes | Jun 2025 |  |
| [**Dolci-Instruct-SFT**](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) | 2.15M | No | Dec 2025 | CC-BY-NC-4.0 |
| [**open-perfectblend**](https://huggingface.co/datasets/mlabonne/open-perfectblend) | 1.42M | No | Oct 2024 |  |
| [**orca-agentinstruct-1M-v1**](https://huggingface.co/datasets/mlabonne/orca-agentinstruct-1M-v1-cleaned) | 1.05M | No | Nov 2024 |  |
| [**KIMI-K2.5-1000000x**](https://huggingface.co/datasets/ianncity/KIMI-K2.5-1000000x) | 693k | Yes | Apr 2026 |  |

## Math

Problems with step-by-step reasoning. See the lesson [Instruction datasets](datasets/02-instruction-datasets.md).

| Dataset | Samples | Thinking | Date | Licence note |
|---|---|---|---|---|
| [**MathX-5M**](https://huggingface.co/datasets/Modotte/MathX-5M) | 5.05M | Yes | Feb 2026 |  |
| [**Nemotron-SFT-Math-v3**](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v3) | 1.24M | Yes | Mar 2026 | NVIDIA licence |
| [**Nemotron-Math-Proofs-v1**](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1) | 925k | Yes | Dec 2025 | CC-BY-SA-4.0 |
| [**OpenThoughts3-1.2M**](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) | 1.2M | Yes | Jun 2025 |  |
| [**NuminaMath-CoT**](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 859k | Yes | Jul 2024 |  |
| [**AM-Thinking-v1-Distilled (Math)**](https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled/blob/main/math.jsonl) | 558k | Yes | May 2025 |  |

## Science

Questions in physics, chemistry and biology, with much reasoning. See the lesson [Instruction datasets](datasets/02-instruction-datasets.md).

| Dataset | Samples | Thinking | Date | Licence note |
|---|---|---|---|---|
| [**MegaScience**](https://huggingface.co/datasets/MegaScience/MegaScience) | 1.25M | Yes | Jul 2025 | CC-BY-NC-SA-4.0 |
| [**Nemotron-Science-v1**](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1) | 226k | Yes | Dec 2025 | CC-BY-4.0 |

## Code

Samples to understand, write and analyse code in many programming languages. See the lesson [Instruction datasets](datasets/02-instruction-datasets.md).

| Dataset | Samples | Thinking | Date | Licence note |
|---|---|---|---|---|
| [**CodeX-7M-Non-Thinking**](https://huggingface.co/datasets/Modotte/CodeX-7M-Non-Thinking) | 7.36M | No | Feb 2026 |  |
| [**Ling-Coder-SFT**](https://huggingface.co/datasets/inclusionAI/Ling-Coder-SFT) | 4.48M | No | Mar 2025 |  |
| [**Nemotron-SFT-Competitive-Programming-v2**](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2) | 845k | Yes | Mar 2026 | NVIDIA licence |
| [**Nemotron-SFT-OpenCode-v1**](https://huggingface.co/datasets/nvidia/Nemotron-SFT-OpenCode-v1) | 459k | No | Mar 2026 | NVIDIA licence |
| [**Nemotron-SFT-SWE-v2**](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2) | 256k | Yes | Mar 2026 | NVIDIA licence |
| [**rStar-Coder**](https://huggingface.co/datasets/microsoft/rStar-Coder) | 1M | Yes | May 2025 |  |
| [**opc-sft-stage2**](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2) | 436k | No | Nov 2024 |  |
| [**AM-Thinking-v1-Distilled (Code)**](https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled/blob/main/code.jsonl) | 324k | Yes | May 2025 |  |
| [**CodeFeedback-Filtered-Instruction**](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | 157k | No | Feb 2024 |  |
| [**open-r1/codeforces**](https://huggingface.co/datasets/open-r1/codeforces) | 29k | Yes | Feb 2025 | CC-BY-4.0 |
| [**synthetic_tex_to_sql**](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) | 100k | No | Apr 2024 |  |

The list writes the last name as "synthetic_tex_to_sql". Its link goes to `gretelai/synthetic_text_to_sql`.

## Instruction following

Samples in which the answer must obey a constraint of the prompt, for example a number of paragraphs or a language. See the lesson [Instruction datasets](datasets/02-instruction-datasets.md).

| Dataset | Samples | Thinking | Date | Licence note |
|---|---|---|---|---|
| [**Nemotron-SFT-Instruction-Following-Chat-v2**](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2) | 2M | Mixed | Mar 2026 | NVIDIA licence |
| [**AutoIF-instruct-61k-with-funcs**](https://huggingface.co/datasets/Post-training-Data-Flywheel/AutoIF-instruct-61k-with-funcs) | 61.5k | No | Oct 2024 |  |
| [**ifeval-like-data**](https://huggingface.co/datasets/argilla/ifeval-like-data) | 56.3k | No | Oct 2024 |  |
| [**tulu-3-sft-personas-instruction-following**](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) | 30k | No | Nov 2024 |  |

## Multilingual

Instruction samples in many languages. See the lesson [Instruction datasets](datasets/02-instruction-datasets.md).

| Dataset | Samples | Thinking | Date | Licence note |
|---|---|---|---|---|
| [**Nemotron-SFT-Multilingual-v1**](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v1) | 3.07M | Yes | Mar 2026 | NVIDIA licence |
| [**luth-sft**](https://huggingface.co/datasets/kurakurai/luth-sft) | 570K | No | Aug 2025 |  |
| [**aya dataset**](https://huggingface.co/datasets/CohereForAI/aya_dataset) | 204k | No | Feb 2024 |  |
| [**M2Lingual**](https://huggingface.co/datasets/ServiceNow-AI/M2Lingual) | 175K | No | Jun 2024 |  |

## Agent and function calling

Samples in which the model calls a function with arguments that it takes from the prompt. See the lesson [Instruction datasets](datasets/02-instruction-datasets.md).

| Dataset | Samples | Thinking | Date | Licence note |
|---|---|---|---|---|
| [**AgentTrove**](https://huggingface.co/datasets/open-thoughts/AgentTrove) | 1.7M | No | Apr 2026 |  |
| [**Nemotron-SFT-Agentic-v2**](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) | 992k | Mixed | Mar 2026 | NVIDIA licence |
| [**ToolMind**](https://huggingface.co/datasets/Nanbeige/ToolMind) | 369k | Yes | Nov 2025 |  |
| [**xlam-function-calling-60k**](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60k | No | Jun 2024 |  |
| [**FunReason-MT**](https://huggingface.co/datasets/Bingguang/FunReason-MT) | 17k | Yes | Oct 2025 |  |
| [**hermes-function-calling-v1**](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | 11.6k | No | Aug 2024 |  |
| [**ToolACE**](https://huggingface.co/datasets/Team-ACE/ToolACE) | 11.3k | No | Aug 2024 |  |
| [**APIGen-MT-5k**](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) | 5k | No | Apr 2025 | CC-BY-NC-4.0 |

## Real conversations

Conversations between real users and models. They show what people ask. See the lesson [Instruction datasets](datasets/02-instruction-datasets.md).

| Dataset | Samples | Thinking | Date | Licence note |
|---|---|---|---|---|
| [**WildChat-4.8M**](https://huggingface.co/datasets/allenai/WildChat-4.8M) | 3.2M | No | Aug 2025 |  |
| [**lmsys-chat-1m**](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 1M | No | Sep 2023 |  |
| [**arena-human-preference-140k**](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) | 136k | No | Jul 2025 | CC-BY-4.0 |

The name of the first dataset says 4.8M, and the list gives a size of 3.2M.

## Preference datasets

Pairs of a chosen and a rejected answer. See the lesson [Preference datasets](datasets/03-preference-datasets.md).

| Dataset | Pairs | Date | Licence note |
|---|---|---|---|
| [**Skywork-Reward-Preference-80K-v0.2**](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2) | 77k | 2024 |  |
| [**ultrafeedback-binarized-preferences-cleaned**](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) | 61.1k | 2023 |  |
| [**Infinity-Preference**](https://huggingface.co/datasets/BAAI/Infinity-Preference) | 59k | Sep 2024 |  |
| [**Code-Preference-Pairs**](https://huggingface.co/datasets/Vezora/Code-Preference-Pairs) | 53k | Jul 2024 |  |
| [**orpo-dpo-mix-40k**](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k) | 44k | May 2024 |  |
| [**HelpSteer3**](https://huggingface.co/datasets/nvidia/HelpSteer3) | 40.5k | Oct 2024 |  |
| [**chatbot_arena_conversations**](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) | 33k | Jul 2023 |  |
| [**FalseReject**](https://huggingface.co/datasets/AmazonScience/FalseReject) | 28.8k | May 2025 | CC-BY-NC-4.0 |
| [**tulu-3-pref-personas-instruction-following**](https://huggingface.co/datasets/allenai/tulu-3-pref-personas-instruction-following) | 19.9k | Nov 2024 |  |
| [**Human-Like-DPO-Dataset**](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset) | 10.9k | May 2024 |  |

## Tools

The list groups the tools in four steps. The lesson [From raw text to a dataset: the tools](datasets/04-tools.md) explains each step with numbers, and gives the limits of each tool.

| Step | Tool | What it is for |
|---|---|---|
| Scraping | [**Trafilatura**](https://github.com/adbar/trafilatura) | Gets the text and the metadata of web pages |
| Scraping | [**Marker**](https://github.com/VikParuchuri/marker) | Changes PDF files into Markdown |
| Filtering | **Rule-based filtering** | A technique, not a tool: remove the samples that contain unwanted words, for example refusals |
| Filtering | [**NeMo Curator**](https://github.com/NVIDIA/NeMo-Curator) | Deduplication, filters and classifiers on GPUs |
| Filtering | [**DataTrove**](https://github.com/huggingface/datatrove) | Text processing in pipeline blocks, on one machine or on a cluster |
| Filtering | [**Dolma**](https://github.com/allenai/dolma) | Taggers and fast deduplication for very large text sets |
| Filtering | [**SemHash**](https://github.com/MinishLab/semhash) | Deduplication of near duplicates with embeddings |
| Filtering | [**Argilla**](https://argilla.io/) | Manual review and annotation by several persons |
| Generation | [**Curator**](https://github.com/bespokelabsai/curator/) | Pipelines that make synthetic data with LLMs |
| Generation | [**Distilabel**](https://github.com/argilla-io/distilabel) | Generation and improvement of instruction and preference data |
| Generation | [**Augmentoolkit**](https://github.com/e-p-armstrong/augmentoolkit) | Changes raw text into datasets |
| Generation | [**Data Prep Kit**](https://github.com/IBM/data-prep-kit) | Preparation of code and language data at scale |
| Exploration | [**Nomic Atlas**](https://github.com/nomic-ai/nomic) | A map of the embeddings of a dataset |

## Source

All the rows of this page come from the README of [LLM Datasets](https://github.com/mlabonne/llm-datasets) by Maxime Labonne, read on 11 November 2026. The descriptions are rewritten. The sizes, dates and licence notes are the values of that list.
