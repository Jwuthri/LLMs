# Notebook guide

The [LLM Course](https://github.com/mlabonne/llm-course) has a list of 23 notebooks. Maxime Labonne wrote them, and they run in Google Colab. This page tells you, for each notebook, what it does, which lesson to read first, and what to look at when you run it.

This project did not run the notebooks: each description comes from the linked article, read on 11 November 2026. A notebook with no article has only the facts of the course README or of another opened page.

## How to learn from a notebook

1. **Read the lesson first.** It gives the calculation with small numbers.
2. **Predict one number before you run a cell**: a file size, a number of parameters, the next token.
3. **Run the cell** and compare.
4. **Change one parameter and run again.** Change only one, or you cannot know which change had the effect.
5. **Write what changed**: the parameter, the old result, the new result.

## Tools

These notebooks are forms: you fill in some fields and run all cells. None of them has an article, so no date was read for them.

### LLM AutoEval

You give a model ID, a benchmark suite (`nous`, `lighteval` or `openllm`) and a GPU. The notebook starts a machine on RunPod (a cloud GPU service), runs the benchmarks, and writes a summary to a GitHub Gist. These facts come from the README of the repository.

**Read first:** [Evaluation](scientist/06-evaluation.md) and [Metrics and losses](fundamentals/05-metrics.md).

**Look at:** the `openllm` suite. The README says that its results do not agree with runs that do not use vLLM. Compare two models only when the same suite gave the two scores.

[Repository](https://github.com/mlabonne/llm-autoeval) · [Colab](https://colab.research.google.com/drive/1Igs3WZuXAIv9X0vwqiE90QlEPys8e8Oa?usp=sharing)

### LazyMergekit

The README says that it merges models with MergeKit in one click. The two merge articles below say how to use it: you paste a merge configuration, give a model name and your Hugging Face name and token, and run the cells.

**Read first:** [New trends](scientist/08-new-trends.md), part 1.

**Look at:** the `merge_method` line of your configuration. Find that method in the lesson before you run.

[Colab](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing)

### LazyAxolotl

The README says that it fine-tunes models in the cloud with Axolotl in one click.

**Read first:** [Supervised fine-tuning](scientist/04-fine-tuning.md) and [Training libraries](scientist/10-training-libraries.md).

**Look at:** the YAML configuration. Find the LoRA rank, the learning rate and the number of epochs.

[Colab](https://colab.research.google.com/drive/1TsDKNo2riwVmU55gjuBgB1AXVtRRfRHW?usp=sharing)

### AutoQuant

The README says that it quantizes a model in the GGUF, GPTQ, EXL2, AWQ and HQQ formats in one click.

**Read first:** [Quantization](scientist/07-quantization.md) and [Quantization beyond rounding](scientist/11-quantization-beyond-rounding.md).

**Look at:** the size of each output file. Divide it by the size of the 16-bit model.

[Colab](https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4?usp=sharing)

### Model Family Tree

The README says that it shows the family tree of merged models.

**Read first:** [New trends](scientist/08-new-trends.md), part 1.

**Look at:** the number of merge levels below one model. The merge article below warns that a merge can contain a model that was trained on test questions.

[Colab](https://colab.research.google.com/drive/1s2eQlolcI1VGgDhqWIANfkfKvcKrMyNr?usp=sharing)

### ZeroSpace

The README says that it creates a Gradio chat interface on a free ZeroGPU.

**Read first:** [Deploying LLMs](engineer/07-deploying.md), part 2.

**Look at:** the Gradio code that the notebook writes. Compare it with the Gradio lines of the lesson.

[Colab](https://colab.research.google.com/drive/1LcVUW5wsJTO2NGmozjji5CkC--646LgC)

### AutoAbliteration

The README says that it abliterates models with custom datasets. The abliteration article below explains the method.

**Read first:** [Interpretability: probes, features and steering](safety/05-interpretability.md), part 3, and [New trends](scientist/08-new-trends.md), part 3.

**Look at:** the two datasets. The direction is a difference of two means, so it depends fully on the prompts in each set.

[Colab](https://colab.research.google.com/drive/1RmLv-pCMBBsQGXQIM8yF-OdCNyoylUR1?usp=sharing)

### AutoDedup

The README says that it removes duplicates from a dataset with the Rensa library.

**Read first:** [Post-training datasets](scientist/03-datasets.md), part 4.

**Look at:** the similarity limit. The lesson removes a sample above a Jaccard similarity of 0.8. Change the limit and count the rows that stay.

[Colab](https://colab.research.google.com/drive/1o1nzwXWAa8kdkEJljbJFW1VuI-3VZLUn?usp=sharing)

## Fine-tuning

### Fine-tune Llama 3.1 with Unsloth

The notebook trains Llama 3.1 8B with QLoRA and the Unsloth library on the FineTome-100k dataset. It loads a 4-bit copy of the model (5.4 GB, not 16 GB), adds LoRA adapters with $r = 16$ and $\alpha = 16$, trains, merges the adapters and exports GGUF files. The article reports 4 hours 45 minutes on an A100 GPU, and about 47 hours on a free T4.

**Read first:** [Supervised fine-tuning](scientist/04-fine-tuning.md) and [Training libraries](scientist/10-training-libraries.md).

**Look at:** the number of trained parameters: 42 million, or 0.5196% of the model. Change $r$ to 32 and read the new number. Use a subset first (the article proposes 10,000 samples).

**Age:** 29 July 2024.

[Article](https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html) · [Colab](https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z?usp=sharing)

### Fine-tune Llama 3 with ORPO

The notebook trains Llama 3 8B with ORPO and the TRL library. ORPO does the supervised fine-tuning and the preference alignment in one stage; the last section of this page explains it. The article reports about 2 hours on an L4 GPU for a short run.

**Read first:** [Preference alignment](scientist/05-preference-alignment.md), part 2.

**Look at:** the line that selects the samples. The text of the article says 1,000 samples, but its code selects 100. Then read the accuracy curve: the article reports a value only a little above 0.5.

**Age:** 19 April 2024.

[Article](https://mlabonne.github.io/blog/posts/2024-04-19_Fine_tune_Llama_3_with_ORPO.html) · [Colab](https://colab.research.google.com/drive/1eHNWg9gnaXErdAa8_mcvjMupbSS6rDvi)

### Fine-tune Mistral-7b with DPO

The notebook applies DPO to OpenHermes-2.5-Mistral-7B. In the data, the chosen answer comes from ChatGPT and the rejected answer from Llama 2 13B chat. The settings are $\beta = 0.1$ and 200 steps. The article reports 6.7 more points on the mean score of the Open LLM Leaderboard.

**Read first:** [Preference alignment](scientist/05-preference-alignment.md), part 2.

**Look at:** $\beta$. The lesson calculates the loss with 0.5 and with 0.1. Then read the accuracy plot: the article says that a perfect score can mean that the two answers are too easy to tell apart.

**Age:** 1 January 2024.

[Article](https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html) · [Colab](https://colab.research.google.com/drive/15iFBr1xWgztXvhrj5I9fBv20c7CFOPBE?usp=sharing)

### Fine-tune Mistral-7b with QLoRA

The README gives only the name. No article.

**Read first:** [Supervised fine-tuning](scientist/04-fine-tuning.md).

**Look at:** $r$ and $\alpha$ in the LoRA configuration. Calculate the scale $\alpha / r$ as in step 4 of the lesson.

**Age:** no date read.

[Colab](https://colab.research.google.com/drive/1o_w0KastmEJNVwT5GoqMCciH-18ca5WS?usp=sharing)

### Fine-tune CodeLlama using Axolotl

The article trains Code Llama 7b on 1,000 Python samples with Axolotl. One YAML file holds all the settings: QLoRA in 4 bits, `lora_r: 32`, `lora_alpha: 16`, 3 epochs. The article runs the training on a rented RTX 3090 on RunPod in 1 hour 11 minutes.

**Read first:** [Training libraries](scientist/10-training-libraries.md) and [Supervised fine-tuning](scientist/04-fine-tuning.md).

**Look at:** `micro_batch_size`. The article sets it to 10 so that the GPU memory is almost full. Read the memory use, then change the value.

**Age:** 27 August 2023. This project did not check that the current Axolotl accepts this configuration.

[Article](https://mlabonne.github.io/blog/posts/A_Beginners_Guide_to_LLM_Finetuning.html) · [Colab](https://colab.research.google.com/drive/1Xu0BrCB7IShwSWKVcfAfhehwjDrDMH5m?usp=sharing)

### Fine-tune Llama 2 with QLoRA

The notebook trains Llama 2 7B chat on 1,000 samples of the Guanaco dataset, on a T4 GPU with 16 GB of memory. The article explains why QLoRA is necessary: the weights alone need 14 GB at 16 bits. The settings are $r = 64$, $\alpha = 16$ and 1 epoch.

**Read first:** [Supervised fine-tuning](scientist/04-fine-tuning.md).

**Look at:** the scale $\alpha / r$. Here it is $16 / 64 = 0.25$. The Axolotl notebook above has $16 / 32 = 0.5$.

**Age:** 24 July 2023. The model is Llama 2. The Unsloth notebook above uses Llama 3.1, one year later.

[Article](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html) · [Colab](https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing)

## Quantization

### Introduction to Quantization

The notebook writes absmax and zero-point quantization in Python and applies them to GPT-2 (487 MB at 32 bits). It compares the perplexity of a generated text: 15.53 for the original model, 17.92 for absmax and 17.97 for zero-point. The article says that one experiment is not reliable.

**Read first:** [Quantization](scientist/07-quantization.md), part 1, and [Metrics and losses](fundamentals/05-metrics.md) for the perplexity.

**Look at:** the example of the article. The largest absolute weight is 3.2 and the weight is 0.1. Calculate the absmax integer by hand with step 2 of the lesson. The article gets 4, and -1 with zero-point.

**Age:** 6 July 2023.

[Article](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html) · [Colab](https://colab.research.google.com/drive/1DPr4mUQ92Cc-xf4GgAaB6dFcFnWIvqYi?usp=sharing)

### 4-bit Quantization using GPTQ

The article explains GPTQ: it quantizes the weights of a layer in order, and it changes the remaining weights to repair the output. The notebook applies it to GPT-2 with the AutoGPTQ library on a free T4 GPU. The settings are `bits=4`, `group_size=128` and 1,024 calibration samples from the C4 dataset.

**Read first:** [Quantization](scientist/07-quantization.md), part 3, and [Quantization beyond rounding](scientist/11-quantization-beyond-rounding.md).

**Look at:** the number of calibration samples. Decrease it and generate text again. The article does not measure the perplexity, so add that measurement to compare two runs.

**Age:** 30 July 2023. The article names the GGML format, which GGUF replaced (see the next notebook).

[Article](https://mlabonne.github.io/blog/4bit_quantization/) · [Colab](https://colab.research.google.com/drive/1lSvVDaRgqQp_mWK_jC9gydz6_-y6Aq4A?usp=sharing)

### Quantization with GGUF and llama.cpp

The notebook compiles llama.cpp, downloads a 7B model, converts it to a 16-bit file, and quantizes it with the `Q4_K_M` and `Q5_K_M` types. The article also shows how a simple 4-bit type works: blocks of 32 weights, with one scale for each block.

**Read first:** [Quantization](scientist/07-quantization.md), part 2.

**Look at:** the file sizes. The article reports 13.5 GB at 16 bits and 4.08 GB for `Q4_K_M`: a ratio of $4.08 / 13.5 = 0.30$. The lesson gives about 31%.

**Age:** 3 September 2023. The article says that the file format changed from GGML to GGUF. The lesson uses only the GGUF names.

[Article](https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html) · [Colab](https://colab.research.google.com/drive/1pL8k7m04mgE5jo2NrjGi8atB0j_37aDD?usp=sharing)

### ExLlamaV2: The Fastest Library to Run LLMs

The notebook quantizes zephyr-7B-beta to the EXL2 format. EXL2 uses the GPTQ algorithm, but it can mix precisions inside a model and inside a layer. You give a target mean number of bits per weight (`-b 5.0` in the article), and the tool tries different settings and measures the error of each. The article reports 2 hours 10 minutes on a T4 GPU for the quantization, then 56.44 tokens per second.

**Read first:** [Quantization](scientist/07-quantization.md), part 3, and [Quantization beyond rounding](scientist/11-quantization-beyond-rounding.md).

**Look at:** the file `measurement.json`, which holds these measurements. Then change `-b` and compare the file size and the speed.

**Age:** 19 November 2023.

[Article](https://mlabonne.github.io/blog/posts/ExLlamaV2_The_Fastest_Library_to_Run%C2%A0LLMs.html) · [Colab](https://colab.research.google.com/drive/1yrq4XBlxiA0fALtMoT2dwiACVc77PHou?usp=sharing)

## Other

### Merge LLMs with MergeKit

The article explains four merge methods: SLERP, TIES, DARE and passthrough. The notebook then merges two 7B models with SLERP in about 10 minutes, with no GPU. The article warns that the result is probably contaminated: some source models can contain test data of the public leaderboard.

**Read first:** [New trends](scientist/08-new-trends.md), part 1, and [Evaluation](scientist/06-evaluation.md) for contamination.

**Look at:** the values of `t` in the SLERP configuration. With $t = 0$ the result is the first model, and with $t = 1$ it is the second. Calculate one SLERP step with the formula of the lesson.

**Age:** 8 January 2024.

[Article](https://mlabonne.github.io/blog/posts/2024-01-08_Merge_LLMs_with_mergekit%20copy.html) · [Colab](https://colab.research.google.com/drive/1_JS7JKJAQozD48-LhYdegcuuZ2ddgXfr?usp=sharing)

### Create MoEs with MergeKit

The article builds a mixture of experts from four fine-tuned Mistral-7B models: chat, code, math and role-play. A configuration file gives each expert a short list of prompts, and MergeKit uses them to set the router. The last section of this page explains how. The Colab link is the LazyMergekit notebook, with the `mixtral` branch selected.

**Read first:** [New trends](scientist/08-new-trends.md), part 1.

**Look at:** the number of parameters of the result. The article gives 24.2 billion, not $4 \times 7 = 28$ billion.

**Age:** 24 March 2024 on the page (its address says 28 March).

[Article](https://mlabonne.github.io/blog/posts/2024-03-28_Create_Mixture_of_Experts_with_MergeKit.html) · [Colab](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing)

### Uncensor any LLM with abliteration

The notebook records the activations of a model on 256 harmful and 256 harmless instructions. The difference of the two means gives one candidate direction for each layer. It tests the candidates, keeps one, and changes the weights so that the model cannot write along that direction. The article measures a lower score on all its benchmarks after this change. A DPO training run then recovers most of the loss.

**Read first:** [Interpretability: probes, features and steering](safety/05-interpretability.md), part 3, and [New trends](scientist/08-new-trends.md), part 3 (steps 5 and 6).

**Look at:** the benchmark tables before and after. They show the cost that the lesson names: the quality on other tasks can fall. The method removes a safeguard, so read the safety lesson first.

**Age:** 4 June 2024.

[Article](https://mlabonne.github.io/blog/posts/2024-06-04_Uncensor_any_LLM_with_abliteration.html) · [Colab](https://colab.research.google.com/drive/1VYm3hOcvCpbGiqKZb141gJwjdmmCcVpR?usp=sharing)

### Improve ChatGPT with Knowledge Graphs

The notebook asks GPT-3.5 when Apple announced the Vision Pro. The model cannot answer, because the event is after its training data. The notebook then builds a small knowledge graph with LangChain, first from one sentence and then from 22 hand-written facts, and asks again through a chain that reads the graph. The answer is now correct. It needs an OpenAI API key.

**Read first:** [Advanced RAG](engineer/04-advanced-rag.md), part 1, then the last section of this page.

**Look at:** the context lines that the chain prints. They are the facts that go into the prompt. Add a fact about a new product and ask a question about it.

**Age:** 20 June 2023. This project did not check that the code runs with the current LangChain and OpenAI libraries.

[Article](https://mlabonne.github.io/blog/posts/Article_Improve_ChatGPT_with_Knowledge_Graphs.html) · [Colab](https://colab.research.google.com/drive/1mwhOSw9Y9bgEaIFKT4CLi0n18pXRM4cj?usp=sharing)

### Decoding Strategies in Large Language Models

The notebook gives "I have a dream" to GPT-2 and generates 5 tokens with greedy search, beam search, top-k sampling and nucleus sampling. The greedy sequence has a score of -1.16. Beam search finds a sequence with a better score, -0.69, but it repeats the input.

**Read first:** [The LLM architecture](scientist/01-architecture.md), part 4.

**Look at:** the two scores. They show the effect of steps 1 and 2 of the lesson: the best first token does not always start the best sequence. Then change `top_k` (the article plots 20) and $p$ (the article uses 0.5).

**Age:** 7 June 2023 on the page.

[Article](https://mlabonne.github.io/blog/posts/2022-06-07-Decoding_strategies.html) · [Colab](https://colab.research.google.com/drive/19CJlOS5lI29g-B3dziNn93Enez1yiHk2?usp=sharing)

## Topics that the lessons do not explain

### ORPO

**ORPO** is a training method that joins supervised fine-tuning and preference alignment in one stage. The usual way has two stages: supervised fine-tuning, then a method such as DPO. The article gives the reason for the second stage: supervised fine-tuning increases the probability of the good answers, but also the probability of the answers that you do not want. ORPO adds a second term to the usual loss:

$$\mathcal{L}_{ORPO} = \mathbb{E}_{(x, y_w, y_l)}\left[\mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR}\right]$$

$\mathcal{L}_{SFT}$ is the usual next-token loss. $\mathcal{L}_{OR}$ is the odds ratio term: it gives a weak penalty to the rejected answer $y_l$ and a strong reward to the chosen answer $y_w$. $\lambda$ sets the weight of this term. TRL calls it `beta`, with a default value of 0.1. The data has the same form as for DPO: a prompt, a chosen answer and a rejected answer.

### The frankenMoE

A **mixture of experts** (MoE) is a model in which each block has several feed-forward networks, the **experts**. A **router** decides, for each token, which experts process it. Mixtral has 8 experts and uses 2 for each token. In a usual MoE, the experts and the router are trained together. A **frankenMoE** is an MoE that is assembled from models that exist. MergeKit copies the layer norm and the self-attention layers from one base model, and the feed-forward layers from each expert model. Thus the 4 experts of the article give 24.2 billion parameters, not 28 billion. The router is not trained. MergeKit can initialise it with random weights, with the raw embeddings of the tokens, or with the **hidden** method: it takes a list of prompts for each expert, reads their representation in the last layer, and uses the normalized mean as the weights of the gate.

### Knowledge graphs for an assistant

A **knowledge graph** stores facts as a graph: each node is an entity and each edge is a relation. One fact is a **triple**: a source node, a relation and a target node, for example (Vision Pro, announced in, 2023). The article uses the graph to give facts to a model that does not know them. First, the chain extracts the entities of the question, here Apple and Vision Pro. Second, it collects triples of these entities: the printed context has the facts that start at Apple and at Vision Pro. Third, the model answers from this context.

## Sources

- [LLM Course](https://github.com/mlabonne/llm-course), Maxime Labonne, Apache-2.0.

Pages opened on 11 November 2026, all by Maxime Labonne:

- [LLM AutoEval repository](https://github.com/mlabonne/llm-autoeval)
- [Fine-tune Llama 3.1 Ultra-Efficiently with Unsloth](https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html)
- [Fine-tune Llama 3 with ORPO](https://mlabonne.github.io/blog/posts/2024-04-19_Fine_tune_Llama_3_with_ORPO.html)
- [Fine-tune Mistral-7b with Direct Preference Optimization](https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html)
- [A Beginner's Guide to LLM Fine-Tuning](https://mlabonne.github.io/blog/posts/A_Beginners_Guide_to_LLM_Finetuning.html)
- [Fine-Tune Your Own Llama 2 Model in a Colab Notebook](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html)
- [Introduction to Weight Quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html)
- [4-bit LLM Quantization with GPTQ](https://mlabonne.github.io/blog/posts/4_bit_Quantization_with_GPTQ.html) (target of the README link)
- [Quantize Llama models with GGUF and llama.cpp](https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html)
- [ExLlamaV2: The Fastest Library to Run LLMs](https://mlabonne.github.io/blog/posts/ExLlamaV2_The_Fastest_Library_to_Run%C2%A0LLMs.html)
- [Merge Large Language Models with MergeKit](https://mlabonne.github.io/blog/posts/2024-01-08_Merge_LLMs_with_mergekit%20copy.html)
- [Create Mixtures of Experts with MergeKit](https://mlabonne.github.io/blog/posts/2024-03-28_Create_Mixture_of_Experts_with_MergeKit.html)
- [Uncensor any LLM with abliteration](https://mlabonne.github.io/blog/posts/2024-06-04_Uncensor_any_LLM_with_abliteration.html)
- [Improve ChatGPT with Knowledge Graphs](https://mlabonne.github.io/blog/posts/Article_Improve_ChatGPT_with_Knowledge_Graphs.html)
- [Decoding Strategies in Large Language Models](https://mlabonne.github.io/blog/posts/2022-06-07-Decoding_strategies.html)
