# 5. Interpretability: probes, features and steering

## Core idea

Interpretability reads the numbers inside a model. It does not rely on the answers of the model.
A concept is often a direction in those numbers.
You can read the direction, find directions without labels, push along a direction, and remove it to test if the model uses it.

## The example

This page uses one small model for all four parts. It has the embeddings of the [toy example](../toy-example.md) and a **hidden state** $h$: the 2 numbers that the model calculates for a text before it writes the next token.
The logit of each token is the dot product of its embedding with $h$. Softmax then gives the probabilities.

$$\text{logits} = E\,h \qquad E = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \quad \text{(rows: the, cat, sat)}$$

| Item | Value |
|---|---|
| Hidden state of parts 3 and 4 (assumed) | $h = [1, 1]$ |
| Logits of `the`, `cat`, `sat` | [1, 1, 2] |
| Probabilities | [0.212, 0.212, 0.576] |
| Hidden states of 6 texts with a label (assumed) | The table in part 1 |
| Directions | $v = [-1, 1]$ (part 1 finds it) and $v_{cat} = [0, 1]$ (the embedding of `cat`) |

<figure class="diagram" data-src="interpretability/directions.svg">
<figcaption>The four operations of this page, with the toy numbers. Box 4 uses the direction of "cat".</figcaption>
</figure>

## 1. Probes: read a concept

### Setup

**Interpretability** is the work that finds what the numbers inside a model mean. **Mechanistic interpretability** is the part of this work that explains the calculation itself: which numbers cause which output.
An **activation** is a vector that the model calculates for one input, between the input and the output. In a Transformer, the most used activation is the **residual stream**: the vector of one token position, to which each block adds its result. Our $h$ is the residual stream at the last position.
A **linear probe** is a small classifier on activations: one direction $v$ and one threshold. It reads one concept.
Assume the concept "the text is about an animal", and 6 texts with a known label.

| Text | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| About an animal? | yes | yes | yes | no | no | no |
| Hidden state | [0.2, 1.2] | [0.4, 1.6] | [0.0, 1.4] | [1.0, 0.6] | [1.4, 0.2] | [1.2, 0.4] |

### Steps

1. **Calculate the mean hidden state of each group.** With the concept: $[0.2, 1.4]$. Without the concept: $[1.2, 0.4]$.

    **Why:** one text has many properties. The mean of a group keeps what its texts have in common.

2. **Subtract the two means.** The result is the **difference of means**. It is the probe direction.

    $$v = [0.2,\ 1.4] - [1.2,\ 0.4] = [-1,\ 1]$$

    **Why:** the difference points from "no animal" to "animal". The **linear representation hypothesis** says that a model stores a concept as one direction. If that is true, one vector is sufficient to read the concept.

3. **Give each hidden state a score: its dot product with $v$.** For text A: $-1 \times 0.2 + 1 \times 1.2 = 1.0$.

    | Text | A | B | C | D | E | F |
    |---|---|---|---|---|---|---|
    | Score | 1.0 | 1.2 | 1.4 | -0.4 | -1.2 | -0.8 |

    **Why:** the dot product measures how far a hidden state goes along the direction.

4. **Put a threshold between the two groups, and count.** The mean scores are 1.2 and -0.8. The middle is 0.2. A score above 0.2 means "animal". The **probe accuracy** is the share of texts with the correct result: $6 / 6 = 1.0$.

    **Why:** a score is not a decision. The threshold makes one, and the accuracy says how good the decisions are.

5. **Test the probe on a new text.** Assume a seventh text about an animal with $h = [1.6, 1.4]$. Its score is $-1.6 + 1.4 = -0.2$. This is below 0.2, so the probe says "no animal". The probe is wrong. Its accuracy on the 7 texts is $6 / 7 = 0.857$.

    **Why:** in the 6 texts, each animal text also had a small first number. The probe learned the two properties together. The new text has the high second number of the animal group (1.4), but its first number is high too.

<div class="key-point">

A probe finds what separates your two sets of examples. That can be the concept, or a property that only occurs together with the concept in your data. A probe finds a correlation, not a cause.

</div>

### From toy to real

- Measure the accuracy of a probe on texts that you did not use to build it.
- "Detecting Strategic Deception Using Linear Probes" (February 2025) built probes on Llama-3.3-70B-Instruct. Their AUROC (1 is a perfect ranking, 0.5 is chance) was between 0.96 and 0.999. With a threshold that flags 1% of normal chat answers, the probes caught 95% to 99% of the deceptive answers. But they also gave high scores to honest answers in scenarios about deception. The authors say that this is not sufficient for a robust defence.
- A Google DeepMind team (March 2025) tried to detect harmful intent in prompts with new types of jailbreak. A linear probe reached an AUROC of 0.999 on this new data. Probes built from the features of part 2 were worse.
- A difference-of-means probe also detected models with a hidden trigger: see [Scheming](04-scheming.md).

## 2. Features, superposition and sparse autoencoders

### Setup

A probe needs labels. To find concepts that you did not think of, you must split an activation into its parts without labels.
A **neuron** is one number of an activation vector. A **feature** is a direction in the activation space that has one meaning.
"One neuron, one feature" is false: many neurons are **polysemantic**, which means active for many unrelated concepts.
The cause is **superposition**: the model stores more features than it has neurons, as directions that overlap.
Our toy model does this. It has 2 neurons and 3 features: the directions of `the`, `cat` and `sat`.

### Steps

1. **Give each feature direction the length 1.** `the` = [1, 0] and `cat` = [0, 1] stay as they are. `sat` = [1, 1] has the length 1.414, so it becomes [0.707, 0.707].

    **Why:** a dot product then compares only the directions. Neuron 1 is active for `the` and for `sat`: it is polysemantic.

2. **Calculate the dot product of each pair of directions.** This overlap is the **interference**.

    | | `the` | `cat` | `sat` |
    |---|---|---|---|
    | `the` | 1 | 0 | 0.707 |
    | `cat` | 0 | 1 | 0.707 |
    | `sat` | 0.707 | 0.707 | 1 |

    **Why:** 2 dimensions have space for only 2 directions at a right angle. A third direction must overlap the others. Interference is the price of 3 features in 2 numbers.

3. **Make only `sat` active, and read the three features.** The activation is $x = [0.707, 0.707]$. Its dot products with the three directions are $[0.707,\ 0.707,\ 1.0]$.

    **Why:** a linear read finds `cat` at 0.707, but `cat` is not in the activation. This is the interference of step 2.

4. **Add a negative bias, then apply ReLU.** Assume the bias $-0.75$. The read becomes $[-0.043,\ -0.043,\ 0.25]$. ReLU changes each negative number to 0:

    $$f = \text{ReLU}([0.707,\ 0.707,\ 1.0] - 0.75) = [0,\ 0,\ 0.25]$$

    **Why:** the interference (0.707) is smaller than a real feature (1.0), so a threshold removes it. This is the trick of the encoder of a sparse autoencoder. The real feature now reads 0.25: a trained encoder uses a larger weight to correct this.

5. **Make `the` and `cat` active together.** The activation is $x = [1, 0] + [0, 1] = [1, 1]$. The read is $[1,\ 1,\ 1.414]$ and the encoder gives $[0.25,\ 0.25,\ 0.664]$. The three features are active, and `sat` is the strongest.

    **Why:** "`the` with `cat`" and "a strong `sat`" are the same activation. No threshold can tell them apart. Superposition only works when features are rarely active together.

<div class="key-point">

A model can store more features than it has neurons, if only a few features are active for one input. A sparse autoencoder uses the same condition to get the features back.

</div>

### From toy to real

- **Dictionary learning** finds, without labels, a set of directions such that each activation is a sum of a few of them. The usual tool is a **sparse autoencoder (SAE)**: an encoder as in step 4, with many more features than neurons, and a decoder that builds the activation again from the active features. **Sparsity** means that only a few features are active for one input. The **reconstruction error** is the difference between the activation and the copy from the decoder. The page [New trends](../scientist/08-new-trends.md) calculates one full SAE step.
- "Toy Models of Superposition" (September 2022) shows in small models that polysemantic neurons occur when a model stores additional sparse features in superposition.
- "Towards Monosemanticity" (Anthropic, October 2023) split a layer with 512 neurons into more than 4,000 features, for example DNA sequences and legal language.
- In May 2024, Anthropic got millions of features from the middle layer of Claude 3 Sonnet. It says that they are a small subset of the concepts of the model.
- A **transcoder** is a variant that is trained to give the activations of the next layer, not a copy of its input.
- The limits: "Open Problems in Mechanistic Interpretability" (January 2025) says that the reconstruction errors are too high. With an SAE of 16 million features in place of its activations, GPT-4 had the loss of a model with about one tenth of the training compute. The features also change with the size of the dictionary. After the probe result of part 1, the Google DeepMind team gave a lower priority to basic SAE research. It also says that one negative result is not a strong proof.

## 3. Steering: push along a direction

### Setup

**Steering** changes the behaviour of a model through its activations while it runs. The weights stay the same.
**Activation addition** adds a vector to the hidden state. This vector is the **steering vector**, and $\alpha$ is the steering strength:

$$h' = h + \alpha\,v$$

In **contrastive activation addition**, the steering vector is the difference of means between two sets of prompts. This is the calculation of part 1, so we use $v = [-1, 1]$ on the hidden state $h = [1, 1]$. Before the change, the probabilities are [0.212, 0.212, 0.576] and the probe score is $h \cdot v = 0$, below the threshold 0.2.

### Steps

1. **Add the vector with $\alpha = 1$.** The new hidden state is $h' = [1, 1] + 1 \times [-1, 1] = [0, 2]$. The logits are $E\,h' = [0, 2, 2]$ and the probabilities are [0.063, 0.468, 0.468]. The probe score is now 2.

    **Why:** the direction points to "animal", so `cat` becomes more probable (0.212 to 0.468). But `the` falls from 0.212 to 0.063, and nobody asked for this. The vector also contains the correlated property of part 1, a small first number.

2. **Increase the strength.**

    | $\alpha$ | $h'$ | Logits | Probabilities (`the`, `cat`, `sat`) |
    |---|---|---|---|
    | 0 | [1, 1] | [1, 1, 2] | [0.212, 0.212, 0.576] |
    | 1 | [0, 2] | [0, 2, 2] | [0.063, 0.468, 0.468] |
    | 2 | [-1, 3] | [-1, 3, 2] | [0.013, 0.721, 0.265] |
    | 4 | [-3, 5] | [-3, 5, 2] | [0.000, 0.952, 0.047] |

    **Why:** each step of 1 adds 1 to the logit of `cat` and subtracts 1 from the logit of `the`. The logit of `sat` stays at 2, because its embedding [1, 1] is at a right angle to $v$.

3. **Find where the output breaks.** At $\alpha = 4$, `cat` has a probability of 0.952. At $\alpha = 8$, it has 0.999. The model can then only write "cat cat cat".

    **Why:** a hidden state that is far from the normal values gives extreme logits. A steering vector has a useful range of $\alpha$.

<figure class="plot">
<img src="/figures/interpretability/steering.svg" alt="Probability of each token against the steering strength from 0 to 6. At a strength of 4, cat rises from 0.21 to 0.95 and sat falls from 0.58 to 0.05.">
<figcaption>A small strength moves the probabilities. A large strength gives all the probability to one token.</figcaption>
</figure>

<div class="llm-widget" data-widget="steering"></div>

<div class="key-point">

A direction that you can read is also a direction that you can push. But steering has a limited range, and it moves all that separates the two prompt sets, not only the concept.

</div>

### From toy to real

- "Steering Language Models With Activation Engineering" (August 2023) calculates the vector from one pair of prompts, for example "Love" and "Hate".
- "Steering Llama 2 via Contrastive Activation Addition" (December 2023) uses the mean difference over many pairs, on Llama 2 Chat. The authors report that larger strengths decrease the quality of the text.
- The **refusal direction** comes from "Refusal in Language Models Is Mediated by a Single Direction" (June 2024). In 13 open-source chat models with up to 72 billion parameters, one difference-of-means direction between harmful and harmless requests controls the refusal. With the direction added, a model refuses harmless requests. With the direction removed, it answers harmful requests. **Abliteration** is the removal of this direction from the weights: see [Jailbreaks](03-jailbreaks.md).
- A **persona vector** (July 2025) is a direction for a character trait, such as evil, sycophancy or a tendency to hallucinate. It is the difference of the mean activations between answers that show the trait and answers that do not. It can monitor the trait, steer against it, and find training data that causes it.
- **Feature clamping** sets one SAE feature to a fixed value. Anthropic set the "Golden Gate Bridge" feature of Claude 3 Sonnet to a high value. The model then spoke about the bridge in almost any answer.

## 4. Causal tests: ablation, patching, circuits

### Setup

Parts 1 and 2 find directions. They do not show that the model uses them.
A causal test changes an activation and measures the output.
We test the direction of `cat`, $v_{cat} = [0, 1]$, on the hidden state $h = [1, 1]$. This $h$ is equal to the embedding of `sat`.

### Steps

1. **Read the direction.** Use the three embeddings as three hidden states. The scores $h \cdot v_{cat}$ are `the` = 0, `cat` = 1, `sat` = 1.

    **Why:** this is a probe. It finds the direction in two of the three states. It does not say what the model does with it.

2. **Remove the direction from $h$.** An **ablation** removes a part of an activation. A **directional ablation** removes the part along one direction of length 1:

    $$h' = h - (h \cdot v_{cat})\,v_{cat} = [1, 1] - 1 \times [0, 1] = [1, 0]$$

    **Why:** if the model uses the direction, an output must change when the direction is gone.

3. **Compare the outputs before and after.** The logit of `sat` reads $h$ with the weights [1, 1]: it changes from 2 to 1. The logit of `the` reads $h$ with the weights [1, 0]: it stays at 1.

    **Why:** the first output uses the direction. The second does not, although the probe found the direction in the same hidden state.

4. **Choose the value that replaces the removed part.** A **zero ablation** sets it to 0, as in step 2. A **mean ablation** sets it to its mean over many inputs. The mean of the scores of step 1 is 0.667, thus $h' = [1, 0.667]$ and the logit of `sat` is 1.667.

    **Why:** the value 0 can be abnormal for the model. Then the output changes because the activation is strange, not because the information is gone.

5. **Copy an activation from a different run.** This is **activation patching**. The clean run is text A of part 1, $h = [0.2, 1.2]$. The corrupted run is text D, $h = [1.0, 0.6]$. The measure is the logit of `cat` minus the logit of `the`: 1.0 in the clean run and -0.4 in the corrupted run. Now copy neuron 2 from the clean run into the corrupted run: $h = [1.0, 1.2]$ and the measure is 0.2.

    $$\text{recovered} = \frac{0.2 - (-0.4)}{1.0 - (-0.4)} = 0.429$$

    With neuron 1 as the patch, $h = [0.2, 0.6]$, the measure is 0.4 and the recovered share is 0.571.

    **Why:** a patch shows where the difference between two behaviours comes from. Here neuron 1 carries more than half of it. An explanation is **faithful** when it describes the calculation that the model really does. "Only neuron 2 is important" is not faithful: it gives back 0.429 of the effect.

<div class="key-point">

A probe direction or an SAE feature is a hypothesis. The evidence is an ablation or a patch that changes the output as the hypothesis predicts.

</div>

### From toy to real

- **Causal tracing** does a patch for each layer and each token position. The result is a map of the places where a piece of information is necessary.
- A **circuit** is a small set of components, with their connections, that does one task. An **attribution graph** shows the circuit for one prompt: its nodes are features, and its edges show how much one feature causes the next.
- "On the Biology of a Large Language Model" (Anthropic, March 2025) builds attribution graphs for Claude 3.5 Haiku with a transcoder of 30 million features. For a question about the capital of the state that contains Dallas, the graph shows a step for Texas before the answer Austin. The authors say that the method gave a satisfying result for about a quarter of their prompts.
- The **logit lens** applies the output matrix to an activation of a middle layer, to see which tokens this layer prefers. Our $E\,h$ is this calculation. It measures only the direct effect of the activation.
- Tools: use TransformerLens to record and change the activations of GPT-style models, nnsight to do the same on any PyTorch model or on a remote server, SAELens to train or load an SAE, Neuronpedia to look at features and steer a model in the browser, and circuit-tracer to calculate attribution graphs on open models.
- The International AI Safety Report 2026 says that the inner workings of models are still poorly understood.

## Minimal code

This code gives the same numbers as the steps on this page.

```python
import numpy as np

E = np.array([[1., 0.], [0., 1.], [1., 1.]])             # the, cat, sat
softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()

# 1. Probe: difference of means, score, threshold
with_c = np.array([[0.2, 1.2], [0.4, 1.6], [0.0, 1.4]])
without = np.array([[1.0, 0.6], [1.4, 0.2], [1.2, 0.4]])
v = with_c.mean(0) - without.mean(0)                     # [-1, 1]
threshold = ((with_c @ v).mean() + (without @ v).mean()) / 2   # 0.2
new_score = np.array([1.6, 1.4]) @ v                     # -0.2: the probe is wrong

# 2. Superposition: 3 unit directions in 2 dimensions, then the encoder of an SAE
F = E / np.linalg.norm(E, axis=1, keepdims=True)
read = F @ F[2]                                          # [0.707, 0.707, 1.0]
features = np.maximum(read - 0.75, 0)                    # [0, 0, 0.25]

# 3. Steering
h = np.array([1., 1.])
probs = softmax(E @ (h + 2 * v))                         # [0.013, 0.721, 0.265]

# 4. Ablation and patching
v_cat = np.array([0., 1.])
ablated = h - (h @ v_cat) * v_cat                        # [1, 0]: logit of sat 2 -> 1
ld = lambda x: (E @ x)[1] - (E @ x)[0]                   # logit of cat minus logit of the
clean, corrupted = with_c[0], without[0]
patched = np.array([corrupted[0], clean[1]])             # neuron 2 from the clean run
recovered = (ld(patched) - ld(corrupted)) / (ld(clean) - ld(corrupted))   # 0.429
```

## Comparison

| Method | Question that it answers | Toy result | Main limit |
|---|---|---|---|
| Linear probe | Is the concept in the activation? | Accuracy 1.0, then 0.857 with a new text | It finds a correlation |
| Sparse autoencoder | Which features are in the activation? | [0, 0, 0.25]: 1 of 3 features active | Reconstruction error; features change with the size |
| Steering | Does a push along the direction change the behaviour? | `cat` from 0.212 to 0.468 at a strength of 1 | Limited range; side effects |
| Ablation | Does the output need the direction? | Logit of `sat` from 2 to 1 | The new activation can be abnormal |
| Activation patching | Which part causes the difference between two prompts? | Neuron 1: 0.571, neuron 2: 0.429 | One pair of prompts at a time |

## Checks and common misconceptions

- **"The probe is accurate, so the model uses this concept."** A probe shows only that the information is in the activations. Check: look for an ablation or a patch that changes the output.
- **"An SAE feature is the true unit of the model."** An SAE of a different size gives different features, and a part of the activation stays in the reconstruction error. Check: clamp or ablate the feature and see if the output changes as its name predicts.
- **"Steering is a safe control for a model."** It has a limited range, and the same calculation can remove a safeguard, as the refusal direction shows. Check: measure the quality of the text at each strength.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. Each statement about a real system was read in the linked source on 11 November 2026.

- [Detecting Strategic Deception Using Linear Probes](https://arxiv.org/abs/2502.03407), Goldowsky-Dill et al., 2025 (AUROC range, 95% to 99%, limits).
- [Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks), Smith et al., Google DeepMind, 26 March 2025 (AUROC 0.999, SAE probes, lower priority).
- [Simple probes can catch sleeper agents](https://www.anthropic.com/research/probes-catch-sleeper-agents), MacDiarmid et al., Anthropic, 23 April 2024 (probe for a hidden trigger).
- [Toy Models of Superposition](https://arxiv.org/abs/2209.10652), Elhage et al., 2022 (polysemantic neurons, superposition).
- [Towards Monosemanticity](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning), Anthropic, 5 October 2023 (512 neurons, more than 4,000 features).
- [Mapping the Mind of a Large Language Model](https://www.anthropic.com/research/mapping-mind-language-model), Anthropic, 21 May 2024 (millions of features, Golden Gate Bridge feature, limits).
- [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496), Sharkey et al., 2025 (SAE limits, transcoders, logit lens).
- [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248), Turner et al., 2023 (one pair of prompts).
- [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681), Panickssery et al., 2023 (mean difference, models, quality).
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717), Arditi et al., 2024 (13 models, up to 72 billion parameters).
- [Persona Vectors](https://arxiv.org/abs/2507.21509), Chen et al., 2025 (traits, calculation, three uses).
- [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html), Lindsey et al., Anthropic, 27 March 2025 (attribution graphs, 30 million features, a quarter of the prompts).
- [International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026), Bengio et al. (inner workings poorly understood).
- Tools: [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), [nnsight](https://nnsight.net/), [SAELens](https://github.com/jbloomAus/SAELens), [Neuronpedia](https://www.neuronpedia.org/), [circuit-tracer](https://github.com/safety-research/circuit-tracer).
