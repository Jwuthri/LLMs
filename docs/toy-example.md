# The toy example

## Why one example

Each page of this course explains a part of a large language model (LLM).
A real LLM has billions of numbers, so you cannot follow a calculation by hand.
This course uses one model that is small enough to calculate on paper, and uses it again on each page where it fits.

## The model

| Item | Value | Real model (Llama 3 8B) |
|---|---|---|
| Vocabulary | `the` = 0, `cat` = 1, `sat` = 2 | 128,256 tokens |
| Embeddings (one vector for each token) | `the` = [1, 0], `cat` = [0, 1], `sat` = [1, 1] | 4,096 numbers for each token |
| Sentence | "the cat sat" = ids `[0, 1, 2]` | Up to 8,192 tokens |
| Logits (scores for the next token) after "the cat" | `[0.1, 1.0, 2.0]` | 128,256 scores |
| One weight matrix $W$ | $\begin{bmatrix} 0.5 & -1.2 \\ 0.8 & 0.3 \end{bmatrix}$ | Hundreds of matrices of 4,096 × 4,096 |

Two numbers come back on many pages:

- The probabilities of the next token. Softmax of the logits gives `[0.099, 0.242, 0.659]`. The model thinks that `sat` comes after "the cat" with a probability of 65.9%.
- The loss when the correct next token is `sat`: $-\ln(0.659) = 0.417$.

## Where each part is used

| Part | Pages |
|---|---|
| Token ids | [LLM architecture](scientist/01-architecture.md), [NLP](fundamentals/04-nlp.md) |
| Embeddings | [LLM architecture](scientist/01-architecture.md) (attention), [Vector storage](engineer/02-vector-storage.md) (similarity search), [Inference optimization](engineer/06-inference-optimization.md) (key-value cache) |
| Logits and probabilities | [LLM architecture](scientist/01-architecture.md) (sampling), [Neural networks](fundamentals/03-neural-networks.md) (loss), [Mathematics](fundamentals/01-math.md) |
| Weight matrix $W$ | [Fine-tuning](scientist/04-fine-tuning.md) (LoRA), [Quantization](scientist/07-quantization.md), [Mathematics](fundamentals/01-math.md) |

Some pages cannot use this model (for example, deployment or security). Those pages use their own small example with real numbers.

## In code

The file `examples/toy.py` in the repository holds these values. Each number on each page comes from a script in `examples/`, and a test checks it.

```python
import numpy as np

VOCAB = ["the", "cat", "sat"]
E = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])   # embeddings
LOGITS = np.array([0.1, 1.0, 2.0])                   # after "the cat"
W = np.array([[0.5, -1.2], [0.8, 0.3]])              # one weight matrix

probs = np.exp(LOGITS) / np.exp(LOGITS).sum()        # [0.099, 0.242, 0.659]
loss = -np.log(probs[2])                             # 0.417
```
