"""Numbers and plots for docs/fundamentals/04-nlp.md.

Run: uv run python -m examples.fundamentals.nlp
"""

import re
from collections import Counter

import numpy as np

from examples.plotstyle import save_light_dark
from examples.scientist.architecture import BPE_CORPUS, bpe_merge, bpe_pair_counts
from examples.toy import VOCAB, E, W, softmax

PAGE = "nlp"

# --- Text preprocessing --------------------------------------------------------------------

RAW_DOCS = ["The cat sat.", "The dog sat.", "The cats sat on the cat."]
STOP_WORDS = {"the", "on"}
LEMMAS = {"cats": "cat", "sat": "sit"}  # a lemmatizer needs a dictionary


def tokenize(text: str) -> list[str]:
    """Lowercase the text, then split it into words and punctuation marks."""
    return re.findall(r"[a-z]+|[^a-z\s]", text.lower())


def drop_punctuation(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token.isalpha()]


def drop_stop_words(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in STOP_WORDS]


def stem(word: str) -> str:
    """Toy stemmer with one rule: cut a final "s" from a word of 4 letters or more."""
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def lemmatize(word: str) -> str:
    """Toy lemmatizer: look up the dictionary form of the word."""
    return LEMMAS.get(word, word)


def preprocess(text: str) -> list[str]:
    """The tokens that the feature extraction steps use: lowercase, no punctuation, stemmed."""
    return [stem(token) for token in drop_punctuation(tokenize(text))]


DOCS = [preprocess(text) for text in RAW_DOCS]
TERMS = sorted({term for doc in DOCS for term in doc})  # cat, dog, on, sat, the


# --- Byte-pair encoding: vocabulary size against text length -------------------------------

def bpe_trace(corpus: dict[str, int]) -> list[dict]:
    """Merge until no pair is left. One row for each state: merge, vocabulary size, token count."""
    words = {tuple(word): count for word, count in corpus.items()}
    vocab = {symbol for symbols in words for symbol in symbols}
    rows = [{"merge": None, "vocab": len(vocab),
             "tokens": sum(len(symbols) * count for symbols, count in words.items())}]
    while counts := bpe_pair_counts(words):
        best = max(counts, key=lambda pair: (counts[pair], pair))  # same tie rule as the architecture page
        words = bpe_merge(words, best)
        vocab.add(best[0] + best[1])
        rows.append({"merge": best, "vocab": len(vocab),
                     "tokens": sum(len(symbols) * count for symbols, count in words.items())})
    return rows


# --- Feature extraction: bag-of-words, TF-IDF, n-grams -------------------------------------

def bag_of_words(docs: list[list[str]], terms: list[str]) -> np.ndarray:
    """One row for each document, one column for each term: the count of the term."""
    return np.array([[doc.count(term) for term in terms] for doc in docs])


def tf_idf(counts: np.ndarray) -> dict[str, np.ndarray]:
    """tf = count / document length. idf = ln(N / number of documents with the term)."""
    tf = counts / counts.sum(axis=1, keepdims=True)
    df = (counts > 0).sum(axis=0)
    idf = np.log(len(counts) / df)
    return {"tf": tf, "df": df, "idf": idf, "tfidf": tf * idf}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def bigram_counts(docs: list[list[str]]) -> Counter:
    return Counter(pair for doc in docs for pair in zip(doc, doc[1:]))


GPT2_VOCAB = 50_257  # vocabulary size of GPT-2 (see the architecture page)


def bigram_probability(counts: Counter, first: str, second: str) -> float:
    """P(second | first) = count(first second) / count(first, followed by a word)."""
    total = sum(count for (a, _), count in counts.items() if a == first)
    return counts[(first, second)] / total


# --- Word embeddings: cosine similarity and one skip-gram step -----------------------------

ONE_HOT = np.eye(len(VOCAB))


def skipgram_pairs(tokens: list[str], window: int = 1) -> list[tuple[str, str]]:
    """Each (center word, context word) pair: the context is at most `window` positions away."""
    return [(center, tokens[j])
            for i, center in enumerate(tokens)
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)) if j != i]


def skipgram_step(center: int, context: int, lr: float = 0.5) -> dict:
    """One gradient step on the vector of the center word for one (center, context) pair.

    Assumption of the page: the toy embeddings are the center vectors and the context vectors.
    """
    v = E[center]
    scores = E @ v
    probs = softmax(scores)
    grad = probs @ E - E[context]  # d(-ln p_context) / d v
    v_new = v - lr * grad
    return {"scores": scores, "probs": probs, "loss": float(-np.log(probs[context])),
            "expected_context": probs @ E, "grad": grad,
            "v_new": v_new, "probs_new": softmax(E @ v_new),
            "loss_new": float(-np.log(softmax(E @ v_new)[context]))}


# --- Recurrent neural network --------------------------------------------------------------

U = np.eye(2)  # input matrix; the page assumes the identity to keep the numbers small


def rnn_forward(xs: np.ndarray) -> np.ndarray:
    """h_t = tanh(W h_{t-1} + U x_t), with h_0 = 0. Returns h_0 ... h_T."""
    states = [np.zeros(2)]
    for x in xs:
        states.append(np.tanh(W @ states[-1] + U @ x))
    return np.array(states)


def rnn_jacobian(h: np.ndarray) -> np.ndarray:
    """d h_t / d h_{t-1} = diag(1 - h_t^2) W."""
    return np.diag(1 - h ** 2) @ W


def gradient_norms(states: np.ndarray) -> np.ndarray:
    """Norm of d h_T / d h_{T-k} for k = 0 ... T-1 (largest singular value)."""
    product = np.eye(2)
    norms = [1.0]
    for h in states[:0:-1][:-1]:  # h_T, h_{T-1}, ..., h_2
        product = product @ rnn_jacobian(h)
        norms.append(float(np.linalg.norm(product, 2)))
    return np.array(norms)


LONG_TEXT = np.tile(E, (20, 1))  # "the cat sat" 20 times: 60 tokens
FORGET_GATE = 0.99  # assumed value of the LSTM forget gate


def layer_parameters(inputs: int, hidden: int, biases: int = 1) -> dict[str, int]:
    """Weights and `biases` bias vectors for each group of equations: RNN 1, GRU 3, LSTM 4."""
    one = hidden * (hidden + inputs) + biases * hidden
    return {"rnn": one, "gru": 3 * one, "lstm": 4 * one}


# --- Plot ----------------------------------------------------------------------------------

def make_figures() -> None:
    norms = gradient_norms(rnn_forward(LONG_TEXT))
    steps = np.arange(len(norms))

    def draw(ax, theme):
        ax.semilogy(steps, norms, color=theme["series"][0], marker="o", markersize=3, label="RNN (toy W)")
        ax.semilogy(steps, FORGET_GATE ** steps, color=theme["series"][1],
                    label=f"LSTM cell path (f = {FORGET_GATE})")
        ax.set_xlabel("Distance k in tokens")
        ax.set_ylabel("Size of the gradient (log scale)")
        ax.legend(loc="lower left")

    save_light_dark(draw, PAGE, "gradient")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    for text in RAW_DOCS:
        tokens = tokenize(text)
        words = drop_punctuation(tokens)
        print(text, "->", tokens, "| no stop words:", drop_stop_words(words),
              "| stems:", [stem(w) for w in words], "| lemmas:", [lemmatize(w) for w in words])
    for row in bpe_trace(BPE_CORPUS):
        print("bpe:", row)

    counts = bag_of_words(DOCS, TERMS)
    features = tf_idf(counts)
    print("terms:", TERMS)
    print("bag of words:", counts, sep="\n")
    for name, value in features.items():
        print(name, value, sep="\n")
    print("cosine D1-D2, counts:", cosine(counts[0], counts[1]), "tf-idf:", cosine(features["tfidf"][0], features["tfidf"][1]))
    print("cosine D1-D3, counts:", cosine(counts[0], counts[2]), "tf-idf:", cosine(features["tfidf"][0], features["tfidf"][2]))
    bigrams = bigram_counts(DOCS)
    print("bigrams:", dict(bigrams))
    print("P(cat | the):", bigram_probability(bigrams, "the", "cat"), "P(dog | the):", bigram_probability(bigrams, "the", "dog"))
    print("bag of words 'the cat sat' == 'sat the cat':",
          bag_of_words([["the", "cat", "sat"]], TERMS).tolist() == bag_of_words([["sat", "the", "cat"]], TERMS).tolist())

    for a, b in ((0, 1), (0, 2), (1, 2)):
        print(f"cosine {VOCAB[a]}-{VOCAB[b]}: toy {cosine(E[a], E[b]):.3f}, one-hot {cosine(ONE_HOT[a], ONE_HOT[b]):.3f}")
    print("skip-gram pairs:", skipgram_pairs(DOCS[0]))
    for name, value in skipgram_step(center=1, context=2).items():
        print("skip-gram", name, value)

    states = rnn_forward(E)
    print("rnn states:", states, sep="\n")
    for t in (1, 2, 3):
        print(f"step {t}: W h =", W @ states[t - 1], "pre-activation =", W @ states[t - 1] + U @ E[t - 1])
    print("jacobian h3/h2:", rnn_jacobian(states[3]), sep="\n")
    norms = gradient_norms(rnn_forward(LONG_TEXT))
    for k in (1, 2, 10, 30, 59):
        print(f"k={k}: rnn gradient {norms[k]:.3e}, lstm cell path {FORGET_GATE ** k:.3f}")
    print("tanh'(h) at h3:", 1 - states[3] ** 2)
    print("parameters:", layer_parameters(2, 2), "with 2 bias vectors:", layer_parameters(2, 2, biases=2))
    print("possible bigrams:", len(TERMS) ** 2, "and for 50,257 tokens:", GPT2_VOCAB ** 2)
    make_figures()
