"""Each number shown in content/fundamentals/04-nlp.md."""

import numpy as np

from examples.fundamentals.nlp import (
    DOCS, FORGET_GATE, GPT2_VOCAB, LONG_TEXT, ONE_HOT, RAW_DOCS, TERMS, U,
    bag_of_words, bigram_counts, bigram_probability, bpe_trace, cosine, drop_punctuation,
    drop_stop_words, gradient_norms, layer_parameters, lemmatize, rnn_forward, rnn_jacobian,
    skipgram_pairs, skipgram_step, stem, tf_idf, tokenize,
)
from examples.scientist.architecture import BPE_CORPUS
from examples.toy import E, W


def test_tokenize():
    assert tokenize(RAW_DOCS[0]) == ["the", "cat", "sat", "."]
    assert tokenize(RAW_DOCS[2]) == ["the", "cats", "sat", "on", "the", "cat", "."]
    assert drop_punctuation(tokenize(RAW_DOCS[2])) == ["the", "cats", "sat", "on", "the", "cat"]


def test_stem_and_lemmatize():
    assert stem("cats") == "cat" and stem("sat") == "sat"
    assert lemmatize("cats") == "cat" and lemmatize("sat") == "sit"
    assert DOCS == [["the", "cat", "sat"], ["the", "dog", "sat"], ["the", "cat", "sat", "on", "the", "cat"]]


def test_stop_words():
    words = [drop_punctuation(tokenize(text)) for text in RAW_DOCS]
    kept = [drop_stop_words(doc) for doc in words]  # step 2 comes before stemming: "cats" stays
    assert kept == [["cat", "sat"], ["dog", "sat"], ["cats", "sat", "cat"]]
    assert (sum(map(len, words)), sum(map(len, kept))) == (12, 7)


def test_bpe_vocabulary_against_length():
    rows = bpe_trace(BPE_CORPUS)
    assert [row["merge"] for row in rows[1:]] == [("a", "t"), ("c", "at"), ("t", "h"), ("th", "e"), ("s", "at")]
    assert [row["vocab"] for row in rows] == [6, 7, 8, 9, 10, 11]
    assert [row["tokens"] for row in rows] == [12, 9, 7, 6, 5, 4]


def test_bag_of_words():
    assert TERMS == ["cat", "dog", "on", "sat", "the"]
    counts = bag_of_words(DOCS, TERMS)
    assert counts.tolist() == [[1, 0, 0, 1, 1], [0, 1, 0, 1, 1], [2, 0, 1, 1, 2]]
    assert bag_of_words([["the", "cat", "sat"]], TERMS).tolist() == bag_of_words([["sat", "the", "cat"]], TERMS).tolist()


def test_tf_idf():
    counts = bag_of_words(DOCS, TERMS)
    features = tf_idf(counts)
    assert features["df"].tolist() == [2, 1, 1, 3, 3]
    assert np.allclose(features["idf"], [0.405, 1.099, 1.099, 0, 0], atol=5e-4)
    assert np.allclose(features["tf"][2], [0.333, 0, 0.167, 0.167, 0.333], atol=5e-4)
    assert np.allclose(features["tfidf"], [[0.135, 0, 0, 0, 0], [0, 0.366, 0, 0, 0], [0.135, 0, 0.183, 0, 0]], atol=5e-4)
    assert round(cosine(counts[0], counts[1]), 3) == 0.667
    assert cosine(features["tfidf"][0], features["tfidf"][1]) == 0
    assert round(cosine(features["tfidf"][0], features["tfidf"][2]), 3) == 0.594


def test_bigrams():
    counts = bigram_counts(DOCS)
    assert counts[("the", "cat")] == 3 and counts[("cat", "sat")] == 2 and counts[("the", "dog")] == 1
    assert len(counts) == 6
    assert bigram_probability(counts, "the", "cat") == 0.75
    assert bigram_probability(counts, "the", "dog") == 0.25
    assert list(zip(DOCS[0], DOCS[0][1:])) == [("the", "cat"), ("cat", "sat")]
    assert len(TERMS) ** 2 == 25 and GPT2_VOCAB ** 2 == 2_525_766_049


def test_cosine_similarity():
    assert cosine(E[0], E[1]) == 0
    assert round(cosine(E[0], E[2]), 3) == 0.707
    assert round(cosine(E[1], E[2]), 3) == 0.707
    assert all(cosine(ONE_HOT[a], ONE_HOT[b]) == 0 for a, b in ((0, 1), (0, 2), (1, 2)))


def test_skipgram_step():
    assert skipgram_pairs(DOCS[0]) == [("the", "cat"), ("cat", "the"), ("cat", "sat"), ("sat", "cat")]
    step = skipgram_step(center=1, context=2, lr=0.5)
    assert step["scores"].tolist() == [0, 1, 1]
    assert np.allclose(step["probs"], [0.155, 0.422, 0.422], atol=5e-4)
    assert round(step["loss"], 3) == 0.862
    assert np.allclose(step["expected_context"], [0.578, 0.845], atol=5e-4)
    assert np.allclose(step["grad"], [-0.422, -0.155], atol=5e-4)
    assert np.allclose(step["v_new"], [0.211, 1.078], atol=5e-4)
    assert np.allclose(step["probs_new"], [0.158, 0.377, 0.465], atol=5e-4)
    assert round(step["loss_new"], 3) == 0.765


def test_rnn_forward():
    states = rnn_forward(E)
    assert np.allclose(states, [[0, 0], [0.762, 0], [0.363, 0.923], [0.074, 0.917]], atol=5e-4)
    assert np.allclose(W @ states[1], [0.381, 0.609], atol=5e-4)
    assert np.allclose(W @ states[2], [-0.926, 0.568], atol=5e-4)
    assert np.allclose(W @ states[1] + U @ E[1], [0.381, 1.609], atol=5e-4)
    assert np.allclose(W @ states[2] + U @ E[2], [0.074, 1.568], atol=5e-4)


def test_rnn_gradient_vanishes():
    states = rnn_forward(E)
    assert np.allclose(1 - states[3] ** 2, [0.995, 0.160], atol=5e-4)
    assert np.allclose(rnn_jacobian(states[3]), [[0.497, -1.193], [0.128, 0.048]], atol=5e-4)
    assert len(LONG_TEXT) == 60
    norms = gradient_norms(rnn_forward(LONG_TEXT))
    assert round(norms[1], 3) == 1.296 and round(norms[10], 3) == 0.338 and round(norms[30], 3) == 0.011
    assert round(norms[59] * 1e5, 1) == 9.7  # 0.000097
    assert 10_000 < norms[0] / norms[59] < 11_000  # "about 10,000 times smaller"
    assert round(FORGET_GATE ** 10, 3) == 0.904 and round(FORGET_GATE ** 59, 3) == 0.553


def test_layer_parameters():
    assert layer_parameters(2, 2) == {"rnn": 10, "gru": 30, "lstm": 40}
    assert layer_parameters(2, 2, biases=2) == {"rnn": 12, "gru": 36, "lstm": 48}
