"""Each number shown in content/scientist/01-architecture.md."""

import numpy as np

from examples.scientist.architecture import (
    BPE_CORPUS, attention, beam_scores, bpe_train, top_k, top_p,
)
from examples.toy import LOGITS, E, encode, softmax


def test_token_ids():
    assert encode("the cat sat") == [0, 1, 2]


def test_bpe_merges():
    merges, history = bpe_train(BPE_CORPUS, 3)
    assert merges == [("a", "t"), ("c", "at"), ("t", "h")]
    assert history[0][("a", "t")] == 3 and history[0][("c", "a")] == 2
    assert history[1][("c", "at")] == 2 and history[1][("s", "at")] == 1


def test_attention():
    result = attention(E)
    assert np.allclose(result["raw"], [[1, 0, 1], [0, 1, 1], [1, 1, 2]])
    assert np.allclose(result["scores"][2], [0.707, 0.707, 1.414], atol=5e-4)
    assert np.allclose(result["weights"], [[1, 0, 0], [0.330, 0.670, 0], [0.248, 0.248, 0.503]], atol=5e-4)
    assert np.allclose(result["output"], [[1, 0], [0.330, 0.670], [0.752, 0.752]], atol=5e-4)


def test_softmax_and_temperature():
    assert np.allclose(softmax(LOGITS), [0.099, 0.242, 0.659], atol=5e-4)
    assert np.allclose(softmax(LOGITS, 0.5), [0.019, 0.117, 0.864], atol=5e-4)
    assert np.allclose(softmax(LOGITS, 2.0), [0.194, 0.304, 0.502], atol=5e-4)


def test_top_k_and_top_p():
    probs = softmax(LOGITS)
    assert np.allclose(top_k(probs, 2), [0, 0.269, 0.731], atol=5e-4)
    assert np.allclose(top_p(probs, 0.9), [0, 0.269, 0.731], atol=5e-4)
    assert np.allclose(top_p(probs, 0.5), [0, 0, 1])


def test_beam_search_beats_greedy():
    scored = dict(beam_scores(softmax(LOGITS)))
    assert round(scored["sat the"], 3) == 0.224  # greedy path
    assert round(scored["cat sat"], 3) == 0.230  # best path
    assert max(scored, key=scored.get) == "cat sat"
