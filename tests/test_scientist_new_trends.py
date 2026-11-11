"""Each number shown in content/scientist/08-new-trends.md."""

from math import comb

import numpy as np

from examples.scientist.new_trends import (
    ACTIVATION, CLIP_TEMPERATURE, DARE_DROP, DARE_MASK, IMAGES, MEAN_HARMFUL, MEAN_HARMLESS, SAE_BIAS,
    SAE_DIRECTIONS, SAE_L1, TAU_A, TAU_B, TEXTS, VOTE_P, W_B, ablate, clip_loss, dare, lerp,
    majority_vote, orthogonalize, patch_count, project_patch, sae, slerp, ties, unit_direction,
)
from examples.toy import E, W


def test_lerp_shrinks_the_matrix():
    assert round(float(np.linalg.norm(W)), 3) == 1.556 and round(float(np.linalg.norm(W_B)), 3) == 1.556
    half = lerp(W, W_B, 0.5)
    assert np.allclose(half, [[0.85, -0.45], [0.65, 0.55]])
    assert round(float(np.linalg.norm(half)), 3) == 1.285


def test_slerp_keeps_the_length():
    s = slerp(W, W_B, 0.5)
    assert np.isclose(W.ravel() @ W_B.ravel(), 0.88)
    assert np.isclose(np.linalg.norm(W) * np.linalg.norm(W_B), 2.42)
    assert round(s["cos"], 3) == 0.364
    assert round(float(np.degrees(s["theta"])), 1) == 68.7
    assert round(float(np.degrees(s["theta"] / 2)), 1) == 34.3
    assert round(float(np.sin(s["theta"] / 2)), 4) == 0.5641 and round(float(np.sin(s["theta"])), 4) == 0.9315
    assert round(s["coef_a"], 3) == 0.606 and round(s["coef_b"], 3) == 0.606
    assert np.allclose(s["merged"], [[1.029, -0.545], [0.787, 0.666]], atol=5e-4)
    assert round(float(np.linalg.norm(s["merged"])), 3) == 1.556


def test_ties_and_dare():
    assert np.allclose((TAU_A + TAU_B) / 2, [0.3, 0.1, -0.275, 0.11])
    t = ties([TAU_A, TAU_B], keep=2)
    assert np.allclose(t["trimmed"], [[0.4, -0.3, 0, 0], [0, 0.5, -0.6, 0]])
    assert t["elected"].tolist() == [1, 1, -1, 0]
    assert np.allclose(t["merged"], [0.4, 0.5, -0.6, 0])
    assert np.allclose(W.ravel() + t["merged"], [0.9, -0.7, 0.2, 0.3])
    kept = dare(TAU_A, DARE_MASK, DARE_DROP)
    assert np.allclose(kept, [0.8, 0, 0.1, 0])
    assert np.isclose((1 - DARE_DROP) * kept[0], TAU_A[0])


def test_clip_loss():
    c = clip_loss(IMAGES, TEXTS, CLIP_TEMPERATURE)
    assert np.allclose(np.linalg.norm(IMAGES, axis=1), 1) and np.allclose(np.linalg.norm(TEXTS, axis=1), 1)
    assert np.allclose(c["similarity"], [[0.8, 0], [0.6, 1.0]])
    assert np.allclose(c["logits"], [[1.6, 0], [1.2, 2.0]])
    assert np.allclose(np.exp(c["logits"][1]), [3.320, 7.389], atol=5e-4)
    assert np.allclose(c["image_to_text"], [[0.832, 0.168], [0.310, 0.690]], atol=5e-4)
    assert np.allclose(c["text_to_image"], [[0.599, 0.119], [0.401, 0.881]], atol=5e-4)
    assert round(c["loss_i2t"], 3) == 0.278 and round(c["loss_t2i"], 3) == 0.320
    assert round(c["loss"], 3) == 0.299
    nll_rows = -np.log(np.diag(c["image_to_text"]))
    nll_cols = -np.log(np.diag(c["text_to_image"]))
    assert np.allclose(nll_rows, [0.184, 0.371], atol=5e-4) and np.allclose(nll_cols, [0.513, 0.127], atol=5e-4)


def test_projector_and_patch_count():
    assert np.allclose(project_patch(IMAGES[0], W), [-0.32, 0.82])
    assert patch_count(336, 14) == 576 and 336 // 14 == 24
    assert np.vstack([project_patch(IMAGES[0], W), E[0], E[1]]).shape == (3, 2)


def test_sparse_autoencoder():
    a = sae(ACTIVATION, SAE_DIRECTIONS, SAE_BIAS, SAE_L1)
    assert np.allclose(SAE_DIRECTIONS @ ACTIVATION, [0.2, 1.0, 0.92])
    assert np.allclose(a["pre"], [-0.3, 0.5, 0.42])
    assert np.allclose(a["features"], [0, 0.5, 0.42])
    assert np.allclose(a["recon"], [0.252, 0.836])
    assert np.allclose(ACTIVATION - a["recon"], [-0.052, 0.164])
    assert round(a["mse"], 4) == 0.0296 and round(a["l1"], 2) == 0.92
    assert round(a["loss"], 3) == 0.122


def test_abliteration():
    r = unit_direction(MEAN_HARMFUL, MEAN_HARMLESS)
    assert np.allclose(MEAN_HARMFUL - MEAN_HARMLESS, [0.6, 0.8]) and np.allclose(r, [0.6, 0.8])
    ab = ablate(ACTIVATION, r)
    assert round(ab["amount"], 2) == 0.92
    assert np.allclose(ab["projection"], [0.552, 0.736])
    assert np.allclose(ab["result"], [-0.352, 0.264])
    assert np.isclose(ab["result"] @ r, 0)
    w_orth = orthogonalize(W, r)
    assert np.allclose(r @ W, [0.94, -0.48])
    assert np.allclose(w_orth, [[-0.064, -0.912], [0.048, 0.684]])
    assert np.allclose(r @ w_orth, 0)


def test_majority_vote():
    assert majority_vote(VOTE_P, 1) == 0.6
    assert round(majority_vote(VOTE_P, 3), 3) == 0.648
    assert round(majority_vote(VOTE_P, 5), 3) == 0.683
    assert np.isclose(VOTE_P ** 3, 0.216) and np.isclose(3 * VOTE_P ** 2 * (1 - VOTE_P), 0.432)
    terms = [comb(5, k) * VOTE_P ** k * (1 - VOTE_P) ** (5 - k) for k in (3, 4, 5)]
    assert np.allclose(terms, [0.3456, 0.2592, 0.0778], atol=5e-5)
    assert round(100 * (majority_vote(VOTE_P, 5) - VOTE_P)) == 8  # points gained with 5 samples
    assert round(majority_vote(VOTE_P, 41), 3) == 0.903
    assert round(majority_vote(0.4, 5), 3) == 0.317 and round(majority_vote(0.4, 41), 3) == 0.097
    assert round(majority_vote(0.8, 5), 3) == 0.942
