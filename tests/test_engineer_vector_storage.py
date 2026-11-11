"""Each number shown in docs/engineer/02-vector-storage.md."""

import numpy as np

from examples.engineer.corpus import CHUNKS, WORD_VECTORS, chunk_vectors, embed, normalize, tokenize
from examples.engineer.vector_storage import (
    QUERY, QUERY_2, RAW_DOCUMENT, body_text, build_chunks, clean, fixed_chunk_count, greedy_walk,
    ivf_recall, ivf_search, kmeans, load, ranking, scale_numbers, similarity_table, split_by_header, split_fixed,
    split_recursive, top_k,
)
from examples.toy import E

VECTORS = chunk_vectors()
UNIT = normalize(VECTORS)


def test_word_table_extends_the_toy():
    assert np.array_equal(WORD_VECTORS["the"], E[0])
    assert np.array_equal(WORD_VECTORS["cat"], E[1])
    assert np.array_equal(WORD_VECTORS["sat"], E[2])
    assert len(WORD_VECTORS) == 17


def test_load_and_clean():
    assert load()["metadata"] == {"source": "cat.md"}
    cleaned = clean(RAW_DOCUMENT)
    assert len(RAW_DOCUMENT) == 112 and len(cleaned) == 96
    assert "Page 1 of 1" not in cleaned and "   " not in cleaned
    assert body_text(cleaned) == "The cat sat on the mat. The cat sleeps all day.\n\nThe cat eats fish."
    assert len(body_text(cleaned).split()) == 15


def test_split_fixed_size():
    body = body_text(clean(RAW_DOCUMENT))
    assert split_fixed(body, size=6, overlap=2) == [
        "The cat sat on the mat.",
        "the mat. The cat sleeps all",
        "sleeps all day. The cat eats",
        "cat eats fish.",
    ]
    assert (15 - 2) / (6 - 2) == 3.25 and fixed_chunk_count(15, 6, 2) == 4
    for size, overlap in [(6, 2), (6, 0), (5, 0), (4, 1), (15, 0), (3, 2)]:
        assert len(split_fixed(body, size, overlap)) == fixed_chunk_count(15, size, overlap)


def test_split_recursive_and_by_header():
    cleaned = clean(RAW_DOCUMENT)
    assert split_recursive(body_text(cleaned), size=6) == [
        "The cat sat on the mat.", "The cat sleeps all day.", "The cat eats fish.",
    ]
    assert split_by_header(cleaned) == [
        ("Habits", "The cat sat on the mat. The cat sleeps all day."),
        ("Food", "The cat eats fish."),
    ]
    assert [len(body.split()) for _, body in split_by_header(cleaned)] == [11, 4]


def test_pipeline_builds_the_shared_corpus():
    assert build_chunks() == CHUNKS
    assert [(c.id, c.section) for c in CHUNKS] == [("c1", "Habits"), ("c2", "Habits"), ("c3", "Food")]


def test_embeddings():
    assert np.allclose(embed(QUERY), [0.667, 0.667], atol=5e-4)
    sums = [np.sum([WORD_VECTORS[word] for word in tokenize(chunk.text)], axis=0) for chunk in CHUNKS]
    assert np.array_equal(sums, [[6, 4], [2, 5], [1, 6]])
    assert [len(tokenize(chunk.text)) for chunk in CHUNKS] == [6, 5, 4]
    assert np.allclose(VECTORS, [[1, 0.667], [0.4, 1.0], [0.25, 1.5]], atol=5e-4)
    assert np.allclose(np.linalg.norm(VECTORS, axis=1), [1.202, 1.077, 1.521], atol=5e-4)
    assert round(float(np.linalg.norm(embed(QUERY))), 3) == 0.943
    assert np.allclose(normalize(embed(QUERY)), [0.707, 0.707], atol=5e-4)
    assert np.allclose(UNIT, [[0.832, 0.555], [0.371, 0.928], [0.164, 0.986]], atol=5e-4)


def test_three_similarity_measures():
    table = similarity_table(embed(QUERY), VECTORS)
    assert np.allclose(table["dot"], [1.111, 0.933, 1.167], atol=5e-4)
    assert np.allclose(table["cosine"], [0.981, 0.919, 0.814], atol=5e-4)
    assert np.allclose(table["euclidean"], [0.333, 0.427, 0.932], atol=5e-4)
    assert ranking(table["dot"]) == ["c3", "c1", "c2"]
    assert ranking(table["cosine"]) == ["c1", "c2", "c3"]
    assert ranking(table["euclidean"], largest_first=False) == ["c1", "c2", "c3"]


def test_normalized_vectors_make_dot_equal_cosine():
    raw = similarity_table(embed(QUERY), VECTORS)
    unit = similarity_table(normalize(embed(QUERY)), UNIT)
    assert np.allclose(unit["dot"], raw["cosine"])
    assert np.allclose(unit["euclidean"], [0.197, 0.402, 0.610], atol=5e-4)
    assert np.allclose(unit["euclidean"] ** 2, 2 - 2 * unit["dot"])
    assert ranking(unit["dot"]) == ranking(unit["euclidean"], largest_first=False) == ["c1", "c2", "c3"]


def test_exact_top_k():
    best = top_k(embed(QUERY), VECTORS, 2)
    assert [chunk_id for chunk_id, _ in best] == ["c1", "c2"]
    assert [round(score, 3) for _, score in best] == [0.981, 0.919]
    assert VECTORS.size == 6  # N x d = 3 x 2 multiplications for one query


def test_ivf_index():
    centroids, assignment = kmeans(UNIT, start=[0, 2])
    assert assignment.tolist() == [0, 1, 1]
    assert np.allclose(centroids, [[0.832, 0.555], [0.269, 0.963]], atol=5e-4)
    # Query 1: the nearest cluster holds the true best chunk.
    assert np.allclose(centroids @ normalize(embed(QUERY)), [0.981, 0.871], atol=5e-4)
    assert ivf_search(embed(QUERY), UNIT, centroids, assignment, nprobe=1) == ("c1", 3)
    # Query 2: the true best chunk is c2, but the nearest centroid is the one of c1.
    q2 = embed(QUERY_2)
    assert np.allclose(q2, [0.6, 0.8])
    exact = top_k(q2, VECTORS, 2)
    assert [chunk_id for chunk_id, _ in exact] == ["c2", "c1"]
    assert [round(score, 3) for _, score in exact] == [0.966, 0.943]
    assert np.allclose(centroids @ normalize(q2), [0.943, 0.932], atol=5e-4)
    assert ivf_search(q2, UNIT, centroids, assignment, nprobe=1) == ("c1", 3)
    assert ivf_search(q2, UNIT, centroids, assignment, nprobe=2) == ("c2", 5)
    assert ivf_recall([QUERY, QUERY_2], VECTORS, nprobe=1) == 0.5
    assert ivf_recall([QUERY, QUERY_2], VECTORS, nprobe=2) == 1.0


def test_graph_walk():
    walk = greedy_walk(embed(QUERY), UNIT, entry="c3")
    assert [node for node, _ in walk] == ["c3", "c2", "c1"]
    assert [round(score, 3) for _, score in walk] == [0.814, 0.919, 0.981]
    walk_2 = greedy_walk(embed(QUERY_2), UNIT, entry="c3")
    assert [node for node, _ in walk_2] == ["c3", "c2"]


def test_scale_numbers():
    numbers = scale_numbers()
    assert numbers["exact_multiplications"] == 1_024_000_000
    assert numbers["vector_gb"] == 4.096
    assert round(numbers["vector_gib"], 2) == 3.81
    assert numbers["hnsw_link_gb"] == 0.128
    assert numbers["hnsw_total_gb"] == 4.224
    assert numbers["ivf_comparisons"] == 8_000
    assert numbers["ivf_fraction"] == 0.008
