"""Each number shown in content/datasets/04-tools.md."""

import numpy as np
import pytest

from examples.datasets.tools import (
    KEEP_RATE, PAGES, QUOTE_SAMPLE, after_deduplication, after_rules, cosine_matrix, counts_by_topic,
    crawl_estimate, decode_tokens_per_second, exact_duplicates, funnel, fuzzy_deduplicate, generation_plan,
    kept_pages, missing_by_topic, page_yield, pair_count, pairs_above, refusal_flags, rule_precision,
    to_generate, total_yield, two_means, COMPARISONS_PER_SECOND,
)
from examples.datasets.toy_dataset import SAMPLES


def test_yield_of_each_page():
    yields = [round(page_yield(raw, clean), 3) for _, raw, clean in PAGES]
    assert yields == [0.3, 0.2, 0.05, 0.02, 0.47]
    assert sum(raw for _, raw, _ in PAGES) == 150_000
    assert sum(clean for _, _, clean in PAGES) == 25_000
    assert round(total_yield(), 3) == 0.167


def test_minimum_length_rule_drops_the_login_page():
    kept = kept_pages()
    assert [name for name, _, _ in kept] == ["Long article", "Short article", "Forum thread", "Documentation page"]
    assert sum(clean for _, _, clean in kept) == 24_700
    assert round(300 / 25_000, 3) == 0.012  # the dropped page has 1.2% of the clean text


def test_crawl_of_one_million_pages():
    crawl = crawl_estimate()
    assert crawl["kept_share"] == 0.8
    assert crawl["kept_pages"] == 800_000
    assert crawl["characters_per_page"] == 6_175
    assert crawl["tokens_per_page"] == 1_543.75
    assert crawl["tokens"] == 1_235_000_000


def test_refusal_rule_and_its_false_positive():
    assert refusal_flags() == [8]
    assert [sample.id for sample in after_rules()] == [1, 2, 3, 4, 5, 6, 7]
    assert refusal_flags(SAMPLES + [QUOTE_SAMPLE]) == [8, 9]
    assert rule_precision() == 0.5


def test_exact_deduplication_finds_nothing():
    assert exact_duplicates() == []
    assert len(SAMPLES[2].instruction) - len(SAMPLES[0].instruction) == 1  # samples 1 and 3: one character


def test_cosine_similarities_shown_on_the_page():
    sim = cosine_matrix()
    shown = {(1, 2): 0.9929, (1, 3): 0.9998, (2, 3): 0.9953, (5, 6): 0.9929, (7, 8): 0.9971,
             (2, 4): 0.9701, (1, 4): 0.9345, (1, 5): 0.1980}
    for (a, b), value in shown.items():
        assert round(float(sim[a - 1, b - 1]), 4) == value
    assert np.allclose(np.diag(sim), 1.0)


def test_pairs_and_kept_samples_at_each_threshold():
    assert [(a, b) for a, b, _ in pairs_above(0.999)] == [(1, 3)]
    assert [(a, b) for a, b, _ in pairs_above(0.99)] == [(1, 2), (1, 3), (2, 3), (5, 6), (7, 8)]
    assert [(a, b) for a, b, _ in pairs_above(0.95)] == [(1, 2), (1, 3), (2, 3), (2, 4), (5, 6), (7, 8)]
    assert fuzzy_deduplicate(0.999) == ([1, 2, 4, 5, 6, 7, 8], {3: 1})
    assert fuzzy_deduplicate(0.99) == ([1, 4, 5, 7], {2: 1, 3: 1, 6: 5, 8: 7})
    assert fuzzy_deduplicate(0.95) == ([1, 4, 5, 7], {2: 1, 3: 1, 6: 5, 8: 7})
    assert fuzzy_deduplicate(0.9) == ([1, 5, 7], {2: 1, 3: 1, 4: 1, 6: 5, 8: 7})
    assert len(fuzzy_deduplicate(1.0)[0]) == 8


def test_number_of_comparisons():
    assert pair_count(8) == 28
    assert pair_count(10_000) == 49_995_000
    assert pair_count(1_000_000) == 499_999_500_000
    assert round(pair_count(10_000) / COMPARISONS_PER_SECOND, 1) == 5.0
    assert round(pair_count(1_000_000) / COMPARISONS_PER_SECOND / 3600, 1) == 13.9
    assert 1_000_000 * 10 == 10_000_000  # an index that compares each sample with 10 candidates


def test_samples_to_generate():
    survivors = after_deduplication()
    assert [sample.id for sample in survivors] == [1, 2, 4, 5, 6, 7]
    assert counts_by_topic(survivors) == {"math": 3, "code": 2, "chat": 1}
    missing = missing_by_topic(survivors)
    assert missing == {"math": 0, "code": 1, "chat": 2}
    assert sum(missing.values()) == 3
    assert to_generate(3) == 5
    assert to_generate(3) * KEEP_RATE == 3.0


def test_generation_at_scale():
    plan = generation_plan()
    assert plan["generated"] == 16_667
    assert plan["input_tokens"] == 5_000_100
    assert plan["output_tokens"] == 8_333_500
    assert decode_tokens_per_second(1) == pytest.approx(50.0)
    assert round(plan["one_by_one_hours"], 1) == 46.3
    assert round(plan["batched_tokens_per_second"]) == 1_963
    assert round(plan["batched_minutes"], 1) == 70.7
    assert round(plan["one_by_one_hours"] * 60 / plan["batched_minutes"], 1) == 39.3
    assert round(plan["dollars"], 2) == 19.17
    assert round(plan["dollars_per_kept_sample"], 3) == 0.002
    assert round(plan["dollars_with_batch_discount"], 2) == 9.58
    assert round(plan["output_tokens_per_kept_sample"], 1) == 833.3


def test_two_clusters_and_the_outlier():
    result = two_means()
    assert result["clusters"] == [[1, 2, 3, 4, 8], [5, 6, 7]]
    assert np.round(result["centres"], 3).tolist() == [[0.836, 0.264], [0.3, 0.867]]
    assert result["outlier"] == 8
    assert round(float(result["distances"][7]), 3) == 0.411
    assert round(float(result["distances"][6]), 3) == 0.343
    assert result["rounds"] == 1


def test_funnel_of_the_lesson():
    steps = funnel()
    assert [count for _, count, _ in steps] == [8, 7, 6, 9]
    assert steps[0][2] == {"math": 4, "code": 2, "chat": 2}
    assert steps[-1][2] == {"math": 3, "code": 3, "chat": 3}
    assert [round(c / 8, 3) for c in steps[0][2].values()] == [0.5, 0.25, 0.25]
    assert [round(c / 9, 3) for c in steps[-1][2].values()] == [0.333, 0.333, 0.333]


def test_worked_cosine_and_error_counts():
    a, b = np.array(SAMPLES[0].embedding), np.array(SAMPLES[2].embedding)
    assert round(float(a @ b), 3) == 0.992
    assert round(float(np.linalg.norm(a)), 3) == 1.005
    assert round(float(np.linalg.norm(b)), 3) == 0.987
    _, removed = fuzzy_deduplicate(0.99)
    assert sorted(removed) == [2, 3, 6, 8]  # only sample 3 is a near duplicate: 3 of 4 are false positives
    assert round(pair_count(1_000_000) / pair_count(10_000)) == 10_001
    assert round(pair_count(1_000_000) / 10_000_000) == 50_000
    assert round(1 - total_yield(), 3) == round(5 / 6, 3)
