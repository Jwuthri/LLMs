"""Each number and each rendered string shown in docs/scientist/03-datasets.md."""

from examples.scientist.datasets import (
    A, B, BAD_OUTPUT, C, CANDIDATES, D, FUNCTION_NAMES, GENERATED, MESSAGES, POOL, SEED_TASKS, SHAREGPT,
    UNIT_TESTS, WORDS, all_prompts, chance_of_one_good, chatml_pieces, filter_pool, generator_prompt,
    is_refusal, jaccard, long_enough, minhash_estimate, minhash_ranks, minhash_signature, parse_sample,
    rejection_sampling, render_alpaca, render_chatml, run_tests, sharegpt_to_messages, standard_error,
    to_messages,
)


def test_sharegpt_maps_to_messages():
    assert sharegpt_to_messages(SHAREGPT) == MESSAGES


def test_chatml_exact_string():
    assert render_chatml(MESSAGES["messages"]) == (
        "<|im_start|>system\n"
        "You are a concise assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Count the words: the cat sat<|im_end|>\n"
        "<|im_start|>assistant\n"
        "3<|im_end|>\n"
    )


def test_chatml_character_counts():
    lengths = [[len(piece) for piece in chatml_pieces(message)] for message in MESSAGES["messages"]]
    assert lengths == [[12, 6, 1, 28, 10, 1], [12, 4, 1, 28, 10, 1], [12, 9, 1, 1, 10, 1]]
    assert [sum(row) for row in lengths] == [58, 56, 34]
    assert len(render_chatml(MESSAGES["messages"])) == 148
    assert sum(len(message["content"]) for message in MESSAGES["messages"]) == 57
    assert 148 - 57 == 91


def test_generation_prompt_and_alpaca():
    prompt = render_chatml(MESSAGES["messages"][:2], add_generation_prompt=True)
    assert prompt.endswith("Count the words: the cat sat<|im_end|>\n<|im_start|>assistant\n")
    assert render_alpaca("Count the words: the cat sat", "3") == (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nCount the words: the cat sat\n\n### Response:\n3"
    )


def test_generation_pipeline():
    assert len(SEED_TASKS) == 3 and len(all_prompts()) == 12
    prompt = generator_prompt(SEED_TASKS[0])
    assert [message["role"] for message in prompt] == ["system", "user"]
    assert prompt[1]["content"].startswith("Seed task: count the words in a text.")
    samples = [parse_sample(raw) for raw in GENERATED]
    assert all(sample is not None for sample in samples) and len(samples) == 3
    assert parse_sample(BAD_OUTPUT) is None
    assert render_chatml(to_messages(samples[0])["messages"]) == (
        "<|im_start|>user\n"
        "Write a Python function count_words(text) that returns the number of words in text.<|im_end|>\n"
        "<|im_start|>assistant\n"
        "def count_words(text):\n    return len(text.split())<|im_end|>\n"
    )


def test_verified_outputs():
    samples = [parse_sample(raw) for raw in GENERATED]
    for name, sample in zip(FUNCTION_NAMES, samples):
        assert all(run_tests(sample["answer"], name, UNIT_TESTS[name]))
    assert run_tests("def count_words(text):\n    return 1 / 0", "count_words", UNIT_TESTS["count_words"]) == [False, False]


def test_rejection_sampling():
    strong = rejection_sampling(CANDIDATES, UNIT_TESTS["count_words"])
    assert strong == {"A": [False, True], "B": [True, False], "C": [True, False], "D": [True, True]}
    assert [label for label, results in strong.items() if all(results)] == ["D"]
    weak = rejection_sampling(CANDIDATES, UNIT_TESTS["count_words"][:1])
    assert [label for label, results in weak.items() if all(results)] == ["B", "C", "D"]
    assert chance_of_one_good(0.25, 1) == 0.25
    assert round(chance_of_one_good(0.25, 4), 3) == 0.684
    assert round(chance_of_one_good(0.25, 10), 3) == 0.944


def test_rule_filters_and_funnel():
    assert not long_enough("ok") and all(long_enough(sample[1]) for sample in POOL[:5])
    assert is_refusal(POOL[4][1]) and not is_refusal(POOL[0][1])
    stages = filter_pool()
    assert [len(pool) for pool in stages.values()] == [6, 5, 4, 3, 2]
    assert [sample[0] for sample in stages["judge"]] == [A, C]
    # The wrong answer of sample 4 passes the rules. A unit test finds it.
    assert POOL[3] in stages["duplicates"]
    assert run_tests(POOL[3][1], "longest_word", UNIT_TESTS["longest_word"]) == [False]


def test_jaccard():
    assert round(jaccard(A, B), 3) == 0.833
    assert round(jaccard(A, C), 3) == 0.667
    assert jaccard(A, D) == 0.125
    assert len(set(A.split())) == 5 and len(set(B.split())) == 6
    sizes = [(len(set(A.split()) & set(other.split())), len(set(A.split()) | set(other.split()))) for other in (B, C, D)]
    assert sizes == [(5, 6), (4, 6), (1, 8)]


def test_minhash_signatures():
    assert len(WORDS) == 10
    ranks = minhash_ranks(4)
    columns = [WORDS.index(word) for word in ("count", "in", "text", "the", "words", "this")]
    assert ranks[:, columns].tolist() == [[4, 2, 5, 9, 1, 0], [2, 3, 4, 8, 1, 7], [5, 9, 2, 1, 3, 6], [4, 7, 3, 1, 8, 9]]
    assert minhash_signature(A, ranks).tolist() == [1, 1, 1, 1]
    assert minhash_signature(B, ranks).tolist() == [0, 1, 1, 1]
    assert minhash_estimate(A, B, 4) == 0.75


def test_minhash_estimates():
    assert round(minhash_estimate(A, B, 128), 3) == 0.844
    assert round(minhash_estimate(A, C, 128), 3) == 0.664
    assert round(minhash_estimate(A, D, 128), 3) == 0.164
    assert round(standard_error(jaccard(A, B), 4), 3) == 0.186
    assert round(standard_error(jaccard(A, B), 128), 3) == 0.033
