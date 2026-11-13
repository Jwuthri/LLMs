// Fuzzy deduplication of the 8 toy samples of the LLM Datasets track: move the similarity threshold
// and see which samples go out. Starts with the numbers of the lesson "From raw text to a dataset: the tools".
// Keep the samples in sync with examples/datasets/toy_dataset.py and the rule with examples/datasets/tools.py.
LLMWidgets.register("dedup-threshold", (el) => {
  const { h, fmt, slider } = LLMWidgets;
  const samples = [
    { id: 1, topic: "math", text: "Count the words: the cat sat", v: [1.0, 0.1] },
    { id: 2, topic: "math", text: "Count the words: the cat", v: [0.9, 0.2] },
    { id: 3, topic: "math", text: "Count the words: the cat sat.", v: [0.98, 0.12] },
    { id: 4, topic: "math", text: "Count the letters: cat", v: [0.8, 0.4] },
    { id: 5, topic: "code", text: "Write Python that counts the words of a text", v: [0.1, 1.0] },
    { id: 6, topic: "code", text: "This function counts words wrong. Fix it, then add a test", v: [0.2, 0.9] },
    { id: 7, topic: "chat", text: "Explain why the cat sat, then make it shorter, then translate it", v: [0.6, 0.7] },
    { id: 8, topic: "chat", text: "Tell me about the cat", v: [0.5, 0.5] },
  ];
  const state = { threshold: 0.999 };

  const cosine = (a, b) => (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(a[0], a[1]) * Math.hypot(b[0], b[1]));

  // Read the samples in order. Keep a sample if no kept sample is at or above the threshold.
  function deduplicate(threshold) {
    const kept = [];
    return samples.map((sample) => {
      const match = kept.find((other) => cosine(sample.v, other.v) >= threshold);
      if (!match) kept.push(sample);
      return { sample, match, similarity: match ? cosine(sample.v, match.v) : null };
    });
  }

  const body = h("tbody");
  const table = h("table", {}, [
    h("thead", {}, [h("tr", {}, ["Sample", "Topic", "Instruction", "Result"].map((name) => h("th", {}, name)))]),
    body,
  ]);
  const note = h("p", { class: "llm-widget__note" });

  function render() {
    const rows = deduplicate(state.threshold);
    const kept = rows.filter((row) => !row.match);
    body.replaceChildren(
      ...rows.map(({ sample, match, similarity }) =>
        h("tr", match ? { style: "color:var(--llm-muted)" } : {}, [
          h("td", {}, sample.id),
          h("td", {}, sample.topic),
          h("td", {}, sample.text),
          h("td", {}, match ? `removed: ${fmt(similarity, 4)} with sample ${match.id}` : "kept"),
        ])
      )
    );
    const topics = ["math", "code", "chat"]
      .map((topic) => `${topic} ${kept.filter((row) => row.sample.topic === topic).length}`)
      .join(", ");
    note.textContent =
      `${kept.length} of ${samples.length} samples stay (${topics}). ` +
      (kept.length < 7
        ? "The threshold now removes samples that are only on the same topic."
        : kept.length === 7
          ? "Only the near duplicate (sample 3) goes out."
          : "No pair is at or above the threshold: the near duplicate stays.");
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Fuzzy deduplication: the similarity threshold decides what goes out"),
    h("div", { class: "llm-widget__controls" }, [
      slider({
        label: "Similarity threshold", min: 0.9, max: 1, step: 0.001, value: state.threshold, digits: 3,
        onInput: (value) => { state.threshold = value; render(); },
      }),
    ]),
    h("div", { style: "overflow-x:auto" }, [table]),
    note
  );
  render();
});
