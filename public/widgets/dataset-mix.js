// A mix of three sources: the cap on the samples of each source, and the mean tokens of a sample.
// Shows the share of each family, the token total and the training time.
// Starts with the numbers of the lesson "Instruction datasets". Keep in sync with
// examples/datasets/instruction_datasets.py (SOURCES, SOURCE_TOKENS, CAP, TOKENS_PER_SECOND).
LLMWidgets.register("dataset-mix", (el) => {
  const { h, fmt, slider, bars } = LLMWidgets;
  const sources = [
    { name: "general", size: 4000000, cap: 500000, tokens: 600 },
    { name: "code", size: 1000000, cap: 500000, tokens: 800 },
    { name: "function calling", size: 100000, cap: 500000, tokens: 400 },
  ];
  const TOKENS_PER_SECOND = 2500;
  const BATCH_SIZE = 16;
  const count = (n) => Math.round(n).toLocaleString("en-US");

  const chart = h("div");
  const note = h("p", { class: "llm-widget__note" });

  function render() {
    const kept = sources.map((s) => Math.min(s.size, s.cap));
    const total = kept.reduce((a, b) => a + b, 0);
    const tokens = sources.reduce((sum, s, i) => sum + kept[i] * s.tokens, 0);
    const hours = tokens / TOKENS_PER_SECOND / 3600;
    const smallest = Math.min(...kept) / total;
    chart.replaceChildren(
      bars(
        sources.map((s, i) => ({
          label: s.name,
          value: kept[i] / total,
          text: `${fmt(kept[i] / total, 3)} (${count(kept[i])})`,
        }))
      )
    );
    note.textContent =
      `Mix: ${count(total)} samples, ${fmt(tokens / 1e6, 0)} million tokens for one epoch. ` +
      `At ${count(TOKENS_PER_SECOND)} tokens/s: ${fmt(hours, 1)} hours (${fmt(hours / 24, 2)} days). ` +
      `With a batch of ${BATCH_SIZE}, the model sees 100 samples of the smallest family after ` +
      `${Math.ceil(100 / (smallest * BATCH_SIZE))} steps.`;
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Cap each source and count the tokens of the mix"),
    h("div", { class: "llm-widget__controls" },
      sources.map((s) =>
        slider({
          label: `${s.name} (${count(s.size)}): cap`, min: 50000, max: 4000000, step: 50000, value: s.cap, digits: 0,
          onInput: (value) => { s.cap = value; render(); },
        })
      )
    ),
    h("div", { class: "llm-widget__controls" },
      sources.map((s) =>
        slider({
          label: `${s.name}: mean tokens`, min: 100, max: 2000, step: 50, value: s.tokens, digits: 0,
          onInput: (value) => { s.tokens = value; render(); },
        })
      )
    ),
    chart,
    note
  );
  render();
});
