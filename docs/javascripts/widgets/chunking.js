// Fixed-size chunks with overlap on the 15-word text of docs/engineer/03-rag.md.
// Same rule as split_fixed() in examples/engineer/vector_storage.py.
LLMWidgets.register("chunking", (el) => {
  const { h, slider } = LLMWidgets;
  const words = "The cat sat on the mat. The cat sleeps all day. The cat eats fish.".split(" ");
  const state = { size: 6, overlap: 2 };

  // Each chunk starts `size - overlap` words after the one before.
  function split() {
    const chunks = [];
    for (let start = 0; start < words.length; start += state.size - state.overlap) {
      chunks.push({ start, end: Math.min(start + state.size, words.length) });
      if (start + state.size >= words.length) break;
    }
    return chunks;
  }

  const pill =
    "display:inline-block;margin:0 0.3em 0.3em 0;padding:0.1em 0.5em;border-radius:4px;" +
    'font-family:"JetBrains Mono",ui-monospace,monospace;color:var(--llm-ink);';
  const plain = pill + "background:var(--llm-s1-soft);border:1px solid var(--llm-s1)";
  const shared = pill + "background:color-mix(in srgb, var(--llm-s2) 28%, transparent);border:1px solid var(--llm-s2)";

  const rows = h("div");
  const summary = h("p", { class: "llm-widget__note" });

  const sizeControl = slider({
    label: "Chunk size (words)", min: 1, max: words.length, step: 1, value: state.size, digits: 0,
    onInput: (size) => {
      state.size = size;
      // The overlap must be smaller than the chunk size. If not, the chunks do not move forward.
      const input = overlapControl.querySelector("input");
      input.max = Math.max(0, size - 1);
      state.overlap = Math.min(state.overlap, size - 1);
      input.value = state.overlap;
      overlapControl.querySelector("output").textContent = String(state.overlap);
      render();
    },
  });
  const overlapControl = slider({
    label: "Overlap (words)", min: 0, max: state.size - 1, step: 1, value: state.overlap, digits: 0,
    onInput: (overlap) => {
      state.overlap = overlap;
      render();
    },
  });

  function render() {
    const chunks = split();
    rows.replaceChildren(
      ...chunks.map((chunk, i) => {
        const previousEnd = i === 0 ? 0 : chunks[i - 1].end;
        return h("div", { style: "display:flex;gap:0.8em;align-items:baseline" }, [
          h("span", { style: "flex:0 0 5.2em;color:var(--llm-ink-2)" }, `Chunk ${i + 1}`),
          h("span", {}, words.slice(chunk.start, chunk.end).map((word, j) =>
            h("span", { style: chunk.start + j < previousEnd ? shared : plain }, word)
          )),
        ]);
      })
    );
    const stored = chunks.reduce((sum, chunk) => sum + chunk.end - chunk.start, 0);
    const step = state.size - state.overlap;
    summary.replaceChildren(
      h("strong", {}, `${chunks.length} ${chunks.length === 1 ? "chunk" : "chunks"}. `),
      `Each chunk starts ${step} ${step === 1 ? "word" : "words"} after the one before ` +
        `(${state.size} − ${state.overlap}). The store holds ${stored} words for a text of ${words.length} words. `,
      h("span", { style: shared }, "orange"),
      " = a word that is also in the chunk before (the overlap)."
    );
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Fixed-size chunks of a 15-word text"),
    h("div", { class: "llm-widget__controls" }, [sizeControl, overlapControl]),
    rows,
    summary
  );
  render();
});
