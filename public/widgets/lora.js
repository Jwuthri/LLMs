// Parameter count of one d x d weight matrix: full fine-tuning (d * d) against LoRA (2 * d * r).
LLMWidgets.register("lora", (el) => {
  const { h, slider, bars } = LLMWidgets;
  const state = { d: 4096, r: 16 };
  const count = (n) => n.toLocaleString("en-US");
  const megabytes = (n) => ((n * 16) / 8 / 1e6).toFixed(2);

  const chart = h("div");
  const summary = h("p", { class: "llm-widget__note" });
  const memory = h("p", { class: "llm-widget__note" });

  function render() {
    const full = state.d * state.d;
    const lora = 2 * state.d * state.r;
    const percent = Number(((100 * lora) / full).toPrecision(3));
    chart.replaceChildren(
      bars(
        [
          { label: "Full: d × d", value: full, text: count(full) },
          { label: "LoRA: 2 × d × r", value: lora, text: count(lora), off: lora >= full },
        ],
        Math.max(full, lora)
      )
    );
    summary.replaceChildren(
      "LoRA trains ",
      h("strong", { style: "color: var(--llm-ink)" }, `${percent}%`),
      ` of the numbers of full fine-tuning (ratio = 2r / d = ${2 * state.r} / ${state.d}).` +
        (lora >= full ? " The rank is too high for this matrix: with r ≥ d / 2, LoRA saves nothing." : "")
    );
    memory.textContent =
      `At 16 bits for each number: full matrix ${megabytes(full)} MB, adapter ${megabytes(lora)} MB.`;
  }

  function change(key) {
    return (value) => {
      state[key] = value;
      render();
    };
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Trained numbers for one d × d weight matrix"),
    h("div", { class: "llm-widget__controls" }, [
      slider({ label: "Matrix dimension d", min: 64, max: 8192, step: 64, value: state.d, digits: 0, onInput: change("d") }),
      slider({ label: "Rank r", min: 1, max: 256, step: 1, value: state.r, digits: 0, onInput: change("r") }),
    ]),
    chart,
    summary,
    memory
  );
  render();
});
