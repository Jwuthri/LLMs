// Mixed precision: give the bits of a budget to the layers with the highest sensitivity.
LLMWidgets.register("bit-budget", (el) => {
  const { h, fmt, bars } = LLMWidgets;
  // Same numbers and formulas as examples/scientist/quantization_beyond_rounding.py.
  const sensitivity = [0.064, 0.016, 0.004]; // assumed error of each layer at 4 bits
  const minBits = 2;
  const maxBits = 8;
  const state = { total: 12 }; // total bits for the 3 layers: a mean of 4

  const layerError = (s, bits) => s * 4 ** (4 - bits);
  const totalError = (bits) => bits.reduce((sum, b, i) => sum + layerError(sensitivity[i], b), 0);

  // Start each layer at minBits. Give one bit at a time to the layer with the largest error.
  function allocate(total) {
    const bits = sensitivity.map(() => minBits);
    for (let n = total - minBits * bits.length; n > 0; n--) {
      let worst = -1;
      bits.forEach((b, i) => {
        if (b < maxBits && (worst < 0 || layerError(sensitivity[i], b) > layerError(sensitivity[worst], bits[worst]))) worst = i;
      });
      bits[worst] += 1;
    }
    return bits;
  }

  const out = h("output", {}, fmt(state.total / 3, 2));
  const input = h("input", { type: "range", min: 6, max: 18, step: 1, value: state.total, "aria-label": "Mean bits for each weight" });
  input.addEventListener("input", () => {
    state.total = Number(input.value);
    out.textContent = fmt(state.total / 3, 2);
    render();
  });
  const table = h("table");
  const chart = h("div");
  const summary = h("p", { class: "llm-widget__note" });

  function render() {
    const mean = state.total / 3;
    const mixed = allocate(state.total);
    const equal = sensitivity.map(() => mean);
    const mixedError = totalError(mixed);
    const equalError = totalError(equal);
    table.replaceChildren(
      h("tr", {}, ["Layer", "Sensitivity", "Equal bits", "Error", "Bits by sensitivity", "Error"].map((t) => h("th", {}, t))),
      ...sensitivity.map((s, i) =>
        h("tr", {}, [
          h("td", {}, `Layer ${i + 1}`),
          h("td", {}, fmt(s, 3)),
          h("td", {}, fmt(mean, 2)),
          h("td", {}, fmt(layerError(s, mean), 4)),
          h("td", { style: "font-weight:600" }, String(mixed[i])),
          h("td", {}, fmt(layerError(s, mixed[i]), 4)),
        ])
      )
    );
    chart.replaceChildren(
      bars(
        [
          { label: "Equal bits", value: equalError, text: fmt(equalError, 4), off: true },
          { label: "By sensitivity", value: mixedError, text: fmt(mixedError, 4) },
        ],
        equalError
      )
    );
    summary.replaceChildren(
      `Budget: ${state.total} bits for 3 layers. `,
      h("strong", {}, `Total error ${fmt(mixedError, 3)} against ${fmt(equalError, 3)}`),
      ` (${fmt((mixedError / equalError) * 100, 0)}% of the error of equal bits).`
    );
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Spend a bit budget on 3 layers"),
    h("div", { class: "llm-widget__controls" }, [
      h("div", { class: "llm-widget__control" }, [h("label", {}, ["Mean bits for each weight", out]), input]),
    ]),
    h("div", { style: "overflow-x:auto" }, [table]),
    chart,
    summary
  );
  render();
});
