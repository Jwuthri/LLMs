// Absmax (symmetric) and zero-point (asymmetric) quantization of the toy matrix W.
LLMWidgets.register("quantization", (el) => {
  const { h, fmt, slider, toy } = LLMWidgets;
  const weights = toy.W.flat();
  const names = ["W[1,1]", "W[1,2]", "W[2,1]", "W[2,2]"];
  const state = { bits: 8, method: "absmax" };

  // Same formulas as examples/scientist/quantization.py.
  function quantize() {
    const qmin = -(2 ** (state.bits - 1));
    const qmax = 2 ** (state.bits - 1) - 1;
    if (state.method === "absmax") {
      const scale = qmax / Math.max(...weights.map(Math.abs));
      const q = weights.map((w) => Math.round(scale * w));
      return { scale, zero: 0, lo: -qmax, hi: qmax, q, dequant: q.map((v) => v / scale) };
    }
    const lo = Math.min(...weights);
    const scale = (qmax - qmin) / (Math.max(...weights) - lo);
    const zero = Math.round(-scale * lo) + qmin;
    const q = weights.map((w) => Math.min(qmax, Math.max(qmin, Math.round(scale * w + zero))));
    return { scale, zero, lo: qmin, hi: qmax, q, dequant: q.map((v) => (v - zero) / scale) };
  }

  const table = h("table");
  const summary = h("p", { class: "llm-widget__note" });
  const select = h("select", { id: "quantization-method", "aria-label": "Quantization method" }, [
    h("option", { value: "absmax" }, "Absmax (symmetric)"),
    h("option", { value: "zeropoint" }, "Zero-point (asymmetric)"),
  ]);
  select.addEventListener("change", () => {
    state.method = select.value;
    render();
  });

  function render() {
    const r = quantize();
    const errors = weights.map((w, i) => Math.abs(w - r.dequant[i]));
    const worst = Math.max(...errors);
    const mae = errors.reduce((a, b) => a + b, 0) / errors.length;
    table.replaceChildren(
      h("tr", {}, ["Weight", "Value", "Integer", "Dequantized", "Error"].map((t) => h("th", {}, t))),
      ...weights.map((w, i) =>
        h("tr", {}, [
          h("td", {}, names[i]),
          h("td", {}, fmt(w, 1)),
          h("td", { style: "font-weight:600" }, String(r.q[i])),
          h("td", {}, fmt(r.dequant[i], 4)),
          h("td", {
            style: `background: color-mix(in srgb, var(--llm-s2) ${worst > 0 ? Math.round((errors[i] / worst) * 45) : 0}%, transparent)`,
          }, fmt(errors[i], 4)),
        ])
      )
    );
    summary.replaceChildren(
      `Integers from ${r.lo} to ${r.hi} (${r.hi - r.lo + 1} values). Scale = ${fmt(r.scale, 3)}` +
        (state.method === "zeropoint" ? `, zero point = ${r.zero}. ` : ". "),
      h("strong", {}, `Mean absolute error = ${fmt(mae, 4)}`),
      "."
    );
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Quantize W = [[0.5, -1.2], [0.8, 0.3]]"),
    h("div", { class: "llm-widget__controls" }, [
      slider({
        label: "Bits for each weight", min: 2, max: 8, step: 1, value: 8, digits: 0,
        onInput: (value) => { state.bits = value; render(); },
      }),
      h("div", { class: "llm-widget__control" }, [h("label", { for: "quantization-method" }, "Method"), select]),
    ]),
    h("div", { style: "overflow-x:auto" }, [table]),
    summary
  );
  render();
});
