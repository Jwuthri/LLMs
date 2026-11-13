// Attention weights for "the cat sat" with the toy embeddings (W_Q = W_K = W_V = identity).
LLMWidgets.register("attention", (el) => {
  const { h, fmt, softmax, toy } = LLMWidgets;
  const tokens = toy.vocab;
  const x = toy.embeddings;
  const dk = x[0].length;
  const state = { causal: true, row: 2 };

  function compute() {
    return x.map((q, i) => {
      const scores = x.map((k) => (q[0] * k[0] + q[1] * k[1]) / Math.sqrt(dk));
      const visible = scores.map((s, j) => (state.causal && j > i ? -Infinity : s));
      const weights = softmax(visible);
      const output = [0, 1].map((d) => weights.reduce((sum, w, j) => sum + w * x[j][d], 0));
      return { scores, weights, output };
    });
  }

  const grid = h("table");
  const detail = h("p", { class: "llm-widget__note" });
  const mask = h("input", { type: "checkbox", id: "attention-mask", checked: "" });
  mask.addEventListener("change", () => {
    state.causal = mask.checked;
    render();
  });

  function render() {
    const rows = compute();
    grid.replaceChildren(
      h("tr", {}, [h("th", {}, "query \\ key"), ...tokens.map((t) => h("th", {}, t))]),
      ...rows.map((row, i) =>
        h("tr", { style: "cursor:pointer", onclick: () => { state.row = i; render(); } }, [
          h("td", { style: i === state.row ? "font-weight:700" : "" }, tokens[i]),
          ...row.weights.map((w) =>
            h("td", {
              style: `background: color-mix(in srgb, var(--llm-s1) ${Math.round(w * 85)}%, transparent);` +
                `color: ${w > 0.55 ? "#fff" : "var(--llm-ink)"}`,
            }, fmt(w, 3))
          ),
        ])
      )
    );
    const row = rows[state.row];
    const scores = row.scores.map((s, j) => (state.causal && j > state.row ? "masked" : fmt(s, 3)));
    detail.textContent =
      `Query "${tokens[state.row]}": scores [${scores.join(", ")}] → weights [` +
      `${row.weights.map((w) => fmt(w, 3)).join(", ")}] → output [` +
      `${row.output.map((v) => fmt(v, 3)).join(", ")}]. Click a row to change the query.`;
  }

  el.append(
    h("p", { class: "llm-widget__title" }, 'Attention weights for "the cat sat"'),
    h("div", { class: "llm-widget__controls" }, [
      h("label", { for: "attention-mask" }, [mask, " Causal mask (a token cannot see later tokens)"]),
    ]),
    grid,
    detail
  );
  render();
});
