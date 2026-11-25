// GRPO, DAPO, GSPO and CISPO on one token: move pi_new("the") and see the ratio, the objective and the gradient weight.
LLMWidgets.register("policy-objectives", (el) => {
  const { h, fmt, slider, bars, softmax, toy } = LLMWidgets;
  // Same numbers as examples/rl/grpo_family.py: group rewards [1, 0, 0, 0], one GRPO step of size 1.8.
  const advPos = Math.sqrt(3); // +1.732, answer "the cat sat"
  const advNeg = -1 / Math.sqrt(3); // -0.577, answer "the sat sat"
  const delta = (1.8 * (advPos + advNeg)) / 12; // logit change of the step at positions 1 and 2
  const pOld = softmax(toy.logits);
  const pos1 = softmax([toy.logits[0] + delta, toy.logits[1], toy.logits[2] - delta]);
  const pos2 = softmax([toy.logits[0], toy.logits[1] + delta, toy.logits[2] - delta]);
  // Ratios of the other tokens stay at the page values: "cat" 1.264 and "sat" 0.894 at position 2, "sat" 1 at position 3.
  const otherPos = (pos2[1] / pOld[1]) * 1;
  const otherNeg = (pos2[2] / pOld[2]) * 1;
  const state = { pNew: pos1[0] };

  // min(r A, clip(r, 1 - lo, 1 + hi) A) and its gradient weight (r A, or 0 when the clipped side wins).
  function clipped(r, a, lo, hi) {
    const unclipped = r * a;
    const c = Math.min(Math.max(r, 1 - lo), 1 + hi) * a;
    return unclipped <= c ? { obj: unclipped, grad: unclipped } : { obj: c, grad: 0 };
  }

  function methods(r, a, other) {
    const s = Math.cbrt(r * other); // GSPO: geometric mean of the 3 token ratios
    return [
      { name: "GRPO", ratio: r, ...clipped(r, a, 0.2, 0.2) },
      { name: "DAPO", ratio: r, ...clipped(r, a, 0.2, 0.28) },
      { name: "GSPO", ratio: s, ...clipped(s, a, 3e-4, 4e-4) },
      { name: "CISPO", ratio: r, obj: null, grad: Math.min(r, 1.28) * a },
    ];
  }

  const info = h("p", { class: "llm-widget__note" });
  const table = h("table");
  const chart = h("div");
  const note = h("p", { class: "llm-widget__note" });

  function cell(x) {
    return h("td", {}, x === null ? "n/a" : fmt(x, 3));
  }

  function gradCell(x) {
    return h("td", { style: x === 0 ? "color:var(--llm-muted)" : "font-weight:600" }, x === 0 ? "0 (clipped)" : fmt(x, 3));
  }

  function render() {
    const r = state.pNew / pOld[0];
    const pos = methods(r, advPos, otherPos);
    const neg = methods(r, advNeg, otherNeg);
    info.replaceChildren(
      `p_old("the") = ${fmt(pOld[0], 3)}, p_new = ${fmt(state.pNew, 3)}, `,
      h("strong", {}, `ratio r = ${fmt(r, 3)}`),
      `. GSPO uses s = (r × other token ratios)^(1/3): ${fmt(pos[2].ratio, 3)} and ${fmt(neg[2].ratio, 3)}.`
    );
    table.replaceChildren(
      h("tr", {}, ["Method", "Ratio used", "Objective, A = +1.732", "Gradient weight", "Ratio used", "Objective, A = −0.577", "Gradient weight"].map((t) => h("th", {}, t))),
      ...pos.map((m, i) =>
        h("tr", {}, [
          h("td", {}, m.name),
          cell(m.ratio),
          cell(m.obj),
          gradCell(m.grad),
          cell(neg[i].ratio),
          cell(neg[i].obj),
          gradCell(neg[i].grad),
        ])
      )
    );
    chart.replaceChildren(
      bars(
        pos.map((m) => ({ label: m.name, value: Math.abs(m.grad), text: m.grad === 0 ? "0" : fmt(m.grad, 3), off: m.grad === 0 })),
        1.6 * advPos
      )
    );
    note.textContent =
      "Bars: gradient weight on the token of the correct answer (A = +1.732). " +
      "CISPO has no clipped objective: it uses sg(clipped r) × A × log π, so only its gradient weight is shown. " +
      "GSPO clips at [0.9997, 1.0004]: almost any visible move clips the full answer.";
  }

  el.append(
    h("p", { class: "llm-widget__title" }, 'Token "the" at position 1, in "the cat sat" (A = +1.732) and "the sat sat" (A = −0.577)'),
    h("div", { class: "llm-widget__controls" }, [
      slider({
        label: 'New probability of "the"',
        min: 0.03,
        max: 0.25,
        step: "any",
        value: state.pNew,
        digits: 3,
        onInput: (value) => {
          state.pNew = value;
          render();
        },
      }),
    ]),
    info,
    h("div", { style: "overflow-x:auto" }, [table]),
    chart,
    note
  );
  render();
});
