// KL estimators k1, k2, k3 on one position of the RL toy (content/rl/05-stable-training.md).
// The reference is the toy start policy (logits [0.1, 1.0, 2.0]). The slider moves the policy logit of "the";
// the logits of "cat" and "sat" stay at 1.0 and 2.0. The page starts at the logit 2.5.
LLMWidgets.register("kl-estimators", (el) => {
  const { h, fmt, slider, softmax, toy } = LLMWidgets;
  const ref = softmax(toy.logits);
  const table = h("div");
  const summary = h("p", { class: "llm-widget__note" });

  function cell(text, strong) {
    return h("td", { style: "padding:0.2em 0.6em;text-align:right" + (strong ? ";font-weight:600" : "") }, text);
  }

  function render(logitThe) {
    const pi = softmax([logitThe, toy.logits[1], toy.logits[2]]);
    const rows = toy.vocab.map((token, i) => {
      const r = ref[i] / pi[i];
      return { token, pi: pi[i], ref: ref[i], r, k1: -Math.log(r), k2: Math.log(r) ** 2 / 2, k3: r - 1 - Math.log(r) };
    });
    const mean = (key) => rows.reduce((sum, row) => sum + row.pi * row[key], 0);
    const exact = rows.reduce((sum, row) => sum + row.pi * Math.log(row.pi / row.ref), 0);

    const head = ["sampled token", "π", "π_ref", "r", "k1", "k2", "k3"].map((t) =>
      h("th", { style: "padding:0.2em 0.6em;text-align:right" }, t)
    );
    const body = rows.map((row) =>
      h("tr", {}, [
        cell(row.token),
        cell(fmt(row.pi)),
        cell(fmt(row.ref)),
        cell(fmt(row.r)),
        cell(fmt(row.k1), row.k1 < 0),
        cell(fmt(row.k2)),
        cell(fmt(row.k3)),
      ])
    );
    const footer = h("tr", {}, [
      cell("mean under π", true), cell(""), cell(""), cell(""),
      cell(fmt(mean("k1")), true), cell(fmt(mean("k2")), true), cell(fmt(mean("k3")), true),
    ]);
    table.replaceChildren(
      h("div", { style: "overflow-x:auto" }, [
        h("table", { style: "border-collapse:collapse;font-variant-numeric:tabular-nums" }, [
          h("thead", {}, [h("tr", {}, head)]),
          h("tbody", {}, [...body, footer]),
        ]),
      ])
    );
    summary.textContent =
      `Exact KL(π || π_ref) = ${fmt(exact)}. The means of k1 and k3 equal it; k2 gives ${fmt(mean("k2"))}. ` +
      "A bold k1 is negative for that sample, but k3 is never negative.";
  }

  el.append(
    h("p", { class: "llm-widget__title" }, 'Position 1: policy vs reference (logits of "cat" and "sat" stay 1.0 and 2.0)'),
    h("div", { class: "llm-widget__controls" }, [
      slider({ label: 'Policy logit of "the"', min: -2, max: 5, step: 0.1, value: 2.5, digits: 1, onInput: render }),
    ]),
    table,
    summary
  );
  render(2.5);
});
