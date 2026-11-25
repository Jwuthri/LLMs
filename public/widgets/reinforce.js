// REINFORCE on the RL toy (content/rl/01-rl-basics.md): 3 positions, 3 tokens, partial reward.
// Start: the page's group of 4 answers. Update with or without a baseline, or sample new groups from the policy.
LLMWidgets.register("reinforce", (el) => {
  const { h, fmt, slider, bars, softmax, toy } = LLMWidgets;
  const V = toy.vocab;
  const TARGET = [0, 1, 2]; // "the cat sat"
  const PAGE_GROUP = [[0, 1, 2], [2, 1, 2], [0, 2, 2], [2, 2, 2]];

  const state = { logits: [], group: [], steps: 0, stepSize: 1, baseline: "none" };

  function reset() {
    state.logits = [0, 1, 2].map(() => toy.logits.slice());
    state.group = PAGE_GROUP.map((y) => y.slice());
    state.steps = 0;
  }

  const probs = () => state.logits.map((row) => softmax(row));
  const partial = (y) => y.filter((tok, pos) => tok === TARGET[pos]).length / 3;

  // Exact E[partial] = mean over positions of the target-token probability.
  function expectedPartial(p) {
    return TARGET.reduce((sum, tok, pos) => sum + p[pos][tok], 0) / 3;
  }

  function baselineValue(p, rewards) {
    if (state.baseline === "exact") return expectedPartial(p);
    if (state.baseline === "group") return rewards.reduce((a, b) => a + b, 0) / rewards.length;
    return 0;
  }

  function update() {
    const p = probs();
    const rewards = state.group.map(partial);
    const b = baselineValue(p, rewards);
    const n = state.group.length;
    state.logits = state.logits.map((row, pos) =>
      row.map((z, k) => {
        let g = 0;
        state.group.forEach((y, i) => {
          g += (rewards[i] - b) * ((y[pos] === k ? 1 : 0) - p[pos][k]);
        });
        return z + state.stepSize * (g / n);
      })
    );
    state.steps += 1;
  }

  function sampleGroup() {
    const p = probs();
    state.group = [0, 1, 2, 3].map(() =>
      p.map((row) => {
        let r = Math.random();
        let k = 0;
        while (k < row.length - 1 && r >= row[k]) r -= row[k++];
        return k;
      })
    );
  }

  const groupBox = h("div");
  const chart = h("div");
  const summary = h("p", { class: "llm-widget__note" });

  function render() {
    const p = probs();
    const rewards = state.group.map(partial);
    const b = baselineValue(p, rewards);
    const rows = state.group.map((y, i) =>
      h("tr", {}, [
        h("td", {}, y.map((k) => V[k]).join(" ")),
        h("td", {}, fmt(rewards[i], 3)),
        h("td", {}, fmt(rewards[i] - b, 3)),
      ])
    );
    groupBox.replaceChildren(
      h("table", {}, [
        h("tr", {}, [h("th", {}, "Answer"), h("th", {}, "Reward R"), h("th", {}, `R − b (b = ${fmt(b, 3)})`)]),
        ...rows,
      ])
    );
    chart.replaceChildren(
      bars(TARGET.map((tok, pos) => ({
        label: `pos ${pos + 1}: ${V[tok]}`,
        value: p[pos][tok],
        text: fmt(p[pos][tok], 3),
      })))
    );
    const pTarget = TARGET.reduce((prod, tok, pos) => prod * p[pos][tok], 1);
    summary.textContent =
      `After ${state.steps} update(s): P(target) = ${fmt(pTarget, 4)}, ` +
      `E[partial reward] = ${fmt(expectedPartial(p), 3)}.`;
  }

  function act(fn) {
    return () => {
      fn();
      render();
    };
  }

  const select = h("select", { "aria-label": "Baseline" }, [
    h("option", { value: "none" }, "No baseline (b = 0)"),
    h("option", { value: "exact" }, "b = E[R] (exact, toy only)"),
    h("option", { value: "group" }, "b = group mean"),
  ]);
  select.addEventListener("change", act(() => { state.baseline = select.value; }));

  reset();
  el.append(
    h("p", { class: "llm-widget__title" }, "REINFORCE on \"Copy: the cat sat\" with the partial reward"),
    h("div", { class: "llm-widget__controls" }, [
      slider({ label: "Step size α", min: 0.1, max: 3, step: 0.1, value: 1, digits: 1,
        onInput: (v) => { state.stepSize = v; } }),
      h("div", { class: "llm-widget__control" }, [h("label", {}, "Baseline"), select]),
    ]),
    groupBox,
    h("p", { class: "llm-widget__note" }, "Probability of the target token at each position:"),
    chart,
    h("div", { style: "margin-top:0.9em;display:flex;flex-wrap:wrap;gap:0.5em" }, [
      h("button", { onclick: act(update) }, "Update with this group"),
      h("button", { class: "secondary", onclick: act(sampleGroup) }, "Sample a new group"),
      h("button", { class: "secondary", onclick: act(() => { for (let i = 0; i < 10; i++) { sampleGroup(); update(); } }) },
        "Sample + update 10 times"),
      h("button", { class: "secondary", onclick: act(reset) }, "Reset"),
    ]),
    summary
  );
  render();
});
