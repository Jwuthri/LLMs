// Sparse, partial and process rewards on the RL toy group, and the chance of a learning signal vs group size.
LLMWidgets.register("reward-types", (el) => {
  const { h, fmt, slider, softmax, toy } = LLMWidgets;
  const target = ["the", "cat", "sat"];
  const group = [
    ["the", "cat", "sat"],
    ["sat", "cat", "sat"],
    ["the", "sat", "sat"],
    ["sat", "sat", "sat"],
  ];
  const probs = softmax(toy.logits);
  const pToken = target.map((t) => probs[toy.vocab.indexOf(t)]); // [0.099, 0.242, 0.659]
  const pSuccess = pToken.reduce((a, b) => a * b, 1); // 0.0157

  // P(k of 3 tokens correct), k = 0..3.
  const partialDist = [0, 0, 0, 0];
  for (let mask = 0; mask < 8; mask++) {
    let prob = 1;
    let k = 0;
    for (let i = 0; i < 3; i++) {
      const hit = (mask >> i) & 1;
      prob *= hit ? pToken[i] : 1 - pToken[i];
      k += hit;
    }
    partialDist[k] += prob;
  }

  const state = { kind: "sparse", g: 4 };
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const correct = (a) => a.map((tok, i) => (tok === target[i] ? 1 : 0));

  function grpo(rewards) {
    const m = mean(rewards);
    const sd = Math.sqrt(mean(rewards.map((r) => (r - m) ** 2)));
    return rewards.map((r) => (sd === 0 ? 0 : (r - m) / sd));
  }

  function rows() {
    if (state.kind !== "process") {
      const rewards = group.map((a) =>
        state.kind === "sparse" ? Math.min(...correct(a)) : mean(correct(a))
      );
      const adv = grpo(rewards);
      return group.map((a, i) => [a.join(" "), fmt(rewards[i]), fmt(adv[i])]);
    }
    // Process: reward 1/3 per correct token; signal = reward-to-go minus the group mean at that position.
    const steps = group.map((a) => correct(a).map((c) => c / 3));
    const rtg = steps.map((r) => r.map((_, t) => r.slice(t).reduce((x, y) => x + y, 0)));
    const posMean = [0, 1, 2].map((t) => mean(rtg.map((row) => row[t])));
    return group.map((a, i) => [
      a.join(" "),
      steps[i].map((r) => fmt(r)).join(" / "),
      rtg[i].map((g, t) => fmt(g - posMean[t])).join(" / "),
    ]);
  }

  const table = h("div", { style: "overflow-x:auto" });
  const buttons = h("div", { style: "display:flex;gap:0.5em;flex-wrap:wrap;margin-bottom:0.8em" });
  const kinds = [
    ["sparse", "Sparse (all correct)"],
    ["partial", "Partial credit"],
    ["process", "Process (per token)"],
  ];
  const signal = h("p", { class: "llm-widget__note" });

  function renderTable() {
    buttons.replaceChildren(
      ...kinds.map(([key, label]) =>
        h("button", {
          class: key === state.kind ? "" : "secondary",
          "aria-pressed": key === state.kind,
          onclick: () => {
            state.kind = key;
            renderTable();
          },
        }, label)
      )
    );
    const process = state.kind === "process";
    table.replaceChildren(
      h("table", {}, [
        h("tr", {}, [
          h("th", {}, "Answer"),
          h("th", {}, process ? "Reward per token" : "Reward"),
          h("th", {}, process ? "Token signal (reward-to-go − position mean)" : "GRPO advantage"),
        ]),
        ...rows().map((cells) => h("tr", {}, cells.map((c) => h("td", {}, c)))),
      ])
    );
  }

  function renderSignal() {
    const g = state.g;
    const sparse = 1 - (1 - pSuccess) ** g;
    const partial = 1 - partialDist.reduce((s, p) => s + p ** g, 0);
    signal.textContent =
      `Group of ${g}: P(at least one sparse reward of 1) = ${fmt(sparse)}. ` +
      `With partial credit, P(the rewards are not all equal) = ${fmt(partial)}. ` +
      `A group with all rewards equal gives a zero advantage to every answer.`;
  }

  el.append(
    h("p", { class: "llm-widget__title" }, 'Prompt "Copy: the cat sat", P(target) = ' + fmt(pSuccess, 4)),
    buttons,
    table,
    h("div", { class: "llm-widget__controls", style: "margin-top:0.9em" }, [
      slider({ label: "Group size G", min: 1, max: 64, step: 1, value: 4, digits: 0, onInput: (v) => { state.g = v; renderSignal(); } }),
    ]),
    signal
  );
  renderTable();
  renderSignal();
});
