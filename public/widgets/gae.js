// GAE on the answer "sat cat sat" of the RL toy task (content/rl/03-ppo.md).
// Reward 0.667 at the last token. Critic: exact values under the start policy, or an assumed
// critic that sees only the position. Sliders change gamma and lambda.
LLMWidgets.register("gae", (el) => {
  const { h, fmt, slider, softmax, toy } = LLMWidgets;
  const answer = ["sat", "cat", "sat"];
  const target = ["the", "cat", "sat"];
  const T = answer.length;
  const reward = answer.filter((tok, t) => tok === target[t]).length / T; // 0.667
  const probs = softmax(toy.logits);
  const pTarget = target.map((tok) => probs[toy.vocab.indexOf(tok)]); // [0.099, 0.242, 0.659]
  const guess = [0.4, 0.45, 0.5];
  const state = { gamma: 1, lambda: 0.95, critic: "true" };

  function trueValues(gamma) {
    return answer.map((_, t) => {
      let expected = 0;
      for (let i = 0; i < t; i++) expected += answer[i] === target[i] ? 1 : 0;
      for (let i = t; i < T; i++) expected += pTarget[i];
      return Math.pow(gamma, T - 1 - t) * (expected / T);
    });
  }

  function compute() {
    const values = state.critic === "true" ? trueValues(state.gamma) : guess;
    const rewards = answer.map((_, t) => (t === T - 1 ? reward : 0));
    const deltas = values.map((v, t) => rewards[t] + state.gamma * (t + 1 < T ? values[t + 1] : 0) - v);
    const adv = new Array(T).fill(0);
    let running = 0;
    for (let t = T - 1; t >= 0; t--) {
      running = deltas[t] + state.gamma * state.lambda * running;
      adv[t] = running;
    }
    return { values, rewards, deltas, adv };
  }

  const table = h("div");
  const chart = h("div");
  const note = h("p", { class: "llm-widget__note" });

  function divergingBars(rows, max = 0.5) {
    const root = h("div", { class: "llm-bars" });
    for (const row of rows) {
      const share = Math.min(Math.abs(row.value) / max, 1) * 50;
      const left = row.value >= 0 ? 50 : 50 - share;
      const colour = row.value >= 0 ? "var(--llm-s1)" : "var(--llm-s2)";
      root.append(
        h("div", { class: "llm-bars__label" }, row.label),
        h("div", { class: "llm-bars__track", style: "position:relative" }, [
          h("div", { style: "position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--llm-axis)" }),
          h("div", { class: "llm-bars__fill", style: `margin-left:${left}%;width:${share}%;background:${colour}` }),
        ]),
        h("div", { class: "llm-bars__value" }, fmt(row.value, 3))
      );
    }
    return root;
  }

  function render() {
    const { values, rewards, deltas, adv } = compute();
    const head = h("tr", {}, ["t", "token", "V(s_t)", "r_t", "δ_t", "A_t"].map((x) => h("th", {}, x)));
    const body = answer.map((tok, t) =>
      h("tr", {}, [t, tok, fmt(values[t], 3), fmt(rewards[t], 3), fmt(deltas[t], 3), fmt(adv[t], 3)].map((x) => h("td", {}, String(x))))
    );
    table.replaceChildren(h("table", {}, [h("thead", {}, head), h("tbody", {}, body)]));
    chart.replaceChildren(divergingBars(answer.map((tok, t) => ({
      label: `A_${t} ${tok}${tok === target[t] ? "" : " (wrong)"}`,
      value: adv[t],
    }))));
    note.textContent = state.lambda === 0
      ? "λ = 0: each token gets only its own TD error (one-step TD). Low variance; correct only if the critic is correct."
      : state.lambda === 1
        ? "λ = 1: each token gets the return minus V(s_t) (Monte Carlo). No critic bias in the gradient, but the most noise."
        : "0 < λ < 1: a weighted mix of the TD errors that follow. λ trades critic bias against noise.";
  }

  const criticButtons = [
    ["true", "Exact values"],
    ["guess", "Position-only critic"],
  ].map(([key, text]) => {
    const button = h("button", {
      class: key === state.critic ? "" : "secondary",
      onclick: () => {
        state.critic = key;
        criticButtons.forEach((b) => b.classList.toggle("secondary", b !== button));
        render();
      },
    }, text);
    return button;
  });

  el.append(
    h("p", { class: "llm-widget__title" }, 'Per-token advantages of "sat cat sat" (reward 0.667 at the end)'),
    h("div", { class: "llm-widget__controls" }, [
      slider({ label: "Discount γ", min: 0.5, max: 1, step: 0.01, value: 1, onInput: (v) => { state.gamma = v; render(); } }),
      slider({ label: "GAE λ", min: 0, max: 1, step: 0.05, value: 0.95, onInput: (v) => { state.lambda = v; render(); } }),
    ]),
    h("div", { style: "display:flex;gap:0.5em;flex-wrap:wrap;margin-bottom:0.9em" }, criticButtons),
    table,
    h("div", { style: "margin-top:0.9em" }, [chart]),
    note
  );
  render();
});
