// Over-optimization of a buggy checker. The checker counts the tokens equal to "the", so it pays
// "the the the" in full. The best policy for (proxy - beta * KL to the start policy) has, at each of
// the 3 positions, pi(a) proportional to pi_start(a) * exp(r(a) / beta), with r = [1/3, 0, 0].
// Starts at beta = 0.2, the KL run of the Reward hacking lesson.
LLMWidgets.register("overoptimization", (el) => {
  const { h, fmt, slider, bars, softmax, toy } = LLMWidgets;
  const start = softmax(toy.logits); // [0.099, 0.242, 0.659] at each position
  const startGold = start[0] * start[1] * start[2];
  const state = { beta: 0.2 };

  const chart = h("div");
  const goldChart = h("div");
  const note = h("p", { class: "llm-widget__note" });

  function policy(beta) {
    const w = start.map((p, i) => p * Math.exp((i === 0 ? 1 / 3 : 0) / beta));
    const total = w.reduce((a, b) => a + b, 0);
    return w.map((x) => x / total);
  }

  function render() {
    const q = policy(state.beta);
    const proxy = q[0];                 // expected share of "the" tokens
    const gold = q[0] * q[1] * q[2];    // P("the cat sat")
    const hack = q[0] ** 3;             // P("the the the")
    const kl = 3 * q.reduce((s, p, i) => s + p * Math.log(p / start[i]), 0);
    chart.replaceChildren(
      bars([
        { label: "proxy (buggy checker)", value: proxy, text: fmt(proxy) },
        { label: 'P("the the the")', value: hack, text: fmt(hack) },
      ])
    );
    goldChart.replaceChildren(
      bars([{ label: 'gold: P("the cat sat")', value: gold, text: fmt(gold) }], 0.035)
    );
    note.textContent =
      `KL to the start policy = ${fmt(kl)}, sqrt(KL) = ${fmt(Math.sqrt(kl))}. ` +
      `The gold starts at ${fmt(startGold)} and is highest (0.029) at beta = 0.219. ` +
      "A smaller beta lets the policy move further: the proxy goes up, and below beta = 0.219 the gold goes down.";
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Proxy against gold as the KL budget grows"),
    h("div", { class: "llm-widget__controls" }, [
      slider({
        label: "KL coefficient beta", min: 0.05, max: 2, step: 0.01, value: state.beta, digits: 2,
        onInput: (value) => { state.beta = value; render(); },
      }),
    ]),
    chart,
    h("p", { class: "llm-widget__note" }, "Gold, on a scale from 0 to 0.035:"),
    goldChart,
    note
  );
  render();
});
