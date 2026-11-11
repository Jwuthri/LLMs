// Trusted monitoring with an audit budget: the catch rate of one attack, and the probability that
// the developers catch the model before one attack succeeds. Starts with the numbers of the
// Control and oversight lesson (budget 2%, separation 2: 0.479, 0.261, 0.647).
LLMWidgets.register("audit-budget", (el) => {
  const { h, fmt, slider, bars } = LLMWidgets;
  const ATTACK_WORKS = 0.5; // share of the attacks that cause harm when they are not audited
  const state = { budget: 2, separation: 2 }; // budget in percent of the outputs

  // Standard normal distribution: cumulative probability (Abramowitz and Stegun 26.2.17).
  function cdf(x) {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const poly = t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    const tail = (Math.exp((-x * x) / 2) / Math.sqrt(2 * Math.PI)) * poly;
    return x >= 0 ? 1 - tail : tail;
  }

  // Score that only `share` of the normal outputs exceed: bisection on the cumulative probability.
  function threshold(share) {
    let low = -8;
    let high = 8;
    for (let i = 0; i < 60; i++) {
      const mid = (low + high) / 2;
      if (1 - cdf(mid) > share) low = mid;
      else high = mid;
    }
    return (low + high) / 2;
  }

  const chart = h("div");
  const note = h("p", { class: "llm-widget__note" });

  function render() {
    const cut = threshold(state.budget / 100);
    const caught = 1 - cdf(cut - state.separation);
    const success = (1 - caught) * ATTACK_WORKS;
    const win = caught / (caught + success);
    chart.replaceChildren(
      bars([
        { label: "one attack is audited (c)", value: caught, text: fmt(caught, 3) },
        { label: "one attack succeeds (s)", value: success, text: fmt(success, 3), off: true },
        { label: "a catch comes before a success", value: win, text: fmt(win, 3) },
      ])
    );
    note.textContent =
      `The developers audit each output with a suspicion score above ${fmt(cut, 3)}. ` +
      `For 1,000 tasks, this is ${Math.round(state.budget * 10)} audits. ` +
      `${fmt(caught, 3)} / (${fmt(caught, 3)} + ${fmt(success, 3)}) = ${fmt(win, 3)}. ` +
      "Half of the attacks that are not audited cause harm (assumed).";
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Audit budget, monitor quality, and who wins the game"),
    h("div", { class: "llm-widget__controls" }, [
      slider({
        label: "Audit budget (% of the outputs)", min: 0.5, max: 20, step: 0.5, value: state.budget, digits: 1,
        onInput: (value) => { state.budget = value; render(); },
      }),
      slider({
        label: "Monitor quality (separation of the scores)", min: 0, max: 4, step: 0.1, value: state.separation, digits: 1,
        onInput: (value) => { state.separation = value; render(); },
      }),
    ]),
    chart,
    note
  );
  render();
});
