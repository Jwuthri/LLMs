// Label noise in a preference dataset: the measured preference rate and its 95% interval, for a true rate,
// a share of swapped labels and a number of pairs. Starts with the numbers of the lesson "Preference datasets".
LLMWidgets.register("label-noise", (el) => {
  const { h, fmt, slider, bars } = LLMWidgets;
  const state = { rate: 0.8, swapped: 0.1, pairs: 100 };

  const chart = h("div");
  const note = h("p", { class: "llm-widget__note" });

  function render() {
    const measured = state.rate * (1 - state.swapped) + (1 - state.rate) * state.swapped;
    const half = 1.96 * Math.sqrt((measured * (1 - measured)) / state.pairs);
    const low = Math.max(0, measured - half);
    const high = Math.min(1, measured + half);
    chart.replaceChildren(
      bars([
        { label: "true rate", value: state.rate, text: fmt(state.rate, 3) },
        { label: "measured rate", value: measured, text: fmt(measured, 3) },
        { label: "interval, lower limit", value: low, text: fmt(low, 3), off: true },
        { label: "interval, upper limit", value: high, text: fmt(high, 3), off: true },
      ])
    );
    const formula =
      `${fmt(state.rate, 2)} × ${fmt(1 - state.swapped, 2)} + ${fmt(1 - state.rate, 2)} × ${fmt(state.swapped, 2)}` +
      ` = ${fmt(measured, 3)}. With ${state.pairs.toLocaleString("en-US")} pairs, the 95% interval is ` +
      `[${fmt(low, 3)}, ${fmt(high, 3)}].`;
    let verdict;
    if (Math.abs(state.swapped - 0.5) < 0.001) verdict = " Half of the labels are swapped: the labels say nothing.";
    else if (low <= 0.5 && high >= 0.5) verdict = " The interval contains 0.5: these pairs do not show a preference.";
    else if (state.rate < low || state.rate > high) verdict = " The true rate is outside the interval: the noise is larger than the sampling error.";
    else verdict = " The true rate is inside the interval: with this number of pairs, you cannot see the noise.";
    note.textContent = formula + verdict;
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Swapped labels pull a preference rate towards 0.5"),
    h("div", { class: "llm-widget__controls" }, [
      slider({
        label: "True preference rate", min: 0.5, max: 1, step: 0.01, value: state.rate,
        onInput: (value) => { state.rate = value; render(); },
      }),
      slider({
        label: "Share of swapped labels", min: 0, max: 0.5, step: 0.01, value: state.swapped,
        onInput: (value) => { state.swapped = value; render(); },
      }),
      slider({
        label: "Number of pairs", min: 10, max: 10000, step: 10, value: state.pairs, digits: 0,
        onInput: (value) => { state.pairs = value; render(); },
      }),
    ]),
    chart,
    note
  );
  render();
});
