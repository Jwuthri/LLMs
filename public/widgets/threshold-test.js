// A capability test against a threshold: the score, its 95% interval and the decision.
// Starts with the numbers of the lesson "Risks and safety frameworks": 7 of 20 tasks, threshold 0.5.
LLMWidgets.register("threshold-test", (el) => {
  const { h, fmt, slider, bars } = LLMWidgets;
  const state = { tasks: 20, score: 0.35, threshold: 0.5 };

  const chart = h("div");
  const note = h("p", { class: "llm-widget__note" });

  function render() {
    const se = Math.sqrt((state.score * (1 - state.score)) / state.tasks);
    const low = Math.max(0, state.score - 1.96 * se);
    const high = Math.min(1, state.score + 1.96 * se);
    chart.replaceChildren(
      bars([
        { label: "lower end of the interval", value: low, text: fmt(low, 3) },
        { label: "score", value: state.score, text: fmt(state.score, 3) },
        { label: "upper end of the interval", value: high, text: fmt(high, 3) },
        { label: "threshold", value: state.threshold, text: fmt(state.threshold, 2) },
      ])
    );
    note.textContent =
      high >= state.threshold
        ? `The upper end ${fmt(high, 3)} is at or above the threshold ${fmt(state.threshold, 2)}: stronger safeguards. ` +
          (state.score < state.threshold ? "The score is below the threshold, but the test is too small to be sure." : "")
        : `The upper end ${fmt(high, 3)} is below the threshold ${fmt(state.threshold, 2)}: current safeguards.`;
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "A test score, its interval and the if-then rule"),
    h("div", { class: "llm-widget__controls" }, [
      slider({
        label: "Number of test tasks", min: 5, max: 400, step: 1, value: state.tasks, digits: 0,
        onInput: (value) => { state.tasks = value; render(); },
      }),
      slider({
        label: "Share of tasks completed", min: 0, max: 1, step: 0.01, value: state.score,
        onInput: (value) => { state.score = value; render(); },
      }),
      slider({
        label: "Threshold", min: 0.1, max: 0.9, step: 0.05, value: state.threshold,
        onInput: (value) => { state.threshold = value; render(); },
      }),
    ]),
    chart,
    note
  );
  render();
});
