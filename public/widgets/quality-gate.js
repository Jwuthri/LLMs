// A judge model gives each of the 8 toy samples a score from 1 to 5. Move the lowest score that is kept
// and read how many samples stay and how many of them are really correct.
// Starts with the numbers of the lesson "What is a good dataset?": threshold 3, 6 kept, accuracy 0.833.
LLMWidgets.register("quality-gate", (el) => {
  const { h, fmt, slider, bars } = LLMWidgets;
  const samples = [
    { id: 1, judge: 5, correct: true }, { id: 2, judge: 2, correct: true },
    { id: 3, judge: 5, correct: true }, { id: 4, judge: 4, correct: false },
    { id: 5, judge: 4, correct: true }, { id: 6, judge: 5, correct: true },
    { id: 7, judge: 4, correct: true }, { id: 8, judge: 1, correct: false },
  ];
  const state = { threshold: 3 };
  const chart = h("div");
  const note = h("p", { class: "llm-widget__note" });

  function render() {
    const kept = samples.filter((s) => s.judge >= state.threshold);
    const good = kept.filter((s) => s.correct).length;
    const lost = samples.filter((s) => s.correct && s.judge < state.threshold).map((s) => s.id);
    const wrong = kept.filter((s) => !s.correct).map((s) => s.id);
    chart.replaceChildren(
      bars([
        { label: "share of samples kept", value: kept.length / 8, text: `${kept.length} of 8` },
        { label: "accuracy of the kept samples", value: kept.length ? good / kept.length : 0, text: kept.length ? fmt(good / kept.length, 3) : "none" },
      ])
    );
    note.textContent =
      `Correct samples that the judge removes: ${lost.length ? lost.join(", ") : "none"}. ` +
      `Wrong samples that the judge keeps: ${wrong.length ? wrong.join(", ") : "none"}.`;
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "A judge score as a filter"),
    h("div", { class: "llm-widget__controls" }, [
      slider({
        label: "Lowest judge score that is kept", min: 1, max: 5, step: 1, value: state.threshold, digits: 0,
        onInput: (value) => { state.threshold = value; render(); },
      }),
    ]),
    chart,
    note
  );
  render();
});
