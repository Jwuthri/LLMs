// Evaluation awareness: what a behaviour test reports, against the rate in real use.
// Starts with the numbers of the lesson "Scheming and alignment evaluations".
LLMWidgets.register("eval-awareness", (el) => {
  const { h, fmt, slider, bars } = LLMWidgets;
  const MAX_RATE = 0.5;
  const state = { awareness: 0.8, tested: 0.05, inUse: 0.25 };

  const chart = h("div");
  const note = h("p", { class: "llm-widget__note" });

  function render() {
    const { awareness, tested, inUse } = state;
    const reported = awareness * tested + (1 - awareness) * inUse;
    chart.replaceChildren(
      bars(
        [
          { label: "the test reports", value: reported, text: fmt(reported, 3) },
          { label: "real use", value: inUse, text: fmt(inUse, 3) },
        ],
        MAX_RATE
      )
    );
    const sum = `${fmt(awareness, 2)} × ${fmt(tested, 2)} + ${fmt(1 - awareness, 2)} × ${fmt(inUse, 2)} = ${fmt(reported, 3)}.`;
    let reading;
    if (Math.abs(inUse - reported) < 1e-9) {
      reading = "The test reports the rate in real use.";
    } else if (reported > inUse) {
      reading = "This model is less safe when it thinks that it is in a test. The test reports more than real use.";
    } else if (reported === 0) {
      reading = "The test reports no unsafe action, but real use has some.";
    } else {
      reading = `The rate in real use is ${fmt(inUse / reported, 2)} times the rate that the test reports.`;
    }
    note.textContent = `${sum} ${reading}`;
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "What a test reports when the model can identify the test"),
    h("div", { class: "llm-widget__controls" }, [
      slider({
        label: "Evaluation awareness", min: 0, max: 1, step: 0.01, value: state.awareness,
        onInput: (value) => { state.awareness = value; render(); },
      }),
      slider({
        label: "Unsafe rate when the model thinks that it is in a test", min: 0, max: MAX_RATE, step: 0.01,
        value: state.tested,
        onInput: (value) => { state.tested = value; render(); },
      }),
      slider({
        label: "Unsafe rate when it thinks that nobody checks (real use)", min: 0, max: MAX_RATE, step: 0.01,
        value: state.inUse,
        onInput: (value) => { state.inUse = value; render(); },
      }),
    ]),
    chart,
    note
  );
  render();
});
