// A detector with 10 scored requests: move the threshold and read the confusion matrix and the rates.
// Starts with the numbers of the lesson "Metrics and losses": threshold 0.5, precision 0.75, recall 0.75.
LLMWidgets.register("threshold-metrics", (el) => {
  const { h, fmt, slider, bars } = LLMWidgets;
  const harmful = [0.9, 0.8, 0.6, 0.3];
  const harmless = [0.7, 0.4, 0.35, 0.2, 0.1, 0.05];
  const state = { threshold: 0.5 };

  const counts = h("p", { class: "llm-widget__note" });
  const chart = h("div");
  const note = h("p", { class: "llm-widget__note" });

  function render() {
    const tp = harmful.filter((s) => s >= state.threshold).length;
    const fp = harmless.filter((s) => s >= state.threshold).length;
    const fn = harmful.length - tp;
    const tn = harmless.length - fp;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp / harmful.length;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    counts.textContent = `True positives ${tp}, false negatives ${fn}, false positives ${fp}, true negatives ${tn}.`;
    chart.replaceChildren(
      bars([
        { label: "accuracy", value: (tp + tn) / 10, text: fmt((tp + tn) / 10, 3) },
        { label: "precision", value: precision, text: tp + fp > 0 ? fmt(precision, 3) : "no flag" },
        { label: "recall", value: recall, text: fmt(recall, 3) },
        { label: "F1", value: f1, text: fmt(f1, 3) },
        { label: "false positive rate", value: fp / harmless.length, text: fmt(fp / harmless.length, 3) },
      ])
    );
    note.textContent =
      "The AUROC stays at 0.833 for each threshold: it measures the order of the scores, not one decision.";
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "One detector, many thresholds"),
    h("div", { class: "llm-widget__controls" }, [
      slider({
        label: "Threshold", min: 0, max: 1, step: 0.05, value: state.threshold,
        onInput: (value) => { state.threshold = value; render(); },
      }),
    ]),
    counts,
    chart,
    note
  );
  render();
});
