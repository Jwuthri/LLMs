// Layers of safeguards: the pass rate of one harmful request, and the chance that an attacker
// with many attempts gets one request through. Starts with the numbers of the lesson "Jailbreaks and safeguards".
LLMWidgets.register("safeguard-layers", (el) => {
  const { h, fmt, slider, bars } = LLMWidgets;
  const names = ["Input classifier", "Model", "Output classifier"];
  const state = { miss: [0.2, 0.3, 0.1], attempts: 500, correlated: false };

  const chart = h("div");
  const note = h("p", { class: "llm-widget__note" });
  const box = h("input", { type: "checkbox", id: "safeguard-layers-correlated" });
  box.addEventListener("change", () => {
    state.correlated = box.checked;
    render();
  });

  function render() {
    const independent = state.miss.reduce((a, b) => a * b, 1);
    const rate = state.correlated ? Math.min(...state.miss) : independent;
    const success = 1 - Math.pow(1 - rate, state.attempts);
    chart.replaceChildren(
      bars([
        { label: "one request gets through", value: rate, text: fmt(rate, 4) },
        { label: `1 of ${state.attempts} attempts gets through`, value: success, text: fmt(success, 3) },
      ])
    );
    note.textContent = state.correlated
      ? `One trick defeats all the layers together. The pass rate is the miss rate of the best layer: ${fmt(rate, 2)}.`
      : `Independent layers: ${state.miss.map((m) => fmt(m, 2)).join(" × ")} = ${fmt(independent, 4)}. ` +
        `Of 1,000 harmful requests, ${Math.round(independent * 1000)} get through.`;
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Three layers of safeguards and an attacker with many attempts"),
    h("div", { class: "llm-widget__controls" }, [
      ...names.map((name, i) =>
        slider({
          label: `${name}: miss rate`, min: 0.01, max: 1, step: 0.01, value: state.miss[i],
          onInput: (value) => { state.miss[i] = value; render(); },
        })
      ),
      slider({
        label: "Attempts", min: 1, max: 1000, step: 1, value: state.attempts, digits: 0,
        onInput: (value) => { state.attempts = value; render(); },
      }),
    ]),
    h("div", { class: "llm-widget__controls" }, [
      h("label", { for: "safeguard-layers-correlated" }, [box, " One trick defeats all the layers (correlated failures)"]),
    ]),
    chart,
    note
  );
  render();
});
