// Steering: add alpha times the direction v = [-1, 1] to the hidden state h = [1, 1].
// Starts with the numbers of the interpretability lesson (alpha = 0: probabilities 0.212, 0.212, 0.576).
LLMWidgets.register("steering", (el) => {
  const { h, fmt, slider, bars, softmax, toy } = LLMWidgets;
  const base = [1, 1];
  const v = [-1, 1];
  const threshold = 0.2;
  const state = { alpha: 0 };

  const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
  const vector = (x, digits = 1) => `[${x.map((value) => fmt(value, digits)).join(", ")}]`;

  const chart = h("div");
  const note = h("p", { class: "llm-widget__note" });

  function render() {
    const steered = base.map((x, i) => x + state.alpha * v[i]);
    const logits = toy.embeddings.map((row) => dot(row, steered));
    const probs = softmax(logits);
    const score = dot(steered, v);
    chart.replaceChildren(
      bars(toy.vocab.map((token, i) => ({ label: token, value: probs[i], text: fmt(probs[i], 3) })))
    );
    const top = probs.indexOf(Math.max(...probs));
    note.textContent =
      `h' = [1, 1] ${state.alpha < 0 ? "−" : "+"} ${fmt(Math.abs(state.alpha), 1)} × [-1, 1] = ${vector(steered)}. Logits: ${vector(logits)}. ` +
      `Probe score h' · v = ${fmt(score, 1)}: ${score > threshold ? "above" : "not above"} the threshold 0.2. ` +
      (probs[top] > 0.95
        ? `The steering is too strong: "${toy.vocab[top]}" takes almost all the probability.`
        : state.alpha < 0
          ? "A negative strength pushes away from the concept."
          : "");
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Steering: h' = h + α · v, then logits = E · h'"),
    h("div", { class: "llm-widget__controls" }, [
      slider({
        label: "Steering strength α", min: -4, max: 8, step: 0.5, value: 0, digits: 1,
        onInput: (value) => { state.alpha = value; render(); },
      }),
    ]),
    chart,
    note
  );
  render();
});
