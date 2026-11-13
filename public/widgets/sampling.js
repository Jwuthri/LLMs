// Temperature, top-k and top-p on the toy logits [0.1, 1.0, 2.0].
LLMWidgets.register("sampling", (el) => {
  const { h, fmt, slider, bars, softmax, toy } = LLMWidgets;
  const state = { temperature: 1, topK: 3, topP: 1 };
  const counts = [0, 0, 0];

  function distribution() {
    const probs = softmax(toy.logits, state.temperature);
    const order = probs.map((p, i) => i).sort((a, b) => probs[b] - probs[a]);
    const kept = new Set();
    let cumulative = 0;
    for (const i of order.slice(0, state.topK)) {
      kept.add(i);
      cumulative += probs[i];
      if (cumulative >= state.topP) break;
    }
    const total = [...kept].reduce((sum, i) => sum + probs[i], 0);
    return probs.map((p, i) => (kept.has(i) ? p / total : 0));
  }

  const chart = h("div");
  const tally = h("p", { class: "llm-widget__note" });

  function render() {
    const probs = distribution();
    chart.replaceChildren(
      bars(toy.vocab.map((token, i) => ({
        label: token,
        value: probs[i],
        text: probs[i] === 0 ? "removed" : fmt(probs[i], 3),
        off: probs[i] === 0,
      })))
    );
    const drawn = counts.reduce((a, b) => a + b, 0);
    tally.textContent = drawn === 0
      ? "Press the button to draw tokens from this distribution."
      : `Drawn ${drawn} tokens: ` + toy.vocab.map((t, i) => `${t} ${counts[i]}`).join(", ") + ".";
  }

  function draw(times) {
    const probs = distribution();
    for (let n = 0; n < times; n++) {
      let r = Math.random();
      let i = 0;
      while (i < probs.length - 1 && r >= probs[i]) r -= probs[i++];
      counts[i] += 1;
    }
    render();
  }

  function change(key) {
    return (value) => {
      state[key] = value;
      counts.fill(0);
      render();
    };
  }

  el.append(
    h("p", { class: "llm-widget__title" }, 'Next token after "the cat": logits [0.1, 1.0, 2.0]'),
    h("div", { class: "llm-widget__controls" }, [
      slider({ label: "Temperature T", min: 0.1, max: 3, step: 0.1, value: 1, digits: 1, onInput: change("temperature") }),
      slider({ label: "Top-k", min: 1, max: 3, step: 1, value: 3, digits: 0, onInput: change("topK") }),
      slider({ label: "Top-p", min: 0.1, max: 1, step: 0.05, value: 1, onInput: change("topP") }),
    ]),
    chart,
    h("div", { style: "margin-top:0.9em" }, [h("button", { onclick: () => draw(100) }, "Draw 100 tokens")]),
    tally
  );
  render();
});
