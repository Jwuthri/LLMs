// Masked diffusion on "the cat sat": start from 3 masks, unmask the most confident positions in each step.
// The logits are the assumed values of the page. Keep in sync with MODEL_LOGITS in examples/scientist/diffusion.py.
LLMWidgets.register("text-diffusion", (el) => {
  const { h, fmt, slider, bars, softmax, toy } = LLMWidgets;
  const MASK = "[M]";
  const LENGTH = 3;
  // Partly masked sentence -> logits at each masked position (one model call).
  const MODEL = {
    "[M] [M] [M]": { 0: [2.0, 0.0, 0.0], 1: [0.3, 0.8, 0.6], 2: [0.2, 0.9, 0.8] },
    "the [M] [M]": { 1: [0.5, 1.5, 0.8], 2: toy.logits },
    "the [M] sat": { 1: [0.0, 3.0, 0.5] },
  };
  const state = { steps: 3, done: 0, tokens: [], fresh: [] };

  // One model call: the best token and its probability (the confidence) at each masked position.
  function predict(tokens) {
    const guesses = [];
    for (const [position, logits] of Object.entries(MODEL[tokens.join(" ")] ?? {})) {
      const probs = softmax(logits);
      const best = probs.indexOf(Math.max(...probs));
      guesses.push({ position: Number(position), token: toy.vocab[best], confidence: probs[best] });
    }
    return guesses;
  }

  function reset() {
    state.done = 0;
    state.tokens = Array(LENGTH).fill(MASK);
    state.fresh = [];
    render();
  }

  function nextStep() {
    const guesses = predict(state.tokens);
    const visible = LENGTH - guesses.length;
    state.done += 1;
    const count = Math.floor((LENGTH * state.done) / state.steps) - visible;
    const chosen = [...guesses].sort((a, b) => b.confidence - a.confidence).slice(0, count);
    for (const guess of chosen) state.tokens[guess.position] = guess.token;
    state.fresh = chosen.map((guess) => guess.position);
    render();
  }

  const sentence = h("div", { style: "display:flex; flex-wrap:wrap; gap:0.5em; margin:0 0 1em" });
  const chart = h("div");
  const calls = h("p", { class: "llm-widget__note" });
  const note = h("p", { class: "llm-widget__note" });
  const nextButton = h("button", { onclick: nextStep }, "Next step");
  const resetButton = h("button", { class: "secondary", onclick: reset }, "Reset");

  function tokenBox(token, isFresh) {
    const masked = token === MASK;
    const border = masked ? "1.5px dashed var(--llm-axis)" : `1.5px solid ${isFresh ? "var(--llm-s1)" : "var(--llm-axis)"}`;
    const background = masked ? "transparent" : isFresh ? "var(--llm-s1-soft)" : "var(--llm-panel)";
    return h(
      "span",
      {
        style:
          `min-width:4.2em; padding:0.35em 0.8em; border:${border}; border-radius:6px; background:${background}; ` +
          `color:${masked ? "var(--llm-muted)" : "var(--llm-ink)"}; text-align:center; ` +
          "font-family:var(--font-mono), ui-monospace, monospace",
      },
      token
    );
  }

  function render() {
    const guesses = predict(state.tokens);
    const finished = guesses.length === 0;
    sentence.replaceChildren(...state.tokens.map((token, i) => tokenBox(token, state.fresh.includes(i))));

    chart.replaceChildren(
      bars(
        state.tokens.map((token, i) => {
          const guess = guesses.find((g) => g.position === i);
          return guess
            ? { label: `position ${i + 1}: ${guess.token}`, value: guess.confidence, text: fmt(guess.confidence, 3) }
            : { label: `position ${i + 1}: ${token}`, value: 0, text: "visible", off: true };
        })
      )
    );

    calls.textContent =
      `Model calls: diffusion ${state.done} of ${state.steps}. ` +
      `An autoregressive model needs ${LENGTH} calls for ${LENGTH} tokens (one call for each token).`;

    const text = state.tokens.join(" ");
    if (!finished) {
      const visible = LENGTH - guesses.length;
      const count = Math.floor((LENGTH * (state.done + 1)) / state.steps) - visible;
      note.textContent =
        `The bars show the best token and its confidence at each masked position. ` +
        `The next step unmasks the ${count === 1 ? "most confident position" : `${count} most confident positions`}.`;
    } else if (text === "the cat sat") {
      note.textContent =
        `Result: "${text}" after ${state.steps} calls. ` +
        (state.steps === LENGTH
          ? "The order was position 1, then 3, then 2: not from left to right."
          : "Positions 2 and 3 were unmasked in the same call, and here the two choices agree.");
    } else {
      note.textContent =
        `Result: "${text}" after 1 call. Positions 2 and 3 chose "cat" in the same call. ` +
        "Each position could not see the choice of the other position.";
    }
    nextButton.disabled = finished;
  }

  el.append(
    h("p", { class: "llm-widget__title" }, 'Masked diffusion: write "the cat sat" from 3 masks'),
    h("div", { class: "llm-widget__controls" }, [
      slider({
        label: "Number of steps",
        min: 1,
        max: LENGTH,
        step: 1,
        value: state.steps,
        digits: 0,
        onInput: (value) => {
          state.steps = value;
          reset();
        },
      }),
    ]),
    sentence,
    chart,
    h("div", { style: "display:flex; gap:0.6em; margin-top:0.9em" }, [nextButton, resetButton]),
    calls,
    note
  );
  reset();
});
