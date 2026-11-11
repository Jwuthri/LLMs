// Work of generation with and without a key-value cache (one layer, one head).
// Pass t reads a context of t tokens. Without a cache it computes t key-value pairs and
// t(t+1)/2 dot products. With a cache it computes 1 pair and t dot products.
LLMWidgets.register("kv-cache", (el) => {
  const { h, bars, toy } = LLMWidgets;
  const START = 3; // "the cat sat": the numbers of the page
  const LIMIT = 1000;
  const state = { passes: 0, kvNoCache: 0, kvCache: 0, dotsNoCache: 0, dotsCache: 0 };

  function runPass() {
    const t = state.passes + 1;
    state.passes = t;
    state.kvNoCache += t;
    state.kvCache += 1;
    state.dotsNoCache += (t * (t + 1)) / 2;
    state.dotsCache += t;
  }

  function reset() {
    for (const key of Object.keys(state)) state[key] = 0;
    for (let i = 0; i < START; i++) runPass();
    render();
  }

  function generate(count) {
    for (let i = 0; i < count && state.passes < LIMIT; i++) runPass();
    render();
  }

  const number = (x) => x.toLocaleString("en-US");
  const context = h("p", { class: "llm-widget__note", style: "margin:0 0 0.8em" });
  const lastPass = h("table", { style: "margin-bottom:1em" });
  const kvChart = h("div");
  const dotsChart = h("div");
  const note = h("p", { class: "llm-widget__note" });
  const heading = (text) =>
    h("p", { style: "margin:0.9em 0 0.4em; font-weight:600; color:var(--llm-ink-2)" }, text);
  const nextButton = h("button", { onclick: () => generate(1) }, "Generate next token");
  const moreButton = h("button", { class: "secondary", onclick: () => generate(100) }, "Generate 100 tokens");
  const resetButton = h("button", { class: "secondary", onclick: reset }, "Reset");

  function pair(target, withoutCache, withCache) {
    target.replaceChildren(
      bars(
        [
          { label: "no cache", value: withoutCache, text: number(withoutCache) },
          { label: "cache", value: withCache, text: number(withCache) },
        ],
        withoutCache
      )
    );
  }

  function render() {
    const t = state.passes;
    const shown = Math.min(t, 8);
    const words = [];
    for (let i = t - shown; i < t; i++) words.push(toy.vocab[i % toy.vocab.length]);
    context.textContent = `Context: ${t > shown ? "… " : ""}${words.join(" ")} (${number(t)} tokens)`;

    lastPass.replaceChildren(
      h("tr", {}, [h("th", {}, `Pass ${number(t)}`), h("th", {}, "No cache"), h("th", {}, "Cache")]),
      h("tr", {}, [h("td", {}, "Key-value pairs computed"), h("td", {}, number(t)), h("td", {}, "1")]),
      h("tr", {}, [h("td", {}, "Dot products"), h("td", {}, number((t * (t + 1)) / 2)), h("td", {}, number(t))])
    );
    pair(kvChart, state.kvNoCache, state.kvCache);
    pair(dotsChart, state.dotsNoCache, state.dotsCache);
    note.textContent =
      `Totals of passes 1 to ${number(t)}. The cache needs ` +
      `${(state.dotsNoCache / state.dotsCache).toFixed(1)} times fewer dot products and holds ` +
      `${number(t)} key-value pairs in memory.`;
    nextButton.disabled = t >= LIMIT;
    moreButton.disabled = t >= LIMIT;
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Key-value cache: work to generate token after token"),
    h("div", { class: "llm-widget__controls" }, [nextButton, moreButton, resetButton]),
    context,
    lastPass,
    heading("Total key-value pairs computed"),
    kvChart,
    heading("Total dot products"),
    dotsChart,
    note
  );
  reset();
});
