// Byte-pair encoding (BPE) merge stepper on a small corpus that the reader can edit.
// The rules are the same as examples/scientist/architecture.py: count each adjacent pair,
// merge the most frequent pair, and for equal counts take the last pair in a-z order.
LLMWidgets.register("bpe", (el) => {
  const { h } = LLMWidgets;
  const DEFAULT_CORPUS = "cat cat sat the";
  let words = []; // [{ symbols: ["c", "a", "t"], count: 2 }]
  let merges = []; // [["a", "t"], ...]
  let baseVocab = [];

  function start(text) {
    const counts = new Map();
    for (const word of text.toLowerCase().split(/\s+/).filter(Boolean)) {
      counts.set(word, (counts.get(word) || 0) + 1);
    }
    words = [...counts].map(([word, count]) => ({ symbols: [...word], count }));
    merges = [];
    baseVocab = [...new Set(words.flatMap((w) => w.symbols))].sort();
    render();
  }

  function pairCounts() {
    const counts = new Map(); // "a\u0000t" -> { pair: ["a", "t"], count: 3 }
    for (const { symbols, count } of words) {
      for (let i = 0; i < symbols.length - 1; i++) {
        const key = symbols[i] + "\u0000" + symbols[i + 1];
        const entry = counts.get(key) || { pair: [symbols[i], symbols[i + 1]], count: 0 };
        entry.count += count;
        counts.set(key, entry);
      }
    }
    return [...counts.values()];
  }

  // The most frequent pair. Equal counts: the last pair in a-z order.
  function bestPair(pairs) {
    const later = (a, b) => a.pair[0] > b.pair[0] || (a.pair[0] === b.pair[0] && a.pair[1] > b.pair[1]);
    return pairs.reduce((best, p) => (p.count > best.count || (p.count === best.count && later(p, best)) ? p : best));
  }

  function mergeNext() {
    const pairs = pairCounts();
    if (pairs.length === 0) return;
    const [left, right] = bestPair(pairs).pair;
    for (const word of words) {
      const out = [];
      for (let i = 0; i < word.symbols.length; i++) {
        if (word.symbols[i] === left && word.symbols[i + 1] === right) {
          out.push(left + right);
          i += 1;
        } else {
          out.push(word.symbols[i]);
        }
      }
      word.symbols = out;
    }
    merges.push([left, right]);
    render();
  }

  const chip = (text, accent) =>
    h("code", {
      style:
        "display:inline-block;margin:0 0.3em 0.3em 0;padding:0.1em 0.45em;border-radius:4px;" +
        `border:1px solid var(${accent ? "--llm-s1" : "--llm-axis"});` +
        `background:var(${accent ? "--llm-s1-soft" : "--llm-panel"});color:var(--llm-ink)`,
    }, text);
  const heading = (text) => h("p", { style: "margin:0.9em 0 0.3em;font-weight:600" }, text);

  const input = h("input", { type: "text", value: DEFAULT_CORPUS, "aria-label": "Corpus", style: "flex:1 1 220px" });
  input.addEventListener("input", () => start(input.value));
  const mergeButton = h("button", { onclick: mergeNext }, "Merge next pair");
  const resetButton = h("button", { class: "secondary", onclick: () => start(input.value) }, "Reset");
  const body = h("div");
  const note = h("p", { class: "llm-widget__note" });

  function render() {
    const pairs = pairCounts().sort((a, b) => b.count - a.count);
    const best = pairs.length ? bestPair(pairs) : null;
    const mergedTokens = merges.map(([a, b]) => a + b);
    const tokenTotal = words.reduce((sum, w) => sum + w.symbols.length * w.count, 0);
    mergeButton.disabled = !best;

    const wordRows = words.map((w) =>
      h("tr", {}, [
        h("td", {}, w.symbols.join("")),
        h("td", { style: "text-align:left" }, w.symbols.map((s) => chip(s, mergedTokens.includes(s)))),
        h("td", {}, w.count),
        h("td", {}, w.symbols.length * w.count),
      ])
    );
    const pairRow = pairs.length
      ? pairs.map((p) => chip(`${p.pair[0]} ${p.pair[1]} : ${p.count}`, p === best))
      : "No pair is left. Each word is one token.";
    const mergeRow = merges.length
      ? merges.map(([a, b], i) => chip(`${i + 1}. ${a} + ${b} → ${a}${b}`, true))
      : "No merge yet.";

    body.replaceChildren(
      heading("Words and their current tokens"),
      // A table with a class does not get the wide-table style of the theme, so it fits a phone.
      h("table", { class: "llm-bpe-words", style: "width:100%" }, [
        h("tr", {}, [h("th", {}, "Word"), h("th", { style: "text-align:left" }, "Tokens"), h("th", {}, "Count"), h("th", {}, "Tokens × count")]),
        ...wordRows,
      ]),
      heading("Pair counts (the next merge is blue)"),
      h("div", {}, pairRow),
      heading("Merges, in order"),
      h("div", {}, mergeRow),
      heading(`Vocabulary: ${baseVocab.length + merges.length} tokens`),
      h("div", {}, [...baseVocab.map((s) => chip(s, false)), ...mergedTokens.map((s) => chip(s, true))])
    );
    note.textContent = words.length
      ? `The corpus is ${tokenTotal} tokens long after ${merges.length} merge${merges.length === 1 ? "" : "s"}. ` +
        "For equal counts, the widget merges the last pair in a-z order."
      : "Type one or more words.";
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Byte-pair encoding: merge the most frequent pair"),
    h("div", { class: "llm-widget__controls", style: "align-items:center" }, [input, mergeButton, resetButton]),
    body,
    note
  );
  start(DEFAULT_CORPUS);
});
