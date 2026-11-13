// Gradient descent on the one-weight loss L(w) = (w - 2)^2 of the mathematics page.
// Start: w = 0, learning rate 0.1. Keep in sync with examples/fundamentals/mathematics.py.
LLMWidgets.register("gradient-descent", (el) => {
  const { h, fmt, slider } = LLMWidgets;
  const target = 2;
  const start = 0;
  const maxLoss = 1e9; // stop the steps when the loss is this large
  const state = { rate: 0.1, path: [start] };

  const loss = (w) => (w - target) ** 2;
  const gradient = (w) => 2 * (w - target);

  // Chart geometry: w from -3 to 7 (the minimum is in the middle), loss from 0 to 25.
  const box = { width: 480, height: 250, left: 34, right: 10, top: 10, bottom: 30 };
  const range = { wMin: -3, wMax: 7, lossMax: 25 };
  const px = (w) => box.left + ((w - range.wMin) / (range.wMax - range.wMin)) * (box.width - box.left - box.right);
  const py = (l) => box.height - box.bottom - (l / range.lossMax) * (box.height - box.top - box.bottom);
  const point = (w) => `${px(w).toFixed(1)},${py(loss(w)).toFixed(1)}`;

  const svg = h("svg", {
    viewBox: `0 0 ${box.width} ${box.height}`,
    role: "img",
    "aria-label": "Loss curve L(w) = (w - 2) squared with the points of each gradient descent step",
    style: "display:block;width:100%;max-width:560px;height:auto;margin:0 auto",
  });
  // A class on the table turns off the page theme for Markdown tables: the cells stay compact.
  const table = h("table", { class: "llm-widget__table" });
  const note = h("p", { class: "llm-widget__note" });
  const stepButton = h("button", { onclick: step }, "Step");

  function axes() {
    const parts = [
      h("line", { x1: box.left, y1: py(0), x2: box.width - box.right, y2: py(0), style: "stroke:var(--llm-axis)" }),
      h("line", { x1: box.left, y1: py(0), x2: box.left, y2: box.top, style: "stroke:var(--llm-axis)" }),
      h("text", { x: box.width - box.right, y: box.height - 4, "text-anchor": "end" }, "weight w"),
      h("text", { x: px(target), y: box.top + 10, "text-anchor": "middle" }, "loss L(w)"),
    ];
    for (let w = -2; w <= 6; w += 2) {
      parts.push(h("text", { x: px(w), y: py(0) + 14, "text-anchor": "middle" }, w));
    }
    for (let l = 0; l <= 20; l += 10) {
      parts.push(h("text", { x: box.left - 6, y: py(l) + 4, "text-anchor": "end" }, l));
    }
    const curve = [];
    for (let w = range.wMin; w <= range.wMax + 1e-9; w += 0.1) curve.push(point(w));
    parts.push(h("polyline", { points: curve.join(" "), style: "fill:none;stroke:var(--llm-s1);stroke-width:2" }));
    return parts;
  }

  function step() {
    const w = state.path[state.path.length - 1];
    state.path.push(w - state.rate * gradient(w));
    render();
  }

  function reset() {
    state.path = [start];
    render();
  }

  function render() {
    const w = state.path[state.path.length - 1];
    const g = gradient(w);
    // Half-width of the tangent line in w units, so that the line is 60 px long on the screen.
    const half = 30 / Math.hypot(px(1) - px(0), g * (py(0) - py(1)));

    svg.replaceChildren(
      ...axes(),
      h("polyline", {
        points: state.path.map(point).join(" "),
        style: "fill:none;stroke:var(--llm-muted);stroke-width:1.2;stroke-dasharray:4 3",
      }),
      ...state.path.slice(0, -1).map((v) =>
        h("circle", { cx: px(v), cy: py(loss(v)), r: 3, style: "fill:var(--llm-muted)" })
      ),
      h("line", {
        x1: px(w - half), y1: py(loss(w) - g * half), x2: px(w + half), y2: py(loss(w) + g * half),
        style: "stroke:var(--llm-s2);stroke-width:2",
      }),
      h("circle", { cx: px(w), cy: py(loss(w)), r: 5, style: "fill:var(--llm-s2)" })
    );

    const first = Math.max(0, state.path.length - 5);
    table.replaceChildren(
      h("tr", {}, ["Step", "Weight w", "Loss L(w)", "Gradient"].map((name) => h("th", {}, name))),
      ...state.path.slice(first).map((v, i) =>
        h("tr", {}, [first + i, fmt(v, 3), fmt(loss(v), 3), fmt(gradient(v), 3)].map((cell) => h("td", {}, cell)))
      )
    );

    const factor = Math.abs(1 - 2 * state.rate);
    const behaviour = factor < 1
      ? (state.rate > 0.5 ? "w jumps over the minimum at each step, but the loss goes down." : "The loss goes down at each step.")
      : factor === 1
        ? "w jumps between two points. The loss does not change."
        : "The loss goes up at each step: the learning rate is too large.";
    const outside = w < range.wMin || w > range.wMax ? " The point is now outside the chart." : "";
    stepButton.disabled = loss(w) > maxLoss;
    note.textContent =
      `Next step: w = ${fmt(w, 3)} − ${fmt(state.rate, 2)} × (${fmt(g, 3)}) = ${fmt(w - state.rate * g, 3)}. ` +
      `Each step multiplies the distance to w = 2 by ${fmt(factor, 2)}. ${behaviour}${outside}`;
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Gradient descent on L(w) = (w − 2)², start at w = 0"),
    h("div", { class: "llm-widget__controls" }, [
      slider({
        label: "Learning rate (a change starts again at w = 0)", min: 0.05, max: 1.2, step: 0.05, value: 0.1,
        onInput: (value) => { state.rate = value; reset(); },
      }),
    ]),
    svg,
    h("div", { style: "display:flex;gap:0.6em;margin:0.7em 0" }, [
      stepButton,
      h("button", { class: "secondary", onclick: reset }, "Reset"),
    ]),
    h("div", { style: "overflow-x:auto" }, [table]),
    note
  );
  render();
});
