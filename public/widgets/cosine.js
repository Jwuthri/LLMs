// Cosine similarity and dot product between a query vector and the 3 chunk vectors
// of docs/engineer/02-vector-storage.md. Keep the vectors in sync with examples/engineer/corpus.py.
LLMWidgets.register("cosine", (el) => {
  const { h, fmt, slider } = LLMWidgets;
  const chunks = [
    { id: "c1", text: "The cat sat on the mat.", v: [1, 2 / 3], colour: "var(--llm-s1)" },
    { id: "c2", text: "The cat sleeps all day.", v: [0.4, 1.0], colour: "var(--llm-s2)" },
    { id: "c3", text: "The cat eats fish.", v: [0.25, 1.5], colour: "var(--llm-s3)" },
  ];
  // The query starts at the embedding of "the cat sat": [0.667, 0.667].
  const state = { q: [2 / 3, 2 / 3], normalize: false };

  // Plane geometry: viewBox units for one vector unit, and the position of the origin.
  const WIDTH = 320;
  const HEIGHT = 300;
  const SCALE = 130;
  const ORIGIN = [36, 264];
  const MAX_LENGTH = 2;
  const px = (v) => [ORIGIN[0] + v[0] * SCALE, ORIGIN[1] - v[1] * SCALE];

  const norm = (v) => Math.hypot(v[0], v[1]);
  const unit = (v) => [v[0] / norm(v), v[1] / norm(v)];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
  const cosine = (a, b) => dot(a, b) / (norm(a) * norm(b));

  const svg = h("svg", {
    viewBox: `0 0 ${WIDTH} ${HEIGHT}`,
    role: "img",
    "aria-label": "Plane with the 3 chunk vectors and the query vector. Drag the query.",
    style: "flex:1 1 260px;max-width:340px;touch-action:none;cursor:crosshair",
  });
  const lineStyle = (colour, width) => `stroke:${colour};stroke-width:${width};fill:none`;

  // Static parts: grid, axes, the circle of length 1.
  for (const tick of [0.5, 1, 1.5, 2]) {
    const [x, y] = px([tick, tick]);
    svg.append(
      h("line", { x1: x, y1: ORIGIN[1], x2: x, y2: px([0, 2])[1], style: lineStyle("var(--llm-grid)", 1) }),
      h("line", { x1: ORIGIN[0], y1: y, x2: px([2, 0])[0], y2: y, style: lineStyle("var(--llm-grid)", 1) })
    );
    if (Number.isInteger(tick)) {
      svg.append(
        h("text", { x, y: ORIGIN[1] + 16, "text-anchor": "middle" }, tick),
        h("text", { x: ORIGIN[0] - 8, y: y + 4, "text-anchor": "end" }, tick)
      );
    }
  }
  svg.append(
    h("line", { x1: ORIGIN[0], y1: ORIGIN[1], x2: px([2, 0])[0], y2: ORIGIN[1], style: lineStyle("var(--llm-axis)", 1.5) }),
    h("line", { x1: ORIGIN[0], y1: ORIGIN[1], x2: ORIGIN[0], y2: px([0, 2])[1], style: lineStyle("var(--llm-axis)", 1.5) }),
    h("path", {
      d: `M${px([1, 0]).join(" ")}A${SCALE} ${SCALE} 0 0 0 ${px([0, 1]).join(" ")}`,
      style: lineStyle("var(--llm-muted)", 1) + ";stroke-dasharray:4 4",
    }),
    h("text", { x: ORIGIN[0] - 8, y: ORIGIN[1] + 16, "text-anchor": "end" }, "0")
  );

  // Moving parts: one line, dot and label per chunk, then the query on top.
  const chunkMarks = chunks.map((chunk) => ({
    line: h("line", { x1: ORIGIN[0], y1: ORIGIN[1], style: lineStyle(chunk.colour, 2.5) }),
    tip: h("circle", { r: 4, style: `fill:${chunk.colour}` }),
    label: h("text", { style: "fill:var(--llm-ink);font-weight:600" }, chunk.id),
  }));
  chunkMarks.forEach((mark) => svg.append(mark.line, mark.tip, mark.label));
  const queryLine = h("line", { x1: ORIGIN[0], y1: ORIGIN[1], style: lineStyle("var(--llm-ink)", 2.5) });
  const queryHandle = h("circle", { r: 9, style: "fill:var(--llm-ink);stroke:var(--llm-surface);stroke-width:2;cursor:grab" });
  const queryLabel = h("text", { "text-anchor": "end", style: "fill:var(--llm-ink);font-weight:600" }, "query");
  svg.append(queryLine, queryHandle, queryLabel);

  const table = h("table");
  const ranks = h("p", { class: "llm-widget__note" });
  const note = h("p", { class: "llm-widget__note" });

  const angleControl = slider({
    label: "Angle of the query (degrees)", min: 0, max: 90, step: 1, value: 45, digits: 0,
    onInput: (angle) => setPolar(norm(state.q), angle),
  });
  const lengthControl = slider({
    label: "Length of the query", min: 0.1, max: MAX_LENGTH, step: 0.001, value: 0.943, digits: 3,
    onInput: (length) => setPolar(length, (Math.atan2(state.q[1], state.q[0]) * 180) / Math.PI),
  });
  const normalizeBox = h("input", { type: "checkbox", id: "cosine-normalize" });
  normalizeBox.addEventListener("change", () => {
    state.normalize = normalizeBox.checked;
    render();
  });

  function setPolar(length, degrees) {
    const radians = (degrees * Math.PI) / 180;
    state.q = [length * Math.cos(radians), length * Math.sin(radians)];
    render();
  }

  function setControl(control, value, digits) {
    control.querySelector("input").value = value;
    control.querySelector("output").textContent = fmt(value, digits);
  }

  // Pointer position -> vector, kept inside the first quadrant and the plane.
  function drag(event) {
    const box = svg.getBoundingClientRect();
    const x = (((event.clientX - box.left) / box.width) * WIDTH - ORIGIN[0]) / SCALE;
    const y = (ORIGIN[1] - ((event.clientY - box.top) / box.height) * HEIGHT) / SCALE;
    let q = [Math.max(0, x), Math.max(0, y)];
    const length = norm(q);
    if (length < 0.1) q = length === 0 ? [0.1, 0] : [(q[0] / length) * 0.1, (q[1] / length) * 0.1];
    if (length > MAX_LENGTH) q = [(q[0] / length) * MAX_LENGTH, (q[1] / length) * MAX_LENGTH];
    state.q = q;
    setControl(angleControl, (Math.atan2(q[1], q[0]) * 180) / Math.PI, 0);
    setControl(lengthControl, norm(q), 3);
    render();
  }
  let dragging = false;
  svg.addEventListener("pointerdown", (event) => {
    dragging = true;
    svg.setPointerCapture(event.pointerId);
    drag(event);
  });
  svg.addEventListener("pointermove", (event) => {
    if (dragging) drag(event);
  });
  for (const type of ["pointerup", "pointercancel"]) {
    svg.addEventListener(type, () => {
      dragging = false;
    });
  }

  function render() {
    const q = state.normalize ? unit(state.q) : state.q;
    const rows = chunks.map((chunk, i) => {
      const v = state.normalize ? unit(chunk.v) : chunk.v;
      const [x, y] = px(v);
      const mark = chunkMarks[i];
      mark.line.setAttribute("x2", x);
      mark.line.setAttribute("y2", y);
      mark.tip.setAttribute("cx", x);
      mark.tip.setAttribute("cy", y);
      mark.label.setAttribute("x", x + 7);
      mark.label.setAttribute("y", y - 6);
      return { chunk, cosine: cosine(q, v), dot: dot(q, v) };
    });
    const [qx, qy] = px(q);
    queryLine.setAttribute("x2", qx);
    queryLine.setAttribute("y2", qy);
    queryHandle.setAttribute("cx", qx);
    queryHandle.setAttribute("cy", qy);
    queryLabel.setAttribute("x", qx - 12);
    queryLabel.setAttribute("y", qy - 10);

    table.replaceChildren(
      h("tr", {}, [h("th", {}, "Chunk"), h("th", {}, "Cosine"), h("th", {}, "Dot product")]),
      ...rows.map((row) =>
        h("tr", {}, [
          h("td", {}, [
            h("span", { style: `display:inline-block;width:0.7em;height:0.7em;margin-right:0.5em;border-radius:2px;background:${row.chunk.colour}` }),
            `${row.chunk.id}: ${row.chunk.text}`,
          ]),
          h("td", {}, fmt(row.cosine, 3)),
          h("td", {}, fmt(row.dot, 3)),
        ])
      )
    );
    const order = (key) => [...rows].sort((a, b) => b[key] - a[key]).map((row) => row.chunk.id).join(" > ");
    const byCosine = order("cosine");
    const byDot = order("dot");
    ranks.replaceChildren(
      h("strong", {}, "Ranking by cosine: "), byCosine, h("br"),
      h("strong", {}, "Ranking by dot product: "), byDot,
      byCosine === byDot ? " (the same)" : " (different)"
    );
    note.textContent =
      `Query vector [${fmt(q[0], 3)}, ${fmt(q[1], 3)}], length ${fmt(norm(q), 3)}. ` +
      (state.normalize
        ? "Each vector has length 1, so the dot product is equal to the cosine."
        : "A longer query changes each dot product, but no cosine. A long chunk vector (c3) gets a high dot product.");
  }

  el.append(
    h("p", { class: "llm-widget__title" }, 'Move the query. Start: "the cat sat" = [0.667, 0.667]'),
    h("div", { class: "llm-widget__controls" }, [
      angleControl,
      lengthControl,
      h("label", { for: "cosine-normalize", style: "flex:1 1 100%" }, [normalizeBox, " Normalize each vector to length 1"]),
    ]),
    h("div", { style: "display:flex;flex-wrap:wrap;gap:0.8em 1.6em;align-items:flex-start" }, [
      svg,
      h("div", { style: "flex:1 1 240px;min-width:0" }, [table, ranks, note]),
    ])
  );
  render();
});
