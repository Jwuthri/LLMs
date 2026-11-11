// Forward pass, loss and backpropagation on the toy network of the neural networks page:
// x = [1, 1] -> z = W x + b -> h = activation(z) -> logits = U h -> softmax -> loss for "sat".
// Keep in sync with examples/fundamentals/neural_networks.py.
LLMWidgets.register("forward-backward", (el) => {
  const { h, fmt, softmax, toy } = LLMWidgets;
  const x = [1, 1];
  const target = 2; // "sat"
  const rate = 0.1;
  const initial = () => ({
    W: toy.W.map((row) => row.slice()),
    b: [0.2, -0.1],
    U: [
      [0.4, 0.1],
      [-0.3, 1.0],
      [0.2, 2.0],
    ],
  });
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const activations = {
    relu: { name: "ReLU", f: (z) => Math.max(z, 0), slope: (z) => (z > 0 ? 1 : 0) },
    sigmoid: { name: "sigmoid", f: sigmoid, slope: (z) => sigmoid(z) * (1 - sigmoid(z)) },
    tanh: { name: "tanh", f: Math.tanh, slope: (z) => 1 - Math.tanh(z) ** 2 },
  };

  let weights = initial();
  let activation = "relu";
  let lossBeforeStep = null;

  // Forward pass, then the gradient of the loss for each weight (backpropagation).
  function compute() {
    const act = activations[activation];
    const z = weights.W.map((row, i) => row[0] * x[0] + row[1] * x[1] + weights.b[i]);
    const hidden = z.map(act.f);
    const logits = weights.U.map((row) => row[0] * hidden[0] + row[1] * hidden[1]);
    const probs = softmax(logits);
    const dLogits = probs.map((p, k) => p - (k === target ? 1 : 0));
    const dHidden = [0, 1].map((i) => dLogits.reduce((sum, d, k) => sum + d * weights.U[k][i], 0));
    const dZ = dHidden.map((d, i) => d * act.slope(z[i]));
    return {
      z, hidden, logits, probs, dLogits,
      loss: -Math.log(probs[target]),
      grads: { W: dZ.map((d) => x.map((v) => d * v)), b: dZ, U: dLogits.map((d) => hidden.map((v) => d * v)) },
    };
  }

  const show = (value) => fmt(value, 3).replace(/^-0\.000$/, "0.000");
  const list = (values) => `[${values.map(show).join(", ")}]`;

  // One editable weight: a number input with its gradient below.
  const tight = "padding:0.3em 0.4em"; // narrow cells, so that a table fits a phone screen
  const cells = [];
  function cell(label, read, write, gradient) {
    const input = h("input", { type: "number", step: "0.1", "aria-label": label, style: "width:4.4em;padding:0.3em 0.35em" });
    const note = h("div", { style: "color:var(--llm-ink-2);font-size:0.85em" });
    input.addEventListener("input", () => {
      const value = Number(input.value);
      if (input.value === "" || !Number.isFinite(value)) return;
      write(value);
      lossBeforeStep = null;
      render();
    });
    cells.push({ input, note, read, gradient });
    return h("td", { style: tight }, [input, note]);
  }

  // Write the weights into the inputs (after a step or a reset).
  function fillInputs() {
    for (const c of cells) c.input.value = String(Number(c.read().toFixed(3)));
  }

  // A class on the table turns off the page theme for Markdown tables: the cells stay compact.
  const hiddenTable = h("table", { class: "llm-widget__table" }, [
    h("tr", {}, ["Unit", "From x1", "From x2", "Bias b"].map((name) => h("th", { style: tight }, name))),
    ...[0, 1].map((i) =>
      h("tr", {}, [
        h("td", { style: tight }, `h${i + 1}`),
        ...[0, 1].map((j) =>
          cell(`W row ${i + 1} column ${j + 1}`, () => weights.W[i][j], (v) => { weights.W[i][j] = v; }, (g) => g.W[i][j])
        ),
        cell(`b ${i + 1}`, () => weights.b[i], (v) => { weights.b[i] = v; }, (g) => g.b[i]),
      ])
    ),
  ]);
  const outputTable = h("table", { class: "llm-widget__table" }, [
    h("tr", {}, ["Logit", "From h1", "From h2"].map((name) => h("th", { style: tight }, name))),
    ...toy.vocab.map((token, k) =>
      h("tr", {}, [
        h("td", { style: tight }, token),
        ...[0, 1].map((i) =>
          cell(`U row ${k + 1} column ${i + 1}`, () => weights.U[k][i], (v) => { weights.U[k][i] = v; }, (g) => g.U[k][i])
        ),
      ])
    ),
  ]);

  // Network picture: line width = size of the weight, colour = sign of the weight.
  const svg = h("svg", {
    viewBox: "0 0 480 200",
    role: "img",
    "aria-label": "Network with 2 inputs, 2 hidden units and 3 outputs. The numbers in the circles are the forward values.",
    style: "display:block;width:100%;max-width:520px;height:auto;margin:0 auto",
  });
  const nodes = {
    input: [[60, 70], [60, 150]],
    hidden: [[235, 70], [235, 150]],
    output: [[405, 50], [405, 110], [405, 170]],
  };

  function edge(from, to, weight) {
    return h("line", {
      x1: from[0], y1: from[1], x2: to[0], y2: to[1],
      style: `stroke:var(${weight >= 0 ? "--llm-s1" : "--llm-s2"});opacity:0.8;` +
        `stroke-width:${(0.6 + 2 * Math.min(Math.abs(weight), 3)).toFixed(2)}`,
    });
  }

  function node([cx, cy], text) {
    return [
      h("circle", { cx, cy, r: 21, style: "fill:var(--llm-panel);stroke:var(--llm-axis);stroke-width:1.5" }),
      h("text", { x: cx, y: cy + 4, "text-anchor": "middle", style: "fill:var(--llm-ink)" }, text),
    ];
  }

  function drawNetwork(result) {
    const parts = [
      h("text", { x: 60, y: 14, "text-anchor": "middle" }, "input x"),
      h("text", { x: 235, y: 14, "text-anchor": "middle" }, `hidden h (${activations[activation].name})`),
      h("text", { x: 405, y: 14, "text-anchor": "middle" }, "probability p"),
    ];
    weights.W.forEach((row, i) => row.forEach((w, j) => parts.push(edge(nodes.input[j], nodes.hidden[i], w))));
    weights.U.forEach((row, k) => row.forEach((w, i) => parts.push(edge(nodes.hidden[i], nodes.output[k], w))));
    x.forEach((value, j) => parts.push(...node(nodes.input[j], fmt(value, 0))));
    result.hidden.forEach((value, i) => parts.push(...node(nodes.hidden[i], show(value))));
    result.probs.forEach((value, k) => {
      parts.push(...node(nodes.output[k], show(value)));
      parts.push(h("text", { x: 432, y: nodes.output[k][1] + 4 }, toy.vocab[k]));
    });
    svg.replaceChildren(...parts);
  }

  const values = h("div", {
    style: "font-family:'JetBrains Mono',ui-monospace,monospace;line-height:1.7;overflow-x:auto;white-space:nowrap",
  });
  const note = h("p", { class: "llm-widget__note" });

  function render() {
    const result = compute();
    drawNetwork(result);
    for (const c of cells) c.note.textContent = `gradient ${show(c.gradient(result.grads))}`;
    values.replaceChildren(
      ...[
        `z = W x + b = ${list(result.z)}`,
        `h = ${activations[activation].name}(z) = ${list(result.hidden)}`,
        `logits = U h = ${list(result.logits)}`,
        `p = softmax(logits) = ${list(result.probs)}`,
        `loss = -ln(p of "sat") = ${show(result.loss)}`,
        `gradient for the logits = p - y = ${list(result.dLogits)}`,
      ].map((line) => h("div", {}, line))
    );
    note.textContent = lossBeforeStep === null
      ? "Blue line: positive weight. Orange line: negative weight. A thicker line is a larger weight. " +
        "A gradient of 0 means that the step does not change this weight."
      : `The step changed the loss from ${show(lossBeforeStep)} to ${show(result.loss)}.`;
  }

  function step() {
    const result = compute();
    lossBeforeStep = result.loss;
    for (const name of ["W", "U"]) {
      weights[name] = weights[name].map((row, i) => row.map((w, j) => w - rate * result.grads[name][i][j]));
    }
    weights.b = weights.b.map((b, i) => b - rate * result.grads.b[i]);
    fillInputs();
    render();
  }

  function reset() {
    weights = initial();
    lossBeforeStep = null;
    fillInputs();
    render();
  }

  const select = h("select", { id: "forward-backward-activation" },
    Object.entries(activations).map(([key, act]) => h("option", { value: key }, act.name))
  );
  select.addEventListener("change", () => {
    activation = select.value;
    lossBeforeStep = null;
    render();
  });

  el.append(
    h("p", { class: "llm-widget__title" }, 'A network with 12 weights. Input [1, 1], correct token "sat"'),
    h("div", { class: "llm-widget__controls" }, [
      h("label", { for: "forward-backward-activation" }, ["Activation ", select]),
    ]),
    svg,
    h("p", { class: "llm-widget__note" }, "Each box is one weight. You can change it. Below it is the gradient of the loss for this weight."),
    h("div", { style: "display:flex;flex-wrap:wrap;align-items:flex-start;gap:0.8em 1.6em;margin:0.8em 0;overflow-x:auto" }, [hiddenTable, outputTable]),
    values,
    h("div", { style: "display:flex;flex-wrap:wrap;gap:0.6em;margin-top:0.8em" }, [
      h("button", { onclick: step }, `Apply one gradient step (learning rate ${rate})`),
      h("button", { class: "secondary", onclick: reset }, "Reset"),
    ]),
    note
  );
  fillInputs();
  render();
});
