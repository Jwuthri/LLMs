// Learning-rate schedule: linear warm-up, then a cosine curve down to 0.
// Same formula as lr_schedule() in examples/scientist/pre_training.py.
LLMWidgets.register("lr-schedule", (el) => {
  const { h, slider } = LLMWidgets;
  const state = { peak: 3e-4, warmup: 100, total: 1000, step: 550 };
  const box = { width: 640, height: 240, left: 62, right: 16, top: 16, bottom: 40 };

  function lr(step) {
    const warmup = Math.min(state.warmup, state.total);
    if (step < warmup) return (state.peak * step) / warmup;
    if (state.total === warmup) return state.peak;
    const progress = (step - warmup) / (state.total - warmup);
    return 0.5 * state.peak * (1 + Math.cos(Math.PI * progress));
  }

  const x = (step) => box.left + (step / state.total) * (box.width - box.left - box.right);
  const y = (value) => box.height - box.bottom - (value / state.peak) * (box.height - box.top - box.bottom);
  const sci = (value) => (value === 0 ? "0" : value.toExponential(2));
  const line = (x1, y1, x2, y2, style) => h("line", { x1, y1, x2, y2, style });

  const chart = h("div");
  const readout = h("p", { class: "llm-widget__note" });

  function render() {
    const warmup = Math.min(state.warmup, state.total);
    // 201 evenly spaced steps, plus the end of the warm-up so that the peak is exact.
    const steps = Array.from({ length: 201 }, (_, i) => (i / 200) * state.total).concat(warmup).sort((a, b) => a - b);
    const points = steps.map((step) => `${x(step).toFixed(1)},${y(lr(step)).toFixed(1)}`);
    const axis = "stroke:var(--llm-axis);stroke-width:1";
    const grid = "stroke:var(--llm-grid);stroke-width:1";
    const bottom = box.height - box.bottom;
    const now = lr(state.step);

    chart.replaceChildren(
      h("svg", { viewBox: `0 0 ${box.width} ${box.height}`, role: "img", style: "display:block;width:100%;height:auto",
        "aria-label": "Learning rate against training step" }, [
        line(box.left, y(state.peak), box.width - box.right, y(state.peak), grid),
        line(box.left, y(state.peak / 2), box.width - box.right, y(state.peak / 2), grid),
        line(box.left, box.top, box.left, bottom, axis),
        line(box.left, bottom, box.width - box.right, bottom, axis),
        line(x(warmup), box.top, x(warmup), bottom, "stroke:var(--llm-s2);stroke-width:1;stroke-dasharray:4 4"),
        h("polyline", { points: points.join(" "), style: "fill:none;stroke:var(--llm-s1);stroke-width:2" }),
        line(x(state.step), y(now), x(state.step), bottom, "stroke:var(--llm-muted);stroke-width:1"),
        h("circle", { cx: x(state.step), cy: y(now), r: 5, style: "fill:var(--llm-s1);stroke:var(--llm-surface);stroke-width:2" }),
        h("text", { x: box.left - 6, y: y(state.peak) + 4, "text-anchor": "end" }, sci(state.peak)),
        h("text", { x: box.left - 6, y: y(state.peak / 2) + 4, "text-anchor": "end" }, sci(state.peak / 2)),
        h("text", { x: box.left - 6, y: bottom + 4, "text-anchor": "end" }, "0"),
        h("text", { x: box.left, y: bottom + 16, "text-anchor": "middle" }, "0"),
        h("text", { x: box.width - box.right, y: bottom + 16, "text-anchor": "end" }, String(state.total)),
        h("text", { x: (box.left + box.width - box.right) / 2, y: box.height - 6, "text-anchor": "middle" }, "Training step"),
        h("text", { x: Math.min(x(warmup) + 6, box.width - 120), y: bottom - 8 }, `end of warm-up: ${warmup}`),
      ])
    );
    const phase = state.step < warmup ? "warm-up" : "cosine";
    readout.textContent =
      `Learning rate at step ${state.step}: ${sci(now)} (${phase} phase). Peak ${sci(state.peak)} at step ${warmup}.`;
  }

  const stepControl = slider({
    label: "Step to read", min: 0, max: state.total, step: 5, value: state.step, digits: 0,
    onInput: (value) => { state.step = value; render(); },
  });
  const stepInput = stepControl.querySelector("input");
  const stepOutput = stepControl.querySelector("output");

  function change(key) {
    return (value) => {
      state[key] = value;
      stepInput.max = state.total; // the step slider always covers the full run
      state.step = Math.min(state.step, state.total);
      stepInput.value = state.step;
      stepOutput.textContent = String(state.step);
      render();
    };
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Learning rate: linear warm-up, then cosine decay"),
    h("div", { class: "llm-widget__controls" }, [
      slider({ label: "Peak learning rate", min: 0.0001, max: 0.001, step: 0.0001, value: state.peak, digits: 4, onInput: change("peak") }),
      slider({ label: "Warm-up steps", min: 0, max: 500, step: 10, value: state.warmup, digits: 0, onInput: change("warmup") }),
      slider({ label: "Total steps", min: 500, max: 5000, step: 100, value: state.total, digits: 0, onInput: change("total") }),
      stepControl,
    ]),
    chart,
    readout
  );
  render();
});
