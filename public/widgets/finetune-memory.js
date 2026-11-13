// Estimate of the GPU memory to fine-tune a model: full fine-tuning, LoRA or QLoRA.
// Same assumptions as memory_parts() in examples/scientist/training_libraries.py. An estimate, not a measurement.
LLMWidgets.register("finetune-memory", (el) => {
  const { h, slider, bars } = LLMWidgets;
  const state = { billions: 8, method: "qlora", gpu: 24 };

  const USABLE_SHARE = 0.9; // keep 10% of the GPU memory free
  const FOUR_BIT_GB = 5.5 / 8; // GB for each billion parameters: 4 bits + 1.5 bits of overhead
  const SIXTEEN_BIT_GB = 2; // GB for each billion parameters
  const STATE_BYTES = 16; // one trained parameter: weight + gradient + optimizer states
  const LORA_SHARE = 41943040 / 8e9; // trained share with rank 16 on all linear layers
  const ACTIVATION_GB = 0.5; // GB for each billion parameters

  // Returns the three parts of the estimate in GB.
  function parts() {
    const b = state.billions;
    const activations = ACTIVATION_GB * b;
    if (state.method === "full") {
      return { weights: SIXTEEN_BIT_GB * b, states: (STATE_BYTES - 2) * b, activations };
    }
    const base = (state.method === "lora" ? SIXTEEN_BIT_GB : FOUR_BIT_GB) * b;
    return { weights: base, states: LORA_SHARE * b * STATE_BYTES, activations };
  }

  const gb = (value) => `${value.toFixed(1)} GB`;
  const chart = h("div");
  const verdict = h("p", { class: "llm-widget__note" });
  const note = h("p", { class: "llm-widget__note" });

  const select = h("select", { id: "finetune-memory-method", "aria-label": "Method" }, [
    h("option", { value: "full" }, "Full fine-tuning"),
    h("option", { value: "lora" }, "LoRA, 16-bit base"),
    h("option", { value: "qlora", selected: "selected" }, "QLoRA, 4-bit base"),
  ]);
  select.addEventListener("change", () => {
    state.method = select.value;
    render();
  });

  function render() {
    const p = parts();
    const total = p.weights + p.states + p.activations;
    const limit = USABLE_SHARE * state.gpu;
    const ok = total <= limit;
    chart.replaceChildren(
      bars(
        [
          { label: "Weights", value: p.weights, text: gb(p.weights) },
          { label: "Training states", value: p.states, text: gb(p.states) },
          { label: "Activations", value: p.activations, text: gb(p.activations) },
          { label: "Total", value: total, text: gb(total) },
          { label: "90% of the GPU", value: limit, text: gb(limit), off: true },
        ],
        Math.max(total, limit)
      )
    );
    verdict.replaceChildren(
      h("strong", { style: `color: var(${ok ? "--llm-s3" : "--llm-s1"})` }, ok ? "Fits." : "Does not fit."),
      ok
        ? ` The estimate is ${gb(total)}. ${gb(limit - total)} stay free below the limit.`
        : ` The estimate is ${gb(total)}. That is ${gb(total - limit)} above the limit.`
    );
    note.textContent =
      "Training states: the gradients and the optimizer states. Assumptions: 16 bytes for each trained parameter, 5.5 bits for each parameter of a 4-bit base, " +
      "LoRA with rank 16 on all linear layers (0.524% of the parameters), " +
      "0.5 GB of activations for each billion parameters. A longer text or a larger batch needs more.";
  }

  function change(key) {
    return (value) => {
      state[key] = value;
      render();
    };
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Does the fine-tuning job fit in the GPU memory?"),
    h("div", { class: "llm-widget__controls" }, [
      slider({ label: "Model size (billion parameters)", min: 1, max: 70, step: 1, value: state.billions, digits: 0, onInput: change("billions") }),
      h("div", { class: "llm-widget__control" }, [h("label", { for: "finetune-memory-method" }, "Method"), select]),
      slider({ label: "GPU memory (GB)", min: 8, max: 80, step: 4, value: state.gpu, digits: 0, onInput: change("gpu") }),
    ]),
    chart,
    verdict,
    note
  );
  render();
});
