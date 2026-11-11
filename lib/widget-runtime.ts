// Runtime for the interactive widgets in public/widgets/.
// A page places `<div class="llm-widget" data-widget="NAME"></div>`.
// A widget file calls `LLMWidgets.register("NAME", (el) => { ...build the widget in el... })`.

type Build = (el: HTMLElement) => void;
type Child = Node | string | number;

const SVG_TAGS = ["svg", "g", "rect", "circle", "line", "path", "text", "polyline", "polygon"];

function createRuntime() {
  const registry: Record<string, Build> = {};

  function mount() {
    document.querySelectorAll<HTMLElement>("[data-widget]").forEach((el) => {
      const build = registry[el.dataset.widget ?? ""];
      if (build && !el.dataset.mounted) {
        el.dataset.mounted = "1";
        build(el);
      }
    });
  }

  function register(name: string, build: Build) {
    registry[name] = build;
    mount();
  }

  // Create an element: h("div", { class: "x", onclick: fn }, [child, "text"]).
  function h(tag: string, attrs: Record<string, unknown> = {}, children: Child | Child[] = []) {
    const el = SVG_TAGS.includes(tag)
      ? document.createElementNS("http://www.w3.org/2000/svg", tag)
      : document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (key.startsWith("on")) el.addEventListener(key.slice(2), value as EventListener);
      else el.setAttribute(key, String(value));
    }
    for (const child of ([] as Child[]).concat(children)) {
      el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return el;
  }

  // Fixed-decimal number as text.
  function fmt(x: number | string, digits = 3) {
    return Number(x).toFixed(digits);
  }

  // Labelled range slider. `onInput(value)` runs on each change. Returns the control element.
  function slider(options: {
    label: string;
    min: number;
    max: number;
    step: number;
    value: number;
    digits?: number;
    onInput: (value: number) => void;
  }) {
    const { label, min, max, step, value, digits = 2, onInput } = options;
    const out = h("output", {}, fmt(value, digits));
    const input = h("input", { type: "range", min, max, step, value, "aria-label": label }) as HTMLInputElement;
    input.addEventListener("input", () => {
      out.textContent = fmt(input.value, digits);
      onInput(Number(input.value));
    });
    return h("div", { class: "llm-widget__control" }, [h("label", {}, [label, out]), input]);
  }

  // Horizontal bars. rows = [{ label, value, text, off }], value in 0..max.
  function bars(rows: { label: string; value: number; text: string; off?: boolean }[], max = 1) {
    const root = h("div", { class: "llm-bars" });
    for (const row of rows) {
      const fill = h("div", {
        class: "llm-bars__fill" + (row.off ? " is-off" : ""),
        style: `width:${Math.max(0, (row.value / max) * 100)}%`,
      });
      root.append(
        h("div", { class: "llm-bars__label" }, row.label),
        h("div", { class: "llm-bars__track" }, [fill]),
        h("div", { class: "llm-bars__value" }, row.text)
      );
    }
    return root;
  }

  function softmax(logits: number[], temperature = 1) {
    const scaled = logits.map((z) => z / temperature);
    const top = Math.max(...scaled);
    const exps = scaled.map((z) => Math.exp(z - top));
    const total = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / total);
  }

  // The shared toy example (content/toy-example.md). Keep in sync with examples/toy.py.
  const toy = {
    vocab: ["the", "cat", "sat"],
    embeddings: [
      [1, 0],
      [0, 1],
      [1, 1],
    ],
    logits: [0.1, 1.0, 2.0],
    W: [
      [0.5, -1.2],
      [0.8, 0.3],
    ],
  };

  return { register, mount, h, fmt, slider, bars, softmax, toy };
}

export type WidgetRuntime = ReturnType<typeof createRuntime>;

declare global {
  interface Window {
    LLMWidgets?: WidgetRuntime;
  }
}

const loaded = new Set<string>();

// Load the script of each widget on the page (once), then build the widgets.
export function mountWidgets(names: string[]) {
  const runtime = (window.LLMWidgets ??= createRuntime());
  for (const name of names) {
    if (loaded.has(name)) continue;
    loaded.add(name);
    const script = document.createElement("script");
    script.src = `/widgets/${name}.js`;
    script.async = true;
    document.head.append(script);
  }
  runtime.mount();
}
