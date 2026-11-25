// Idle slot time of a synchronous batch vs asynchronous rollouts (content/rl/07-agentic-rl.md, part 3).
// Rollout lengths are the (i + 0.5) / G quantiles of a log-normal with median 250 tokens, as in the script.
LLMWidgets.register("async-rollouts", (el) => {
  const { h, fmt, slider, bars } = LLMWidgets;
  const MEDIAN = 250;
  const state = { group: 8, spread: 0.6 };

  // Inverse of the standard normal CDF (Wichura AS241, the algorithm of Python's NormalDist.inv_cdf).
  function invCdf(p) {
    const q = p - 0.5;
    if (Math.abs(q) <= 0.425) {
      const r = 0.180625 - q * q;
      const num = (((((((2509.0809287301226727 * r + 33430.575583588128105) * r + 67265.770927008700853) * r
        + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r
        + 133.14166789178437745) * r + 3.387132872796366608);
      const den = (((((((5226.495278852545925 * r + 28729.085735721942674) * r + 39307.89580009271061) * r
        + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r
        + 42.313330701600911252) * r + 1.0);
      return (q * num) / den;
    }
    let r = q <= 0 ? p : 1 - p;
    r = Math.sqrt(-Math.log(r));
    let x;
    if (r <= 5) {
      r -= 1.6;
      x = (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r + 0.24178072517745061177) * r
        + 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r
        + 4.6303378461565452959) * r + 1.42343711074968357734)
        / (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r
        + 0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r
        + 2.05319162663775882187) * r + 1.0);
    } else {
      r -= 5;
      x = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r
        + 0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r
        + 5.4637849111641143699) * r + 6.6579046435011037772)
        / (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r
        + 7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r
        + 0.59983220655588793769) * r + 1.0);
    }
    return q < 0 ? -x : x;
  }

  function lengths(group, spread) {
    return Array.from({ length: group }, (_, i) => Math.round(MEDIAN * Math.exp(spread * invCdf((i + 0.5) / group))));
  }

  const chart = h("div");
  const summary = h("div");
  const note = h("p", { class: "llm-widget__note" });

  function render() {
    const ls = lengths(state.group, state.spread);
    const max = Math.max(...ls);
    const mean = ls.reduce((a, b) => a + b, 0) / ls.length;
    const idle = 1 - mean / max;
    const shown = ls.length <= 16 ? ls : [];
    chart.replaceChildren(
      shown.length
        ? bars(shown.map((n, i) => ({ label: `rollout ${i + 1}`, value: n / max, text: `${n} tok` })))
        : h("p", { class: "llm-widget__note" }, `${ls.length} rollouts: ${ls[0]} to ${max} tokens.`)
    );
    summary.replaceChildren(
      bars([
        { label: "sync idle", value: idle, text: `${fmt(idle * 100, 1)}%` },
        { label: "async idle", value: 0, text: "≈ 0%", off: true },
      ])
    );
    note.textContent =
      `Synchronous: every slot waits for the longest rollout (${max} tokens, mean ${fmt(mean, 1)}). ` +
      `Asynchronous: a slot starts a new rollout as soon as one ends, so generation is ${fmt(max / mean, 2)}× faster, ` +
      `but the longest rollout is ${Math.floor(max / mean)} policy versions old when the trainer uses it.`;
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "Rollouts of one batch: median 250 tokens"),
    h("div", { class: "llm-widget__controls" }, [
      slider({ label: "Group size G", min: 1, max: 128, step: 1, value: 8, digits: 0,
        onInput: (v) => { state.group = v; render(); } }),
      slider({ label: "Length spread", min: 0, max: 1.2, step: 0.1, value: 0.6, digits: 1,
        onInput: (v) => { state.spread = v; render(); } }),
    ]),
    chart,
    h("div", { style: "margin-top:0.9em" }, [summary]),
    note
  );
  render();
});
