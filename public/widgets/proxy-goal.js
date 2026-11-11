// A reward on visible tests against the goal (hidden tests), for an honest policy and a policy
// that special-cases the visible tests. Starts with the numbers of the Alignment lesson.
LLMWidgets.register("proxy-goal", (el) => {
  const { h, fmt, slider, bars } = LLMWidgets;
  const TESTS = 4;
  const state = { passed: 3, audit: 0, hidden: false };

  const chart = h("div");
  const note = h("p", { class: "llm-widget__note" });
  const box = h("input", { type: "checkbox", id: "proxy-goal-hidden" });
  box.addEventListener("change", () => {
    state.hidden = box.checked;
    render();
  });

  function render() {
    const goal = { honest: state.passed / TESTS, cheat: 0 };
    const reward = state.hidden
      ? { honest: goal.honest, cheat: goal.cheat }
      : { honest: state.passed / TESTS, cheat: 1 - state.audit };
    const kept = reward.cheat > reward.honest ? "cheat" : reward.honest > reward.cheat ? "honest" : null;
    chart.replaceChildren(
      bars([
        { label: "honest: reward", value: reward.honest, text: fmt(reward.honest, 2), off: kept === "cheat" },
        { label: "cheat: reward", value: reward.cheat, text: fmt(reward.cheat, 2), off: kept === "honest" },
        { label: "honest: goal", value: goal.honest, text: fmt(goal.honest, 2), off: kept === "cheat" },
        { label: "cheat: goal", value: goal.cheat, text: fmt(goal.cheat, 2), off: kept === "honest" },
      ])
    );
    const source = state.hidden
      ? "The reward is the share of hidden tests that pass."
      : `The reward is the share of visible tests that pass. An audit finds the special cases with a probability of ${fmt(state.audit, 2)}.`;
    const result = kept === null
      ? "The two rewards are equal. The optimiser has no preference."
      : `The optimiser keeps the ${kept} policy: reward ${fmt(reward[kept], 2)} against ` +
        `${fmt(reward[kept === "cheat" ? "honest" : "cheat"], 2)}. Its score on the goal is ${fmt(goal[kept], 2)}.`;
    note.textContent = `${source} ${result}`;
  }

  el.append(
    h("p", { class: "llm-widget__title" }, "The reward on visible tests against the goal"),
    h("div", { class: "llm-widget__controls" }, [
      slider({
        label: `Tests that the honest code passes (of ${TESTS})`, min: 0, max: TESTS, step: 1, value: state.passed, digits: 0,
        onInput: (value) => { state.passed = value; render(); },
      }),
      slider({
        label: "Audit probability", min: 0, max: 1, step: 0.05, value: state.audit,
        onInput: (value) => { state.audit = value; render(); },
      }),
    ]),
    h("div", { class: "llm-widget__controls" }, [
      h("label", { for: "proxy-goal-hidden" }, [box, " Score on the hidden tests (the fix)"]),
    ]),
    chart,
    note
  );
  render();
});
