# How to write a page

Each page explains one section of the [LLM Course](https://github.com/mlabonne/llm-course).
The reference page is `docs/scientist/01-architecture.md`. Read it first and copy its structure.

## The method (8 rules)

1. **Core idea**: 2-3 sentences. No jargon.
2. For each sub-part, use the structure **Setup → Steps → Key point**. The sub-parts are the bullets of the
   course section (usually 4).
3. Use **one tiny concrete example with real numbers**. Show every calculation step. Use the shared toy
   example (`docs/toy-example.md`, `examples/toy.py`) where it fits. If it does not fit, define one small
   example for the page and reuse it in all sub-parts.
4. After each step, say **why** it is done, not only what is done.
5. Go from concrete to abstract: the toy case first, then how it scales to the real system.
6. End with a **comparison table** and **2-3 practical checks or common misconceptions**.
7. Style: Simplified Technical English (ASD-STE100). Short sentences, one idea each, active voice, common
   words. Define a term the first time you use it (bold it at the definition).
8. Keep the technical depth. Include the formula, and a minimal code snippet if it helps.

## Page template

```markdown
# N. Title of the course section

## Core idea
(2-3 sentences)

## The example
(what small example this page uses; a table of its values)

## 1. First sub-part
### Setup
### Steps
1. **Do this.** The calculation, with the numbers.

    **Why:** the reason.
    { .why }

!!! success "Key point"
    One or two sentences.

### From toy to real
(real sizes, real names, real tools; cite the source of each real number)

## 2. Second sub-part
(same structure) ... ## 3. ... ## 4. ...

## Minimal code
## Comparison
## Checks and common misconceptions
## Sources
```

Length: 1,800 to 2,600 words for each page.

## Rules for numbers and facts

- Each number of the worked example comes from a script: `examples/<track>/<name>.py` (NumPy and the standard
  library only). Run the script and copy its output. Do not calculate in your head.
- Each of those numbers has an assert in `tests/test_<track>_<name>.py`. Round to 3 decimals on the page.
- A number about a real system (model size, token count, price, speed) needs a source link in **Sources**.
  If you are not sure of a number, do not write it.
- When a value is an assumption for the example, say "assume".
- Code that the tests do not run (for example, a call to a real library) goes under a line
  "Real-library sketch (the tests do not run this code)". Keep it short. Never write a string that looks like
  an API key; read keys from the environment.
- The **Sources** section starts with the credit line of the reference page, then lists the reference links
  of that course section.

## Formulas, figures, widgets

- **Formulas**: KaTeX. Inline `$...$`, display `$$...$$` on its own paragraph. Inside a list item, indent 4
  spaces.
- **Diagrams**: a hand-written SVG file in `docs/assets/figures/<page>/<name>.svg`. No XML header, no blank
  lines, `viewBox="0 0 720 H"`, an `aria-label`, unique marker ids. Use only the CSS classes `dgm-*` from
  `docs/stylesheets/extra.css` (no fixed colours), so the diagram works in light and dark mode. Keep all text
  inside the viewBox. Include it with:

    ```markdown
    <figure class="diagram">
    --8<-- "docs/assets/figures/<page>/<name>.svg"
    <figcaption>One sentence.</figcaption>
    </figure>
    ```

- **Plots**: draw them in the example script with `examples.plotstyle.save_light_dark(draw, page, name)`.
  It writes `<name>.svg` and `<name>-dark.svg`. One y-axis only. Use `theme["series"]` colours in order.
  Label the axes. Include both files:

    ```markdown
    <figure class="plot" markdown>
    ![Alt text with the main numbers.](../assets/figures/<page>/<name>.svg#only-light)
    ![Alt text with the main numbers.](../assets/figures/<page>/<name>-dark.svg#only-dark)
    <figcaption>One sentence that says what to see.</figcaption>
    </figure>
    ```

- **Widgets**: one file `docs/javascripts/widgets/<name>.js` that calls
  `LLMWidgets.register("<name>", (el) => { ... })`. Use the helpers of `docs/javascripts/widgets.js`
  (`h`, `fmt`, `slider`, `bars`, `softmax`, `toy`) and the `llm-widget*` / `llm-bars*` classes. Use the CSS
  variables `--llm-*` for colours. The widget must start with the numbers of the page. Place it with
  `<div class="llm-widget" data-widget="<name>"></div>`. The hook `hooks/widgets.py` loads each widget file.

## Checks before a page is done

```bash
uv run python -m examples.<track>.<name>        # prints the numbers, writes the plots
uv run pytest -q                                # all asserted numbers pass
uv run mkdocs build --strict                    # no broken link, no warning
```

Then open the page with `uv run mkdocs serve` and check: formulas render, figures show in light and dark
mode, the widget works, no horizontal scroll at phone width.
