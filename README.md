# LLM Course, Explained

A learning website that explains each section of the [LLM Course](https://github.com/mlabonne/llm-course) by
Maxime Labonne (Apache-2.0) with one tiny example, real numbers, formulas, figures and small interactive widgets.
Three parts are additions to the course: a track on AI safety (6 lessons and a glossary), and lessons on
diffusion models, training libraries, metrics and losses, and newer quantization methods. A fifth track, LLM Datasets (4 lessons and a
catalogue), follows the list [LLM Datasets](https://github.com/mlabonne/llm-datasets) of the same author.

## Run

```bash
npm ci                 # Next.js site
npm run dev            # http://localhost:3210
npm run build          # static site in out/
npm run preview        # serves out/ at http://127.0.0.1:3211

uv sync --frozen       # Python, for the example scripts
uv run pytest -q       # checks each number shown on the pages
```

## Layout

| Path | Content |
|---|---|
| `content/` | The 30 lessons, the toy-example page, the AI safety glossary and the notebook guide (Markdown) |
| `app/`, `components/`, `lib/` | The Next.js site. `lib/markdown.ts` turns a lesson into HTML at build time |
| `app/globals.css` | The theme (cream, white, light orange) and all styles |
| `public/figures/` | Diagrams (hand-written SVG) and plots (written by the example scripts) |
| `public/widgets/` | The interactive widgets (plain JavaScript, no build step) |
| `examples/` | One script for each lesson. It calculates each number of the lesson and writes its plots |
| `tests/` | Asserts for the numbers of the lessons, and a check of the lesson template |
| `AUTHORING.md` | The method and the template of a lesson |
