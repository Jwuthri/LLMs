# LLM Course, Explained

A website that explains each section of the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime
Labonne (Apache-2.0) with one tiny example, real numbers, formulas, figures and small interactive widgets.

## Run

```bash
uv sync --frozen
uv run mkdocs serve          # http://127.0.0.1:8000
uv run pytest -q             # checks each number shown on the pages
uv run mkdocs build --strict # static site in site/
```

## Layout

| Path | Content |
|---|---|
| `docs/` | The pages (Markdown), figures, widgets, CSS |
| `examples/` | One script for each page. It calculates each number on the page and writes the plots |
| `tests/` | Asserts for the numbers on the pages |
| `AUTHORING.md` | The method and the template of a page |
