"""Each section page follows the template in AUTHORING.md."""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
PAGES = sorted(p for track in ("fundamentals", "scientist", "engineer") for p in (DOCS / track).glob("*.md"))

REQUIRED_HEADINGS = [
    "## Core idea",
    "## The example",
    "## Minimal code",
    "## Comparison",
    "## Checks and common misconceptions",
    "## Sources",
]


def test_all_20_pages_exist():
    assert len(PAGES) == 20


@pytest.mark.parametrize("page", PAGES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_page_follows_template(page: Path):
    text = page.read_text()
    lines = text.splitlines()

    for heading in REQUIRED_HEADINGS:
        assert heading in lines, f"missing heading: {heading}"

    sub_parts = [line for line in lines if re.match(r"## \d+\. ", line)]
    assert len(sub_parts) >= 3, "a page has one numbered section for each course bullet"
    for block in ("### Setup", "### Steps", "### From toy to real"):
        assert lines.count(block) == len(sub_parts), f"each sub-part needs one '{block}'"
    assert text.count('!!! success "Key point"') == len(sub_parts)

    steps = len(re.findall(r"^\d+\. \*\*", text, flags=re.MULTILINE))
    whys = len(re.findall(r"^\s*\*\*Why:\*\*", text, flags=re.MULTILINE))
    assert whys >= steps >= 3 * len(sub_parts), f"{steps} steps but {whys} 'Why' paragraphs"
    assert text.count("{ .why }") == whys, "each 'Why' paragraph ends with '{ .why }'"

    assert "https://github.com/mlabonne/llm-course" in text, "credit the course in Sources"

    for figure in re.findall(r"docs/assets/figures/[\w./-]+\.svg|\.\./assets/figures/[\w./-]+\.svg", text):
        path = DOCS.parent / figure if figure.startswith("docs/") else (page.parent / figure).resolve()
        assert path.exists(), f"missing figure: {figure}"

    for widget in re.findall(r'data-widget="([\w-]+)"', text):
        assert (DOCS / "javascripts" / "widgets" / f"{widget}.js").exists(), f"missing widget: {widget}"

    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    words = len(re.findall(r"[A-Za-z][A-Za-z'-]*", prose))
    assert 1500 <= words <= 3600, f"{words} words"
