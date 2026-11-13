"""Each section page follows the template in AUTHORING.md."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted(p for track in ("fundamentals", "scientist", "engineer", "safety", "datasets") for p in (ROOT / "content" / track).glob("*.md"))

REQUIRED_HEADINGS = [
    "## Core idea",
    "## The example",
    "## Minimal code",
    "## Comparison",
    "## Checks and common misconceptions",
    "## Sources",
]


def test_all_34_pages_exist():
    assert len(PAGES) == 34


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
    assert text.count('<div class="key-point">') == len(sub_parts)

    steps = len(re.findall(r"^\d+\. \*\*", text, flags=re.MULTILINE))
    whys = len(re.findall(r"^\s*\*\*Why:\*\*", text, flags=re.MULTILINE))
    assert whys >= steps >= 3 * len(sub_parts), f"{steps} steps but {whys} 'Why' paragraphs"

    assert "https://github.com/mlabonne/llm-course" in text, "credit the course in Sources"

    figures = re.findall(r'<figure class="diagram" data-src="([^"]+)">', text)
    figures += re.findall(r'<img src="/figures/([^"]+)"', text)
    assert figures, "a page has one figure or more"
    for figure in figures:
        assert (ROOT / "public" / "figures" / figure).exists(), f"missing figure: {figure}"

    for widget in re.findall(r'data-widget="([\w-]+)"', text):
        assert (ROOT / "public" / "widgets" / f"{widget}.js").exists(), f"missing widget: {widget}"

    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    words = len(re.findall(r"[A-Za-z][A-Za-z'-]*", prose))
    assert 1500 <= words <= 3600, f"{words} words"
