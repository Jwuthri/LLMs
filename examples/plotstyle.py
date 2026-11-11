"""Shared plot style. A plot is saved as public/figures/<page>/<name>.svg and shown on a white card."""

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent.parent / "public" / "figures"

# Colours of the site theme (app/globals.css). The series order is fixed: orange, blue, green, yellow.
# Checked with the palette validator: lightness band, chroma, colour-blind separation of adjacent series.
THEME = {
    "series": ["#eb6834", "#2a78d6", "#1baf7a", "#eda100"],
    "ink": "#2b2420",
    "ink2": "#665b52",
    "muted": "#9a8e83",
    "grid": "#efe7dc",
    "axis": "#ddd0bf",
    "surface": "#ffffff",
}


def save_figure(draw: Callable[[plt.Axes, dict], None], page: str, name: str,
                size: tuple[float, float] = (6.4, 3.6)) -> None:
    """Call `draw(ax, theme)` and save the SVG file under figures/<page>/."""
    out_dir = FIGURES_DIR / page
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = {
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": THEME["ink"],
        "axes.labelcolor": THEME["ink2"],
        "axes.edgecolor": THEME["axis"],
        "axes.facecolor": THEME["surface"],
        "figure.facecolor": THEME["surface"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": THEME["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": THEME["muted"],
        "ytick.color": THEME["muted"],
        "lines.linewidth": 2,
        "svg.hashsalt": "llm-course-explained",
        "legend.frameon": False,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=size)
        draw(ax, THEME)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.svg", metadata={"Date": None})
        plt.close(fig)
