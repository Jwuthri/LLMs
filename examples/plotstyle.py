"""Shared plot style. Each plot is saved twice: `<name>.svg` (light) and `<name>-dark.svg` (dark)."""

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets" / "figures"

# Validated categorical palette: fixed order, light and dark steps.
THEMES = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "surface": "#fcfcfb",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "surface": "#1a1a19",
    },
}


def save_light_dark(draw: Callable[[plt.Axes, dict], None], page: str, name: str,
                    size: tuple[float, float] = (6.4, 3.6)) -> None:
    """Call `draw(ax, theme)` once per theme and save both SVG files under figures/<page>/."""
    out_dir = FIGURES_DIR / page
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        rc = {
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": theme["ink"],
            "axes.labelcolor": theme["ink2"],
            "axes.edgecolor": theme["axis"],
            "axes.facecolor": theme["surface"],
            "figure.facecolor": theme["surface"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": theme["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "lines.linewidth": 2,
            "svg.hashsalt": "llm-course-explained",
            "legend.frameon": False,
        }
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=size)
            draw(ax, theme)
            fig.tight_layout()
            suffix = "" if mode == "light" else "-dark"
            fig.savefig(out_dir / f"{name}{suffix}.svg", metadata={"Date": None})
            plt.close(fig)
