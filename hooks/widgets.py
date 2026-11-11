"""MkDocs hook: load every widget script in docs/javascripts/widgets/ on each page."""

from pathlib import Path


def on_config(config):
    widgets_dir = Path(config["docs_dir"]) / "javascripts" / "widgets"
    for script in sorted(widgets_dir.glob("*.js")):
        config["extra_javascript"].append(f"javascripts/widgets/{script.name}")
    return config
