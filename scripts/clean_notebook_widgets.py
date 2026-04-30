"""Remove `metadata.widgets` from notebook files passed on the command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def clean_notebook(path: Path) -> bool:
    """Remove widget metadata from one notebook and report if it existed."""
    with path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    metadata = nb.get("metadata", {})
    had_widgets = "widgets" in metadata
    metadata.pop("widgets", None)

    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")

    return had_widgets


def main() -> int:
    """CLI entrypoint for cleaning widget metadata in one or more notebooks."""
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/clean_notebook_widgets.py "
            "notebook.ipynb [...]"
        )
        return 1

    for arg in sys.argv[1:]:
        path = Path(arg)
        changed = clean_notebook(path)
        widgets_status = (
            "removed metadata.widgets"
            if changed
            else "no widgets metadata found"
        )
        print(f"{path}: {widgets_status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
