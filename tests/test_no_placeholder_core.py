from __future__ import annotations

from pathlib import Path


def test_no_obvious_placeholders() -> None:
    offenders: list[str] = []
    for path in Path("src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "return []" in text:
            offenders.append(str(path))
    assert offenders == []
