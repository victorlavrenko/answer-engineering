from __future__ import annotations

from pathlib import Path

from answer_engineering import GenerationPolicy

SNAPSHOT_PATH = Path("tests/golden/default_system_prompt.txt")


def test_default_system_prompt_matches_golden() -> None:
    assert SNAPSHOT_PATH.exists(), (
        f"Golden snapshot missing: {SNAPSHOT_PATH}. "
        "Run `python tests/regenerate_goldens.py` and inspect the diff."
    )

    expected = SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert expected == GenerationPolicy.default_system_prompt, (
        "GenerationPolicy.default_system_prompt changed from golden snapshot. "
        "If intentional, regenerate with `python tests/regenerate_goldens.py`."
    )
