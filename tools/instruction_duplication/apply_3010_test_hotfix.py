#!/usr/bin/env python3
"""Apply the non-scientific 3.0.10 hotfixes found by the full test suite.

This hotfix does not change generation prompts or the deterministic counterfactual
judge introduced in 3.0.10. It only:
  * keeps the optional 199-task human-audit export from breaking tiny smoke/test runs;
  * prevents the human display from visually highlighting the grammatical token 'not';
  * restores explicit generated-state .gitignore entries required by the quality test.

The audit.py and regression-test file are supplied by the overlay itself. This
script only patches .gitignore because that file is intentionally user-maintained.
"""

from __future__ import annotations

from pathlib import Path

REQUIRED_GITIGNORE = (
    "/run/.lock",
    "/run/state/run.sqlite3-wal",
    "/run/state/run.sqlite3-shm",
)


def main() -> int:
    root = Path.cwd()
    init_path = root / "src" / "instruction_duplication" / "__init__.py"
    audit_path = root / "src" / "instruction_duplication" / "audit.py"
    gitignore_path = root / ".gitignore"
    if not init_path.is_file() or not audit_path.is_file() or not gitignore_path.is_file():
        raise SystemExit(f"not an instruction-duplication source checkout: {root}")

    init_text = init_path.read_text(encoding="utf-8")
    if '__version__ = "3.0.10"' not in init_text:
        raise SystemExit("this hotfix expects package version 3.0.10")

    audit_text = audit_path.read_text(encoding="utf-8")
    required_markers = (
        'DISPLAY_STOPWORDS = TFIDF_STOPWORDS | {"not"}',
        '"skipped": True,',
        "frozen human audit requires at least",
    )
    missing = [marker for marker in required_markers if marker not in audit_text]
    if missing:
        raise SystemExit(
            "the overlay audit.py does not appear to be installed; extract the tgz with "
            "--strip-components=1 before running this script"
        )

    text = gitignore_path.read_text(encoding="utf-8")
    missing_patterns = [pattern for pattern in REQUIRED_GITIGNORE if pattern not in text]
    if missing_patterns:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n# SQLite/runtime state inside the canonical run workspace\n"
        text += "\n".join(missing_patterns) + "\n"
        gitignore_path.write_text(text, encoding="utf-8")
        print("added explicit runtime-state .gitignore entries")
    else:
        print("runtime-state .gitignore entries already present")

    print("3.0.10 test hotfix is installed")
    print("counterfactual judge and generation code were not changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
