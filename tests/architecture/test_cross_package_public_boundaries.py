from __future__ import annotations

import ast
import json
from pathlib import Path

_ALLOWED_PREFIXES = (
    "answer_engineering",
    "answer_engineering.rules",
    "answer_engineering.telemetry",
)
_FORBIDDEN_PREFIXES = (
    "answer_engineering.engine.",
    "answer_engineering.inference.",
    "answer_engineering.config.",
)


def _iter_import_targets(py_file: Path) -> list[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            targets.append(node.module)
    return targets


def test_ae_paper_reproduction_uses_public_ae_boundaries_only() -> None:
    bad: list[tuple[str, str]] = []
    for py_file in Path("src/ae_paper_reproduction").rglob("*.py"):
        for target in _iter_import_targets(py_file):
            if not target.startswith("answer_engineering"):
                continue
            if target.startswith(_FORBIDDEN_PREFIXES):
                bad.append((str(py_file), target))
                continue
            if not target.startswith(_ALLOWED_PREFIXES):
                bad.append((str(py_file), target))
    assert bad == []


def test_reproduce_notebook_uses_package_root_imports_only() -> None:
    nb = json.loads(Path("notebooks/reproduce.ipynb").read_text())
    offending: list[str] = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for raw_line in cell.get("source", []):
            line = raw_line.strip()
            if not line.startswith(("from ", "import ")):
                continue
            if "answer_engineering" in line and (
                "from answer_engineering." in line
                or "import answer_engineering." in line
            ):
                offending.append(line)
            if "ae_paper_reproduction" in line and (
                "from ae_paper_reproduction." in line
                or "import ae_paper_reproduction." in line
            ):
                offending.append(line)
    assert offending == []
