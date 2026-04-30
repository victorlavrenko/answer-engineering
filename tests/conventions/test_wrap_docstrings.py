from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.wrap_docstrings import reflow_docstring_content, render_docstring

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "wrap_docstrings"


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8").strip()


def _repo_root() -> Path:
    return _REPO_ROOT


def test_repository_docstrings_are_already_canonical() -> None:
    repo_root = _repo_root()
    copied_repo = Path(
        shutil.copytree(
            repo_root,
            Path(tempfile.mkdtemp(prefix="ae-docstrings-")) / "repo",
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                ".pytest_cache",
                "__pycache__",
            ),
            dirs_exist_ok=False,
        )
    )
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(copied_repo / "scripts" / "wrap_docstrings.py"),
                str(copied_repo),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        shutil.rmtree(copied_repo, ignore_errors=True)

    assert result.returncode == 0, result.stderr
    expected = "Processed docstrings. Files changed: 0"
    assert expected in result.stdout, (
        "Repository docstrings are not canonical. Run "
        "`python scripts/wrap_docstrings.py .` and commit the changes.\n"
        f"Output:\n{result.stdout}"
    )


def test_repairs_corrupted_reproduction_init_docstring_shape() -> None:
    canonical = _fixture("canonical_reproduction_init.txt")
    corrupted = _fixture("corrupted_reproduction_init.txt")
    got = "\n".join(reflow_docstring_content(corrupted, 80))
    assert got == canonical


def test_repairs_collapsed_heading_docstring_exactly() -> None:
    doc = (
        "Responsibilities ---------------- - Represent a proposed patch "
        "operation in a\n"
        "stable, serializable form. - Capture the minimal context required."
    )
    got = "\n".join(reflow_docstring_content(doc, 88))
    expected = (
        "Responsibilities\n"
        "----------------\n"
        "- Represent a proposed patch operation in a stable, "
        "serializable form.\n"
        "- Capture the minimal context required."
    )
    assert got == expected


def test_repairs_requested_nested_wrapping_shape() -> None:
    doc = (
        "A:\n"
        "  - b c d - e1 e2 - f1 2 3 -g g1 g2\n"
        "  E:\n"
        "     - f g h\n"
        "     - z x c"
    )
    got = "\n".join(reflow_docstring_content(doc, 40))
    expected = (
        "A:\n"
        "    - b c d\n"
        "    - e1 e2\n"
        "    - f1 2 3\n"
        "    - g g1 g2\n"
        "    E:\n"
        "        - f g h\n"
        "        - z x c"
    )
    assert got == expected


def test_render_docstring_preserves_structured_todo_nesting() -> None:
    doc = (
        "Planner boundary.\n\n"
        "TODO:\n"
        "    Target:\n"
        "        Keep planner as the only required proposal entry point while "
        "moving\n"
        "        provider-specific runtime adaptation behind narrower internal "
        "seams.\n\n"
        "    Boundary note:\n"
        "        The planner still reaches into concrete avoid-provider "
        "runtime state\n"
        "        during configure_runtime."
    )
    got = render_docstring(doc, "    ", 80)
    expected = (
        '    """Planner boundary.\n'
        "\n"
        "    TODO:\n"
        "        Target:\n"
        "            Keep planner as the only required proposal entry point "
        "while moving\n"
        "            provider-specific runtime adaptation behind narrower "
        "internal seams.\n"
        "\n"
        "        Boundary note:\n"
        "            The planner still reaches into concrete avoid-provider "
        "runtime state\n"
        "            during configure_runtime.\n"
        "\n"
        '    """'
    )
    assert got == expected
