from __future__ import annotations

from pathlib import Path

from conventions.enforcement.find_any_usage import find_any_usage
from conventions.enforcement.find_crlf_line_endings import (
    find_crlf_line_endings,
)
from conventions.enforcement.find_global_builders import find_global_builders
from conventions.enforcement.find_type_checking_guards import (
    find_type_checking_guards,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_no_module_level_global_builders() -> None:
    matches = find_global_builders(_repo_root())
    assert matches == []


def test_no_type_checking_guards() -> None:
    matches = find_type_checking_guards(_repo_root())
    assert matches == []


def test_no_any_usage_in_src_package() -> None:
    matches = find_any_usage(_repo_root() / "src" / "answer_engineering")
    assert matches == []


def test_no_crlf_line_endings() -> None:
    matches = find_crlf_line_endings(_repo_root())
    assert matches == []


def test_crlf_ignored_directories_are_relative_to_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    source_file = root / "src" / "bad.py"
    ignored_file = root / "tmp" / "ignored.py"
    source_file.parent.mkdir(parents=True)
    ignored_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"x = 1\r\n")
    ignored_file.write_bytes(b"x = 2\r\n")

    matches = find_crlf_line_endings(root)

    assert [match.path for match in matches] == [source_file]
