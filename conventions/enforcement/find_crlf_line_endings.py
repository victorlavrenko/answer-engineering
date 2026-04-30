#!/usr/bin/env python3
"""Detect CRLF files so repository text encoding stays deterministic.

Architectural role:
    Cross-platform newline drift causes noisy diffs and brittle golden outputs.
    This scanner enforces a repository-level invariant: source artifacts should
    use LF line endings unless explicitly exempted.

Input provenance:
    The root path comes from command-line callers in local workflows or CI.

Output usage:
    ``CrlfMatch`` entries are returned for programmatic consumers and formatted
    by ``main()`` for enforcement logs.

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CrlfMatch:
    """Represents one file containing CRLF bytes.

    This record is consumed by CLI reporting and can be reused by tests that
    assert line-ending policy invariants.

    """

    path: Path


def find_crlf_line_endings(root: Path) -> list[CrlfMatch]:
    """Traverse ``root`` and return files that contain CRLF line endings.

    The ``root`` argument is supplied by repository-level tooling. Returned
    matches are sorted for stable output in CI and developer consoles.

    """
    matches: list[CrlfMatch] = []
    for path in root.rglob("*"):
        if not path.is_file() or any(
            part
            in {
                ".git",
                ".venv",
                "venv",
                "__pycache__",
                ".pytest_cache",
                "coverage",
                "paper",
                "dist",
                "build",
            }
            for part in path.parts
        ):
            continue
        data = path.read_bytes()
        if b"\r\n" in data:
            matches.append(CrlfMatch(path=path))
    matches.sort(key=lambda item: str(item.path))
    return matches


def main() -> None:
    """Expose CRLF scanning as a CLI command for convention enforcement.

    Parses argv, executes ``find_crlf_line_endings``, and prints deterministic
    findings consumed by local checks and CI jobs.

    """
    parser = argparse.ArgumentParser(
        description="Find files with CRLF line endings."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Directory tree to scan (default: current directory)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    matches = find_crlf_line_endings(root)
    if not matches:
        print("No CRLF line endings found.")
        return

    print(f"Found {len(matches)} file(s) with CRLF line endings:")
    for item in matches:
        print(item.path.relative_to(root))


if __name__ == "__main__":
    main()
