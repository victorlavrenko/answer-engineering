#!/usr/bin/env python3
"""Find ``if TYPE_CHECKING:`` guards that indicate boundary pressure.

Architectural role:
    In this codebase, repeated ``TYPE_CHECKING`` guards are treated as a design
    smell pointing to import-cycle or layering issues. This utility provides a
    deterministic inventory used by convention checks and refactoring work.

Input provenance:
    The scan root is provided by command-line callers (developers or CI).

Output usage:
    Matches are returned as structured records and rendered by ``main()`` so
    pipelines can report and potentially gate architectural drift.

"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

TYPE_CHECKING_GUARD_RE = re.compile(
    r"^\s*if\s+TYPE_CHECKING\s*:\s*$", re.MULTILINE
)


@dataclass(frozen=True)
class TypeCheckingGuardMatch:
    """Structured report entry for one ``if TYPE_CHECKING:`` line.

    This value object bridges raw regex findings into stable outputs consumed by
    terminal reporting and automated convention checks.

    """

    path: Path
    line: int
    text: str


def find_type_checking_guards(root: Path) -> list[TypeCheckingGuardMatch]:
    """Scan Python files under ``root`` and collect TYPE_CHECKING guards.

    ``root`` is supplied by repository-level tooling. The sorted output is used
    downstream by the CLI layer and any tests asserting convention compliance.

    """
    matches: list[TypeCheckingGuardMatch] = []
    for path in root.rglob("*.py"):
        if any(
            part in {".git", ".venv", "venv", "__pycache__"}
            for part in path.parts
        ):
            continue
        text = path.read_text(encoding="utf-8")
        for match in TYPE_CHECKING_GUARD_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            matches.append(
                TypeCheckingGuardMatch(
                    path=path, line=line, text=match.group(0).strip()
                )
            )
    matches.sort(key=lambda item: (str(item.path), item.line))
    return matches


def main() -> None:
    """Execute the scanner as a command-line enforcement helper.

    Inputs come from argv, and output is a deterministic text report consumed by
    developer workflows and CI logs.

    """
    parser = argparse.ArgumentParser(
        description="Find `if TYPE_CHECKING:` guards in Python files."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Repository root to scan (default: current directory)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    matches = find_type_checking_guards(root)
    if not matches:
        print("No `if TYPE_CHECKING:` guards found.")
        return

    print(f"Found {len(matches)} TYPE_CHECKING guard(s):")
    for item in matches:
        print(f"{item.path.relative_to(root)}:{item.line} {item.text}")


if __name__ == "__main__":
    main()
