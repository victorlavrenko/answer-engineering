#!/usr/bin/env python3
"""Detect ``Any`` usage as part of typed-boundary enforcement.

Architectural role:
    The project treats ``Any`` as a narrow escape hatch. This utility
    inventories where ``Any`` appears so typed architecture constraints can be
    enforced by humans and CI jobs.

Input provenance:
    The root directory is provided by command-line callers and usually points at
    repository source folders.

Output usage:
    The scanner returns ``AnyUsageMatch`` records, which ``main()`` emits in a
    stable text format for lint-like reporting and convention gates.

"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

ANY_TOKEN_RE = re.compile(r"\bAny\b")


@dataclass(frozen=True)
class AnyUsageMatch:
    """One ``Any`` token occurrence discovered during repository scanning.

    This data class is the architectural handoff between raw text scanning and
    reporting layers that decide whether the usage is acceptable.

    """

    path: Path
    line: int
    text: str


def find_any_usage(root: Path) -> list[AnyUsageMatch]:
    """Scan Python files under ``root`` and collect lines containing ``Any``.

    ``root`` comes from command-line orchestration. The sorted output feeds CLI
    reporting and can be consumed by tests or policy checks.

    """
    matches: list[AnyUsageMatch] = []
    for path in root.rglob("*.py"):
        if any(
            part in {".git", ".venv", "venv", "__pycache__"}
            for part in path.parts
        ):
            continue
        for idx, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if ANY_TOKEN_RE.search(line):
                matches.append(
                    AnyUsageMatch(path=path, line=idx, text=line.strip())
                )
    matches.sort(key=lambda item: (str(item.path), item.line))
    return matches


def main() -> None:
    """Run the ``Any`` scanner from the command line.

    This function converts argv input into a repository scan and prints
    deterministic output used by local checks and CI diagnostics.

    """
    parser = argparse.ArgumentParser(
        description="Find `Any` usage in Python files."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Directory tree to scan (default: current directory)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    matches = find_any_usage(root)
    if not matches:
        print("No `Any` usage found.")
        return

    print(f"Found {len(matches)} `Any` usage(s):")
    for item in matches:
        print(f"{item.path.relative_to(root)}:{item.line} {item.text}")


if __name__ == "__main__":
    main()
