"""Inspect cross-subpackage imports under `answer_engineering.engine`.

Purpose:
    Report import edges between engine subpackages so maintainers can see when
    subsystem boundaries are becoming tangled.

Architectural role:
    Repository-maintenance analysis tool for dependency-boundary review.

Inputs (architectural provenance):
    Reads Python source files under the configured engine root and optional
    command-line filters that choose summary, zoom, matrix, or graph views.

Outputs (downstream usage):
    Prints occurrence lists, edge summaries, or adjacency matrices consumed by
    maintainers during refactors and architecture reviews.

Invariants/constraints:
    The tool is observational only. It should not rewrite imports or treat every
    cross-edge as wrong; it exposes evidence for human boundary decisions.

"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path("src/answer_engineering/engine")
PACKAGE_PREFIX = "answer_engineering.engine"


@dataclass(frozen=True, slots=True)
class ImportOccurrence:
    """One concrete import edge occurrence discovered from AST analysis."""

    importer_package: str
    dependency_package: str
    importer_file: Path
    imported_module: str
    imported_names: tuple[str, ...]
    line: int


def subpackage_of_engine_file(path: Path) -> str | None:
    """Return the engine subpackage name inferred from a source file path."""
    parts = path.parts
    if "engine" not in parts:
        return None
    index = parts.index("engine")
    if len(parts) <= index + 1:
        return None
    return parts[index + 1]


def subpackage_of_module_name(module_name: str) -> str | None:
    """Return the engine subpackage name from a fully qualified module name."""
    if not module_name.startswith(PACKAGE_PREFIX):
        return None
    parts = module_name.split(".")
    if len(parts) < 3:
        return None
    return parts[2]


def collect_occurrences() -> list[ImportOccurrence]:
    """Collect cross-subpackage `from ... import ...` rows via AST scan."""
    occurrences: list[ImportOccurrence] = []

    for file_path in PACKAGE_ROOT.rglob("*.py"):
        importer_package = subpackage_of_engine_file(file_path)
        if importer_package is None:
            continue

        try:
            source_text = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source_text)
        except Exception:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module is None:
                continue
            if not node.module.startswith(PACKAGE_PREFIX):
                continue

            dependency_package = subpackage_of_module_name(node.module)
            if dependency_package is None:
                continue
            if dependency_package == importer_package:
                continue

            imported_names = tuple(alias.name for alias in node.names)

            occurrences.append(
                ImportOccurrence(
                    importer_package=importer_package,
                    dependency_package=dependency_package,
                    importer_file=file_path,
                    imported_module=node.module,
                    imported_names=imported_names,
                    line=node.lineno,
                )
            )

    occurrences.sort(
        key=lambda item: (
            item.importer_package,
            item.dependency_package,
            str(item.importer_file),
            item.line,
            item.imported_module,
            item.imported_names,
        )
    )
    return occurrences


def edge_set(
    occurrences: list[ImportOccurrence],
) -> set[tuple[str, str]]:
    """Collapse occurrence rows into unique importer/dependency edges."""
    return {
        (item.importer_package, item.dependency_package) for item in occurrences
    }


def print_helicopter_view(occurrences: list[ImportOccurrence]) -> None:
    """Print one line per unique dependency edge."""
    edges = sorted(edge_set(occurrences))
    for importer_package, dependency_package in edges:
        print(f"{importer_package} <- {dependency_package}")


def print_edge_summary(
    occurrences: list[ImportOccurrence],
    importer: str | None,
    dependency: str | None,
) -> None:
    """Print aggregated import counts for filtered edge groups."""
    filtered = [
        item
        for item in occurrences
        if (importer is None or item.importer_package == importer)
        and (dependency is None or item.dependency_package == dependency)
    ]

    if not filtered:
        print("No matches.")
        return

    grouped: dict[tuple[str, str], list[ImportOccurrence]] = defaultdict(list)
    for item in filtered:
        grouped[(item.importer_package, item.dependency_package)].append(item)

    for importer_package, dependency_package in sorted(grouped):
        items = grouped[(importer_package, dependency_package)]
        file_count = len({item.importer_file for item in items})
        print(
            f"{importer_package} <- {dependency_package} "
            f"(imports: {len(items)}, files: {file_count})"
        )


def print_zoom_view(
    occurrences: list[ImportOccurrence],
    importer: str | None,
    dependency: str | None,
) -> None:
    """Print per-file import statements for filtered edge groups."""
    filtered = [
        item
        for item in occurrences
        if (importer is None or item.importer_package == importer)
        and (dependency is None or item.dependency_package == dependency)
    ]

    if not filtered:
        print("No matches.")
        return

    current_edge: tuple[str, str] | None = None
    current_file: Path | None = None

    for item in filtered:
        edge = (item.importer_package, item.dependency_package)
        if edge != current_edge:
            if current_edge is not None:
                print()
            print(f"{item.importer_package} <- {item.dependency_package}")
            current_edge = edge
            current_file = None

        if item.importer_file != current_file:
            rel_path = item.importer_file.as_posix()
            print(f"  {rel_path}")
            current_file = item.importer_file

        imported_names_text = ", ".join(item.imported_names)
        print(
            f"    L{item.line}: from {item.imported_module} "
            f"import {imported_names_text}"
        )


def print_matrix(occurrences: list[ImportOccurrence]) -> None:
    """Print an adjacency matrix of cross-subpackage dependencies."""
    edges = edge_set(occurrences)
    packages = sorted(
        {item.importer_package for item in occurrences}
        | {item.dependency_package for item in occurrences}
    )

    header = " " * 14 + " ".join(f"{pkg:12}" for pkg in packages)
    print(header)

    for importer_package in packages:
        row = f"{importer_package:12} "
        for dependency_package in packages:
            row += (
                "     X      "
                if (
                    importer_package,
                    dependency_package,
                )
                in edges
                else "            "
            )
        print(row)


def main() -> None:
    """CLI entrypoint for dependency edge summaries and views."""
    parser = argparse.ArgumentParser(
        description=(
            "Explore engine subpackage dependencies. "
            "Convention: A <- B means A imports from B."
        )
    )
    parser.add_argument(
        "--importer",
        help="Filter by importer subpackage name, e.g. proposal",
    )
    parser.add_argument(
        "--dependency",
        help="Filter by dependency subpackage name, e.g. runtime",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Show subpackage dependency matrix.",
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        help=(
            "Show file-level import occurrences for the selected edge or "
            "filter. This is the real zoom mode."
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show filtered edge summary with occurrence counts.",
    )
    args = parser.parse_args()

    occurrences = collect_occurrences()

    if args.matrix:
        print_matrix(occurrences)
        return

    if args.zoom:
        print_zoom_view(
            occurrences,
            importer=args.importer,
            dependency=args.dependency,
        )
        return

    if args.summary:
        print_edge_summary(
            occurrences,
            importer=args.importer,
            dependency=args.dependency,
        )
        return

    print_helicopter_view(occurrences)


if __name__ == "__main__":
    main()
