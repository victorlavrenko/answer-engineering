#!/usr/bin/env python3
"""Inventory missing docstrings across ``src`` with architectural.

Architectural role:
    This script is the planning entry point for docstring migration work. It
    turns the current source tree into an actionable backlog so documentation
    can be implemented in architecture-first batches.

Input provenance:
    The scanner root and optional output path are provided by command-line
    callers (developers/CI planning jobs). Source code is read from repository
    files under that root.

Output usage:
    The script emits a markdown report consumed by maintainers to plan and track
    phased docstring work. The report is designed to be committed as a project
    artifact for review and progress tracking.

"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

type _DocumentableNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
type _DocstringOwnerNode = _DocumentableNode | ast.Module


@dataclass(frozen=True)
class MissingDocItem:
    """One missing-docstring symbol discovered during source inventory.

    This record carries both local symbol context and architectural bucket data
    so downstream planning can prioritize work by subsystem boundary impact.

    """

    path: Path
    symbol_kind: str
    symbol_name: str
    line: int
    bucket: str
    priority: int


_EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
}


def _priority_bucket_for_path(path: Path) -> tuple[str, int]:
    """Map a source path to an architectural priority bucket.

    The input ``path`` comes from filesystem traversal in ``find_missing_doc``.
    Returned bucket/priority values are used in report sorting and in the
    checklist that drives phased documentation rollout.

    """
    normalized = str(path).replace("\\", "/")
    if "/engine/api/" in normalized:
        return ("P1 API and extension boundaries", 1)
    if (
        "/engine/orchestration/" in normalized
        or "/engine/runtime/" in normalized
    ):
        return ("P2 orchestration and runtime core", 2)
    if "/engine/scoring/" in normalized or "/engine/proposal/" in normalized:
        return ("P2 scoring and proposal core", 2)
    if "/inference/" in normalized:
        return ("P3 inference subsystem", 3)
    if "/reproduction/" in normalized:
        return ("P3 reproduction subsystem", 3)
    if "/cli/" in normalized:
        return ("P4 CLI and support", 4)
    return ("P4 remaining src modules", 4)


def _iter_top_level_symbols(module: ast.Module) -> list[_DocumentableNode]:
    """Return top-level classes and functions from a module AST.

    ``module`` originates from ``ast.parse`` and the returned list is consumed
    by missing-docstring detection to inventory boundary-level symbols.

    """
    nodes: list[_DocumentableNode] = []
    for node in module.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            nodes.append(node)
    return nodes


def _iter_public_methods(
    class_node: ast.ClassDef,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return public methods for class-level inventory.

    The class node is produced by AST parsing. Returned methods are used to
    identify callable API surfaces that need docstrings during migration.

    """
    methods: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in class_node.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and not node.name.startswith("_"):
            methods.append(node)
    return methods


def _has_docstring(node: _DocstringOwnerNode) -> bool:
    """Return whether an abstract-syntax-tree node has an attached docstring."""
    return ast.get_docstring(node) is not None


def find_missing_doc(root: Path) -> list[MissingDocItem]:
    """Scan ``root`` for missing module/symbol docstrings in ``src`` Python.

    ``root`` is typically the repository root. The returned records feed both
    terminal summary output and a markdown checklist artifact used for phased
    documentation implementation.

    """
    items: list[MissingDocItem] = []
    src_root = root / "src"
    for path in src_root.rglob("*.py"):
        if any(part in _EXCLUDED_PARTS for part in path.parts):
            continue

        text = path.read_text(encoding="utf-8")
        rel_path = path.relative_to(root)
        bucket, priority = _priority_bucket_for_path(rel_path)

        try:
            module = ast.parse(text)
        except SyntaxError:
            continue

        if not _has_docstring(module):
            items.append(
                MissingDocItem(
                    path=rel_path,
                    symbol_kind="module",
                    symbol_name="(module)",
                    line=1,
                    bucket=bucket,
                    priority=priority,
                )
            )

        for node in _iter_top_level_symbols(module):
            if not _has_docstring(node):
                symbol_kind = (
                    "class" if isinstance(node, ast.ClassDef) else "function"
                )
                items.append(
                    MissingDocItem(
                        path=rel_path,
                        symbol_kind=symbol_kind,
                        symbol_name=node.name,
                        line=node.lineno,
                        bucket=bucket,
                        priority=priority,
                    )
                )

            if isinstance(node, ast.ClassDef):
                for method in _iter_public_methods(node):
                    if not _has_docstring(method):
                        items.append(
                            MissingDocItem(
                                path=rel_path,
                                symbol_kind="method",
                                symbol_name=f"{node.name}.{method.name}",
                                line=method.lineno,
                                bucket=bucket,
                                priority=priority,
                            )
                        )

    items.sort(
        key=lambda item: (
            item.priority,
            item.bucket,
            str(item.path),
            item.line,
            item.symbol_kind,
            item.symbol_name,
        )
    )
    return items


def _render_markdown(items: list[MissingDocItem]) -> str:
    """Render a markdown backlog for Step 1 planning output.

    The markdown is committed as a planning artifact and consumed by subsequent
    migration steps to select implementation batches.

    """
    by_bucket: dict[str, list[MissingDocItem]] = {}
    for item in items:
        by_bucket.setdefault(item.bucket, []).append(item)

    lines: list[str] = []
    lines.append("# Step 1 — Missing docstring inventory (src)")
    lines.append("")
    lines.append(
        "Generated by `conventions/enforcement/inventory_docstrings.py`."
    )
    lines.append("")
    lines.append(f"Total missing docstring targets: **{len(items)}**")
    lines.append("")

    if not items:
        lines.append("No missing docstrings detected in `src/`.")
        return "\n".join(lines) + "\n"

    lines.append("## Priority buckets")
    lines.append("")
    for bucket in sorted(
        by_bucket.keys(), key=lambda b: by_bucket[b][0].priority
    ):
        lines.append(f"- **{bucket}**: {len(by_bucket[bucket])} target(s)")
    lines.append("")

    lines.append("## Detailed checklist")
    lines.append("")
    for bucket in sorted(
        by_bucket.keys(), key=lambda b: by_bucket[b][0].priority
    ):
        lines.append(f"### {bucket}")
        lines.append("")
        for item in by_bucket[bucket]:
            lines.append(
                "- [ ] "
                f"`{item.path}:{item.line}` "
                f"{item.symbol_kind} `{item.symbol_name}`"
            )
        lines.append("")

    lines.append("## Suggested first batch (high impact)")
    lines.append("")
    first_batch = [item for item in items if item.priority <= 2][:30]
    for item in first_batch:
        lines.append(
            "- "
            f"`{item.path}:{item.line}` "
            f"{item.symbol_kind} `{item.symbol_name}`"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- This inventory intentionally includes modules, classes, "
        "functions, and public methods."
    )
    lines.append(
        "- Private methods (`_name`) are excluded from Step 1 backlog "
        "by default."
    )
    lines.append(
        "- Syntax-error files are skipped to keep the report generation "
        "deterministic."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    """Generate and optionally write the Step 1 docstring inventory report.

    Inputs originate from command-line invocation. Outputs are printed to stdout
    and optionally persisted to a markdown file used for planning/review.

    """
    parser = argparse.ArgumentParser(
        description=(
            "Inventory missing docstrings in src/ with "
            "architectural priority buckets."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--output",
        default="docs/docstring-inventory-step1.md",
        help="Markdown report path relative to repository root",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    items = find_missing_doc(root)
    report = _render_markdown(items)

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"Wrote inventory: {output_path.relative_to(root)}")
    print(f"Total missing docstring targets: {len(items)}")


if __name__ == "__main__":
    main()
