#!/usr/bin/env python3
"""Detect global builder factories that can blur architectural ownership.

Architectural role:
    This scanner is a convention-enforcement guard used by CI and developer
    checks to detect module-level ``build_*``/``make_*``/``create_*`` factories
    that return uppercase (type-like) annotations. In this codebase, those
    patterns are treated as architecture smells because they can hide lifecycle
    ownership behind broad factory entry points.

Input provenance:
    The ``root`` path is supplied by command-line callers (pre-commit, developer
    shell, or test harnesses) and represents a repository subtree. The scanner
    loads Python source files from that subtree.

Output usage:
    Results are returned as ``Match`` records for programmatic checks and are
    also rendered by ``main()`` into deterministic CLI output that can fail a
    conventions job or guide refactors.

"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path

# Non-greedy args capture as requested.
FUNC_SIG_RE = re.compile(
    (
        r"^def\s+([a-z_][a-z0-9_]*)\s*\((.*?)\)\s*->\s*"
        r"([A-Z][A-Za-z0-9_\[\], .|]*)\s*:"
    ),
    re.DOTALL,
)


@dataclass(frozen=True)
class Match:
    """One global-builder occurrence reported by the conventions scanner.

    Architecturally, this value object is the stable handoff format between the
    abstract-syntax-tree-based detection phase and reporting/CI consumers.

    """

    path: Path
    line: int
    name: str
    return_hint: str


def _is_builder_name(name: str) -> bool:
    """Return whether a function name matches the builder prefix policy.

    The input ``name`` comes from function definitions discovered in module AST
    bodies. The boolean output is consumed by ``find_global_builders`` to gate
    which definitions proceed to return-type validation.

    """
    return (
        name.startswith("build_")
        or name.startswith("make_")
        or name.startswith("create_")
    )


def _has_upper_return_annotation(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Check whether a function's return annotation looks type-like.

    ``node`` is produced by ``ast.parse`` during repository scanning. The result
    feeds the enforcement heuristic that flags global factories returning
    uppercase type names.

    """
    ann = node.returns
    if ann is None:
        return False
    if isinstance(ann, ast.Name):
        return ann.id != "None" and ann.id[:1].isupper()
    if isinstance(ann, ast.Attribute):
        return ann.attr[:1].isupper()
    return False


def find_global_builders(root: Path) -> list[Match]:
    """Scan Python modules under ``root`` for prohibited global builder shapes.

    Architecturally, this is the detection core for the global-builder
    convention rule. ``root`` is typically passed from repository-level tooling.
    The returned matches are sorted and consumed by CLI output and automated
    enforcement checks.

    """
    out: list[Match] = []
    for path in root.rglob("*.py"):
        if any(
            part in {".git", ".venv", "venv", "__pycache__", "tests"}
            for part in path.parts
        ):
            continue

        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)

        try:
            module = ast.parse(text)
        except SyntaxError:
            continue

        for node in module.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _is_builder_name(node.name):
                continue
            if not _has_upper_return_annotation(node):
                continue

            start = node.lineno - 1
            end = node.end_lineno or node.lineno
            segment = "".join(lines[start:end])
            m = FUNC_SIG_RE.search(segment)
            if not m:
                continue
            return_hint = m.group(3).strip()
            if return_hint == "None":
                continue

            out.append(
                Match(
                    path=path,
                    line=node.lineno,
                    name=node.name,
                    return_hint=return_hint,
                )
            )

    out.sort(key=lambda x: (str(x.path), x.line, x.name))
    return out


def main() -> None:
    """Run the global-builder scanner as a CLI command.

    CLI arguments come from local developers or CI jobs. This function routes
    them into ``find_global_builders`` and emits deterministic human-readable
    lines that are consumed by logs and enforcement pipelines.

    """
    parser = argparse.ArgumentParser(
        description=(
            "Find module-level global builder factories "
            "(build_/make_/create_) with uppercase return types, "
            "excluding class named constructors."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Repository root to scan (default: current directory)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    matches = find_global_builders(root)

    if not matches:
        print("No matching global builder patterns found.")
        return

    print(f"Found {len(matches)} global builder candidate(s):")
    for item in matches:
        rel_path = item.path.relative_to(root)
        print(
            f"{rel_path}:{item.line} def {item.name}(...) -> {item.return_hint}"
        )


if __name__ == "__main__":
    main()
