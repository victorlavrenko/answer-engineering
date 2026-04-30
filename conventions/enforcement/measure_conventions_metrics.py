#!/usr/bin/env python3
"""Compute a conventions metrics snapshot for the current repository state.

This script automates the subset of metrics from
`docs/conventions_dods_and_metrics.md` that can be measured reliably from a
single working tree snapshot.

"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_BOUNDARY_ROOTS = {
    "rules",
    "inference",
    "engine",
    "infra",
}

_EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
}

_TYPE_IGNORE_RE = re.compile(r"#\s*type:\s*ignore")
_ANY_TOKEN_RE = re.compile(r"\bAny\b")
_TYPE_CHECKING_GUARD_RE = re.compile(r"^\s*if\s+TYPE_CHECKING\s*:\s*$")


@dataclass(frozen=True)
class MetricRecord:
    """One conventions metric value and associated status metadata."""

    metric: str
    value: int | float | str | None
    target: str
    status: str
    notes: str


def _iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in _EXCLUDED_PARTS for part in path.parts):
            continue
        files.append(path)
    files.sort()
    return files


def _parse_module(path: Path) -> ast.Module | None:
    text = path.read_text(encoding="utf-8")
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def _count_relative_imports(paths: list[Path]) -> int:
    count = 0
    for path in paths:
        module = _parse_module(path)
        if module is None:
            continue
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                count += 1
    return count


def _count_type_ignores(paths: list[Path]) -> int:
    count = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if _TYPE_IGNORE_RE.search(line):
                count += 1
    return count


def _count_any_usage(paths: list[Path]) -> int:
    count = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if _ANY_TOKEN_RE.search(line):
                count += 1
    return count


def _count_type_checking_guards(paths: list[Path]) -> int:
    count = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if _TYPE_CHECKING_GUARD_RE.match(line):
                count += 1
    return count


def _count_global_builder_candidates(paths: list[Path]) -> int:
    count = 0
    for path in paths:
        module = _parse_module(path)
        if module is None:
            continue
        for node in module.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not (
                node.name.startswith("build_")
                or node.name.startswith("make_")
                or node.name.startswith("create_")
            ):
                continue
            ann = node.returns
            if ann is None:
                continue
            return_hint = ast.unparse(ann).strip()
            if return_hint[:1].isupper():
                count += 1
    return count


def _count_ambiguous_names(paths: list[Path]) -> tuple[int, int]:
    ambiguous_pkg = 0
    ambiguous_file = 0
    for path in paths:
        for part in path.parts:
            if part in {"utils", "common"}:
                ambiguous_pkg += 1
        if path.stem in {"utils", "common"}:
            ambiguous_file += 1
    return ambiguous_pkg, ambiguous_file


def _imported_modules(module: ast.Module) -> list[str]:
    imported: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    return imported


def _count_forbidden_boundary_to_legacy_imports(src_root: Path) -> int:
    count = 0
    for path in _iter_python_files(src_root):
        rel = path.relative_to(src_root)
        if not rel.parts:
            continue
        if rel.parts[0] not in _BOUNDARY_ROOTS:
            continue
        module = _parse_module(path)
        if module is None:
            continue
        for mod in _imported_modules(module):
            if mod.startswith("answer_engineering.engine.") or mod.startswith(
                "answer_engineering.inference."
            ):
                count += 1
    return count


def _count_legacy_package_paths(src_root: Path) -> int:
    legacy = [
        src_root / "engine",
        src_root / "inference",
        src_root / "reproduction",
    ]
    return sum(1 for item in legacy if item.exists())


def _coverage_files_with_explicit_boundary_owner(
    src_root: Path,
) -> tuple[int, int]:
    files = _iter_python_files(src_root)
    in_scope = [p for p in files if p.parent != src_root]
    owned = [
        p
        for p in in_scope
        if p.relative_to(src_root).parts[0] in _BOUNDARY_ROOTS | {"config"}
    ]
    return len(owned), len(in_scope)


def _status_zero(value: int) -> str:
    return "pass" if value == 0 else "fail"


def measure(repo_root: Path) -> list[MetricRecord]:
    """Measure conventions metrics from repository snapshot."""
    src_root = repo_root / "src" / "answer_engineering"
    test_root = repo_root / "tests"

    src_files = _iter_python_files(src_root)
    test_files = _iter_python_files(test_root) if test_root.exists() else []
    any_usage_count = _count_any_usage(src_files)
    type_checking_guard_count = _count_type_checking_guards(src_files)
    type_ignore_src = _count_type_ignores(src_files)
    type_ignore_tests = _count_type_ignores(test_files)
    forbidden_legacy_imports = _count_forbidden_boundary_to_legacy_imports(
        src_root
    )
    relative_import_count = _count_relative_imports(src_files)
    legacy_package_paths = _count_legacy_package_paths(src_root)
    global_builder_candidates = _count_global_builder_candidates(src_files)

    owned, in_scope = _coverage_files_with_explicit_boundary_owner(src_root)
    owner_pct = 100.0 if in_scope == 0 else (owned / in_scope) * 100.0

    ambiguous_pkg, ambiguous_file = _count_ambiguous_names(src_files)

    records: list[MetricRecord] = [
        MetricRecord(
            metric="forbidden_dependency_edges",
            value=forbidden_legacy_imports,
            target="0",
            status=_status_zero(forbidden_legacy_imports),
            notes=(
                "Counts boundary-module imports of legacy "
                "answer_engineering.engine*/inference* namespaces."
            ),
        ),
        MetricRecord(
            metric="ambiguous_package_names_count",
            value=ambiguous_pkg,
            target="0",
            status=_status_zero(ambiguous_pkg),
            notes="Counts path segments named utils/common under src.",
        ),
        MetricRecord(
            metric="ambiguous_file_names_count",
            value=ambiguous_file,
            target="0",
            status=_status_zero(ambiguous_file),
            notes="Counts Python modules named utils.py/common.py under src.",
        ),
        MetricRecord(
            metric="files_with_explicit_boundary_owner_ratio",
            value=round(owner_pct, 2),
            target="100%",
            status="pass" if round(owner_pct, 2) == 100.0 else "warn",
            notes=(
                f"{owned}/{in_scope} files are under declared boundary roots."
            ),
        ),
        MetricRecord(
            metric="any_usage_count",
            value=any_usage_count,
            target="0 (or bounded allowlist)",
            status=_status_zero(any_usage_count),
            notes="Token-level Any usage inventory in production source tree.",
        ),
        MetricRecord(
            metric="type_checking_guard_count",
            value=type_checking_guard_count,
            target="trend down",
            status="pass" if type_checking_guard_count == 0 else "warn",
            notes="Counts `if TYPE_CHECKING:` import-cycle workaround guards.",
        ),
        MetricRecord(
            metric="unjustified_type_ignore_count_src",
            value=type_ignore_src,
            target="0",
            status=_status_zero(type_ignore_src),
            notes="Raw `# type: ignore` occurrences in production source.",
        ),
        MetricRecord(
            metric="type_ignore_count_tests",
            value=type_ignore_tests,
            target="0",
            status="warn" if type_ignore_tests > 0 else "pass",
            notes="Raw `# type: ignore` occurrences in tests.",
        ),
        MetricRecord(
            metric="parallel_internal_paths_per_capability_proxy",
            value=legacy_package_paths,
            target="0 legacy package trees",
            status=_status_zero(legacy_package_paths),
            notes="Proxy for canonical-path convergence: legacy trees present.",
        ),
        MetricRecord(
            metric="relative_import_count",
            value=relative_import_count,
            target="0",
            status=_status_zero(relative_import_count),
            notes="Counts `from .` / `from ..` import statements in src.",
        ),
        MetricRecord(
            metric="global_builder_candidate_count",
            value=global_builder_candidates,
            target="0",
            status=_status_zero(global_builder_candidates),
            notes="Counts module-level build_/make_/create_ factories.",
        ),
        MetricRecord(
            metric="gate_pass_rate",
            value=None,
            target=">=98%",
            status="not_automated",
            notes="Requires PR/CI historical dataset, not a single checkout.",
        ),
        MetricRecord(
            metric="boundary_api_adoption",
            value=None,
            target="100%",
            status="not_automated",
            notes="Requires call-site classification across boundaries.",
        ),
        MetricRecord(
            metric="typed_public_api_coverage",
            value=None,
            target="100%",
            status="not_automated",
            notes="Needs API-surface definition + type-annotation analyzer.",
        ),
        MetricRecord(
            metric="phase_validation_record_completeness",
            value=None,
            target="100%",
            status="not_automated",
            notes="Requires phase ledger and expected-phase registry.",
        ),
    ]
    return records


def _summary(records: list[MetricRecord]) -> dict[str, int]:
    by_status: dict[str, int] = {}
    for record in records:
        by_status[record.status] = by_status.get(record.status, 0) + 1
    return by_status


def main() -> None:
    """Run conventions metrics measurement and emit deterministic output."""
    parser = argparse.ArgumentParser(
        description="Measure conventions DoD metrics from repository snapshot."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root path (default: current directory).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write JSON report.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    records = measure(repo_root)

    payload: dict[str, Any] = {
        "repo_root": str(repo_root),
        "metrics": [asdict(item) for item in records],
        "summary": _summary(records),
    }

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    print("Conventions metrics snapshot")
    print(f"repo_root: {repo_root}")
    print("-")
    for item in records:
        print(
            f"{item.metric}: value={item.value!r} target={item.target} "
            f"status={item.status}"
        )
    print("-")
    print(f"summary: {payload['summary']}")


if __name__ == "__main__":
    main()
