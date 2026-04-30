#!/usr/bin/env python3
"""Extract Python docstrings into a reviewable YAML inventory.

Purpose:
    Scan repository Python files, collect existing docstrings with stable target
    metadata, and write an inventory that maintainers can review, edit, and pass
    to `apply_docstrings_from_yaml.py`.

Architectural role:
    Repository-maintenance tool at the scripts boundary. It observes source
    structure and emits review data; it does not decide final documentation
    content.

Inputs (architectural provenance):
    Receives file or directory paths, filtering options, and dummy- docstring
    patterns from command-line arguments.

Outputs (downstream usage):
    Writes YAML entries containing path, module, qualified name, kind, line
    number, extracted docstring text, and dummy classification for later manual
    editing or automated replacement.

Invariants/constraints:
    Extraction must preserve enough target identity for safe replacement after
    review. It should not rewrite Python files or silently invent docstrings for
    targets that do not already own one.

"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml

DocstringOwner = (
    ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
)
NamedDocstringOwner = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
EntryKind = Literal["module", "class", "function", "async_function"]


@dataclass(frozen=True, slots=True)
class ExtractedDocstring:
    """Serializable record for one extracted module/class/function docstring."""

    id: str
    path: str
    module: str
    qualname: str
    kind: EntryKind
    lineno: int
    end_lineno: int | None
    is_public: bool
    is_dummy: bool
    dummy_reasons: tuple[str, ...]
    docstring: str


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Runtime configuration for extraction filters and dummy classification."""

    root: Path
    output: Path
    include_path_regex: str | None
    exclude_path_regex: str | None
    module_regex: str | None
    dummy_only: bool
    public_only: bool
    include_without_docstring: bool
    min_dummy_length: int
    dummy_patterns: tuple[str, ...]


class DocstringExtractor:
    """Walk Python files and emit structured docstring extraction records."""

    def __init__(self, config: ExtractionConfig) -> None:
        """Store extraction config and precompile regex filters."""
        self._config = config
        self._include_path_re = (
            re.compile(config.include_path_regex)
            if config.include_path_regex
            else None
        )
        self._exclude_path_re = (
            re.compile(config.exclude_path_regex)
            if config.exclude_path_regex
            else None
        )
        self._module_re = (
            re.compile(config.module_regex) if config.module_regex else None
        )
        self._dummy_res = tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in config.dummy_patterns
        )

    def extract(self) -> list[ExtractedDocstring]:
        """Scan configured files and return extracted docstring records."""
        entries: list[ExtractedDocstring] = []
        py_files = sorted(self._iter_python_files(self._config.root))
        counter = 1

        for file_path in py_files:
            rel_path = file_path.relative_to(self._config.root)
            path_str = rel_path.as_posix()

            if self._include_path_re and not self._include_path_re.search(
                path_str
            ):
                continue
            if self._exclude_path_re and self._exclude_path_re.search(path_str):
                continue

            module_name = self._path_to_module(rel_path)
            if self._module_re and not self._module_re.search(module_name):
                continue

            try:
                source = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                print(
                    f"Skipping unreadable file (encoding): {file_path}: {exc}",
                    file=sys.stderr,
                )
                continue
            except OSError as exc:
                print(
                    f"Skipping unreadable file: {file_path}: {exc}",
                    file=sys.stderr,
                )
                continue

            try:
                tree = ast.parse(source, filename=str(file_path))
            except SyntaxError as exc:
                print(
                    f"Skipping unparsable file: {file_path}: {exc}",
                    file=sys.stderr,
                )
                continue

            module_entry = self._build_entry(
                node=tree,
                file_path=path_str,
                module_name=module_name,
                qualname=module_name,
                kind="module",
                stable_index=counter,
            )
            if module_entry is not None:
                entries.append(module_entry)
                counter += 1

            visitor = _QualifiedNodeVisitor(
                file_path=path_str,
                module_name=module_name,
                build_entry=self._build_entry,
                starting_index=counter,
            )
            visitor.visit(tree)
            entries.extend(visitor.entries)
            counter = visitor.next_index

        return entries

    def _iter_python_files(self, root: Path) -> Iterable[Path]:
        yield from root.rglob("*.py")

    def _path_to_module(self, rel_path: Path) -> str:
        parts = list(rel_path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _build_entry(
        self,
        *,
        node: DocstringOwner,
        file_path: str,
        module_name: str,
        qualname: str,
        kind: EntryKind,
        stable_index: int,
    ) -> ExtractedDocstring | None:
        raw_docstring = ast.get_docstring(node, clean=False)
        if raw_docstring is None and not self._config.include_without_docstring:
            return None

        docstring = raw_docstring or ""
        is_public = self._is_public_qualname(qualname, kind)
        is_dummy, dummy_reasons = self._classify_dummy(docstring)

        if self._config.public_only and not is_public:
            return None
        if self._config.dummy_only and not is_dummy:
            return None

        lineno = getattr(node, "lineno", 1)
        end_lineno = getattr(node, "end_lineno", None)

        return ExtractedDocstring(
            id=f"DOC{stable_index:06d}",
            path=file_path,
            module=module_name,
            qualname=qualname,
            kind=kind,
            lineno=lineno,
            end_lineno=end_lineno,
            is_public=is_public,
            is_dummy=is_dummy,
            dummy_reasons=dummy_reasons,
            docstring=docstring,
        )

    def _is_public_qualname(self, qualname: str, kind: str) -> bool:
        if kind == "module":
            return True
        last_part = qualname.split(".")[-1]
        return not last_part.startswith("_")

    def _classify_dummy(self, docstring: str) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        stripped = docstring.strip()

        if not stripped:
            return True, ("empty",)

        if len(stripped) <= self._config.min_dummy_length:
            reasons.append("very_short")

        normalized = re.sub(r"\s+", " ", stripped).strip().lower()

        exact_dummy_values = {
            "todo",
            "tbd",
            "fixme",
            "placeholder",
            "dummy",
            "dummy docstring",
            "temp",
            "temporary",
            "description",
            "summary",
            "docstring",
            "not implemented",
        }
        if normalized in exact_dummy_values:
            reasons.append("exact_dummy_value")

        matched_pattern_strings: list[str] = []
        for pattern in self._dummy_res:
            if pattern.search(stripped):
                matched_pattern_strings.append(pattern.pattern)
                reasons.append(f"pattern:{pattern.pattern}")

        weak_starts = (
            "todo:",
            "tbd:",
            "fixme:",
            "placeholder:",
            "dummy:",
        )
        if normalized.startswith(weak_starts):
            reasons.append("dummy_prefix")

        boilerplate_headers = (
            "purpose:",
            "architectural role:",
            "inputs (architectural provenance):",
            "outputs (downstream usage):",
        )
        matched_headers = sum(
            1 for header in boilerplate_headers if header in normalized
        )
        if matched_headers >= 3:
            reasons.append("template_contract_structure")

        if "internal contract" in normalized:
            reasons.append("internal_contract_template")

        strong_reason_keys = {
            "exact_dummy_value",
            "dummy_prefix",
            "internal_contract_template",
        }
        strong_pattern_fragments = (
            "internal contract",
            "implement one coherent behavior unit",
            "adjacent pipeline collaborators",
            "downstream runtime/scoring/reporting stages",
            "write me",
            "fill me",
            "lorem ipsum",
        )

        has_strong_reason = any(
            reason in strong_reason_keys for reason in reasons
        )
        has_strong_pattern = any(
            any(
                fragment in pattern_text.lower()
                for fragment in strong_pattern_fragments
            )
            for pattern_text in matched_pattern_strings
        )

        # A structured docstring is not dummy merely because it uses the
        # project template. Treat template shape as a warning signal only and
        # require a stronger placeholder indicator before classifying it as
        # dummy.
        if has_strong_reason or has_strong_pattern:
            return True, tuple(dict.fromkeys(reasons))

        if reasons == ["very_short"]:
            return True, ("very_short",)

        return False, tuple(dict.fromkeys(reasons))


class _QualifiedNodeVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        file_path: str,
        module_name: str,
        build_entry: Callable[..., ExtractedDocstring | None],
        starting_index: int,
    ) -> None:
        self._file_path = file_path
        self._module_name = module_name
        self._build_entry = build_entry
        self._stack: list[str] = []
        self.entries: list[ExtractedDocstring] = []
        self.next_index = starting_index

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._push(node.name)
        try:
            self._maybe_add(node=node, kind="class")
            self.generic_visit(node)
        finally:
            self._pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._push(node.name)
        try:
            self._maybe_add(node=node, kind="function")
            self.generic_visit(node)
        finally:
            self._pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._push(node.name)
        try:
            self._maybe_add(node=node, kind="async_function")
            self.generic_visit(node)
        finally:
            self._pop()

    def _push(self, name: str) -> None:
        self._stack.append(name)

    def _pop(self) -> None:
        self._stack.pop()

    def _qualname(self) -> str:
        if not self._stack:
            return self._module_name
        return ".".join([self._module_name, *self._stack])

    def _maybe_add(self, *, node: NamedDocstringOwner, kind: EntryKind) -> None:
        entry = self._build_entry(
            node=node,
            file_path=self._file_path,
            module_name=self._module_name,
            qualname=self._qualname(),
            kind=kind,
            stable_index=self.next_index,
        )
        if entry is not None:
            self.entries.append(entry)
            self.next_index += 1


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Python docstrings into one YAML file."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory of the Python codebase to scan.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("docstrings_review.yaml"),
        help="Output YAML path.",
    )
    parser.add_argument(
        "--include-path-regex",
        type=str,
        default=None,
        help="Only include files whose relative path matches this regex.",
    )
    parser.add_argument(
        "--exclude-path-regex",
        type=str,
        default=r"(^|/)(\.venv|venv|build|dist|site-packages|\.git|__pycache__)(/|$)",
        help="Exclude files whose relative path matches this regex.",
    )
    parser.add_argument(
        "--module-regex",
        type=str,
        default=None,
        help="Only include entries whose module name matches this regex.",
    )
    parser.add_argument(
        "--dummy-only",
        action="store_true",
        help="Only emit entries classified as dummy/weak.",
    )
    parser.add_argument(
        "--public-only",
        action="store_true",
        help="Only emit public classes/functions/methods.",
    )
    parser.add_argument(
        "--include-without-docstring",
        action="store_true",
        help="Also include objects with no docstring at all.",
    )
    parser.add_argument(
        "--min-dummy-length",
        type=int,
        default=12,
        help="Docstrings at or below this length are flagged as dummy-like.",
    )
    parser.add_argument(
        "--dummy-pattern",
        action="append",
        default=[],
        help=(
            "Regex for dummy detection. Can be passed multiple times. "
            "Example: --dummy-pattern '^TODO' --dummy-pattern 'placeholder'"
        ),
    )
    return parser


def _default_dummy_patterns() -> tuple[str, ...]:
    return (
        r"^\s*todo\b",
        r"^\s*tbd\b",
        r"^\s*fixme\b",
        r"\bplaceholder\b",
        r"\bdummy\b",
        r"\bwrite me\b",
        r"\bfill me\b",
        r"\blorem ipsum\b",
        r"\binternal contract\b",
        r"\bimplement one coherent behavior unit\b",
        r"\badjacent pipeline collaborators\b",
        r"\bdownstream runtime/scoring/reporting stages\b",
    )


def _normalize_docstring_for_review(docstring: str) -> str:
    """Normalize formatting for YAML review without heavy semantic edits."""
    lines = docstring.splitlines()
    lines = [line.rstrip() for line in lines]

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)


def _entry_to_yaml_dict(entry: ExtractedDocstring) -> dict[str, object]:
    data = asdict(entry)
    docstring = _normalize_docstring_for_review(entry.docstring)
    data["docstring"] = docstring
    data["dummy_reasons"] = list(data["dummy_reasons"])
    return data


def _write_yaml(
    output_path: Path,
    entries: list[ExtractedDocstring],
    root: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "root": str(root.resolve()),
        "count": len(entries),
        "entries": [_entry_to_yaml_dict(entry) for entry in entries],
    }

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            payload,
            handle,
            sort_keys=False,
            allow_unicode=True,
            width=80,
            default_flow_style=False,
        )


def main() -> int:
    """CLI entrypoint for extracting docstrings into YAML."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"Root is not a directory: {root}", file=sys.stderr)
        return 2

    dummy_patterns = tuple(args.dummy_pattern) or _default_dummy_patterns()

    config = ExtractionConfig(
        root=root,
        output=args.output.resolve(),
        include_path_regex=args.include_path_regex,
        exclude_path_regex=args.exclude_path_regex,
        module_regex=args.module_regex,
        dummy_only=args.dummy_only,
        public_only=args.public_only,
        include_without_docstring=args.include_without_docstring,
        min_dummy_length=args.min_dummy_length,
        dummy_patterns=dummy_patterns,
    )

    extractor = DocstringExtractor(config)
    entries = extractor.extract()
    _write_yaml(config.output, entries, root)

    print(f"Wrote {len(entries)} entries to {config.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
