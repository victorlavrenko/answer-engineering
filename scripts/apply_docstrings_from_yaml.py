#!/usr/bin/env python3
"""Apply curated YAML docstrings to existing Python docstring nodes.

Purpose:
    Replace existing module, class, function, and method docstrings with curated
    text stored in a structured YAML inventory. The script is intended for
    reviewable documentation cleanups where docstrings are edited outside the
    Python source and then applied back in one deterministic batch.

Architectural role:
    Repository-maintenance tool at the scripts boundary. It operates on source
    text and Python AST coordinates, but it does not infer documentation
    content, format prose, or decide which symbols deserve expansion.

Accepted YAML format:
    The YAML root must be a mapping with an `entries` list. Each
    entry must have these fields:

    - `path`: repository-relative Python file path from the provided root.
    - `module`: dotted module name used to interpret qualified names.
    - `qualname`: module-relative or fully qualified target name.
    - `kind`: one of `module`, `class`, `function`, `method`, or
      `async_function`.
    - `lineno`: current line number of the target definition.
    - `docstring`: replacement docstring text without surrounding triple quotes.
    - `is_dummy`: boolean flag; true entries are intentionally skipped.

Inputs (architectural provenance):
    Receives a repository root, a YAML inventory path, and optional safety
    thresholds from the command line. The inventory is normally produced or
    reviewed by maintainers after running docstring inventory tooling.

Outputs (downstream usage):
    Rewrites files in place when valid replacements are found, prints planned,
    skipped, and updated targets, and returns a process status for scripts/check
    or manual maintenance workflows.

Invariants/constraints:
    Only existing docstring literals are replaced. The script deliberately does
    not insert missing docstrings because that would require statement-level
    source editing and stronger formatting guarantees. Replacements are applied
    from the end of each file toward the beginning, then the generated source is
    parsed before writing. Very short replacements are skipped by default to
    reduce accidental documentation loss.

"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict, cast

import yaml

_DEFAULT_MIN_LENGTH_RATIO = 0.8
_DEFAULT_MIN_CHAR_DELTA = 200

DocstringOwner = (
    ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
)
NamedDocstringOwner = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


class DocstringYamlEntry(TypedDict):
    """Validated YAML entry for one docstring replacement target.

    Purpose:
        Describe the exact keys that a parsed YAML entry must contain before it
        can participate in source rewriting.

    Architectural role:
        TypedDict boundary between untrusted YAML data and the replacement
        planner.

    Inputs (architectural provenance):
        Populated by `_parse_entry` after runtime type checks on raw YAML
        values.

    Outputs (downstream usage):
        Consumed by target lookup, replacement construction, grouping by file
        path, and skip reporting.

    Invariants/constraints:
        Values must remain simple YAML-serializable scalars so inventories are
        easy to inspect, diff, and edit manually.

    """

    path: str
    module: str
    qualname: str
    kind: str
    lineno: int
    docstring: str
    is_dummy: bool


def _parse_entry(raw: object) -> DocstringYamlEntry | None:
    if not isinstance(raw, dict):
        return None
    mapping = cast(dict[str, object], raw)

    path = mapping.get("path")
    module = mapping.get("module")
    qualname = mapping.get("qualname")
    kind = mapping.get("kind")
    lineno = mapping.get("lineno")
    docstring = mapping.get("docstring")
    is_dummy = mapping.get("is_dummy", False)

    if not isinstance(path, str):
        return None
    if not isinstance(module, str):
        return None
    if not isinstance(qualname, str):
        return None
    if not isinstance(kind, str):
        return None
    if not isinstance(lineno, int):
        return None
    if not isinstance(docstring, str):
        return None
    if not isinstance(is_dummy, bool):
        return None

    return {
        "path": path,
        "module": module,
        "qualname": qualname,
        "kind": kind,
        "lineno": lineno,
        "docstring": docstring,
        "is_dummy": is_dummy,
    }


def parse_entries(data: object) -> list[DocstringYamlEntry]:
    """Parse YAML root data into validated docstring entries.

    Purpose:
        Extract the `entries` sequence from loaded YAML and discard malformed
        rows before any file I/O or abstract-syntax-tree matching occurs.

    Architectural role:
        Validation boundary between PyYAML output and the source-rewrite
        pipeline.

    Inputs (architectural provenance):
        Receives arbitrary Python data returned by `yaml.safe_load`.

    Outputs (downstream usage):
        Returns only entries that satisfy `DocstringYamlEntry` field
        requirements; `main` groups those entries by repository-relative path.

    Invariants/constraints:
        Invalid root shapes, missing `entries`, non-list entries, and malformed
        rows are ignored rather than partially trusted.

    """
    if not isinstance(data, dict):
        return []
    mapping = cast(dict[str, object], data)
    entries = mapping.get("entries")
    if not isinstance(entries, list):
        return []

    parsed: list[DocstringYamlEntry] = []
    for raw in cast(list[object], entries):
        entry = _parse_entry(raw)
        if entry is not None:
            parsed.append(entry)
    return parsed


def iter_nodes_with_qualnames(
    tree: ast.Module,
) -> Iterator[tuple[str, NamedDocstringOwner]]:
    """Yield class/function nodes with module-relative qualified names.

    Purpose:
        Traverse a parsed Python module and expose every class, function, and
        method node that can own a replacement docstring.

    Architectural role:
        AST inventory helper used by YAML target resolution. It provides the
        qualified-name view shared by the matcher and the reviewed inventory.

    Inputs (architectural provenance):
        Receives an `ast.Module` parsed from the current source file.

    Outputs (downstream usage):
        Yields `(qualname, node)` pairs consumed by `find_target_node`.

    Invariants/constraints:
        Qualified names are module-relative and preserve lexical nesting. The
        traversal intentionally ignores non-docstring-owning statements.

    """

    def visit(
        body: list[ast.stmt],
        prefix: str,
    ) -> Iterator[tuple[str, NamedDocstringOwner]]:
        for node in body:
            if not isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue
            qualname = f"{prefix}.{node.name}" if prefix else node.name
            yield qualname, node
            yield from visit(node.body, qualname)

    yield from visit(tree.body, "")


def find_target_node(
    tree: ast.Module,
    module_name: str,
    qualname: str,
    kind: str,
    lineno: int,
) -> DocstringOwner | None:
    """Locate the abstract-syntax-tree node that matches one YAML entry target.

    Purpose:
        Resolve a validated inventory entry to the exact module, class,
        function, or method node in the current source tree.

    Architectural role:
        Matching boundary between stable YAML metadata and mutable source files.

    Inputs (architectural provenance):
        Receives the parsed module, the inventory module name, the stored
        qualified name, the expected node kind, and the recorded definition line
        number.

    Outputs (downstream usage):
        Returns the matched docstring owner for replacement planning, or `None`
        when the entry no longer describes the current file.

    Invariants/constraints:
        Fully qualified names may include the module prefix; local matching
        strips that prefix before lookup. The line number must still match to
        avoid applying a reviewed docstring to a different symbol after source
        drift.

    """
    if kind == "module":
        return tree

    prefix = module_name + "."
    local_qualname = (
        qualname[len(prefix) :] if qualname.startswith(prefix) else qualname
    )

    for node_qualname, node in iter_nodes_with_qualnames(tree):
        if node_qualname != local_qualname:
            continue

        if kind == "class" and not isinstance(node, ast.ClassDef):
            continue
        if kind in {"function", "method", "async_function"} and not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue

        if node.lineno == lineno:
            return node

    return None


def get_docstring_expr_node(node: DocstringOwner) -> ast.Expr | None:
    """Return the AST expression node that currently owns the docstring.

    Purpose:
        Find the concrete first-statement string literal that Python recognizes
        as a docstring for a module, class, function, or method.

    Architectural role:
        Source-coordinate bridge between semantic AST targets and string-slice
        replacement.

    Inputs (architectural provenance):
        Receives a docstring-capable AST node selected by `find_target_node`.

    Outputs (downstream usage):
        Returns the `ast.Expr` whose value is the existing string literal, or
        `None` when the target has no replaceable docstring.

    Invariants/constraints:
        Only existing docstring literals are eligible. This helper deliberately
        does not insert new statements or treat later string constants as
        docstrings.

    """
    body = node.body
    if not body:
        return None

    first = body[0]
    if not isinstance(first, ast.Expr):
        return None

    value = first.value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return first

    return None


def line_starts(text: str) -> list[int]:
    """Return 0-based start offsets for each line in `text`.

    Purpose:
        Precompute line-start positions so abstract-syntax-tree line/column
        coordinates can be converted into absolute source offsets cheaply and
        deterministically.

    Architectural role:
        Text-coordinate utility in the source-rewrite boundary.

    Inputs (architectural provenance):
        Receives the complete current file text before replacement planning.

    Outputs (downstream usage):
        Returns a list indexed by 1-based source line minus one, consumed by
        `absolute_offset`.

    Invariants/constraints:
        Offsets are based on the exact string being rewritten. The first line
        always starts at zero, including for empty files.

    """
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def absolute_offset(starts: list[int], lineno: int, col_offset: int) -> int:
    """Convert a 1-based line/column pair into an absolute string offset.

    Purpose:
        Translate abstract-syntax-tree source coordinates into offsets suitable
        for Python string slicing.

    Architectural role:
        Coordinate conversion helper used while building a replacement edit.

    Inputs (architectural provenance):
        Receives line starts from `line_starts`, a 1-based line number from the
        abstract-syntax-tree, and a 0-based column offset from the
        abstract-syntax-tree.

    Outputs (downstream usage):
        Returns the absolute offset used as a replacement boundary.

    Invariants/constraints:
        Callers must pass coordinates derived from the same source string used
        to build the line-start table.

    """
    return starts[lineno - 1] + col_offset


def detect_indent_from_original_docstring(
    source_lines: list[str],
    doc_expr: ast.Expr,
) -> str:
    """Extract indentation used by the original docstring literal.

    Purpose:
        Recover the exact leading whitespace that should prefix rendered
        replacement docstring lines.

    Architectural role:
        Formatting-preservation helper between abstract-syntax-tree coordinates
        and source text.

    Inputs (architectural provenance):
        Receives split source lines and the existing docstring expression node.

    Outputs (downstream usage):
        Returns the indentation string passed to `render_docstring`.

    Invariants/constraints:
        The indentation is read from the original literal line, not recomputed
        from abstract-syntax-tree depth, so tabs or unusual but valid source
        spacing are preserved.

    """
    line = source_lines[doc_expr.lineno - 1]
    return line[: doc_expr.col_offset]


def render_docstring(text: str, indent: str) -> str:
    """Render replacement text as a triple-quoted docstring literal.

    Purpose:
        Convert plain YAML docstring text into source code that can replace the
        existing string literal while preserving the target node's indentation.

    Architectural role:
        Formatting boundary between structured inventory data and concrete
        Python source text.

    Inputs (architectural provenance):
        Receives the replacement docstring text from a validated YAML entry and
        the indentation extracted from the original source location.

    Outputs (downstream usage):
        Returns a complete triple-double-quoted literal used by
        `build_replacement_for_entry` as the replacement payload.

    Invariants/constraints:
        Existing triple-double-quote sequences inside replacement text are
        escaped. The first summary line stays on the opening quote line,
        matching the repository's docstring style and ruff D212 expectations.

    """
    text = text.replace('"""', '\\"""')
    lines = text.splitlines()

    if not lines:
        return '""""""'

    rendered: list[str] = [f'"""{lines[0]}']
    for line in lines[1:]:
        rendered.append(f"{indent}{line}")
    rendered.append(f'{indent}"""')
    return "\n".join(rendered)


def _normalized_docstring_length(text: str) -> int:
    return len(text.strip())


def _is_much_shorter_than_original(
    original_docstring: str,
    new_docstring: str,
    *,
    min_length_ratio: float,
    min_char_delta: int,
) -> bool:
    original_length = _normalized_docstring_length(original_docstring)
    new_length = _normalized_docstring_length(new_docstring)

    if original_length == 0:
        return False

    length_ratio = new_length / original_length
    char_delta = original_length - new_length
    return length_ratio < min_length_ratio and char_delta >= min_char_delta


def build_replacement_for_entry(
    source: str,
    source_lines: list[str],
    starts: list[int],
    entry: DocstringYamlEntry,
    *,
    min_length_ratio: float,
    min_char_delta: int,
) -> tuple[int, int, str] | None:
    """Build the source edit needed for one validated YAML entry.

    Purpose:
        Match one inventory entry to its current abstract-syntax-tree target and
        compute the exact source slice that should be replaced by the rendered
        docstring literal.

    Architectural role:
        Core planning step of the docstring-application pipeline.

    Inputs (architectural provenance):
        Receives the current file source, precomputed line offsets, one
        validated YAML entry, and shortening-protection thresholds from
        `apply_file_entries`.

    Outputs (downstream usage):
        Returns a `(start, end, replacement)` tuple consumed by the per-file
        rewrite stage, or `None` when the entry must be skipped.

    Invariants/constraints:
        The target must still exist at the recorded line number and already own
        a docstring expression. Dummy entries, blank replacement text, missing
        targets, missing docstring literals, and suspiciously shorter
        replacements are skipped with explicit console diagnostics.

    """
    if entry["is_dummy"]:
        return None

    new_docstring = entry["docstring"]
    if not new_docstring.strip():
        return None

    tree = ast.parse(source)
    target = find_target_node(
        tree=tree,
        module_name=entry["module"],
        qualname=entry["qualname"],
        kind=entry["kind"],
        lineno=entry["lineno"],
    )
    if target is None:
        print(f"SKIP: target not found: {entry['path']} :: {entry['qualname']}")
        return None

    doc_expr = get_docstring_expr_node(target)
    if doc_expr is None:
        print(
            "SKIP: no existing docstring: "
            f"{entry['path']} :: {entry['qualname']}"
        )
        return None

    original_docstring = ast.get_docstring(target, clean=False) or ""
    if _is_much_shorter_than_original(
        original_docstring,
        new_docstring,
        min_length_ratio=min_length_ratio,
        min_char_delta=min_char_delta,
    ):
        old_len = _normalized_docstring_length(original_docstring)
        new_len = _normalized_docstring_length(new_docstring)
        print(
            "SKIP: replacement docstring is much shorter than original: "
            f"{entry['path']} :: {entry['qualname']} "
            f"(old={old_len}, new={new_len})"
        )
        return None

    value = doc_expr.value
    if not isinstance(value, ast.Constant):
        return None
    indent = detect_indent_from_original_docstring(source_lines, doc_expr)
    replacement = render_docstring(new_docstring, indent)

    if value.end_lineno is None or value.end_col_offset is None:
        print(
            "SKIP: missing source coordinates: "
            f"{entry['path']} :: {entry['qualname']}"
        )
        return None

    start = absolute_offset(starts, value.lineno, value.col_offset)
    end = absolute_offset(starts, value.end_lineno, value.end_col_offset)

    return (start, end, replacement)


def apply_file_entries(
    root: Path,
    rel_path: str,
    entries: list[DocstringYamlEntry],
    *,
    min_length_ratio: float,
    min_char_delta: int,
) -> int:
    """Apply all valid docstring replacements for one source file.

    Purpose:
        Plan, validate, and write the replacements associated with a single
        repository-relative Python file.

    Architectural role:
        File-level mutation boundary of the docstring-application tool.

    Inputs (architectural provenance):
        Receives the repository root, relative file path, entries already
        grouped by path, and global safety thresholds from `main`.

    Outputs (downstream usage):
        Writes the updated source file when replacements parse successfully and
        returns the number of applied replacements for final reporting.

    Invariants/constraints:
        Replacements are sorted in descending source-offset order to avoid
        offset drift. The complete generated file is parsed before writing;
        syntax errors cancel the file update.

    """
    file_path = root / rel_path
    source = file_path.read_text(encoding="utf-8")
    source_lines = source.splitlines(keepends=True)
    starts = line_starts(source)

    replacements: list[tuple[int, int, str]] = []

    for entry in entries:
        try:
            replacement = build_replacement_for_entry(
                source=source,
                source_lines=source_lines,
                starts=starts,
                entry=entry,
                min_length_ratio=min_length_ratio,
                min_char_delta=min_char_delta,
            )
            if replacement is not None:
                replacements.append(replacement)
                print(f"PLANNED: {entry['path']} :: {entry['qualname']}")
        except Exception as exc:
            print(f"ERROR: {entry['path']} :: {entry['qualname']} :: {exc}")

    if not replacements:
        return 0

    replacements.sort(key=lambda item: item[0], reverse=True)

    new_source = source
    for start, end, replacement in replacements:
        new_source = new_source[:start] + replacement + new_source[end:]

    try:
        ast.parse(new_source)
    except SyntaxError as exc:
        print(f"ERROR: {rel_path} :: generated invalid syntax: {exc}")
        return 0

    file_path.write_text(new_source, encoding="utf-8", newline="\n")
    print(f"UPDATED FILE: {rel_path} :: {len(replacements)} replacements")
    return len(replacements)


def main() -> int:
    """Run the command-line docstring application workflow.

    Purpose:
        Parse CLI arguments, load the YAML inventory, validate safety
        thresholds, group entries by source path, and apply all planned
        replacements.

    Architectural role:
        Script entrypoint that connects shell usage to the file-level rewrite
        pipeline.

    Inputs (architectural provenance):
        Reads `sys.argv` in the form `root yaml_file [min_length_ratio]
        [min_char_delta]`.

    Outputs (downstream usage):
        Prints a summary of changed docstrings and returns a process exit
        status.

    Invariants/constraints:
        `min_length_ratio` must be in `(0, 1]` and `min_char_delta` must be
        non-negative. These guards keep accidental large documentation
        regressions visible during maintenance batches.

    """
    if len(sys.argv) not in {3, 4, 5}:
        print(
            "Usage: python apply_docstrings_from_yaml.py <root> <yaml_file> "
            "[min_length_ratio] [min_char_delta]"
        )
        return 1

    root = Path(sys.argv[1]).resolve()
    yaml_file = Path(sys.argv[2]).resolve()
    min_length_ratio = (
        float(sys.argv[3]) if len(sys.argv) >= 4 else _DEFAULT_MIN_LENGTH_RATIO
    )
    min_char_delta = (
        int(sys.argv[4]) if len(sys.argv) >= 5 else _DEFAULT_MIN_CHAR_DELTA
    )

    if not 0 < min_length_ratio <= 1:
        print("min_length_ratio must be in the interval (0, 1].")
        return 1
    if min_char_delta < 0:
        print("min_char_delta must be >= 0.")
        return 1

    raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    entries = parse_entries(raw)

    entries_by_path: defaultdict[str, list[DocstringYamlEntry]] = defaultdict(
        list
    )
    for entry in entries:
        entries_by_path[entry["path"]].append(entry)

    changed = 0
    for rel_path, file_entries in entries_by_path.items():
        changed += apply_file_entries(
            root,
            rel_path,
            file_entries,
            min_length_ratio=min_length_ratio,
            min_char_delta=min_char_delta,
        )

    print(
        f"\nDone. Updated {changed} docstrings. "
        "(min_length_ratio="
        f"{min_length_ratio}, "
        f"min_char_delta={min_char_delta})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
