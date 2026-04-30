#!/usr/bin/env python3
"""Normalize and reflow Python docstrings to repository style.

Purpose:
    Rewrite existing docstring literals so summaries, section bodies, bullets,
    and wrapped prose follow the repository's line- length and formatting
    expectations.

Architectural role:
    Repository-maintenance formatter for docstring text. It complements ruff by
    handling project-specific documentation shape that a generic formatter does
    not fully own.

Inputs (architectural provenance):
    Receives Python files or directories, target width, and rewrite/check flags
    from command-line arguments.

Outputs (downstream usage):
    Rewrites source files or reports files that would change, enabling manual
    cleanup and CI-style docstring formatting checks.

Invariants/constraints:
    The script operates only on existing docstrings. It should preserve Python
    parseability, keep the summary on the opening triple-quote line, and avoid
    flattening structured sections into ambiguous prose.

"""

from __future__ import annotations

import ast
import re
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

DEFAULT_WIDTH = 80

BULLET_RE = re.compile(r"^(\s*)([-*+])\s+(.*\S)\s*$")
ENUM_RE = re.compile(r"^(\s*)(\d+[.)]|[a-zA-Z][.)])\s+(.*\S)\s*$")
UNDERLINE_RE = re.compile(r"^\s*([=\-~`:#^\"'])\1{2,}\s*$")
COLLAPSED_HEADING_RE = re.compile(
    r"^(?P<indent>\s*)(?P<title>.+?\S)\s+(?P<underline>[-=~`:#^\"']{3,})\s+(?P<rest>\S.*)$"
)
LABEL_RE = re.compile(r"^\s*[^:\n]{1,60}:\s*$")
FIELD_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*):\s+(.+\S)\s*$")
FIELD_SECTION_LABELS = frozenset(
    {
        "args",
        "arguments",
        "parameters",
        "keyword args",
        "keyword arguments",
        "other parameters",
        "attributes",
        "raises",
        "warns",
        "yields",
        "methods",
        "consumes",
        "produces",
        "see also",
        "returns",
        "developer notes",
    }
)

REST_FIELD_RE = re.compile(
    r"^(\s*)"
    r"("
    r"(?::(?:meth|class|func|mod|attr|data|exc|obj):`[^`]+`)"
    r"(?:\s+\([^)]+\))?"
    r")"
    r"\s*$"
)
CODE_SECTION_LABELS = frozenset(
    {
        "cli",
        "configuration",
        "example",
        "examples",
        "snippet",
        "typical use",
        "usage",
    }
)

FENCE_RE = re.compile(r"^(?P<indent>\s*)```(?P<info>[^`\n]*)$")
COLLAPSED_FENCE_RE = re.compile(
    r"^(?P<indent>\s*)```(?P<info>[A-Za-z0-9_.+\-]*)\s+(?P<rest>\S.*)$"
)


def iter_py_files(path: Path) -> Iterator[Path]:
    """Yield Python files from a file path or recursively from a directory."""
    if path.is_file():
        if path.suffix == ".py":
            yield path
        return
    yield from path.rglob("*.py")


def get_docstring_expr(node: ast.AST) -> ast.Expr | None:
    """Return the first body expression when it is a string docstring node."""
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if not isinstance(first, ast.Expr):
        return None
    value = first.value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    return first


def leading_ws(s: str) -> str:
    """Return the leading whitespace prefix of `s`."""
    return s[: len(s) - len(s.lstrip())]


def is_bullet(line: str) -> bool:
    """Return whether a line starts with a bullet or enumerated marker."""
    return bool(BULLET_RE.match(line) or ENUM_RE.match(line))


def is_underline(line: str) -> bool:
    """Return whether a line looks like a heading underline delimiter."""
    stripped = line.strip()
    if stripped.startswith("```"):
        return False
    return bool(UNDERLINE_RE.match(line))


def is_label(line: str) -> bool:
    """Return whether a line looks like a section label ending in `:`."""
    return bool(LABEL_RE.match(line))


def is_field_entry(line: str) -> bool:
    """Return whether a line looks like a structured section entry."""
    return bool(FIELD_RE.match(line) or REST_FIELD_RE.match(line))


def is_field_section_label(line: str) -> bool:
    """Return whether a label conventionally contains field entries."""
    stripped = line.strip()
    if not stripped.endswith(":"):
        return False
    return stripped[:-1].lower() in FIELD_SECTION_LABELS


def label_name(line: str) -> str:
    """Return a normalized label name without the trailing colon."""
    stripped = line.strip()
    return stripped[:-1].lower() if stripped.endswith(":") else stripped.lower()


def is_code_section_label(line: str) -> bool:
    """Return whether a label conventionally introduces literal code."""
    return is_label(line) and label_name(line) in CODE_SECTION_LABELS


def is_fence(line: str) -> bool:
    """Return whether a line is a standalone Markdown code fence."""
    return bool(FENCE_RE.match(line.rstrip()))


def starts_fence(line: str) -> bool:
    """Return whether a line starts a Markdown fenced code block."""
    stripped = line.lstrip()
    return stripped.startswith("```")


def repair_literal_code_lines(lines: list[str]) -> list[str]:
    """Repair only fence-marker corruption inside literal code sections.

    The formatter must never prose-wrap fenced examples.  This helper handles
    the common damaged form produced by older wrapper versions::

        "```python" joined with "result = call(...)"

    by splitting it back into the fence opener and the first code line.  It does
    not attempt to pretty-print or reflow code inside the fence.

    """
    repaired: list[str] = []
    for line in lines:
        match = COLLAPSED_FENCE_RE.match(line.rstrip())
        if match:
            indent = match.group("indent")
            info = match.group("info")
            repaired.append(f"{indent}```{info}".rstrip())
            repaired.append(f"{indent}{match.group('rest')}")
            continue
        repaired.append(line.rstrip())
    return repaired


def append_literal_lines(out: list[str], literal_lines: list[str]) -> None:
    """Append literal lines preserving blank lines and existing indentation."""
    for raw in repair_literal_code_lines(literal_lines):
        if raw.strip():
            out.append(raw.rstrip())
        elif out and out[-1] != "":
            out.append("")


def collect_fenced_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Collect a Markdown fenced code block without wrapping its contents."""
    block = [lines[start].rstrip()]
    i = start + 1
    while i < len(lines):
        cur = lines[i].rstrip()
        block.append(cur)
        if is_fence(cur):
            i += 1
            break
        i += 1
    return repair_literal_code_lines(block), i


def is_literal_code_section(label: str, body: list[str]) -> bool:
    """Return whether a labeled section should bypass prose wrapping.

    Purpose:
        Protect explicitly code-oriented docstring sections from being flattened
        into prose.

    Architectural role:
        Parser guard used before ordinary label-body wrapping.

    Inputs (architectural provenance):
        Receives one label and its raw collected body from `parse_lines` or the
        reporting pass.

    Outputs (downstream usage):
        Returns true only for labels that are semantically intended to contain
        examples or snippets.

    Invariants/constraints:
        This predicate is intentionally label-based. General prose sections such
        as `Purpose:`, `Inputs:`, and `Notes:` may contain dotted names, inline
        code, parentheses, or words like "return" without becoming literal code
        sections.

    """
    del body
    if not is_label(label) or is_field_section_label(label):
        return False
    return is_code_section_label(label)


def is_directive(line: str) -> bool:
    """Return whether a line starts or ends a directive-style literal block."""
    stripped = line.strip()
    return (
        stripped.endswith("::")
        or stripped.startswith(">>>")
        or stripped.startswith("... ")
    )


def normalize_docstring_lines(docstring: str) -> list[str]:
    """Normalize indentation and blank-line boundaries in docstring text."""
    raw_lines = docstring.expandtabs().splitlines()

    while raw_lines and not raw_lines[0].strip():
        raw_lines.pop(0)
    while raw_lines and not raw_lines[-1].strip():
        raw_lines.pop()

    if not raw_lines:
        return []

    indents = [len(leading_ws(line)) for line in raw_lines[1:] if line.strip()]
    common = min(indents) if indents else 0

    normalized = [raw_lines[0].strip()]
    for line in raw_lines[1:]:
        if not line.strip():
            normalized.append("")
        else:
            normalized.append(line[common:].rstrip())

    return repair_structure(normalized)


def split_collapsed_bullet_runs(text: str) -> list[str]:
    """Split run-on bullet text into separate bullet-item fragments."""
    parts = [text.strip()]
    changed = True
    while changed:
        changed = False
        new_parts: list[str] = []
        for part in parts:
            split = re.split(r"(?<=[.!?])\s+-\s+(?=[A-Z`\"'(])", part)
            if len(split) == 1:
                split = re.split(r"\s+-\s+(?=[A-Za-z0-9_`\"'(])", part)
            if len(split) == 1:
                split = re.split(r"\s+-(?=[a-z])", part)
            if len(split) > 1:
                changed = True
                new_parts.extend(s.strip() for s in split if s.strip())
            else:
                new_parts.append(part.strip())
        parts = new_parts
    return parts


def repair_structure(lines: list[str]) -> list[str]:
    """Repair collapsed heading or bullet structure before wrapping.

    Purpose:
        Recover common docstring structures that were flattened into a single
        line before normal wrapping occurs.

    Architectural role:
        Pre-parser cleanup stage for the docstring formatting tool.

    Inputs (architectural provenance):
        Receives normalized raw docstring lines extracted from source.

    Outputs (downstream usage):
        Returns structurally repaired lines for `parse_lines`.

    Invariants/constraints:
        Repairs should be conservative. The function may split obvious collapsed
        labels or bullets, but it should not invent semantic sections that are
        not present in the text.

    """
    repaired: list[str] = []

    for line in lines:
        m = COLLAPSED_HEADING_RE.match(line)
        if m:
            indent = m.group("indent")
            repaired.append(f"{indent}{m.group('title').rstrip()}")
            repaired.append(f"{indent}{m.group('underline')}")
            line = f"{indent}{m.group('rest').lstrip()}"

        if is_bullet(line):
            indent = leading_ws(line)
            marker_match = BULLET_RE.match(line) or ENUM_RE.match(line)
            assert marker_match is not None
            marker = marker_match.group(2)
            text = marker_match.group(3)
            items = split_collapsed_bullet_runs(text)
            for item in items:
                repaired.append(f"{indent}{marker} {item}")
            continue

        repaired.append(line)

    return repaired


def wrap_text(
    text: str,
    width: int,
    initial_indent: str = "",
    subsequent_indent: str | None = None,
) -> list[str]:
    """Wrap one text block with stable indentation and width settings."""
    if subsequent_indent is None:
        subsequent_indent = initial_indent
    return textwrap.fill(
        text,
        width=width,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    ).splitlines()


def wrap_plain_paragraph(lines: list[str], width: int) -> list[str]:
    """Wrap a plain paragraph assembled from multiple source lines."""
    text = " ".join(line.strip() for line in lines if line.strip())
    indent = leading_ws(lines[0]) if lines else ""
    return wrap_text(text, width, indent, indent) if text else [""]


def wrap_bullet_item(
    first_line: str, continuation_lines: list[str], width: int
) -> list[str]:
    """Wrap one bullet item while preserving marker indentation.

    Purpose:
        Reflow long bullet text without changing the bullet marker or visual
        nesting used by the original docstring.

    Architectural role:
        Bullet-specific formatting helper in the docstring wrapper.

    Inputs (architectural provenance):
        Receives one bullet line or repaired bullet fragment from body parsing.

    Outputs (downstream usage):
        Returns wrapped bullet lines consumed by the final docstring renderer.

    Invariants/constraints:
        Continuation lines align under the bullet content rather than under
        column zero, preserving readable list structure.

    """
    m = BULLET_RE.match(first_line) or ENUM_RE.match(first_line)
    if not m:
        return wrap_plain_paragraph([first_line, *continuation_lines], width)

    indent, marker, text0 = m.groups()
    text_parts = [
        text0.strip(),
        *[line.strip() for line in continuation_lines if line.strip()],
    ]
    text = " ".join(text_parts)
    prefix = f"{indent}{marker} "
    items = split_collapsed_bullet_runs(text)
    out: list[str] = []
    for item in items:
        out.extend(wrap_text(item, width, prefix, " " * len(prefix)))
    return out


def wrap_field_entry(
    first_line: str, continuation_lines: list[str], width: int
) -> list[str]:
    """Wrap one structured section entry without merging sibling entries."""
    m = FIELD_RE.match(first_line)
    if m:
        indent, name, text0 = m.groups()
        text_parts = [
            text0.strip(),
            *[line.strip() for line in continuation_lines if line.strip()],
        ]
        text = " ".join(text_parts)
        prefix = f"{indent}{name}: "
        return wrap_text(text, width, prefix, f"{indent}    ")

    rest_m = REST_FIELD_RE.match(first_line)
    if rest_m:
        indent, ref = rest_m.groups()
        out = [f"{indent}{ref}"]

        text = " ".join(
            line.strip() for line in continuation_lines if line.strip()
        )
        if text:
            out.extend(
                wrap_text(
                    text,
                    width,
                    initial_indent=f"{indent}    ",
                    subsequent_indent=f"{indent}    ",
                )
            )

        return out

    return wrap_plain_paragraph([first_line, *continuation_lines], width)


def wrap_label_paragraph(lines: list[str], width: int) -> list[str]:
    """Wrap paragraph content that belongs under a label header.

    Purpose:
        Format prose associated with section labels such as `Purpose:` or
        `Inputs:` while preserving the label boundary.

    Architectural role:
        Paragraph-formatting helper for convention-style docstring sections.

    Inputs (architectural provenance):
        Receives label-adjacent lines detected by `parse_lines`.

    Outputs (downstream usage):
        Returns wrapped lines that remain visually nested under the label.

    Invariants/constraints:
        The label itself must stay distinct from the prose it introduces.
        Wrapping should not collapse the section into an ordinary paragraph.

    """
    indent = leading_ws(lines[0]) if lines else "    "
    text = " ".join(line.strip() for line in lines if line.strip())
    return wrap_text(text, width, indent, indent)


def parse_lines(lines: list[str], width: int) -> list[str]:
    """Parse normalized lines into wrapped output blocks.

    Purpose:
        Classify docstring body lines and route them to the correct wrapping
        logic.

    Architectural role:
        Structural parser inside the docstring wrapping tool.

    Inputs (architectural provenance):
        Receives normalized docstring lines after indentation cleanup and
        structure repair.

    Outputs (downstream usage):
        Returns wrapped lines consumed by `reflow_docstring_content`.

    Invariants/constraints:
        Label sections, bullets, heading underlines, and directive-style blocks
        are handled separately from plain prose so semantic structure is not
        flattened.

    """
    out: list[str] = []
    i = 0

    def emit_blank() -> None:
        if out and out[-1] != "":
            out.append("")

    def collect_until_boundary(
        start: int, *, stop_on_top_level_label: bool
    ) -> tuple[list[str], int]:
        body: list[str] = []
        j = start
        while j < len(lines):
            cur = lines[j]

            if not cur.strip():
                k = j + 1
                while k < len(lines) and not lines[k].strip():
                    k += 1
                if k >= len(lines):
                    break
                if (
                    stop_on_top_level_label
                    and is_label(lines[k])
                    and leading_ws(lines[k]) == ""
                    and (not body or body[-1] == "")
                ):
                    break
                if not body or body[-1] != "":
                    body.append("")
                j += 1
                continue

            if j + 1 < len(lines) and is_underline(lines[j + 1]):
                break
            if is_underline(cur) or is_directive(cur):
                break
            if (
                stop_on_top_level_label
                and is_label(cur)
                and leading_ws(cur) == ""
                and (not body or body[-1] == "")
            ):
                break

            body.append(cur.rstrip())
            j += 1

        while body and body[-1] == "":
            body.pop()
        return body, j

    def normalize_label_body(body: list[str]) -> list[str]:
        if not body:
            return []

        result: list[str] = []
        body_indents = [len(leading_ws(line)) for line in body if line.strip()]
        min_body_indent = min(body_indents) if body_indents else 0

        for line in body:
            if not line.strip():
                result.append("")
                continue

            indent = len(leading_ws(line))
            stripped = line.strip()
            rel_indent = max(0, indent - min_body_indent)
            base_indent = 4
            indent_step = ((rel_indent + 3) // 4) * 4
            effective_indent = base_indent + indent_step
            result.append((" " * effective_indent) + stripped)

        return result

    def collect_field_continuation(
        block_lines: list[str], start: int, first: str, parent_indent_len: int
    ) -> tuple[list[str], int]:
        cont: list[str] = []
        j = start
        first_indent = leading_ws(first)
        while j < len(block_lines):
            nxt = block_lines[j]
            if not nxt.strip():
                break
            if is_field_entry(nxt) and leading_ws(nxt) == first_indent:
                break
            if is_bullet(nxt) and leading_ws(nxt) == first_indent:
                break
            if is_label(nxt) and len(leading_ws(nxt)) <= parent_indent_len:
                break
            if j + 1 < len(block_lines) and is_underline(block_lines[j + 1]):
                break
            if is_underline(nxt) or is_directive(nxt):
                break
            cont.append(nxt.rstrip())
            j += 1
        return cont, j

    while i < len(lines):
        line = lines[i]

        if not line.strip():
            emit_blank()
            i += 1
            continue

        next_line = lines[i + 1] if i + 1 < len(lines) else None

        if next_line is not None and is_underline(next_line):
            out.append(line.rstrip())
            out.append(next_line.rstrip())
            i += 2
            continue

        if is_underline(line) or is_directive(line):
            out.append(line.rstrip())
            i += 1
            continue

        if starts_fence(line):
            block, i = collect_fenced_block(lines, i)
            append_literal_lines(out, block)
            continue

        if is_label(line):
            if out and out[-1] != "":
                out.append("")
            out.append(line.rstrip())
            section_allows_fields = is_field_section_label(line)
            i += 1

            if i < len(lines) and not lines[i].strip():
                emit_blank()
                i += 1
                continue

            body, i = collect_until_boundary(i, stop_on_top_level_label=True)

            if is_literal_code_section(line, body):
                append_literal_lines(out, body)
                continue

            normalized_body = normalize_label_body(body)

            j = 0
            while j < len(normalized_body):
                cur = normalized_body[j]

                if not cur.strip():
                    emit_blank()
                    j += 1
                    continue

                if starts_fence(cur):
                    block, j = collect_fenced_block(normalized_body, j)
                    append_literal_lines(out, block)
                    continue

                cur_next = (
                    normalized_body[j + 1]
                    if j + 1 < len(normalized_body)
                    else None
                )

                if cur_next is not None and is_underline(cur_next):
                    out.append(cur.rstrip())
                    out.append(cur_next.rstrip())
                    j += 2
                    continue

                if is_label(cur):
                    out.append(cur.rstrip())
                    j += 1
                    nested_body: list[str] = []
                    while j < len(normalized_body):
                        nxt = normalized_body[j]
                        if not nxt.strip():
                            break
                        if is_label(nxt) and len(leading_ws(nxt)) == len(
                            leading_ws(cur)
                        ):
                            break
                        if j + 1 < len(normalized_body) and is_underline(
                            normalized_body[j + 1]
                        ):
                            break
                        nested_body.append(nxt.rstrip())
                        j += 1

                    k = 0
                    while k < len(nested_body):
                        first = nested_body[k]
                        if not first.strip():
                            emit_blank()
                            k += 1
                            continue
                        if section_allows_fields and is_field_entry(first):
                            k += 1
                            cont, k = collect_field_continuation(
                                nested_body, k, first, len(leading_ws(cur))
                            )
                            out.extend(wrap_field_entry(first, cont, width))
                        elif is_bullet(first):
                            cont: list[str] = []
                            k += 1
                            while k < len(nested_body):
                                nxt = nested_body[k]
                                if not nxt.strip():
                                    break
                                if is_bullet(nxt) and leading_ws(
                                    nxt
                                ) == leading_ws(first):
                                    break
                                if (
                                    section_allows_fields
                                    and is_field_entry(nxt)
                                    and leading_ws(nxt) == leading_ws(first)
                                ):
                                    break
                                if is_label(nxt) and len(
                                    leading_ws(nxt)
                                ) <= len(leading_ws(cur)):
                                    break
                                cont.append(nxt.rstrip())
                                k += 1
                            out.extend(wrap_bullet_item(first, cont, width))
                        else:
                            para = [first]
                            k += 1
                            while k < len(nested_body):
                                nxt = nested_body[k]
                                if not nxt.strip():
                                    break
                                if is_bullet(nxt) or (
                                    section_allows_fields
                                    and is_field_entry(nxt)
                                ):
                                    break
                                if is_label(nxt) and len(
                                    leading_ws(nxt)
                                ) <= len(leading_ws(cur)):
                                    break
                                para.append(nxt.rstrip())
                                k += 1
                            out.extend(wrap_label_paragraph(para, width))
                    continue

                if section_allows_fields and is_field_entry(cur):
                    first = cur
                    j += 1
                    cont, j = collect_field_continuation(
                        normalized_body, j, first, 0
                    )
                    out.extend(wrap_field_entry(first, cont, width))
                    continue

                if is_bullet(cur):
                    first = cur
                    j += 1
                    cont: list[str] = []
                    while j < len(normalized_body):
                        nxt = normalized_body[j]
                        if not nxt.strip():
                            break
                        if is_bullet(nxt) and leading_ws(nxt) == leading_ws(
                            first
                        ):
                            break
                        if (
                            section_allows_fields
                            and is_field_entry(nxt)
                            and leading_ws(nxt) == leading_ws(first)
                        ):
                            break
                        if is_label(nxt) and len(leading_ws(nxt)) <= len(
                            leading_ws(cur)
                        ):
                            break
                        cont.append(nxt.rstrip())
                        j += 1
                    out.extend(wrap_bullet_item(first, cont, width))
                    continue

                para = [cur]
                j += 1
                while j < len(normalized_body):
                    nxt = normalized_body[j]
                    if not nxt.strip():
                        break
                    if (
                        is_bullet(nxt)
                        or starts_fence(nxt)
                        or (section_allows_fields and is_field_entry(nxt))
                        or is_label(nxt)
                    ):
                        break
                    if j + 1 < len(normalized_body) and is_underline(
                        normalized_body[j + 1]
                    ):
                        break
                    para.append(nxt.rstrip())
                    j += 1
                out.extend(wrap_label_paragraph(para, width))

            continue

        if is_bullet(line):
            first = line.rstrip()
            i += 1
            cont: list[str] = []
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    break
                if (
                    is_bullet(nxt)
                    or is_label(nxt)
                    or starts_fence(nxt)
                    or (i + 1 < len(lines) and is_underline(lines[i + 1]))
                ):
                    break
                if is_underline(nxt):
                    break
                cont.append(nxt.rstrip())
                i += 1
            out.extend(wrap_bullet_item(first, cont, width))
            continue

        para = [line.rstrip()]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip():
                break
            if (
                is_bullet(nxt)
                or is_label(nxt)
                or starts_fence(nxt)
                or is_underline(nxt)
                or is_directive(nxt)
            ):
                break
            if i + 1 < len(lines) and is_underline(lines[i + 1]):
                break
            para.append(nxt.rstrip())
            i += 1
        out.extend(wrap_plain_paragraph(para, width))

    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return out


def collect_potential_code_sections(
    lines: list[str],
) -> list[tuple[int, str, list[str]]]:
    """Return labeled docstring sections that look like literal code.

    Purpose:
        Support audit reporting for sections such as `Typical use:` and
        `Examples:` that should not be flattened into prose.

    Architectural role:
        Read-only companion to the parser's literal-section protection. The
        report intentionally uses the same `is_literal_code_section` predicate
        so reported sections match the sections bypassed by wrapping.

    Inputs (architectural provenance):
        Receives normalized docstring lines with the summary already split away
        by the caller.

    Outputs (downstream usage):
        Returns normalized line offsets, labels, and raw section bodies for CLI
        reporting.

    Invariants/constraints:
        The function is conservative and reports only explicitly code-oriented
        labels such as `Typical use:` and `Examples:`. It does not treat
        ordinary prose sections as code merely because they contain dotted
        names, parentheses, inline literals, or method references.

    """
    sections: list[tuple[int, str, list[str]]] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if not is_label(line):
            i += 1
            continue

        label_index = i
        label = line.rstrip()
        i += 1

        body: list[str] = []
        while i < len(lines):
            cur = lines[i]

            if is_label(cur) and leading_ws(cur) == "":
                break
            if i + 1 < len(lines) and is_underline(lines[i + 1]):
                break
            if is_underline(cur):
                break

            body.append(cur.rstrip())
            i += 1

        while body and body[-1] == "":
            body.pop()

        if is_literal_code_section(label, body):
            sections.append((label_index, label, body))

    return sections


def report_docstring_code_sections(
    path: Path,
    docstring: str,
    doc_lineno: int,
) -> None:
    """Print locations of labeled docstring sections treated as code."""
    normalized_lines = normalize_docstring_lines(docstring)
    for line_offset, label, body in collect_potential_code_sections(
        normalized_lines
    ):
        preview_lines = [line.strip() for line in body if line.strip()]
        preview = preview_lines[0] if preview_lines else ""
        if len(preview) > 100:
            preview = f"{preview[:97]}..."
        print(
            f"{path}:{doc_lineno + line_offset}: code-like docstring section "
            f"{label!r} preview={preview!r}",
            file=sys.stderr,
        )


def reflow_docstring_content(docstring: str, width: int) -> list[str]:
    """Reflow raw docstring content into canonical wrapped lines.

    Purpose:
        Normalize and wrap the inside of a docstring without the surrounding
        triple quotes.

    Architectural role:
        Main formatting pipeline for repository docstring cleanup.

    Inputs (architectural provenance):
        Receives raw docstring text extracted from an abstract-syntax-tree
        string literal.

    Outputs (downstream usage):
        Returns lines that `render_docstring` can place back into source code.

    Invariants/constraints:
        The summary line is kept as the first logical line. Body wrapping
        preserves labels, bullets, directive-style blocks, and intentional
        blank-line structure where the parser can recognize it.

    """
    normalized = normalize_docstring_lines(docstring)
    return parse_lines(normalized, width)


def split_summary_body(lines: list[str]) -> tuple[str, list[str]]:
    """Split docstring lines into summary line and remaining body lines.

    Purpose:
        Separate the PEP 257 summary from the rest of the docstring so
        formatting can keep the summary on the opening quote line.

    Architectural role:
        Structural helper inside the docstring wrapping pipeline.

    Inputs (architectural provenance):
        Receives normalized docstring lines before final rendering.

    Outputs (downstream usage):
        Returns the summary and body line lists consumed by `render_docstring`.

    Invariants/constraints:
        The first non-empty logical line is the summary. The helper should not
        wrap the summary with body text because ruff D212 expects it on the
        first line.

    """
    if not lines:
        return "", []

    summary_parts: list[str] = []
    index = 0

    while index < len(lines) and lines[index].strip():
        summary_parts.append(lines[index].strip())
        index += 1

    while index < len(lines) and not lines[index].strip():
        index += 1

    return " ".join(summary_parts), lines[index:]


def render_docstring(
    docstring: str,
    doc_indent: str,
    width: int,
    *,
    path: Path | None = None,
    lineno: int = 1,
) -> str:
    """Render normalized/wrapped docstring text back to Python literal form.

    Purpose:
        Convert formatted docstring lines into a triple-double-quoted source
        literal with repository-preferred placement.

    Architectural role:
        Final source-rendering boundary of the docstring wrapper.

    Inputs (architectural provenance):
        Receives summary/body lines from the wrapping pipeline and indentation
        from the original docstring expression.

    Outputs (downstream usage):
        Returns the replacement source text inserted into the Python file.

    Invariants/constraints:
        The summary stays on the opening quote line. Multiline docstrings close
        on their own line and body lines preserve the target indentation.

    """
    content_width = max(20, width - len(doc_indent))
    normalized_lines = normalize_docstring_lines(docstring)
    summary, body_lines = split_summary_body(normalized_lines)

    if not summary:
        return f'{doc_indent}""""""'

    if not body_lines:
        summary_line_len = len(f'{doc_indent}"""{summary}"""')
        if summary_line_len > width and path is not None:
            print(
                f"{path}:{lineno}: warning: one-line docstring exceeds "
                f"{width} chars after rendering: {summary_line_len}",
                file=sys.stderr,
            )
        return f'{doc_indent}"""{summary}"""'

    summary_line_len = len(f'{doc_indent}"""{summary}')
    if summary_line_len > width and path is not None:
        print(
            f"{path}:{lineno}: warning: docstring summary exceeds "
            f"{width} chars after rendering: {summary_line_len}",
            file=sys.stderr,
        )

    body = parse_lines(body_lines, content_width)

    rendered = [f'{doc_indent}"""{summary}', ""]
    rendered.extend(f"{doc_indent}{line}" if line else "" for line in body)
    if body and rendered[-1].strip():
        rendered.append("")

    rendered.append(f'{doc_indent}"""')
    return "\n".join(rendered)


def process_file(
    path: Path, width: int, *, report_code_sections: bool = False
) -> bool:
    """Rewrite docstrings in one file and return whether content changed.

    Purpose:
        Apply the wrapping pipeline to every existing docstring literal in a
        single Python source file.

    Architectural role:
        File-level mutation boundary for the docstring formatting script.

    Inputs (architectural provenance):
        Receives a Python file path from CLI traversal.

    Outputs (downstream usage):
        Writes updated source when needed and returns a boolean used by CLI
        summary reporting.

    Invariants/constraints:
        Replacements must preserve valid Python syntax. The function rewrites
        only recognized docstring literals and leaves non-docstring strings
        untouched.

    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    source_lines = source.splitlines()

    replacements: list[tuple[int, int, ast.Expr, str]] = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue

        expr = get_docstring_expr(node)
        if expr is None:
            continue

        start = expr.lineno - 1
        end = expr.end_lineno
        if end is None:
            continue

        doc_indent = leading_ws(source_lines[start])
        replacements.append((start, end, expr, doc_indent))

    if not replacements:
        return False

    new_lines: list[str] = []
    cursor = 0

    for start, end, expr, doc_indent in sorted(replacements):
        new_lines.extend(source_lines[cursor:start])

        value = expr.value
        assert isinstance(value, ast.Constant)
        assert isinstance(value.value, str)

        output_lineno = len(new_lines) + 1

        if report_code_sections:
            report_docstring_code_sections(path, value.value, expr.lineno)

        replacement = render_docstring(
            value.value,
            doc_indent,
            width,
            path=path,
            lineno=output_lineno,
        )

        replacement_lines = replacement.splitlines()

        if len(replacement_lines) > 2:
            escaped = [
                replacement_lines[0],
                *[
                    line.replace('"""', '\\"\\"\\"')
                    for line in replacement_lines[1:-1]
                ],
                replacement_lines[-1],
            ]

            replacement = "\n".join(escaped)

        new_lines.extend(replacement.splitlines())
        cursor = end

    new_lines.extend(source_lines[cursor:])
    new_source = "\n".join(new_lines) + "\n"

    if new_source != source:
        path.write_text(new_source, encoding="utf-8", newline="\n")
        return True
    return False


def main() -> int:
    """CLI entrypoint for docstring wrapping across files or directories."""
    args = sys.argv[1:]
    report_code_sections = False

    if "--report-code-sections" in args:
        report_code_sections = True
        args = [arg for arg in args if arg != "--report-code-sections"]

    if len(args) not in (1, 2):
        print(
            "Usage: wrap_docstrings.py <file-or-dir> [width] "
            "[--report-code-sections]"
        )
        return 1

    target = Path(args[0])
    width = int(args[1]) if len(args) >= 2 else DEFAULT_WIDTH

    changed = 0
    for file in iter_py_files(target):
        if process_file(
            file,
            width,
            report_code_sections=report_code_sections,
        ):
            changed += 1

    print(f"Processed docstrings. Files changed: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
