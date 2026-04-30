"""Runtime scope-window derivation helpers.

Purpose:
    Own derivation of scoped ``TextView`` windows from ``DocumentState`` and
    compiled ``ScopeSpec`` values.

Architectural role:
    Runtime helper module that turns one document snapshot into scoped views for
    proposal-time guard matching and edit targeting.

Supported scope kinds:
    - full document
    - tail chars
    - tail sentences
    - tail clauses

Output contract:
    ``TextView`` values with absolute-coordinate mapping metadata.

"""

from __future__ import annotations

from answer_engineering.config.engine_defaults import ScopeDefaults
from answer_engineering.engine.runtime.document_state import DocumentState
from answer_engineering.rules.compile.plan import (
    ScopeSpec,
)


def resolve_scoped_text(
    doc: DocumentState,
    spec: ScopeSpec,
) -> tuple[str, int]:
    """Return scoped raw text and absolute start requested by ``spec``.

    Purpose:
        Slice the current document into the exact window used for guard matching
        and edit targeting, while preserving absolute coordinates into the
        source document version.

    """
    if spec.kind == "tail_chars":
        n = (
            spec.max_chars
            if spec.max_chars is not None
            else (spec.n or ScopeDefaults().tail_chars)
        )
        abs_start = max(0, len(doc.text) - n)
        raw = doc.text[abs_start:]
    elif spec.kind == "tail_sentences":
        n = spec.n or ScopeDefaults().sentences
        raw, abs_start = _tail_sentences(doc.text, n)
    elif spec.kind == "tail_clauses":
        n = spec.n or ScopeDefaults().clauses
        raw, abs_start = _tail_clauses(
            doc.text,
            n,
            include_leading_delimiter=spec.include_leading_delimiter,
        )
    else:
        abs_start = 0
        raw = doc.text
    return raw, abs_start


def _tail_sentences(text: str, n: int) -> tuple[str, int]:
    """Return the last ``n`` sentences (or full text) plus absolute start."""
    scan_text, scan_end, slice_end = _scan_window(text)
    sentence_ends = _find_sentence_ends(scan_text, scan_end)
    if not sentence_ends:
        return text, 0

    if len(sentence_ends) <= n:
        return text, 0

    start = sentence_ends[-(n + 1)]
    return _left_trimmed_slice(text, start=start, end=slice_end)


def _tail_clauses(
    text: str, n: int, *, include_leading_delimiter: bool = False
) -> tuple[str, int]:
    """Return the last ``n`` clauses (or sentence fallback) plus start index."""
    scan_text, scan_end, slice_end = _scan_window(text)
    clause_starts = _find_clause_starts(scan_text, scan_end)
    if len(clause_starts) <= 1:
        return _tail_sentences(text, n)

    start = 0 if n >= len(clause_starts) else clause_starts[-n]
    tail_text, abs_start = _left_trimmed_slice(text, start=start, end=slice_end)
    if include_leading_delimiter:
        abs_start = _expand_start_to_leading_delimiter(
            text, abs_start=abs_start
        )
        tail_text = text[abs_start:slice_end]
    return tail_text, abs_start


def _scan_window(text: str) -> tuple[str, int, int]:
    """Return right-trimmed scan text with scan and original slice end."""
    scan_text = text.rstrip()
    return scan_text, len(scan_text), len(text)


def _find_sentence_ends(scan_text: str, scan_end: int) -> list[int]:
    """Locate sentence end boundaries in scan-window text."""
    sentence_ends = [
        index + 1 for index, ch in enumerate(scan_text) if ch in ".!?"
    ]
    if not sentence_ends:
        return list()

    has_trailing_fragment = bool(scan_text[sentence_ends[-1] :].strip())
    if has_trailing_fragment:
        sentence_ends.append(scan_end)
    return sentence_ends


def _find_clause_starts(scan_text: str, scan_end: int) -> list[int]:
    """Locate clause starts while skipping introductory-comma boundaries."""
    delimiters = ",;:.?!\n"
    clause_starts = [0]
    for index, ch in enumerate(scan_text):
        if ch not in delimiters:
            continue
        if _is_introductory_comma(scan_text, index):
            continue
        clause_starts.append(index + 1)

    if clause_starts[-1] == scan_end:
        clause_starts.pop()
    return clause_starts


def _left_trimmed_slice(text: str, *, start: int, end: int) -> tuple[str, int]:
    """Return a left-trimmed slice and its adjusted absolute start."""
    raw = text[start:end]
    trimmed = raw.lstrip()
    trimmed_start = start + (len(raw) - len(trimmed))
    return trimmed, trimmed_start


def _expand_start_to_leading_delimiter(text: str, *, abs_start: int) -> int:
    """Expand a clause start backward to include its leading delimiter."""
    delimiters = ",;:.?!\n"
    back = abs_start - 1
    while back >= 0 and text[back].isspace():
        back -= 1
    if back >= 0 and text[back] in delimiters:
        return back
    return abs_start


def _is_introductory_comma(text: str, index: int) -> bool:
    """Return whether the comma closes a sentence-leading adverbial phrase."""
    if text[index] != ",":
        return False
    sentence_breaks = ".?!\n"
    sentence_start = 0
    for pos in range(index - 1, -1, -1):
        if text[pos] in sentence_breaks:
            sentence_start = pos + 1
            break
    leader = text[sentence_start:index].strip().casefold()
    return leader in {
        "however",
        "therefore",
        "moreover",
        "furthermore",
        "instead",
        "meanwhile",
    }


def sentence_floor_start(*, text: str, span_abs: tuple[int, int]) -> int:
    """Return the sentence-floor start inside an avoid span.

    Purpose:
        Prevent avoid-scope edits from starting in the middle of a later
        sentence when a span contains multiple sentence-like segments.

    Architectural role:
        Runtime scope-normalization helper used before avoid proposals are
        anchored to document coordinates.

    Inputs (architectural provenance):
        Receives the full document text and an absolute half-open span selected
        by matching or scope expansion.

    Outputs (downstream usage):
        Returns either the original span start or the first non-space character
        after the last sentence boundary inside the span.

    Invariants/constraints:
        Empty spans and spans whose final boundary consumes the whole snippet
        fall back to the original start. The helper is deterministic and does
        not mutate document state.

    """
    start, end = span_abs
    if start >= end:
        return start
    snippet = text[start:end]
    last_boundary = -1
    for idx, ch in enumerate(snippet):
        if ch in ".!?\n":
            last_boundary = idx + 1
    if last_boundary <= 0:
        return start
    while last_boundary < len(snippet) and snippet[last_boundary].isspace():
        last_boundary += 1
    if last_boundary >= len(snippet):
        return start
    return start + last_boundary
