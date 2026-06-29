"""Central span and token-alignment safety helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from answer_engineering.engine.runtime.runtime_types import TokenCharAlignment

type Span = tuple[int, int]
NormalizeMode = Literal["fallback_then_clamp", "clamp", "drop"]
AlignmentErrorKind = Literal[
    "token_index_not_monotonic",
    "negative_char_start",
    "char_end_before_start",
    "char_end_out_of_bounds",
    "char_start_decreased",
    "char_end_decreased",
    "char_span_overlap",
    "piece_text_mismatch",
]
OverlapKind = Literal[
    "identical_span",
    "same_start_shorter_or_longer",
    "partial_overlap",
    "current_inside_previous",
    "previous_inside_current",
]


@dataclass(frozen=True, slots=True)
class SpanNormalizationResult:
    """Result of validating, repairing, or dropping a span."""

    original: Span
    span: Span | None
    reason: str | None
    changed: bool
    fallback_used: bool = False
    clamped: bool = False


@dataclass(frozen=True, slots=True)
class AlignmentValidationError:
    """Structured token/character-alignment validation diagnostic."""

    kind: AlignmentErrorKind
    token_index: int | None = None
    previous_token_index: int | None = None
    token_id: int | None = None
    previous_token_id: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    previous_char_start: int | None = None
    previous_char_end: int | None = None
    piece_text: str | None = None
    previous_piece_text: str | None = None
    text_slice: str | None = None
    previous_text_slice: str | None = None
    text_slice_codepoints: str | None = None
    previous_text_slice_codepoints: str | None = None
    doc_len: int | None = None
    overlap_kind: OverlapKind | None = None

    def compact(self) -> str:
        """Render a compact, log-safe one-line diagnostic."""
        parts = [self.kind]
        fields: list[tuple[str, object | None]] = [
            ("token_index", self.token_index),
            ("token_id", self.token_id),
            (
                "span",
                (
                    None
                    if self.char_start is None or self.char_end is None
                    else (self.char_start, self.char_end)
                ),
            ),
            ("char_start", self.char_start),
            ("char_end", self.char_end),
            ("slice", self.text_slice),
            ("codepoints", self.text_slice_codepoints),
            ("piece", self.piece_text),
            ("previous_token_index", self.previous_token_index),
            ("previous_token_id", self.previous_token_id),
            (
                "previous_span",
                (
                    None
                    if self.previous_char_start is None
                    or self.previous_char_end is None
                    else (self.previous_char_start, self.previous_char_end)
                ),
            ),
            ("previous_char_start", self.previous_char_start),
            ("previous_char_end", self.previous_char_end),
            ("previous_slice", self.previous_text_slice),
            ("previous_codepoints", self.previous_text_slice_codepoints),
            ("previous_piece", self.previous_piece_text),
            ("doc_len", self.doc_len),
            ("overlap_kind", self.overlap_kind),
        ]
        for name, value in fields:
            if value is not None:
                parts.append(
                    f"{name}={value!r}"
                    if isinstance(value, str) and name != "overlap_kind"
                    else f"{name}={value}"
                )
        return " ".join(parts)


class InvalidTokenAlignmentError(ValueError):
    """ValueError carrying a structured alignment validation diagnostic."""

    def __init__(self, error: AlignmentValidationError) -> None:
        """Build the exception from a structured validation diagnostic."""
        self.error = error
        super().__init__(error.compact())


def _codepoints(s: str, *, limit: int = 16) -> str:
    suffix = " …" if len(s) > limit else ""
    return " ".join(f"U+{ord(ch):04X}" for ch in s[:limit]) + suffix


def _debug_text_slice(
    text: str, start: int | None, end: int | None, *, limit: int = 80
) -> str | None:
    """Return a bounded diagnostic slice without trusting span validity."""
    if start is None or end is None:
        return None
    lo = min(max(start, 0), len(text))
    hi = min(max(end, 0), len(text))
    if hi < lo:
        hi = lo
    snippet = text[lo:hi]
    if len(snippet) > limit:
        snippet = snippet[:limit] + "…"
    return snippet


def _overlap_kind(
    start: int, end: int, prev_start: int, prev_end: int
) -> OverlapKind:
    if (start, end) == (prev_start, prev_end):
        return "identical_span"
    if start == prev_start:
        return "same_start_shorter_or_longer"
    if prev_start <= start and end <= prev_end:
        return "current_inside_previous"
    if start <= prev_start and prev_end <= end:
        return "previous_inside_current"
    return "partial_overlap"


def _alignment_error(
    kind: AlignmentErrorKind,
    *,
    item: TokenCharAlignment,
    text: str,
    previous: TokenCharAlignment | None = None,
    overlap_kind: OverlapKind | None = None,
) -> AlignmentValidationError:
    text_slice = _debug_text_slice(text, item.char_start, item.char_end)
    previous_text_slice = (
        None
        if previous is None
        else _debug_text_slice(text, previous.char_start, previous.char_end)
    )
    return AlignmentValidationError(
        kind=kind,
        token_index=item.token_index,
        previous_token_index=None if previous is None else previous.token_index,
        token_id=item.token_id,
        previous_token_id=None if previous is None else previous.token_id,
        char_start=item.char_start,
        char_end=item.char_end,
        previous_char_start=None if previous is None else previous.char_start,
        previous_char_end=None if previous is None else previous.char_end,
        piece_text=item.piece_text,
        previous_piece_text=None if previous is None else previous.piece_text,
        text_slice=text_slice,
        previous_text_slice=previous_text_slice,
        text_slice_codepoints=None
        if text_slice is None
        else _codepoints(text_slice),
        previous_text_slice_codepoints=(
            None
            if previous_text_slice is None
            else _codepoints(previous_text_slice)
        ),
        doc_len=len(text),
        overlap_kind=overlap_kind,
    )


def is_valid_index(index: int, text: str) -> bool:
    """Return whether index is inside half-open text boundaries."""
    return 0 <= index <= len(text)


def clamp_index(index: int, text: str) -> int:
    """Clamp index into the inclusive [0, len(text)] range."""
    return min(max(index, 0), len(text))


def is_valid_span(span: Span | None, text: str) -> bool:
    """Return whether span is a valid half-open code-point span."""
    if span is None:
        return False
    start, end = span
    return 0 <= start <= end <= len(text)


def span_error(span: Span | None, text: str) -> str | None:
    """Return a human-readable span validation error, if any."""
    if span is None:
        return "span is None"
    start, end = span
    if start < 0:
        return f"start {start} < 0"
    if end < start:
        return f"end {end} < start {start}"
    if end > len(text):
        return f"end {end} > doc_len {len(text)}"
    return None


def clamp_span(span: Span, text: str) -> Span:
    """Clamp span coordinates into text bounds preserving ordering."""
    start, end = span
    start = clamp_index(start, text)
    end = clamp_index(end, text)
    if end < start:
        end = start
    return start, end


def normalize_span(
    span: Span,
    text: str,
    *,
    fallback: Span | None = None,
    mode: NormalizeMode = "fallback_then_clamp",
    max_clamp_delta: int = 8,
) -> SpanNormalizationResult:
    """Repair span using fallback or bounded clamping, else drop it."""
    if is_valid_span(span, text):
        return SpanNormalizationResult(span, span, None, False)
    if mode == "drop":
        return SpanNormalizationResult(span, None, "invalid_span_dropped", True)
    if mode == "fallback_then_clamp" and is_valid_span(fallback, text):
        return SpanNormalizationResult(
            span, fallback, "invalid_span_fallback", True, fallback_used=True
        )
    clamped = clamp_span(span, text)
    delta = abs(clamped[0] - span[0]) + abs(clamped[1] - span[1])
    if mode == "clamp" or delta <= max_clamp_delta:
        return SpanNormalizationResult(
            span, clamped, "invalid_span_clamped", True, clamped=True
        )
    return SpanNormalizationResult(span, None, "invalid_span_dropped", True)


def describe_span(span: Span | None, text: str, *, context: int = 120) -> str:
    """Describe span validity and nearby text for diagnostics."""
    if span is None:
        return f"span=None doc_len={len(text)}"
    start, end = span
    lo = clamp_index(start - context, text)
    hi = clamp_index(end + context, text)
    valid = is_valid_span(span, text)
    slice_text = text[start:end] if valid else "<invalid>"
    return (
        f"doc_len={len(text)} span={span} valid={valid} "
        f"slice={slice_text!r} around={text[lo:hi]!r}"
    )


def validate_token_alignment(
    alignment: Sequence[TokenCharAlignment],
    text: str,
    *,
    require_piece_match: bool = True,
) -> str | None:
    """Return None if generated token character alignment is valid."""
    error = validate_token_alignment_detailed(
        alignment, text, require_piece_match=require_piece_match
    )
    return None if error is None else error.compact()


def validate_token_alignment_detailed(
    alignment: Sequence[TokenCharAlignment],
    text: str,
    *,
    require_piece_match: bool = True,
) -> AlignmentValidationError | None:
    """Return a structured diagnostic for invalid token alignment."""
    prev_token = -1
    previous: TokenCharAlignment | None = None
    for item in alignment:
        if item.token_index <= prev_token:
            return AlignmentValidationError(
                kind="token_index_not_monotonic",
                token_index=item.token_index,
                previous_token_index=prev_token,
                token_id=item.token_id,
                char_start=item.char_start,
                char_end=item.char_end,
                piece_text=item.piece_text,
                text_slice=_debug_text_slice(
                    text, item.char_start, item.char_end
                ),
                doc_len=len(text),
            )
        if item.char_start < 0:
            return _alignment_error("negative_char_start", item=item, text=text)
        if item.char_end < item.char_start:
            return _alignment_error(
                "char_end_before_start", item=item, text=text
            )
        if item.char_end > len(text):
            return _alignment_error(
                "char_end_out_of_bounds", item=item, text=text
            )
        if previous is not None and item.char_start < previous.char_start:
            return _alignment_error(
                "char_start_decreased", item=item, previous=previous, text=text
            )
        if previous is not None and item.char_start < previous.char_end:
            return _alignment_error(
                "char_span_overlap",
                item=item,
                previous=previous,
                text=text,
                overlap_kind=_overlap_kind(
                    item.char_start,
                    item.char_end,
                    previous.char_start,
                    previous.char_end,
                ),
            )
        if previous is not None and item.char_end < previous.char_end:
            return _alignment_error(
                "char_end_decreased", item=item, previous=previous, text=text
            )
        if (
            require_piece_match
            and item.piece_text != text[item.char_start : item.char_end]
        ):
            return _alignment_error("piece_text_mismatch", item=item, text=text)
        prev_token = item.token_index
        previous = item
    return None
