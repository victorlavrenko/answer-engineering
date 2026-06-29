"""Tokenizer-offset reconstruction helpers for runtime text/span alignment.

Purpose:
    Own tokenizer-offset reconstruction and char-span ↔ token-span conversion
    helpers used by scoring and related runtime text alignment flows.

Architectural role:
    Runtime alignment helper module for tokenizer-driven offset mapping.

Non-ownership:
    Does not own decode-session incremental alignment accumulation; it only
    provides reconstruction/conversion helpers from tokenizer offsets.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from answer_engineering.inference.model_types import TextCodec


@dataclass(frozen=True, slots=True, init=False)
class TokenizedTextWithOffsets:
    """Tokenization result paired with per-token character offsets for one text.

    Invariants:
        ``token_ids`` and ``offsets`` share the same token order and both were
        produced from the same source text/tokenizer call.

    """

    token_ids: list[int]
    offsets: list[tuple[int, int]]

    def __init__(self, tokenizer: TextCodec, text: str) -> None:
        """Tokenize text and retain token ids with character offsets.

        Purpose:
            Build the alignment table needed to translate between character
            spans in visible text and token spans used by model-backed scoring
            or probing.

        Architectural role:
            Constructor boundary between tokenizer output and runtime coordinate
            logic. It keeps token ids and offsets together so downstream code
            cannot mix incompatible alignment sources.

        Inputs (architectural provenance):
            Receives a text codec/tokenizer and the exact visible text snapshot
            to tokenize.

        Outputs (downstream usage):
            Stores token ids and character-offset metadata consumed by span
            conversion helpers.

        Invariants/constraints:
            Offsets must refer to the same text string that was tokenized. Span
            conversion is valid only while the text snapshot remains unchanged.

        """
        encoded = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        token_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        if len(token_ids) != len(offsets):
            raise ValueError("token id / offset length mismatch")
        for i, (start, end) in enumerate(offsets):
            if not 0 <= start <= end <= len(text):
                raise ValueError(
                    "offset out of bounds at token "
                    f"{i}: {(start, end)} doc_len={len(text)}"
                )
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "offsets", offsets)


def char_span_to_token_span(
    offsets: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int]:
    """Map a half-open character span to a token-span slice.

    Purpose:
        Translate document character coordinates into tokenizer-offset
        coordinates for model scoring and patch-context construction.

    Architectural role:
        Alignment boundary between text-level runtime spans and token-level
        scoring tasks.

    Inputs (architectural provenance):
        Receives tokenizer offset mappings and the character start/end pair
        selected by proposal or scoring logic.

    Outputs (downstream usage):
        Returns a half-open token slice covering all tokens that overlap the
        character span.

    Invariants/constraints:
        Invalid spans, spans outside the encoded text, and inconsistent mappings
        raise `ValueError`. A zero-length span at the end of the text maps to
        the end token position.

    """
    if start < 0 or end < start:
        raise ValueError("invalid char span")
    if not offsets:
        if start == end == 0:
            return 0, 0
        raise ValueError("cannot map non-empty span with empty token offsets")

    for i, (os, oe) in enumerate(offsets):
        if os < 0 or oe < os:
            raise ValueError(f"invalid offset at token {i}: {(os, oe)}")
    text_end = max(item[1] for item in offsets)
    if end > text_end:
        raise ValueError(f"span end {end} exceeds text length {text_end}")

    non_empty = [(i, os, oe) for i, (os, oe) in enumerate(offsets) if oe > os]
    tok_start = next((i for i, _, oe in non_empty if oe > start), None)
    tok_end = next((i for i, os, _ in non_empty if os >= end), len(offsets))

    if tok_start is None:
        if start == end == text_end:
            return len(offsets), len(offsets)
        raise ValueError(f"could not map span start {start} to token index")
    if tok_end < tok_start:
        raise ValueError("token span mapping is inconsistent")
    return tok_start, tok_end
