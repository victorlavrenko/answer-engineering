"""Utilities for locating sentence and clause boundaries inside scoped text.

Purpose:
    Provide small deterministic boundary finders reused when computing runtime
    guard/edit views and anchor-relative spans.

Architectural role:
    Low-level text segmentation helper module for proposal targeting.

Inputs:
    Called with current document text plus start/limit coordinates from scope
    and target-building logic.

Outputs:
    Returns boundary indices consumed by scope construction and edit- span
    resolution.

"""

from __future__ import annotations

from answer_engineering.engine.span_utils import clamp_index


def find_sentence_end(text: str, start: int, limit: int) -> int:
    """Return the first sentence boundary between start and limit.

    Purpose:
        Locate a conservative end point for sentence-scoped proposal and patch
        operations.

    Architectural role:
        Text-boundary primitive shared by runtime span expansion helpers.

    Inputs (architectural provenance):
        Receives document text plus the inclusive scan start and exclusive scan
        limit selected by a caller.

    Outputs (downstream usage):
        Returns the index of the first terminator or the bounded scan end when
        no terminator is present.

    Invariants/constraints:
        Scanning is clamped to the document length. The helper does not parse
        abbreviations or nested syntax; it is a deterministic character-boundary
        heuristic.

    """
    i = clamp_index(start, text)
    hi = max(i, clamp_index(limit, text))
    while i < hi:
        if text[i] in ".?!\n":
            return i
        i += 1
    return hi


def find_clause_end(text: str, start: int, limit: int) -> int:
    """Return the first clause boundary between start and limit.

    Purpose:
        Find a conservative punctuation boundary for clause-sized edit scopes.

    Architectural role:
        Text-boundary primitive used by scope and proposal logic before
        canonical patch spans are formed.

    Inputs (architectural provenance):
        Receives document text, a start index, and a caller-provided limit.

    Outputs (downstream usage):
        Returns the first comma, semicolon, sentence terminator, newline, or the
        bounded scan end.

    Invariants/constraints:
        The function is intentionally heuristic and side-effect free. Callers
        own any higher-level linguistic interpretation of the returned index.

    """
    i = clamp_index(start, text)
    hi = max(i, clamp_index(limit, text))
    while i < hi:
        if text[i] in ",;:.?!\n":
            return i
        i += 1
    return hi


def find_clause_start(text: str, pos: int, limit: int) -> int:
    """Scan backward from ``pos`` to the start of the enclosing clause."""
    lo = clamp_index(limit, text)
    i = max(lo, clamp_index(pos, text))
    while i > lo:
        if text[i - 1] in ",;:.?!\n":
            return i
        i -= 1
    return lo
