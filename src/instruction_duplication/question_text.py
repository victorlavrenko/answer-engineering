"""Deterministic normalization of question text before selection or comparison."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .answer_utils import normalize_text

_ANSWER_CHOICES_MARKER_RE = re.compile(r"\s+Answer\s+Choices\s*:\s*", re.IGNORECASE)
_PARENTHESIZED_CHOICE_RE = re.compile(r"\(([A-Z])\)\s*", re.IGNORECASE)


def strip_exact_embedded_choice_suffix(
    stem: str,
    choices: Mapping[str, str],
) -> tuple[str, bool]:
    """Strip a terminal ``Answer Choices`` block only when it exactly mirrors choices.

    Some upstream datasets expose answer choices twice: once inside the question text
    and again as structured options.  The experiment renders structured options itself.
    This helper removes only a terminal block whose labels and normalized texts exactly
    reproduce those options; ambiguous prose is left untouched.
    """
    markers = tuple(_ANSWER_CHOICES_MARKER_RE.finditer(stem))
    if not markers:
        return stem, False
    marker = markers[-1]
    suffix = stem[marker.end() :].strip()
    matches = tuple(_PARENTHESIZED_CHOICE_RE.finditer(suffix))
    if not matches:
        return stem, False
    labels = tuple(match.group(1).upper() for match in matches)
    if labels != tuple(choices):
        return stem, False
    parsed: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(suffix)
        parsed[match.group(1).upper()] = suffix[match.end() : end].strip()
    if any(
        normalize_text(parsed[label]) != normalize_text(text) for label, text in choices.items()
    ):
        return stem, False
    cleaned = stem[: marker.start()].rstrip()
    return (cleaned, True) if cleaned else (stem, False)
