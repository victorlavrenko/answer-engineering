"""Pattern compilation helpers for rule matching and gating.

Purpose:
    Provide a shared matcher for literal phrases, numeric-range shorthand, and
    explicit regex expressions in rule guard/anchor fields.

Architectural role:
    Keeps matching semantics centralized so guard checks and anchor resolution
    evolve together as domain-specific language matching features expand.

"""

from __future__ import annotations

import re
from functools import lru_cache

from answer_engineering.rules.matching.options import ResolvedMatchOptions

_RANGE_TOKEN = re.compile(r"(?P<low>\d+)\s*-\s*(?P<high>\d+)")
_ACRONYM_LITERAL = re.compile(r"[A-Z]{2,}")


def search_span(
    text: str,
    expression: str,
    *,
    match_options: ResolvedMatchOptions,
) -> tuple[int, int] | None:
    """Return the first span matched by one DSL expression in ``text``."""
    match = _compile_expression(expression, match_options=match_options).search(
        text
    )
    if match is None:
        return None
    return (int(match.start()), int(match.end()))


def find_spans(
    text: str,
    expression: str,
    *,
    match_options: ResolvedMatchOptions,
) -> tuple[tuple[int, int], ...]:
    """Return all non-overlapping spans matched by one DSL expression."""
    pattern = _compile_expression(expression, match_options=match_options)
    return tuple((int(m.start()), int(m.end())) for m in pattern.finditer(text))


@lru_cache(maxsize=512)
def _compile_expression(
    expression: str, *, match_options: ResolvedMatchOptions
) -> re.Pattern[str]:
    """Compile one rule expression into a cached regex matcher.

    Purpose:
        Convert authored domain-specific language match expressions into the
        concrete regular expression object used for repeated span searches.

    Architectural role:
        Pattern-compilation boundary between rule matching options and runtime
        text search.

    Inputs (architectural provenance):
        Receives an authored expression and resolved match options such as case
        folding and word-boundary policy.

    Outputs (downstream usage):
        Returns a compiled `re.Pattern` reused by `search_span` and
        `find_spans`.

    Invariants/constraints:
        Slash-delimited expressions are treated as regex bodies. Other
        expressions are escaped as literals, with inline numeric ranges expanded
        before optional word-boundary guards are added.

    """
    stripped = expression.strip()
    flags = re.IGNORECASE if match_options.casefold else 0
    if (
        len(stripped) >= 2
        and stripped.startswith("/")
        and stripped.endswith("/")
    ):
        return re.compile(stripped[1:-1], flags=flags)

    literal_pattern = _literal_with_ranges_to_regex(expression)
    literal_stripped = expression.strip()
    enforce_word_boundaries = match_options.word or bool(
        _ACRONYM_LITERAL.fullmatch(literal_stripped)
    )
    if enforce_word_boundaries:
        starts_word = bool(literal_stripped) and literal_stripped[0].isalnum()
        ends_word = bool(literal_stripped) and literal_stripped[-1].isalnum()
        if starts_word:
            literal_pattern = rf"(?<!\w){literal_pattern}"
        if ends_word:
            literal_pattern = rf"{literal_pattern}(?!\w)"
    return re.compile(literal_pattern, flags=flags)


def _literal_with_ranges_to_regex(text: str) -> str:
    """Escape literal text while expanding inline numeric ranges.

    Purpose:
        Preserve literal matching semantics while allowing compact authored
        numeric ranges in rule expressions.

    Architectural role:
        Pattern-normalization helper used before regex compilation.

    Inputs (architectural provenance):
        Receives the non-regex expression text from the rule domain-specific
        language matcher.

    Outputs (downstream usage):
        Returns regex source where ordinary text is escaped and recognized
        ranges are replaced by numeric alternations.

    Invariants/constraints:
        The helper should not interpret arbitrary regex syntax. Only the
        supported range token syntax is expanded.

    """
    parts: list[str] = []
    cursor = 0
    for match in _RANGE_TOKEN.finditer(text):
        start, end = match.span()
        parts.append(re.escape(text[cursor:start]))
        low = int(match.group("low"))
        high = int(match.group("high"))
        if low > high:
            low, high = high, low
        parts.append(_numeric_range_alternation(low, high))
        cursor = end
    parts.append(re.escape(text[cursor:]))
    return "".join(parts)


def _numeric_range_alternation(low: int, high: int) -> str:
    """Return a regex alternation matching every integer from ``low`` to."""
    values = "|".join(str(n) for n in range(low, high + 1))
    return rf"(?:{values})"
