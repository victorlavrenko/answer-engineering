from answer_engineering.engine.pipeline.text_patterns import (
    search_span,
)
from answer_engineering.rules.matching.options import ResolvedMatchOptions


def test_acronym_literal_respects_word_boundaries() -> None:
    assert (
        search_span(
            "the patient",
            "ENT",
            match_options=ResolvedMatchOptions(casefold=True, word=False),
        )
        is None
    )


def test_acronym_literal_still_finds_standalone_word() -> None:
    assert search_span(
        "Urgent ENT evaluation",
        "ENT",
        match_options=ResolvedMatchOptions(casefold=True, word=False),
    ) == (7, 10)


def test_non_acronym_literal_allows_substring_match() -> None:
    assert search_span(
        "Start corticosteroids now",
        "steroid",
        match_options=ResolvedMatchOptions(casefold=True, word=False),
    ) == (13, 20)


def test_word_modifier_enforces_word_boundaries() -> None:
    assert (
        search_span(
            "Start corticosteroids now",
            "steroid",
            match_options=ResolvedMatchOptions(casefold=True, word=True),
        )
        is None
    )
    assert search_span(
        "Start steroid now",
        "steroid",
        match_options=ResolvedMatchOptions(casefold=True, word=True),
    ) == (6, 13)


def test_non_keyword_parenthesized_text_remains_literal() -> None:
    assert search_span(
        "Use (steroids).",
        "(steroids)",
        match_options=ResolvedMatchOptions(casefold=True, word=False),
    ) == (
        4,
        14,
    )
    assert search_span(
        "Use (steroids).",
        "(steroids).",
        match_options=ResolvedMatchOptions(casefold=True, word=False),
    ) == (
        4,
        15,
    )
