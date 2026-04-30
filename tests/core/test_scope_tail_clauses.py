from __future__ import annotations

import pytest

from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    TextView,
)
from answer_engineering.rules.compile.plan import (
    ScopeSpec,
)


def _tail_clause_view(
    text: str,
    *,
    n: int = 1,
    include_leading_delimiter: bool = False,
) -> tuple[str, int]:
    view = TextView(
        DocumentState(text=text, version_id="v1"),
        ScopeSpec(
            kind="tail_clauses",
            n=n,
            casefold=False,
            include_leading_delimiter=include_leading_delimiter,
        ),
    )
    return view.text, view.abs_start


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("alpha, beta, gamma", ", gamma"),
        ("alpha, beta, gamma, ", ", gamma, "),
        ("alpha, beta, gamma.", ", gamma."),
        ("alpha, beta, gamma. \n", ", gamma. \n"),
    ],
)
def test_tail_clauses_include_leading_delimiter_corner_cases(
    text: str, expected: str
) -> None:
    tail, _ = _tail_clause_view(text, include_leading_delimiter=True)
    assert tail == expected


def test_tail_clauses_default_excludes_leading_delimiter() -> None:
    tail, _ = _tail_clause_view(
        "alpha, beta, gamma", include_leading_delimiter=False
    )
    assert tail == "gamma"


def test_tail_clauses_include_leading_delimiter_toggle() -> None:
    tail, _ = _tail_clause_view(
        "alpha, beta, gamma", include_leading_delimiter=True
    )
    assert tail == ", gamma"


def test_tail_clauses_accumulate_across_sentences() -> None:
    tail, _ = _tail_clause_view(
        "a, b. c, d.", n=3, include_leading_delimiter=True
    )
    assert tail == ", b. c, d."


def test_tail_clauses_keeps_introductory_however_with_clause() -> None:
    tail, start = _tail_clause_view(
        "However, a b c d", include_leading_delimiter=True
    )
    assert tail == "However, a b c d"
    assert start == 0


def test_tail_clauses_preserve_trailing_whitespace() -> None:
    tail, start = _tail_clause_view(
        "alpha, beta   ", include_leading_delimiter=False
    )
    assert tail == "beta   "
    assert start == len("alpha, ")


def test_tail_clauses_return_whole_text_when_n_covers_all_clauses() -> None:
    text = "a, b"
    tail, start = _tail_clause_view(text, n=99, include_leading_delimiter=False)
    assert tail == text
    assert start == 0


def test_tail_clauses_fallback_to_sentence_behavior_without_delimiters() -> (
    None
):
    text = "One. Two   "
    tail, start = _tail_clause_view(text, include_leading_delimiter=False)
    assert tail == "Two   "
    assert start == len("One. ")
