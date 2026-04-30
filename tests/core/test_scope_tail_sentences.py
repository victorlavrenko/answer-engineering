from __future__ import annotations

import pytest

from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    TextView,
)
from answer_engineering.rules.compile.plan import (
    ScopeSpec,
)


def _tail_sentence_view(text: str, *, n: int = 1) -> tuple[str, int]:
    view = TextView(
        DocumentState(text=text, version_id="v1"),
        ScopeSpec(kind="tail_sentences", n=n, casefold=False),
    )
    return view.text, view.abs_start


def test_tail_sentences_counts_unfinished_trailing_sentence() -> None:
    text = "One. Two. Three"
    tail, start = _tail_sentence_view(text)

    assert tail == "Three"
    assert start == len("One. Two. ")


def test_tail_sentences_last_sentence_when_terminal_punctuation_present() -> (
    None
):
    text = "One. Two. Three."
    tail, start = _tail_sentence_view(text)

    assert tail == "Three."
    assert start == len("One. Two. ")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two. Three   ", "Three   "),
        ("One. Two. Three.   ", "Three.   "),
        ("One. Two. Three.\n", "Three.\n"),
    ],
)
def test_tail_sentences_preserve_trailing_whitespace(
    text: str, expected: str
) -> None:
    tail, start = _tail_sentence_view(text)

    assert tail == expected
    assert start == len("One. Two. ")


def test_tail_sentences_return_whole_text_when_n_covers_all_sentences() -> None:
    text = "One. Two. Three"
    tail, start = _tail_sentence_view(text, n=3)

    assert tail == text
    assert start == 0


def test_tail_sentences_handles_trailing_fragment_as_sentence_boundary() -> (
    None
):
    text = "One. Two unfinished"
    tail, start = _tail_sentence_view(text)

    assert tail == "Two unfinished"
    assert start == len("One. ")


def test_tail_sentences_no_punctuation_preserves_text() -> None:
    text = "No punctuation here   "
    tail, start = _tail_sentence_view(text)

    assert tail == text
    assert start == 0
