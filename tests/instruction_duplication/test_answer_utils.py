from __future__ import annotations

import pytest

from instruction_duplication.answer_utils import (
    ChoiceNormalizationError,
    canonical_gold_labels,
    canonicalize_choices,
    extract_answer,
    normalize_text,
)

CHOICES = {"A": "Alpha therapy", "B": "Beta therapy", "C": "Gamma therapy"}


def test_numeric_string_index_is_zero_based_for_index_field():
    assert canonical_gold_labels("0", CHOICES, index_base=0) == {"A"}
    assert canonical_gold_labels("1", CHOICES, index_base=1) == {"A"}


def test_choice_number_label_does_not_match_letter_in_word():
    assert canonicalize_choices({"label": ["choice_1", "choice_2"], "text": ["one", "two"]}) == {
        "A": "one",
        "B": "two",
    }


def test_strings_are_not_parallel_sequences():
    with pytest.raises(ChoiceNormalizationError):
        canonicalize_choices({"label": "AB", "text": "xy"})


def test_duplicate_choice_collision_is_rejected():
    with pytest.raises(ChoiceNormalizationError):
        canonicalize_choices({"A": "one", "option_A": "two"})


def test_unicode_normalization_preserves_non_ascii_letters():
    assert (
        normalize_text("Ménière — \u043b\u0435\u0432\u043e\u0435 \u0443\u0445\u043e")
        == "ménière \u043b\u0435\u0432\u043e\u0435 \u0443\u0445\u043e"
    )


def test_answer_extraction_uses_final_region_and_detects_conflict():
    parsed = extract_answer("Option B is less likely.\nFinal answer: A", CHOICES)
    assert parsed.option == "A"
    conflict = extract_answer("Final answer: A; answer: B", CHOICES)
    assert conflict.status == "conflicting"


def test_lowercase_article_is_not_option_a():
    parsed = extract_answer("The answer is a treatment chosen after review.", CHOICES)
    assert parsed.option is None


def test_choice_shapes_are_normalized_without_silent_guessing():
    assert canonicalize_choices(["one", "two"]) == {"A": "one", "B": "two"}
    assert canonicalize_choices([{"label": "A", "text": "one"}, {"label": "B", "text": "two"}]) == {
        "A": "one",
        "B": "two",
    }
    assert canonicalize_choices("A. one\nB. two") == {"A": "one", "B": "two"}
    assert canonicalize_choices("1) one\n2) two") == {"A": "one", "B": "two"}


def test_gold_parsing_handles_text_lists_and_rejects_unknowns():
    assert canonical_gold_labels("option B", CHOICES) == {"B"}
    assert canonical_gold_labels("Beta therapy", CHOICES) == {"B"}
    assert canonical_gold_labels(["A", "C"], CHOICES) == {"A", "C"}
    assert canonical_gold_labels("not represented", CHOICES) == set()


def test_terminal_choice_text_can_be_extracted():
    parsed = extract_answer("Review complete.\nBeta therapy", CHOICES)
    assert parsed.option == "B"


def test_numeric_gold_requires_explicit_schema_index_base():
    with pytest.raises(ValueError, match="index_base"):
        canonical_gold_labels("1", CHOICES)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Final Answer: B", "B"),
        ("Answer: B. Beta therapy", "B"),
        ("**Final Answer: B**", "B"),
        ("### Final Answer: **B. Beta therapy**", "B"),
        ("Final Answer: (B) Beta therapy", "B"),
    ],
)
def test_common_markdown_answer_formats_are_parseable(text: str, expected: str):
    parsed = extract_answer(text, CHOICES)
    assert parsed.status == "parsed"
    assert parsed.option == expected
