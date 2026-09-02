from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from instruction_duplication.datasets_loader import (
    DATASET_SPECS,
    DatasetNormalizationError,
    load_hf_dataset,
    normalize_row,
    select_questions,
)
from instruction_duplication.json_types import object_value
from instruction_duplication.protocol import format_question


def base_row():
    return {
        "id": "1",
        "question": "A patient has sudden hearing loss. What is the diagnosis?",
        "options": {"A": "SSNHL", "B": "Cerumen"},
        "answer_idx": "0",
    }


def test_malformed_earlier_gold_field_does_not_block_valid_later_field():
    row = base_row() | {"answer_idx": "bad", "correct_answer": "A"}
    question = normalize_row(row, dataset="d", source_split="test", index=0)
    assert question.gold == "A"
    assert question.gold_source == "correct_answer"


def test_conflicting_gold_fields_are_rejected():
    row = base_row() | {"correct_answer": "B"}
    with pytest.raises(DatasetNormalizationError, match="conflicting gold"):
        normalize_row(row, dataset="d", source_split="test", index=0)


def test_conflicting_choice_encodings_are_rejected():
    row = base_row() | {"choices": {"A": "Different", "B": "Cerumen"}}
    with pytest.raises(DatasetNormalizationError, match="conflicting choice"):
        normalize_row(row, dataset="d", source_split="test", index=0)


@pytest.mark.parametrize(
    "stem",
    [
        "The radiograph demonstrates a lesion. What is it?",
        "In the following table, which value is abnormal?",
        "The ECG shown below indicates what diagnosis?",
    ],
)
def test_media_dependent_questions_are_rejected(stem):
    row = base_row() | {"question": stem}
    with pytest.raises(DatasetNormalizationError, match="media"):
        normalize_row(row, dataset="d", source_split="test", index=0)


def test_local_selection_filters_deduplicates_and_uses_seed(tmp_path: Path):
    rows = []
    for dataset in ("d1", "d2"):
        rows.extend(
            {
                "id": f"{dataset}-{index}",
                "dataset": dataset,
                "stem": f"Unique {dataset} stem {index}",
                "choices": {"A": "Yes", "B": "No"},
                "gold": "A",
                "gold_text": "Yes",
                "gold_source": "fixture",
                "gold_raw": "A",
                "source_split": "test",
            }
            for index in range(3)
        )
        rows.append(rows[-1] | {"id": f"{dataset}-duplicate"})
    path = tmp_path / "input.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    first, audit = select_questions(
        dataset_names=["d1", "d2"], questions_per_dataset=2, seed=7, input_jsonl=path
    )
    second, _ = select_questions(
        dataset_names=["d1", "d2"], questions_per_dataset=2, seed=7, input_jsonl=path
    )
    assert [q.id for q in first] == [q.id for q in second]
    assert len(first) == 4
    assert object_value(audit["selection"], name="audit.selection")["seed"] == 7


def test_pinned_hf_source_is_passed_without_fallback(monkeypatch):
    calls = []

    def fake_load_dataset(repository, config, **kwargs):
        calls.append((repository, config, kwargs))
        return [base_row()]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )
    questions, audit = load_hf_dataset("medqa")
    spec = DATASET_SPECS["medqa"]
    assert len(questions) == 1
    assert calls == [
        (spec.repository, spec.config, {"split": spec.split, "revision": spec.revision})
    ]
    assert object_value(audit["source"], name="audit.source")["revision"] == spec.revision


def test_all_dataset_revisions_are_full_hashes():
    assert all(len(spec.revision) == 40 for spec in DATASET_SPECS.values())


def test_afrimedqa_string_options_and_clean_question_are_normalized():
    row = {
        "sample_id": "sample-1",
        "split": "test",
        "question_type": "MCQ",
        "question": "uncleanquestion",
        "question_clean": "Clean question?",
        "answer_options": "{'option1': 'First', 'option2': 'Second', 'option3': 'Third'}",
        "correct_answer": "option2",
    }
    question = normalize_row(
        row,
        dataset="afrimedqa",
        source_split="internal test partition",
        index=0,
    )
    assert question.id == "afrimedqa:sample-1"
    assert question.stem == "Clean question?"
    assert dict(question.choices) == {"A": "First", "B": "Second", "C": "Third"}
    assert question.gold == "B"


def test_embedded_answer_choices_are_removed_only_when_they_match_structured_options():
    row = {
        "id": "medxpert-1",
        "question": (
            "A patient has a finding. What is the diagnosis?\n"
            "Answer Choices: (A) Alpha disease (B) Beta disease (C) Gamma disease"
        ),
        "options": {"A": "Alpha disease", "B": "Beta disease", "C": "Gamma disease"},
        "answer_idx": "A",
    }
    question = normalize_row(
        row,
        dataset="medxpertqa",
        source_split="test",
        index=0,
        embedded_choice_block=True,
    )
    assert question.stem == "A patient has a finding. What is the diagnosis?"
    assert "Answer Choices" not in question.stem


def test_mismatched_embedded_answer_choices_are_rejected_in_required_source():
    row = {
        "id": "medxpert-1",
        "question": "Question? Answer Choices: (A) Wrong text (B) Beta disease",
        "options": {"A": "Alpha disease", "B": "Beta disease"},
        "answer_idx": "A",
    }
    with pytest.raises(DatasetNormalizationError, match="expected terminal Answer Choices"):
        normalize_row(
            row,
            dataset="medxpertqa",
            source_split="test",
            index=0,
            embedded_choice_block=True,
        )


def test_question_exclusion_accepts_current_and_hidden_legacy_workspaces(tmp_path: Path):
    from instruction_duplication.exclusions import load_question_exclusions

    current = tmp_path / "current"
    legacy = tmp_path / "legacy"
    (current / "data").mkdir(parents=True)
    (legacy / ".instruction-duplication" / "data").mkdir(parents=True)
    current_row = {
        "id": "d1-old",
        "dataset": "d1",
        "stem": "Unique d1 stem 0",
        "choices": {"A": "Yes", "B": "No"},
        "gold": "A",
        "gold_text": "Yes",
        "gold_source": "fixture",
        "gold_raw": "A",
        "source_split": "test",
    }
    legacy_row = current_row | {"id": "d2-old", "dataset": "d2", "stem": "Unique d2 stem 0"}
    (current / "data" / "questions.jsonl").write_text(
        json.dumps(current_row) + "\n", encoding="utf-8"
    )
    (legacy / ".instruction-duplication" / "data" / "questions.jsonl").write_text(
        json.dumps(legacy_row) + "\n", encoding="utf-8"
    )
    exclusions, identity, audit = load_question_exclusions((current, legacy))
    assert current_row["id"] in exclusions.ids
    assert legacy_row["id"] in exclusions.ids
    assert identity["unique_ids"] == 2
    assert len(audit) == 2


def test_medxpert_cleaned_prompt_contains_each_structured_choice_once():
    row = {
        "id": "medxpert-2",
        "question": (
            "Which diagnosis fits? Answer Choices: "
            "(A) Alpha disease (B) Beta disease (C) Gamma disease"
        ),
        "options": {"A": "Alpha disease", "B": "Beta disease", "C": "Gamma disease"},
        "answer_idx": "B",
    }
    question = normalize_row(
        row,
        dataset="medxpertqa",
        source_split="test",
        index=0,
        embedded_choice_block=True,
    )
    rendered = format_question(question.stem, question.choices)
    assert "Answer Choices" not in rendered
    for choice in question.choices.values():
        assert rendered.count(choice) == 1


def test_cross_dataset_duplicate_stem_is_resampled(tmp_path: Path):
    rows = [
        {
            "id": "d1-shared",
            "dataset": "d1",
            "stem": "Exactly shared question?",
            "choices": {"A": "Yes", "B": "No"},
            "gold": "A",
            "gold_text": "Yes",
            "gold_source": "fixture",
            "gold_raw": "A",
            "source_split": "test",
        },
        {
            "id": "d2-shared",
            "dataset": "d2",
            "stem": "Exactly shared question?",
            "choices": {"A": "Yes", "B": "No"},
            "gold": "A",
            "gold_text": "Yes",
            "gold_source": "fixture",
            "gold_raw": "A",
            "source_split": "test",
        },
        {
            "id": "d2-unique",
            "dataset": "d2",
            "stem": "A different question?",
            "choices": {"A": "Yes", "B": "No"},
            "gold": "A",
            "gold_text": "Yes",
            "gold_source": "fixture",
            "gold_raw": "A",
            "source_split": "test",
        },
    ]
    path = tmp_path / "input.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    selected, audit = select_questions(
        dataset_names=["d1", "d2"],
        questions_per_dataset=1,
        seed=11,
        input_jsonl=path,
    )
    assert [question.id for question in selected] == ["d1-shared", "d2-unique"]
    dataset_audits = audit["datasets"]
    assert isinstance(dataset_audits, list)
    assert (
        object_value(dataset_audits[1], name="audit.datasets[1]")[
            "cross_dataset_duplicates_removed"
        ]
        == 1
    )


def test_prior_workspace_stem_exclusion_is_global_across_dataset_names(tmp_path: Path):
    from instruction_duplication.exclusions import load_question_exclusions
    from instruction_duplication.types import Question

    prior = tmp_path / "prior"
    (prior / "data").mkdir(parents=True)
    old = {
        "id": "old-source-id",
        "dataset": "old-dataset",
        "stem": "Same underlying question?",
    }
    (prior / "data" / "questions.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    exclusions, identity, _audit = load_question_exclusions((prior,))
    candidate = Question(
        id="new-source-id",
        dataset="new-dataset",
        stem="Same underlying question?",
        choices={"A": "Yes", "B": "No"},
        gold="A",
        gold_text="Yes",
        gold_source="fixture",
        gold_raw="A",
        source_split="test",
    )
    assert exclusions.contains(candidate)
    assert identity["unique_stems"] == 1


def test_prior_dirty_medxpert_stem_excludes_cleaned_version_even_if_source_id_changes(
    tmp_path: Path,
):
    from instruction_duplication.exclusions import load_question_exclusions
    from instruction_duplication.types import Question

    prior = tmp_path / "prior"
    (prior / "data").mkdir(parents=True)
    old = {
        "id": "medxpertqa:Text-42",
        "dataset": "medxpertqa",
        "stem": "Question? Answer Choices: (A) Alpha (B) Beta",
        "choices": {"A": "Alpha", "B": "Beta"},
    }
    (prior / "data" / "questions.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    exclusions, _identity, _audit = load_question_exclusions((prior,))
    cleaned = Question(
        id="renamed-source:42",
        dataset="renamed-source",
        stem="Question?",
        choices={"A": "Alpha", "B": "Beta"},
        gold="A",
        gold_text="Alpha",
        gold_source="fixture",
        gold_raw="A",
        source_split="test",
    )
    assert exclusions.contains(cleaned)


def test_question_exclusion_accepts_historical_hidden_workspace_layout(tmp_path: Path):
    from instruction_duplication.exclusions import load_question_exclusions
    from instruction_duplication.types import Question

    prior = tmp_path / "prior"
    ledger = prior / ".instruction-duplication" / "data" / "questions.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "id": "legacy-id",
                "dataset": "legacy-dataset",
                "stem": "Previously used question?",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    exclusions, identity, _audit = load_question_exclusions((prior,))
    candidate = Question(
        id="new-id",
        dataset="new-dataset",
        stem="Previously used question?",
        choices={"A": "Yes", "B": "No"},
        gold="A",
        gold_text="Yes",
        gold_source="fixture",
        gold_raw="A",
        source_split="test",
    )
    assert exclusions.contains(candidate)
    assert identity["unique_stems"] == 1
