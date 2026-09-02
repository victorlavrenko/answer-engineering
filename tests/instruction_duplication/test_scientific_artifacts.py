from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from instruction_duplication.audit import export_blinded_matched_pairs
from instruction_duplication.facts import build_question_facts, score_fact_inventory
from instruction_duplication.pubmed_idf import read_document_frequencies
from instruction_duplication.schedule import schedule_key
from instruction_duplication.types import Question


def test_topic_only_question_is_not_repair_endpoint_eligible() -> None:
    question = Question(
        id="topic",
        dataset="test",
        stem="Concerning Dopamine Stress Echocardiography, which is true?",
        choices={"A": "One", "B": "Two"},
        gold="A",
        gold_text="One",
        gold_source="fixture",
        gold_raw="A",
        source_split="test",
    )
    inventory = build_question_facts(question)
    assert not inventory.applicable
    assert inventory.complexity == "inapplicable"


def _question_with_stem(stem: str) -> Question:
    return Question(
        id="stem-fixture",
        dataset="test",
        stem=stem,
        choices={"A": "One", "B": "Two"},
        gold="A",
        gold_text="One",
        gold_source="fixture",
        gold_raw="A",
        source_split="test",
    )


@pytest.mark.parametrize(
    "stem",
    (
        "Cullen's sign is seen in",
        "Group O individuals:",
        "Regarding normal deliveries",
        "Aganglionic megacolon typically presents with the following except",
        "the first step to improve oxygenation in a patient with chest trauma is",
        "With regard to treatment of CIN, which statement is false?",
    ),
)
def test_topic_or_completion_fragments_are_not_fact_repair_eligible(stem: str) -> None:
    assert build_question_facts(_question_with_stem(stem)).applicable is False


def test_conditional_scenario_prefix_remains_fact_repair_eligible() -> None:
    inventory = build_question_facts(
        _question_with_stem(
            "If tetanus is suspected in any wound, how many units of immunoglobulin should be given"
        )
    )
    assert inventory.applicable is True
    assert [fact.source_text for fact in inventory.facts] == [
        "If tetanus is suspected in any wound"
    ]


def test_relative_clause_is_not_mistaken_for_question_tail() -> None:
    inventory = build_question_facts(
        _question_with_stem(
            "A patient has pain, which is relieved by standing. Which diagnosis is most likely?"
        )
    )
    assert inventory.applicable is True
    assert inventory.facts[0].source_text == "A patient has pain, which is relieved by standing."


def test_atomic_fact_scoring_requires_qualifiers_and_inference_cues(question: Question) -> None:
    inventory = build_question_facts(question)
    assert inventory.complexity == "two_facts"
    complete = score_fact_inventory(
        inventory,
        "The sudden left hearing loss began 2 hours ago and there is no ear pain.",
        (
            "The sudden left hearing loss for 2 hours and no ear pain support a "
            "sensorineural process."
        ),
    )
    omitted = score_fact_inventory(
        inventory,
        "The patient has hearing loss and ear pain.",
        "Hearing loss is present.",
    )
    assert complete["atomic_fact_coverage"] == 1.0
    assert complete["atomic_implication_trace_coverage"] == 1.0
    assert omitted["atomic_fact_coverage"] == 0.0
    assert omitted["atomic_implication_trace_coverage"] == 0.0


def test_pubmed_reader_selects_terms_and_uses_terminal_document_total(tmp_path: Path) -> None:
    path = tmp_path / "counts.csv.gz"
    years = [str(year) for year in range(2010, 2019)]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("word," + ",".join(years) + "\n")
        handle.write("hearing," + ",".join("2" for _ in years) + "\n")
        handle.write("ordinary," + ",".join("7" for _ in years) + "\n")
        handle.write("," + ",".join("100" for _ in years) + "\n")
    documents, frequencies = read_document_frequencies(
        path,
        {"hearing", "missing"},
        verify_published_source=False,
    )
    assert documents == 900
    assert frequencies == {"hearing": 18, "missing": 0}


def test_schedule_key_is_stable_and_opaque() -> None:
    first = schedule_key("cell-a")
    assert first == schedule_key("cell-a")
    assert first != schedule_key("cell-b")
    assert len(first) == 64


def test_blinded_audit_keeps_condition_identity_only_in_key(tmp_path: Path) -> None:
    rows = [
        {
            "question_id": "q1",
            "model_id": "m1",
            "condition_id": condition,
            "dataset": "test",
            "stem": "A factual stem.",
            "choices": {"A": "One", "B": "Two"},
            "status": "completed",
            "content": "first response" if condition == "system" else "second response",
            "judgment": {"minimum_repair_scaffold": 1.0},
        }
        for condition in ("system", "system_before")
    ]
    audit = tmp_path / "audit.jsonl"
    key = tmp_path / "key.jsonl"
    schema = tmp_path / "schema.json"
    metadata = export_blinded_matched_pairs(
        rows,
        audit_path=audit,
        key_path=key,
        schema_path=schema,
    )
    blind_text = audit.read_text()
    key_row = json.loads(key.read_text())
    assert metadata["exported_pairs"] == 1
    assert "system_before" not in blind_text
    assert {key_row["response_a_condition"], key_row["response_b_condition"]} == {
        "system",
        "system_before",
    }
