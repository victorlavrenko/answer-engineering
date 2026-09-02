from __future__ import annotations

from instruction_duplication.audit import HUMAN_AUDIT_VERSION, ROLE_AUDIT_GUIDANCE, ROLE_SPECS
from instruction_duplication.judge import judge
from instruction_duplication.provider import fake_response
from instruction_duplication.records import GenerationCell


def valid_response(question):
    cell = GenerationCell(
        cell_id="b" * 64,
        question_id=question.id,
        model_id="test-model",
        condition_id="system",
        copies=1,
        dataset=question.dataset,
        stem=question.stem,
        choices=question.choices,
        gold=question.gold,
    )
    return fake_response(cell)["choices"][0]["message"]["content"]


def _semantic_sentence(question):
    other = next(label for label in question.choices if label != question.gold)
    return f"If the decisive finding changed to support {question.choices[other]}, option {other} would become best."


def test_validated_aggregate_uses_simple_step6_nontrivial_metric(question, lexical_reference):
    raw = valid_response(question)
    old = _semantic_sentence(question)
    assert old in raw
    raw = raw.replace(
        old,
        "This section contains meaningful generated material, but it does not explicitly establish that the stated best alternative would become the winning answer.",
        1,
    )
    r = judge(question, "completed", raw, "system", lexical_reference)
    assert r["role_answer_changing_change_complete"] == 0.0
    assert r["role_answer_changing_change_nontrivial"] == 1.0
    assert r["validated_role_count"] == 8
    assert r["all_roles_validated_complete"] == 1.0
    assert r["validated_complete_role_scaffold"] == 1.0


def test_short_step6_filler_fails_validated_aggregate(question, lexical_reference):
    raw = valid_response(question)
    old = _semantic_sentence(question)
    assert old in raw
    raw = raw.replace(old, "See above.", 1)
    r = judge(question, "completed", raw, "system", lexical_reference)
    assert r["role_answer_changing_change_nontrivial"] == 0.0
    assert r["validated_role_count"] == 7
    assert r["all_roles_validated_complete"] == 0.0


def test_human_audit_v9_mirrors_simple_step6_construct():
    assert HUMAN_AUDIT_VERSION == "atomic-blinded-pairs-v9-step6-nontrivial"
    spec = next(s for s in ROLE_SPECS if s[0] == "answer_changing_change")
    assert spec[1] == "role_answer_changing_change_nontrivial"
    assert spec[3] == ("answer_changing_change",)
    assert "non-trivial generated content" in ROLE_AUDIT_GUIDANCE["answer_changing_change"]["rule"]
