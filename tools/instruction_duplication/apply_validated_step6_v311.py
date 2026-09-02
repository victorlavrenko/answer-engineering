#!/usr/bin/env python3
"""Upgrade patched 3.0.10 to 3.0.11 paper-facing validated role metrics."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MARKER = 'PAPER_VALIDATED_ROLE_AGGREGATE = "v1"'


def r1(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {n}")
    return text.replace(old, new, 1)


def rf(text, old, new, label):
    n = text.count(old)
    if n < 1:
        raise RuntimeError(f"{label}: expected at least 1 match, found {n}")
    return text.replace(old, new, 1)


def patch_judge(text):
    if MARKER in text:
        return text
    if 'HUMAN_VALIDATED_COUNTERFACTUAL_JUDGE = "v1"' not in text:
        raise RuntimeError("expected 3.0.10 human-validated judge")
    text = r1(
        text,
        'HUMAN_VALIDATED_COUNTERFACTUAL_JUDGE = "v1"\n',
        'HUMAN_VALIDATED_COUNTERFACTUAL_JUDGE = "v1"\n' + MARKER + "\n",
        "marker",
    )
    text = r1(
        text,
        "    contrastive_discussion_count: int | None\n    contrastive_discussion_score: float | None\n",
        "    contrastive_discussion_count: int | None\n    contrastive_discussion_score: float | None\n    validated_contrastive_discussion_count: int | None\n    validated_contrastive_discussion_score: float | None\n",
        "typed contrastive",
    )
    text = r1(
        text,
        "    substantive_role_count: int | None\n",
        "    substantive_role_count: int | None\n    validated_role_count: int | None\n    validated_role_completeness_score: float | None\n",
        "typed roles",
    )
    text = r1(
        text,
        "    all_roles_substantive: float | None\n",
        "    all_roles_substantive: float | None\n    all_roles_validated_complete: float | None\n    validated_complete_role_scaffold: float | None\n",
        "typed all roles",
    )
    text = r1(
        text,
        "    role_answer_changing_change_complete: float | None\n",
        "    role_answer_changing_change_complete: float | None\n    role_answer_changing_change_nontrivial: float | None\n",
        "typed step6",
    )
    text = r1(
        text,
        '        "contrastive_discussion_score": all_question_zero,\n',
        '        "contrastive_discussion_score": all_question_zero,\n        "validated_contrastive_discussion_count": 0 if instructed else None,\n        "validated_contrastive_discussion_score": all_question_zero,\n',
        "failure contrastive",
    )
    text = r1(
        text,
        '        "substantive_role_count": 0 if instructed else None,\n',
        '        "substantive_role_count": 0 if instructed else None,\n        "validated_role_count": 0 if instructed else None,\n        "validated_role_completeness_score": all_question_zero,\n',
        "failure roles",
    )
    text = r1(
        text,
        '        "all_roles_substantive": all_question_zero,\n',
        '        "all_roles_substantive": all_question_zero,\n        "all_roles_validated_complete": all_question_zero,\n        "validated_complete_role_scaffold": all_question_zero,\n',
        "failure all roles",
    )
    text = r1(
        text,
        '        "role_answer_changing_change_complete": all_question_zero,\n',
        '        "role_answer_changing_change_complete": all_question_zero,\n        "role_answer_changing_change_nontrivial": all_question_zero,\n',
        "failure step6",
    )
    text = r1(
        text,
        '        "contrastive_discussion_score": None,\n',
        '        "contrastive_discussion_score": None,\n        "validated_contrastive_discussion_count": None,\n        "validated_contrastive_discussion_score": None,\n',
        "baseline contrastive",
    )
    text = r1(
        text,
        '        "substantive_role_count": None,\n',
        '        "substantive_role_count": None,\n        "validated_role_count": None,\n        "validated_role_completeness_score": None,\n',
        "baseline roles",
    )
    text = r1(
        text,
        '        "all_roles_substantive": None,\n',
        '        "all_roles_substantive": None,\n        "all_roles_validated_complete": None,\n        "validated_complete_role_scaffold": None,\n',
        "baseline all roles",
    )
    text = r1(
        text,
        '        "role_answer_changing_change_complete": None,\n',
        '        "role_answer_changing_change_complete": None,\n        "role_answer_changing_change_nontrivial": None,\n',
        "baseline step6",
    )
    old = "    nontrivial = _nontrivial_sections(recovered)\n    section_diagnostics = _section_diagnostics(recovered, substantive, nontrivial)\n"
    new = """    nontrivial = _nontrivial_sections(recovered)\n    # Paper-facing validated completion uses seven semantic role checks plus a\n    # deliberately simpler Step-6 observable: non-trivial generated content.\n    # The strict semantic Step-6 result remains exported as exploratory.\n    validated_substantive = dict(substantive)\n    validated_substantive["answer_changing_change"] = bool(\n        nontrivial["answer_changing_change"]\n    )\n    validated_role_count = sum(bool(value) for value in validated_substantive.values())\n    validated_role_completeness_score = validated_role_count / len(CONTENT_TAGS)\n    all_roles_validated_complete = bool(all(validated_substantive.values()))\n    section_diagnostics = _section_diagnostics(recovered, substantive, nontrivial)\n"""
    text = r1(text, old, new, "validated computation")
    old = "    contrastive_discussion_count = sum(bool(value) for value in contrastive_discussion_components)\n    contrastive_discussion_score = contrastive_discussion_count / len(contrastive_discussion_components)\n"
    new = (
        old
        + """    validated_contrastive_discussion_components = (\n        validated_substantive["provisional_answer"],\n        validated_substantive["second_best"],\n        validated_substantive["decisive_fact"],\n        validated_substantive["answer_changing_change"],\n        validated_substantive["rereasoning"],\n    )\n    validated_contrastive_discussion_count = sum(\n        bool(value) for value in validated_contrastive_discussion_components\n    )\n    validated_contrastive_discussion_score = (\n        validated_contrastive_discussion_count / len(validated_contrastive_discussion_components)\n    )\n"""
    )
    text = r1(text, old, new, "validated contrastive")
    old = """    complete_role_scaffold = bool(\n        all_roles_substantive\n        and all(section_diagnostics["unique"].values())\n        and roles_in_requested_order\n    )\n"""
    new = (
        old
        + """    validated_complete_role_scaffold = bool(\n        all_roles_validated_complete\n        and all(section_diagnostics["unique"].values())\n        and roles_in_requested_order\n    )\n"""
    )
    text = r1(text, old, new, "validated scaffold")
    text = r1(
        text,
        '        "contrastive_discussion_score": contrastive_discussion_score,\n',
        '        "contrastive_discussion_score": contrastive_discussion_score,\n        "validated_contrastive_discussion_count": validated_contrastive_discussion_count,\n        "validated_contrastive_discussion_score": validated_contrastive_discussion_score,\n',
        "return contrastive",
    )
    text = r1(
        text,
        '        "substantive_role_count": substantive_role_count,\n',
        '        "substantive_role_count": substantive_role_count,\n        "validated_role_count": validated_role_count,\n        "validated_role_completeness_score": validated_role_completeness_score,\n',
        "return roles",
    )
    text = r1(
        text,
        '        "all_roles_substantive": float(all_roles_substantive),\n',
        '        "all_roles_substantive": float(all_roles_substantive),\n        "all_roles_validated_complete": float(all_roles_validated_complete),\n        "validated_complete_role_scaffold": float(validated_complete_role_scaffold),\n',
        "return all roles",
    )
    text = r1(
        text,
        '        "role_answer_changing_change_complete": float(substantive["answer_changing_change"]),\n',
        '        "role_answer_changing_change_complete": float(substantive["answer_changing_change"]),\n        "role_answer_changing_change_nontrivial": float(validated_substantive["answer_changing_change"]),\n',
        "return step6",
    )
    return text


def patch_stats(text):
    if '"all_roles_validated_complete"' in text and '"validated_role_count"' in text:
        return text
    text = r1(
        text,
        '    "contrastive_discussion_score",\n    "substantive_role_count",\n    "complete_role_scaffold",\n',
        '    "validated_contrastive_discussion_score",\n    "validated_role_count",\n    "all_roles_validated_complete",\n',
        "primary",
    )
    text = r1(
        text,
        '    "contrastive_discussion_score",\n    "substantive_role_count",\n    "preprovisional_commitment_itt",\n    "accuracy",\n',
        '    "validated_contrastive_discussion_score",\n    "validated_role_count",\n    "preprovisional_commitment_itt",\n    "accuracy",\n',
        "key secondary",
    )
    # First occurrence = role diagnostics.
    text = rf(
        text,
        '    "contrastive_discussion_count",\n    "contrastive_discussion_score",\n',
        '    "contrastive_discussion_count",\n    "contrastive_discussion_score",\n    "validated_contrastive_discussion_count",\n    "validated_contrastive_discussion_score",\n',
        "role diagnostic contrastive",
    )
    text = rf(
        text,
        '    "substantive_role_count",\n    "role_completeness_score",\n    "all_roles_substantive",\n',
        '    "substantive_role_count",\n    "role_completeness_score",\n    "all_roles_substantive",\n    "validated_role_count",\n    "validated_role_completeness_score",\n    "all_roles_validated_complete",\n',
        "role diagnostic aggregate",
    )
    text = rf(
        text,
        '    "complete_role_scaffold",\n    "role_facts_complete",\n',
        '    "complete_role_scaffold",\n    "validated_complete_role_scaffold",\n    "role_facts_complete",\n',
        "role diagnostic scaffold",
    )
    text = rf(
        text,
        '    "role_answer_changing_change_complete",\n    "role_reconsideration_complete",\n',
        '    "role_answer_changing_change_complete",\n    "role_answer_changing_change_nontrivial",\n    "role_reconsideration_complete",\n',
        "role diagnostic step6",
    )
    text = r1(
        text,
        '    "nontrivial_section_count",\n    "preprovisional_tfidf_recall",\n    "contrastive_discussion_score",\n    "atomic_fact_coverage",\n',
        '    "nontrivial_section_count",\n    "preprovisional_tfidf_recall",\n    "validated_contrastive_discussion_score",\n    "atomic_fact_coverage",\n',
        "complexity",
    )
    text = r1(
        text,
        '    "required_section_count",\n    "nontrivial_section_count",\n    "preprovisional_tfidf_recall",\n    "contrastive_discussion_score",\n    "accuracy",\n',
        '    "required_section_count",\n    "nontrivial_section_count",\n    "preprovisional_tfidf_recall",\n    "validated_contrastive_discussion_score",\n    "accuracy",\n',
        "factorial",
    )
    # Remaining occurrences belong to ZERO_AS_ABSENT_PROTOCOL.
    for old, new, label in [
        (
            '    "contrastive_discussion_count",\n    "contrastive_discussion_score",\n',
            '    "contrastive_discussion_count",\n    "contrastive_discussion_score",\n    "validated_contrastive_discussion_count",\n    "validated_contrastive_discussion_score",\n',
            "zero contrastive",
        ),
        (
            '    "substantive_role_count",\n    "role_completeness_score",\n    "all_roles_substantive",\n',
            '    "substantive_role_count",\n    "role_completeness_score",\n    "all_roles_substantive",\n    "validated_role_count",\n    "validated_role_completeness_score",\n    "all_roles_validated_complete",\n',
            "zero aggregate",
        ),
        (
            '    "complete_role_scaffold",\n    "role_facts_complete",\n',
            '    "complete_role_scaffold",\n    "validated_complete_role_scaffold",\n    "role_facts_complete",\n',
            "zero scaffold",
        ),
        (
            '    "role_answer_changing_change_complete",\n    "role_reconsideration_complete",\n',
            '    "role_answer_changing_change_complete",\n    "role_answer_changing_change_nontrivial",\n    "role_reconsideration_complete",\n',
            "zero step6",
        ),
    ]:
        prefix, zero = text.split("ZERO_AS_ABSENT_PROTOCOL = {", 1)
        zero = r1(zero, old, new, label)
        text = prefix + "ZERO_AS_ABSENT_PROTOCOL = {" + zero
    return text


def patch_report(text):
    if '"all_roles_validated_complete": "All 8 roles meet validated completion criteria"' in text:
        return text
    text = r1(
        text,
        '    "contrastive_discussion_score": "Provisional/contrastive discussion score",\n',
        '    "contrastive_discussion_score": "Provisional/contrastive discussion score",\n    "validated_contrastive_discussion_count": "Validated provisional/contrastive stages completed (of 5)",\n    "validated_contrastive_discussion_score": "Validated provisional/contrastive discussion score",\n',
        "labels contrastive",
    )
    text = r1(
        text,
        '    "substantive_role_count": "Substantively completed reasoning roles (of 8)",\n',
        '    "substantive_role_count": "Substantively completed reasoning roles (of 8; exploratory semantic Step 6)",\n    "validated_role_count": "Roles meeting validated completion criteria (of 8)",\n    "validated_role_completeness_score": "Validated role-completion fraction",\n',
        "labels roles",
    )
    text = r1(
        text,
        '    "all_roles_substantive": "All 8 roles substantively completed",\n',
        '    "all_roles_substantive": "All 8 roles substantively completed (exploratory semantic Step 6)",\n    "all_roles_validated_complete": "All 8 roles meet validated completion criteria",\n',
        "labels all",
    )
    text = r1(
        text,
        '    "complete_role_scaffold": "Complete ordered substantive role scaffold",\n',
        '    "complete_role_scaffold": "Complete ordered substantive role scaffold (exploratory semantic Step 6)",\n    "validated_complete_role_scaffold": "Complete ordered validated role scaffold",\n',
        "labels scaffold",
    )
    text = r1(
        text,
        '    "role_answer_changing_change_complete": "What-would-change section substantive",\n',
        '    "role_answer_changing_change_complete": "What-would-change semantic completion (exploratory)",\n    "role_answer_changing_change_nontrivial": "What-would-change section has non-trivial content",\n',
        "labels step6",
    )
    text = r1(
        text,
        '    "contrastive_discussion_score",\n    "substantive_role_count",\n    "all_roles_substantive",\n    "complete_role_scaffold",\n',
        '    "validated_contrastive_discussion_score",\n    "validated_role_count",\n    "all_roles_validated_complete",\n    "validated_complete_role_scaffold",\n',
        "headline aggregate",
    )
    text = rf(
        text,
        '    "role_answer_changing_change_complete",\n    "role_reconsideration_complete",\n',
        '    "role_answer_changing_change_nontrivial",\n    "role_reconsideration_complete",\n',
        "headline step6",
    )
    old = """        "The primary role-completeness outcome counts identifiable, substantive reasoning "\n        "roles and is independent of XML, Markdown, or heading punctuation. Presence, "\n        "uniqueness, order, each role's completion, lexical exposure, premature commitment, "\n        "generation success, and answer accuracy are reported separately rather than added "\n        "into a composite. TF-IDF and automatic fact/qualifier checks measure visible lexical "\n        "exposure, not medical truth. Length robustness reports both pre-answer content-token "\n        "count and credited TF-IDF mass per 100 content tokens. Accuracy uses the option or "\n        "unambiguous answer text in the Final answer role."\n"""
    new = """        "Paper-facing validated role completion uses seven human-validated semantic role "\n        "judges plus a deliberately simple Step-6 observable: the identified What-would-change "\n        "section contains non-trivial generated content. The stricter semantic Step-6 "\n        "counterfactual judge is retained only as an exploratory diagnostic and does not enter "\n        "the validated aggregates. Presence, uniqueness, order, lexical exposure, premature "\n        "commitment, generation success, and answer accuracy remain separate measurements. "\n        "TF-IDF and automatic fact/qualifier checks measure visible lexical exposure, not medical truth."\n"""
    text = r1(text, old, new, "full method note")
    old = """            "identified_role_count",\n            "unique_role_count",\n            "substantive_role_count",\n            "all_roles_substantive",\n            "roles_in_requested_order",\n            "complete_role_scaffold",\n            "role_facts_complete",\n            "role_implications_complete",\n            "role_provisional_answer_complete",\n            "role_best_alternative_complete",\n            "role_decisive_distinction_complete",\n            "role_answer_changing_change_complete",\n            "role_reconsideration_complete",\n            "role_final_answer_complete","""
    new = """            "identified_role_count",\n            "unique_role_count",\n            "validated_role_count",\n            "all_roles_validated_complete",\n            "roles_in_requested_order",\n            "validated_complete_role_scaffold",\n            "role_facts_complete",\n            "role_implications_complete",\n            "role_provisional_answer_complete",\n            "role_best_alternative_complete",\n            "role_decisive_distinction_complete",\n            "role_answer_changing_change_nontrivial",\n            "role_reconsideration_complete",\n            "role_final_answer_complete","""
    text = r1(text, old, new, "paper role set")
    old = """                "substantive_role_count",\n                "all_roles_substantive",\n                "complete_role_scaffold",\n                "role_facts_complete",\n                "role_implications_complete",\n                "role_provisional_answer_complete",\n                "role_best_alternative_complete",\n                "role_decisive_distinction_complete",\n                "role_answer_changing_change_complete",\n                "role_reconsideration_complete","""
    new = """                "validated_role_count",\n                "all_roles_validated_complete",\n                "validated_complete_role_scaffold",\n                "role_facts_complete",\n                "role_implications_complete",\n                "role_provisional_answer_complete",\n                "role_best_alternative_complete",\n                "role_decisive_distinction_complete",\n                "role_answer_changing_change_nontrivial",\n                "role_reconsideration_complete","""
    text = r1(text, old, new, "paper model order")
    old = """                "  Judge v2 reports four core constructs separately: required-section presence, "\n                "non-trivial section content, PubMed-IDF lexical exposure before the provisional "\n                "answer, and substantive provisional/contrastive discussion. No composite score "\n                "is formed. The 2026-08-12 run was used to develop this judge and is exploratory."\n"""
    new = """                "  Paper-facing validated role metrics use seven semantic role judges plus "\n                "non-trivial Step-6 content. The stricter semantic Step-6 counterfactual judge "\n                "is retained as an exploratory diagnostic because it did not generalize reliably "\n                "under blinded human validation. No medical-correctness judgment is implied."\n"""
    text = r1(text, old, new, "paper method note")
    return text


def patch_test_stats(text):
    if '"validated_role_count": 8.0 * value' in text:
        return text
    text = r1(
        text,
        '                        "substantive_role_count": 8.0 * value,\n',
        '                        "substantive_role_count": 8.0 * value,\n                        "validated_role_count": 8.0 * value,\n                        "validated_role_completeness_score": value,\n',
        "test roles",
    )
    text = r1(
        text,
        '                        "all_roles_substantive": value,\n',
        '                        "all_roles_substantive": value,\n                        "all_roles_validated_complete": value,\n                        "validated_complete_role_scaffold": value,\n',
        "test all roles",
    )
    text = r1(
        text,
        '                        "role_answer_changing_change_complete": value,\n',
        '                        "role_answer_changing_change_complete": value,\n                        "role_answer_changing_change_nontrivial": value,\n',
        "test step6",
    )
    text = r1(
        text,
        '                        "contrastive_role_score": value,\n',
        '                        "contrastive_role_score": value,\n                        "validated_contrastive_discussion_count": 5.0 * value,\n                        "validated_contrastive_discussion_score": value,\n',
        "test contrastive",
    )
    old = """        "required_section_count",\n        "nontrivial_section_count",\n        "preprovisional_tfidf_recall",\n        "contrastive_discussion_score",\n        "substantive_role_count",\n        "complete_role_scaffold",\n        "preprovisional_commitment_itt","""
    new = """        "required_section_count",\n        "nontrivial_section_count",\n        "preprovisional_tfidf_recall",\n        "validated_contrastive_discussion_score",\n        "validated_role_count",\n        "all_roles_validated_complete",\n        "preprovisional_commitment_itt","""
    text = r1(text, old, new, "test expected primary")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    args = ap.parse_args()
    root = args.root.resolve()
    P = {
        k: root / v
        for k, v in {
            "judge": "src/instruction_duplication/judge.py",
            "stats": "src/instruction_duplication/stats.py",
            "report": "src/instruction_duplication/report.py",
            "init": "src/instruction_duplication/__init__.py",
            "test_stats": "tests/test_stats.py",
        }.items()
    }
    if any(not p.is_file() for p in P.values()):
        raise SystemExit("not an instruction-duplication source checkout")
    init = P["init"].read_text(encoding="utf-8")
    if '__version__ = "3.0.11"' in init and MARKER in P["judge"].read_text(encoding="utf-8"):
        print("3.0.11 already applied")
        return 0
    if '__version__ = "3.0.10"' not in init:
        raise SystemExit("overlay expects version 3.0.10")
    patched = {
        "judge": patch_judge(P["judge"].read_text(encoding="utf-8")),
        "stats": patch_stats(P["stats"].read_text(encoding="utf-8")),
        "report": patch_report(P["report"].read_text(encoding="utf-8")),
        "init": init.replace('__version__ = "3.0.10"', '__version__ = "3.0.11"', 1),
        "test_stats": patch_test_stats(P["test_stats"].read_text(encoding="utf-8")),
    }
    for k, p in P.items():
        b = p.with_suffix(p.suffix + ".v310.bak")
        if not b.exists():
            shutil.copy2(p, b)
        p.write_text(patched[k], encoding="utf-8")
    pkg = root / "PKG-INFO"
    if pkg.is_file():
        t = pkg.read_text(encoding="utf-8")
        if "Version: 3.0.10" in t:
            pkg.write_text(t.replace("Version: 3.0.10", "Version: 3.0.11", 1), encoding="utf-8")
    print("applied 3.0.11 validated Step-6 metric; generation code/prompts unchanged")
    print("rejudge and reanalyze existing generations; regeneration is not required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
