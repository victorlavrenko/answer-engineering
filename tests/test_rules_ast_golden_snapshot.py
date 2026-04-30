from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from answer_engineering.rules.parse.ast import (
    AfterRuleAST,
    AvoidRuleAST,
    ForceRuleAST,
    ReplaceRuleAST,
    RuleAST,
)
from answer_engineering.rules.parse.parser import MarkdownRulesParser

RULESET_PATH = Path("tests/fixtures/ent_ssnhl_doctor_rules.ae")
SNAPSHOT_PATH = Path("tests/golden/ent_ssnhl_ruleset_ast.json")


def _trim_ruleset_front_matter(text: str) -> str:
    marker = "\n---\n"
    if text.startswith("# Answer Engineering Rules") and marker in text:
        return text.split(marker, maxsplit=1)[1].lstrip()
    return text


def _rule_to_primitive(rule: RuleAST) -> dict[str, Any]:
    base: dict[str, Any] = {
        "rule_id": rule.rule_id,
        "fire": rule.fire,
        "scope": {
            "kind": rule.scope.kind if rule.scope is not None else None,
            "n": rule.scope.n if rule.scope is not None else None,
            "casefold": rule.scope.casefold if rule.scope is not None else None,
        },
    }
    if isinstance(rule, ReplaceRuleAST):
        base.update(
            {
                "kind": "replace",
                "target": rule.target,
                "candidates": list(rule.candidates),
                "gate": {
                    "any_of": list(rule.gate.any_of)
                    if rule.gate is not None
                    else [],
                    "all_of": list(rule.gate.all_of)
                    if rule.gate is not None
                    else [],
                    "none_of": list(rule.gate.none_of)
                    if rule.gate is not None
                    else [],
                },
            }
        )
    elif isinstance(rule, AfterRuleAST):
        base.update(
            {
                "kind": "after",
                "target": rule.target,
                "candidates": list(rule.candidates),
                "gate": {
                    "any_of": list(rule.gate.any_of)
                    if rule.gate is not None
                    else [],
                    "all_of": list(rule.gate.all_of)
                    if rule.gate is not None
                    else [],
                    "none_of": list(rule.gate.none_of)
                    if rule.gate is not None
                    else [],
                },
            }
        )
    elif isinstance(rule, AvoidRuleAST):
        base.update(
            {
                "kind": "avoid",
                "target": rule.target,
                "edit": {
                    "kind": rule.edit.kind,
                    "n_sentences": rule.edit.n_sentences,
                    "n_clauses": rule.edit.n_clauses,
                },
                "required_before_all": list(rule.required_before_all),
                "required_before_any": list(rule.required_before_any),
                "connector_terms": list(rule.connector_terms),
                "required_after_any": list(rule.required_after_any),
                "fallback": list(rule.fallback),
                "options": dict(rule.options),
            }
        )
    elif isinstance(rule, ForceRuleAST):
        base.update(
            {
                "kind": "force",
                "target": rule.target,
                "add": list(rule.add),
            }
        )
    return base


def _first_diff(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> str:
    if len(actual) != len(expected):
        return f"length differs: actual={len(actual)} expected={len(expected)}"
    for idx, (actual_rule, expected_rule) in enumerate(zip(actual, expected)):
        if actual_rule != expected_rule:
            return f"first differing rule index={idx}"
    return "unknown diff"


def test_rules_ast_parser_matches_golden_snapshot() -> None:
    assert RULESET_PATH.exists(), f"Ruleset file missing: {RULESET_PATH}"
    assert SNAPSHOT_PATH.exists(), (
        f"Golden snapshot missing: {SNAPSHOT_PATH}. "
        "Run `python tests/regenerate_goldens.py` and inspect the diff."
    )

    ruleset_text = _trim_ruleset_front_matter(
        RULESET_PATH.read_text(encoding="utf-8")
    )
    ruleset = MarkdownRulesParser().parse(ruleset_text)
    actual = [_rule_to_primitive(rule) for rule in ruleset.rules]

    expected_raw = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    expected = cast(list[dict[str, Any]], expected_raw)

    assert actual == expected, (
        "Rules abstract-syntax-tree parser output drifted from snapshot; "
        f"{_first_diff(actual, expected)}. "
        "If intentional, regenerate with `python tests/regenerate_goldens.py`."
    )
