from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from answer_engineering import GenerationPolicy
from answer_engineering.rules.compile.compiler import (
    FullPlanCompiler,
)
from answer_engineering.rules.parse.ast import (
    AfterRuleAST,
    AvoidRuleAST,
    ForceRuleAST,
    ReplaceRuleAST,
    RuleAST,
)
from answer_engineering.rules.parse.parser import (
    MarkdownRulesParser,
)

RULESET_PATH = Path("tests/fixtures/ent_ssnhl_doctor_rules.ae")
RULES_AST_GOLDEN_PATH = Path("tests/golden/ent_ssnhl_ruleset_ast.json")
PLAN_IR_GOLDEN_PATH = Path("tests/golden/ent_ssnhl_plan_ir.json")
PROMPT_GOLDEN_PATH = Path("tests/golden/default_system_prompt.txt")


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


def regenerate_goldens() -> None:
    RULES_AST_GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)

    ruleset = MarkdownRulesParser().parse(
        _trim_ruleset_front_matter(RULESET_PATH.read_text(encoding="utf-8"))
    )

    primitive = [_rule_to_primitive(rule) for rule in ruleset.rules]
    RULES_AST_GOLDEN_PATH.write_text(
        json.dumps(primitive, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compiled = FullPlanCompiler().compile(
        MarkdownRulesParser().parse(
            _trim_ruleset_front_matter(RULESET_PATH.read_text(encoding="utf-8"))
        )
    )
    plan_primitive = [
        {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "target_kind": rule.target.kind,
            "candidate_ops": [
                candidate.op.value for candidate in rule.candidates
            ],
            "candidate_kinds": [
                candidate.kind for candidate in rule.candidates
            ],
            "candidate_ids": [
                candidate.candidate_id for candidate in rule.candidates
            ],
            "allow_noop": rule.policy.allow_noop,
            "fire_mode": rule.fire.mode,
        }
        for rule in compiled.rules
    ]
    PLAN_IR_GOLDEN_PATH.write_text(
        json.dumps(plan_primitive, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    PROMPT_GOLDEN_PATH.write_text(
        GenerationPolicy.default_system_prompt, encoding="utf-8"
    )


if __name__ == "__main__":
    regenerate_goldens()
    print(f"Updated {RULES_AST_GOLDEN_PATH}")
    print(f"Updated {PLAN_IR_GOLDEN_PATH}")
    print(f"Updated {PROMPT_GOLDEN_PATH}")
    print("Please inspect diffs before committing.")
