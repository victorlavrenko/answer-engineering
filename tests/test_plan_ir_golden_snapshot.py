from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from answer_engineering.rules.compile.compiler import (
    FullPlanCompiler,
)
from answer_engineering.rules.parse.parser import (
    MarkdownRulesParser,
)

RULESET_PATH = Path("tests/fixtures/ent_ssnhl_doctor_rules.ae")
SNAPSHOT_PATH = Path("tests/golden/ent_ssnhl_plan_ir.json")


def _trim_ruleset_front_matter(text: str) -> str:
    marker = "\n---\n"
    if text.startswith("# Answer Engineering Rules") and marker in text:
        return text.split(marker, maxsplit=1)[1].lstrip()
    return text


def _plan_to_primitive() -> list[dict[str, Any]]:
    parser = MarkdownRulesParser()
    compiler = FullPlanCompiler()
    rules_text = _trim_ruleset_front_matter(
        RULESET_PATH.read_text(encoding="utf-8")
    )
    plan = compiler.compile(parser.parse(rules_text))
    out: list[dict[str, Any]] = []
    for rule in plan.rules:
        out.append(
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
        )
    return out


def test_plan_ir_compilation_matches_golden_snapshot() -> None:
    assert RULESET_PATH.exists()
    assert SNAPSHOT_PATH.exists()

    actual = _plan_to_primitive()
    expected_raw = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    expected = cast(list[dict[str, Any]], expected_raw)
    assert actual == expected
