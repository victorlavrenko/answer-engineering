from __future__ import annotations

import re

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false
from collections.abc import Mapping, Sequence
from pathlib import Path

from ae_paper_reproduction.core.aggregation.rule_stats import (
    AggregatedRunStats,
    CandidateTelemetry,
    ConditionTelemetry,
    RuleTelemetry,
    TelemetryItem,
    annotate_rules_with_run_stats,
)
from answer_engineering.engine.telemetry.aggregation.aggregator import (
    RuntimeTelemetryAggregator,
)
from answer_engineering.inference.decode.session_orchestration import (
    ExecutionSession,
)
from answer_engineering.rules.compile.compiled_rules import (
    CompiledRules,
)
from answer_engineering.rules.compile.compiler import (
    FullPlanCompiler,
)
from answer_engineering.rules.parse.parser import (
    MarkdownRulesParser,
)
from answer_engineering.telemetry import RuntimeTelemetrySnapshot
from tests._support.core_helpers import (
    step_test,
)


def _telemetry_items(
    rows: Sequence[Mapping[str, object]],
) -> tuple[TelemetryItem, ...]:
    """Build TelemetryItem rows using only the current telemetry schema."""
    items: list[TelemetryItem] = []
    for row in rows:
        raw_rules = row.get("rules", {})
        if not isinstance(raw_rules, Mapping):
            raw_rules = {}
        rules: list[RuleTelemetry] = []
        for rule_id, raw_rule in raw_rules.items():
            if not isinstance(raw_rule, Mapping):
                continue
            raw_conditions = raw_rule.get("conditions", {})
            if not isinstance(raw_conditions, Mapping):
                raw_conditions = {}
            conditions = tuple(
                ConditionTelemetry(
                    condition_id=str(
                        raw_condition.get("condition_id", condition_key) or ""
                    ),
                    node_path=str(raw_condition.get("node_path", "") or ""),
                    node_type=str(raw_condition.get("node_type", "") or ""),
                    debug_expression=str(
                        raw_condition.get("debug_expression", "") or ""
                    ),
                    matched=int(raw_condition.get("matched", 0) or 0),
                    seen=int(raw_condition.get("seen", 0) or 0),
                )
                for condition_key, raw_condition in raw_conditions.items()
                if isinstance(raw_condition, Mapping)
            )

            raw_candidates = raw_rule.get("candidate_choices", {})
            if not isinstance(raw_candidates, Mapping):
                raw_candidates = {}
            candidates = tuple(
                CandidateTelemetry(
                    kind=str(raw_candidate.get("kind", "") or ""),
                    candidate_id=str(
                        raw_candidate.get("candidate_id", candidate_key) or ""
                    ),
                    label=str(raw_candidate.get("label", "") or ""),
                    chosen=int(raw_candidate.get("chosen", 0) or 0),
                )
                for candidate_key, raw_candidate in raw_candidates.items()
                if isinstance(raw_candidate, Mapping)
            )

            rules.append(
                RuleTelemetry(
                    rule_id=str(raw_rule.get("rule_id", rule_id) or ""),
                    rule_name=str(raw_rule.get("rule_name", "") or ""),
                    evaluations=int(raw_rule.get("evaluations", 0) or 0),
                    applied=int(raw_rule.get("applied", 0) or 0),
                    conditions=conditions,
                    candidate_choices=candidates,
                )
            )
        items.append(
            TelemetryItem(
                RuntimeTelemetrySnapshot(
                    runtime_sec=None,
                    applied_decisions=int(row.get("applied_decisions", 0) or 0),
                    decision_limit_reached=bool(
                        row.get("decision_limit_reached", False)
                    ),
                    rules=tuple(rules),
                    events=tuple(),
                )
            )
        )
    return tuple(items)


def test_rules_parser_accepts_double_slash_comment_lines() -> None:
    rules = """
// ae-stats: evaluations=10 applied=5
## Replace (once): sensorineural hearing loss

Prefix:

- sudden

With:

- SSNHL

// ae-stats condition section=prefix op=any expression='sudden' matched=3/10

---
""".strip()
    parsed = MarkdownRulesParser().parse(rules)
    assert parsed.rules


def test_annotated_rules_parse_after_stats_injection() -> None:
    rules = """
## Replace (once): sensorineural hearing loss

Prefix:

- sudden

With:

- sudden sensorineural hearing loss
- SSNHL
""".strip()
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "applied_decisions": 2,
                    "decision_limit_reached": False,
                    "rules": {
                        rid: {
                            "rule_id": rid,
                            "rule_name": (
                                "Replace (once): sensorineural hearing loss"
                            ),
                            "evaluations": 4,
                            "applied": 2,
                            "conditions": {
                                "prefix:any:sudden": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "sudden",
                                    "matched": 3,
                                    "seen": 4,
                                }
                            },
                            "candidate_choices": {
                                "fallback:fallback_1": {
                                    "kind": "fallback",
                                    "label": "fallback_1",
                                    "chosen": 1,
                                }
                            },
                        }
                    },
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)
    assert "// ae-stats:" in annotated
    assert "## Replace (once): sensorineural hearing loss" in annotated

    parsed = MarkdownRulesParser().parse(
        _strip_rule_activity_summary(annotated)
    )
    assert parsed.rules


def _strip_rule_activity_summary(markdown: str) -> str:
    """Remove the synthetic 'Rule activity summary' section from annotated

    Intended for test normalization only. Assumes the summary section, if
    present, is a trailing block.

    """
    lines = markdown.splitlines()
    output: list[str] = []

    for line in lines:
        if line.strip() == "## Rule activity summary":
            break
        output.append(line)

    result = "\n".join(output)
    return f"{result}\n" if result else ""


def test_merge_ae_telemetry_accumulates_rule_condition_and_candidate_cnts() -> (
    None
):
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "applied_decisions": 1,
                    "rules": {
                        "rule-1": {
                            "rule_name": "rule one",
                            "evaluations": 2,
                            "applied": 1,
                            "conditions": {
                                "prefix:any:a": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "a",
                                    "matched": 1,
                                    "seen": 2,
                                }
                            },
                            "candidate_choices": {
                                "fallback:f1": {
                                    "kind": "fallback",
                                    "label": "f1",
                                    "chosen": 1,
                                }
                            },
                        }
                    },
                },
                {
                    "applied_decisions": 2,
                    "decision_limit_reached": True,
                    "rules": {
                        "rule-1": {
                            "rule_name": "rule one",
                            "evaluations": 3,
                            "applied": 2,
                            "conditions": {
                                "prefix:any:a": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "a",
                                    "matched": 2,
                                    "seen": 3,
                                }
                            },
                            "candidate_choices": {
                                "generated:g1": {
                                    "kind": "generated",
                                    "label": "g1",
                                    "chosen": 2,
                                }
                            },
                        }
                    },
                },
            ]
        )
    )

    assert merged.applied_decisions == 3
    assert merged.decision_limit_reached is True
    assert len(merged.rules) == 1
    rule = merged.rules[0]
    assert rule.evaluations == 5
    assert rule.applied == 3
    assert rule.fired_generations == 2
    assert rule.total_generations == 2
    assert {
        (c.kind, c.label, c.chosen, c.chosen_generations)
        for c in rule.candidate_choices
    } == {
        ("fallback", "f1", 1, 1),
        ("generated", "g1", 2, 1),
    }
    assert {
        (c.section, c.expression, c.matched, c.seen, c.matched_generations)
        for c in rule.conditions
    } == {("prefix", "a", 3, 5, 2)}


def test_annotate_rules_with_run_stats_reports_human_readable_breakdown() -> (
    None
):
    rules = "## Replace (once): sensorineural hearing loss\n"
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        rid: {
                            "rule_name": (
                                "Replace (once): sensorineural hearing loss"
                            ),
                            "evaluations": 10,
                            "applied": 1,
                            "conditions": {
                                "prefix:any:abrupt": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "abrupt",
                                    "matched": 1,
                                    "seen": 10,
                                }
                            },
                            "candidate_choices": {
                                "generated:sudden_sensorineural_hearing_loss": {
                                    "kind": "generated",
                                    "label": (
                                        "sudden sensorineural hearing loss"
                                    ),
                                    "chosen": 1,
                                }
                            },
                        }
                    }
                },
                {
                    "rules": {
                        "r1": {
                            "rule_name": (
                                "Replace (once): sensorineural hearing loss"
                            ),
                            "evaluations": 10,
                            "applied": 0,
                            "conditions": {
                                "prefix:any:abrupt": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "abrupt",
                                    "matched": 0,
                                    "seen": 10,
                                }
                            },
                            "candidate_choices": {
                                "generated:ssnhl": {
                                    "kind": "generated",
                                    "label": "SSNHL",
                                    "chosen": 0,
                                }
                            },
                        }
                    }
                },
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)

    assert "fired=1/2 (50.0%)" in annotated
    assert "avg_repeat_when_fired=1.00" in annotated


def test_annotate_matches_stats_by_compiled_rule_id_when_heading_differs() -> (
    None
):
    rules = """
## Replace (once): sensorineural hearing loss

With:

- SSNHL

## Avoid (repeat): hearing loss

With:

- auditory loss
""".strip()
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        "ng-r1": {
                            "rule_name": "replace:sensorineural hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                        },
                        "ng-r2": {
                            "rule_name": "avoid:hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                        },
                    }
                }
            ]
        )
    )

    parser = MarkdownRulesParser()
    plan = parser.parse(rules)
    compiled = FullPlanCompiler().compile(plan)
    first_rule_id = compiled.rules[0].rule_id
    second_rule_id = compiled.rules[1].rule_id

    # Rebuild merged stats with stable ids expected from runtime telemetry.
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        first_rule_id: {
                            "rule_name": "replace:sensorineural hearing loss",
                            "evaluations": 2,
                            "applied": 1,
                        },
                        second_rule_id: {
                            "rule_name": "avoid:hearing loss",
                            "evaluations": 2,
                            "applied": 1,
                        },
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)

    assert annotated.count("// ae-stats:") >= 3
    assert (
        f"// ae-rule-id: {first_rule_id} "
        "canonical=replace:sensorineural hearing loss" in annotated
    )
    assert (
        f"// ae-rule-id: {second_rule_id} canonical=avoid:hearing loss"
        in annotated
    )


def test_annotate_emits_rule_id_comments_and_output_is_parseable() -> None:
    rules = "## Replace (once): term\n"
    compiled = FullPlanCompiler().compile(MarkdownRulesParser().parse(rules))
    rid = compiled.rules[0].rule_id
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        rid: {
                            "rule_name": "replace:term",
                            "evaluations": 1,
                            "applied": 0,
                        }
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)

    assert f"// ae-rule-id: {rid} canonical=replace:term" in annotated
    assert (
        MarkdownRulesParser()
        .parse(_strip_rule_activity_summary(annotated))
        .rules
    )


def test_annotate_falls_back_when_parse_compile_fails() -> None:
    rules = "## Replace (once): term\n\n- malformed"
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "applied_decisions": 3,
                    "decision_limit_reached": False,
                    "rules": {
                        "r1": {
                            "rule_name": "Replace (once): term",
                            "evaluations": 4,
                            "applied": 1,
                        }
                    },
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)

    assert "// ae-rule-id:" not in annotated
    assert "fired=1/1 (100.0%)" in annotated
    assert "// ae-stats: run-summary" in annotated


def test_annotate_expanded_authored_heading_uses_structured_merge() -> None:
    rules = """
## Replace (once): hearing loss

Prefix (any):

- sudden | abrupt

Postfix: conductive
""".strip()
    compiled = FullPlanCompiler().compile(MarkdownRulesParser().parse(rules))
    assert len(compiled.rules) == 2
    expanded_rule_a = compiled.rules[0].rule_id
    expanded_rule_b = compiled.rules[1].rule_id

    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        expanded_rule_a: {
                            "rule_name": "replace:hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prefix:any:sudden": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "sudden",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "postfix:any:conductive": {
                                    "condition_id": "postfix:any:conductive",
                                    "node_path": "postfix",
                                    "node_type": "any",
                                    "debug_expression": "conductive",
                                    "matched": 1,
                                    "seen": 1,
                                },
                            },
                        },
                        expanded_rule_b: {
                            "rule_name": "replace:hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prefix:any:abrupt": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "abrupt",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "postfix:any:conductive:second": {
                                    "condition_id": "postfix:any:conductive",
                                    "node_path": "postfix",
                                    "node_type": "any",
                                    "debug_expression": "conductive",
                                    "matched": 1,
                                    "seen": 1,
                                },
                            },
                        },
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)

    assert "// ae-rule-id:" in annotated
    assert "fired=1/1 (100.0%)" in annotated
    assert "Postfix: conductive // ae-stats: matched=1/1" in annotated


def test_annotate_inline_scalar_fallback_receives_candidate_stats() -> None:
    rules = """
## Avoid (repeat): conductive

Fallback: some text
""".strip()
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        rid: {
                            "rule_name": "avoid:conductive",
                            "evaluations": 2,
                            "applied": 2,
                            "candidate_choices": {
                                "fallback:some text": {
                                    "kind": "fallback",
                                    "candidate_id": "some text",
                                    "label": "some text",
                                    "chosen": 2,
                                }
                            },
                        }
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)

    assert (
        "Fallback: some text // ae-stats: chosen=1/1 (100.0%) "
        "avg_hits_when_chosen=2.00 total_hits=2" in annotated
    )


def test_annotate_inline_bullet_behavior_unchanged_with_scalar_support() -> (
    None
):
    rules = """
## Replace (once): sensorineural hearing loss

Prefix:

- sudden
""".strip()
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        rid: {
                            "rule_name": "replace:sensorineural hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prefix:any:sudden": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "sudden",
                                    "matched": 1,
                                    "seen": 1,
                                }
                            },
                        }
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)
    assert "- sudden // ae-stats: matched=1/1" in annotated


def test_annotate_template_bullet_alternatives_scatter_stats_inline() -> None:
    rules = """
## Replace (once): hearing loss

Prefix:

- left | right
""".strip()
    compiled = FullPlanCompiler().compile(MarkdownRulesParser().parse(rules))
    assert len(compiled.rules) == 2
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        compiled.rules[0].rule_id: {
                            "rule_name": "replace:hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prefix:any:left": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "left",
                                    "matched": 1,
                                    "seen": 1,
                                }
                            },
                        },
                        compiled.rules[1].rule_id: {
                            "rule_name": "replace:hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prefix:any:right": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "right",
                                    "matched": 0,
                                    "seen": 1,
                                }
                            },
                        },
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)
    assert "- left | right // ae-stats: left 1/1, right 0/1" in annotated


def test_annotate_template_scalar_postfix_scatter_stats_inline() -> None:
    rules = """
## Replace (once): hearing loss

Postfix: conductive | sensorineural
""".strip()
    compiled = FullPlanCompiler().compile(MarkdownRulesParser().parse(rules))
    assert len(compiled.rules) == 2
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        compiled.rules[0].rule_id: {
                            "rule_name": "replace:hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "postfix:any:conductive": {
                                    "node_path": "postfix",
                                    "node_type": "any",
                                    "debug_expression": "conductive",
                                    "matched": 1,
                                    "seen": 1,
                                }
                            },
                        },
                        compiled.rules[1].rule_id: {
                            "rule_name": "replace:hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "postfix:any:sensorineural": {
                                    "node_path": "postfix",
                                    "node_type": "any",
                                    "debug_expression": "sensorineural",
                                    "matched": 1,
                                    "seen": 1,
                                }
                            },
                        },
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)
    assert (
        "Postfix: conductive | sensorineural // ae-stats: "
        "conductive 1/1, sensorineural 1/1" in annotated
    )


def test_annotate_template_scalar_prompt_four_alternatives_inline() -> None:
    rules = """
## Replace (once): hearing loss

Prompt: a | b | c | d
""".strip()
    compiled = FullPlanCompiler().compile(MarkdownRulesParser().parse(rules))
    assert len(compiled.rules) == 4
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        compiled.rules[0].rule_id: {
                            "rule_name": "replace:hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prompt:any:a": {
                                    "node_path": "prompt",
                                    "node_type": "any",
                                    "debug_expression": "a",
                                    "matched": 1,
                                    "seen": 1,
                                }
                            },
                        },
                        compiled.rules[1].rule_id: {
                            "rule_name": "replace:hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prompt:any:b": {
                                    "node_path": "prompt",
                                    "node_type": "any",
                                    "debug_expression": "b",
                                    "matched": 0,
                                    "seen": 1,
                                }
                            },
                        },
                        compiled.rules[2].rule_id: {
                            "rule_name": "replace:hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prompt:any:c": {
                                    "node_path": "prompt",
                                    "node_type": "any",
                                    "debug_expression": "c",
                                    "matched": 1,
                                    "seen": 1,
                                }
                            },
                        },
                        compiled.rules[3].rule_id: {
                            "rule_name": "replace:hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prompt:any:d": {
                                    "node_path": "prompt",
                                    "node_type": "any",
                                    "debug_expression": "d",
                                    "matched": 1,
                                    "seen": 1,
                                }
                            },
                        },
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)
    assert (
        "Prompt: a | b | c | d // ae-stats: a 1/1, b 0/1, c 1/1, d 1/1"
        in annotated
    )


def test_annotate_template_expansion_two_rules_matches_golden() -> None:
    rules = """
## Replace (once): hearing loss

Postfix: conductive | sensorineural

Fallback: keep original phrase
""".strip()
    compiled = FullPlanCompiler().compile(MarkdownRulesParser().parse(rules))
    assert len(compiled.rules) == 2
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        compiled.rules[0].rule_id: {
                            "rule_name": "replace:hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "postfix:any:conductive": {
                                    "node_path": "postfix",
                                    "node_type": "any",
                                    "debug_expression": "conductive",
                                    "matched": 1,
                                    "seen": 1,
                                }
                            },
                            "candidate_choices": {
                                "fallback:keep original phrase": {
                                    "kind": "fallback",
                                    "candidate_id": "keep original phrase",
                                    "label": "keep original phrase",
                                    "chosen": 1,
                                }
                            },
                        },
                        compiled.rules[1].rule_id: {
                            "rule_name": "replace:hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "postfix:any:sensorineural": {
                                    "node_path": "postfix",
                                    "node_type": "any",
                                    "debug_expression": "sensorineural",
                                    "matched": 1,
                                    "seen": 1,
                                }
                            },
                            "candidate_choices": {
                                "fallback:keep original phrase": {
                                    "kind": "fallback",
                                    "candidate_id": "keep original phrase",
                                    "label": "keep original phrase",
                                    "chosen": 1,
                                }
                            },
                        },
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)
    expected = Path(
        "tests/golden/rules_stats_reporting_template_expand_2_golden.md"
    ).read_text()
    assert _normalize_dynamic_rule_ids(annotated) == expected


def test_annotate_template_expansion_four_rules_matches_golden() -> None:
    rules = """
## Avoid (last clause): contralateral conductive inference Weber

Scope: all

Prefix:

- Weber | forehead
- left || right

Postfix:

- right || left
- conductive
""".strip()
    compiled = FullPlanCompiler().compile(MarkdownRulesParser().parse(rules))
    assert len(compiled.rules) == 4
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        compiled.rules[0].rule_id: {
                            "rule_name": (
                                "avoid:contralateral conductive inference Weber"
                            ),
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prefix:any:Weber": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "Weber",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "prefix:any:left": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "left",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "postfix:any:right": {
                                    "node_path": "postfix",
                                    "node_type": "any",
                                    "debug_expression": "right",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "postfix:any:conductive": {
                                    "node_path": "postfix",
                                    "node_type": "any",
                                    "debug_expression": "conductive",
                                    "matched": 1,
                                    "seen": 1,
                                },
                            },
                        },
                        compiled.rules[1].rule_id: {
                            "rule_name": (
                                "avoid:contralateral conductive inference Weber"
                            ),
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prefix:any:Weber": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "Weber",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "prefix:any:right": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "right",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "postfix:any:left": {
                                    "node_path": "postfix",
                                    "node_type": "any",
                                    "debug_expression": "left",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "postfix:any:conductive": {
                                    "node_path": "postfix",
                                    "node_type": "any",
                                    "debug_expression": "conductive",
                                    "matched": 1,
                                    "seen": 1,
                                },
                            },
                        },
                        compiled.rules[2].rule_id: {
                            "rule_name": (
                                "avoid:contralateral conductive inference Weber"
                            ),
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prefix:any:forehead": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "forehead",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "prefix:any:left": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "left",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "postfix:any:right": {
                                    "node_path": "postfix",
                                    "node_type": "any",
                                    "debug_expression": "right",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "postfix:any:conductive": {
                                    "node_path": "postfix",
                                    "node_type": "any",
                                    "debug_expression": "conductive",
                                    "matched": 1,
                                    "seen": 1,
                                },
                            },
                        },
                        compiled.rules[3].rule_id: {
                            "rule_name": (
                                "avoid:contralateral conductive inference Weber"
                            ),
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prefix:any:forehead": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "forehead",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "prefix:any:right": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "right",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "postfix:any:left": {
                                    "node_path": "postfix",
                                    "node_type": "any",
                                    "debug_expression": "left",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "postfix:any:conductive": {
                                    "node_path": "postfix",
                                    "node_type": "any",
                                    "debug_expression": "conductive",
                                    "matched": 1,
                                    "seen": 1,
                                },
                            },
                        },
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)
    expected = Path(
        "tests/golden/rules_stats_reporting_template_expand_4_golden.md"
    ).read_text()
    assert _normalize_dynamic_rule_ids(annotated) == expected


def test_annotate_rules_with_run_stats_matches_golden_fixture() -> None:
    rules = """
## Avoid (postfix, repeat): conductive

Prefix (all):

- Rinne
- positive

Postfix (all):

- conductive

With:

- probe_1
- probe_2
- probe_3
""".strip()
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        rid: {
                            "rule_name": "avoid:Rinne positive then conductive",
                            "evaluations": 1,
                            "applied": 3,
                            "conditions": {
                                "prefix:all:Rinne": {
                                    "node_path": "prefix",
                                    "node_type": "all",
                                    "debug_expression": "Rinne",
                                    "matched": 89,
                                    "seen": 313,
                                },
                                "prefix:all:positive": {
                                    "node_path": "prefix",
                                    "node_type": "all",
                                    "debug_expression": "positive",
                                    "matched": 86,
                                    "seen": 313,
                                },
                                "postfix:all:conductive": {
                                    "node_path": "postfix",
                                    "node_type": "all",
                                    "debug_expression": "conductive",
                                    "matched": 13,
                                    "seen": 313,
                                },
                            },
                            "candidate_choices": {
                                "generated:probe_1": {
                                    "kind": "generated",
                                    "label": "probe_1",
                                    "chosen": 1,
                                },
                                "generated:probe_2": {
                                    "kind": "generated",
                                    "label": "probe_2",
                                    "chosen": 1,
                                },
                                "generated:probe_3": {
                                    "kind": "generated",
                                    "label": "probe_3",
                                    "chosen": 1,
                                },
                            },
                        }
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)
    expected = Path(
        "tests/golden/rules_stats_reporting_human_avoid_golden.md"
    ).read_text()
    assert _normalize_dynamic_rule_ids(annotated) == expected


def test_annotate_replace_rule_includes_prefix_and_candidate_gen_stats() -> (
    None
):
    rules = """
## Replace (once): sensorineural hearing loss

Prefix (any):

- sudden

With:

- sudden sensorineural hearing loss
- SSNHL
""".strip()
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        rid: {
                            "rule_name": "replace:sensorineural hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prefix:any:sudden": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "sudden",
                                    "matched": 1,
                                    "seen": 1,
                                }
                            },
                            "candidate_choices": {
                                "generated:sudden sensorineural hearing loss": {
                                    "kind": "generated",
                                    "label": (
                                        "sudden sensorineural hearing loss"
                                    ),
                                    "chosen": 1,
                                }
                            },
                        }
                    }
                },
                {
                    "rules": {
                        rid: {
                            "rule_name": "replace:sensorineural hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prefix:any:sudden": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "sudden",
                                    "matched": 1,
                                    "seen": 1,
                                }
                            },
                            "candidate_choices": {
                                "generated:SSNHL": {
                                    "kind": "generated",
                                    "label": "SSNHL",
                                    "chosen": 1,
                                }
                            },
                        }
                    }
                },
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)

    assert "top trigger terms: sudden 2/2" in annotated
    assert "- sudden // ae-stats: matched=2/2" in annotated
    assert "- sudden sensorineural hearing loss" in annotated
    assert "chosen=1/2 (50.0%)" in annotated
    assert "- SSNHL" in annotated
    assert annotated.count("chosen=1/2 (50.0%)") >= 2


def test_fake_generation_scoring_splits_stats_by_chosen_generation() -> None:
    rules = """
## Replace (once): sensorineural hearing loss

With:

- sudden sensorineural hearing loss
- SSNHL
""".strip()
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )

    def _fake_generation(scores: tuple[float, float]) -> dict[str, object]:
        best = 0 if scores[0] >= scores[1] else 1
        labels = ("sudden sensorineural hearing loss", "SSNHL")
        chosen_label = labels[best]
        return {
            "rules": {
                rid: {
                    "rule_name": "replace:sensorineural hearing loss",
                    "evaluations": 1,
                    "applied": 1,
                    "candidate_choices": {
                        f"generated:{chosen_label}": {
                            "kind": "generated",
                            "label": chosen_label,
                            "chosen": 1,
                        }
                    },
                }
            }
        }

    merged = AggregatedRunStats(
        _telemetry_items(
            [
                _fake_generation(
                    (0.90, 0.10)
                ),  # generation #1: candidate 1 wins
                _fake_generation(
                    (0.10, 0.95)
                ),  # generation #2: candidate 2 wins
                _fake_generation(
                    (0.20, 0.80)
                ),  # generation #3: candidate 2 wins
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)

    assert "- sudden sensorineural hearing loss" in annotated
    assert "- SSNHL" in annotated
    assert "chosen=1/3 (33.3%)" in annotated
    assert "chosen=2/3 (66.7%)" in annotated
    assert "chosen=3/3" not in annotated


def test_annotate_avoid_rule_renders_fallback_section_in_body() -> None:
    rules = """
## Avoid (repeat): conductive

Prefix (all):

- rinne

Fallback:

- these findings require further evaluation.
""".strip()
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        rid: {
                            "rule_name": "avoid:conductive",
                            "evaluations": 1,
                            "applied": 0,
                            "candidate_choices": {
                                "fallback:fallback_1": {
                                    "kind": "fallback",
                                    "candidate_id": "fallback_1",
                                    "label": "fallback_1",
                                    "chosen": 0,
                                }
                            },
                        }
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)

    assert "\nFallback:\n" in annotated
    assert "\nWith:\n" not in annotated
    assert "chosen candidates:" not in annotated


def test_annotate_replace_static_candidate_stats_matches_e2e_golden() -> None:
    rules = """
## Replace (once): sensorineural hearing loss

With:

- sudden sensorineural hearing loss
- SSNHL
""".strip()
    telemetry_item = _run_real_static_rule_flow_once(
        rules_markdown=rules,
        source_text="The patient has sensorineural hearing loss.",
        token_index=20,
    )
    merged = AggregatedRunStats((telemetry_item,))

    annotated = annotate_rules_with_run_stats(rules, merged)
    expected = Path(
        "tests/golden/rules_stats_reporting_replace_static_e2e_golden.md"
    ).read_text()
    assert _normalize_dynamic_rule_ids(annotated) == expected


def test_annotate_after_static_candidate_stats_matches_e2e_golden() -> None:
    rules = """
## After (once): SSNHL

Add:

- This condition requires urgent treatment.
- Prompt ENT follow-up.
""".strip()
    telemetry_item = _run_real_static_rule_flow_once(
        rules_markdown=rules,
        source_text="Findings are consistent with SSNHL",
        token_index=6,
    )
    merged = AggregatedRunStats((telemetry_item,))

    annotated = annotate_rules_with_run_stats(rules, merged)
    expected = Path(
        "tests/golden/rules_stats_reporting_after_static_e2e_golden.md"
    ).read_text()
    assert _normalize_dynamic_rule_ids(annotated) == expected


def test_annotate_replace_fake_generation_matches_golden_fixture() -> None:
    rules = """
## Replace (once): sensorineural hearing loss

Prefix (any):

- sudden

With:

- sudden sensorineural hearing loss
- SSNHL
""".strip()
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )

    def _fake_generation(scores: tuple[float, float]) -> dict[str, object]:
        best = 0 if scores[0] >= scores[1] else 1
        labels = ("sudden sensorineural hearing loss", "SSNHL")
        chosen_label = labels[best]
        return {
            "rules": {
                rid: {
                    "rule_name": "replace:sensorineural hearing loss",
                    "evaluations": 1,
                    "applied": 1,
                    "conditions": {
                        "required_before_any_1": {
                            "condition_id": "required_before_any_1",
                            "node_path": "prefix",
                            "node_type": "any",
                            "debug_expression": "sudden",
                            "matched": 1,
                            "seen": 1,
                        }
                    },
                    "candidate_choices": {
                        f"generated:{chosen_label}": {
                            "kind": "generated",
                            "candidate_id": chosen_label,
                            "label": chosen_label,
                            "chosen": 1,
                        }
                    },
                }
            }
        }

    merged = AggregatedRunStats(
        _telemetry_items(
            [
                _fake_generation((0.90, 0.10)),
                _fake_generation((0.10, 0.95)),
                _fake_generation((0.20, 0.80)),
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)
    expected = Path(
        "tests/golden/rules_stats_reporting_human_replace_golden.md"
    ).read_text()
    assert _normalize_dynamic_rule_ids(annotated) == expected


def test_annotated_rules_stats_are_idempotent_input_rules() -> None:
    rules = """
## Replace (once): sensorineural hearing loss

Prefix (any):

- sudden

With:

- sudden sensorineural hearing loss
- SSNHL
""".strip()
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        rid: {
                            "rule_name": "replace:sensorineural hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prefix:any:sudden": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "sudden",
                                    "matched": 1,
                                    "seen": 1,
                                }
                            },
                            "candidate_choices": {
                                "generated:SSNHL": {
                                    "kind": "generated",
                                    "candidate_id": "SSNHL",
                                    "label": "SSNHL",
                                    "chosen": 1,
                                }
                            },
                        }
                    }
                }
            ]
        )
    )

    first = annotate_rules_with_run_stats(rules, merged)
    parsed = MarkdownRulesParser().parse(_strip_rule_activity_summary(first))
    assert parsed.rules
    second = annotate_rules_with_run_stats(
        _strip_rule_activity_summary(first), merged
    )
    assert second == _strip_rule_activity_summary(first)


def test_annotated_static_replace_candidate_stats_are_idempotent() -> None:
    rules = """
## Replace (once): sensorineural hearing loss

With:

- sudden sensorineural hearing loss
- SSNHL
""".strip()
    telemetry_item = _run_real_static_rule_flow_once(
        rules_markdown=rules,
        source_text="sudden sensorineural hearing loss",
        token_index=1,
    )
    merged = AggregatedRunStats((telemetry_item,))

    first = annotate_rules_with_run_stats(rules, merged)
    stripped = _strip_rule_activity_summary(first)
    second = annotate_rules_with_run_stats(stripped, merged)
    assert second == stripped


def test_runtime_replace_prefix_terms_render_inline_trigger_stats() -> None:
    rules = """
## Replace (once): sensorineural hearing loss

Prefix (any):

- sudden
- abrupt

With:

- SSNHL
""".strip()
    telemetry_item = _run_real_static_rule_flow_once(
        rules_markdown=rules,
        source_text="The patient has sudden sensorineural hearing loss.",
        token_index=20,
    )
    merged = AggregatedRunStats((telemetry_item,))

    annotated = annotate_rules_with_run_stats(rules, merged)

    assert "fired=1/1 (100.0%)" in annotated
    assert "top trigger terms: sudden 1/1" in annotated
    assert "- sudden // ae-stats: matched=1/1" in annotated
    assert "- abrupt // ae-stats:" not in annotated
    assert "chosen=1/1 (100.0%)" in annotated


def test_runtime_after_prefix_terms_render_inline_trigger_stats() -> None:
    rules = """
## After (once): SSNHL

Prefix (any):

- urgent

Add:

- This condition requires urgent treatment.
- Prompt ENT follow-up.
""".strip()
    telemetry_item = _run_real_static_rule_flow_once(
        rules_markdown=rules,
        source_text="urgent SSNHL",
        token_index=8,
    )
    merged = AggregatedRunStats((telemetry_item,))

    annotated = annotate_rules_with_run_stats(rules, merged)

    assert "fired=1/1 (100.0%)" in annotated
    assert "top trigger terms: urgent 1/1" in annotated
    assert "- urgent // ae-stats: matched=1/1" in annotated
    assert "chosen=1/1 (100.0%)" in annotated


def test_runtime_avoid_trigger_annotation_has_no_regression() -> None:
    rules = """
## Avoid (repeat): conductive hearing loss

Prefix (all):

- rinne
- positive

Postfix (all):

- conductive

Fallback:

- findings require specialist follow-up.
""".strip()
    telemetry_item = _run_real_static_rule_flow_once(
        rules_markdown=rules,
        source_text="Rinne positive pattern suggests conductive findings.",
        token_index=20,
    )
    merged = AggregatedRunStats((telemetry_item,))

    annotated = annotate_rules_with_run_stats(rules, merged)

    assert "- rinne // ae-stats: matched=1/1" in annotated
    assert "- positive // ae-stats: matched=1/1" in annotated
    assert "- conductive // ae-stats: matched=1/1" in annotated


def test_runtime_avoid_prompt_terms_render_inline_trigger_stats() -> None:
    rules = """
## Avoid (repeat): conductive hearing loss

Prompt (any):

- left
- right

Postfix (all):

- conductive

Fallback:

- findings require specialist follow-up.
""".strip()
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        rid: {
                            "rule_name": "avoid:conductive hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prompt_left": {
                                    "node_path": "prompt",
                                    "node_type": "any",
                                    "debug_expression": "left",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "prompt_right": {
                                    "node_path": "prompt",
                                    "node_type": "any",
                                    "debug_expression": "right",
                                    "matched": 0,
                                    "seen": 1,
                                },
                                "postfix_conductive": {
                                    "node_path": "postfix",
                                    "node_type": "all",
                                    "debug_expression": "conductive",
                                    "matched": 1,
                                    "seen": 1,
                                },
                            },
                        }
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)
    assert "- left // ae-stats: matched=1/1" in annotated
    assert "- right // ae-stats: matched=" not in annotated
    assert "top trigger terms: conductive 1/1, left 1/1" in annotated


def test_runtime_avoid_reports_prompt_prefix_connector_and_postfix() -> None:
    rules = """
## Avoid (repeat): conductive hearing loss

Prompt (all):

- left

Prefix (none):

- bilateral

Connector:

- suggests

Postfix (all):

- conductive

Fallback:

- findings require specialist follow-up.
""".strip()
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        rid: {
                            "rule_name": "avoid:conductive hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "prompt_left": {
                                    "node_path": "prompt",
                                    "node_type": "all",
                                    "debug_expression": "left",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "prefix_bilateral": {
                                    "node_path": "prefix",
                                    "node_type": "none",
                                    "debug_expression": "bilateral",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "connector_suggests": {
                                    "node_path": "connector",
                                    "node_type": "any",
                                    "debug_expression": "suggests",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "postfix_conductive": {
                                    "node_path": "postfix",
                                    "node_type": "all",
                                    "debug_expression": "conductive",
                                    "matched": 1,
                                    "seen": 1,
                                },
                            },
                            "candidate_choices": {
                                "fallback:fallback_1": {
                                    "kind": "fallback",
                                    "candidate_id": "fallback_1",
                                    "label": (
                                        "findings require specialist follow-up."
                                    ),
                                    "chosen": 1,
                                }
                            },
                        }
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)
    assert "- left // ae-stats: matched=1/1" in annotated
    assert "- bilateral // ae-stats: matched=1/1" in annotated
    assert "- suggests // ae-stats: matched=1/1" in annotated
    assert "- conductive // ae-stats: matched=1/1" in annotated
    assert (
        "top trigger terms: bilateral 1/1, conductive 1/1, left 1/1"
        in annotated
    )


def test_runtime_avoid_connector_zero_match_stays_clean() -> None:
    rules = """
## Avoid (repeat): conductive hearing loss

Connector:

- suggests

Fallback:

- findings require specialist follow-up.
""".strip()
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        rid: {
                            "rule_name": "avoid:conductive hearing loss",
                            "evaluations": 1,
                            "applied": 0,
                            "conditions": {
                                "connector_suggests": {
                                    "node_path": "connector",
                                    "node_type": "any",
                                    "debug_expression": "suggests",
                                    "matched": 0,
                                    "seen": 1,
                                }
                            },
                        }
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)
    assert "- suggests // ae-stats: matched=" not in annotated
    assert "top trigger terms:" not in annotated


def test_trigger_summary_excludes_structural_nodes() -> None:
    rules = """
## Replace (once): sensorineural hearing loss

Prefix:

- sudden

With:

- SSNHL
""".strip()
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        rid: {
                            "rule_name": "replace:sensorineural hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "structural": {
                                    "condition_id": "structural",
                                    "node_path": "prefix",
                                    "node_type": "MatchAny",
                                    "debug_expression": "MatchAny",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "leaf": {
                                    "condition_id": "leaf",
                                    "node_path": "prefix",
                                    "node_type": "MatchTerm",
                                    "debug_expression": "sudden",
                                    "matched": 1,
                                    "seen": 1,
                                },
                            },
                            "candidate_choices": {
                                "generated:SSNHL": {
                                    "kind": "generated",
                                    "label": "SSNHL",
                                    "chosen": 1,
                                }
                            },
                        }
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)
    assert "top trigger terms: sudden 1/1" in annotated
    assert "MatchAny 1/1" not in annotated


def test_runtime_replace_prefix_zero_match_stays_clean() -> None:
    rules = """
## Replace (once): sensorineural hearing loss

Prefix (any):

- sudden

With:

- SSNHL
""".strip()
    rid = (
        FullPlanCompiler()
        .compile(MarkdownRulesParser().parse(rules))
        .rules[0]
        .rule_id
    )
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        rid: {
                            "rule_name": "replace:sensorineural hearing loss",
                            "evaluations": 1,
                            "applied": 0,
                            "conditions": {
                                "prefix_sudden": {
                                    "node_path": "prefix",
                                    "node_type": "any",
                                    "debug_expression": "sudden",
                                    "matched": 0,
                                    "seen": 1,
                                }
                            },
                        }
                    }
                }
            ]
        )
    )

    annotated = annotate_rules_with_run_stats(rules, merged)

    assert "fired=0/1 (0.0%)" in annotated
    assert "- sudden // ae-stats: matched=" not in annotated
    assert "top trigger terms:" not in annotated


def test_mixed_replace_after_avoid_annotation_and_idempotence() -> None:
    rules = """
## Replace (once): sensorineural hearing loss

Prefix:

- sudden

With:

- SSNHL

## After (once): SSNHL

Prefix:

- urgent

Add:

- Prompt ENT follow-up.

## Avoid (repeat): conductive hearing loss

Prefix:

- rinne

Postfix:

- conductive

Fallback:

- findings require specialist follow-up.
""".strip()
    compiled_rules = FullPlanCompiler().compile(
        MarkdownRulesParser().parse(rules)
    )
    replace_rid, after_rid, avoid_rid = (
        compiled_rules.rules[0].rule_id,
        compiled_rules.rules[1].rule_id,
        compiled_rules.rules[2].rule_id,
    )
    merged = AggregatedRunStats(
        _telemetry_items(
            [
                {
                    "rules": {
                        replace_rid: {
                            "rule_name": "replace:sensorineural hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "replace_prefix": {
                                    "node_path": "prefix",
                                    "node_type": "all",
                                    "debug_expression": "sudden",
                                    "matched": 1,
                                    "seen": 1,
                                }
                            },
                            "candidate_choices": {
                                "generated:SSNHL": {
                                    "kind": "generated",
                                    "label": "SSNHL",
                                    "chosen": 1,
                                }
                            },
                        },
                        after_rid: {
                            "rule_name": "after:SSNHL",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "after_prefix": {
                                    "node_path": "prefix",
                                    "node_type": "all",
                                    "debug_expression": "urgent",
                                    "matched": 1,
                                    "seen": 1,
                                }
                            },
                            "candidate_choices": {
                                "generated:Prompt ENT follow-up.": {
                                    "kind": "generated",
                                    "label": "Prompt ENT follow-up.",
                                    "chosen": 1,
                                }
                            },
                        },
                        avoid_rid: {
                            "rule_name": "avoid:conductive hearing loss",
                            "evaluations": 1,
                            "applied": 1,
                            "conditions": {
                                "avoid_prefix": {
                                    "node_path": "prefix",
                                    "node_type": "all",
                                    "debug_expression": "rinne",
                                    "matched": 1,
                                    "seen": 1,
                                },
                                "avoid_postfix": {
                                    "node_path": "postfix",
                                    "node_type": "all",
                                    "debug_expression": "conductive",
                                    "matched": 1,
                                    "seen": 1,
                                },
                            },
                            "candidate_choices": {
                                "fallback:fallback_1": {
                                    "kind": "fallback",
                                    "candidate_id": "fallback_1",
                                    "label": (
                                        "findings require specialist follow-up."
                                    ),
                                    "chosen": 1,
                                }
                            },
                        },
                    }
                }
            ]
        )
    )

    first = annotate_rules_with_run_stats(rules, merged)
    assert "- sudden // ae-stats: matched=1/1" in first
    assert "- urgent // ae-stats: matched=1/1" in first
    assert "- rinne // ae-stats: matched=1/1" in first
    assert "- conductive // ae-stats: matched=1/1" in first
    assert "top trigger terms: sudden 1/1" in first
    assert "top trigger terms: urgent 1/1" in first

    stripped = _strip_rule_activity_summary(first)
    second = annotate_rules_with_run_stats(stripped, merged)
    assert second == stripped


def _normalize_dynamic_rule_ids(text: str) -> str:
    return re.sub(
        pattern=r"^// ae-rule-id: .+$",
        repl="// ae-rule-id: <dynamic>",
        string=text,
        flags=re.MULTILINE,
    )


def _run_real_static_rule_flow_once(
    *,
    rules_markdown: str,
    source_text: str,
    token_index: int,
) -> TelemetryItem:
    engine = ExecutionSession(plan=CompiledRules(rules_markdown).plan)
    decision = step_test(
        engine,
        snapshot_text=source_text,
        token_index=token_index,
    )
    aggregator = RuntimeTelemetryAggregator(rule_name_for=engine.rule_name)
    aggregator.observe_events(decision.events)
    telemetry = aggregator.build_snapshot(decision_limit_reached=False)
    return TelemetryItem(telemetry)
