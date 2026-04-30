from __future__ import annotations

from pathlib import Path

from answer_engineering.engine.orchestration.orchestrator import (
    PlanRunner,
)
from answer_engineering.rules.compile.compiler import (
    FullPlanCompiler,
)
from answer_engineering.rules.parse.parser import (
    MarkdownRulesParser,
)
from tests._support.core_helpers import (
    create_step_snapshot,
)


def test_avoid_postfix_rewrites_suffix() -> None:
    md = Path("tests/fixtures/rules_full_syntax.md").read_text(encoding="utf-8")
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))
    text = (
        "weber rinne left right positive this suggests conductive hearing loss."
    )
    out = PlanRunner().run(
        plan, create_step_snapshot(snapshot_text=text, token_index=10)
    )

    assert out.applied_patches
    assert "these findings require further evaluation." in out.final_doc.text
    assert "conductive" not in out.final_doc.text


def test_runner_result_events_include_generation_and_apply_trace() -> None:
    md = Path("tests/fixtures/rules_full_syntax.md").read_text(encoding="utf-8")
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))
    out = PlanRunner().run(
        plan,
        create_step_snapshot(
            snapshot_text=(
                "weber rinne left right positive "
                "this suggests conductive hearing loss."
            ),
            token_index=10,
        ),
    )

    event_types = {type(event).__name__ for event in out.events}
    assert "RuleEvaluationStarted" in event_types
    assert "ProposalsGenerated" in event_types
    assert "PatchApplied" in event_types


def test_avoid_everything_rewrites_entire_edit_scope() -> None:
    md = """## Avoid (repeat): conductive

Prefix (all):

* weber
* rinne

Connector:

* this suggests

Postfix (any):

* conductive

Scope:

* 800 chars

Fallback:

* these findings require further evaluation.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))
    text = (
        "weber rinne left right positive. "
        "this suggests conductive hearing loss."
    )
    out = PlanRunner().run(
        plan, create_step_snapshot(snapshot_text=text, token_index=10)
    )

    assert out.applied_patches
    assert "these findings require further evaluation." in out.final_doc.text
    assert "this suggests" not in out.final_doc.text
