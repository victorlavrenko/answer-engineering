from __future__ import annotations

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


def test_planrunner_single_step_trace_smoke() -> None:
    md = """## Avoid (once): conductive

Connector:

* this suggests

Postfix (any):

* conductive

Fallback:

* these findings require further evaluation.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))
    result = PlanRunner(verbose=False).run(
        plan,
        create_step_snapshot(
            snapshot_text="this suggests conductive hearing loss.",
            token_index=10,
        ),
    )

    assert result.applied_patches
    assert "these findings require further evaluation." in result.final_doc.text
    event_types = [type(event).__name__ for event in result.events]
    assert "RuleEvaluationStarted" in event_types
    assert "ProposalsGenerated" in event_types
    assert "ProposalScored" in event_types
    assert "ProposalAccepted" in event_types
    assert "PatchApplied" in event_types
