from __future__ import annotations

from answer_engineering.engine.orchestration.orchestrator import (
    PlanRunner,
)
from answer_engineering.engine.runtime.runtime_types import (
    PatchOp,
)
from answer_engineering.rules.compile.plan import (
    AnchorQuerySpec,
    CandidateSpec,
    EditTargetSpec,
    PlanIR,
    RulePlan,
    ScopeSpec,
)
from tests._support.core_helpers import (
    create_step_snapshot,
)


def test_missing_anchor_generates_noop_with_reason() -> None:
    plan = PlanIR(
        rules=[
            RulePlan(
                rule_id="r1",
                scope=ScopeSpec(kind="tail_chars", n=80, casefold=True),
                anchors=(
                    AnchorQuerySpec(
                        anchor_id="match",
                        match_phrase_any=("needle",),
                        match_mode="last",
                    ),
                ),
                target=EditTargetSpec(kind="match_span", anchor_id="match"),
                candidates=(CandidateSpec(op=PatchOp.REPLACE, text="new"),),
            )
        ]
    )

    out = PlanRunner().run(
        plan,
        create_step_snapshot(
            snapshot_text="text without target", token_index=0
        ),
    )
    assert out.proposals == []


def test_empty_candidates_generates_noop_with_reason() -> None:
    plan = PlanIR(
        rules=[
            RulePlan(
                rule_id="r1",
                scope=ScopeSpec(kind="tail_chars", n=80, casefold=True),
                target=EditTargetSpec(kind="scope_entire"),
                candidates=(),
            )
        ]
    )

    out = PlanRunner().run(
        plan,
        create_step_snapshot(
            snapshot_text="contains needle at end needle",
            token_index=0,
        ),
    )
    assert any(p.reason == "no candidates" for p in out.proposals)
