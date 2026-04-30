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
    DecisionPolicySpec,
    EditTargetSpec,
    FirePolicySpec,
    PlanIR,
    RulePlan,
    ScopeSpec,
)
from tests._support.core_helpers import (
    create_step_snapshot,
)


def test_repeat_rules_do_not_iterate_to_stable_within_single_run() -> None:
    scope = ScopeSpec(kind="tail_chars", max_chars=200, casefold=True)
    plan = PlanIR(
        rules=(
            RulePlan(
                rule_id="to_show",
                guard_scope=scope,
                edit_scope=scope,
                anchors=(
                    AnchorQuerySpec(
                        anchor_id="suggest_anchor",
                        match_phrase_any=("suggest",),
                    ),
                ),
                target=EditTargetSpec(
                    kind="match_span", anchor_id="suggest_anchor"
                ),
                candidates=(
                    CandidateSpec(op=PatchOp.REPLACE, text="show", priority=10),
                ),
                policy=DecisionPolicySpec(skip_tokens=0),
                fire=FirePolicySpec(mode="repeat"),
            ),
            RulePlan(
                rule_id="to_suggest",
                guard_scope=scope,
                edit_scope=scope,
                anchors=(
                    AnchorQuerySpec(
                        anchor_id="show_anchor", match_phrase_any=("show",)
                    ),
                ),
                target=EditTargetSpec(
                    kind="match_span", anchor_id="show_anchor"
                ),
                candidates=(
                    CandidateSpec(
                        op=PatchOp.REPLACE, text="suggest", priority=10
                    ),
                ),
                policy=DecisionPolicySpec(skip_tokens=0),
                fire=FirePolicySpec(mode="repeat"),
            ),
            RulePlan(
                rule_id="append_done",
                guard_scope=scope,
                edit_scope=scope,
                anchors=(
                    AnchorQuerySpec(
                        anchor_id="token_anchor", match_phrase_any=("token",)
                    ),
                ),
                target=EditTargetSpec(
                    kind="match_span", anchor_id="token_anchor"
                ),
                candidates=(
                    CandidateSpec(
                        op=PatchOp.INSERT_AFTER, text="done", priority=1
                    ),
                ),
                policy=DecisionPolicySpec(skip_tokens=0),
                fire=FirePolicySpec(mode="once"),
            ),
        )
    )

    result = PlanRunner().run(
        plan,
        create_step_snapshot(snapshot_text="suggest token", token_index=3),
    )

    assert result.final_doc.text == "show token"
