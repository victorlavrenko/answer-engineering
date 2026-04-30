from __future__ import annotations

from answer_engineering.engine.orchestration.orchestrator import (
    PlanRunner,
)
from answer_engineering.engine.pipeline.context import (
    StepContext,
)
from answer_engineering.engine.proposal.match_tree.match_tree import (
    MatchAll,
    MatchAndThen,
    MatchTerm,
)
from answer_engineering.engine.proposal.proposal_logic import (
    GenerationPrecheck,
)
from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    PatchOp,
    TextView,
)
from answer_engineering.rules.compile.plan import (
    AnchorQuerySpec,
    CandidateSpec,
    DecisionPolicySpec,
    EditTargetSpec,
    FirePolicySpec,
    GuardSpec,
    PlanIR,
    RulePlan,
    ScopeSpec,
)
from tests._support.core_helpers import (
    apply_proposal_to_text,
    create_step_snapshot,
)
from tests.core.match_tree_guard_factory import build_guard_expression


def _avoid_plan(
    *, guard: GuardSpec, target: EditTargetSpec, edit_scope: ScopeSpec
) -> PlanIR:
    return PlanIR(
        rules=(
            RulePlan(
                rule_id="r-avoid",
                name="avoid:postfix overlap",
                guard_scope=ScopeSpec(kind="whole_doc", casefold=True),
                edit_scope=edit_scope,
                guard=guard,
                anchors=(
                    AnchorQuerySpec(
                        anchor_id="topic",
                        match_phrase_any=("TOPIC",),
                        match_mode="first",
                    ),
                ),
                target=target,
                candidates=(
                    CandidateSpec(
                        op=PatchOp.REPLACE,
                        text="SAFE",
                        label="fallback_1",
                        priority=10,
                    ),
                ),
                policy=DecisionPolicySpec(skip_tokens=0),
                fire=FirePolicySpec(mode="repeat"),
            ),
        )
    )


def _guard(**legacy_fields: object) -> GuardSpec:
    if "require_order" not in legacy_fields and any(
        key.startswith("postfix_") for key in legacy_fields
    ):
        legacy_fields["require_order"] = "prefix_before_postfix"
    return GuardSpec(expression=build_guard_expression(**legacy_fields))


def test_required_after_any_outside_edit_scope_does_not_trigger() -> None:
    plan = _avoid_plan(
        guard=_guard(
            required_before_all=("left",), required_after_any=("sensorineural",)
        ),
        target=EditTargetSpec(kind="scope_entire"),
        edit_scope=ScopeSpec(kind="tail_sentences", n=1, casefold=True),
    )
    text = "Left loss appears sensorineural. Final sentence has no diagnosis."

    out = PlanRunner().run(
        plan, create_step_snapshot(snapshot_text=text, token_index=10)
    )

    assert out.applied_patches == []
    assert out.proposals == []


def test_required_after_any_with_one_overlapping_match_triggers() -> None:
    plan = _avoid_plan(
        guard=_guard(
            required_before_all=("left",),
            required_after_any=("sensorineural", "conductive"),
        ),
        target=EditTargetSpec(kind="scope_entire"),
        edit_scope=ScopeSpec(kind="tail_sentences", n=1, casefold=True),
    )
    text = (
        "Left loss appears sensorineural. Final sentence says conductive loss."
    )

    out = PlanRunner().run(
        plan, create_step_snapshot(snapshot_text=text, token_index=10)
    )

    assert out.applied_patches
    assert (
        apply_proposal_to_text(text, out.applied_patches[0].proposal)
        == "Left loss appears sensorineural. SAFE"
    )


def test_required_after_all_requires_full_overlap_of_required_terms() -> None:
    partial = _avoid_plan(
        guard=_guard(
            required_before_all=("left",),
            required_after_all=("sensorineural", "urgent"),
        ),
        target=EditTargetSpec(kind="scope_entire"),
        edit_scope=ScopeSpec(kind="tail_sentences", n=1, casefold=True),
    )
    partial_text = "Left loss appears sensorineural. Final sentence is urgent."

    partial_out = PlanRunner().run(
        partial,
        create_step_snapshot(snapshot_text=partial_text, token_index=10),
    )
    assert partial_out.applied_patches == []

    full = _avoid_plan(
        guard=_guard(
            required_before_all=("left",),
            required_after_all=("sensorineural", "urgent"),
        ),
        target=EditTargetSpec(kind="scope_entire"),
        edit_scope=ScopeSpec(kind="tail_sentences", n=1, casefold=True),
    )
    full_text = (
        "Left loss appears baseline. "
        "Final sentence is sensorineural and urgent."
    )

    full_out = PlanRunner().run(
        full,
        create_step_snapshot(snapshot_text=full_text, token_index=10),
    )
    assert full_out.applied_patches


def test_overlap_boundary_and_point_insertion_semantics() -> None:
    text = "bad X"
    doc = DocumentState(text=text, version_id="v1")
    guard_view = TextView(
        doc=doc,
        abs_start=0,
        abs_end=len(text),
    )
    rule = RulePlan(
        rule_id="r",
        name="avoid:boundary",
        guard_scope=ScopeSpec(kind="whole_doc", casefold=True),
        edit_scope=ScopeSpec(kind="whole_doc", casefold=True),
        guard=_guard(required_after_any=("bad",)),
        target=EditTargetSpec(kind="scope_entire"),
        candidates=(
            CandidateSpec(op=PatchOp.REPLACE, text="SAFE", priority=1),
        ),
        fire=FirePolicySpec(mode="repeat"),
    )

    touch_start = StepContext(
        plan=PlanIR(rules=(rule,)),
        rule=rule,
        doc=doc,
        step=create_step_snapshot(snapshot_text=doc.text, token_index=0),
        guard_view=guard_view,
        edit_view=TextView(
            doc=doc,
            abs_start=3,
            abs_end=len(text),
        ),
    )
    assert GenerationPrecheck(touch_start).span is None

    point = StepContext(
        plan=PlanIR(rules=(rule,)),
        rule=rule,
        doc=doc,
        step=create_step_snapshot(snapshot_text=doc.text, token_index=0),
        guard_view=guard_view,
        edit_view=TextView(
            doc=doc,
            abs_start=1,
            abs_end=1,
        ),
    )
    assert GenerationPrecheck(point).span is None


def test_target_kinds_apply_same_overlap_policy() -> None:
    base_text = "Intro left. TOPIC sensorineural urgent."

    scope_plan = _avoid_plan(
        guard=_guard(
            required_before_all=("left",), required_after_any=("sensorineural",)
        ),
        target=EditTargetSpec(kind="scope_entire"),
        edit_scope=ScopeSpec(kind="tail_sentences", n=1, casefold=True),
    )
    assert (
        PlanRunner()
        .run(
            scope_plan,
            create_step_snapshot(snapshot_text=base_text, token_index=10),
        )
        .applied_patches
    )

    match_plan = _avoid_plan(
        guard=_guard(
            required_before_all=("left",), required_after_any=("sensorineural",)
        ),
        target=EditTargetSpec(kind="match_span", anchor_id="topic"),
        edit_scope=ScopeSpec(kind="whole_doc", casefold=True),
    )
    assert (
        PlanRunner()
        .run(
            match_plan,
            create_step_snapshot(snapshot_text=base_text, token_index=10),
        )
        .applied_patches
        == []
    )

    sent_plan = _avoid_plan(
        guard=_guard(
            required_before_all=("left",), required_after_any=("sensorineural",)
        ),
        target=EditTargetSpec(
            kind="after_anchor_to_sentence_end", anchor_id="topic"
        ),
        edit_scope=ScopeSpec(kind="whole_doc", casefold=True),
    )
    assert (
        PlanRunner()
        .run(
            sent_plan,
            create_step_snapshot(snapshot_text=base_text, token_index=10),
        )
        .applied_patches
    )

    clause_plan = _avoid_plan(
        guard=_guard(
            required_before_all=("left",), required_after_any=("sensorineural",)
        ),
        target=EditTargetSpec(
            kind="after_anchor_to_clause_end", anchor_id="topic"
        ),
        edit_scope=ScopeSpec(kind="whole_doc", casefold=True),
    )
    assert (
        PlanRunner()
        .run(
            clause_plan,
            create_step_snapshot(snapshot_text=base_text, token_index=10),
        )
        .applied_patches
    )


def test_overlap_uses_nested_ordered_topology_not_only_top_level_andthen() -> (
    None
):
    nested_guard = GuardSpec(
        expression=MatchAll(
            (
                MatchTerm("left"),
                MatchAndThen(MatchTerm("left"), MatchTerm("sensorineural")),
            )
        )
    )
    plan = _avoid_plan(
        guard=nested_guard,
        target=EditTargetSpec(kind="scope_entire"),
        edit_scope=ScopeSpec(kind="tail_sentences", n=1, casefold=True),
    )
    out = PlanRunner().run(
        plan,
        create_step_snapshot(
            snapshot_text="left baseline. final sensorineural clause.",
            token_index=10,
        ),
    )
    assert out.applied_patches
