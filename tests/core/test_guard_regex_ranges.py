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


def _guard(**legacy_fields: object) -> GuardSpec:
    return GuardSpec(expression=build_guard_expression(**legacy_fields))


def test_prefix_range_gate_matches_inclusive_values() -> None:
    plan = PlanIR(
        rules=(
            RulePlan(
                rule_id="r-range",
                name="r-range",
                guard_scope=ScopeSpec(kind="whole_doc", casefold=True),
                edit_scope=ScopeSpec(kind="whole_doc", casefold=True),
                guard=_guard(required_before_any=("within 1-72 hours",)),
                target=EditTargetSpec(kind="scope_entire"),
                candidates=(
                    CandidateSpec(
                        op=PatchOp.REPLACE, text="MATCHED", priority=10
                    ),
                ),
                policy=DecisionPolicySpec(),
                fire=FirePolicySpec(mode="repeat"),
            ),
        )
    )

    matched = PlanRunner().run(
        plan,
        create_step_snapshot(
            snapshot_text="Symptoms started within 24 hours.", token_index=0
        ),
    )
    assert matched.applied_patches

    unmatched = PlanRunner().run(
        plan,
        create_step_snapshot(
            snapshot_text="Symptoms started within 99 hours.", token_index=0
        ),
    )
    assert not unmatched.applied_patches


def test_connector_range_can_resolve_anchor_for_postfix_target() -> None:
    plan = PlanIR(
        rules=(
            RulePlan(
                rule_id="r-avoid-range-connector",
                name="avoid:range-connector",
                guard_scope=ScopeSpec(kind="whole_doc", casefold=True),
                edit_scope=ScopeSpec(kind="whole_doc", casefold=True),
                guard=_guard(
                    connector_terms=("noticed 1-72 hours",),
                    required_after_any=("conductive",),
                ),
                anchors=(
                    AnchorQuerySpec(
                        anchor_id="connector",
                        match_phrase_any=("noticed 1-72 hours",),
                        match_mode="last",
                    ),
                ),
                target=EditTargetSpec(
                    kind="after_anchor_to_clause_end", anchor_id="connector"
                ),
                candidates=(
                    CandidateSpec(
                        op=PatchOp.REPLACE,
                        text=" and needs reassessment",
                        priority=10,
                    ),
                ),
                policy=DecisionPolicySpec(),
                fire=FirePolicySpec(mode="repeat"),
            ),
        )
    )

    text = "Findings noticed 24 hours conductive etiology is likely."
    out = PlanRunner().run(
        plan, create_step_snapshot(snapshot_text=text, token_index=0)
    )
    assert out.applied_patches
    edited = apply_proposal_to_text(text, out.applied_patches[0].proposal)
    assert "noticed 24 hours" in edited
    assert "conductive" not in edited


def test_guard_supports_explicit_regex_syntax() -> None:
    plan = PlanIR(
        rules=(
            RulePlan(
                rule_id="r-regex",
                name="r-regex",
                guard_scope=ScopeSpec(kind="whole_doc", casefold=True),
                edit_scope=ScopeSpec(kind="whole_doc", casefold=True),
                guard=_guard(required_before_any=(r"/within\s+\d+\s+hours/",)),
                target=EditTargetSpec(kind="scope_entire"),
                candidates=(
                    CandidateSpec(
                        op=PatchOp.REPLACE, text="REGEX", priority=10
                    ),
                ),
                policy=DecisionPolicySpec(),
                fire=FirePolicySpec(mode="repeat"),
            ),
        )
    )

    out = PlanRunner().run(
        plan,
        create_step_snapshot(
            snapshot_text="Started within 36 hours.", token_index=0
        ),
    )
    assert out.applied_patches


def test_required_before_incomplete_matches_when_no_side_present() -> None:
    plan = PlanIR(
        rules=(
            RulePlan(
                rule_id="r-incomplete-none",
                name="avoid:incomplete",
                guard_scope=ScopeSpec(kind="whole_doc", casefold=True),
                edit_scope=ScopeSpec(kind="whole_doc", casefold=True),
                guard=_guard(
                    required_before_incomplete=("left", "right"),
                    required_after_any=("sensorineural",),
                ),
                target=EditTargetSpec(kind="scope_entire"),
                candidates=(
                    CandidateSpec(
                        op=PatchOp.REPLACE, text="BLOCK", priority=10
                    ),
                ),
                policy=DecisionPolicySpec(),
                fire=FirePolicySpec(mode="repeat"),
            ),
        )
    )

    out = PlanRunner().run(
        plan,
        create_step_snapshot(
            snapshot_text="The patient has sensorineural hearing loss.",
            token_index=0,
        ),
    )
    assert out.applied_patches


def test_required_before_incomplete_matches_when_only_one_side_present() -> (
    None
):
    plan = PlanIR(
        rules=(
            RulePlan(
                rule_id="r-incomplete-one-side",
                name="avoid:incomplete",
                guard_scope=ScopeSpec(kind="whole_doc", casefold=True),
                edit_scope=ScopeSpec(kind="whole_doc", casefold=True),
                guard=_guard(
                    required_before_incomplete=("left", "right"),
                    required_after_any=("sensorineural",),
                ),
                target=EditTargetSpec(kind="scope_entire"),
                candidates=(
                    CandidateSpec(
                        op=PatchOp.REPLACE, text="BLOCK", priority=10
                    ),
                ),
                policy=DecisionPolicySpec(),
                fire=FirePolicySpec(mode="repeat"),
            ),
        )
    )

    out = PlanRunner().run(
        plan,
        create_step_snapshot(
            snapshot_text="The left ear has sensorineural hearing loss.",
            token_index=0,
        ),
    )
    assert out.applied_patches


def test_required_before_incomplete_does_not_match() -> None:
    plan = PlanIR(
        rules=(
            RulePlan(
                rule_id="r-incomplete-both",
                name="avoid:incomplete",
                guard_scope=ScopeSpec(kind="whole_doc", casefold=True),
                edit_scope=ScopeSpec(kind="whole_doc", casefold=True),
                guard=_guard(
                    required_before_incomplete=("left", "right"),
                    required_after_any=("sensorineural",),
                ),
                target=EditTargetSpec(kind="scope_entire"),
                candidates=(
                    CandidateSpec(
                        op=PatchOp.REPLACE, text="BLOCK", priority=10
                    ),
                ),
                policy=DecisionPolicySpec(),
                fire=FirePolicySpec(mode="repeat"),
            ),
        )
    )

    out = PlanRunner().run(
        plan,
        create_step_snapshot(
            snapshot_text=(
                "The left and right ears have sensorineural hearing loss."
            ),
            token_index=0,
        ),
    )
    assert not out.applied_patches
