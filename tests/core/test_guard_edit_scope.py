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


def _postfix_plan() -> PlanIR:
    return PlanIR(
        rules=(
            RulePlan(
                rule_id="r-avoid",
                name="avoid:conductive",
                guard_scope=ScopeSpec(
                    kind="tail_sentences", n=2, casefold=True
                ),
                edit_scope=ScopeSpec(kind="tail_sentences", n=2, casefold=True),
                guard=_guard(
                    required_before_all=(
                        "weber",
                        "rinne",
                        "left",
                        "right",
                        "positive",
                    ),
                    connector_terms=("this suggests",),
                    required_after_any=("conductive",),
                    require_order="prefix_before_postfix",
                ),
                anchors=(
                    AnchorQuerySpec(
                        anchor_id="connector",
                        match_phrase_any=("this suggests",),
                        match_mode="last",
                    ),
                ),
                target=EditTargetSpec(
                    kind="after_anchor_to_sentence_end",
                    anchor_id="connector",
                ),
                candidates=(
                    CandidateSpec(
                        op=PatchOp.REPLACE,
                        text=" these findings require further evaluation",
                        label="fallback_1",
                        priority=10,
                    ),
                ),
                policy=DecisionPolicySpec(skip_tokens=0),
                fire=FirePolicySpec(mode="repeat"),
            ),
        )
    )


def _everything_plan() -> PlanIR:
    return PlanIR(
        rules=(
            RulePlan(
                rule_id="r-avoid-everything",
                name="avoid:conductive:everything",
                guard_scope=ScopeSpec(
                    kind="tail_sentences", n=2, casefold=True
                ),
                edit_scope=ScopeSpec(kind="tail_sentences", n=2, casefold=True),
                guard=_guard(
                    required_before_all=(
                        "weber",
                        "rinne",
                        "left",
                        "right",
                        "positive",
                    ),
                    required_after_any=("conductive",),
                    require_order="prefix_before_postfix",
                ),
                target=EditTargetSpec(kind="scope_entire"),
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


def test_anchor_prev_sentence_rewrites_when_scope_includes_it() -> None:
    text = (
        "weber rinne left right positive this suggests "
        "conductive hearing loss. "
        "Current sentence has no connector."
    )
    out = PlanRunner().run(
        _postfix_plan(),
        create_step_snapshot(snapshot_text=text, token_index=10),
    )
    assert out.applied_patches
    proposal = out.applied_patches[0].proposal
    assert proposal.span_abs is not None
    start, end = proposal.span_abs
    first_sentence_end = text.index(".") + 1
    assert start < first_sentence_end
    assert end <= first_sentence_end

    edited = apply_proposal_to_text(text, proposal)
    first_sentence = edited.split(".", maxsplit=1)[0]
    assert "conductive" not in first_sentence
    assert "Current sentence has no connector." in edited


def test_anchor_in_current_sentence_rewrites_postfix_only() -> None:
    text = (
        "weber rinne left right positive in exam. "
        "In this sentence this suggests conductive hearing loss."
    )
    out = PlanRunner().run(
        _postfix_plan(),
        create_step_snapshot(snapshot_text=text, token_index=10),
    )
    assert out.applied_patches
    proposal = out.applied_patches[0].proposal
    assert proposal.span_abs is not None
    start, _end = proposal.span_abs
    assert text[start:].startswith(" conductive")
    edited = apply_proposal_to_text(text, proposal)
    assert "conductive" not in edited
    assert "weber rinne left right positive" in edited
    assert edited.endswith(".")


def test_avoid_everything_rewrites_trigger_sentence_not_next() -> None:
    text = (
        "weber rinne left right positive conductive hearing loss is likely. "
        "Given the acute onset, correlate clinically."
    )
    out = PlanRunner().run(
        _everything_plan(),
        create_step_snapshot(snapshot_text=text, token_index=10),
    )
    assert out.applied_patches
    proposal = out.applied_patches[0].proposal
    assert proposal.span_abs is not None
    start, end = proposal.span_abs
    first_sentence_end = text.index(".") + 1
    assert start == 0
    assert end >= first_sentence_end

    edited = apply_proposal_to_text(text, proposal)
    assert edited == "SAFE"


def test_repeat_replace_rules_noop_when_already_satisfied() -> None:
    plan = PlanIR(
        rules=(
            RulePlan(
                rule_id="r1",
                name="r1",
                guard_scope=ScopeSpec(
                    kind="tail_sentences", n=1, casefold=True
                ),
                edit_scope=ScopeSpec(kind="tail_sentences", n=1, casefold=True),
                anchors=(
                    AnchorQuerySpec(
                        anchor_id="a1",
                        match_phrase_any=("foo",),
                        match_mode="first",
                    ),
                ),
                target=EditTargetSpec(kind="match_span", anchor_id="a1"),
                candidates=(
                    CandidateSpec(op=PatchOp.REPLACE, text="FOO", priority=10),
                ),
                policy=DecisionPolicySpec(),
                fire=FirePolicySpec(mode="repeat"),
            ),
            RulePlan(
                rule_id="r2",
                name="r2",
                guard_scope=ScopeSpec(
                    kind="tail_sentences", n=1, casefold=True
                ),
                edit_scope=ScopeSpec(kind="tail_sentences", n=1, casefold=True),
                anchors=(
                    AnchorQuerySpec(
                        anchor_id="a2",
                        match_phrase_any=("bar",),
                        match_mode="last",
                    ),
                ),
                target=EditTargetSpec(kind="match_span", anchor_id="a2"),
                candidates=(
                    CandidateSpec(op=PatchOp.REPLACE, text="BAR", priority=10),
                ),
                policy=DecisionPolicySpec(),
                fire=FirePolicySpec(mode="repeat"),
            ),
        )
    )
    out = PlanRunner().run(
        plan,
        create_step_snapshot(snapshot_text="foo and bar.", token_index=5),
    )
    assert len(out.applied_patches) == 0
    assert out.proposals == []


def test_avoid_can_still_trigger_when_postfix_overlap_is_actionable() -> None:
    plan = PlanIR(
        rules=(
            RulePlan(
                rule_id="r-avoid-helpless",
                name="avoid:incomplete laterality then diagnosis",
                guard_scope=ScopeSpec(kind="whole_doc", casefold=True),
                edit_scope=ScopeSpec(kind="tail_sentences", n=1, casefold=True),
                guard=_guard(
                    required_before_all=("left",),
                    required_after_any=("sensorineural",),
                ),
                target=EditTargetSpec(kind="scope_entire"),
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
    text = (
        "Hearing loss in the left ear appears sensorineural. "
        "This condition requires urgent treatment. "
        "The diagnosis may still be sensorineural."
    )

    out = PlanRunner().run(
        plan, create_step_snapshot(snapshot_text=text, token_index=10)
    )

    assert out.applied_patches


def test_avoid_triggers_when_target_overlaps_all_matches() -> None:
    plan = PlanIR(
        rules=(
            RulePlan(
                rule_id="r-avoid-effective",
                name="avoid:single sentence diagnosis",
                guard_scope=ScopeSpec(
                    kind="tail_sentences", n=1, casefold=True
                ),
                edit_scope=ScopeSpec(kind="tail_sentences", n=1, casefold=True),
                guard=_guard(
                    required_before_all=("left",),
                    required_after_any=("sensorineural",),
                ),
                target=EditTargetSpec(kind="scope_entire"),
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
    text = "The hearing loss in the left ear is likely sensorineural."

    out = PlanRunner().run(
        plan, create_step_snapshot(snapshot_text=text, token_index=10)
    )

    assert out.applied_patches
    edited = apply_proposal_to_text(text, out.applied_patches[0].proposal)
    assert edited == "SAFE"
