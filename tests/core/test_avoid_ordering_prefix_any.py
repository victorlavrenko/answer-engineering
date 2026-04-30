from __future__ import annotations

from answer_engineering.engine.orchestration.orchestrator import (
    PlanRunner,
)
from answer_engineering.engine.runtime.runtime_types import (
    PatchOp,
)
from answer_engineering.rules.compile.plan import (
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
    create_step_snapshot,
)
from tests.core.match_tree_guard_factory import build_guard_expression


def _guard(**legacy_fields: object) -> GuardSpec:
    return GuardSpec(expression=build_guard_expression(**legacy_fields))


def _plan() -> PlanIR:
    return PlanIR(
        rules=(
            RulePlan(
                rule_id="r-avoid",
                name="avoid:diagnosis then tests",
                guard_scope=ScopeSpec(
                    kind="tail_sentences", n=2, casefold=True
                ),
                edit_scope=ScopeSpec(kind="tail_sentences", n=2, casefold=True),
                guard=_guard(
                    required_before_any=("conductive",),
                    required_after_any=("test",),
                    require_order="prefix_before_postfix",
                ),
                target=EditTargetSpec(kind="scope_entire"),
                candidates=(
                    CandidateSpec(
                        op=PatchOp.REPLACE,
                        text="The test results shall be analyzed carefully.",
                        label="fallback_1",
                        priority=10,
                    ),
                ),
                policy=DecisionPolicySpec(skip_tokens=0),
                fire=FirePolicySpec(mode="repeat"),
            ),
        )
    )


def test_avoid_ordering_no_fire_postfix_before_required() -> None:
    """Repro for postfix-before-prefix ordering."""
    text = (
        "The Weber test lateralizing to the left ear suggests a conductive"
        "hearing loss."
    )
    out = PlanRunner().run(
        _plan(),
        create_step_snapshot(snapshot_text=text, token_index=20),
    )
    assert not out.applied_patches


def test_avoid_ordering_fires_when_postfix_after_required_before_any() -> None:
    text = (
        "The findings suggest conductive hearing loss and further test"
        "correlation is needed."
    )
    out = PlanRunner().run(
        _plan(),
        create_step_snapshot(snapshot_text=text, token_index=20),
    )
    assert out.applied_patches


def _plan_required_after_all() -> PlanIR:
    return PlanIR(
        rules=(
            RulePlan(
                rule_id="r-avoid-all",
                name="avoid:diagnosis then tests all",
                guard_scope=ScopeSpec(
                    kind="tail_sentences", n=2, casefold=True
                ),
                edit_scope=ScopeSpec(kind="tail_sentences", n=2, casefold=True),
                guard=_guard(
                    required_before_any=("conductive",),
                    required_after_all=("test", "results"),
                    require_order="prefix_before_postfix",
                ),
                target=EditTargetSpec(kind="scope_entire"),
                candidates=(
                    CandidateSpec(
                        op=PatchOp.REPLACE,
                        text="The test results shall be analyzed carefully.",
                        label="fallback_1",
                        priority=10,
                    ),
                ),
                policy=DecisionPolicySpec(skip_tokens=0),
                fire=FirePolicySpec(mode="repeat"),
            ),
        )
    )


def test_required_after_all_no_fire_when_only_before() -> None:
    text = (
        "Test results suggest concern, but later this may be conductive hearing"
        "loss."
    )
    out = PlanRunner().run(
        _plan_required_after_all(),
        create_step_snapshot(snapshot_text=text, token_index=20),
    )
    assert not out.applied_patches


def test_required_after_all_fires_when_all_after() -> None:
    text = (
        "This appears conductive and test results suggest middle-ear pathology."
    )
    out = PlanRunner().run(
        _plan_required_after_all(),
        create_step_snapshot(snapshot_text=text, token_index=20),
    )
    assert out.applied_patches


def _plan_required_before_any_with_two_required_after_any() -> PlanIR:
    return PlanIR(
        rules=(
            RulePlan(
                rule_id="r-avoid-postfix-any-mixed",
                name="avoid:mixed postfix ordering",
                guard_scope=ScopeSpec(
                    kind="tail_sentences", n=2, casefold=True
                ),
                edit_scope=ScopeSpec(kind="tail_sentences", n=2, casefold=True),
                guard=_guard(
                    required_before_any=("conductive",),
                    required_after_any=("test", "results"),
                    require_order="prefix_before_postfix",
                ),
                target=EditTargetSpec(kind="scope_entire"),
                candidates=(
                    CandidateSpec(
                        op=PatchOp.REPLACE,
                        text="The test results shall be analyzed carefully.",
                        label="fallback_1",
                        priority=10,
                    ),
                ),
                policy=DecisionPolicySpec(skip_tokens=0),
                fire=FirePolicySpec(mode="repeat"),
            ),
        )
    )


def test_any_postfix_after_prefix_fires_even_if_other_before() -> None:
    text = (
        "Test comments were noted; conductive hearing loss with later results"
        "pending."
    )
    out = PlanRunner().run(
        _plan_required_before_any_with_two_required_after_any(),
        create_step_snapshot(snapshot_text=text, token_index=20),
    )
    assert out.applied_patches


def test_postfix_on_both_sides_of_required_before_any_fires() -> None:
    """Guard fires when any postfix appears after a prefix."""
    text = (
        "The tuning fork tests are consistent with a conductive hearing "
        "loss on the right side, "
        "as the Rinne test is positive and the Weber test lateralizes to "
        "the left ear."
    )
    out = PlanRunner().run(
        _plan(),
        create_step_snapshot(snapshot_text=text, token_index=40),
    )
    assert out.applied_patches


def test_avoid_ordering_fires_when_prefix_and_postfix_are_adjacent() -> None:
    text = "Findings are conductivetest driven."
    out = PlanRunner().run(
        _plan(),
        create_step_snapshot(snapshot_text=text, token_index=20),
    )
    assert out.applied_patches
