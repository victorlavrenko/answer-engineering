from __future__ import annotations

from collections.abc import Callable, Iterable

import pytest

from answer_engineering.engine.pipeline import events as runtime_events
from answer_engineering.engine.pipeline.events import (
    GuardConditionEvaluated,
    ProposalAccepted,
    ProposalsGenerated,
    RuleEvaluationStarted,
)
from answer_engineering.engine.telemetry.aggregation.aggregator import (
    RuntimeTelemetryAggregator,
)
from answer_engineering.telemetry import RuntimeTelemetrySnapshot


def _build_snapshot(
    *,
    events: Iterable[runtime_events.Event],
    rule_name_for: Callable[[str], str],
    decision_limit_reached: bool,
) -> RuntimeTelemetrySnapshot:
    aggregator = RuntimeTelemetryAggregator(rule_name_for=rule_name_for)
    aggregator.observe_events(events)
    return aggregator.build_snapshot(
        decision_limit_reached=decision_limit_reached
    )


def test_runtime_telemetry_aggregator_builds_typed_snapshot() -> None:
    snapshot = _build_snapshot(
        events=[
            RuleEvaluationStarted(
                rule_id="r1",
                doc_version_id="v1",
                scope_spec={},
            ),
            GuardConditionEvaluated(
                rule_id="r1",
                node_id="",
                node_path="guard.prefix",
                node_type="all",
                marker="prefix_all",
                debug_expression="x",
                matched=True,
                spans=((0, 1),),
            ),
            ProposalsGenerated(
                rule_id="r1",
                proposals_count=2,
                generated_count=1,
                fallback_count=1,
                static_count=0,
                noop_count=0,
            ),
            ProposalAccepted(
                rule_id="r1",
                proposal_summary="selected",
                patch_hash="abc",
                patch_bytes_len=8,
                candidate_kind="fallback",
                candidate_id="c42",
                candidate_label="fallback sentence",
            ),
        ],
        rule_name_for=lambda rule_id: f"name-{rule_id}",
        decision_limit_reached=True,
    )

    assert snapshot.applied_decisions == 1
    assert snapshot.decision_limit_reached is True
    assert snapshot.rules_triggered_count == 1
    assert snapshot.rules_applied_count == 1
    assert len(snapshot.events) == 4

    rule = snapshot.rules[0]
    assert rule.rule_name == "name-r1"
    assert rule.evaluations == 1
    assert rule.applied == 1
    assert rule.proposals_generated == 2

    condition = rule.conditions[0]
    assert condition.node_path == "prefix"
    assert condition.matched == 1
    assert condition.seen == 1

    choice = rule.candidate_choices[0]
    assert choice.kind == "fallback"
    assert choice.candidate_id == "c42"
    assert choice.chosen == 1


def test_runtime_telemetry_aggregator_rejects_empty_candidate_id() -> None:
    with pytest.raises(ValueError, match="candidate_id must be non-empty"):
        _build_snapshot(
            events=[
                ProposalAccepted(
                    rule_id="r2",
                    proposal_summary="selected",
                    patch_hash="def",
                    patch_bytes_len=4,
                    candidate_kind="fallback",
                    candidate_id="",
                    candidate_label="fallback sentence",
                )
            ],
            rule_name_for=lambda rule_id: rule_id,
            decision_limit_reached=False,
        )


def test_runtime_telemetry_aggregator_keeps_static_candidate_label() -> None:
    snapshot = _build_snapshot(
        events=[
            RuleEvaluationStarted(
                rule_id="r3",
                doc_version_id="v1",
                scope_spec={},
            ),
            ProposalAccepted(
                rule_id="r3",
                proposal_summary="selected",
                patch_hash="ghi",
                patch_bytes_len=9,
                candidate_kind="static",
                candidate_id="rewrite_2",
                candidate_label="SSNHL",
            ),
        ],
        rule_name_for=lambda rule_id: rule_id,
        decision_limit_reached=False,
    )

    choice = snapshot.rules[0].candidate_choices[0]
    assert choice.kind == "static"
    assert choice.candidate_id == "rewrite_2"
    assert choice.label == "SSNHL"


def test_runtime_telem_aggregator_merges_same_auth_condition_idtity() -> None:
    snapshot = _build_snapshot(
        events=[
            RuleEvaluationStarted(
                rule_id="r1",
                doc_version_id="v1",
                scope_spec={},
            ),
            GuardConditionEvaluated(
                rule_id="r1",
                node_id="guard.0",
                node_path="guard.0",
                node_type="MatchTerm",
                marker="prefix_any",
                debug_expression="sudden",
                matched=True,
                spans=((0, 6),),
            ),
            GuardConditionEvaluated(
                rule_id="r1",
                node_id="guard.3",
                node_path="guard.3",
                node_type="MatchTerm",
                marker="prefix_any",
                debug_expression="sudden",
                matched=True,
                spans=((15, 21),),
            ),
        ],
        rule_name_for=lambda rule_id: rule_id,
        decision_limit_reached=False,
    )

    assert len(snapshot.rules[0].conditions) == 1
    condition = snapshot.rules[0].conditions[0]
    assert condition.condition_id == "prefix:any:sudden"
    assert condition.node_path == "prefix"
    assert condition.node_type == "any"
    assert condition.matched == 2
    assert condition.seen == 2


def test_runtime_telemetry_aggregator_keeps_none_and_incomplete_distinct() -> (
    None
):
    snapshot = _build_snapshot(
        events=[
            RuleEvaluationStarted(
                rule_id="r1",
                doc_version_id="v1",
                scope_spec={},
            ),
            GuardConditionEvaluated(
                rule_id="r1",
                node_id="guard.none",
                node_path="guard.none",
                node_type="MatchTerm",
                marker="prefix_none",
                debug_expression="bilateral",
                matched=True,
                spans=((0, 9),),
            ),
            GuardConditionEvaluated(
                rule_id="r1",
                node_id="guard.incomplete",
                node_path="guard.incomplete",
                node_type="MatchTerm",
                marker="prefix_incomplete",
                debug_expression="bilateral",
                matched=True,
                spans=((0, 9),),
            ),
        ],
        rule_name_for=lambda rule_id: rule_id,
        decision_limit_reached=False,
    )

    conditions = sorted(
        snapshot.rules[0].conditions,
        key=lambda condition: condition.condition_id,
    )
    assert [condition.condition_id for condition in conditions] == [
        "prefix:incomplete:bilateral",
        "prefix:none:bilateral",
    ]
