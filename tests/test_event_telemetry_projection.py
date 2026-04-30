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
from answer_engineering.engine.telemetry.events.event_sink import (
    CompositeRuntimeEventSink,
    RecordingRuntimeEventSink,
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


def test_runtime_telemetry_from_events_aggregates_rule_metrics() -> None:
    telemetry = _build_snapshot(
        events=[
            RuleEvaluationStarted(
                rule_id="r1", doc_version_id="v1", scope_spec={}
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
                proposals_count=3,
                generated_count=1,
                fallback_count=1,
                static_count=1,
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

    assert telemetry.applied_decisions == 1
    assert telemetry.decision_limit_reached is True
    assert len(telemetry.events) == 4

    rule = telemetry.rules[0]
    assert rule.rule_name == "name-r1"
    assert rule.evaluations == 1
    assert rule.applied == 1
    assert rule.trigger_firings == 1
    assert rule.proposals_generated == 3
    assert rule.generated_candidates_considered == 1
    assert rule.fallback_candidates_considered == 1
    assert rule.static_candidates_considered == 1
    assert rule.noop_candidates_generated == 0

    condition = rule.conditions[0]
    assert condition.node_path == "prefix"
    assert condition.seen == 1
    assert condition.matched == 1

    choice = rule.candidate_choices[0]
    assert choice.kind == "fallback"
    assert choice.candidate_id == "c42"
    assert choice.label == "fallback sentence"
    assert choice.chosen == 1


def test_runtime_telemetry_from_events_requires_nonempty_candidate() -> None:
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


def test_recording_sink_collects_events_for_aggregator() -> None:
    recording_sink = RecordingRuntimeEventSink()
    composite = CompositeRuntimeEventSink((recording_sink,))
    composite.emit(
        RuleEvaluationStarted(
            rule_id="r1",
            doc_version_id="v1",
            scope_spec={},
        )
    )
    composite.emit(
        ProposalAccepted(
            rule_id="r1",
            proposal_summary="selected",
            patch_hash="abc",
            patch_bytes_len=10,
            candidate_kind="static",
            candidate_id="c1",
            candidate_label="SSNHL",
        )
    )

    telemetry = _build_snapshot(
        events=recording_sink.events,
        rule_name_for=lambda rule_id: f"name-{rule_id}",
        decision_limit_reached=False,
    )
    assert telemetry.applied_decisions == 1
    rule = telemetry.rules[0]
    assert rule.evaluations == 1
    assert rule.candidate_choices[0].chosen == 1


def test_runtime_telemetry_preserves_static_candidate_authored_label() -> None:
    telemetry = _build_snapshot(
        events=[
            RuleEvaluationStarted(
                rule_id="r-static",
                doc_version_id="v1",
                scope_spec={},
            ),
            ProposalAccepted(
                rule_id="r-static",
                proposal_summary="selected",
                patch_hash="hash",
                patch_bytes_len=12,
                candidate_kind="static",
                candidate_id="rewrite_2",
                candidate_label="SSNHL",
            ),
        ],
        rule_name_for=lambda rule_id: f"name-{rule_id}",
        decision_limit_reached=False,
    )

    candidate = telemetry.rules[0].candidate_choices[0]
    assert candidate.kind == "static"
    assert candidate.candidate_id == "rewrite_2"
    assert candidate.label == "SSNHL"
