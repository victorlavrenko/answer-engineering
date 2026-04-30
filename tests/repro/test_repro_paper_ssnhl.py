from __future__ import annotations

from typing import Any, cast

from ae_paper_reproduction.telemetry import (
    ssnhl_experiment,
)
from answer_engineering.engine.pipeline import (
    events as runtime_events,
)
from answer_engineering.telemetry import (
    CandidateTelemetrySnapshot,
    RuleTelemetrySnapshot,
    RuntimeTelemetrySnapshot,
)

summarize_telemetry = ssnhl_experiment.summarize_telemetry


def _snapshot_rows(
    rows: list[dict[str, Any]],
) -> list[RuntimeTelemetrySnapshot]:
    snapshots: list[RuntimeTelemetrySnapshot] = []

    def _event_from_payload(
        payload: dict[str, Any],
    ) -> runtime_events.Event:
        event_type = str(payload.get("type", ""))
        match event_type:
            case "ProposalsGenerated":
                return runtime_events.ProposalsGenerated(
                    rule_id=str(payload.get("rule_id", "")),
                    proposals_count=int(payload.get("proposals_count", 0)),
                    generated_count=int(payload.get("generated_count", 0)),
                    fallback_count=int(payload.get("fallback_count", 0)),
                    static_count=int(payload.get("static_count", 0)),
                    noop_count=int(payload.get("noop_count", 0)),
                )
            case "AvoidProbeSetGenerated":
                return runtime_events.AvoidProbeSetGenerated(
                    rule_id=str(payload.get("rule_id", "")),
                    cache_key=str(payload.get("cache_key", "")),
                    generated_count=int(payload.get("generated_count", 0)),
                    probe_budget=int(payload.get("probe_budget", 0)),
                )
            case "AvoidProbeEpisodeStarted":
                return runtime_events.AvoidProbeEpisodeStarted(
                    rule_id=str(payload.get("rule_id", "")),
                    cache_key=str(payload.get("cache_key", "")),
                    generated_count=int(payload.get("generated_count", 0)),
                )
            case "AvoidProbeCandidatePopped":
                return runtime_events.AvoidProbeCandidatePopped(
                    rule_id=str(payload.get("rule_id", "")),
                    cache_key=str(payload.get("cache_key", "")),
                    candidate_id=str(payload.get("candidate_id", "")),
                )
            case "AvoidProbeCacheExhausted":
                return runtime_events.AvoidProbeCacheExhausted(
                    rule_id=str(payload.get("rule_id", "")),
                    cache_key=str(payload.get("cache_key", "")),
                    empty_request_count=int(
                        payload.get("empty_request_count", 0)
                    ),
                )
            case "ProposalRejected":
                return runtime_events.ProposalRejected(
                    rule_id=str(payload.get("rule_id", "")),
                    reason=str(payload.get("reason", "")),
                )
            case "PatchSkipped":
                return runtime_events.PatchSkipped(
                    rule_id=str(payload.get("rule_id", "")),
                    reason=str(payload.get("reason", "")),
                )
            case _:
                return runtime_events.DebugEvent(
                    msg=f"unsupported:{event_type}"
                )

    for row in rows:
        events = tuple(
            _event_from_payload(event)
            for event in cast(list[dict[str, Any]], row.get("events", []))
        )
        rules: list[RuleTelemetrySnapshot] = []
        for raw_rule in cast(dict[str, Any], row.get("rules", {})).values():
            candidate_choices: list[CandidateTelemetrySnapshot] = []
            for raw_choice in cast(
                dict[str, Any], raw_rule.get("candidate_choices", {})
            ).values():
                candidate_choices.append(
                    CandidateTelemetrySnapshot(
                        kind=str(raw_choice.get("kind", "")),
                        candidate_id=str(raw_choice.get("candidate_id", "")),
                        label=str(
                            raw_choice.get(
                                "label", raw_choice.get("candidate_id", "")
                            )
                        ),
                        chosen=int(raw_choice.get("chosen", 0)),
                    )
                )
            rules.append(
                RuleTelemetrySnapshot(
                    rule_id=str(raw_rule.get("rule_id", "")),
                    rule_name=str(raw_rule.get("rule_name", "")),
                    evaluations=int(raw_rule.get("evaluations", 0)),
                    applied=int(raw_rule.get("applied", 0)),
                    trigger_firings=int(raw_rule.get("trigger_firings", 0)),
                    proposals_generated=int(
                        raw_rule.get("proposals_generated", 0)
                    ),
                    generated_candidates_considered=int(
                        raw_rule.get("generated_candidates_considered", 0)
                    ),
                    fallback_candidates_considered=int(
                        raw_rule.get("fallback_candidates_considered", 0)
                    ),
                    static_candidates_considered=int(
                        raw_rule.get("static_candidates_considered", 0)
                    ),
                    noop_candidates_generated=int(
                        raw_rule.get("noop_candidates_generated", 0)
                    ),
                    conditions=(),
                    candidate_choices=tuple(candidate_choices),
                )
            )
        snapshots.append(
            RuntimeTelemetrySnapshot(
                runtime_sec=cast(float | None, row.get("runtime_sec")),
                applied_decisions=int(row.get("applied_decisions", 0)),
                decision_limit_reached=bool(
                    row.get("decision_limit_reached", False)
                ),
                rules=tuple(rules),
                events=events,
            )
        )
    return snapshots


def test_summarize_telemetry_includes_probe_depth_mean_and_fields() -> None:
    telemetry: list[dict[str, Any]] = [
        {
            "applied_decisions": 3,
            "runtime_sec": 0.75,
            "events": [
                {
                    "type": "ProposalsGenerated",
                    "rule_id": "avoid_rule",
                    "generated_count": 3,
                    "fallback_count": 0,
                },
                {
                    "type": "ProposalsGenerated",
                    "rule_id": "avoid_rule",
                    "generated_count": 2,
                    "fallback_count": 1,
                },
            ],
            "rules": {
                "1": {
                    "rule_id": "avoid_rule",
                    "rule_name": "avoid:inference",
                    "evaluations": 2,
                    "trigger_firings": 2,
                    "generated_candidates_considered": 5,
                    "applied": 2,
                    "candidate_choices": {
                        "generated:probe_1": {
                            "kind": "generated",
                            "candidate_id": "probe_1",
                            "chosen": 1,
                        },
                        "fallback:fallback_1": {
                            "kind": "fallback",
                            "candidate_id": "fallback_1",
                            "chosen": 1,
                        },
                    },
                },
                "2": {
                    "rule_name": "replace:ssnhl",
                    "evaluations": 3,
                    "trigger_firings": 3,
                    "applied": 1,
                    "candidate_choices": {},
                },
            },
        }
    ]

    summary = summarize_telemetry(_snapshot_rows(telemetry), fallback_depth=11)

    assert summary.trigger_firings_total == 5
    assert summary.probe_count_total == 5
    assert summary.probe_count_mean == 5.0
    assert summary.probe_depth_max == 11
    assert summary.probe_depth_mean == 6.0
    assert summary.runtime_seconds_total == 0.75
    assert summary.runtime_seconds_per_case == 0.75
    assert summary.avg_triggers_per_case == 5.0
    assert summary.avg_interventions_per_case == 3.0
    assert summary.fallbacks_used == 1
    assert summary.edits_proposed == 5
    assert summary.edits_accepted == 3
    assert sum(summary.rule_family_counts.values()) == summary.edits_accepted
    assert summary.rule_family_counts["avoid"] == 2
    assert summary.rule_family_counts["replace"] == 1
    assert summary.avoid_firings == 2
    assert summary.avoid_firings_with_generated_candidates == 2
    assert summary.avoid_interventions_per_case == 2.0
    assert summary.avoid_probe_episodes_per_case == 2.0
    assert summary.avoid_outcomes_total == 2
    assert summary.accepted_rank_1_share == 0.5
    assert summary.accepted_rank_gt_1_share == 0.0
    assert summary.resolved_within_top_3_share == 0.5
    assert summary.resolved_within_top_5_share == 0.5
    assert summary.not_enough_probes_count == 0
    assert summary.not_enough_probes_share == 0.0
    assert summary.avg_probes_per_avoid_firing == 2.5
    assert summary.rule_family_counts["avoid"] == 2
    assert summary.rule_family_trigger_counts["avoid"] == 2
    assert summary.rule_family_case_counts["avoid"] == 1
    assert summary.rule_family_counts["replace"] == 1
    assert summary.rule_family_trigger_counts["replace"] == 3


def test_summarize_telemetry_defaults_when_no_candidates_are_chosen() -> None:
    summary = summarize_telemetry(
        _snapshot_rows(
            [{"applied_decisions": 0, "rules": cast(dict[str, Any], {})}]
        )
    )

    assert summary.runtime_seconds_total == 0.0
    assert summary.runtime_seconds_per_case == 0.0
    assert summary.trigger_firings_total == 0
    assert summary.probe_count_total == 0
    assert summary.probe_count_mean == 0.0
    assert summary.probe_depth_mean == 0.0
    assert summary.probe_depth_max == 0
    assert summary.avg_interventions_per_case == 0.0
    assert summary.avg_probes_per_avoid_firing == 0.0
    assert summary.avoid_interventions_per_case == 0.0
    assert summary.avoid_probe_episodes_per_case == 0.0


def test_summarize_telemetry_tracks_probe_generation_and_coverage() -> None:
    summary = summarize_telemetry(
        _snapshot_rows(
            [
                {
                    "applied_decisions": 0,
                    "events": [
                        {
                            "type": "AvoidProbeSetGenerated",
                            "rule_id": "avoid_rule_a",
                            "cache_key": "cache-a",
                            "generated_count": 10,
                            "probe_budget": 10,
                        },
                        {
                            "type": "AvoidProbeEpisodeStarted",
                            "rule_id": "avoid_rule_a",
                            "cache_key": "cache-a",
                            "generated_count": 10,
                        },
                        {
                            "type": "AvoidProbeCandidatePopped",
                            "rule_id": "avoid_rule_a",
                            "cache_key": "cache-a",
                            "candidate_id": "probe_1",
                        },
                        {
                            "type": "AvoidProbeCandidatePopped",
                            "rule_id": "avoid_rule_a",
                            "cache_key": "cache-a",
                            "candidate_id": "probe_2",
                        },
                        {
                            "type": "AvoidProbeCandidatePopped",
                            "rule_id": "avoid_rule_a",
                            "cache_key": "cache-a",
                            "candidate_id": "probe_3",
                        },
                    ],
                    "rules": {
                        "avoid_a": {
                            "rule_id": "avoid_rule_a",
                            "rule_name": "avoid:first",
                            "trigger_firings": 1,
                            "candidate_choices": {},
                        }
                    },
                },
                {
                    "applied_decisions": 0,
                    "events": [
                        {
                            "type": "AvoidProbeSetGenerated",
                            "rule_id": "avoid_rule_b",
                            "cache_key": "cache-b",
                            "generated_count": 8,
                            "probe_budget": 10,
                        },
                        {
                            "type": "AvoidProbeEpisodeStarted",
                            "rule_id": "avoid_rule_b",
                            "cache_key": "cache-b",
                            "generated_count": 8,
                        },
                        *[
                            {
                                "type": "AvoidProbeCandidatePopped",
                                "rule_id": "avoid_rule_b",
                                "cache_key": "cache-b",
                                "candidate_id": f"probe_{idx}",
                            }
                            for idx in range(1, 9)
                        ],
                        {
                            "type": "AvoidProbeCacheExhausted",
                            "rule_id": "avoid_rule_b",
                            "cache_key": "cache-b",
                            "empty_request_count": 1,
                        },
                    ],
                    "rules": {
                        "avoid_b": {
                            "rule_id": "avoid_rule_b",
                            "rule_name": "avoid:second",
                            "trigger_firings": 1,
                            "candidate_choices": {},
                        }
                    },
                },
            ]
        )
    )

    assert summary.probe_sets_generated_total == 2
    assert summary.probe_sets_generated_per_case == 1.0
    assert summary.requested_probes_total == 20
    assert summary.avg_valid_probes_per_generated_set == 9.0
    assert summary.avg_requested_probes_per_generated_set == 10.0
    assert summary.generated_probes_total == 18
    assert summary.consumed_generated_probes_total == 11
    assert summary.generated_probes_consumed_share == 11 / 18
    assert summary.probe_budget_for_50_coverage == 3
    assert summary.probe_budget_for_80_coverage == 9
    assert summary.not_enough_probes_count == 1
    assert summary.not_enough_probes_share == 0.5


def test_summarize_telemetry_separates_applied_edits_and_avoid_probe() -> None:
    summary = summarize_telemetry(
        _snapshot_rows(
            [
                {
                    "applied_decisions": 1,
                    "events": [
                        {
                            "type": "ProposalsGenerated",
                            "rule_id": "avoid_rule",
                            "generated_count": 3,
                            "fallback_count": 0,
                        }
                    ],
                    "rules": {
                        "avoid": {
                            "rule_id": "avoid_rule",
                            "rule_name": "avoid:probe",
                            "trigger_firings": 3,
                            "generated_candidates_considered": 3,
                            "candidate_choices": {
                                "generated:probe_1": {
                                    "kind": "generated",
                                    "candidate_id": "probe_1",
                                    "chosen": 1,
                                },
                            },
                        },
                        "replace": {
                            "rule_name": "replace:ssnhl",
                            "trigger_firings": 2,
                            "candidate_choices": {},
                        },
                    },
                },
                {
                    "applied_decisions": 0,
                    "rules": {},
                },
            ]
        )
    )

    assert summary.avg_interventions_per_case == 0.5
    assert summary.avg_triggers_per_case == 2.5
    assert summary.avoid_interventions_per_case == 0.0
    assert summary.avoid_probe_episodes_per_case == 0.5
    assert summary.avg_probes_per_avoid_firing == 3.0
    assert (
        summary.avg_interventions_per_case
        != summary.avoid_interventions_per_case
    )


def test_summarize_telemetry_excludes_fallback_only_from_probe_average() -> (
    None
):
    summary = summarize_telemetry(
        _snapshot_rows(
            [
                {
                    "applied_decisions": 0,
                    "events": [
                        {
                            "type": "ProposalsGenerated",
                            "rule_id": "avoid_rule",
                            "generated_count": 2,
                            "fallback_count": 0,
                        },
                        {
                            "type": "ProposalsGenerated",
                            "rule_id": "avoid_rule",
                            "generated_count": 0,
                            "fallback_count": 1,
                        },
                        {
                            "type": "ProposalsGenerated",
                            "rule_id": "avoid_rule",
                            "generated_count": 0,
                            "fallback_count": 1,
                        },
                    ],
                    "rules": {
                        "avoid_with_mixed_firings": {
                            "rule_id": "avoid_rule",
                            "rule_name": "avoid:probe",
                            "trigger_firings": 3,
                            "generated_candidates_considered": 2,
                            "candidate_choices": {
                                "generated:probe_1": {
                                    "kind": "generated",
                                    "candidate_id": "probe_1",
                                    "chosen": 1,
                                },
                                "fallback:fallback_a": {
                                    "kind": "fallback",
                                    "candidate_id": "fallback_a",
                                    "chosen": 1,
                                },
                                "fallback:fallback_b": {
                                    "kind": "fallback",
                                    "candidate_id": "fallback_b",
                                    "chosen": 1,
                                },
                            },
                        },
                    },
                }
            ]
        )
    )

    assert summary.avoid_firings == 3
    assert summary.avoid_firings_with_generated_candidates == 1
    assert summary.avg_probes_per_avoid_firing == 2.0


def test_summarize_telemetry_avoid_interventions_eq_episodes_times_probes() -> (
    None
):
    summary = summarize_telemetry(
        _snapshot_rows(
            [
                {
                    "applied_decisions": 0,
                    "events": [
                        {
                            "type": "ProposalsGenerated",
                            "rule_id": "avoid_rule",
                            "generated_count": 4,
                            "fallback_count": 0,
                        },
                        {
                            "type": "ProposalsGenerated",
                            "rule_id": "avoid_rule",
                            "generated_count": 2,
                            "fallback_count": 0,
                        },
                    ],
                    "rules": {
                        "avoid": {
                            "rule_id": "avoid_rule",
                            "rule_name": "avoid:probe",
                            "trigger_firings": 2,
                            "candidate_choices": {},
                        }
                    },
                },
                {
                    "applied_decisions": 0,
                    "rules": {},
                },
            ]
        )
    )

    assert summary.avoid_interventions_per_case == 0.0
    assert summary.avoid_probe_episodes_per_case == 1.0
    assert summary.avg_probes_per_avoid_firing == 3.0


def test_summarize_telemetry_uses_per_firing_events_for_probe_average() -> None:
    summary = summarize_telemetry(
        _snapshot_rows(
            [
                {
                    "applied_decisions": 0,
                    "events": [
                        {
                            "type": "ProposalsGenerated",
                            "rule_id": "avoid_rule",
                            "generated_count": 4,
                            "fallback_count": 0,
                        },
                        {
                            "type": "ProposalsGenerated",
                            "rule_id": "avoid_rule",
                            "generated_count": 2,
                            "fallback_count": 0,
                        },
                    ],
                    "rules": {
                        "avoid_with_stale_aggregate": {
                            "rule_id": "avoid_rule",
                            "rule_name": "avoid:probe",
                            "trigger_firings": 2,
                            "generated_candidates_considered": 2,
                            "candidate_choices": {
                                "generated:probe_4": {
                                    "kind": "generated",
                                    "candidate_id": "probe_4",
                                    "chosen": 1,
                                },
                                "generated:probe_2": {
                                    "kind": "generated",
                                    "candidate_id": "probe_2",
                                    "chosen": 1,
                                },
                            },
                        },
                    },
                }
            ]
        )
    )

    assert summary.probe_count_total == 6
    assert summary.avoid_firings_with_generated_candidates == 2
    assert summary.avg_probes_per_avoid_firing == 3.0


def test_summarize_telemetry_reports_probe_budget_shares_from_exhaustion() -> (
    None
):
    summary = summarize_telemetry(
        _snapshot_rows(
            [
                {
                    "applied_decisions": 3,
                    "events": [
                        {
                            "type": "ProposalsGenerated",
                            "rule_id": "avoid_depth_1",
                            "generated_count": 1,
                            "fallback_count": 0,
                        },
                        {
                            "type": "ProposalsGenerated",
                            "rule_id": "avoid_depth_3",
                            "generated_count": 3,
                            "fallback_count": 0,
                        },
                        {
                            "type": "ProposalsGenerated",
                            "rule_id": "avoid_depth_5",
                            "generated_count": 5,
                            "fallback_count": 0,
                        },
                        {
                            "type": "ProposalsGenerated",
                            "rule_id": "avoid_fallback",
                            "generated_count": 0,
                            "fallback_count": 1,
                        },
                        {
                            "type": "AvoidProbeCacheExhausted",
                            "rule_id": "avoid_fallback",
                            "empty_request_count": 1,
                        },
                    ],
                    "rules": {
                        "avoid_depth_1": {
                            "rule_id": "avoid_depth_1",
                            "rule_name": "avoid:depth-1",
                            "trigger_firings": 1,
                            "generated_candidates_considered": 1,
                            "candidate_choices": {
                                "generated:probe_1": {
                                    "kind": "generated",
                                    "candidate_id": "probe_1",
                                    "chosen": 1,
                                }
                            },
                        },
                        "avoid_depth_3": {
                            "rule_id": "avoid_depth_3",
                            "rule_name": "avoid:depth-3",
                            "trigger_firings": 1,
                            "generated_candidates_considered": 3,
                            "candidate_choices": {
                                "generated:probe_3": {
                                    "kind": "generated",
                                    "candidate_id": "probe_3",
                                    "chosen": 1,
                                }
                            },
                        },
                        "avoid_depth_5": {
                            "rule_id": "avoid_depth_5",
                            "rule_name": "avoid:depth-5",
                            "trigger_firings": 1,
                            "generated_candidates_considered": 5,
                            "candidate_choices": {
                                "generated:probe_5": {
                                    "kind": "generated",
                                    "candidate_id": "probe_5",
                                    "chosen": 1,
                                }
                            },
                        },
                        "avoid_fallback": {
                            "rule_id": "avoid_fallback",
                            "rule_name": "avoid:fallback",
                            "trigger_firings": 1,
                            "generated_candidates_considered": 0,
                            "candidate_choices": {
                                "fallback:fallback_1": {
                                    "kind": "fallback",
                                    "candidate_id": "fallback_1",
                                    "chosen": 1,
                                }
                            },
                        },
                    },
                }
            ]
        )
    )

    assert summary.avoid_outcomes_total == 4
    assert summary.avoid_required_gt_1_share == 0.5
    assert summary.avoid_required_gt_3_share == 0.25
    assert summary.avoid_required_gt_5_share == 0.0
    assert summary.not_enough_probes_count == 1
    assert summary.not_enough_probes_share == 1 / 3


def test_summarize_telemetry_counts_rule_family_case_shares_once_per_case() -> (
    None
):
    summary = summarize_telemetry(
        _snapshot_rows(
            [
                {
                    "applied_decisions": 0,
                    "rules": {
                        "avoid_a": {
                            "rule_name": "avoid:first",
                            "trigger_firings": 2,
                            "candidate_choices": {},
                        },
                        "avoid_b": {
                            "rule_name": "avoid:second",
                            "trigger_firings": 1,
                            "candidate_choices": {},
                        },
                    },
                },
                {
                    "applied_decisions": 0,
                    "rules": {
                        "avoid_c": {
                            "rule_name": "avoid:third",
                            "trigger_firings": 1,
                            "candidate_choices": {},
                        }
                    },
                },
            ]
        )
    )

    assert summary.rule_family_trigger_counts["avoid"] == 4
    assert summary.rule_family_case_counts["avoid"] == 2
