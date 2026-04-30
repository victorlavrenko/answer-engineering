"""Paper-specific telemetry summaries for the SSNHL experiment.

Purpose:
    Compute SSNHL experiment summaries and generate TeX fragments tailored to
    the paper-facing reporting workflow.

Architectural role:
    Paper-facing summary and TeX generation helper for the SSNHL experiment
    reporting pipeline.

Inputs:
    Experiment-level runtime telemetry, evaluation counts, and generated- table
    configuration for the SSNHL paper pipeline.

Outputs:
    Paper-specific summaries, runtime-telemetry rows, and generated TeX
    fragments for inclusion in the manuscript.

Ownership:
    Owned by
    `answer_engineering.telemetry.representation.paper.ssnhl_experiment` within
    the downstream telemetry representation boundary.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

from answer_engineering.telemetry import (
    AvoidProbeCacheExhausted,
    AvoidProbeCandidatePopped,
    AvoidProbeEpisodeStarted,
    AvoidProbeSetGenerated,
    PatchSkipped,
    ProposalRejected,
    ProposalsGenerated,
    RuntimeTelemetrySnapshot,
)


@dataclass(frozen=True, slots=True)
class TelemetrySummary:
    """Aggregate runtime telemetry metrics used in SSNHL paper reporting.

    Purpose:
        Hold precomputed aggregate metrics so later rendering code does not need
        to recompute them from raw telemetry.

    Architectural role:
        Paper-facing summary and TeX generation helper for the SSNHL experiment
        reporting pipeline.

    Inputs:
        Experiment-level runtime telemetry, evaluation counts, and generated-
        table configuration for the SSNHL paper pipeline.

    Outputs:
        Paper-specific summaries, runtime-telemetry rows, and generated TeX
        fragments for inclusion in the manuscript.

    Ownership:
        Owned by
        `answer_engineering.telemetry.representation.paper.ssnhl_experiment`
        within the downstream telemetry representation boundary.

    """

    n_cases: int
    runtime_seconds_total: float
    runtime_seconds_per_case: float
    trigger_firings_total: int
    avg_triggers_per_case: float
    avg_interventions_per_case: float
    probe_count_total: int
    probe_count_mean: float
    probe_depth_mean: float
    probe_depth_max: int
    rule_family_counts: dict[str, int]
    rule_family_trigger_counts: dict[str, int]
    rule_family_case_counts: dict[str, int]
    edits_proposed: int
    edits_accepted: int
    avoid_firings: int
    avoid_firings_with_generated_candidates: int
    avoid_interventions_per_case: float
    avoid_probe_episodes_per_case: float
    avoid_outcomes_total: int
    accepted_rank_1_share: float
    accepted_rank_gt_1_share: float
    resolved_within_top_3_share: float
    resolved_within_top_5_share: float
    avoid_required_gt_1_share: float
    avoid_required_gt_3_share: float
    avoid_required_gt_5_share: float
    probe_sets_generated_total: int
    probe_sets_generated_per_case: float
    requested_probes_total: int
    avg_valid_probes_per_generated_set: float
    avg_requested_probes_per_generated_set: float
    generated_probes_total: int
    consumed_generated_probes_total: int
    generated_probes_consumed_share: float
    probe_budget_for_50_coverage: int
    probe_budget_for_80_coverage: int
    not_enough_probes_count: int
    not_enough_probes_share: float
    avg_probes_per_avoid_firing: float
    fallbacks_used: int
    conflict_resolutions: int
    no_op_steps: int


def _safe_div(num: int, den: int) -> float:
    """Divide two counts while returning zero when the denominator is empty.

    Purpose:
        Compute a division for reporting metrics while preserving a stable
        fallback when the denominator is zero.

    Architectural role:
        Paper-facing summary and TeX generation helper for the SSNHL experiment
        reporting pipeline.

    Inputs:
        Experiment-level runtime telemetry, evaluation counts, and generated-
        table configuration for the SSNHL paper pipeline.

    Outputs:
        Paper-specific summaries, runtime-telemetry rows, and generated TeX
        fragments for inclusion in the manuscript.

    Ownership:
        Owned by
        `answer_engineering.telemetry.representation.paper.ssnhl_experiment`
        within the downstream telemetry representation boundary.

    Invariants:
        Returns a deterministic derived value for the supplied inputs.

    """
    return (num / den) if den else 0.0


def _coverage_budget(required_probes: list[int], coverage_share: float) -> int:
    """Return the probe budget needed to reach the requested empirical coverage.

    Purpose:
        Derive the reporting budget or denominator used for coverage-style
        telemetry summaries.

    Architectural role:
        Paper-facing summary and TeX generation helper for the SSNHL experiment
        reporting pipeline.

    Inputs:
        Experiment-level runtime telemetry, evaluation counts, and generated-
        table configuration for the SSNHL paper pipeline.

    Outputs:
        Paper-specific summaries, runtime-telemetry rows, and generated TeX
        fragments for inclusion in the manuscript.

    Ownership:
        Owned by
        `answer_engineering.telemetry.representation.paper.ssnhl_experiment`
        within the downstream telemetry representation boundary.

    """
    if not required_probes:
        return 0
    ordered = sorted(required_probes)
    rank = max(
        1,
        min(len(ordered), int((len(ordered) * coverage_share) + 0.9999999999)),
    )
    return ordered[rank - 1]


def _probe_depth_from_candidate_id(candidate_id: str) -> int | None:
    """Extract the probe depth encoded in a generated probe candidate id.

    Purpose:
        Recover a probe-depth signal from candidate identifiers so runtime
        telemetry can break down probing behavior.

    Architectural role:
        Paper-facing summary and TeX generation helper for the SSNHL experiment
        reporting pipeline.

    Inputs:
        Experiment-level runtime telemetry, evaluation counts, and generated-
        table configuration for the SSNHL paper pipeline.

    Outputs:
        Paper-specific summaries, runtime-telemetry rows, and generated TeX
        fragments for inclusion in the manuscript.

    Ownership:
        Owned by
        `answer_engineering.telemetry.representation.paper.ssnhl_experiment`
        within the downstream telemetry representation boundary.

    """
    if not candidate_id.startswith("probe_"):
        return None
    try:
        return int(candidate_id.split("_", maxsplit=1)[1])
    except (IndexError, ValueError):
        return None


def summarize_telemetry(
    telemetry_items: Sequence[RuntimeTelemetrySnapshot],
    *,
    fallback_depth: int = 11,
) -> TelemetrySummary:
    """Summarize many runtime telemetry snapshots into SSNHL paper metrics.

    Purpose:
        Aggregate intervention counts, fallback behavior, probe-depth behavior,
        and per-case runtime across the supplied snapshots into the
        ``TelemetrySummary`` consumed by the manuscript tables.

    Ownership:
        Owned by
        ``answer_engineering.telemetry.representation.paper.ssnhl_experiment``.

    """
    rule_family_counts: dict[str, int] = {
        "replace": 0,
        "after": 0,
        "avoid": 0,
        "force": 0,
    }
    rule_family_trigger_counts: dict[str, int] = {
        "replace": 0,
        "after": 0,
        "avoid": 0,
        "force": 0,
    }
    rule_family_case_counts: dict[str, int] = {
        "replace": 0,
        "after": 0,
        "avoid": 0,
        "force": 0,
    }
    edits_proposed = 0
    edits_accepted = 0
    fallbacks_used = 0
    conflict_resolutions = 0
    no_op_steps = 0
    probe_depths: list[int] = []
    probe_count_total = 0
    runtime_seconds_total = 0.0
    trigger_firings_total = 0
    avoid_firings = 0
    avoid_firings_with_generated_candidates = 0
    avoid_outcomes_total = 0
    accepted_rank_1 = 0
    accepted_rank_gt_1 = 0
    resolved_within_top_3 = 0
    resolved_within_top_5 = 0
    avoid_required_gt_1 = 0
    avoid_required_gt_3 = 0
    avoid_required_gt_5 = 0
    not_enough_probes_count = 0
    probe_sets_generated_total = 0
    requested_probes_total = 0
    valid_probes_generated_total = 0
    generated_probes_total = 0
    consumed_generated_probes_total = 0
    episode_generated_counts: dict[tuple[int, str], int] = {}
    episode_popped_counts: dict[tuple[int, str], int] = {}
    episode_exhausted: set[tuple[int, str]] = set()

    for case_index, telemetry in enumerate(telemetry_items):
        edits_accepted += telemetry.applied_decisions
        runtime_seconds_total += telemetry.runtime_sec or 0.0
        rule_family_by_id: dict[str, str] = {}
        family_seen_in_case: set[str] = set()
        for stats in telemetry.rules:
            edits_proposed += stats.evaluations
            rule_id = stats.rule_id
            family = stats.rule_name.split(":", maxsplit=1)[0]
            if rule_id:
                rule_family_by_id[rule_id] = family
            applied = stats.applied
            trigger_firings = stats.trigger_firings
            if family in rule_family_counts:
                rule_family_counts[family] += applied
                rule_family_trigger_counts[family] += trigger_firings
                trigger_firings_total += trigger_firings
                if trigger_firings > 0:
                    family_seen_in_case.add(family)
            if family == "avoid":
                avoid_firings += trigger_firings
            for candidate in stats.candidate_choices:
                chosen = candidate.chosen
                if chosen <= 0:
                    continue
                kind = candidate.kind
                if kind == "fallback":
                    fallbacks_used += chosen
                    probe_depths.extend([fallback_depth] * chosen)
                    if family == "avoid":
                        avoid_outcomes_total += chosen
                elif kind == "generated":
                    depth = _probe_depth_from_candidate_id(
                        candidate.candidate_id
                    )
                    if depth is not None:
                        probe_depths.extend([depth] * chosen)
                        if family == "avoid":
                            avoid_outcomes_total += chosen
                            if depth == 1:
                                accepted_rank_1 += chosen
                            elif depth > 1:
                                accepted_rank_gt_1 += chosen
                                avoid_required_gt_1 += chosen
                            if depth > 3:
                                avoid_required_gt_3 += chosen
                            if depth > 5:
                                avoid_required_gt_5 += chosen
                            if depth <= 3:
                                resolved_within_top_3 += chosen
                            if depth <= 5:
                                resolved_within_top_5 += chosen

        for family in family_seen_in_case:
            rule_family_case_counts[family] += 1

        for event in telemetry.events:
            if (
                isinstance(event, ProposalsGenerated)
                and rule_family_by_id.get(event.rule_id) == "avoid"
            ):
                probe_count_total += event.generated_count
                if event.generated_count > 0:
                    avoid_firings_with_generated_candidates += 1
            if isinstance(event, AvoidProbeSetGenerated):
                probe_sets_generated_total += 1
                requested_probes_total += event.probe_budget
                valid_probes_generated_total += event.generated_count
            if isinstance(event, AvoidProbeEpisodeStarted):
                cache_key = event.cache_key
                if cache_key:
                    episode_key = (case_index, cache_key)
                    episode_generated_counts[episode_key] = (
                        event.generated_count
                    )
                    generated_probes_total += event.generated_count
            if isinstance(event, AvoidProbeCandidatePopped):
                cache_key = event.cache_key
                if cache_key:
                    episode_key = (case_index, cache_key)
                    episode_popped_counts[episode_key] = (
                        episode_popped_counts.get(episode_key, 0) + 1
                    )
                    consumed_generated_probes_total += 1
            if isinstance(event, AvoidProbeCacheExhausted):
                not_enough_probes_count += 1
                if event.cache_key:
                    episode_exhausted.add((case_index, event.cache_key))
            if (
                isinstance(event, ProposalRejected)
                and event.reason == "conflict"
            ):
                conflict_resolutions += 1
            if isinstance(event, PatchSkipped):
                no_op_steps += 1

    probe_count_mean = _safe_div(probe_count_total, len(telemetry_items))
    runtime_seconds_per_case = (
        runtime_seconds_total / len(telemetry_items) if telemetry_items else 0.0
    )
    avg_triggers_per_case = (
        trigger_firings_total / len(telemetry_items) if telemetry_items else 0.0
    )
    avg_interventions_per_case = (
        edits_accepted / len(telemetry_items) if telemetry_items else 0.0
    )
    avg_probes_per_avoid_firing = (
        probe_count_total / avoid_firings_with_generated_candidates
        if avoid_firings_with_generated_candidates
        else 0.0
    )
    avoid_interventions_per_case = (
        rule_family_counts["avoid"] / len(telemetry_items)
        if telemetry_items
        else 0.0
    )
    avoid_probe_episodes_per_case = (
        avoid_firings_with_generated_candidates / len(telemetry_items)
        if telemetry_items
        else 0.0
    )
    episode_keys = list(episode_generated_counts.keys())
    required_probe_budgets: list[int] = []
    for episode_key in episode_keys:
        popped_count = episode_popped_counts.get(episode_key, 0)
        exhausted = episode_key in episode_exhausted
        required_probe_budgets.append(popped_count + (1 if exhausted else 0))
    probe_sets_generated_per_case = (
        probe_sets_generated_total / len(telemetry_items)
        if telemetry_items
        else 0.0
    )
    avg_valid_probes_per_generated_set = (
        valid_probes_generated_total / probe_sets_generated_total
        if probe_sets_generated_total
        else 0.0
    )
    avg_requested_probes_per_generated_set = (
        requested_probes_total / probe_sets_generated_total
        if probe_sets_generated_total
        else 0.0
    )
    generated_probes_consumed_share = _safe_div(
        consumed_generated_probes_total, generated_probes_total
    )
    not_enough_probes_denominator = (
        len(episode_keys)
        if episode_keys
        else avoid_firings_with_generated_candidates
    )
    return TelemetrySummary(
        n_cases=len(telemetry_items),
        runtime_seconds_total=runtime_seconds_total,
        runtime_seconds_per_case=runtime_seconds_per_case,
        trigger_firings_total=trigger_firings_total,
        avg_triggers_per_case=avg_triggers_per_case,
        avg_interventions_per_case=avg_interventions_per_case,
        probe_count_total=probe_count_total,
        probe_count_mean=probe_count_mean,
        probe_depth_mean=mean(probe_depths) if probe_depths else 0.0,
        probe_depth_max=max(probe_depths) if probe_depths else 0,
        rule_family_counts=rule_family_counts,
        rule_family_trigger_counts=rule_family_trigger_counts,
        rule_family_case_counts=rule_family_case_counts,
        edits_proposed=edits_proposed,
        edits_accepted=edits_accepted,
        avoid_firings=avoid_firings,
        avoid_firings_with_generated_candidates=(
            avoid_firings_with_generated_candidates
        ),
        avoid_interventions_per_case=avoid_interventions_per_case,
        avoid_probe_episodes_per_case=avoid_probe_episodes_per_case,
        avoid_outcomes_total=avoid_outcomes_total,
        accepted_rank_1_share=_safe_div(accepted_rank_1, avoid_firings),
        accepted_rank_gt_1_share=_safe_div(accepted_rank_gt_1, avoid_firings),
        resolved_within_top_3_share=_safe_div(
            resolved_within_top_3, avoid_firings
        ),
        resolved_within_top_5_share=_safe_div(
            resolved_within_top_5, avoid_firings
        ),
        avoid_required_gt_1_share=_safe_div(
            avoid_required_gt_1, avoid_outcomes_total
        ),
        avoid_required_gt_3_share=_safe_div(
            avoid_required_gt_3, avoid_outcomes_total
        ),
        avoid_required_gt_5_share=_safe_div(
            avoid_required_gt_5, avoid_outcomes_total
        ),
        probe_sets_generated_total=probe_sets_generated_total,
        probe_sets_generated_per_case=probe_sets_generated_per_case,
        requested_probes_total=requested_probes_total,
        avg_valid_probes_per_generated_set=avg_valid_probes_per_generated_set,
        avg_requested_probes_per_generated_set=(
            avg_requested_probes_per_generated_set
        ),
        generated_probes_total=generated_probes_total,
        consumed_generated_probes_total=consumed_generated_probes_total,
        generated_probes_consumed_share=generated_probes_consumed_share,
        probe_budget_for_50_coverage=_coverage_budget(
            required_probe_budgets, 0.50
        ),
        probe_budget_for_80_coverage=_coverage_budget(
            required_probe_budgets, 0.80
        ),
        not_enough_probes_count=not_enough_probes_count,
        not_enough_probes_share=_safe_div(
            not_enough_probes_count, not_enough_probes_denominator
        ),
        avg_probes_per_avoid_firing=avg_probes_per_avoid_firing,
        fallbacks_used=fallbacks_used,
        conflict_resolutions=conflict_resolutions,
        no_op_steps=no_op_steps,
    )
