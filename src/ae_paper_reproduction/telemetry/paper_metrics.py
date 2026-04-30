"""Paper-metrics macro materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ae_paper_reproduction.core.planning.notebook_extractor import (
    GenerationMode,
    PaperRole,
)
from ae_paper_reproduction.telemetry import ssnhl_experiment
from ae_paper_reproduction.telemetry.telemetry_types import (
    AnswerTelemetryRow,
    SubrunTelemetry,
)

_REQUIRED_MODES: tuple[GenerationMode, ...] = (
    "baseline",
    "reasoning",
    "trajectory",
)
_REQUIRED_SCOPES: tuple[str, ...] = ("ssnhl", "conductive")

_SCOPE_DISPLAY_BY_KEY: dict[str, str] = {
    "ssnhl": "SSNHL",
    "conductive": "Conductive",
}

# Canonical paper variant order.
#
# This list defines the stable metrics grid exposed to LaTeX.
# All variants must appear here even if data is currently unavailable.
# Missing data is represented by "--" values emitted by the generator.
#
# Any new experimental method must be added here explicitly.
#
# This list is a schema, not a convenience.
_VARIANT_ORDER: tuple[str, ...] = (
    "Baseline",
    "Reasoning",
    "ReplaceAfter",
    "GlobalValidation",
    "LocalValidation",
    "Trajectory",
)

_VARIANT_SPEC: dict[str, tuple[PaperRole, str]] = {
    "Baseline": ("primary", "baseline"),
    "Reasoning": ("primary", "reasoning"),
    "ReplaceAfter": ("ablation", "replace-after"),
    "GlobalValidation": ("ablation", "global-validation"),
    "LocalValidation": ("ablation", "local-validation"),
    "Trajectory": ("primary", "trajectory"),
}

NA_TEX = "--"


@dataclass(frozen=True, slots=True)
class TaskRunSummary:
    """Structured metrics for one (paper_role, variant, scope) tuple."""

    paper_role: PaperRole
    variant: str
    mode: GenerationMode
    scope: str
    n_cases: int
    n_correct: int
    accuracy: float
    seconds_per_case: float
    cases_per_second: float
    answer_rows: tuple[AnswerTelemetryRow, ...]
    telemetry: ssnhl_experiment.TelemetrySummary


@dataclass(frozen=True, slots=True)
class ModeAggregateSummary:
    """Structured metrics for one primary mode across both scopes."""

    mode: GenerationMode
    ssnhl: TaskRunSummary
    conductive: TaskRunSummary


type PaperRunKey = tuple[PaperRole, str, str]


def _macro(lines: list[str], name: str, value: str) -> None:
    new_line = f"\\newcommand{{\\{name}}}{{{value}}}"
    command = f"\\newcommand{{\\{name}}}"

    for line in lines:
        if line == new_line:
            return

        if line.startswith(command + "{"):
            msg = (
                f"Conflicting paper metric macro emitted: {name}. "
                f"Existing line: {line!r}; new line: {new_line!r}"
            )
            raise ValueError(msg)

    lines.append(new_line)


def _pct(value: float | None) -> str:
    if value is None:
        return NA_TEX
    return f"{value * 100.0:.1f}\\%"


def _pct_raw(value: float | None) -> str:
    if value is None:
        return NA_TEX
    return f"{value * 100.0:.1f}"


def _ratio_raw(value: float | None) -> str:
    if value is None:
        return NA_TEX
    return f"{value:.2f}"


def _ratio_x(value: float | None) -> str:
    if value is None:
        return NA_TEX
    return f"{value:.2f}$\\times$"


def _pp(value: float | None) -> str:
    if value is None:
        return NA_TEX
    return f"{value:+.1f} pp"


def _pp_raw(value: float | None) -> str:
    if value is None:
        return NA_TEX
    return f"{value:.1f}"


def _count(value: int | None) -> str:
    if value is None:
        return NA_TEX
    return str(value)


def _scope_name(case_type_filter: str | None) -> str:
    normalized = (case_type_filter or "").lower().replace("_", "-")
    if "ssnhl" in normalized:
        return "ssnhl"
    if "conductive" in normalized:
        return "conductive"
    return "other"


def _runtime_per_case(rows: Sequence[AnswerTelemetryRow]) -> float | None:
    total_seconds = 0.0
    for row in rows:
        runtime = row.edited_runtime_sec
        if runtime is None:
            runtime = row.baseline_runtime_sec
        if runtime is None:
            return None
        total_seconds += runtime
    if not rows:
        return None
    return total_seconds / len(rows)


def _build_run_summaries(
    subrun_telemetry: Sequence[SubrunTelemetry],
) -> dict[PaperRunKey, TaskRunSummary]:
    summaries: dict[PaperRunKey, TaskRunSummary] = {}
    for telemetry in subrun_telemetry:
        scope = _scope_name(telemetry.case_type_filter)
        if scope not in _REQUIRED_SCOPES:
            continue
        if telemetry.paper_role is None or telemetry.paper_variant is None:
            msg = (
                "Paper metrics require explicit paper_role and variant "
                "metadata; "
                f"missing metadata for ruleset {telemetry.ruleset_name!r}."
            )
            raise ValueError(msg)

        rows = tuple(telemetry.answer_rows)
        seconds_per_case = _runtime_per_case(rows)
        if seconds_per_case is None:
            msg = (
                "seconds_per_case is missing for run "
                f"{telemetry.ruleset_name!r} and scope {scope!r}."
            )
            raise ValueError(msg)
        n_cases = len(rows)
        n_correct = sum(1 for row in rows if bool(row.edited_ok))
        accuracy = (n_correct / n_cases) if n_cases else 0.0
        telemetry_items = [
            row.ae_telemetry for row in rows if row.ae_telemetry is not None
        ]

        key = (telemetry.paper_role, telemetry.paper_variant, scope)
        if key in summaries:
            msg = (
                "Duplicate paper metadata key detected for paper metrics: "
                f"{key!r}."
            )
            raise ValueError(msg)

        summaries[key] = TaskRunSummary(
            paper_role=telemetry.paper_role,
            variant=telemetry.paper_variant,
            mode=telemetry.mode,
            scope=scope,
            n_cases=n_cases,
            n_correct=n_correct,
            accuracy=accuracy,
            seconds_per_case=seconds_per_case,
            cases_per_second=(1.0 / seconds_per_case),
            answer_rows=rows,
            telemetry=ssnhl_experiment.summarize_telemetry(telemetry_items),
        )
    return summaries


def _canonical_mode_runs(
    run_summaries: Mapping[PaperRunKey, TaskRunSummary],
) -> dict[GenerationMode, ModeAggregateSummary]:
    by_mode: dict[GenerationMode, ModeAggregateSummary] = {}
    for mode in _REQUIRED_MODES:
        ssnhl = run_summaries.get(("primary", mode, "ssnhl"))
        conductive = run_summaries.get(("primary", mode, "conductive"))
        if ssnhl is None or conductive is None:
            missing_scope = "ssnhl" if ssnhl is None else "conductive"
            msg = (
                "Missing required mode/scope summary for paper metrics: "
                f"mode={mode!r}, scope={missing_scope!r}."
            )
            raise ValueError(msg)
        by_mode[mode] = ModeAggregateSummary(
            mode=mode,
            ssnhl=ssnhl,
            conductive=conductive,
        )
    return by_mode


def _delta_pp(current: float, reference: float) -> float:
    return (current - reference) * 100.0


def _emit_method_macros(
    *,
    lines: list[str],
    run_summaries: Mapping[PaperRunKey, TaskRunSummary],
    reasoning_by_scope: Mapping[str, TaskRunSummary],
) -> None:
    for scope_key in _REQUIRED_SCOPES:
        scope_display = _SCOPE_DISPLAY_BY_KEY[scope_key]
        reasoning = reasoning_by_scope.get(scope_key)
        reasoning_seconds = (
            reasoning.seconds_per_case
            if reasoning is not None and reasoning.seconds_per_case > 0.0
            else None
        )
        for variant in _VARIANT_ORDER:
            paper_role, paper_variant = _VARIANT_SPEC[variant]
            run = run_summaries.get((paper_role, paper_variant, scope_key))
            prefix = f"{scope_display}{variant}"

            acceptance = run.accuracy if run is not None else None
            n_correct = run.n_correct if run is not None else None
            seconds_per_case = (
                run.seconds_per_case
                if run is not None and run.seconds_per_case > 0.0
                else None
            )
            cases_per_second = (
                run.cases_per_second
                if run is not None and run.cases_per_second > 0.0
                else None
            )
            slowdown = None
            if seconds_per_case is not None and reasoning_seconds is not None:
                slowdown = seconds_per_case / reasoning_seconds

            _macro(lines, f"{prefix}AcceptedPct", _pct(acceptance))
            _macro(lines, f"{prefix}AcceptedRaw", _pct_raw(acceptance))
            _macro(lines, f"{prefix}AcceptedCount", _count(n_correct))
            _macro(
                lines,
                f"{prefix}SecondsPerCaseRaw",
                _ratio_raw(seconds_per_case),
            )
            _macro(
                lines,
                f"{prefix}CasesPerSecondRaw",
                _ratio_raw(cases_per_second),
            )
            _macro(lines, f"{prefix}SlowdownX", _ratio_x(slowdown))
            _macro(lines, f"{prefix}SlowdownRaw", _ratio_raw(slowdown))


def _emit_combined_balanced_accuracy_macros(
    *,
    lines: list[str],
    run_summaries: Mapping[PaperRunKey, TaskRunSummary],
) -> None:
    for variant in _VARIANT_ORDER:
        paper_role, paper_variant = _VARIANT_SPEC[variant]
        ssnhl = run_summaries.get((paper_role, paper_variant, "ssnhl"))
        conductive = run_summaries.get(
            (paper_role, paper_variant, "conductive")
        )
        balanced_accuracy = None
        if ssnhl is not None and conductive is not None:
            # Balanced accuracy is defined as the unweighted mean of scope
            # accuracies. Each diagnostic scope contributes equally regardless
            # of case count.
            #
            # This matches the paper definition of balanced accuracy and avoids
            # dominance of larger datasets.
            balanced_accuracy = (ssnhl.accuracy + conductive.accuracy) / 2.0
        prefix = f"Combined{variant}BalancedAccuracy"
        _macro(lines, f"{prefix}Pct", _pct(balanced_accuracy))
        _macro(lines, f"{prefix}Raw", _pct_raw(balanced_accuracy))


def _emit_pairwise_change_macros(
    *,
    lines: list[str],
    run_summaries: Mapping[PaperRunKey, TaskRunSummary],
) -> None:
    for scope_key in _REQUIRED_SCOPES:
        scope_display = _SCOPE_DISPLAY_BY_KEY[scope_key]
        reasoning = run_summaries.get(("primary", "reasoning", scope_key))
        reasoning_ok_by_case = None
        reasoning_accuracy = None
        if reasoning is not None:
            reasoning_ok_by_case = {
                row.id: bool(row.edited_ok) for row in reasoning.answer_rows
            }
            reasoning_accuracy = reasoning.accuracy

        for variant in _VARIANT_ORDER:
            paper_role, paper_variant = _VARIANT_SPEC[variant]
            candidate = run_summaries.get(
                (paper_role, paper_variant, scope_key)
            )
            prefix = f"{scope_display}{variant}"

            improved_count = None
            degraded_count = None
            improved_pp = None
            degraded_pp = None
            delta_pp = None
            additional_accepted = None

            if (
                candidate is not None
                and reasoning is not None
                and reasoning_ok_by_case is not None
                and reasoning_accuracy is not None
            ):
                improved = 0
                degraded = 0
                matched_cases = 0
                for row in candidate.answer_rows:
                    anchor_ok = reasoning_ok_by_case.get(row.id)
                    if anchor_ok is None:
                        continue
                    matched_cases += 1
                    candidate_ok = bool(row.edited_ok)
                    if candidate_ok and not anchor_ok:
                        improved += 1
                    elif anchor_ok and not candidate_ok:
                        degraded += 1

                improved_count = improved
                degraded_count = degraded
                if matched_cases > 0:
                    improved_pp = improved / matched_cases * 100.0
                    degraded_pp = degraded / matched_cases * 100.0
                else:
                    improved_pp = 0.0
                    degraded_pp = 0.0

                delta_pp = _delta_pp(candidate.accuracy, reasoning_accuracy)
                additional_accepted = candidate.n_correct - reasoning.n_correct

            _macro(lines, f"{prefix}ImprovedPP", _pp(improved_pp))
            _macro(lines, f"{prefix}ImprovedPPRaw", _pp_raw(improved_pp))
            _macro(lines, f"{prefix}ImprovedCount", _count(improved_count))
            _macro(lines, f"{prefix}DegradedPP", _pp(degraded_pp))
            _macro(lines, f"{prefix}DegradedPPRaw", _pp_raw(degraded_pp))
            _macro(lines, f"{prefix}DegradedCount", _count(degraded_count))
            _macro(lines, f"{prefix}DeltaPP", _pp(delta_pp))
            _macro(lines, f"{prefix}DeltaPPRaw", _pp_raw(delta_pp))
            _macro(
                lines,
                f"{prefix}AdditionalAcceptedCount",
                _count(additional_accepted),
            )


def _paired_reasoning_trajectory_n(
    run_summaries: Mapping[PaperRunKey, TaskRunSummary],
) -> int:
    total_pairs = 0
    for scope in _REQUIRED_SCOPES:
        reasoning = run_summaries.get(("primary", "reasoning", scope))
        trajectory = run_summaries.get(("primary", "trajectory", scope))
        if reasoning is None or trajectory is None:
            continue
        reasoning_ids = {row.id for row in reasoning.answer_rows}
        trajectory_ids = {row.id for row in trajectory.answer_rows}
        total_pairs += len(reasoning_ids & trajectory_ids)
    return total_pairs


def render_paper_metrics_tex(
    *,
    subrun_telemetry: Sequence[SubrunTelemetry],
) -> str:
    """Render manuscript macros into ``paper-metrics.tex``."""
    run_summaries = _build_run_summaries(subrun_telemetry)
    by_mode = _canonical_mode_runs(run_summaries)

    reasoning = by_mode["reasoning"]
    reasoning_by_scope = {
        "ssnhl": reasoning.ssnhl,
        "conductive": reasoning.conductive,
    }

    lines = [
        "% Auto-generated by ae_paper_reproduction.telemetry.paper_metrics",
        "% Do not edit manually.",
        "",
        "% =========================================",
        "% Dataset size",
        "% =========================================",
    ]
    _macro(lines, "PaperEvalN", str(reasoning.ssnhl.n_cases))
    _macro(
        lines,
        "PaperEvalPairN",
        str(_paired_reasoning_trajectory_n(run_summaries)),
    )

    lines.extend(
        [
            "",
            "% =========================================",
            "% Scope and method metrics",
            "% =========================================",
        ]
    )
    _emit_method_macros(
        lines=lines,
        run_summaries=run_summaries,
        reasoning_by_scope=reasoning_by_scope,
    )

    lines.extend(
        [
            "",
            "% =========================================",
            "% Combined balanced accuracy",
            "% =========================================",
        ]
    )
    _emit_combined_balanced_accuracy_macros(
        lines=lines,
        run_summaries=run_summaries,
    )

    lines.extend(
        [
            "",
            "% =========================================",
            "% Pairwise change metrics versus reasoning",
            "% =========================================",
        ]
    )
    _emit_pairwise_change_macros(lines=lines, run_summaries=run_summaries)

    lines.extend(
        [
            "",
            "% =========================================",
            "% Intervention/process metrics",
            "% =========================================",
        ]
    )
    trajectory_source = run_summaries.get(("primary", "trajectory", "ssnhl"))
    if trajectory_source is None:
        raise ValueError("Missing required trajectory primary run for ssnhl.")
    trajectory_ssnhl = trajectory_source.telemetry
    avoid_share = (
        trajectory_ssnhl.rule_family_counts["avoid"]
        / trajectory_ssnhl.edits_accepted
        if trajectory_ssnhl.edits_accepted
        else 0.0
    )
    avg_alternatives_tried = (
        trajectory_ssnhl.consumed_generated_probes_total
        / trajectory_ssnhl.probe_sets_generated_total
        if trajectory_ssnhl.probe_sets_generated_total
        else 0.0
    )
    _macro(
        lines,
        "RuntimeAvgInterventionsPerCase",
        f"{trajectory_ssnhl.avg_interventions_per_case:.1f}",
    )
    _macro(
        lines,
        "RuntimeAvoidInterventionsPerCase",
        f"{trajectory_ssnhl.avoid_interventions_per_case:.1f}",
    )
    _macro(
        lines, "RuntimeAvgAlternativesTried", f"{avg_alternatives_tried:.2f}"
    )
    _macro(
        lines,
        "RuntimeAlternativesForFiftyPctResolution",
        str(trajectory_ssnhl.probe_budget_for_50_coverage),
    )
    _macro(
        lines,
        "RuntimeAlternativesForEightyPctResolution",
        str(trajectory_ssnhl.probe_budget_for_80_coverage),
    )
    _macro(
        lines,
        "RuntimeRanOutOfAlternatives",
        f"{trajectory_ssnhl.not_enough_probes_share * 100.0:.1f}\\%",
    )
    _macro(
        lines,
        "RuntimeAvoidInterventionsShare",
        f"{avoid_share * 100.0:.1f}\\%",
    )

    _macro(
        lines, "AvgInterventionsPerCase", r"\RuntimeAvgInterventionsPerCase{}"
    )
    _macro(
        lines,
        "AvgAvoidInterventionsPerCase",
        r"\RuntimeAvoidInterventionsPerCase{}",
    )
    _macro(lines, "AvgAlternativesTried", r"\RuntimeAvgAlternativesTried{}")
    _macro(
        lines,
        "FiftyPercentResolutionBudget",
        r"\RuntimeAlternativesForFiftyPctResolution{}",
    )
    _macro(
        lines,
        "EightyPercentResolutionBudget",
        r"\RuntimeAlternativesForEightyPctResolution{}",
    )
    _macro(lines, "RanOutOfAlternativesRate", r"\RuntimeRanOutOfAlternatives{}")

    return "\n".join(lines) + "\n"


def write_paper_metrics_file(
    *,
    subrun_telemetry: Sequence[SubrunTelemetry],
    paper_generated_dir: Path,
) -> Path:
    """Write ``paper-metrics.tex`` as the only paper-facing TeX artifact."""
    paper_generated_dir.mkdir(parents=True, exist_ok=True)
    path = paper_generated_dir / "paper-metrics.tex"
    path.write_text(
        render_paper_metrics_tex(subrun_telemetry=subrun_telemetry),
        encoding="utf-8",
    )
    return path
