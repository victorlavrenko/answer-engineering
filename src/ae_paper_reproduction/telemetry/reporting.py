"""Telemetry report rendering, artifact materialization, and publication.

Purpose:
    Assemble reproduction telemetry reports, artifact bundles, and publication
    operations from typed telemetry inputs.

Architectural role:
    Practical reporting assembly layer between typed telemetry models and output
    artifact consumers.

Architectural direction:
    Move toward clearer separation among typed telemetry representation, report
    rendering, artifact materialization, and publication workflows.

Why this matters:
    This module currently assembles table rendering, artifact path/layout
    policy, and publication preparation in one place. That concentration is
    functional, but heavier than a long-term clean boundary should be.

What better would look like:
    Report generation and publication flows depend on explicit seams rather than
    one large assembly surface.

How improvement can be recognized:
    - Clearer boundaries between rendering, artifact writing, and publication
    - Lower cross-cutting edits when adding new report consumers
    - Easier explanation of data flow from telemetry input to published
      artifacts

Open constraint:
    Reporting shape should remain responsive to real artifact consumers and
    product/reporting demand.

"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from ae_paper_reproduction.config.hf_defaults import HuggingFaceDefaults
from ae_paper_reproduction.core.aggregation import rule_stats
from ae_paper_reproduction.core.aggregation.comparison_results import (
    GroupComparisonRow,
    SubrunComparisonResult,
    SubrunEvaluationResult,
)
from ae_paper_reproduction.core.aggregation.rule_stats import AggregatedRunStats
from ae_paper_reproduction.core.evaluation.reports import (
    RunOutcomeTransitions,
)
from ae_paper_reproduction.core.evaluation.run_session import (
    RunSession,
    SubrunSession,
)
from ae_paper_reproduction.infra.remote.connectors import (
    ArtifactPublisher,
    HuggingFaceArtifactPublisher,
    HuggingFaceAuthResolver,
)
from ae_paper_reproduction.telemetry.telemetry_types import (
    AnswerTelemetryRow,
    AnswerTelemetryRows,
    ArtifactManifest,
    EvaluationArtifactFiles,
    GroupArtifactFiles,
    GroupRunContext,
    GroupTelemetry,
    RunCaseTypeStatsRow,
    RunContext,
    RunTelemetry,
    SubrunCaseTypeStatsRow,
    SubrunContext,
    SubrunTelemetry,
    TelemetryRows,
)


@dataclass(frozen=True, slots=True)
class GroupSubrunReportRow:
    """Immutable row model for one subrun entry in a group report.

    Purpose:
        Carry one already-shaped reporting row so downstream serializers and
        table renderers can operate on stable field names.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    Invariants:
        Instances are transport-oriented values and should remain stable after
        construction.

    """

    subrun_id: str
    ruleset_name: str
    accuracy: float
    delta_accuracy: float
    report_md_path: str


@dataclass(frozen=True, slots=True, init=False)
class RuntimeSummary:
    """Immutable row model for run-level runtime summary values used in reports.

    Purpose:
        Hold precomputed aggregate metrics so later rendering code does not need
        to recompute them from raw telemetry.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """

    baseline_seconds_per_case: float | None
    edited_seconds_per_case: float | None
    slowdown_vs_baseline: float | None

    def __init__(
        self,
        answer_rows: Iterable[AnswerTelemetryRow],
    ) -> None:
        """Summarize per-answer baseline and edited runtimes for reporting.

        Purpose:
            Build a compact runtime comparison view from answer-level timing
            data.

        Architectural role:
            Reporting constructor that separates telemetry summarization from
            table and markdown rendering.

        Inputs (architectural provenance):
            Receives answer rows or runtime measurements produced by evaluation
            runs.

        Outputs (downstream usage):
            Stores aggregate runtime fields consumed by report renderers and
            paper artifact generation.

        Invariants/constraints:
            Runtime calculations should be deterministic and should preserve the
            distinction between baseline and edited runs.

        """
        rows = list(answer_rows)
        baseline_per_case = _mean_runtime(
            row.baseline_runtime_sec for row in rows
        )
        edited_per_case = _mean_runtime(row.edited_runtime_sec for row in rows)
        slowdown_vs_baseline = None
        if (
            edited_per_case is not None
            and baseline_per_case is not None
            and baseline_per_case > 0
        ):
            slowdown_vs_baseline = edited_per_case / baseline_per_case
        object.__setattr__(self, "baseline_seconds_per_case", baseline_per_case)
        object.__setattr__(self, "edited_seconds_per_case", edited_per_case)
        object.__setattr__(self, "slowdown_vs_baseline", slowdown_vs_baseline)


@dataclass(frozen=True, slots=True)
class RunSummaryRow:
    """Immutable row model for one summary row in the run-level report.

    Purpose:
        Carry one already-shaped reporting row so downstream serializers and
        table renderers can operate on stable field names.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    Invariants:
        Instances are transport-oriented values and should remain stable after
        construction.

    """

    created_at_utc: str
    model_id: str
    case_type_filter: str | None
    accuracy: float
    delta_accuracy: float
    report_md_path: str


__all__ = [
    "ArtifactManifest",
    "EvaluationArtifactFiles",
    "GroupArtifactFiles",
    "GroupComparisonRow",
    "GroupRunContext",
    "GroupSubrunReportRow",
    "GroupTelemetry",
    "RunTelemetry",
    "RunSummaryRow",
    "RunContext",
    "SubrunComparisonResult",
    "SubrunContext",
    "SubrunTelemetry",
    "SubrunEvaluationResult",
    "append_rows_to_config",
    "build_run_id",
    "build_subrun_id",
    "ensure_dataset_repo",
    "git_commit_sha",
    "push_telemetry_bundle",
    "render_group_report",
    "serialize_rows",
    "render_run_report",
    "render_subrun_report",
    "require_hf_token",
    "update_reports_index",
    "upload_run_artifacts",
    "utc_now",
    "write_local_run_artifacts",
]


def utc_now() -> datetime:
    """Return the current UTC timestamp string used in telemetry metadata.

    Purpose:
        Carry out the specific telemetry repr transformation or helper step
        represented by this function while keeping the surrounding boundary code
        small and predictable.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    return datetime.now(UTC)


def git_commit_sha() -> str:
    """Return the current Git commit SHA used to tag telemetry artifacts.

    Purpose:
        Carry out the specific telemetry repr transformation or helper step
        represented by this function while keeping the surrounding boundary code
        small and predictable.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_run_id(
    *, now: datetime | None = None, run_tag: str | None = None
) -> str:
    """Build the stable run id used for telemetry artifacts and reports.

    Purpose:
        Assemble a higher-level reporting object from lower-level telemetry
        inputs so later stages can consume one stable structure.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    return RunSession(now=now or utc_now(), run_tag=run_tag).run_id


def serialize_rows(rows: Iterable[GroupComparisonRow]) -> TelemetryRows:
    """Serialize row objects into JSON text for artifact output.

    Purpose:
        Convert telemetry-domain objects into JSON-safe dictionaries and
        primitive values expected by storage, export, or upload surfaces.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    return [asdict(row) for row in rows]


def _mean_runtime(values: Iterable[float | None]) -> float | None:
    """Compute the mean runtime from a sequence of runtime values while.

    Purpose:
        Carry out the specific telemetry repr transformation or helper step
        represented by this function while keeping the surrounding boundary code
        small and predictable.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    samples = [value for value in values if value is not None]
    if not samples:
        return None
    return sum(samples) / len(samples)


def _format_runtime_sentence(
    seconds_per_case: float | None, slowdown: float | None
) -> str:
    """Format one human-readable runtime summary sentence for reports.

    Purpose:
        Carry out the specific telemetry repr transformation or helper step
        represented by this function while keeping the surrounding boundary code
        small and predictable.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    if seconds_per_case is None:
        return "N/A"
    if slowdown is None:
        return f"{seconds_per_case:.1f} sec/case"
    return f"{seconds_per_case:.1f} sec/case ({slowdown:.2f}× baseline)"


def _build_runtime_discussion_lines(
    runtime_summary: RuntimeSummary | None,
    *,
    subject: str,
) -> list[str]:
    """Build markdown bullet lines describing runtime in a report-friendly form.

    Purpose:
        Turn a ``RuntimeSummary`` into the small "Runtime discussion" section
        inserted into generated reports, including the edited-vs-baseline
        comparison when both measurements are present.

    """
    if runtime_summary is None:
        return list()
    baseline_per_case = runtime_summary.baseline_seconds_per_case
    edited_per_case = runtime_summary.edited_seconds_per_case
    slowdown = runtime_summary.slowdown_vs_baseline
    if baseline_per_case is None and edited_per_case is None:
        return list()

    lines = ["", "## Runtime discussion"]
    if edited_per_case is not None and baseline_per_case is not None:
        lines.extend(
            [
                "- "
                f"{subject} averaged "
                f"{_format_runtime_sentence(edited_per_case, slowdown)} "
                f"versus "
                f"{_format_runtime_sentence(baseline_per_case, None)} "
                f"for the baseline run.",
            ]
        )
        return lines
    if edited_per_case is not None:
        lines.append(
            f"- {subject} averaged "
            f"{_format_runtime_sentence(edited_per_case, None)}."
        )
        return lines
    lines.append(
        f"- {subject} averaged "
        f"{_format_runtime_sentence(baseline_per_case, None)}."
    )
    return lines


def render_run_report(
    *,
    ctx: RunContext,
    gpu_name: str,
    case_type_stats_rows: Sequence[RunCaseTypeStatsRow],
    outcome_transitions: RunOutcomeTransitions,
    rules_original_markdown: str,
    rules_with_stats_markdown: str,
    runtime_summary: RuntimeSummary | None = None,
) -> str:
    """Render the markdown report for a full run.

    Purpose:
        Produce the final textual representation consumed by reports, TeX
        fragments, or other presentation-facing outputs.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    scope = ctx.case_type_filter if ctx.case_type_filter is not None else "all"
    run_tag = ctx.run_tag or ""
    lines = [
        f"# Run {ctx.run_id}",
        "",
        "## Run metadata",
        f"- created_at_utc: {ctx.created_at_utc}",
        f"- code_commit_sha: {ctx.code_commit_sha}",
        f"- model_id: {ctx.model_id}",
        f"- dataset: {ctx.dataset_id}/{ctx.split}",
        f"- scope: {scope}",
        f"- gpu: {gpu_name}",
        f"- run_tag: {run_tag}",
        "",
        "## Accuracy",
        f"- baseline_accuracy: {ctx.baseline_accuracy:.4f}",
        f"- edited_accuracy: {ctx.edited_accuracy:.4f}",
        f"- delta_accuracy: {ctx.delta_accuracy:.4f}",
        "",
        "## Aggregate telemetry",
        f"- applied_decisions_total: {ctx.applied_decisions_total}",
        f"- decision_limit_reached: {str(ctx.decision_limit_reached).lower()}",
        f"- rules_triggered_count: {ctx.rules_triggered_count}",
        f"- rules_applied_count: {ctx.rules_applied_count}",
        "",
        "## Case-type summary",
        (
            "| case_type | n_cases | baseline_accuracy | edited_accuracy | "
            "delta_accuracy |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for row in case_type_stats_rows:
        lines.append(
            "| "
            f"{row.case_type} | {row.n_cases} | "
            f"{row.baseline_accuracy:.4f} | "
            f"{row.edited_accuracy:.4f} | "
            f"{row.delta_accuracy:.4f} |"
        )

    bc_to_ec = outcome_transitions.baseline_correct_to_edited_correct
    bc_to_ei = outcome_transitions.baseline_correct_to_edited_incorrect
    bi_to_ec = outcome_transitions.baseline_incorrect_to_edited_correct
    bi_to_bi = outcome_transitions.baseline_incorrect_to_edited_incorrect
    lines.extend(
        [
            "",
            "## Outcome transition summary",
            "| transition | count |",
            "|---|---:|",
            f"| baseline correct → edited correct | {bc_to_ec} |",
            f"| baseline correct → edited incorrect | {bc_to_ei} |",
            f"| baseline incorrect → edited correct | {bi_to_ec} |",
            f"| baseline incorrect → edited incorrect | {bi_to_bi} |",
        ]
    )

    lines.extend(
        _build_runtime_discussion_lines(
            runtime_summary, subject="Edited decoding"
        )
    )

    lines.extend(
        [
            "",
            "## Annotated rules",
            rules_with_stats_markdown,
            "",
            "## Links",
            "- [Original rules](rules_original.md)",
            "- [Annotated rules](rules_with_stats.md)",
            "- [Run summary JSON](run_summary.json)",
            "",
            "## Original rules",
            rules_original_markdown,
        ]
    )
    return "\n".join(lines)


def ensure_dataset_repo(
    *, publisher: ArtifactPublisher, dataset_id: str, private: bool, token: str
) -> None:
    """Ensure that the remote dataset repository used for telemetry artifacts.

    Purpose:
        Validate that the required external destination exists and create or
        configure it when necessary.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    publisher.ensure_dataset_repo(
        dataset_id=dataset_id, private=private, token=token
    )


def append_rows_to_config(
    *,
    publisher: ArtifactPublisher,
    dataset_id: str,
    config_name: str,
    run_id: str,
    rows: TelemetryRows,
    token: str,
) -> str:
    """Append telemetry row paths to the artifact configuration/index data.

    Purpose:
        Extend the persisted configuration or row collection with the newly
        generated telemetry records.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    path_in_repo = f"{config_name}/{run_id}.jsonl"
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    publisher.upload_file(
        path_or_fileobj=payload.encode("utf-8"),
        path_in_repo=path_in_repo,
        dataset_id=dataset_id,
        token=token,
    )
    return path_in_repo


def upload_run_artifacts(
    *,
    publisher: ArtifactPublisher,
    dataset_id: str,
    run_id: str,
    artifact_files: EvaluationArtifactFiles | GroupArtifactFiles,
    token: str,
) -> dict[str, str]:
    """Upload locally generated telemetry artifacts for one run to the remote.

    Purpose:
        Push the generated artifact files and metadata through the configured
        publication backend.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    uploaded: dict[str, str] = {}
    for name, local_path in artifact_files.upload_files().items():
        path_in_repo = f"artifacts/{run_id}/{name}"
        publisher.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            dataset_id=dataset_id,
            token=token,
        )
        uploaded[name] = path_in_repo
    return uploaded


def update_reports_index(
    *, reports_dir: Path, run_summaries: Sequence[RunSummaryRow]
) -> Path:
    """Update the reports index with the generated run artifacts.

    Purpose:
        Refresh the persisted index or manifest so newly generated telemetry
        artifacts become discoverable.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    index_path = reports_dir / "index.md"

    latest = sorted(
        run_summaries, key=lambda x: x.created_at_utc, reverse=True
    )[:10]
    best_delta = sorted(
        run_summaries, key=lambda x: x.delta_accuracy, reverse=True
    )[:10]

    def _rows(items: Sequence[RunSummaryRow]) -> list[str]:
        """Render and publish telemetry reports and bundles.

        Purpose:
            Normalize either dataclass rows or iterables of rows into the list
            form expected by downstream serialization helpers.

        Architectural role:
            Reporting and publication helper inside the downstream telemetry
            representation boundary.

        Inputs:
            Run/subrun/group telemetry values, artifact paths, and publication
            metadata prepared by evaluation code.

        Outputs:
            Human-readable reports, TeX fragments, artifact bundles, and
            publication operations for telemetry outputs.

        Ownership:
            Owned by `answer_engineering.telemetry.representation.telemetry`
            within the downstream telemetry representation boundary.

        """
        lines = [
            "| timestamp | model | scope | edited_accuracy | delta | report |",
            "|---|---|---|---:|---:|---|",
        ]
        for item in items:
            case_type = item.case_type_filter or "all"
            lines.append(
                "| "
                f"{item.created_at_utc} | {item.model_id} | "
                f"{case_type} | "
                f"{item.accuracy:.4f} | "
                f"{item.delta_accuracy:+.4f} | [open]({item.report_md_path}) |"
            )
        return lines

    content = "\n".join(
        [
            "# Evaluation run reports",
            "",
            "## Latest 10 runs",
            *_rows(latest),
            "",
            "## Best delta runs",
            *_rows(best_delta),
            "",
        ]
    )
    index_path.write_text(content, encoding="utf-8")
    return index_path


def write_local_run_artifacts(
    *,
    run_reports_dir: str,
    ctx: RunContext,
    rules_markdown: str,
    run_stats: AggregatedRunStats,
    case_type_stats_rows: Sequence[RunCaseTypeStatsRow],
    outcome_transitions: RunOutcomeTransitions,
    answer_rows: AnswerTelemetryRows | None = None,
    gpu_name: str,
) -> EvaluationArtifactFiles:
    """Write the local filesystem artifacts for one run.

    Purpose:
        Materialize the prepared reporting artifacts to disk using the file
        layout expected by downstream consumers.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    run_dir = Path(run_reports_dir) / ctx.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rules_with_stats = rule_stats.annotate_rules_with_run_stats(
        rules_markdown, run_stats
    )
    report_md = render_run_report(
        ctx=ctx,
        gpu_name=gpu_name,
        case_type_stats_rows=case_type_stats_rows,
        outcome_transitions=outcome_transitions,
        rules_original_markdown=rules_markdown,
        rules_with_stats_markdown=rules_with_stats,
        runtime_summary=(RuntimeSummary(answer_rows) if answer_rows else None),
    )

    report_path = run_dir / "run_report.md"
    original_path = run_dir / "rules_original.md"
    annotated_path = run_dir / "rules_with_stats.md"
    summary_path = run_dir / "run_summary.json"
    answers_path = run_dir / "answers.json"

    report_path.write_text(report_md, encoding="utf-8")
    original_path.write_text(rules_markdown, encoding="utf-8")
    annotated_path.write_text(rules_with_stats, encoding="utf-8")

    summary = {
        **asdict(ctx),
        "report_md_path": os.path.relpath(report_path, Path.cwd()),
        "rules_original_md_path": os.path.relpath(original_path, Path.cwd()),
        "rules_with_stats_md_path": os.path.relpath(annotated_path, Path.cwd()),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    answers_path.write_text(
        json.dumps(
            _serialize_answer_rows(answer_rows), indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )

    return EvaluationArtifactFiles(
        run_report_md=report_path,
        rules_original_md=original_path,
        rules_with_stats_md=annotated_path,
        run_summary_json=summary_path,
        answers_json=answers_path,
    )


def require_hf_token(env_name: str) -> str:
    """Return a Hugging Face token or raise a clear configuration error.

    Purpose:
        Centralize the required-token check for workflows that upload or
        download Hugging Face artifacts.

    Architectural role:
        Infrastructure guard at the reporting/remote-storage boundary.

    Inputs (architectural provenance):
        Reads the token from explicit arguments or environment-backed
        configuration used by telemetry publishing code.

    Outputs (downstream usage):
        Returns a non-empty token for connector calls that require
        authentication.

    Invariants/constraints:
        Failure should occur before remote operations start so callers do not
        create partial uploads or misleading telemetry artifacts.

    """
    return HuggingFaceAuthResolver(
        defaults=HuggingFaceDefaults()
    ).require_token(env_name)


def build_subrun_id(*, index: int, ruleset_name: str) -> str:
    """Build the stable subrun id used in rows, reports, and artifact paths.

    Purpose:
        Assemble a higher-level reporting object from lower-level telemetry
        inputs so later stages can consume one stable structure.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    return SubrunSession(index=index, ruleset_name=ruleset_name).subrun_id


def render_subrun_report(
    *,
    ctx: SubrunContext,
    gpu_name: str,
    case_type_stats_rows: Sequence[SubrunCaseTypeStatsRow],
    rules_original_markdown: str,
    rules_with_stats_markdown: str,
    runtime_summary: RuntimeSummary | None = None,
) -> str:
    """Render the markdown report for one subrun.

    Purpose:
        Produce the final textual representation consumed by reports, TeX
        fragments, or other presentation-facing outputs.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    scope = ctx.case_type_filter if ctx.case_type_filter is not None else "all"
    run_tag = ctx.run_tag or ""
    lines = [
        f"# Subrun {ctx.subrun_id}",
        "",
        "## Metadata",
        f"- group_run_id: {ctx.group_run_id}",
        f"- ruleset_name: {ctx.ruleset_name}",
        f"- created_at_utc: {ctx.created_at_utc}",
        f"- code_commit_sha: {ctx.code_commit_sha}",
        f"- model_id: {ctx.model_id}",
        f"- dataset: {ctx.dataset_id}/{ctx.split}",
        f"- scope: {scope}",
        f"- gpu: {gpu_name}",
        f"- run_tag: {run_tag}",
        "",
        "## Accuracy",
        f"- accuracy: {ctx.accuracy:.4f}",
    ]
    if (
        ctx.anchor_subrun_id is not None
        and ctx.anchor_accuracy is not None
        and ctx.delta_accuracy is not None
    ):
        lines.extend(
            [
                f"- anchor_subrun_id: {ctx.anchor_subrun_id}",
                f"- anchor_accuracy: {ctx.anchor_accuracy:.4f}",
                f"- delta_accuracy_vs_anchor: {ctx.delta_accuracy:.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Aggregate telemetry",
            f"- applied_decisions_total: {ctx.applied_decisions_total}",
            (
                "- decision_limit_reached: "
                f"{str(ctx.decision_limit_reached).lower()}"
            ),
            f"- rules_triggered_count: {ctx.rules_triggered_count}",
            f"- rules_applied_count: {ctx.rules_applied_count}",
            "",
            "## Case-type summary",
            "| case_type | n_cases | accuracy | delta_vs_anchor |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in case_type_stats_rows:
        lines.append(
            "| "
            f"{row.case_type} | {row.n_cases} | {row.accuracy:.4f} | "
            f"{row.delta_accuracy_vs_anchor:.4f} |"
        )

    lines.extend(
        _build_runtime_discussion_lines(runtime_summary, subject="This subrun")
    )

    lines.extend(
        [
            "",
            "## Annotated rules",
            rules_with_stats_markdown,
            "",
            "## Links",
            "- [Original rules](rules_original.md)",
            "- [Annotated rules](rules_with_stats.md)",
            "- [Run summary JSON](run_summary.json)",
            "",
            "## Original rules",
            rules_original_markdown,
        ]
    )
    return "\n".join(lines)


def render_group_report(
    *,
    ctx: GroupRunContext,
    subrun_rows: Sequence[GroupSubrunReportRow],
    comparison_rows: Sequence[GroupComparisonRow],
) -> str:
    """Render the markdown report for one comparison group.

    Purpose:
        Produce the final textual representation consumed by reports, TeX
        fragments, or other presentation-facing outputs.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    scope = ctx.case_type_filter if ctx.case_type_filter is not None else "all"
    run_tag = ctx.run_tag or ""
    lines = [
        f"# Group run {ctx.group_run_id}",
        "",
        "## Metadata",
        f"- created_at_utc: {ctx.created_at_utc}",
        f"- code_commit_sha: {ctx.code_commit_sha}",
        f"- model_id: {ctx.model_id}",
        f"- dataset: {ctx.dataset_id}/{ctx.split}",
        f"- scope: {scope}",
        f"- run_tag: {run_tag}",
        "",
        "## Subruns",
        "| subrun_id | ruleset_name | accuracy | delta_vs_anchor | report |",
        "|---|---|---:|---:|---|",
    ]
    for row in subrun_rows:
        lines.append(
            "| "
            f"{row.subrun_id} | {row.ruleset_name} | {row.accuracy:.4f} | "
            f"{row.delta_accuracy:+.4f} | [open]({row.report_md_path}) |"
        )

    lines.extend(
        [
            "",
            "## Comparisons",
            (
                "| anchor_subrun_id | candidate_subrun_id | anchor_ruleset | "
                "candidate_ruleset | delta_accuracy | improved | degraded | "
                "unchanged_correct | unchanged_incorrect |"
            ),
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        lines.append(
            "| "
            f"{row.anchor_subrun_id} | {row.candidate_subrun_id} | "
            f"{row.anchor_ruleset_name} | {row.candidate_ruleset_name} | "
            f"{row.delta_accuracy:+.4f} | "
            f"{row.improved} | {row.degraded} | "
            f"{row.unchanged_correct} | {row.unchanged_incorrect} |"
        )
    return "\n".join(lines)


def _build_rows_commit_operation(
    *,
    path_in_repo: str,
    rows: TelemetryRows,
) -> CommitOperationAdd:
    """Build a commit operation for JSON-serialized telemetry rows.

    Purpose:
        Package row serialization and remote path metadata into one upload
        operation for repository-backed artifact publishing.

    Architectural role:
        Infrastructure helper at the reporting-to-remote-storage boundary.

    Inputs (architectural provenance):
        Receives telemetry rows and destination metadata prepared by the
        reporting pipeline.

    Outputs (downstream usage):
        Returns a commit operation consumed by the remote connector batching
        layer.

    Invariants/constraints:
        The helper should prepare upload data only. It should not perform the
        remote commit itself.

    """
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    return CommitOperationAdd(
        path_in_repo=path_in_repo,
        path_or_fileobj=payload.encode("utf-8"),
    )


def _build_answer_rows_commit_operation(
    *,
    path_in_repo: str,
    rows: AnswerTelemetryRows,
) -> CommitOperationAdd:
    """Build a commit operation for serialized answer-level rows.

    Purpose:
        Package answer-row telemetry as a remote upload operation with stable
        path and content metadata.

    Architectural role:
        Reporting infrastructure helper used before batching remote artifact
        commits.

    Inputs (architectural provenance):
        Receives answer-level rows generated by evaluation reporting.

    Outputs (downstream usage):
        Returns a commit operation consumed by upload orchestration.

    Invariants/constraints:
        Serialization should be deterministic and side-effect-free; remote
        writes belong to the connector layer.

    """
    serialized_rows = _serialize_answer_rows(rows)
    payload = (
        "\n".join(json.dumps(row, sort_keys=True) for row in serialized_rows)
        + "\n"
    )
    return CommitOperationAdd(
        path_in_repo=path_in_repo,
        path_or_fileobj=payload.encode("utf-8"),
    )


def _serialize_answer_rows(
    answer_rows: AnswerTelemetryRows | None,
) -> TelemetryRows:
    """Serialize answer-level telemetry rows for artifact publication.

    Purpose:
        Carry out the specific telemetry repr transformation or helper step
        represented by this function while keeping the surrounding boundary code
        small and predictable.

    Architectural role:
        Reporting and publication helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Run/subrun/group telemetry values, artifact paths, and publication
        metadata prepared by evaluation code.

    Outputs:
        Human-readable reports, TeX fragments, artifact bundles, and publication
        operations for telemetry outputs.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry` within
        the downstream telemetry representation boundary.

    """
    return [row.to_serialized_row() for row in answer_rows or []]


def _create_telemetry_commit(
    *,
    publisher: ArtifactPublisher,
    dataset_id: str,
    token: str,
    group_run_id: str,
    operations: list[CommitOperationAdd],
) -> None:
    """Publish one prepared telemetry commit to the remote artifact store.

    Purpose:
        Delegate the final commit creation to the configured artifact publisher
        using the dataset id, group-run-specific commit message, and the already
        materialized file operations.

    """
    publisher.commit(
        dataset_id=dataset_id,
        operations=operations,
        message=f"Add telemetry bundle for {group_run_id}",
        token=token,
    )


def push_telemetry_bundle(
    *,
    dataset_id: str,
    private: bool,
    token: str,
    prepared_subruns: Iterable[tuple[SubrunTelemetry, EvaluationArtifactFiles]],
    payload: GroupTelemetry,
    artifact_files: GroupArtifactFiles,
    publisher: ArtifactPublisher | None = None,
) -> dict[str, dict[str, str]]:
    """Publish a grouped telemetry bundle and its generated artifacts.

    Purpose:
        Ensure the target dataset exists, assemble upload operations for the
        group-level and subrun-level reports, rows, and paper artifacts, and
        publish them through the configured artifact backend.

    Outputs:
        A manifest-like mapping of the published artifact paths grouped by
        reporting scope.

    Ownership:
        Owned by ``answer_engineering.telemetry.representation.telemetry``.

    """
    resolved_publisher = publisher or HuggingFaceArtifactPublisher(
        api=HfApi(),
        defaults=HuggingFaceDefaults(),
    )
    ensure_dataset_repo(
        publisher=resolved_publisher,
        dataset_id=dataset_id,
        private=private,
        token=token,
    )

    operations: list[CommitOperationAdd] = []
    subrun_uploads: dict[str, str] = {}
    group_run_id: str | None = None

    for built_rows, subrun_artifact_files in prepared_subruns:
        current_group_run_id = built_rows.group_run_id
        if not current_group_run_id:
            msg = "prepared_subruns entries must include non-empty group_run_id"
            raise ValueError(msg)
        subrun_id = built_rows.subrun_id
        if not subrun_id:
            msg = "prepared_subruns entries must include non-empty subrun_id"
            raise ValueError(msg)
        if group_run_id is None:
            group_run_id = current_group_run_id
        elif group_run_id != current_group_run_id:
            msg = "prepared_subruns must all share the same group_run_id"
            raise ValueError(msg)

        prefix = f"run-{current_group_run_id}/subrun-{subrun_id}"
        subrun_row = built_rows.to_row()
        subrun_uploads[subrun_id] = f"{prefix}/subrun.jsonl"
        operations.extend(
            [
                _build_rows_commit_operation(
                    path_in_repo=f"{prefix}/subrun.jsonl",
                    rows=[subrun_row],
                ),
                _build_rows_commit_operation(
                    path_in_repo=f"{prefix}/data/runs.jsonl",
                    rows=[subrun_row],
                ),
                _build_rows_commit_operation(
                    path_in_repo=f"{prefix}/data/rule_stats.jsonl",
                    rows=[row.to_row() for row in built_rows.rule_rows],
                ),
                _build_rows_commit_operation(
                    path_in_repo=f"{prefix}/data/case_type_stats.jsonl",
                    rows=[row.to_row() for row in built_rows.case_rows],
                ),
                _build_answer_rows_commit_operation(
                    path_in_repo=f"{prefix}/data/answers.jsonl",
                    rows=built_rows.answer_rows,
                ),
            ]
        )
        for name, local_path in subrun_artifact_files.upload_files().items():
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"{prefix}/artifacts/{name}",
                    path_or_fileobj=local_path,
                )
            )

    group_run_id = group_run_id or payload.group_run_id
    prefix = f"run-{group_run_id}"
    group_uploads = {
        "group": f"{prefix}/group.jsonl",
        "comparisons": f"{prefix}/data/comparisons.jsonl",
    }
    operations.extend(
        [
            _build_rows_commit_operation(
                path_in_repo=f"{prefix}/group.jsonl",
                rows=[payload.group_row.to_row()],
            ),
            _build_rows_commit_operation(
                path_in_repo=f"{prefix}/data/comparisons.jsonl",
                rows=serialize_rows(payload.comparison_rows),
            ),
        ]
    )
    for name, local_path in artifact_files.upload_files().items():
        group_uploads[f"artifact:{name}"] = f"{prefix}/artifacts/{name}"
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"{prefix}/artifacts/{name}",
                path_or_fileobj=local_path,
            )
        )
    for name, local_path in artifact_files.generated_files().items():
        group_uploads[f"generated:{name}"] = f"{prefix}/generated/{name}"
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"{prefix}/generated/{name}",
                path_or_fileobj=local_path,
            )
        )

    _create_telemetry_commit(
        publisher=resolved_publisher,
        dataset_id=dataset_id,
        token=token,
        group_run_id=group_run_id,
        operations=operations,
    )
    return {"subruns": subrun_uploads, "group": group_uploads}
