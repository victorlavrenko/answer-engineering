"""Write telemetry representation artifacts to the local filesystem.

Purpose:
    Write telemetry representation outputs to their on-disk artifact layout for
    local inspection and later publication.

Architectural role:
    Artifact materialization helper inside the downstream telemetry
    representation boundary.

Inputs:
    Structured subrun or group report payloads and filesystem targets chosen by
    the telemetry representation layer.

Outputs:
    Written artifact files laid out for later inspection, bundling, or upload.

Ownership:
    Owned by
    `answer_engineering.telemetry.representation.artifacts.materializer` within
    the downstream telemetry representation boundary.

"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from ae_paper_reproduction.core.aggregation import rule_stats
from ae_paper_reproduction.core.aggregation.rule_stats import AggregatedRunStats
from ae_paper_reproduction.telemetry import paper_metrics, reporting
from ae_paper_reproduction.telemetry.reporting import (
    GroupSubrunReportRow,
)
from ae_paper_reproduction.telemetry.telemetry_types import (
    AnswerTelemetryRows,
    EvaluationArtifactFiles,
    GroupArtifactFiles,
    GroupComparisonRow,
    GroupRunContext,
    SubrunCaseTypeStatsRow,
    SubrunContext,
    SubrunTelemetry,
)

_DEFAULT_GENERATED_TEX_DIR: Final[Path] = Path("docs/paper/generated")


@dataclass(frozen=True, slots=True)
class ArtifactMaterializer:
    """Write rendered telemetry artifacts into local output directories.

    Purpose:
        Write generated reporting artifacts to their on-disk layout so later
        publication steps can reuse the same structure.

    Architectural role:
        Artifact materialization helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Structured subrun or group report payloads and filesystem targets chosen
        by the telemetry representation layer.

    Outputs:
        Written artifact files laid out for later inspection, bundling, or
        upload.

    Ownership:
        Owned by
        `answer_engineering.telemetry.representation.artifacts.materializer`
        within the downstream telemetry representation boundary.

    Lifecycle:
        Constructed by runtime or reporting orchestration and reused across the
        local operation scope it serves.

    """

    run_reports_dir: str
    generated_tex_dir: Path = _DEFAULT_GENERATED_TEX_DIR

    def write_subrun(
        self,
        *,
        ctx: SubrunContext,
        rules_markdown: str,
        run_stats: AggregatedRunStats,
        case_type_stats_rows: tuple[SubrunCaseTypeStatsRow, ...],
        answer_rows: AnswerTelemetryRows | None,
        gpu_name: str,
    ) -> EvaluationArtifactFiles:
        """Write one subrun's telemetry artifacts to the target directory.

        Purpose:
            Materialize the prepared reporting artifacts to disk using the file
            layout expected by downstream consumers.

        Architectural role:
            Artifact materialization helper inside the downstream telemetry
            representation boundary.

        Inputs:
            Structured subrun or group report payloads and filesystem targets
            chosen by the telemetry representation layer.

        Outputs:
            Written artifact files laid out for later inspection, bundling, or
            upload.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.artifacts.materializer`
            within the downstream telemetry representation boundary.

        """
        subrun_dir = (
            Path(self.run_reports_dir)
            / f"run-{ctx.group_run_id}"
            / f"subrun-{ctx.subrun_id}"
        )
        subrun_dir.mkdir(parents=True, exist_ok=True)

        rules_with_stats = rule_stats.annotate_rules_with_run_stats(
            rules_markdown, run_stats
        )
        report_md = reporting.render_subrun_report(
            ctx=ctx,
            gpu_name=gpu_name,
            case_type_stats_rows=case_type_stats_rows,
            rules_original_markdown=rules_markdown,
            rules_with_stats_markdown=rules_with_stats,
            runtime_summary=(
                reporting.RuntimeSummary(answer_rows) if answer_rows else None
            ),
        )

        report_path = subrun_dir / "run_report.md"
        original_path = subrun_dir / "rules_original.md"
        annotated_path = subrun_dir / "rules_with_stats.md"
        summary_path = subrun_dir / "run_summary.json"
        answers_path = subrun_dir / "answers.json"

        report_path.write_text(report_md, encoding="utf-8")
        original_path.write_text(rules_markdown, encoding="utf-8")
        annotated_path.write_text(rules_with_stats, encoding="utf-8")

        summary = {
            **asdict(ctx),
            "report_md_path": os.path.relpath(report_path, Path.cwd()),
            "rules_original_md_path": os.path.relpath(
                original_path, Path.cwd()
            ),
            "rules_with_stats_md_path": os.path.relpath(
                annotated_path, Path.cwd()
            ),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        serialized_answer_rows = (
            [row.to_serialized_row() for row in answer_rows]
            if answer_rows is not None
            else []
        )
        answers_path.write_text(
            json.dumps(serialized_answer_rows, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return EvaluationArtifactFiles(
            run_report_md=report_path,
            rules_original_md=original_path,
            rules_with_stats_md=annotated_path,
            run_summary_json=summary_path,
            answers_json=answers_path,
        )

    def write_group(
        self,
        *,
        ctx: GroupRunContext,
        subrun_telemetry: tuple[SubrunTelemetry, ...],
        comparison_rows: tuple[GroupComparisonRow, ...],
    ) -> GroupArtifactFiles:
        """Write one group's telemetry artifacts to the target directory.

        Purpose:
            Materialize the prepared reporting artifacts to disk using the file
            layout expected by downstream consumers.

        Architectural role:
            Artifact materialization helper inside the downstream telemetry
            representation boundary.

        Inputs:
            Structured subrun or group report payloads and filesystem targets
            chosen by the telemetry representation layer.

        Outputs:
            Written artifact files laid out for later inspection, bundling, or
            upload.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.artifacts.materializer`
            within the downstream telemetry representation boundary.

        """
        group_dir = Path(self.run_reports_dir) / f"run-{ctx.group_run_id}"
        group_dir.mkdir(parents=True, exist_ok=True)

        report_path = group_dir / "group_report.md"
        summary_path = group_dir / "group_summary.json"
        subruns_path = group_dir / "subruns.json"
        comparisons_path = group_dir / "comparisons.json"
        paper_metrics_path = group_dir / "paper_metrics.json"

        subrun_rows = [
            GroupSubrunReportRow(
                subrun_id=item.subrun_id,
                ruleset_name=item.ruleset_name,
                accuracy=item.accuracy,
                delta_accuracy=item.delta_accuracy,
                report_md_path=item.report_md_path,
            )
            for item in subrun_telemetry
        ]
        subrun_rows_json = [item.to_row() for item in subrun_telemetry]
        generated_paper_metrics = paper_metrics.write_paper_metrics_file(
            subrun_telemetry=subrun_telemetry,
            paper_generated_dir=self.generated_tex_dir,
        )
        report_md = reporting.render_group_report(
            ctx=ctx,
            subrun_rows=subrun_rows,
            comparison_rows=comparison_rows,
        )
        report_path.write_text(report_md, encoding="utf-8")
        summary_path.write_text(
            json.dumps(asdict(ctx), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        subruns_path.write_text(
            json.dumps(subrun_rows_json, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        comparisons_path.write_text(
            json.dumps(
                reporting.serialize_rows(comparison_rows),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        paper_metrics_path.write_text(
            json.dumps(
                {"paper_metrics_tex": str(generated_paper_metrics)},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        return GroupArtifactFiles(
            group_report_md=report_path,
            group_summary_json=summary_path,
            subruns_json=subruns_path,
            comparisons_json=comparisons_path,
            paper_metrics_json=paper_metrics_path,
            generated_tex_dir=self.generated_tex_dir,
        )
