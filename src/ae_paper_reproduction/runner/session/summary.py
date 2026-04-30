"""Assemble reporting payloads and artifacts for completed reproduction runs.

Purpose:
    Take completed subrun results, derive anchor-versus-candidate comparisons,
    materialize telemetry artifacts, and prepare an optional push-to-hub payload
    for the whole run.

Architectural role:
    Summary-building module at the end of the reproduction session pipeline.

Inputs (architectural provenance):
    Consumes completed subrun results, merged telemetry, artifact materializers,
    and hub-auth helpers.

Outputs (downstream usage):
    Group-level telemetry payloads, artifact files, and hub-publish helpers
    consumed by notebooks and downstream reporting workflows.

Invariants/constraints:
    Summary construction should happen after execution is complete and must
    preserve the run and subrun identities used to join artifacts, telemetry
    rows, and comparisons.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from ae_paper_reproduction.config.hf_defaults import HuggingFaceDefaults
from ae_paper_reproduction.core.aggregation.rule_stats import (
    AggregatedRunStats,
)
from ae_paper_reproduction.core.planning.subruns import SubrunResult
from ae_paper_reproduction.infra.datasets.datasets import Dataset
from ae_paper_reproduction.infra.remote.connectors import (
    HuggingFaceAuthResolver,
)
from ae_paper_reproduction.telemetry import reporting
from ae_paper_reproduction.telemetry.artifacts import (
    ArtifactMaterializer,
)
from ae_paper_reproduction.telemetry.reporting import (
    EvaluationArtifactFiles,
    GroupArtifactFiles,
    GroupComparisonRow,
    GroupRunContext,
    GroupTelemetry,
    SubrunComparisonResult,
    SubrunContext,
    SubrunTelemetry,
)
from ae_paper_reproduction.telemetry.telemetry_types import (
    ArtifactManifest,
    GroupSummaryRow,
    SubrunCaseTypeStatsRow,
)
from answer_engineering import GenerationPolicy, GenerationRuntime


@dataclass(frozen=True, slots=True)
class _PreparedSubrunTelemetry:
    """Pair built subrun telemetry rows with the artifact files generated for.

    Purpose:
        Keep the two outputs of subrun-level summary preparation together so
        group-level summary construction can reuse both the rows and the written
        artifacts consistently.

    Architectural role:
        Private bridge record inside final summary assembly.

    Inputs (architectural provenance):
        Constructed while each completed subrun is converted into telemetry rows
        and artifacts.

    Outputs (downstream usage):
        A subrun-level bundle consumed by later group summary and artifact
        generation steps.

    Invariants/constraints:
        Both fields must describe the same prepared subrun.

    """

    built_rows: SubrunTelemetry
    artifact_files: EvaluationArtifactFiles


@dataclass(frozen=True, slots=True, init=False)
class Summary:
    """Aggregate reproduction results and generated artifacts.

    Summarize completed subrun results, compute comparisons, prepare artifact
    files, and optionally push generated reports to the Hugging Face Hub. This
    is the main post-run object used by the reproduction notebook.

    .. note::
        Treat ``Summary`` as the aggregation boundary after generation has
        finished. Do not use it to select tasks or run the model.

    Examples:
        ```python
        summary = Summary(subresults)
        print(summary.artifact_files.group_report_md)
        uploaded = summary.push_to_hub(DATASET_ID)
        ```

    Attributes:
        subresults: Completed subrun results included in the summary.
        dataset: Dataset metadata inferred from subrun results when available.
        model: Model/runtime metadata inferred from subrun results when
            available.
        group_context: Context shared by generated group reports.
        comparisons: Pairwise and aggregate comparisons between subruns.
        artifact_files: Generated artifact file payloads.
        payload: Serialized summary payload for export.
        prepared_subruns: Prepared subrun telemetry/report records.

    Methods:
        :meth:`~ae_paper_reproduction.Summary.push_to_hub`: Upload generated
        artifacts to a Hugging Face dataset repository.

    Runtime behavior:
        Construction aggregates already-computed results and prepares report
        payloads. Upload happens only when ``push_to_hub`` is called.

    Architectural role:
        Reporting and artifact boundary for paper reproduction.

    Consumes:
        Sequence of :class:`~ae_paper_reproduction.SubrunResult` objects.

    Produces:
        Markdown/JSON/TeX-oriented artifacts, comparison summaries, and optional
        Hub uploads.

    Invariants:
        Generated paper/report artifacts should be derived from structured
        results, not from console output or private runtime sessions.

    Developer Notes:
        This class is moving toward a single-source-of-truth reporting model for
        generated paper metrics. Keep aggregation explicit and avoid duplicated
        manual numbers where possible.

    Todo:
        Make paper metrics generation the canonical source for LaTeX macros and
        remove remaining manual synchronization points as the reporting
        architecture stabilizes.

    See Also:
        :class:`~ae_paper_reproduction.SubrunResult`
        :class:`~ae_paper_reproduction.RulesetEvaluationResult`

    """

    dataset: Dataset
    model: GenerationRuntime
    group_context: GroupRunContext
    subresults: tuple[SubrunResult, ...]
    comparisons: tuple[SubrunComparisonResult, ...]
    artifact_files: GroupArtifactFiles
    payload: GroupTelemetry
    prepared_subruns: tuple[_PreparedSubrunTelemetry, ...] = field(repr=False)

    def __init__(self, subresults: Sequence[SubrunResult]) -> None:
        """Build a final summary from completed subrun results.

        Examples:
            ```python
            subresults = [
                SubrunResult(subrun, rows)
                for subrun, rows in completed
            ]
            summary = Summary(subresults)
            ```

        Args:
            subresults: Ordered SubrunResult objects, typically one baseline and
                one or more rule-enabled runs.

        What happens internally:
            The constructor stores the ordered subrun reports, prepares grouped
            artifact metadata, and makes the final reporting state available to
            the notebook. It assumes generation and per-task evaluation have
            already finished.

        What users should check:
            The order of subresults is the order readers will usually see in
            reports. Keep baseline and rule-enabled runs clearly labeled so
            paper metrics and private analyses are easy to interpret.

        Developer notes:
            Keep construction deterministic. Expensive publication or remote
            writes should remain in explicit methods such as push_to_hub().

        Todo:
            Make the constructor feed a single paper-metrics emitter once legacy
            table generation is removed.

        """
        if not subresults:
            raise ValueError("subresults must not be empty")

        materialized_subresults = tuple(subresults)
        first = materialized_subresults[0]
        run_created_at = reporting.utc_now()
        group_run_id = reporting.build_run_id(now=run_created_at)
        created_at_utc = run_created_at.isoformat()
        code_commit_sha = reporting.git_commit_sha()
        default_max_new_tokens = GenerationPolicy().max_new_tokens
        gpu_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        )
        group_context = GroupRunContext(
            group_run_id=group_run_id,
            created_at_utc=created_at_utc,
            code_commit_sha=code_commit_sha,
            subresults=materialized_subresults,
            default_max_new_tokens=default_max_new_tokens,
        )

        anchor_subruns_by_scope: dict[str, SubrunResult] = {}
        comparison_outputs: list[SubrunComparisonResult] = []
        for subresult in materialized_subresults:
            anchor = anchor_subruns_by_scope.setdefault(
                subresult.subrun.scope_label, subresult
            )
            if anchor is not subresult:
                comparison_outputs.append(
                    SubrunComparisonResult(anchor, subresult)
                )

        prepared_subruns: list[_PreparedSubrunTelemetry] = []
        artifact_materializer = ArtifactMaterializer(
            run_reports_dir="reports/runs"
        )

        for subresult in materialized_subresults:
            merged_telemetry = AggregatedRunStats(subresult.telemetry_items())
            anchor_subrun = anchor_subruns_by_scope[
                subresult.subrun.scope_label
            ]
            anchor_report = anchor_subrun.report

            case_type_stats_rows: list[SubrunCaseTypeStatsRow] = []
            anchor_by_case = anchor_report.by_case
            for case_type, case_counts in subresult.report.by_case.items():
                accuracy = (
                    (case_counts.correct / case_counts.total)
                    if case_counts.total
                    else 0.0
                )
                if subresult.subrun.subrun_id == anchor_subrun.subrun.subrun_id:
                    delta_accuracy_vs_anchor = 0.0
                else:
                    anchor_counts = anchor_by_case.get(case_type, case_counts)
                    anchor_case_accuracy = (
                        (anchor_counts.correct / anchor_counts.total)
                        if anchor_counts.total
                        else 0.0
                    )
                    delta_accuracy_vs_anchor = accuracy - anchor_case_accuracy
                case_type_stats_rows.append(
                    SubrunCaseTypeStatsRow(
                        case_type=case_type,
                        n_cases=case_counts.total,
                        accuracy=accuracy,
                        delta_accuracy_vs_anchor=delta_accuracy_vs_anchor,
                    )
                )

            subrun_ctx = SubrunContext(
                group_context=group_context,
                subresult=subresult,
                anchor_subrun=anchor_subrun,
                run_stats=merged_telemetry,
                default_max_new_tokens=default_max_new_tokens,
                created_at_utc=created_at_utc,
                code_commit_sha=code_commit_sha,
            )
            built_rows = SubrunTelemetry(
                ctx=subrun_ctx,
                run_stats=merged_telemetry,
                case_type_stats_rows=case_type_stats_rows,
                artifact_manifest=ArtifactManifest(
                    report_md_path="",
                    rules_original_md_path="",
                    rules_with_stats_md_path="",
                ),
                eval_results=subresult.results,
            )
            artifacts = artifact_materializer.write_subrun(
                ctx=subrun_ctx,
                rules_markdown=subresult.subrun.rules_markdown,
                run_stats=merged_telemetry,
                case_type_stats_rows=tuple(case_type_stats_rows),
                answer_rows=built_rows.answer_rows,
                gpu_name=gpu_name,
            )
            built_rows = SubrunTelemetry(
                ctx=subrun_ctx,
                run_stats=merged_telemetry,
                case_type_stats_rows=case_type_stats_rows,
                artifact_manifest=ArtifactManifest(artifacts),
                eval_results=subresult.results,
            )
            prepared_subruns.append(
                _PreparedSubrunTelemetry(
                    built_rows=built_rows, artifact_files=artifacts
                )
            )

        comparison_rows: tuple[GroupComparisonRow, ...] = tuple(
            GroupComparisonRow(
                group_context=group_context,
                comparison_output=comparison,
            )
            for comparison in comparison_outputs
        )
        group_artifacts = artifact_materializer.write_group(
            ctx=group_context,
            subrun_telemetry=tuple(
                prepared.built_rows for prepared in prepared_subruns
            ),
            comparison_rows=comparison_rows,
        )
        payload = GroupTelemetry(
            group_run_id=group_run_id,
            group_row=GroupSummaryRow(
                group_context,
                group_report_md_path=str(group_artifacts.group_report_md),
            ),
            comparison_rows=comparison_rows,
        )

        object.__setattr__(self, "dataset", first.subrun.dataset)
        object.__setattr__(self, "model", first.subrun.model)
        object.__setattr__(self, "group_context", group_context)
        object.__setattr__(self, "subresults", materialized_subresults)
        object.__setattr__(self, "comparisons", tuple(comparison_outputs))
        object.__setattr__(self, "artifact_files", group_artifacts)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "prepared_subruns", tuple(prepared_subruns))

    def push_to_hub(self, dataset_id: str) -> dict[str, dict[str, str]]:
        """Publish the prepared reproduction artifact bundle to HuggingFace Hub.

        Call this after building a ``Summary`` when you want to store generated
        reports, telemetry payloads, and artifact files in a Hub dataset
        repository. The method uses the already-prepared summary state; it does
        not rerun generation, reselect tasks, or rebuild subrun results.

        Example:
            ```python
            summary = Summary(subresults)
            uploaded = summary.push_to_hub(
                "your-org/answer-engineering-runs"
            )
            ```

        Args:
            dataset_id: Hugging Face dataset repository id to receive the
                artifacts, for example ``"your-org/answer-engineering-runs"``.
                The repository must be writable by the active token or by the
                user after the interactive login retry flow completes.

        Returns:
            Mapping describing the uploaded artifact paths and remote locations.
            The exact nested keys come from the reporting layer and are intended
            for notebook display, audit logs, or follow-up scripts.

        Notes:
            Publishing is optional. Users can inspect ``Summary.artifact_files``
            and the generated local files without pushing anything to the Hub.
            For private or unpublished reproduction work, keep artifacts local
            until the generated answers and telemetry are ready to share.

        Telemetry context:
            The pushed bundle is useful for experiments beyond the paper tables.
            Users can compare intervention frequency, runtime, rule-trigger
            patterns, per-case correctness transitions, and other telemetry
            extracted from the stored ``RulesetEvaluationResult`` rows.

        Developer notes:
            This method is an I/O boundary on top of a finalized summary. Keep
            it as a publisher of prepared artifacts, not a hidden builder of new
            reports. Hub authentication retries belong here because they are
            user-facing publishing concerns, not core reporting semantics.

        Todo:
            If the reporting model consolidates around a single generated
            ``paper-metrics.tex`` file, ensure this publisher uploads that file
            as the canonical metrics artifact and does not revive legacy
            generated table outputs.

        """
        prepared_subruns = [
            (prepared.built_rows, prepared.artifact_files)
            for prepared in self.prepared_subruns
        ]
        return _retry_hf_push_after_login(
            lambda token: reporting.push_telemetry_bundle(
                dataset_id=dataset_id,
                private=False,
                token=token,
                prepared_subruns=prepared_subruns,
                payload=self.payload,
                artifact_files=self.artifact_files,
            ),
            token_env_name=HuggingFaceDefaults().token_env_name,
        )


def _looks_like_hf_auth_error(exc: Exception) -> bool:
    """Detect whether an exception looks like a Hugging Face authentication.

    Purpose:
        Classify whether an exception looks like a Hugging Face authentication
        failure.

    Architectural role:
        Private publishing/authentication helper inside summary publication.

    Inputs (architectural provenance):
        Consumes exceptions, environment names, or push callbacks used during
        hub publication.

    Outputs (downstream usage):
        Authentication decisions or side effects consumed by
        `Summary.push_to_hub`.

    Invariants/constraints:
        These helpers should remain tightly scoped to publication behavior.

    """
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "401",
            "403",
            "forbidden",
            "hf_token",
            "hugging face",
            "huggingface",
            "token",
            "unauthorized",
            "whoami",
        )
    )


def require_hf_token(env_name: str) -> str:
    """Resolve and require the Hugging Face token needed for publishing.

    Purpose:
        Fetch the Hugging Face token needed for publishing.

    Architectural role:
        Private publishing/authentication helper inside summary publication.

    Inputs (architectural provenance):
        Consumes exceptions, environment names, or push callbacks used during
        hub publication.

    Outputs (downstream usage):
        Authentication decisions or side effects consumed by
        `Summary.push_to_hub`.

    Invariants/constraints:
        These helpers should remain tightly scoped to publication behavior.

    """
    return HuggingFaceAuthResolver(
        defaults=HuggingFaceDefaults()
    ).require_token(env_name)


def _retry_hf_push_after_login[T](
    push: Callable[[str], T],
    *,
    token_env_name: str,
) -> T:
    """Retry a push operation after resolving interactive Hugging Face.

    Purpose:
        Repeat a push operation after resolving interactive Hugging Face
        authentication.

    Architectural role:
        Private publishing/authentication helper inside summary publication.

    Inputs (architectural provenance):
        Consumes exceptions, environment names, or push callbacks used during
        hub publication.

    Outputs (downstream usage):
        Authentication decisions or side effects consumed by
        `Summary.push_to_hub`.

    Invariants/constraints:
        These helpers should remain tightly scoped to publication behavior.

    """
    auth_resolver = HuggingFaceAuthResolver(defaults=HuggingFaceDefaults())
    token = require_hf_token(token_env_name)
    try:
        return push(token)
    except (OSError, RuntimeError, ValueError) as exc:
        if not _looks_like_hf_auth_error(exc):
            raise
        token = require_hf_token(token_env_name)
        auth_resolver.relogin(token=token)
        return push(token)


__all__ = ["Summary"]
