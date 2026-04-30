"""Build comparison-oriented views over evaluated subruns.

Purpose:
    Package evaluated subruns into stable records that can expose telemetry,
    compute anchor-versus-candidate comparisons, and materialize group-level
    comparison rows.

Architectural role:
    Aggregation module that bridges evaluation reports to run-summary comparison
    tables.

Inputs (architectural provenance):
    Consumes evaluated subruns, accuracy reports, and group-run context values
    produced by session summarization.

Outputs (downstream usage):
    Subrun comparison records and normalized group comparison rows consumed by
    summary payload builders.

Invariants/constraints:
    This module assumes comparisons happen between subruns that already share a
    scope and evaluation basis.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ae_paper_reproduction.core.aggregation.rule_stats import TelemetryItem
from ae_paper_reproduction.core.evaluation.report_types import (
    ComparisonReportSessionAccuracy,
)
from ae_paper_reproduction.core.evaluation.reports import (
    AccuracyReport,
    PairwiseComparisonReport,
    PairwiseOutcomeTransitions,
    RulesetEvaluationResult,
)


class ComparisonSubrun(Protocol):
    """Describe the minimum subrun view needed to build pairwise comparisons.

    Purpose:
        Define the structural contract that comparison builders use when they
        need identifiers, scope, raw evaluation results, and an accuracy report
        from an evaluated subrun.

    Architectural role:
        Protocol boundary between evaluated subrun carriers and
        comparison-building helpers.

    Inputs (architectural provenance):
        Implemented by subrun result objects produced after session execution.

    Outputs (downstream usage):
        A stable, read-only view consumed by pairwise comparison builders.

    Invariants/constraints:
        Implementations must expose one coherent evaluated subrun; mixed-scope
        or partially evaluated views would make comparisons invalid.

    """

    @property
    def subrun_id(self) -> str:
        """Return the stable subrun identifier used to join comparison outputs.

        Purpose:
            Expose the stable subrun identifier used to join comparison outputs.

        Architectural role:
            Read-only protocol member consumed by comparison builders.

        Inputs (architectural provenance):
            Accessed by aggregation code while constructing pairwise comparison
            records.

        Outputs (downstream usage):
            A single value drawn from an evaluated subrun implementation.

        Invariants/constraints:
            The returned value must describe the same evaluated subrun as the
            other protocol members.

        """
        raise NotImplementedError

    @property
    def ruleset_name(self) -> str:
        """Return the human-facing ruleset name for this evaluated subrun.

        Purpose:
            Expose the human-facing ruleset name for evaluated subrun.

        Architectural role:
            Read-only protocol member consumed by comparison builders.

        Inputs (architectural provenance):
            Accessed by aggregation code while constructing pairwise comparison
            records.

        Outputs (downstream usage):
            A single value drawn from an evaluated subrun implementation.

        Invariants/constraints:
            The returned value must describe the same evaluated subrun as the
            other protocol members.

        """
        raise NotImplementedError

    @property
    def scope_label(self) -> str:
        """Return the normalized scope label for aligned subrun comparisons.

        Purpose:
            Expose the normalized scope label used to compare like-for-like
            subruns.

        Architectural role:
            Read-only protocol member consumed by comparison builders.

        Inputs (architectural provenance):
            Accessed by aggregation code while constructing pairwise comparison
            records.

        Outputs (downstream usage):
            A single value drawn from an evaluated subrun implementation.

        Invariants/constraints:
            The returned value must describe the same evaluated subrun as the
            other protocol members.

        """
        raise NotImplementedError

    @property
    def results(self) -> Sequence[RulesetEvaluationResult]:
        """Return the per-case evaluation results for this subrun.

        Purpose:
            Expose the per-case evaluation results for subrun.

        Architectural role:
            Read-only protocol member consumed by comparison builders.

        Inputs (architectural provenance):
            Accessed by aggregation code while constructing pairwise comparison
            records.

        Outputs (downstream usage):
            A single value drawn from an evaluated subrun implementation.

        Invariants/constraints:
            The returned value must describe the same evaluated subrun as the
            other protocol members.

        """
        raise NotImplementedError

    @property
    def report(self) -> AccuracyReport:
        """Return the precomputed accuracy report for this subrun.

        Purpose:
            Expose the precomputed accuracy report for subrun.

        Architectural role:
            Read-only protocol member consumed by comparison builders.

        Inputs (architectural provenance):
            Accessed by aggregation code while constructing pairwise comparison
            records.

        Outputs (downstream usage):
            A single value drawn from an evaluated subrun implementation.

        Invariants/constraints:
            The returned value must describe the same evaluated subrun as the
            other protocol members.

        """
        raise NotImplementedError


class GroupRunContextView(Protocol):
    """Describe the run-level context needed to stamp group comparison rows.

    Purpose:
        Define the minimal structural contract for attaching run identity and
        creation time to comparison rows without depending on the full
        group-context implementation.

    Architectural role:
        Protocol used by group-row builders inside session summarization.

    Inputs (architectural provenance):
        Implemented by group context objects created during summary assembly.

    Outputs (downstream usage):
        Read-only group identifiers and timestamps consumed by comparison row
        builders.

    Invariants/constraints:
        The values must come from one group run context so generated rows share
        a consistent run identity.

    """

    @property
    def group_run_id(self) -> str:
        """Return the run identifier shared by all comparison rows in the group.

        Purpose:
            Expose the run identifier shared by all comparison rows in the
            group.

        Architectural role:
            Protocol member used when stamping comparison rows.

        Inputs (architectural provenance):
            Accessed by group comparison row builders during summary assembly.

        Outputs (downstream usage):
            A scalar context value copied into generated rows.

        Invariants/constraints:
            The value should already be normalized before it reaches aggregation
            code.

        """
        raise NotImplementedError

    @property
    def created_at_utc(self) -> str:
        """Return the UTC timestamp recorded for the group run.

        Purpose:
            Expose the UTC timestamp recorded for the group run.

        Architectural role:
            Protocol member used when stamping comparison rows.

        Inputs (architectural provenance):
            Accessed by group comparison row builders during summary assembly.

        Outputs (downstream usage):
            A scalar context value copied into generated rows.

        Invariants/constraints:
            The value should already be normalized before it reaches aggregation
            code.

        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SubrunEvaluationResult:
    """Capture one evaluated subrun for telemetry and reporting.

    Purpose:
        Store the evaluated answers, report, rules markdown, and naming metadata
        for one subrun so later summary code can derive telemetry rows and
        comparisons without re-running evaluation.

    Architectural role:
        Immutable aggregation record between subrun execution and final summary
        building.

    Inputs (architectural provenance):
        Constructed from an executed subrun and its evaluation outputs.

    Outputs (downstream usage):
        A stable subrun-level bundle consumed by telemetry extraction and
        comparison code.

    Invariants/constraints:
        All fields must describe the same executed subrun and evaluation batch.

    """

    ruleset_name: str
    rules_markdown: str
    subrun_id: str
    case_type_filter: str | None
    scope_label: str
    results: Sequence[RulesetEvaluationResult]
    report: AccuracyReport
    n_eval_actual: int

    def telemetry_items(self) -> tuple[TelemetryItem, ...]:
        """Extract merged-telemetry inputs from the stored evaluation results.

        Purpose:
            Collect the per-case runtime telemetry snapshots attached to
            evaluated answers and normalize them into `TelemetryItem` objects
            for later aggregation.

        Architectural role:
            Subrun-level adapter from evaluation results to telemetry
            aggregation.

        Inputs (architectural provenance):
            Reads the evaluation results stored on this record.

        Outputs (downstream usage):
            A tuple of telemetry items consumed by `merge_ae_telemetry`.

        Invariants/constraints:
            Only results that actually carry runtime telemetry should contribute
            items.

        """
        return tuple(
            TelemetryItem(result.ae_telemetry)
            for result in self.results
            if result.ae_telemetry is not None
        )


@dataclass(frozen=True, slots=True, init=False)
class SubrunComparisonResult:
    """Represent a pairwise comparison of anchor and candidate subruns.

    Purpose:
        Hold the identities, overall comparison report, and case-type accuracy
        rows for one anchor-versus-candidate pairing within a shared scope.

    Architectural role:
        Aggregation record produced during run summary assembly.

    Inputs (architectural provenance):
        Constructed from two evaluated subruns that are ready for comparison.

    Outputs (downstream usage):
        A stable comparison object consumed by group summary and reporting code.

    Invariants/constraints:
        Both subruns must already be evaluated and should belong to the same
        comparison scope.

    """

    anchor_subrun_id: str
    anchor_ruleset_name: str
    candidate_subrun_id: str
    candidate_ruleset_name: str
    scope_label: str
    report: PairwiseComparisonReport
    case_type_rows: ComparisonReportSessionAccuracy

    def __init__(
        self,
        anchor_subrun: ComparisonSubrun,
        candidate_subrun: ComparisonSubrun,
    ) -> None:
        """Build a pairwise comparison from two evaluated subruns.

        Purpose:
            Compute overall and case-type comparison reports by treating the
            first subrun as the anchor baseline and the second as the candidate
            to compare against it.

        Architectural role:
            Constructor path used by summary code when anchor and candidate
            subruns are paired.

        Inputs (architectural provenance):
            Consumes two evaluated subruns that satisfy the `ComparisonSubrun`
            protocol.

        Outputs (downstream usage):
            A fully populated `SubrunComparisonResult`.

        Invariants/constraints:
            The caller is responsible for pairing compatible subruns; this
            method does not realign scopes or datasets.

        """
        comparison_report = PairwiseComparisonReport(
            anchor_results=anchor_subrun.results,
            anchor_report=anchor_subrun.report,
            candidate_results=candidate_subrun.results,
            candidate_report=candidate_subrun.report,
        )
        object.__setattr__(self, "anchor_subrun_id", anchor_subrun.subrun_id)
        object.__setattr__(
            self, "anchor_ruleset_name", anchor_subrun.ruleset_name
        )
        object.__setattr__(
            self, "candidate_subrun_id", candidate_subrun.subrun_id
        )
        object.__setattr__(
            self, "candidate_ruleset_name", candidate_subrun.ruleset_name
        )
        object.__setattr__(self, "scope_label", candidate_subrun.scope_label)
        object.__setattr__(self, "report", comparison_report)
        object.__setattr__(
            self,
            "case_type_rows",
            ComparisonReportSessionAccuracy(
                anchor_subrun.report,
                candidate_subrun.report,
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class GroupComparisonRow:
    """Materialize one reporting row for an anchor-versus-candidate comparison.

    Purpose:
        Flatten a pairwise comparison into the compact scalar values used by
        group-level comparison tables and exported telemetry bundles.

    Architectural role:
        Final row type between comparison logic and group reporting artifacts.

    Inputs (architectural provenance):
        Constructed from a group run context and a pairwise comparison result.

    Outputs (downstream usage):
        A single comparison row consumed by group telemetry payloads and
        markdown/table renderers.

    Invariants/constraints:
        The row should preserve the run and subrun identifiers needed to join it
        back to richer artifacts.

    """

    group_run_id: str
    created_at_utc: str
    anchor_subrun_id: str
    candidate_subrun_id: str
    anchor_ruleset_name: str
    candidate_ruleset_name: str
    delta_accuracy: float
    improved: int
    degraded: int
    unchanged_correct: int
    unchanged_incorrect: int

    def __init__(
        self,
        *,
        group_context: GroupRunContextView,
        comparison_output: SubrunComparisonResult,
    ) -> None:
        """Build a reporting row from group context and one comparison result.

        Purpose:
            Copy run identifiers and timestamps from the group context and
            combine them with improvement/degradation counts from the pairwise
            comparison report.

        Architectural role:
            Constructor path used at the final flattening step of run summary
            assembly.

        Inputs (architectural provenance):
            Consumes a group-level context object and one completed subrun
            comparison.

        Outputs (downstream usage):
            A `GroupComparisonRow` ready for inclusion in group telemetry.

        Invariants/constraints:
            The comparison report must already contain outcome transition counts
            and delta accuracy values.

        """
        transitions: PairwiseOutcomeTransitions = (
            comparison_output.report.outcome_transitions
        )
        object.__setattr__(self, "group_run_id", group_context.group_run_id)
        object.__setattr__(self, "created_at_utc", group_context.created_at_utc)
        object.__setattr__(
            self, "anchor_subrun_id", comparison_output.anchor_subrun_id
        )
        object.__setattr__(
            self, "candidate_subrun_id", comparison_output.candidate_subrun_id
        )
        object.__setattr__(
            self, "anchor_ruleset_name", comparison_output.anchor_ruleset_name
        )
        object.__setattr__(
            self,
            "candidate_ruleset_name",
            comparison_output.candidate_ruleset_name,
        )
        object.__setattr__(
            self, "delta_accuracy", comparison_output.report.delta_overall
        )
        object.__setattr__(
            self, "improved", transitions.anchor_incorrect_to_candidate_correct
        )
        object.__setattr__(
            self, "degraded", transitions.anchor_correct_to_candidate_incorrect
        )
        object.__setattr__(
            self,
            "unchanged_correct",
            transitions.anchor_correct_to_candidate_correct,
        )
        object.__setattr__(
            self,
            "unchanged_incorrect",
            transitions.anchor_incorrect_to_candidate_incorrect,
        )


__all__ = [
    "GroupComparisonRow",
    "SubrunComparisonResult",
    "SubrunEvaluationResult",
]
