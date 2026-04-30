"""Define report row types for case-type comparisons.

Purpose:
    Represent per-case-type accuracy deltas between anchor and candidate runs in
    small immutable rows that reporting code can pass around without recomputing
    metrics.

Architectural role:
    Shared report-type module between pairwise comparison logic and summary
    rendering.

Inputs (architectural provenance):
    Consumes already-computed accuracy values produced by evaluation comparison
    logic.

Outputs (downstream usage):
    Typed row containers consumed by comparison summaries and telemetry
    builders.

Invariants/constraints:
    These types should remain passive data carriers rather than performing
    comparison calculations themselves.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


class CaseCountView(Protocol):
    """Typed interface for per-case correctness-count records.

    Purpose:
        Describe the minimal count surface required by report consumers without
        coupling them to the concrete `CaseCount` implementation.

    Architectural role:
        Structural typing boundary between evaluation report values and
        row-summary code in the reproduction layer.

    Inputs (architectural provenance):
        Implemented by concrete count objects produced by accuracy reporting.

    Outputs (downstream usage):
        Allows summary builders to read correct and total counts for each case
        type.

    Invariants/constraints:
        Implementations should expose counts derived from one coherent evaluated
        result set.

    """

    @property
    def correct(self) -> int:
        """Return the number of correct answers in this case bucket."""
        raise NotImplementedError

    @property
    def total(self) -> int:
        """Return the total number of answers in this case bucket."""
        raise NotImplementedError


class AccuracyReportView(Protocol):
    """Typed interface for accuracy reports consumed by summaries.

    Purpose:
        Let session and comparison summary code depend on the report shape it
        needs rather than on the concrete `AccuracyReport` class.

    Architectural role:
        Read-only protocol boundary between evaluation reports and downstream
        rendering or aggregation helpers.

    Inputs (architectural provenance):
        Implemented by concrete accuracy reports built from evaluated results or
        reconstructed from stored counts.

    Outputs (downstream usage):
        Exposes per-case counts used to compute display rows and comparison
        fields.

    Invariants/constraints:
        The returned mapping should be keyed by stable case-type labels and
        should contain count views from the same evaluated population.

    """

    @property
    def by_case(self) -> Mapping[str, CaseCountView]:
        """Return per-case count views keyed by case type.

        Purpose:
            Expose the case-level accuracy breakdown behind a stable read-only
            view.

        Architectural role:
            Report-view accessor used by comparison tables and notebook
            displays.

        Inputs (architectural provenance):
            Reads normalized case-count data stored on the report view during
            construction.

        Outputs (downstream usage):
            Returns case-type keys mapped to count views consumed by renderers
            and tests.

        Invariants/constraints:
            The returned mapping should reflect canonical report data and should
            not recompute correctness from raw answer rows.

        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class CaseTypeAccuracyRow:
    """Store anchor-versus-candidate accuracy values for one case type.

    Purpose:
        Represent the number of cases and accuracy delta for one case-type slice
        so summary code can render comparisons without recalculating metrics.

    Architectural role:
        Small immutable report row in the evaluation boundary.

    Inputs (architectural provenance):
        Constructed from already-computed per-case-type accuracy values.

    Outputs (downstream usage):
        A typed row consumed by session comparison summaries and telemetry
        builders.

    Invariants/constraints:
        The row should describe one case type only.

    """

    case_type: str
    n_cases: int
    anchor_accuracy: float
    candidate_accuracy: float
    delta_accuracy: float


@dataclass(frozen=True, slots=True, init=False)
class ComparisonReportSessionAccuracy:
    """Store the full set of case-type comparison rows for one anchor/candidate.

    Purpose:
        Bundle the case-type accuracy rows computed for one comparison session
        into a stable value object passed across aggregation and summary code.

    Architectural role:
        Container type between pairwise comparison logic and summary rendering.

    Inputs (architectural provenance):
        Constructed from case-type rows computed from two accuracy reports.

    Outputs (downstream usage):
        A tuple-backed comparison summary consumed by reporting layers.

    Invariants/constraints:
        All rows should belong to the same anchor/candidate comparison.

    """

    rows: tuple[CaseTypeAccuracyRow, ...]

    def __init__(
        self,
        anchor_report: AccuracyReportView,
        candidate_report: AccuracyReportView,
    ) -> None:
        """Build case-type comparison rows from canonical source reports.

        Purpose:
            Construct the session-level accuracy comparison view shown in
            reproduction notebooks and reports.

        Architectural role:
            Report-facing constructor that adapts raw accuracy reports into a
            stable comparison table model.

        Inputs (architectural provenance):
            Receives baseline and edited accuracy reports from evaluation
            aggregation.

        Outputs (downstream usage):
            Stores normalized comparison rows consumed by markdown and notebook
            display code.

        Invariants/constraints:
            Construction should align rows by case type and avoid hiding missing
            or asymmetric case buckets.

        """
        anchor_by_case = anchor_report.by_case
        candidate_by_case = candidate_report.by_case
        computed_rows: list[CaseTypeAccuracyRow] = []
        for case_type in sorted(set(anchor_by_case) | set(candidate_by_case)):
            anchor_counts = anchor_by_case.get(case_type)
            candidate_counts = candidate_by_case.get(case_type)
            anchor_correct = anchor_counts.correct if anchor_counts else 0
            anchor_total = anchor_counts.total if anchor_counts else 0
            candidate_correct = (
                candidate_counts.correct if candidate_counts else 0
            )
            candidate_total = candidate_counts.total if candidate_counts else 0
            total = max(anchor_total, candidate_total)
            anchor_accuracy = (
                (anchor_correct / anchor_total) if anchor_total else 0.0
            )
            candidate_accuracy = (
                (candidate_correct / candidate_total)
                if candidate_total
                else 0.0
            )
            computed_rows.append(
                CaseTypeAccuracyRow(
                    case_type=case_type,
                    n_cases=total,
                    anchor_accuracy=anchor_accuracy,
                    candidate_accuracy=candidate_accuracy,
                    delta_accuracy=candidate_accuracy - anchor_accuracy,
                )
            )
        object.__setattr__(self, "rows", tuple(computed_rows))
