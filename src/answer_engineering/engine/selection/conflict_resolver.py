"""Deterministic conflict resolution for scored proposals.

Purpose:
    Choose stable winners and reject scored proposals whose spans overlap
    already-accepted winners.

Architectural role:
    Global selection stage between scoring and patch application.

Inputs:
    ScoredProposal collections produced by the scoring stage.

Outputs:
    Accepted and rejected proposal partitions used by conflict stages.

"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from answer_engineering.engine.scoring.base import (
    ScoredProposal,
)
from answer_engineering.engine.selection.base import Selector


@dataclass(frozen=True, slots=True)
class ConflictResolutionPolicy:
    """Policy controlling global overlap-rejection semantics.

    Purpose:
        Hold the rejection reason recorded when a scored proposal loses during
        global conflict resolution.

    Architectural role:
        Configuration value object for the global selection boundary.

    Inputs (architectural provenance):
        Provided when the conflict resolver selector is constructed.

    Outputs (downstream usage):
        Read when rejected proposals are annotated with a stable conflict
        reason.

    Invariants/constraints:
        The reject reason should remain stable enough for downstream events and
        reporting.

    """

    reject_reason: str = "conflict"


@dataclass(frozen=True, slots=True)
class ConflictSet:
    """Input wrapper for globally conflicting scored proposals.

    Purpose:
        Hold the scored proposals that must be ordered and filtered by overlap
        rules before apply-stage execution.

    Architectural role:
        Input value object for the global conflict-resolution stage.

    Inputs (architectural provenance):
        Built from scoring-stage output once all local winner decisions have
        been attached.

    Outputs (downstream usage):
        Consumed by the conflict resolver selector.

    Invariants/constraints:
        The tuple should contain only proposals eligible for one shared round of
        conflict arbitration.

    """

    scored: tuple[ScoredProposal, ...]


@dataclass(frozen=True, slots=True)
class ConflictResolutionResult:
    """Accepted and rejected outputs from one global conflict-resolution pass.

    Purpose:
        Separate scored proposals into those allowed to proceed and those
        rejected because a higher-ranked overlapping proposal won.

    Architectural role:
        Canonical return object for the global selection boundary.

    Inputs (architectural provenance):
        Constructed by the conflict resolver after ordering and overlap checks
        complete.

    Outputs (downstream usage):
        Consumed by the global-conflict stage and downstream rejection-event
        emission.

    Invariants/constraints:
        Accepted proposals must be mutually non-overlapping under the resolver's
        overlap semantics.

    """

    accepted: tuple[ScoredProposal, ...]
    rejected: tuple[ScoredProposal, ...]


def _tie_key(item: ScoredProposal) -> tuple[float, str, int, int, str, int]:
    """Deterministic sort key for globally ranking scored proposals.

    Purpose:
        Produce a stable ordering key based on score, rule identity, span,
        operation, and candidate index before winner selection or conflict
        checks.

    Architectural role:
        Ordering helper inside the global conflict-resolution boundary.

    Inputs (architectural provenance):
        Consumes one scored proposal from scoring-stage output.

    Outputs (downstream usage):
        Returned key is consumed by sorting operations that need stable global
        ordering.

    Invariants/constraints:
        The key must order equal-score items deterministically across runs.

    """
    proposal = item.proposal
    start, end = proposal.span_abs or (-1, -1)
    return (
        -item.score,
        proposal.rule_id,
        start,
        end,
        proposal.op.value,
        proposal.candidate_index,
    )


def _is_point(span: tuple[int, int]) -> bool:
    """Return whether a span is a zero-width point span.

    Purpose:
        Distinguish insertion-style point spans from interval spans so overlap
        checks can apply the correct semantics.

    Architectural role:
        Span helper inside the global conflict-resolution boundary.

    Inputs (architectural provenance):
        Consumes a normalized absolute span from a patch proposal.

    Outputs (downstream usage):
        Boolean result is consumed by `_overlaps` and related span predicates.

    Invariants/constraints:
        A point span is defined strictly as `start == end`.

    """
    return span[0] == span[1]


def _contains(span: tuple[int, int], point: int) -> bool:
    """Return whether an interval span contains a given point.

    Purpose:
        Support point-versus-interval overlap logic when conflict resolution
        compares insertion spans with non-zero-width spans.

    Architectural role:
        Span containment helper inside global conflict resolution.

    Inputs (architectural provenance):
        Consumes a normalized interval span and a point coordinate derived from
        proposal spans.

    Outputs (downstream usage):
        Boolean result is consumed by `_overlaps`.

    Invariants/constraints:
        Containment uses the resolver's inclusive boundary semantics for point
        spans.

    """
    return span[0] <= point <= span[1]


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return whether two proposal spans conflict under the resolver's overlap.

    Purpose:
        Apply the mixed point-span and interval-span rules used by global
        conflict resolution.

    Architectural role:
        Core span-comparison helper in the global selection boundary.

    Inputs (architectural provenance):
        Consumes normalized absolute spans from scored proposals.

    Outputs (downstream usage):
        Boolean result is consumed when deciding whether a lower-ranked proposal
        must be rejected.

    Invariants/constraints:
        The predicate must treat point spans consistently with `_is_point` and
        `_contains`.

    """
    if _is_point(a) and _is_point(b):
        return a[0] == b[0]
    if _is_point(a):
        return _contains(b, a[0])
    if _is_point(b):
        return _contains(a, b[0])
    return max(a[0], b[0]) < min(a[1], b[1])


@dataclass(slots=True)
class ConflictResolverSelector(Selector):
    """Deterministic selector for global proposal conflict resolution.

    Purpose:
        Rank scored proposals and reject lower-ranked proposals whose spans
        overlap accepted edits.

    Architectural role:
        Global selection boundary after per-rule scoring and before patch
        application.

    Inputs (architectural provenance):
        Receives scored proposals produced by scoring stages and grouped by the
        orchestrator for conflict handling.

    Outputs (downstream usage):
        Returns accepted proposals for patch application and rejected proposals
        for telemetry/reporting.

    Invariants/constraints:
        Ranking and overlap behavior must be deterministic. Rejected proposals
        are returned with the policy's rejection reason so downstream telemetry
        can explain why they were not applied.

    """

    policy: ConflictResolutionPolicy = ConflictResolutionPolicy()

    def select(self, scored: Iterable[ScoredProposal]) -> ScoredProposal | None:
        """Return the top-ranked scored proposal from one iterable.

        Purpose:
            Sort scored proposals with the resolver's deterministic tie key and
            pick the first item as the local winner for that set.

        Architectural role:
            Deterministic winner-picking method on the conflict resolver
            selector.

        Inputs (architectural provenance):
            Consumes scored proposals emitted by the scoring stage or grouped by
            later orchestration helpers.

        Outputs (downstream usage):
            The selected proposal is consumed by callers that need one stable
            winner before broader conflict handling.

        Invariants/constraints:
            Selection must be deterministic for equal-score inputs.

        """
        ordered = sorted(scored, key=_tie_key)
        if not ordered:
            return None
        return ordered[0]

    def resolve_conflicts(
        self, conflict_set: ConflictSet
    ) -> ConflictResolutionResult:
        """Resolve overlaps and return accepted/rejected proposals.

        Purpose:
            Apply global conflict rules so only mutually compatible scored
            proposals are accepted for a step.

        Architectural role:
            Global selection boundary between scoring and patch application.

        Inputs (architectural provenance):
            Receives scored proposals that may overlap in source span or
            otherwise compete for application.

        Outputs (downstream usage):
            Returns accepted proposals and rejected proposals consumed by apply
            stages and conflict telemetry.

        Invariants/constraints:
            Resolution must be deterministic for identical inputs. It should
            reject incompatible lower-priority proposals rather than relying on
            patch application to fail later.

        """
        ordered = sorted(conflict_set.scored, key=_tie_key)
        accepted: list[ScoredProposal] = []
        rejected: list[ScoredProposal] = []
        for item in ordered:
            span = item.proposal.span_abs
            if span is None:
                accepted.append(item)
                continue
            if any(
                kept.proposal.span_abs is not None
                and _overlaps(span, kept.proposal.span_abs)
                for kept in accepted
            ):
                rejected.append(
                    ScoredProposal(
                        proposal=item.proposal.with_updates(
                            reason=self.policy.reject_reason
                        ),
                        score=item.score,
                        prob_ratio_to_best=item.prob_ratio_to_best,
                    )
                )
                continue
            accepted.append(item)
        return ConflictResolutionResult(
            accepted=tuple(accepted), rejected=tuple(rejected)
        )

    def resolve(
        self,
        scored: Iterable[ScoredProposal],
    ) -> tuple[list[ScoredProposal], list[ScoredProposal]]:
        """Resolve scored proposals into accepted and rejected lists.

        Purpose:
            Provide the legacy-friendly iterable entry point for global conflict
            resolution while delegating actual overlap semantics to
            `resolve_conflicts`.

        Architectural role:
            Convenience adapter on the global selection boundary between scoring
            and patch application.

        Inputs (architectural provenance):
            Receives scored proposals produced by scoring/orchestration stages
            for one resolution pass.

        Outputs (downstream usage):
            Returns accepted and rejected proposals as mutable lists consumed by
            callers that do not need the richer `ConflictResolutionResult`
            wrapper.

        Invariants/constraints:
            Resolution semantics must remain identical to `resolve_conflicts`;
            this method should not introduce separate ranking or overlap policy.

        """
        resolved = self.resolve_conflicts(ConflictSet(scored=tuple(scored)))
        return list(resolved.accepted), list(resolved.rejected)
