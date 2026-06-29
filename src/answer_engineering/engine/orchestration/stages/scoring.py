"""Stage that scores proposals and emits scoring telemetry events.

Purpose:
    Group proposals, score them in phase order, annotate winners with local
    probability ratios, and emit ProposalScored events.

Architectural role:
    Runtime stage for the scoring stage.

"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite

from answer_engineering.engine.patching import patcher
from answer_engineering.engine.patching.proposals import PatchProposal
from answer_engineering.engine.pipeline.attempts import (
    AttemptKey,
    AttemptState,
)
from answer_engineering.engine.pipeline.events import (
    PatchSkipped,
    ProposalScored,
)
from answer_engineering.engine.pipeline.messages import (
    ProposalsReady,
    ScoresReady,
)
from answer_engineering.engine.runtime.runtime_types import (
    PatchOp,
)
from answer_engineering.engine.scoring.base import (
    ScoredProposal,
    Scorer,
    ScoreResult,
    ScoringDiagnostics,
)
from answer_engineering.engine.selection import rule_winner
from answer_engineering.engine.span_utils import describe_span, normalize_span
from answer_engineering.engine.telemetry.events.event_sink import (
    NullRuntimeEventSink,
    RuntimeEventSink,
)
from answer_engineering.inference.prompting import prompt_prefix

_LOG = logging.getLogger(__name__)


def _editable_proposals(
    event: ProposalsReady,
    *,
    event_sink: RuntimeEventSink | None = None,
) -> list[PatchProposal]:
    """Filter invalid/non-editing proposals before scoring."""
    editable: list[PatchProposal] = []
    text = event.ctx.doc.text

    def skip(
        proposal: PatchProposal,
        reason: str,
        *,
        original_span: tuple[int, int] | None = None,
        corrected_span: tuple[int, int] | None = None,
    ) -> None:
        event_original_span = (
            original_span if original_span is not None else proposal.span_abs
        )
        if event_sink is not None:
            event_sink.emit(
                PatchSkipped(
                    rule_id=proposal.rule_id,
                    reason=reason,
                    rule_name=event.ctx.rule.name,
                    doc_len=len(text),
                    original_span=event_original_span,
                    corrected_span=corrected_span,
                    span_abs=proposal.span_abs,
                    nearby_text=describe_span(event_original_span, text),
                    stage="scoring",
                )
            )
        _LOG.warning(
            "%s rule_id=%s rule_name=%r span_abs=%s original_span=%s "
            "corrected_span=%s doc_len=%s op=%s %s",
            reason,
            proposal.rule_id,
            event.ctx.rule.name,
            proposal.span_abs,
            original_span,
            corrected_span,
            len(text),
            proposal.op.value,
            describe_span(event_original_span, text),
        )

    for proposal in event.proposals:
        if proposal.op == PatchOp.NOOP:
            continue
        if proposal.base_version_id != event.ctx.doc.version_id:
            skip(proposal, "proposal_base_version_mismatch")
            continue
        if proposal.span_abs is None:
            skip(proposal, "invalid_span_dropped")
            continue
        fixed = normalize_span(
            proposal.span_abs, event.ctx.doc.text, mode="fallback_then_clamp"
        )
        if fixed.span is None:
            skip(proposal, "invalid_span_dropped")
            continue
        if fixed.span != proposal.span_abs:
            reason = fixed.reason or "invalid_span_clamped"
            if event_sink is not None:
                event_sink.emit(
                    PatchSkipped(
                        rule_id=proposal.rule_id,
                        reason=reason,
                        rule_name=event.ctx.rule.name,
                        doc_len=len(text),
                        original_span=fixed.original,
                        corrected_span=fixed.span,
                        span_abs=proposal.span_abs,
                        nearby_text=describe_span(fixed.original, text),
                        stage="scoring",
                    )
                )
            proposal = proposal.with_updates(span_abs=fixed.span)
            _LOG.warning(
                "%s rule_id=%s rule_name=%r original_span=%s "
                "corrected_span=%s doc_len=%s %s",
                reason,
                proposal.rule_id,
                event.ctx.rule.name,
                fixed.original,
                fixed.span,
                len(text),
                describe_span(fixed.original, text),
            )
        try:
            first = patcher.apply_patch(event.ctx.doc, proposal).text
            second = patcher.apply_patch(event.ctx.doc, proposal).text
        except ValueError as exc:
            skip(proposal, "patcher_rejected_proposal")
            _LOG.warning("patcher rejected proposal before scoring: %s", exc)
            continue
        if first != second:
            skip(proposal, "unstable_patch_application")
            continue
        editable.append(proposal)
    return editable


def _group_by_span_and_op(
    proposals: list[PatchProposal],
) -> dict[tuple[tuple[int, int] | None, str], list[PatchProposal]]:
    grouped: dict[tuple[tuple[int, int] | None, str], list[PatchProposal]] = {}
    for proposal in proposals:
        key = (proposal.span_abs, proposal.op.value)
        grouped.setdefault(key, []).append(proposal)
    return grouped


@dataclass(slots=True)
class ScoringStage:
    """Score grouped proposals and emit proposal-scoring events.

    Purpose:
        Convert generated proposal groups into scored decisions that selection
        can compare deterministically.

    Architectural role:
        Orchestration stage between proposal generation and local/global
        selection. It owns scorer invocation but not scoring policy internals.

    Inputs (architectural provenance):
        Receives proposal groups, document context, runtime scoring services,
        and event sinks from the plan runner context.

    Outputs (downstream usage):
        Returns scored proposal groups and emits telemetry records used by
        reporting and golden tests.

    Invariants/constraints:
        The stage should preserve scorer ownership of model or deterministic
        score calculations and keep selection out of scoring execution.

    """

    scorer: Scorer
    event_sink: RuntimeEventSink = field(default_factory=NullRuntimeEventSink)
    attempt_state: AttemptState = field(default_factory=AttemptState)

    def handle(
        self,
        event: ProposalsReady,
        *,
        on_group_begin: Callable[[tuple[int, int] | None, str, int], None]
        | None = None,
        on_candidate_scored: Callable[[ScoredProposal, int, int], None]
        | None = None,
    ) -> ScoresReady:
        """Score incoming proposals and return scored stage output.

        Purpose:
            Invoke the configured scorer for one stage input and prepare scored
            proposal records for downstream selection.

        Architectural role:
            Orchestration adapter around the scoring boundary.

        Inputs (architectural provenance):
            Receives the current step context and proposal batch produced by
            earlier orchestration stages.

        Outputs (downstream usage):
            Returns scored proposals and scoring events consumed by selection
            and telemetry pipelines.

        Invariants/constraints:
            The stage coordinates scoring only; it should not decide global
            conflicts or apply patches.

        """
        editable = _editable_proposals(event, event_sink=self.event_sink)
        grouped = _group_by_span_and_op(editable)

        scored_all: list[ScoredProposal] = []
        model_scored_any = False
        num_calls = 0
        for (span, op), proposals in grouped.items():
            pivot_prefix_hash = prompt_prefix.stable_prefix_fingerprint(
                event.ctx.doc.text[: (span[0] if span else 0)]
            )
            state_key = AttemptKey(
                span=span, op=op, prefix_hash=pivot_prefix_hash
            )
            filtered = [
                p
                for p in proposals
                if not self.attempt_state.reject_duplicate(
                    state_key, p.candidate_hash
                )
            ]
            generated_or_static = [
                proposal
                for proposal in filtered
                if proposal.candidate_kind in {"generated", "static"}
            ]
            fallback = [
                proposal
                for proposal in filtered
                if proposal.candidate_kind == "fallback"
            ]
            ordered = [
                group for group in (generated_or_static, fallback) if group
            ]
            if on_group_begin is not None:
                on_group_begin(span, op, len(filtered))

            phase_winner: ScoredProposal | None = None
            rank = 0
            for phase in ordered:
                phase_result = self._score_phase(event, phase)
                raw_group = phase_result.scored
                model_scored_any = (
                    model_scored_any or phase_result.model_scored_any
                )
                num_calls += phase_result.num_calls

                decision = rule_winner.RuleWinnerDecision(
                    [item.score for item in raw_group],
                    min_prob_ratio_to_best=(
                        event.ctx.rule.policy.min_prob_ratio_to_best
                    ),
                )
                for idx, item in enumerate(raw_group):
                    rank += 1
                    ratio = (
                        decision.ratios_to_best[idx]
                        if idx < len(decision.ratios_to_best)
                        else None
                    )
                    scored_item = ScoredProposal(
                        proposal=item.proposal.with_updates(
                            cached_prob_ratio_to_best=ratio
                        ),
                        score=item.score,
                        prob_ratio_to_best=ratio,
                    )
                    scored_all.append(scored_item)
                    if on_candidate_scored is not None:
                        on_candidate_scored(scored_item, rank, len(filtered))
                if decision.winner_index is not None:
                    winner_idx = decision.winner_index
                    winner_ratio = (
                        decision.ratios_to_best[winner_idx]
                        if winner_idx < len(decision.ratios_to_best)
                        else None
                    )
                    phase_winner = ScoredProposal(
                        proposal=raw_group[winner_idx].proposal.with_updates(
                            cached_prob_ratio_to_best=winner_ratio
                        ),
                        score=raw_group[winner_idx].score,
                        prob_ratio_to_best=winner_ratio,
                    )
                    break

            winning_hash = (
                phase_winner.proposal.candidate_hash
                if phase_winner is not None
                and phase_winner.proposal.candidate_kind != "fallback"
                else None
            )
            self.attempt_state.record_attempt(
                state_key, winning_candidate_hash=winning_hash
            )

        result = ScoreResult(
            scored=scored_all,
            diagnostics=ScoringDiagnostics(
                model_scored=model_scored_any, num_calls=num_calls
            ),
        )
        for scored in result.scored:
            self.event_sink.emit(
                ProposalScored(
                    rule_id=scored.proposal.rule_id,
                    patch_hash=scored.proposal.patch_hash,
                )
            )
        return ScoresReady(ctx=event.ctx, result=result)

    def _score_phase(
        self, event: ProposalsReady, phase: list[PatchProposal]
    ) -> ScorePhaseResult:
        raw_group: list[ScoredProposal | None] = [None for _ in phase]
        proposals_to_score: list[PatchProposal] = []
        uncached_indexes: list[int] = []
        for index, proposal in enumerate(phase):
            cached_probe_score = proposal.cached_score_logprob
            if (
                proposal.candidate_kind == "generated"
                and cached_probe_score is not None
                and isfinite(cached_probe_score)
            ):
                raw_group[index] = ScoredProposal(
                    proposal=proposal, score=cached_probe_score
                )
                continue
            proposals_to_score.append(proposal)
            uncached_indexes.append(index)

        model_scored_any = False
        num_calls = 0
        if proposals_to_score:
            partial_result = self.scorer.score(event.ctx, proposals_to_score)
            if partial_result.diagnostics is not None:
                model_scored_any = partial_result.diagnostics.model_scored
                num_calls = partial_result.diagnostics.num_calls
            if len(partial_result.scored) != len(proposals_to_score):
                raise ValueError(
                    "scorer returned mismatched scored proposal count"
                )
            for index, partial in zip(
                uncached_indexes, partial_result.scored, strict=True
            ):
                raw_group[index] = ScoredProposal(
                    proposal=partial.proposal, score=partial.score
                )
        return self.ScorePhaseResult(
            scored=[item for item in raw_group if item is not None],
            model_scored_any=model_scored_any,
            num_calls=num_calls,
        )

    @dataclass(frozen=True, slots=True)
    class ScorePhaseResult:
        """Internal result of scoring one proposal phase.

        Purpose:
            Carry the scored proposals plus per-phase diagnostics after the
            stage scores either generated/static candidates or fallback
            candidates.

        Architectural role:
            Phase-level result object inside the scoring stage.

        Inputs (architectural provenance):
            Constructed by `_score_phase()` while the stage evaluates one
            ordered candidate phase.

        Outputs (downstream usage):
            Consumed by `handle()` when merging phases, attaching ratios, and
            recording diagnostics.

        Invariants/constraints:
            All scored proposals and diagnostics in this object must correspond
            to the same phase.

        """

        scored: list[ScoredProposal]
        model_scored_any: bool
        num_calls: int
