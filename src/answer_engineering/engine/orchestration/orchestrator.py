"""Runtime orchestration for proposal, scoring, conflict resolution, and apply.

Purpose:
    Coordinate runtime stages to produce deterministic patch decisions and
    runtime events.

Architectural role:
    Core runtime coordinator between compiled plan execution and final text
    output.

Architectural direction:
    The orchestration boundary should remain explicit but become more legible as
    a coordination seam rather than a knowledge-concentration point.

Why this matters:
    When one coordinator knows too much about many stage details, extension cost
    rises and boundary ownership becomes harder to explain.

What better would look like:
    Stage contracts remain explicit and easier to reason about without requiring
    broad orchestration internals knowledge for every change.

How improvement can be recognized:
    - Fewer unrelated reasons for this module to change
    - Simpler explanation of stage responsibilities and handoffs
    - Lower cross-stage coupling in orchestration-specific logic

Open constraint:
    The orchestration shape should continue to evolve with runtime behavior
    rather than being frozen prematurely.

"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from math import exp

from answer_engineering.config.patch_score_policy import PatchScorePolicy
from answer_engineering.engine.orchestration.event_loop import (
    RuntimeQueueRunner,
)
from answer_engineering.engine.orchestration.stages.apply import ApplyStage
from answer_engineering.engine.orchestration.stages.global_conflict import (
    GlobalConflictStage,
)
from answer_engineering.engine.orchestration.stages.proposal import (
    ProposalStage,
)
from answer_engineering.engine.orchestration.stages.scoring import (
    ScoringStage,
)
from answer_engineering.engine.pipeline.attempts import (
    AttemptState,
)
from answer_engineering.engine.pipeline.context import (
    StepContext,
    StepSnapshot,
)
from answer_engineering.engine.pipeline.events import (
    Event,
    ProposalRejected,
)
from answer_engineering.engine.pipeline.messages import (
    GlobalWinnersReady,
    RunnerEvent,
    StepRequested,
)
from answer_engineering.engine.proposal.proposal_engine import (
    ProposalPlanner,
)
from answer_engineering.engine.runtime.runtime_types import (
    AppliedPatch,
    DocumentState,
    PatchOp,
    PatchProposal,
)
from answer_engineering.engine.scoring.base import (
    ConfigurableScorer,
    ScoredProposal,
    Scorer,
)
from answer_engineering.engine.scoring.logits.scorer import LogitsScorer
from answer_engineering.engine.selection.base import Selector
from answer_engineering.engine.selection.conflict_resolver import (
    ConflictResolverSelector,
)
from answer_engineering.engine.telemetry.events import decision_logging
from answer_engineering.engine.telemetry.events.decision_logging import (
    CandidateRow,
    DecisionEmitter,
    DecisionEvent,
    DecisionFormatter,
    DecisionGroupContext,
    ScoredCandidateRecord,
)
from answer_engineering.engine.telemetry.events.event_sink import (
    ConsoleRuntimeEventSink,
    DebugEventEmitter,
    NullRuntimeEventSink,
    RecordingRuntimeEventSink,
    RuntimeEventSink,
)
from answer_engineering.inference.model_types import (
    GenerationRuntimeProtocol,
)
from answer_engineering.rules.compile.plan import (
    PlanIR,
    RulePlan,
)

__all__ = [
    "CandidateRow",
    "DecisionEvent",
    "DecisionFormatter",
    "OrchestratorResult",
    "PlanRunner",
]


@dataclass(slots=True)
class OrchestratorResult:
    """Aggregate runtime outcome produced by ``PlanRunner.run``.

    Purpose:
        Carry the final runtime outcome assembled after queue execution,
        telemetry capture, and patch application complete.

    Architectural role:
        Orchestration layer above proposal, scoring, selection, and apply
        stages.

    Inputs:
        Constructed from final runtime queue state after proposal/scoring/apply.

    Outputs:
        Consumed by API adapters and converted into ``CoreDecision`` payloads.

    """

    final_doc: DocumentState
    events: list[Event]
    applied_patches: list[AppliedPatch]
    proposals: list[PatchProposal]


@dataclass(slots=True)
class _StageBundle:
    """Concrete stages and emitters assembled for one ``PlanRunner`` execution.

    Purpose:
        Keep the proposal, scoring, conflict-resolution, apply, and decision
        emission collaborators together while the runner wires the event loop.

    """

    proposal: ProposalStage
    scoring: ScoringStage
    decision_emitter: DecisionEmitter
    decision_event_ids: dict[tuple[str, tuple[int, int] | None, str], int] = (
        field(default_factory=lambda: {})
    )
    global_conflict: GlobalConflictStage = field(
        default_factory=lambda: GlobalConflictStage(
            selector=ConflictResolverSelector()
        )
    )
    apply: ApplyStage = field(default_factory=ApplyStage)


def _global_source_scored(
    *,
    item: GlobalWinnersReady,
    scored_proposals: list[ScoredProposal],
) -> list[ScoredProposal]:
    if item.winners:
        return [
            ScoredProposal(
                proposal=winner,
                score=(
                    winner.cached_score_logprob
                    if winner.cached_score_logprob is not None
                    else winner.score
                ),
            )
            for winner in item.winners
        ]
    return scored_proposals


def _emit_noop_rejections(
    proposals: list[PatchProposal],
    recorder: RecordingRuntimeEventSink,
) -> None:
    for proposal in proposals:
        if proposal.op == PatchOp.NOOP:
            recorder.emit(
                ProposalRejected(
                    rule_id=proposal.rule_id, reason=proposal.reason
                )
            )


@dataclass(frozen=True, slots=True)
class _RunInitialization:
    """Initial document, accumulators, and recorder prepared before queue.

    Purpose:
        Bundle the mutable run state that ``PlanRunner.run`` seeds before
        handing control to the queue runner.

    """

    doc: DocumentState
    token_cursor: int
    all_proposals: list[PatchProposal]
    all_applied: list[AppliedPatch]
    recorder: RecordingRuntimeEventSink

    def __iter__(self):
        """Yield initial run components in the tuple order expected by the.

        Purpose:
            Support concise unpacking of the initialization bundle without
            exposing a positional tuple type directly.

        Architectural role:
            Convenience adapter around structured run-start state.

        Inputs (architectural provenance):
            Operates on fields prepared at execution start: document snapshot,
            token cursor, proposal/apply accumulators, and event recorder.

        Outputs (downstream usage):
            Streams those fields in a stable order to the calling orchestration
            code.

        Invariants/constraints:
            Yield order must stay aligned with the unpacking sites that consume
            this bundle.

        """
        yield self.doc
        yield self.token_cursor
        yield self.all_proposals
        yield self.all_applied
        yield self.recorder


@dataclass(frozen=True, slots=True)
class _RuntimeStagesBuild:
    """Configured stage bundle plus the seed queue returned by runner setup.

    Purpose:
        Transport the stage objects and initial ``RunnerEvent`` deque from the
        setup helpers into the main orchestration path.

    """

    stages: _StageBundle
    queue: deque[RunnerEvent]

    def __iter__(self):
        """Yield configured stages and their initial queue in unpacking order.

        Purpose:
            Provide tuple-style unpacking for the runtime stage bundle returned
            during runner construction.

        Architectural role:
            Small transport adapter between stage construction and orchestrator
            setup.

        Inputs:
            Operates on the already-built stage bundle and seed queue.

        Outputs:
            Returns those components in the order expected by orchestration
            callers.

        """
        yield self.stages
        yield self.queue


@dataclass(slots=True)
class PlanRunner:
    """Coordinate proposal, scoring, conflict, and apply stages.

    Purpose:
        Own one runtime execution by configuring collaborators, seeding the
        queue, and assembling the final result.

    Architectural role:
        Orchestration layer above proposal, scoring, selection, and apply
        stages.

    Inputs:
        Consumes run-owned state, queue items, or configured collaborators
        prepared by orchestration.

    Outputs:
        Returns or emits values that are consumed by downstream runtime stages,
        telemetry, or final result assembly.

    """

    proposal_engine: ProposalPlanner = field(default_factory=ProposalPlanner)
    scorer_component: Scorer = field(default_factory=LogitsScorer)
    selector: Selector = field(default_factory=ConflictResolverSelector)
    event_sink: RuntimeEventSink = field(default_factory=NullRuntimeEventSink)
    attempt_state: AttemptState = field(default_factory=AttemptState)
    runtime: GenerationRuntimeProtocol | None = None
    require_model_scoring: bool = False
    verbose: bool = True
    patch_score_policy: PatchScorePolicy | None = None
    trajectory_debug: bool = False
    decision_formatter: DecisionFormatter = field(
        default_factory=DecisionFormatter
    )
    debug_emitter: DebugEventEmitter = field(default_factory=DebugEventEmitter)
    _event_seq: int = 0

    def _configure_scorer(self) -> None:
        """Apply runtime-dependent scorer configuration before execution begins.

        Purpose:
            Inject patch-score policy into ``LogitsScorer`` and call the
            optional ``ConfigurableScorer`` hook with the active generation
            runtime.

        Architectural role:
            Runtime-configuration bridge between orchestrator state and the
            scoring subsystem.

        Inputs:
            Uses ``self.runtime``, ``self.require_model_scoring``, and optional
            ``patch_score_policy`` already stored on ``PlanRunner``.

        Outputs:
            Mutates the scorer component so later scoring calls use the correct
            runtime and policy configuration.

        """
        if (
            isinstance(self.scorer_component, LogitsScorer)
            and self.patch_score_policy is not None
        ):
            self.scorer_component.policy = self.patch_score_policy
        if isinstance(self.scorer_component, ConfigurableScorer):
            self.scorer_component.configure(
                runtime=self.runtime,
                require_model_scoring=self.require_model_scoring,
            )

    def _resolve_overlaps(
        self,
        proposals: list[PatchProposal],
    ) -> tuple[list[PatchProposal], list[PatchProposal]]:
        scored = [
            ScoredProposal(
                proposal=proposal,
                score=(
                    proposal.cached_score_logprob
                    if proposal.cached_score_logprob is not None
                    else proposal.score
                ),
            )
            for proposal in proposals
        ]
        accepted_scored, rejected_scored = self.selector.resolve(scored)
        if self.verbose:
            for rejected in rejected_scored:
                proposal = rejected.proposal
                winner = next(
                    (
                        kept.proposal
                        for kept in accepted_scored
                        if kept.proposal.span_abs is not None
                        and proposal.span_abs is not None
                        and decision_logging.conflicts(
                            kept.proposal.span_abs, proposal.span_abs
                        )
                    ),
                    None,
                )
                if winner is not None and self.verbose:
                    self.debug_emitter.emit(
                        f"core: CONFLICT winner={winner.rule_id} "
                        f"loser={proposal.rule_id}"
                    )
        return [item.proposal for item in accepted_scored], [
            item.proposal for item in rejected_scored
        ]

    def run(self, plan: PlanIR, execution: StepSnapshot) -> OrchestratorResult:
        """Run the orchestrator pipeline for one document snapshot.

        Purpose:
            Implement the operation performed by this orchestration component.

        Architectural role:
            Orchestration helper inside the runtime owner that coordinates stage
            transitions and run-level mutable state.

        Inputs:
            Consumes run-owned state, queue items, or configured collaborators
            prepared by orchestration.

        Outputs:
            Returns or emits values that are consumed by downstream runtime
            stages, telemetry, or final result assembly.

        """
        init = self._initialize_run(
            text=execution.snapshot_text,
            token_index=execution.token_index,
        )
        runtime = self._build_runtime_stages(
            doc=init.doc,
            plan=plan,
            execution=execution,
            recorder=init.recorder,
        )
        doc = self._run_event_queue(
            queue=runtime.queue,
            stages=runtime.stages,
            doc=init.doc,
            all_proposals=init.all_proposals,
            all_applied=init.all_applied,
            recorder=init.recorder,
        )
        _emit_noop_rejections(init.all_proposals, init.recorder)
        return OrchestratorResult(
            final_doc=doc,
            events=init.recorder.events,
            applied_patches=init.all_applied,
            proposals=init.all_proposals,
        )

    def _initialize_run(
        self, *, text: str, token_index: int
    ) -> _RunInitialization:
        self._configure_scorer()
        if (self.trajectory_debug or self.verbose) and isinstance(
            self.event_sink, NullRuntimeEventSink
        ):
            self.event_sink = ConsoleRuntimeEventSink()
        self.debug_emitter.event_sink = self.event_sink
        self.proposal_engine.configure_runtime(
            runtime=self.runtime,
            trajectory_debug=self.trajectory_debug,
        )
        self.proposal_engine.reset_run_state()
        doc = DocumentState(text)
        token_cursor = token_index
        all_proposals: list[PatchProposal] = []
        all_applied: list[AppliedPatch] = []
        recorder = RecordingRuntimeEventSink(delegate=self.event_sink)
        self.proposal_engine.event_sink = recorder
        return _RunInitialization(
            doc=doc,
            token_cursor=token_cursor,
            all_proposals=all_proposals,
            all_applied=all_applied,
            recorder=recorder,
        )

    def _build_runtime_stages(
        self,
        *,
        doc: DocumentState,
        plan: PlanIR,
        execution: StepSnapshot,
        recorder: RecordingRuntimeEventSink,
    ) -> _RuntimeStagesBuild:
        proposal = ProposalStage(
            proposal_engine=self.proposal_engine,
            trajectory_debug=self.trajectory_debug,
            event_sink=recorder,
        )
        scoring = ScoringStage(
            scorer=self.scorer_component,
            event_sink=recorder,
            attempt_state=self.attempt_state,
        )
        stages = _StageBundle(
            proposal=proposal,
            scoring=scoring,
            decision_emitter=DecisionEmitter(
                enabled=self.verbose,
                next_event_id=self._next_event_id,
                formatter=self.decision_formatter,
                debug_emitter=DebugEventEmitter(event_sink=recorder),
            ),
            global_conflict=GlobalConflictStage(selector=self.selector),
            apply=ApplyStage(),
        )
        queue: deque[RunnerEvent] = deque(
            [
                StepRequested(
                    doc=doc,
                    plan=plan,
                    execution=execution,
                    tokenizer=(
                        None
                        if self.runtime is None
                        else self.runtime.text_codec()
                    ),
                )
            ]
        )
        return _RuntimeStagesBuild(stages=stages, queue=queue)

    def _run_event_queue(
        self,
        *,
        queue: deque[RunnerEvent],
        stages: _StageBundle,
        doc: DocumentState,
        all_proposals: list[PatchProposal],
        all_applied: list[AppliedPatch],
        recorder: RecordingRuntimeEventSink,
    ) -> DocumentState:
        runner = RuntimeQueueRunner(
            queue=queue,
            stages=stages,
            doc=doc,
            all_proposals=all_proposals,
            all_applied=all_applied,
            recorder=recorder,
            on_group_begin=lambda ctx: self._make_on_group_begin(ctx, stages),
            on_candidate_scored=lambda ctx: self._make_on_candidate_scored(
                ctx, stages
            ),
            on_scores_ready=lambda ctx, scored: self._emit_group_decisions(
                ctx=ctx, scored=scored, stages=stages
            ),
            global_source_scored=lambda item, scored: _global_source_scored(
                item=item, scored_proposals=scored
            ),
        )
        return runner.run()

    def _make_on_group_begin(
        self, ctx: StepContext, stages: _StageBundle
    ) -> Callable[[tuple[int, int] | None, str, int], None]:
        def _on_group_begin(
            span: tuple[int, int] | None, op: str, total: int
        ) -> None:
            event_id = stages.decision_emitter.begin(
                context=DecisionGroupContext(
                    rule_key=_rule_label(ctx.rule),
                    rule_id_full=ctx.rule.rule_id,
                    rule_id_short=decision_logging.short_id(ctx.rule.rule_id),
                    priority=max(
                        (proposal.priority for proposal in ctx.rule.candidates),
                        default=0,
                    ),
                    repeat=(ctx.rule.fire.mode == "repeat"),
                    guard_span=(
                        ctx.guard_view.abs_start,
                        ctx.guard_view.abs_end,
                    ),
                    guard_text=ctx.guard_view.text,
                    edit_span=(ctx.edit_view.abs_start, ctx.edit_view.abs_end),
                    edit_text=ctx.edit_view.text,
                    doc_text=ctx.doc.text,
                ),
                span=span,
                op=PatchOp(op),
                n=total,
            )
            stages.decision_event_ids[(ctx.rule.rule_id, span, op)] = event_id

        return _on_group_begin

    def _make_on_candidate_scored(
        self, ctx: StepContext, stages: _StageBundle
    ) -> Callable[[ScoredProposal, int, int], None]:
        def _on_candidate_scored(
            scored: ScoredProposal, rank: int, total: int
        ) -> None:
            _ = total
            stages.decision_emitter.row(
                event_id=stages.decision_event_ids.get(
                    (
                        ctx.rule.rule_id,
                        scored.proposal.span_abs,
                        scored.proposal.op.value,
                    ),
                    0,
                ),
                rank=rank,
                candidate_name=_extract_candidate_label(scored.proposal.reason),
                score=scored.score,
                prob_ratio_to_best=scored.prob_ratio_to_best,
                is_winner=False,
                text_excerpt=self.decision_formatter.candidate_text_excerpt(
                    ctx.doc.text, scored.proposal
                ),
            )

        return _on_candidate_scored

    def _emit_group_decisions(
        self,
        *,
        ctx: StepContext,
        scored: list[ScoredProposal],
        stages: _StageBundle,
    ) -> None:
        if not self.verbose or not scored:
            return
        grouped = decision_logging.group_candidates_by_span(
            [
                ScoredCandidateRecord(
                    proposal=item.proposal,
                    score=item.score,
                    prob_ratio_to_best=item.prob_ratio_to_best,
                )
                for item in scored
            ]
        )
        for (span, op), group in grouped.items():
            key = (ctx.rule.rule_id, span, op)
            event_id = stages.decision_event_ids.get(key)
            if event_id is None:
                event_id = self._next_event_id()
            decision = self._build_decision_event(
                ctx=ctx, scored_group=group, event_id=event_id
            )
            stages.decision_emitter.end(event=decision)

    def _next_event_id(self) -> int:
        """Allocate a monotonically increasing event id for runtime telemetry.

        Purpose:
            Centralize sequencing of event identifiers so emitted runtime events
            remain stable and ordered within one run.

        Architectural role:
            Internal sequencing utility owned by ``PlanRunner``.

        Inputs:
            Uses the mutable ``_event_seq`` counter stored on the runner
            instance.

        Outputs:
            Returns the next integer id and advances the internal counter for
            the next emission.

        """
        self._event_seq += 1
        return self._event_seq

    def _build_decision_event(
        self,
        *,
        ctx: StepContext,
        scored_group: list[ScoredCandidateRecord],
        event_id: int,
    ) -> DecisionEvent:
        ranked = sorted(scored_group, key=lambda item: item.score, reverse=True)
        winner = ranked[0]
        second_score = ranked[1].score if len(ranked) > 1 else winner.score
        best_score = winner.score
        gap2 = best_score - second_score
        ratio2 = exp(second_score - best_score)
        span = winner.proposal.span_abs
        old_excerpt, new_excerpt = self.decision_formatter.apply_excerpts(
            ctx.doc.text, winner.proposal
        )
        rows: list[CandidateRow] = []
        for rank, scored in enumerate(ranked, start=1):
            proposal = scored.proposal
            rows.append(
                CandidateRow(
                    rank=rank,
                    name=_extract_candidate_label(proposal.reason),
                    score=scored.score,
                    ratio_to_best=(
                        scored.prob_ratio_to_best
                        if scored.prob_ratio_to_best is not None
                        else exp(scored.score - best_score)
                    ),
                    is_winner=(rank == 1),
                    text_excerpt=self.decision_formatter.candidate_text_excerpt(
                        ctx.doc.text, proposal
                    ),
                )
            )
        return DecisionEvent(
            event_id=event_id,
            scope="core",
            op=winner.proposal.op,
            span_start=(None if span is None else span[0]),
            span_end=(None if span is None else span[1]),
            rule_key=_rule_label(ctx.rule),
            rule_id_full=ctx.rule.rule_id,
            rule_id_short=decision_logging.short_id(ctx.rule.rule_id),
            priority=max(
                (proposal.priority for proposal in ctx.rule.candidates),
                default=0,
            ),
            repeat=(ctx.rule.fire.mode == "repeat"),
            around_excerpt=self.decision_formatter.context_excerpt(
                ctx.doc.text, span or (0, 0)
            ),
            old_excerpt=old_excerpt,
            new_excerpt=new_excerpt,
            winner_name=rows[0].name,
            winner_score=best_score,
            gap2=gap2,
            ratio2=ratio2,
            candidates=rows,
        )


def _extract_candidate_label(reason: str) -> str:
    """Return stable candidate label parsed from proposal reason text.

    Purpose:
        Implement the operation performed by this orchestration component.

    Architectural role:
        Orchestration helper inside the runtime owner that coordinates stage
        transitions and run-level mutable state.

    """
    marker = "candidate="
    if marker not in reason:
        return "unknown"
    tail = reason.split(marker, maxsplit=1)[1]
    return tail.split(maxsplit=1)[0]


def _rule_label(rule: RulePlan) -> str:
    """Return display label for rule telemetry and decision tables.

    Purpose:
        Implement the operation performed by this orchestration component.

    Architectural role:
        Orchestration helper inside the runtime owner that coordinates stage
        transitions and run-level mutable state.

    """
    return rule.name or rule.rule_id
