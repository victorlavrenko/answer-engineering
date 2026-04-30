"""Runtime control-plane event loop and queue dispatch.

Purpose:
    Define the deterministic queue runner and stage bundle contracts that route
    runtime control-plane messages across proposal, scoring, selection, and
    apply stages.

Architectural role:
    Control-plane dispatch seam above concrete runtime stages.

Architectural direction:
    Keep queue dispatch explicit and deterministic while clarifying the final
    ownership boundary between event-loop control and stage behavior.

Why this matters:
    Queue runners can silently accumulate framework-like responsibilities that
    spread control-plane knowledge across unrelated modules.

What better would look like:
    Dispatch logic remains local, and stage ownership can be reasoned about
    without requiring event-loop internals knowledge in many places.

How improvement can be recognized:
    - Clearer separation between dispatch mechanics and stage semantics
    - Fewer modules depending on event-loop-specific conventions
    - Easier explanation of queue item lifecycle and ownership

Open constraint:
    The exact control-plane boundary should remain responsive to runtime
    evolution and not be treated as final.

"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from functools import singledispatchmethod
from typing import Protocol

from answer_engineering.engine.orchestration.stages.apply import (
    ApplyStage,
)
from answer_engineering.engine.orchestration.stages.global_conflict import (
    GlobalConflictStage,
)
from answer_engineering.engine.orchestration.stages.proposal import (
    ProposalStage,
)
from answer_engineering.engine.orchestration.stages.scoring import (
    ScoringStage,
)
from answer_engineering.engine.patching.proposals import PatchProposal
from answer_engineering.engine.pipeline.context import (
    StepContext,
)
from answer_engineering.engine.pipeline.messages import (
    AcceptedPatchesReady,
    GlobalWinnersReady,
    PatchAppliedReady,
    ProposalsReady,
    RunnerEvent,
    ScoredProposalsReady,
    ScoresReady,
    StepRequested,
)
from answer_engineering.engine.runtime.runtime_types import (
    AppliedPatch,
    DocumentState,
)
from answer_engineering.engine.scoring.base import ScoredProposal
from answer_engineering.engine.telemetry.events.event_sink import (
    RecordingRuntimeEventSink,
)


class RuntimeStageBundle(Protocol):
    """Protocol grouping stage collaborators used by the runtime queue loop.

    Purpose:
        Group the stage collaborators that the queue runner dispatches through
        one explicit orchestration dependency.

    Architectural role:
        Dependency bundle interface consumed by ``RuntimeQueueRunner`` so the
        dispatcher can remain decoupled from concrete stage construction.

    Inputs:
        Consumes run-owned state, queue items, or configured collaborators
        prepared by orchestration.

    Outputs:
        Returns or emits values that are consumed by downstream runtime stages,
        telemetry, or final result assembly.

    """

    proposal: ProposalStage
    scoring: ScoringStage
    global_conflict: GlobalConflictStage
    apply: ApplyStage


class RuntimeQueueRunner:
    """Stateful dispatcher for orchestrator queue events.

    Purpose:
        Own the FIFO queue loop that dispatches runtime stage messages,
        accumulates run-level state, and emits follow-up events.

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

    def __init__(
        self,
        *,
        queue: deque[RunnerEvent],
        stages: RuntimeStageBundle,
        doc: DocumentState,
        all_proposals: list[PatchProposal],
        all_applied: list[AppliedPatch],
        recorder: RecordingRuntimeEventSink,
        on_group_begin: Callable[
            [StepContext], Callable[[tuple[int, int] | None, str, int], None]
        ],
        on_candidate_scored: Callable[
            [StepContext], Callable[[ScoredProposal, int, int], None]
        ],
        on_scores_ready: Callable[[StepContext, list[ScoredProposal]], None],
        global_source_scored: Callable[
            [GlobalWinnersReady, list[ScoredProposal]], list[ScoredProposal]
        ],
    ) -> None:
        """Bind queue-owned collaborators and initialize run-scoped state.

        Purpose:
            Assemble the orchestration collaborators required to drive one
            runtime queue from proposed edits through scoring, selection,
            application, and telemetry.

        Architectural role:
            Constructor boundary for the decode-time control plane. It wires
            together runtime context, proposal providers, scoring, selection,
            patching, and event recording without executing the loop yet.

        Inputs (architectural provenance):
            Receives the current step context plus queue-stage collaborators
            built by runtime/session setup.

        Outputs (downstream usage):
            Stores run-scoped state and collaborator references used by `run`
            and its stage handlers.

        Invariants/constraints:
            The runner owns orchestration state for one queue execution. It
            should not mutate caller configuration objects or perform model
            generation during construction.

        """
        self.queue = queue
        self.stages = stages
        self.doc = doc
        self.all_proposals = all_proposals
        self.all_applied = all_applied
        self.recorder = recorder
        self.on_group_begin = on_group_begin
        self.on_candidate_scored = on_candidate_scored
        self.on_scores_ready = on_scores_ready
        self.global_source_scored = global_source_scored
        self.scored_proposals: list[ScoredProposal] = []

    def run(self) -> DocumentState:
        """Drain runner events in deterministic FIFO order.

        Purpose:
            Drain queued runtime events until the current orchestration cycle
            reaches a stable point.

        Architectural role:
            Orchestration helper inside the runtime owner that coordinates stage
            transitions and run-level mutable state.

        Inputs:
            Consumes runner-owned queue items, document state, and stage outputs
            produced by the current orchestration cycle.

        Outputs:
            Returns or emits values that are consumed by downstream runtime
            stages, telemetry, or final result assembly.

        """
        while self.queue:
            item = self.queue.popleft()
            self._dispatch(item)
        return self.doc

    @singledispatchmethod
    def _dispatch(self, item: RunnerEvent) -> None:
        """Dispatch one queued runtime message to the handler registered for.

        Purpose:
            Act as the control-plane switchboard for the FIFO runner queue and
            reject unsupported event types early.

        Architectural role:
            Central event-loop dispatch point inside ``RuntimeQueueRunner``.

        Inputs (architectural provenance):
            Receives ``RunnerEvent`` instances popped from the orchestrator
            queue.

        Outputs (downstream usage):
            Routes control to the corresponding typed handler, which may mutate
            runner state and append follow-up events.

        Invariants/constraints:
            Dispatch is type-based and deterministic; unknown event classes
            raise ``TypeError`` rather than being ignored.

        """
        raise TypeError(f"Unsupported runner event: {type(item).__name__}")

    @_dispatch.register
    def _(self, item: StepRequested) -> None:
        """Handle the start of one step evaluation by delegating proposal.

        Purpose:
            Provide the typed ``singledispatch`` branch for one concrete queue
            event.

        Architectural role:
            Thin adapter layer between generic queue dispatch and the named
            handler methods that own the real work.

        Inputs:
            Receives one specific runtime message subclass popped from the
            orchestrator queue.

        Outputs:
            Forwards control to the matching handler, preserving deterministic
            queue progression and state updates.

        """
        self._handle_step_requested(item)

    @_dispatch.register
    def _(self, item: ProposalsReady) -> None:
        """Handle freshly generated proposals by recording them and scheduling.

        Purpose:
            Provide the typed ``singledispatch`` branch for one concrete queue
            event.

        Architectural role:
            Thin adapter layer between generic queue dispatch and the named
            handler methods that own the real work.

        Inputs:
            Receives one specific runtime message subclass popped from the
            orchestrator queue.

        Outputs:
            Forwards control to the matching handler, preserving deterministic
            queue progression and state updates.

        """
        self._handle_proposals_ready(item)

    @_dispatch.register
    def _(self, item: ScoresReady) -> None:
        """Handle completed scoring output by accumulating scored proposals and.

        Purpose:
            Provide the typed ``singledispatch`` branch for one concrete queue
            event.

        Architectural role:
            Thin adapter layer between generic queue dispatch and the named
            handler methods that own the real work.

        Inputs:
            Receives one specific runtime message subclass popped from the
            orchestrator queue.

        Outputs:
            Forwards control to the matching handler, preserving deterministic
            queue progression and state updates.

        """
        self._handle_scores_ready(item)

    @_dispatch.register
    def _(self, item: GlobalWinnersReady) -> None:
        """Handle global-winner candidates once local proposal/scoring work has.

        Purpose:
            Provide the typed ``singledispatch`` branch for one concrete queue
            event.

        Architectural role:
            Thin adapter layer between generic queue dispatch and the named
            handler methods that own the real work.

        Inputs:
            Receives one specific runtime message subclass popped from the
            orchestrator queue.

        Outputs:
            Forwards control to the matching handler, preserving deterministic
            queue progression and state updates.

        """
        self._handle_global_winners_ready(item)

    @_dispatch.register
    def _(self, item: AcceptedPatchesReady) -> None:
        """Handle globally accepted patches by scheduling the apply stage.

        Purpose:
            Provide the typed ``singledispatch`` branch for one concrete queue
            event.

        Architectural role:
            Thin adapter layer between generic queue dispatch and the named
            handler methods that own the real work.

        Inputs:
            Receives one specific runtime message subclass popped from the
            orchestrator queue.

        Outputs:
            Forwards control to the matching handler, preserving deterministic
            queue progression and state updates.

        """
        self._handle_accepted_patches_ready(item)

    @_dispatch.register
    def _(self, item: PatchAppliedReady) -> None:
        """Handle post-apply results by advancing runner document state and.

        Purpose:
            Provide the typed ``singledispatch`` branch for one concrete queue
            event.

        Architectural role:
            Thin adapter layer between generic queue dispatch and the named
            handler methods that own the real work.

        Inputs:
            Receives one specific runtime message subclass popped from the
            orchestrator queue.

        Outputs:
            Forwards control to the matching handler, preserving deterministic
            queue progression and state updates.

        """
        self._handle_patch_applied_ready(item)

    def _handle_step_requested(self, item: StepRequested) -> None:
        """Generate proposals for one requested step and seed the queue with.

        Purpose:
            Delegate ``StepRequested`` to the proposal stage and then enqueue
            the placeholder global-winner message used to close the group.

        Architectural role:
            Step-to-proposal transition inside the runtime queue loop.

        Inputs:
            Called from dispatch when orchestration wants proposals for a
            prepared ``StepContext``.

        Outputs:
            Extends the queue with proposal-stage output and appends an empty
            ``GlobalWinnersReady`` marker to trigger downstream conflict
            resolution.

        """
        self.queue.extend(self.stages.proposal.handle(item))
        self.queue.append(GlobalWinnersReady(winners=[]))

    def _handle_proposals_ready(self, item: ProposalsReady) -> None:
        """Record proposal output and schedule scoring for the associated step.

        Purpose:
            Preserve all generated proposals for run-level bookkeeping and
            invoke the scoring stage with the per-group telemetry callbacks.

        Architectural role:
            Proposal-to-scoring transition inside the queue runner.

        Inputs:
            Receives ``ProposalsReady`` emitted by the proposal stage.

        Outputs:
            Appends the resulting ``ScoresReady`` message to the queue and
            extends the run-level proposal history.

        """
        self.all_proposals.extend(item.proposals)
        self.queue.append(
            self.stages.scoring.handle(
                item,
                on_group_begin=self.on_group_begin(item.ctx),
                on_candidate_scored=self.on_candidate_scored(item.ctx),
            )
        )

    def _handle_scores_ready(self, item: ScoresReady) -> None:
        """Merge one scoring batch into runner state and publish score.

        Purpose:
            Accumulate scored proposals for later global conflict resolution
            while also invoking the external ``on_scores_ready`` callback.

        Architectural role:
            Scoring-result collector within the runtime event loop.

        Inputs:
            Receives ``ScoresReady`` from the scoring stage for one step
            context.

        Outputs:
            Updates ``scored_proposals`` and emits score information to
            downstream callbacks used by telemetry/reporting layers.

        """
        self.scored_proposals.extend(item.result.scored)
        self.on_scores_ready(item.ctx, item.result.scored)

    def _handle_global_winners_ready(self, item: GlobalWinnersReady) -> None:
        """Resolve globally conflicting scored proposals once no more local.

        Purpose:
            Delay global conflict resolution until proposal and scoring messages
            have drained, then run the global-conflict stage on the accumulated
            scores.

        Architectural role:
            Queue barrier between local per-step work and cross-step patch
            acceptance.

        Inputs:
            Called for ``GlobalWinnersReady`` markers emitted after step
            processing.

        Outputs:
            Either requeues the marker until local work is finished or appends
            the accepted-patches message produced by global conflict resolution.

        """
        if any(
            isinstance(queued, ProposalsReady | ScoresReady)
            for queued in self.queue
        ):
            self.queue.append(item)
            return
        source_scored = self.global_source_scored(item, self.scored_proposals)
        conflict_result = self.stages.global_conflict.handle(
            ScoredProposalsReady(scored=source_scored)
        )
        self.scored_proposals = []
        for rejected_event in conflict_result.rejected_events:
            self.recorder.emit(rejected_event)
        self.queue.append(conflict_result.accepted)

    def _handle_accepted_patches_ready(
        self, item: AcceptedPatchesReady
    ) -> None:
        self.queue.append(
            self.stages.apply.handle(
                item,
                doc=self.doc,
                applied_count=len(self.all_applied),
            )
        )

    def _handle_patch_applied_ready(self, item: PatchAppliedReady) -> None:
        """Commit apply-stage output back into runner-owned document and event.

        Purpose:
            Replace the active document snapshot, extend the applied-patch list,
            and forward emitted events to the recorder.

        Architectural role:
            Final state-commit step of one queue cycle after patch application.

        Inputs:
            Receives ``PatchAppliedReady`` emitted by the apply stage.

        Outputs:
            Updates runner state that will be observed by subsequent queue
            iterations and final run result construction.

        """
        self.doc = item.doc
        self.all_applied.extend(item.applied_patches)
        for emitted_event in item.emitted_events:
            self.recorder.emit(emitted_event)
