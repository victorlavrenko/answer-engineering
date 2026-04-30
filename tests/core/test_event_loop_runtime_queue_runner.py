from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import cast

from answer_engineering.engine.orchestration.event_loop import (
    RuntimeQueueRunner,
    RuntimeStageBundle,
)
from answer_engineering.engine.patching.proposals import PatchProposal
from answer_engineering.engine.pipeline.context import (
    StepContext,
)
from answer_engineering.engine.pipeline.events import (
    DebugEvent,
)
from answer_engineering.engine.pipeline.messages import (
    AcceptedPatchesReady,
    GlobalWinnersReady,
    PatchAppliedReady,
    ProposalsReady,
    RunnerEvent,
    ScoresReady,
    StepRequested,
)
from answer_engineering.engine.runtime.runtime_types import (
    AppliedPatch,
    DocumentState,
    PatchOp,
)
from answer_engineering.engine.scoring.base import (
    ScoredProposal,
    ScoreResult,
)
from answer_engineering.engine.telemetry.events.event_sink import (
    RecordingRuntimeEventSink,
)
from answer_engineering.rules.compile.plan import PlanIR
from tests._support.core_helpers import create_step_snapshot


@dataclass(slots=True)
class _FakeProposalStage:
    proposal: PatchProposal
    ctx: StepContext

    def handle(self, event: StepRequested) -> list[ProposalsReady]:
        del event
        return [ProposalsReady(ctx=self.ctx, proposals=[self.proposal])]


@dataclass(slots=True)
class _FakeScoringStage:
    scored: ScoredProposal
    ctx: StepContext

    def handle(
        self,
        event: ProposalsReady,
        *,
        on_group_begin: object,
        on_candidate_scored: object,
    ) -> ScoresReady:
        del event
        del on_group_begin
        del on_candidate_scored
        return ScoresReady(
            ctx=self.ctx, result=ScoreResult(scored=[self.scored])
        )


@dataclass(frozen=True, slots=True)
class _FakeConflictResult:
    accepted: AcceptedPatchesReady
    rejected_events: list[DebugEvent]


@dataclass(slots=True)
class _FakeGlobalConflictStage:
    proposal: PatchProposal

    def handle(self, event: object) -> _FakeConflictResult:
        del event
        return _FakeConflictResult(
            accepted=AcceptedPatchesReady(accepted=[self.proposal]),
            rejected_events=[DebugEvent(msg="rejected")],
        )


@dataclass(slots=True)
class _FakeApplyStage:
    proposal: PatchProposal

    def handle(
        self,
        event: AcceptedPatchesReady,
        *,
        doc: DocumentState,
        applied_count: int,
    ) -> PatchAppliedReady:
        del event
        del applied_count
        applied = AppliedPatch(
            patch_id="p1", proposal=self.proposal, new_version_id="v2"
        )
        return PatchAppliedReady(
            doc=DocumentState(text=f"{doc.text}!", version_id="v2"),
            applied_patches=[applied],
            emitted_events=[DebugEvent(msg="applied")],
        )


@dataclass(slots=True)
class _FakeStages:
    proposal: _FakeProposalStage
    scoring: _FakeScoringStage
    global_conflict: _FakeGlobalConflictStage
    apply: _FakeApplyStage


def test_runtime_queue_runner_handles_step_to_apply_flow() -> None:
    proposal = PatchProposal(
        op=PatchOp.REPLACE,
        span_abs=(0, 1),
        payload="b",
        base_version_id="v1",
        rule_id="r1",
        score=1.0,
        reason="candidate=a",
    )
    scored = ScoredProposal(proposal=proposal, score=1.0)
    ctx = cast(StepContext, object())
    stages = cast(
        RuntimeStageBundle,
        _FakeStages(
            proposal=_FakeProposalStage(proposal=proposal, ctx=ctx),
            scoring=_FakeScoringStage(scored=scored, ctx=ctx),
            global_conflict=_FakeGlobalConflictStage(proposal=proposal),
            apply=_FakeApplyStage(proposal=proposal),
        ),
    )
    queue = cast(
        deque[RunnerEvent],
        deque(
            [
                StepRequested(
                    doc=DocumentState(text="a", version_id="v1"),
                    plan=PlanIR(rules=()),
                    execution=create_step_snapshot(
                        snapshot_text="a", token_index=0
                    ),
                )
            ]
        ),
    )
    all_proposals: list[PatchProposal] = []
    all_applied: list[AppliedPatch] = []
    recorder = RecordingRuntimeEventSink()
    seen_scores: list[ScoredProposal] = []

    runner = RuntimeQueueRunner(
        queue=queue,
        stages=stages,
        doc=DocumentState(text="a", version_id="v1"),
        all_proposals=all_proposals,
        all_applied=all_applied,
        recorder=recorder,
        on_group_begin=lambda _ctx: lambda _span, _op, _n: None,
        on_candidate_scored=lambda _ctx: lambda _scored, _rank, _n: None,
        on_scores_ready=lambda _ctx, scored_items: seen_scores.extend(
            scored_items
        ),
        global_source_scored=lambda _gw, items: items,
    )

    final_doc = runner.run()

    assert final_doc.text == "a!"
    assert len(all_proposals) == 1
    assert len(seen_scores) == 1
    assert len(all_applied) == 1
    assert [
        event.msg for event in recorder.events if isinstance(event, DebugEvent)
    ] == [
        "rejected",
        "applied",
    ]


def test_runtime_requeues_global_winners_until_scores_drained() -> None:
    proposal = PatchProposal(
        op=PatchOp.REPLACE,
        span_abs=(0, 1),
        payload="b",
        base_version_id="v1",
        rule_id="r1",
        score=1.0,
        reason="candidate=a",
    )
    scored = ScoredProposal(proposal=proposal, score=1.0)
    ctx = cast(StepContext, object())
    stages = cast(
        RuntimeStageBundle,
        _FakeStages(
            proposal=_FakeProposalStage(proposal=proposal, ctx=ctx),
            scoring=_FakeScoringStage(scored=scored, ctx=ctx),
            global_conflict=_FakeGlobalConflictStage(proposal=proposal),
            apply=_FakeApplyStage(proposal=proposal),
        ),
    )
    queue = cast(
        deque[RunnerEvent],
        deque(
            [
                GlobalWinnersReady(winners=[]),
                ScoresReady(ctx=ctx, result=ScoreResult(scored=[scored])),
            ]
        ),
    )
    runner = RuntimeQueueRunner(
        queue=queue,
        stages=stages,
        doc=DocumentState(text="a", version_id="v1"),
        all_proposals=[],
        all_applied=[],
        recorder=RecordingRuntimeEventSink(),
        on_group_begin=lambda _ctx: lambda _span, _op, _n: None,
        on_candidate_scored=lambda _ctx: lambda _scored, _rank, _n: None,
        on_scores_ready=lambda _ctx, _scored: None,
        global_source_scored=lambda _gw, items: items,
    )

    final_doc = runner.run()

    assert final_doc.text == "a!"
