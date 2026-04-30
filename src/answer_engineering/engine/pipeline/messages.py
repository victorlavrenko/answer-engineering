"""Execution-stage handoff messages for the runtime pipeline.

Purpose:
    Define the immutable message envelopes exchanged between runtime stages as
    proposal, scoring, conflict resolution, and patch application progress.

Architectural role:
    Control-plane contracts for the execution pipeline. These messages are
    routed by orchestration and queue runners, but they are kept separate from
    orchestration logic so stage handoff schemas remain explicit and reusable.

Contents:
    - request messages that enter a runtime stage
    - result messages emitted by proposal, scoring, selection, and apply stages
    - the union type describing all supported runtime runner events

Invariants:
    Objects in this module describe pipeline progression and stage outputs. They
    are message envelopes, not execution-context carriers. They should remain
    immutable and should not own orchestration behavior.

"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from answer_engineering.engine.patching.proposals import PatchProposal
from answer_engineering.engine.pipeline.context import (
    StepContext,
    StepSnapshot,
)
from answer_engineering.engine.pipeline.events import Event
from answer_engineering.engine.runtime.runtime_types import (
    AppliedPatch,
    DocumentState,
)
from answer_engineering.engine.scoring.base import (
    ScoredProposal,
    ScoreResult,
)
from answer_engineering.inference.model_types import TextCodec
from answer_engineering.rules.compile.plan import PlanIR


@dataclass(frozen=True, slots=True)
class StepRequested:
    """Queue message requesting proposal generation for one prepared step.

    Purpose:
        Mark the start of local processing for a single rule and step pair.

    Architectural role:
        Control-plane message passed through the runtime queue loop.

    Inputs:
        Emitted by orchestrator stepping logic after it constructs a
        ``StepContext``.

    Outputs:
        Consumed by the proposal-stage path in ``RuntimeQueueRunner``.

    """

    doc: DocumentState
    plan: PlanIR
    execution: StepSnapshot
    tokenizer: TextCodec | None = None


@dataclass(frozen=True, slots=True)
class ProposalsReady:
    """Message carrying patch proposals produced for one step.

    Purpose:
        Preserve a typed, immutable handoff record between runtime stages
        without embedding orchestration behavior.

    Architectural role:
        Control-plane value type in the runtime pipeline. Produced by one stage
        and consumed by the next stage through orchestration.

    Inputs:
        Emitted by proposal stage after matching/guard evaluation.

    Outputs:
        Consumed by scoring stage.

    """

    ctx: StepContext
    proposals: Iterable[PatchProposal]


@dataclass(frozen=True, slots=True)
class ScoresReady:
    """Message carrying score results for one step context.

    Purpose:
        Preserve a typed, immutable handoff record between runtime stages
        without embedding orchestration behavior.

    Architectural role:
        Control-plane value type in the runtime pipeline. Produced by one stage
        and consumed by the next stage through orchestration.

    Inputs:
        Emitted by scoring stage after model/deterministic scoring.

    Outputs:
        Consumed by scored-proposal selection and winner logic.

    """

    ctx: StepContext
    result: ScoreResult


@dataclass(frozen=True, slots=True)
class ScoredProposalsReady:
    """Message carrying scored proposals for winner resolution.

    Purpose:
        Preserve a typed, immutable handoff record between runtime stages
        without embedding orchestration behavior.

    Architectural role:
        Control-plane value type in the runtime pipeline. Produced by one stage
        and consumed by the next stage through orchestration.

    Inputs:
        Built from scoring outputs in orchestration flow.

    Outputs:
        Consumed by rule-winner and global-conflict resolution stages.

    """

    scored: Iterable[ScoredProposal]


@dataclass(frozen=True, slots=True)
class RuleWinnerReady:
    """Message carrying per-rule winning proposal selection.

    Purpose:
        Preserve a typed, immutable handoff record between runtime stages
        without embedding orchestration behavior.

    Architectural role:
        Control-plane value type in the runtime pipeline. Produced by one stage
        and consumed by the next stage through orchestration.

    Inputs:
        Built by the upstream runtime stage from the current document snapshot,
        step context, or scored proposal set.

    Outputs:
        Consumed by the downstream runtime stage through the queue runner.

    """

    ctx: StepContext
    winner: PatchProposal | None


@dataclass(frozen=True, slots=True)
class GlobalWinnersReady:
    """Message carrying winners after cross-rule conflict resolution.

    Purpose:
        Preserve a typed, immutable handoff record between runtime stages
        without embedding orchestration behavior.

    Architectural role:
        Control-plane value type in the runtime pipeline. Produced by one stage
        and consumed by the next stage through orchestration.

    Inputs:
        Built by the upstream runtime stage from the current document snapshot,
        step context, or scored proposal set.

    Outputs:
        Consumed by the downstream runtime stage through the queue runner.

    """

    winners: Iterable[PatchProposal]


@dataclass(frozen=True, slots=True)
class AcceptedPatchesReady:
    """Message carrying accepted patches prior to application.

    Purpose:
        Preserve a typed, immutable handoff record between runtime stages
        without embedding orchestration behavior.

    Architectural role:
        Control-plane value type in the runtime pipeline. Produced by one stage
        and consumed by the next stage through orchestration.

    Inputs:
        Built by the upstream runtime stage from the current document snapshot,
        step context, or scored proposal set.

    Outputs:
        Consumed by the downstream runtime stage through the queue runner.

    """

    accepted: Iterable[PatchProposal]


@dataclass(frozen=True, slots=True)
class PatchAppliedReady:
    """Message with updated document and emitted events after apply phase.

    Purpose:
        Preserve a typed, immutable handoff record between runtime stages
        without embedding orchestration behavior.

    Architectural role:
        Control-plane value type in the runtime pipeline. Produced by one stage
        and consumed by the next stage through orchestration.

    Inputs:
        Built by the upstream runtime stage from the current document snapshot,
        step context, or scored proposal set.

    Outputs:
        Consumed by the downstream runtime stage through the queue runner.

    """

    doc: DocumentState
    applied_patches: Iterable[AppliedPatch]
    emitted_events: Iterable[Event]


@dataclass(frozen=True, slots=True)
class RunCompleted:
    """Terminal message carrying final document and applied patch list.

    Purpose:
        Preserve a typed, immutable handoff record between runtime stages
        without embedding orchestration behavior.

    Architectural role:
        Control-plane value type in the runtime pipeline. Produced by one stage
        and consumed by the next stage through orchestration.

    Inputs:
        Built by the upstream runtime stage from the current document snapshot,
        step context, or scored proposal set.

    Outputs:
        Consumed by the downstream runtime stage through the queue runner.

    """

    doc: DocumentState
    applied_patches: Iterable[AppliedPatch]


type RunnerEvent = (
    StepRequested
    | ProposalsReady
    | ScoresReady
    | ScoredProposalsReady
    | RuleWinnerReady
    | GlobalWinnersReady
    | AcceptedPatchesReady
    | PatchAppliedReady
    | RunCompleted
)
