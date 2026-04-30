"""Scoring protocols and canonical result containers.

Purpose:
    Define the stable handoff between proposal scoring implementations and the
    downstream selection stages.

Architectural role:
    Core scoring boundary in the engine runtime.

Inputs:
    StepContext values for the current decoding step and PatchProposal batches
    assembled by stages.

Outputs:
    Protocols for scorer implementations and immutable score containers consumed
    by rule-winner and conflict-resolution logic.

Ownership:
    Owned by the engine scoring boundary.

Non-ownership:
    Does not decide winners and does not resolve cross-proposal conflicts.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from answer_engineering.engine.patching.proposals import (
    PatchProposal,
)
from answer_engineering.engine.pipeline.context import (
    StepContext,
)
from answer_engineering.inference.model_types import (
    TokenGenerationRuntime,
)


@dataclass(frozen=True, slots=True)
class ScoredProposal:
    """Immutable score attached to one patch proposal.

    Purpose:
        Pair a generated patch proposal with the scalar score and optional ratio
        used to compare it against alternatives.

    Architectural role:
        Canonical handoff value from scoring implementations to rule-winner and
        conflict-resolution selection.

    Inputs (architectural provenance):
        Constructed by a scorer after evaluating proposals for a step context.

    Outputs (downstream usage):
        Consumed by selectors, conflict resolution, telemetry formatting, and
        tests.

    Invariants/constraints:
        `prob_ratio_to_best` is optional because deterministic fallback scoring
        and model-backed probability scoring do not always expose the same
        diagnostics.

    """

    proposal: PatchProposal
    score: float
    prob_ratio_to_best: float | None = None


@dataclass(frozen=True, slots=True)
class ScoringDiagnostics:
    """Diagnostics describing how one scoring batch was produced.

    Purpose:
        Record whether a scorer used model-backed evaluation or fallback logic
        and how many runtime scoring calls were spent on the batch.

    Architectural role:
        Immutable diagnostics value object in the scoring boundary.

    Inputs:
        Execution facts collected by a scorer while producing a ScoreResult.

    Outputs:
        Batch-level scoring metadata consumed by stages, telemetry, and tests.

    Invariants:
        `num_calls` counts runtime scoring invocations for this batch only.
        `model_scored` is true only when model-backed scoring actually ran.

    Ownership:
        Owned by the engine scoring boundary.

    """

    model_scored: bool
    num_calls: int


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Canonical result payload returned by scorer implementations.

    Purpose:
        Bundle scored proposals with optional diagnostics from one scoring pass
        so stages and selectors can consume one stable object.

    Architectural role:
        Primary return value of the runtime scoring boundary.

    Inputs (architectural provenance):
        Constructed by scorer implementations after evaluating proposal
        alternatives for one step context.

    Outputs (downstream usage):
        Consumed by scoring stages, local winner selection, and later global
        conflict resolution.

    Invariants/constraints:
        Every item in `scored` must correspond to the same scoring pass and step
        context.

    """

    scored: list[ScoredProposal]
    diagnostics: ScoringDiagnostics | None = None


@runtime_checkable
class ConfigurableScorer(Protocol):
    """Protocol for scorers that can be bound to a runtime before scoring.

    Purpose:
        Define the configuration step used when a scorer must be told which
        runtime to use and whether model-backed scoring is mandatory.

    Architectural role:
        Optional runtime-binding extension of the scoring protocol surface.

    Inputs (architectural provenance):
        Called by orchestration or session setup before stages start scoring
        proposals.

    Outputs (downstream usage):
        Mutates scorer configuration so later `score()` calls use the intended
        runtime policy.

    Invariants/constraints:
        Configuration must happen before dependent scoring calls and must leave
        the scorer internally consistent.

    """

    def configure(
        self,
        *,
        runtime: TokenGenerationRuntime | None,
        require_model_scoring: bool,
    ) -> None:
        """Configure the scorer with the runtime to use and model-score policy.

        Purpose:
            Bind scorer instances to the runtime resources and scoring mode
            required for a generation run.

        Architectural role:
            Lifecycle hook between orchestration setup and concrete scorer
            execution.

        Inputs (architectural provenance):
            Receives the active runtime plus the policy flag indicating whether
            model scoring is allowed or required.

        Outputs (downstream usage):
            Mutates scorer-local configuration that later `score` calls consult.

        Invariants/constraints:
            Scorers should be configured before use. Implementations should keep
            runtime binding local to scoring and avoid leaking backend details
            into selection or proposal layers.

        """
        raise NotImplementedError


class Scorer(Protocol):
    """Protocol for components that score patch proposals.

    Purpose:
        Define the stable scoring operation used by runtime stages regardless of
        the concrete scoring strategy.

    Architectural role:
        Proposal-to-selection boundary in the engine pipeline.

    Inputs (architectural provenance):
        Implementations receive the current `StepContext` and the proposals
        produced by the proposal stage.

    Outputs (downstream usage):
        Returns a `ScoreResult` consumed by local winner selection and global
        conflict resolution.

    Invariants/constraints:
        Implementations must preserve proposal identity and return scores that
        are comparable within the provided batch.

    """

    def score(
        self, ctx: StepContext, proposals: list[PatchProposal]
    ) -> ScoreResult:
        """Score proposals and return canonical results.

        Purpose:
            Assign comparable scoring records to proposals emitted during one
            runtime step.

        Architectural role:
            Abstract scoring boundary between proposal generation and selection.

        Inputs (architectural provenance):
            Receives a `StepContext` from orchestration and an iterable of
            proposal objects from proposal stages.

        Outputs (downstream usage):
            Returns `ProposalScore` values consumed by local selection and
            global conflict resolution.

        Invariants/constraints:
            Implementations should not apply patches or mutate the step
            document. They only evaluate proposals and attach scoring evidence.

        """
        raise NotImplementedError
