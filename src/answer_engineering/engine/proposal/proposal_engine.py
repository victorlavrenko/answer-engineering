"""Proposal planner and final proposal canonicalization.

Purpose:
    Select a candidate provider for one rule evaluation, apply shared proposal
    semantics, and freeze normalized patch proposals for downstream use.

Architectural role:
    Main proposal-layer entry module between step-scoped orchestration context
    and scoring/selection.

Current architecture notes:
    This module owns provider selection, proposal telemetry, and the last
    proposal-freezing step. It also depends on shared precheck/proposal logic
    that still exposes more internals than the eventual closed boundary likely
    should.

Architectural TODO:
    Keep provider responsibilities narrow and move more proposal-internal
    helpers behind the planner boundary so external callers depend only on the
    planner-facing API.

"""

from __future__ import annotations

from dataclasses import dataclass, field

from answer_engineering.engine.patching import patch_canonical
from answer_engineering.engine.patching.proposals import PatchProposal
from answer_engineering.engine.pipeline.context import (
    StepContext,
)
from answer_engineering.engine.pipeline.events import (
    GuardConditionEvaluated,
    PatchSkipped,
    ProposalsGenerated,
    RuleEvaluationStarted,
    ViewMatchSettings,
    ViewProduced,
)
from answer_engineering.engine.proposal.candidates.avoid import (
    AvoidCandidatesProvider,
)
from answer_engineering.engine.proposal.candidates.base import (
    CandidateProvider,
    CandidateRequest,
    StaticCandidatesProvider,
)
from answer_engineering.engine.proposal.proposal_logic import (
    GenerationPrecheck,
    StandardProposalGenerator,
)
from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    PatchOp,
)
from answer_engineering.engine.telemetry.events.event_sink import (
    DebugEventEmitter,
    NullRuntimeEventSink,
    RuntimeEventSink,
)
from answer_engineering.inference.model_types import (
    GenerationRuntimeProtocol,
)


@dataclass(frozen=True, slots=True)
class ProposalInput:
    """Normalized planner input for one proposal-generation call.

    Purpose:
        Carry the active StepContext through planner internals as one canonical
        request object.

    Architectural role:
        Thin input record at the public edge of ProposalPlanner.

    Inputs (architectural provenance):
        Built by ProposalPlanner.generate from the current orchestration step.

    Outputs (downstream usage):
        Consumed by provider selection, telemetry emission, and shared proposal
        logic.

    """

    ctx: StepContext


@dataclass(frozen=True, slots=True)
class ProposalOutcome:
    """Immutable result of one proposal-planner call.

    Purpose:
        Carry the finalized proposal tuple and optional skip reason produced for
        one rule evaluation.

    Architectural role:
        Proposal-stage handoff from `ProposalPlanner` to orchestration and
        scoring.

    Inputs (architectural provenance):
        Constructed after candidate providers and shared proposal generation
        finish for the current `StepContext`.

    Outputs (downstream usage):
        `proposals` are consumed by scoring. `skip_reason` explains
        planner-level short circuits to telemetry and debug output.

    Invariants/constraints:
        Proposals are stored as a tuple so downstream stages receive a stable
        snapshot rather than a mutable planner-owned list.

    """

    proposals: tuple[PatchProposal, ...]
    skip_reason: str | None = None


@dataclass(slots=True)
class ProposalPlanner:
    """Proposal-boundary orchestrator for provider selection and proposal.

    Purpose:
        Coordinate precheck, candidate-provider routing, proposal telemetry, and
        canonical patch-proposal freezing for one ``StepContext``.

    Architectural role:
        Main API object of ``engine.proposal`` used by proposal stages and
        orchestrator wiring.

    Inputs:
        Receives per-rule ``StepContext`` plus runtime attachment via
        ``configure_runtime``.

    Outputs:
        Returns normalized ``PatchProposal`` values for scoring/selection/apply
        and emits proposal-stage events.

    State:
        Holds provider instances plus event/debug collaborators across calls.

    Todo:
        Target:
            Keep planner as the only required proposal entry point while moving
            provider-specific runtime adaptation behind narrower internal seams.

        Boundary note:
            The planner still reaches into concrete avoid-provider runtime state
            during ``configure_runtime``.

    """

    candidates_providers: tuple[CandidateProvider, ...] | None = None
    event_sink: RuntimeEventSink = field(default_factory=NullRuntimeEventSink)
    debug_emitter: DebugEventEmitter = field(default_factory=DebugEventEmitter)
    generator: StandardProposalGenerator = field(
        default_factory=StandardProposalGenerator
    )

    def __post_init__(self) -> None:
        """Initialize provider collaborators after dataclass construction.

        Purpose:
            Ensure the planner always has configured candidate providers, even
            when it is created before a generation runtime is attached.

        """
        self.configure_runtime(runtime=None, trajectory_debug=False)

    def reset_run_state(self) -> None:
        """Reset planner-owned per-run state before a new orchestrator run.

        Purpose:
            Provide the lifecycle hook invoked between runs.

        Current behavior:
            This is intentionally a no-op because avoid-provider runtime state
            is preserved across reruns rather than treated as planner-owned
            run-local state.

        Todo:
            Target:
                Reset only genuinely run-scoped planner state here once
                provider/runtime ownership is separated more cleanly.

        """
        return

    def _debug(self, ctx: StepContext, msg: str) -> None:
        """Emit a planner debug message only when trajectory debugging is."""
        if not ctx.trajectory_debug:
            return
        self.debug_emitter.emit(msg)

    def configure_runtime(
        self,
        *,
        runtime: GenerationRuntimeProtocol | None = None,
        trajectory_debug: bool = False,
    ) -> None:
        """Install or refresh provider state for the active generation runtime.

        Purpose:
            Keep static and avoid providers synchronized with the currently
            attached runtime and debug-emission settings.

        Architectural role:
            Runtime-configuration entry point for the proposal subsystem.

        Inputs (architectural provenance):
            Called during planner construction and again when orchestration
            attaches or replaces the generation runtime.

        Outputs (downstream usage):
            Updates or rebuilds the provider tuple used by later generate calls.

        Architectural TODO:
            Reduce direct knowledge of concrete provider internals so runtime
            attachment does not need to reach through provider-specific state.

        """
        existing_avoid = None
        if self.candidates_providers is not None:
            existing_avoid = next(
                (
                    provider
                    for provider in self.candidates_providers
                    if isinstance(provider, AvoidCandidatesProvider)
                ),
                None,
            )
        if existing_avoid is not None:
            existing_avoid.runtime.generation_runtime = runtime
            existing_avoid.runtime.trajectory_debug = trajectory_debug
            existing_avoid.runtime.debug_emitter = self.debug_emitter
            return
        self.candidates_providers = (
            StaticCandidatesProvider(("replace:", "after:", "force:")),
            AvoidCandidatesProvider(
                runtime=runtime,
                trajectory_debug=trajectory_debug,
                debug_emitter=self.debug_emitter,
            ),
        )

    def generate(self, ctx: StepContext) -> list[PatchProposal]:
        """Generate canonical proposals for one rule evaluation context.

        Purpose:
            Run proposal precheck, select a candidate provider, emit proposal
            telemetry, and freeze normalized proposals for downstream stages.

        Architectural role:
            Main per-rule execution method on the proposal boundary.

        Inputs (architectural provenance):
            Receives one StepContext built by proposal-stage orchestration.

        Outputs (downstream usage):
            Returns PatchProposal objects consumed by scoring, conflict
            resolution, and apply stages.

        Architectural TODO:
            Replace the current proposal-shaped probing seam with a narrower
            adapter so the planner no longer exposes provider/precheck coupling
            across subsystem boundaries.

        """
        if not self.candidates_providers:
            raise RuntimeError("ProposalPlanner runtime not configured")
        request = ProposalInput(ctx=ctx)
        self.debug_emitter.event_sink = (
            request.ctx.event_sink or self.event_sink
        )
        self.event_sink.emit(
            RuleEvaluationStarted(
                rule_id=request.ctx.rule.rule_id,
                doc_version_id=request.ctx.doc.version_id,
                scope_spec=request.ctx.rule.effective_guard_scope(),
            )
        )
        self.event_sink.emit(
            ViewProduced(
                rule_id=ctx.rule.rule_id,
                base_version_id=ctx.guard_view.base_version_id,
                abs_start=ctx.guard_view.abs_start,
                abs_end=ctx.guard_view.abs_end,
                match_settings=ViewMatchSettings(
                    casefold=ctx.rule.effective_guard_scope().casefold
                ),
            )
        )

        provider = next(
            (p for p in self.candidates_providers if p.supports(ctx)),
            self.candidates_providers[0],
        )
        precheck = GenerationPrecheck(ctx)
        for observation in precheck.guard_observations:
            self.event_sink.emit(
                GuardConditionEvaluated(
                    rule_id=ctx.rule.rule_id,
                    node_id=observation.node_id,
                    node_path=observation.node_path,
                    node_type=observation.node_type,
                    marker=observation.marker,
                    debug_expression=observation.debug_expression,
                    matched=observation.matched,
                    spans=observation.spans,
                )
            )

        provider_request = CandidateRequest(
            ctx=ctx,
            precheck=precheck,
            event_sink=self.event_sink,
        )
        provision = provider.provide(provider_request)
        proposals, skip_reason = self.generator.generate(
            provision.ctx, provision.candidates
        )
        normalized = [
            self.freeze_normalized_proposal(ctx.doc, proposal)
            for proposal in proposals
        ]
        if skip_reason is not None:
            self.event_sink.emit(
                PatchSkipped(rule_id=ctx.rule.rule_id, reason=skip_reason)
            )
        generated_count = sum(
            1
            for proposal in normalized
            if proposal.candidate_kind == "generated"
        )
        fallback_count = sum(
            1
            for proposal in normalized
            if proposal.candidate_kind == "fallback"
        )
        static_count = sum(
            1 for proposal in normalized if proposal.candidate_kind == "static"
        )
        noop_count = sum(
            1 for proposal in normalized if proposal.op == PatchOp.NOOP
        )
        self.event_sink.emit(
            ProposalsGenerated(
                rule_id=ctx.rule.rule_id,
                proposals_count=len(normalized),
                generated_count=generated_count,
                fallback_count=fallback_count,
                static_count=static_count,
                noop_count=noop_count,
            )
        )
        outcome = ProposalOutcome(
            proposals=tuple(normalized), skip_reason=skip_reason
        )
        return list(outcome.proposals)

    def freeze_normalized_proposal(
        self, base_doc: DocumentState, proposal: PatchProposal
    ) -> PatchProposal:
        """Finalize one proposal into the canonical immutable downstream form.

        Purpose:
            Validate payload expectations for the proposal operation and compute
            deterministic patch bytes and hash values before the proposal leaves
            the planner.

        Architectural role:
            Last normalization gate inside proposal generation.

        Inputs (architectural provenance):
            Receives the current base document and one proposal produced by
            shared proposal logic.

        Outputs (downstream usage):
            Returns a PatchProposal ready for scoring, conflict resolution, and
            patch application.

        """
        _ = base_doc
        payload_norm = proposal.payload_norm
        if proposal.op in {PatchOp.NOOP, PatchOp.DELETE}:
            payload_norm = None
        elif payload_norm is None:
            raise ValueError(
                "payload_norm must be pre-canonicalized for non-noop proposals"
            )
        canonical_bytes = patch_canonical.patch_bytes(
            proposal.op, proposal.span_abs, payload_norm
        )
        return proposal.with_updates(
            payload_norm=payload_norm,
            patch_bytes=canonical_bytes,
            patch_hash=patch_canonical.patch_hash(canonical_bytes),
        )
