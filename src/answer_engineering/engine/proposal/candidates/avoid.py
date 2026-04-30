"""Avoid-rule candidate adaptation and stream-serving policy.

Purpose:
    Serve generated and fallback candidates for avoid rules and adapt probing
    results into proposal-facing candidate specs.

Architectural role:
    Proposal-side adapter between avoid-rule semantics and probing-backed
    generated candidates.

Current architecture notes:
    This module currently owns mutable state for remaining generated candidates,
    exhaustion tracking, and pop arbitration. That is functional, but much of
    that state is architecturally closer to probing lifecycle than to proposal
    semantics.

Architectural direction:
    This module should evolve toward clearer separation between proposal
    semantics and serving lifecycle behavior.

Why this matters:
    Mixed responsibilities increase maintenance cost and reduce subsystem
    clarity.

What better would look like:
    Serving mechanics handled by dedicated components while this module focuses
    on proposal adaptation.

How improvement can be recognized:
    - Reduced mutable state
    - Simpler serving logic

Open constraint:
    Serving policies may evolve as runtime behavior changes.

"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

from answer_engineering.engine.pipeline.context import StepContext
from answer_engineering.engine.pipeline.events import (
    AvoidProbeCacheExhausted,
    AvoidProbeCandidatePopped,
    AvoidProbeEpisodeStarted,
)
from answer_engineering.engine.proposal import proposal_logic
from answer_engineering.engine.proposal.candidates.base import (
    CandidateProvider,
    CandidateProvision,
    CandidateRequest,
)
from answer_engineering.engine.runtime import scope
from answer_engineering.engine.telemetry.events.event_sink import (
    DebugEventEmitter,
    RuntimeEventSink,
)
from answer_engineering.inference.model_types import (
    GenerationRuntimeProtocol,
)
from answer_engineering.inference.probing.runtime.probe_runtime import (
    ProbeRuntime,
)
from answer_engineering.rules.compile.plan import CandidateSpec


def _new_avoid_remaining_candidates() -> dict[
    AvoidCandidateStreamKey, deque[CandidateSpec]
]:
    return {}


def _new_avoid_empty_requests() -> dict[AvoidCandidateStreamKey, int]:
    return {}


def _new_avoid_pop_arbiter() -> dict[AvoidPopArbiterKey, AvoidPopArbiterState]:
    return {}


def _fallback_candidates(ctx: StepContext) -> tuple[CandidateSpec, ...]:
    return tuple(
        CandidateSpec(
            op=candidate.op,
            text=candidate.text,
            kind="fallback",
            priority=candidate.priority,
            label=candidate.label,
            candidate_id=candidate.candidate_id,
        )
        for candidate in ctx.rule.candidates
    )


def _compose_avoid_candidates(
    *,
    popped: CandidateSpec | None,
    fallback_candidates: tuple[CandidateSpec, ...],
) -> tuple[CandidateSpec, ...]:
    generated_part = (popped,) if popped is not None else tuple()
    return (*generated_part, *fallback_candidates)


def _emit_probe_episode_started(
    *,
    event_sink: RuntimeEventSink | None,
    rule_id: str,
    key: AvoidCandidateStreamKey,
    generated_count: int,
) -> None:
    if event_sink is None:
        return
    event_sink.emit(
        AvoidProbeEpisodeStarted(
            rule_id=rule_id,
            cache_key=_avoid_stream_key_id(key),
            generated_count=generated_count,
        )
    )


def _emit_probe_candidate_popped(
    *,
    event_sink: RuntimeEventSink | None,
    rule_id: str,
    key: AvoidCandidateStreamKey,
    candidate: CandidateSpec,
) -> None:
    if event_sink is None:
        return
    event_sink.emit(
        AvoidProbeCandidatePopped(
            rule_id=rule_id,
            cache_key=_avoid_stream_key_id(key),
            candidate_id=candidate.candidate_id or candidate.label or "",
        )
    )


def _emit_probe_cache_exhausted(
    *,
    event_sink: RuntimeEventSink | None,
    rule_id: str,
    key: AvoidCandidateStreamKey,
    empty_request_count: int,
) -> None:
    if event_sink is None:
        return
    event_sink.emit(
        AvoidProbeCacheExhausted(
            rule_id=rule_id,
            cache_key=_avoid_stream_key_id(key),
            empty_request_count=empty_request_count,
        )
    )


@dataclass(slots=True)
class AvoidCandidatesProvider(CandidateProvider):
    """Proposal-side provider for avoid-rule candidate sets.

    Purpose:
        Route avoid rules to probing-backed candidate generation and return one
        atomic CandidateProvision for the planner.

    Architectural role:
        Concrete CandidateProvider implementation for the avoid rule family.

    Current architecture notes:
        The provider delegates most serving policy to AvoidStreamSession and
        uses ProbeRuntime for generated candidate production.

    Architectural TODO:
        Shrink this provider into a thin adapter once probe-serving lifecycle
        state moves behind probing-owned APIs.

    """

    runtime: ProbeRuntime
    session: AvoidStreamSession

    def __init__(
        self,
        runtime: GenerationRuntimeProtocol | None = None,
        trajectory_debug: bool = False,
        debug_emitter: DebugEventEmitter | None = None,
    ) -> None:
        """Create probing collaborators for avoid-rule candidate provision.

        Purpose:
            Assemble the probe runtime and avoid stream session that serve
            generated and fallback candidates for avoid rules.

        Architectural role:
            Construction boundary for the proposal-side avoid candidate
            provider.

        Inputs (architectural provenance):
            Receives an optional generation runtime, trajectory-debug flag, and
            debug emitter from provider setup or tests.

        Outputs (downstream usage):
            Stores a `ProbeRuntime` and `AvoidStreamSession` consumed by
            `provide` when planner code asks for avoid-rule candidates.

        Invariants/constraints:
            The provider owns serving state only for its lifetime.
            Generated-candidate lifecycle state should eventually move behind
            probing-owned APIs rather than leaking further into proposal logic.

        """
        self.runtime = ProbeRuntime(
            generation_runtime=runtime,
            trajectory_debug=trajectory_debug,
            debug_emitter=debug_emitter
            if debug_emitter is not None
            else DebugEventEmitter(),
        )
        self.session = AvoidStreamSession(runtime=self.runtime)

    def supports(self, ctx: StepContext) -> bool:
        """Return whether the current rule should be served by the avoid-.

        Purpose:
            Restrict probe-backed candidate generation to rule families whose
            names begin with the avoid prefix.

        Architectural role:
            Provider-selection predicate inside the proposal candidate boundary.

        Inputs (architectural provenance):
            Receives the active ``StepContext`` from planner provider selection.

        Outputs (downstream usage):
            Boolean determines whether the planner delegates candidate
            provisioning to this provider.

        """
        return ctx.rule.name.startswith("avoid:")

    def provide(self, request: CandidateRequest) -> CandidateProvision:
        """Serve one avoid-rule candidate provision request.

        Purpose:
            Delegate an avoid-rule candidate request to the provider-owned
            stream session and return the resulting generated/fallback candidate
            set.

        Architectural role:
            Proposal-facing `CandidateProvider` method for the avoid rule
            family.

        Inputs (architectural provenance):
            Receives a `CandidateRequest` from proposal planning after provider
            selection has established that this provider supports the active
            rule.

        Outputs (downstream usage):
            Returns a `CandidateProvision` consumed by shared proposal logic for
            scoring, selection, and later patch application.

        Invariants/constraints:
            This method should remain an adapter: request interpretation and
            serving state belong to `AvoidStreamSession`, while generated
            candidate production belongs to probing.

        """
        return self.session.provide(request=request)


@dataclass(slots=True)
class AvoidStreamSession:
    """Mutable owner of avoid candidate-serving state and policy.

    Purpose:
        Remember remaining generated candidates, exhaustion counts, and
        pop-order state across repeated avoid-rule requests.

    Architectural role:
        Stateful serving subsystem currently owned by the proposal avoid
        provider.

    Ownership:
        Owns remaining generated candidates, empty-request counts, and per-step
        pop arbitration for avoid probe streams.

    Current architecture notes:
        This class is the clearest place where proposal and probing boundaries
        are still mixed: it combines proposal-facing fallback composition with
        probing-shaped serving lifecycle state.

    Architectural TODO:
        Move generated-candidate serving, exhaustion tracking, and pop
        arbitration into probing, while keeping fallback candidate composition
        proposal-side.

    """

    runtime: ProbeRuntime
    _avoid_remaining_candidates: dict[
        AvoidCandidateStreamKey, deque[CandidateSpec]
    ] = field(default_factory=_new_avoid_remaining_candidates)
    _avoid_empty_requests: dict[AvoidCandidateStreamKey, int] = field(
        default_factory=_new_avoid_empty_requests
    )
    _avoid_pop_arbiter: dict[AvoidPopArbiterKey, AvoidPopArbiterState] = field(
        default_factory=_new_avoid_pop_arbiter
    )

    def provide(self, request: CandidateRequest) -> CandidateProvision:
        """Provide the next avoid candidate group for one rule evaluation.

        Purpose:
            Serve cached or newly generated avoid candidates while preserving
            the session-level lifecycle for one stream of alternatives.

        Architectural role:
            Candidate-provider method at the proposal/probing boundary. It
            adapts probe runtime output into proposal groups without owning
            probe-prefix assembly or scoring policy.

        Inputs (architectural provenance):
            Receives the current document view, compiled rule information,
            generation context, and probe cache state from orchestration.

        Outputs (downstream usage):
            Returns proposal candidates that downstream scoring and selection
            can rank against other rule alternatives.

        Invariants/constraints:
            Cache reuse must be keyed by stable probe identity. The method
            should not hide probe failures as successful proposal generation.

        """
        ctx = request.ctx
        precheck = (
            request.precheck
            if request.precheck is not None
            else proposal_logic.GenerationPrecheck(ctx)
        )
        if precheck.noop_reason is not None:
            return CandidateProvision(ctx=ctx, candidates=tuple())
        fallback_candidates = _fallback_candidates(ctx)
        if proposal_logic.already_satisfied(
            ctx.edit_view,
            fallback_candidates,
            casefold_compare=ctx.rule.effective_edit_scope().casefold,
        ):
            return CandidateProvision(ctx=ctx, candidates=fallback_candidates)
        assert precheck.span is not None
        precheck_span = precheck.span
        key_ctx = ctx
        key_start = precheck_span[0]
        base_key = self._avoid_cache_key(
            ctx=ctx, span_abs_start=precheck_span[0]
        )
        base_empty_count = self._avoid_empty_requests.get(base_key, 0)
        floor_allowed = _allow_avoid_scope_floor(ctx)
        if base_empty_count > 0 and floor_allowed:
            floor_abs_start = scope.sentence_floor_start(
                text=ctx.doc.text,
                span_abs=precheck_span,
            )
            if floor_abs_start > precheck_span[0]:
                key_ctx = replace(
                    ctx,
                    avoid_edit_floor_abs_start=floor_abs_start,
                )
                key_start = floor_abs_start
                self._debug(
                    ctx,
                    "AVOID_PROBE_SCOPE_FLOOR "
                    f"rule_id={ctx.rule.rule_id} "
                    f"old_start={precheck_span[0]} "
                    f"new_start={floor_abs_start} "
                    f"base_empty_count={base_empty_count}",
                )
        key = self._avoid_cache_key(ctx=key_ctx, span_abs_start=key_start)
        remaining = self._avoid_remaining_candidates.get(key)
        pop_request_cause = "cache_hit"
        sink = request.event_sink or ctx.event_sink
        if remaining is None:
            generated = self.runtime.generate(key_ctx, precheck=precheck)
            shared_generated = [
                candidate
                for candidate in generated
                if candidate.kind == "generated"
            ]
            remaining = deque(shared_generated)
            self._avoid_remaining_candidates[key] = remaining
            self._avoid_empty_requests[key] = 0
            pop_request_cause = "cache_miss"
            _emit_probe_episode_started(
                event_sink=sink,
                rule_id=ctx.rule.rule_id,
                key=key,
                generated_count=len(shared_generated),
            )
        self._debug(
            ctx,
            "AVOID_POP_REQUESTED "
            f"rule_id={ctx.rule.rule_id} "
            f"step_id={ctx.step.token_index} "
            f"cause={pop_request_cause}",
        )
        pop_result = self._pop_avoid_stream_candidate(
            key=key,
            remaining=remaining,
            step_id=ctx.step.token_index,
            request_cause=pop_request_cause,
            rule_id=ctx.rule.rule_id,
            event_sink=sink,
        )
        popped = pop_result.candidate
        if popped is not None:
            _emit_probe_candidate_popped(
                event_sink=sink,
                rule_id=ctx.rule.rule_id,
                key=key,
                candidate=popped,
            )
            self._debug(
                ctx,
                "AVOID_POP_GRANTED "
                f"rule_id={ctx.rule.rule_id} "
                f"step_id={ctx.step.token_index} "
                f"candidate={popped.label} "
                f"remaining_generated={len(remaining)}",
            )
        else:
            self._debug(
                ctx,
                "AVOID_POP_DENIED "
                f"rule_id={ctx.rule.rule_id} "
                f"step_id={ctx.step.token_index} "
                "reason=duplicate_step_key "
                f"seen_before={str(pop_result.empty_seen_before).lower()} "
                f"empty_request_count={pop_result.empty_request_count}",
            )
        candidates = _compose_avoid_candidates(
            popped=popped,
            fallback_candidates=fallback_candidates,
        )
        ctx_for_generation = key_ctx
        if pop_result.empty_seen_before and floor_allowed:
            floor_abs_start = scope.sentence_floor_start(
                text=ctx.doc.text,
                span_abs=precheck_span,
            )
            if floor_abs_start > precheck_span[0]:
                ctx_for_generation = replace(
                    ctx,
                    avoid_edit_floor_abs_start=floor_abs_start,
                )
                self._debug(
                    ctx,
                    "AVOID_SCOPE_FLOOR "
                    f"rule_id={ctx.rule.rule_id} "
                    f"old_start={precheck_span[0]} "
                    f"new_start={floor_abs_start}",
                )
        return CandidateProvision(ctx=ctx_for_generation, candidates=candidates)

    def _avoid_cache_key(
        self,
        *,
        ctx: StepContext,
        span_abs_start: int,
    ) -> AvoidCandidateStreamKey:
        probe_key = self.runtime.avoid_stream_key(
            ctx,
            abs_start=span_abs_start,
        )
        return AvoidCandidateStreamKey(
            span_abs_start=span_abs_start,
            prefix_hash=probe_key.prefix_hash,
            probe_num_beams=probe_key.probe_num_beams,
            probe_max_new_tokens=probe_key.probe_max_new_tokens,
        )

    def _pop_avoid_stream_candidate(
        self,
        *,
        key: AvoidCandidateStreamKey,
        remaining: deque[CandidateSpec],
        step_id: int,
        request_cause: str,
        rule_id: str,
        event_sink: RuntimeEventSink | None,
    ) -> AvoidStreamPopResult:
        arbiter_key = AvoidPopArbiterKey(step_id=step_id, avoid_key=key)
        arbiter_state = self._avoid_pop_arbiter.setdefault(
            arbiter_key, AvoidPopArbiterState()
        )
        if arbiter_state.generated_on_miss and arbiter_state.first_pop_consumed:
            return AvoidStreamPopResult(
                candidate=None,
                empty_seen_before=False,
                empty_request_count=self._avoid_empty_requests.get(key, 0),
            )
        if arbiter_state.granted:
            return AvoidStreamPopResult(
                candidate=None,
                empty_seen_before=False,
                empty_request_count=self._avoid_empty_requests.get(key, 0),
            )
        if remaining:
            arbiter_state.granted = True
            if request_cause == "cache_miss":
                arbiter_state.generated_on_miss = True
            arbiter_state.first_pop_consumed = True
            return AvoidStreamPopResult(
                candidate=remaining.popleft(),
                empty_seen_before=False,
                empty_request_count=self._avoid_empty_requests.get(key, 0),
            )
        empty_request_count = self._avoid_empty_requests.get(key, 0) + 1
        self._avoid_empty_requests[key] = empty_request_count
        if empty_request_count == 1:
            _emit_probe_cache_exhausted(
                event_sink=event_sink,
                rule_id=rule_id,
                key=key,
                empty_request_count=empty_request_count,
            )
        return AvoidStreamPopResult(
            candidate=None,
            empty_seen_before=empty_request_count > 1,
            empty_request_count=empty_request_count,
        )

    def _debug(self, ctx: StepContext, msg: str) -> None:
        if not ctx.trajectory_debug:
            return
        self.runtime.debug_emitter.emit(msg)


@dataclass(frozen=True, slots=True)
class AvoidCandidateStreamKey:
    """Identity of one cached avoid-probe stream.

    Purpose:
        Distinguish reusable generated-candidate streams by the rule, prefix,
        and span context that produced them.

    Architectural role:
        Cache-key value object for avoid probing.

    Inputs (architectural provenance):
        Built from step context, resolved span, and probe prefix information
        inside the avoid provider.

    Outputs (downstream usage):
        Used to index remaining candidates, empty-request counts, and
        pop-arbiter state.

    """

    span_abs_start: int
    prefix_hash: str
    probe_num_beams: int
    probe_max_new_tokens: int

    def cache_key_id(self) -> str:
        """Return the stable cache identity for one avoid-candidate stream.

        Purpose:
            Convert the structured stream key into the string identifier used by
            probe caches and telemetry events.

        Architectural role:
            Identity helper at the proposal/probing lifecycle boundary.

        Inputs (architectural provenance):
            Reads the immutable rule, document, match, prefix, and generation
            fields stored on the stream key.

        Outputs (downstream usage):
            Produces a stable cache key consumed by avoid-probe caches, debug
            output, and event records.

        Invariants/constraints:
            The returned identity must change when probe-relevant inputs change
            and stay stable across repeated serving of the same probe set.

        """
        return (
            f"{self.span_abs_start}:{self.prefix_hash}:"
            f"{self.probe_num_beams}:{self.probe_max_new_tokens}"
        )


def _avoid_stream_key_id(key: AvoidCandidateStreamKey) -> str:
    """Format a stable probe cache key id for one avoid stream key."""
    return key.cache_key_id()


@dataclass(frozen=True, slots=True)
class AvoidStreamPopResult:
    """Result of popping the next candidate from an avoid stream.

    Purpose:
        Return both the chosen candidate, if any, and metadata about stream
        state after the pop decision.

    Architectural role:
        Internal transport record between avoid-stream serving logic and
        provider orchestration.

    Inputs (architectural provenance):
        Constructed by the stream session after consulting cached candidates and
        pop-arbiter state.

    Outputs (downstream usage):
        Consumed by avoid-provider logic to decide whether to emit a candidate,
        regenerate probes, or mark exhaustion.

    """

    candidate: CandidateSpec | None
    empty_seen_before: bool
    empty_request_count: int


@dataclass(frozen=True, slots=True)
class AvoidPopArbiterKey:
    """Key for sharing pop-order state across related avoid streams.

    Purpose:
        Identify the granularity at which candidate-pop fairness and exhaustion
        are coordinated.

    Architectural role:
        Internal coordination key for avoid probing.

    Inputs (architectural provenance):
        Derived from the same contextual information that determines when two
        probe streams should share serving state.

    Outputs (downstream usage):
        Used to index ``AvoidPopArbiterState`` records.

    """

    step_id: int
    avoid_key: AvoidCandidateStreamKey


@dataclass(slots=True)
class AvoidPopArbiterState:
    """Mutable state tracking pop order and exhaustion for related avoid.

    Purpose:
        Remember which generated candidates were already served so repeated
        requests do not recycle the same probe output prematurely.

    Architectural role:
        In-memory coordination state owned by the avoid provider.

    Inputs (architectural provenance):
        Updated as avoid-probe streams are popped during proposal generation.

    Outputs (downstream usage):
        Guides future stream-serving decisions and exhaustion handling.

    """

    granted: bool = False
    generated_on_miss: bool = False
    first_pop_consumed: bool = False


def _allow_avoid_scope_floor(ctx: StepContext) -> bool:
    """Return whether avoid edits may be floored to a narrower scope start.

    Purpose:
        Encapsulate the policy that decides when avoid-generated candidates may
        snap forward to the last-sentence boundary instead of using the full
        span start.

    Architectural role:
        Policy helper inside avoid proposal generation.

    Inputs (architectural provenance):
        Receives the active step context, including rule name and scope
        metadata.

    Outputs (downstream usage):
        Boolean is consumed by avoid-provider logic when computing edit floors.

    """
    if not ctx.rule.policy.validation_for_all:
        return True
    return not (
        ctx.rule.effective_guard_scope().kind == "whole_doc"
        and ctx.rule.effective_edit_scope().kind == "whole_doc"
    )
