"""Patch proposal domain objects.

This module defines the core value objects that represent candidate document
edits produced during proposal generation and consumed by downstream stages such
as scoring and application.

These objects are patch-domain artifacts rather than runtime infrastructure or
proposal-stage orchestration. They describe *what change is proposed* and *under
what construction context*, independent of execution mechanics.

Responsibilities
----------------
- Represent a proposed patch operation in a stable, serializable form.
- Capture the minimal context required to construct or interpret a proposal.
- Serve as the shared contract between proposal generation, scoring, and patch
  application stages.

Non-responsibilities
--------------------
- Executing patches.
- Scheduling or coordinating stages.
- Maintaining runtime lifecycle or state.
- Emitting events or messages.

Architectural role
------------------
This module belongs to the patching domain boundary. It intentionally sits
outside `engine.runtime` to avoid coupling shared domain objects to execution
infrastructure.

Objects defined here are expected to be:
    - Immutable or effectively immutable.
    - Deterministic representations of candidate edits.
    - Safe to pass across stage and process boundaries.
Typical consumers
-----------------
- Proposal planners and candidate generators.
- Scoring components evaluating candidate edits.
- Stages applying accepted patches.
- Telemetry and reporting components inspecting proposal outcomes.

"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from answer_engineering.engine.runtime.runtime_primitives import (
    PatchOp,
    SpanAbs,
)
from answer_engineering.rules.compile.plan import CandidateSpec


@dataclass(frozen=True, slots=True)
class ProposalContext:
    """Context fields shared by all proposals produced in one rule step.

    Inputs (architectural provenance):
        Derived from orchestrator step context and rule metadata.

    Outputs (downstream usage):
        Applied to each proposal for version checks, traceability, and scoring.

    """

    base_version_id: str
    rule_id: str
    guard_abs_start: int | None = None


@dataclass(frozen=True, slots=True, init=False)
class PatchProposal:
    """Immutable record describing one canonical patch proposal or noop.

    Purpose:
        Carry the full normalized editing intent for one candidate, including
        span, payload, provenance, scoring metadata, and cached hash
        information.

    Architectural role:
        Core patch-proposal value object shared across proposal generation,
        scoring, conflict resolution, application, and telemetry.

    Inputs (architectural provenance):
        Built from proposal-generation context or helper constructors such as
        noop and from_candidate.

    Outputs (downstream usage):
        Consumed by scorers, selectors, apply stages, and reporting code.

    Invariants/constraints:
        Instances are frozen. Non-noop proposals are expected to carry
        canonicalized span and payload data.

    """

    op: PatchOp
    span_abs: SpanAbs | None
    payload: str | None
    base_version_id: str
    rule_id: str
    score: float
    reason: str
    cached_final_text: str | None = None
    cached_score_logprob: float | None = None
    cached_prob_ratio_to_best: float | None = None
    payload_norm: str | None = None
    patch_bytes: bytes = b""
    patch_hash: str = ""
    candidate_index: int = 0
    candidate_kind: Literal["static", "generated", "fallback"] = "static"
    candidate_id: str = ""
    candidate_label: str = ""
    candidate_hash: str = ""
    guard_abs_start: int | None = None

    def __init__(
        self,
        *,
        op: PatchOp,
        base_version_id: str,
        rule_id: str,
        reason: str,
        span_abs: SpanAbs | None = None,
        payload: str | None = None,
        score: float = 0.0,
        cached_final_text: str | None = None,
        cached_score_logprob: float | None = None,
        cached_prob_ratio_to_best: float | None = None,
        payload_norm: str | None = None,
        patch_bytes: bytes = b"",
        patch_hash: str = "",
        candidate_index: int = 0,
        candidate_kind: Literal["static", "generated", "fallback"] = "static",
        candidate_id: str = "",
        candidate_label: str = "",
        candidate_hash: str = "",
        guard_abs_start: int | None = None,
    ) -> None:
        """Initialize proposal while enforcing noop/non-noop invariants.

        Inputs (architectural provenance):
            Called by proposal-construction paths in orchestration stages.

        Outputs (downstream usage):
            Produces validated immutable proposal consumed by downstream
            scoring, selection, and apply phases.

        """
        resolved_span = span_abs
        resolved_payload = payload
        resolved_score = score
        resolved_reason = reason
        if op == PatchOp.NOOP:
            if span_abs is not None:
                raise ValueError("noop proposal span_abs must be None")
            if payload is not None:
                raise ValueError("noop proposal payload must be None")
            resolved_span = None
            resolved_payload = None
            resolved_score = 0.0
            if not resolved_reason:
                resolved_reason = "noop"
        else:
            if resolved_span is None:
                raise ValueError("span_abs is required for non-noop proposals")
            if resolved_payload is None:
                raise ValueError("payload is required for non-noop proposals")
        if not resolved_reason:
            raise ValueError("reason is required for non-noop proposals")

        object.__setattr__(self, "op", op)
        object.__setattr__(self, "span_abs", resolved_span)
        object.__setattr__(self, "payload", resolved_payload)
        object.__setattr__(self, "base_version_id", base_version_id)
        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "score", float(resolved_score))
        object.__setattr__(self, "reason", resolved_reason)
        object.__setattr__(self, "cached_final_text", cached_final_text)
        object.__setattr__(self, "cached_score_logprob", cached_score_logprob)
        object.__setattr__(
            self, "cached_prob_ratio_to_best", cached_prob_ratio_to_best
        )
        object.__setattr__(self, "payload_norm", payload_norm)
        object.__setattr__(self, "patch_bytes", patch_bytes)
        object.__setattr__(self, "patch_hash", patch_hash)
        object.__setattr__(self, "candidate_index", candidate_index)
        object.__setattr__(self, "candidate_kind", candidate_kind)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "candidate_label", candidate_label)
        object.__setattr__(self, "candidate_hash", candidate_hash)
        object.__setattr__(self, "guard_abs_start", guard_abs_start)

    @classmethod
    def from_candidate(
        cls,
        *,
        op: PatchOp,
        span_abs: SpanAbs,
        payload: str,
        reason: str,
        context: ProposalContext,
        candidate: CandidateSpec,
        payload_norm: str | None = None,
        candidate_index: int = 0,
        candidate_hash: str = "",
    ) -> PatchProposal:
        """Build proposal from candidate-source metadata and shared context.

        Inputs (architectural provenance):
            ``context`` comes from orchestrator step state; ``candidate`` comes
            from static/generated/fallback candidate providers.

        Outputs (downstream usage):
            Returns standardized proposal with candidate provenance fields used
            by scoring, ranking, telemetry, and conflict resolution.

        """
        return cls(
            op=op,
            span_abs=span_abs,
            payload=payload,
            base_version_id=context.base_version_id,
            rule_id=context.rule_id,
            score=float(candidate.priority),
            reason=reason,
            cached_score_logprob=float(candidate.logprob)
            if candidate.logprob is not None
            else None,
            payload_norm=payload_norm,
            candidate_index=candidate_index,
            candidate_kind=candidate.kind,
            candidate_id=candidate.candidate_id
            or f"candidate_{candidate_index + 1}",
            candidate_label=(
                candidate.label
                or candidate.candidate_id
                or f"candidate_{candidate_index + 1}"
            ),
            candidate_hash=candidate_hash,
            guard_abs_start=context.guard_abs_start,
        )

    @classmethod
    def noop(cls, *, context: ProposalContext, reason: str) -> PatchProposal:
        """Build a noop proposal for rule steps with no applicable edit.

        Inputs (architectural provenance):
            ``context`` comes from active step/rule metadata.

        Outputs (downstream usage):
            Noop proposal participates in deterministic selection flow without
            mutating document state.

        """
        return cls(
            op=PatchOp.NOOP,
            base_version_id=context.base_version_id,
            rule_id=context.rule_id,
            score=0.0,
            reason=reason,
            guard_abs_start=context.guard_abs_start,
        )

    def with_updates(self, **changes: object) -> PatchProposal:
        """Return a new proposal with selected fields updated.

        Inputs (architectural provenance):
            Called by downstream scoring/conflict/apply steps that enrich
            proposal metadata.

        Outputs (downstream usage):
            Produces immutable updated proposal for subsequent pipeline stages.

        """
        return replace(self, **changes)
