"""Execution-stage event contracts and structured diagnostics records.

Purpose:
    Define immutable event payloads emitted by runtime stages during execution,
    including proposal generation, scoring, conflict resolution, and patch
    application.

Architectural role:
    Observability boundary for the execution pipeline. These events describe
    what happened during execution but do not control pipeline sequencing.

Contents:
    - structured runtime event records emitted by stages and runtime stages
    - base event envelope types and serialization helpers
    - debug and diagnostic events used for telemetry and analysis

Invariants:
    Events are data-only records. They must not execute behavior, mutate runtime
    state, or coordinate stage transitions.

Non-goals:
    This module does not define message routing, queue logic, or execution
    context objects. It defines observability records only.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TypeGuard
from uuid import uuid4

type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


def _event_id() -> str:
    """Generate the unique identifier stored on each emitted event."""
    return uuid4().hex


def _event_ts() -> str:
    """Return the current UTC timestamp string stored on each event."""
    return datetime.now(UTC).isoformat()


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    """Return ``True`` when ``value`` is a dictionary for JSON-safe."""
    return isinstance(value, dict)


def _is_object_list_or_tuple(
    value: object,
) -> TypeGuard[list[object] | tuple[object, ...]]:
    """Return ``True`` when ``value`` is a list or tuple for JSON-safe."""
    return isinstance(value, list | tuple)


def _json_safe(value: object) -> JsonValue:
    """Recursively normalize event field values into JSON-compatible."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if _is_object_dict(value):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if _is_object_list_or_tuple(value):
        return [_json_safe(item) for item in value]

    return str(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Base envelope for all orchestration/runtime telemetry events.

    Purpose:
        Preserve structured telemetry data describing one runtime occurrence
        without affecting control flow.

    Architectural role:
        Defines shared metadata and serialization behavior for all event types.

    Inputs:
        Populated from runtime-stage state at the moment the event is emitted.

    Outputs:
        Serialized and consumed by telemetry sinks, debugging tools, or
        reproducibility artifacts.

    """

    event_id: str = field(default_factory=_event_id)
    ts: str = field(default_factory=_event_ts)
    trace_id: str = "core"

    def serialize(self) -> dict[str, JsonValue]:
        """Serialize the event envelope and payload into a JSON-safe mapping."""
        payload = {
            key: _json_safe(value) for key, value in asdict(self).items()
        }
        payload["type"] = self.__class__.__name__
        return payload


@dataclass(frozen=True, slots=True)
class RuleEvaluationStarted(Event):
    """Event emitted when evaluation starts for one rule on a document view.

    Purpose:
        Capture one structured runtime observation as immutable data rather than
        encoding it as ad-hoc debug text.

    Architectural role:
        Telemetry event in the execution-pipeline observability boundary. It
        anchors rule-scoped telemetry before matching, proposal, and scoring
        events are emitted.

    Inputs (architectural provenance):
        Populated by the stage or runtime collaborator that owns the observed
        lifecycle transition.

    Outputs (downstream usage):
        Serialized by event sinks and consumed by debugging, telemetry
        snapshots, golden tests, and reproduction artifacts.

    Invariants/constraints:
        Rule id, document version id, and scope spec describe the evaluated
        view. The event is data-only: it must not mutate runtime state, control
        stage ordering, or compute follow-up decisions.

    """

    rule_id: str
    doc_version_id: str
    scope_spec: object


@dataclass(frozen=True, slots=True)
class ViewMatchSettings:
    """Typed match-policy settings associated with one produced view.

    Purpose:
        Capture one structured runtime observation as immutable data rather than
        encoding it as ad-hoc debug text.

    Architectural role:
        Telemetry event in the execution-pipeline observability boundary. It
        records the effective matching policy attached to extracted text views.

    Inputs (architectural provenance):
        Populated by the stage or runtime collaborator that owns the observed
        lifecycle transition.

    Outputs (downstream usage):
        Serialized by event sinks and consumed by debugging, telemetry
        snapshots, golden tests, and reproduction artifacts.

    Invariants/constraints:
        Settings must mirror the policy used to produce matches for that view.
        The event is data-only: it must not mutate runtime state, control stage
        ordering, or compute follow-up decisions.

    """

    casefold: bool


@dataclass(frozen=True, slots=True)
class ViewProduced(Event):
    """Event emitted after scoped text view extraction for a rule.

    Purpose:
        Capture one structured runtime observation as immutable data rather than
        encoding it as ad-hoc debug text.

    Architectural role:
        Telemetry event in the execution-pipeline observability boundary. It
        links rule evaluation to the exact absolute text span inspected by
        matchers.

    Inputs (architectural provenance):
        Populated by the stage or runtime collaborator that owns the observed
        lifecycle transition.

    Outputs (downstream usage):
        Serialized by event sinks and consumed by debugging, telemetry
        snapshots, golden tests, and reproduction artifacts.

    Invariants/constraints:
        Absolute offsets and match settings must describe the produced base
        view. The event is data-only: it must not mutate runtime state, control
        stage ordering, or compute follow-up decisions.

    """

    rule_id: str
    base_version_id: str
    abs_start: int
    abs_end: int
    match_settings: ViewMatchSettings


@dataclass(frozen=True, slots=True)
class GuardConditionEvaluated(Event):
    """Event emitted for one guard-node evaluation result.

    Purpose:
        Capture one structured runtime observation as immutable data rather than
        encoding it as ad-hoc debug text.

    Architectural role:
        Telemetry event in the execution-pipeline observability boundary. It
        exposes guard-tree evidence without coupling telemetry consumers to
        matcher internals.

    Inputs (architectural provenance):
        Populated by the stage or runtime collaborator that owns the observed
        lifecycle transition.

    Outputs (downstream usage):
        Serialized by event sinks and consumed by debugging, telemetry
        snapshots, golden tests, and reproduction artifacts.

    Invariants/constraints:
        Node identity, debug expression, match status, and spans must describe
        one guard node. The event is data-only: it must not mutate runtime
        state, control stage ordering, or compute follow-up decisions.

    """

    rule_id: str
    node_id: str
    node_path: str
    node_type: str
    marker: str | None
    debug_expression: str
    matched: bool
    spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ProposalsGenerated(Event):
    """Event emitted after proposal generation for a rule.

    Purpose:
        Capture one structured runtime observation as immutable data rather than
        encoding it as ad-hoc debug text.

    Architectural role:
        Telemetry event in the execution-pipeline observability boundary. It
        summarizes proposal-provider output before scoring and selection occur.

    Inputs (architectural provenance):
        Populated by the stage or runtime collaborator that owns the observed
        lifecycle transition.

    Outputs (downstream usage):
        Serialized by event sinks and consumed by debugging, telemetry
        snapshots, golden tests, and reproduction artifacts.

    Invariants/constraints:
        Counts should be internally consistent for generated, fallback, static,
        and noop proposals. The event is data-only: it must not mutate runtime
        state, control stage ordering, or compute follow-up decisions.

    """

    rule_id: str
    proposals_count: int
    generated_count: int
    fallback_count: int
    static_count: int
    noop_count: int


@dataclass(frozen=True, slots=True)
class ProposalRejected(Event):
    """Event emitted when a proposal is rejected before application.

    Purpose:
        Capture one structured runtime observation as immutable data rather than
        encoding it as ad-hoc debug text.

    Architectural role:
        Telemetry event in the execution-pipeline observability boundary. It
        records pre-application rejection reasons for debugging and aggregate
        reporting.

    Inputs (architectural provenance):
        Populated by the stage or runtime collaborator that owns the observed
        lifecycle transition.

    Outputs (downstream usage):
        Serialized by event sinks and consumed by debugging, telemetry
        snapshots, golden tests, and reproduction artifacts.

    Invariants/constraints:
        The reason should be stable enough for tests and human diagnostics. The
        event is data-only: it must not mutate runtime state, control stage
        ordering, or compute follow-up decisions.

    """

    rule_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class AvoidProbeCacheExhausted(Event):
    """Event emitted when avoid-probe cache cannot provide candidates.

    Purpose:
        Capture one structured runtime observation as immutable data rather than
        encoding it as ad-hoc debug text.

    Architectural role:
        Telemetry event in the execution-pipeline observability boundary. It
        marks a probe-serving lifecycle miss after cached candidates are
        unavailable.

    Inputs (architectural provenance):
        Populated by the stage or runtime collaborator that owns the observed
        lifecycle transition.

    Outputs (downstream usage):
        Serialized by event sinks and consumed by debugging, telemetry
        snapshots, golden tests, and reproduction artifacts.

    Invariants/constraints:
        Cache key and empty-request count must identify the exhausted probe set.
        The event is data-only: it must not mutate runtime state, control stage
        ordering, or compute follow-up decisions.

    """

    rule_id: str
    cache_key: str
    empty_request_count: int


@dataclass(frozen=True, slots=True)
class AvoidProbeSetGenerated(Event):
    """Event emitted when avoid-probe candidates are generated for a key.

    Purpose:
        Capture one structured runtime observation as immutable data rather than
        encoding it as ad-hoc debug text.

    Architectural role:
        Telemetry event in the execution-pipeline observability boundary. It
        records generation of a reusable candidate set before individual
        consumption.

    Inputs (architectural provenance):
        Populated by the stage or runtime collaborator that owns the observed
        lifecycle transition.

    Outputs (downstream usage):
        Serialized by event sinks and consumed by debugging, telemetry
        snapshots, golden tests, and reproduction artifacts.

    Invariants/constraints:
        Generated count and probe budget must describe the produced candidate
        pool. The event is data-only: it must not mutate runtime state, control
        stage ordering, or compute follow-up decisions.

    """

    rule_id: str
    cache_key: str
    generated_count: int
    probe_budget: int


@dataclass(frozen=True, slots=True)
class AvoidProbeEpisodeStarted(Event):
    """Event emitted when an avoid-probe episode starts for a cache key.

    Purpose:
        Capture one structured runtime observation as immutable data rather than
        encoding it as ad-hoc debug text.

    Architectural role:
        Telemetry event in the execution-pipeline observability boundary. It
        starts the observable lifecycle for serving candidates from one probe
        set.

    Inputs (architectural provenance):
        Populated by the stage or runtime collaborator that owns the observed
        lifecycle transition.

    Outputs (downstream usage):
        Serialized by event sinks and consumed by debugging, telemetry
        snapshots, golden tests, and reproduction artifacts.

    Invariants/constraints:
        Generated count must match the cache entry available for that episode.
        The event is data-only: it must not mutate runtime state, control stage
        ordering, or compute follow-up decisions.

    """

    rule_id: str
    cache_key: str
    generated_count: int


@dataclass(frozen=True, slots=True)
class AvoidProbeCandidatePopped(Event):
    """Event emitted when one avoid-probe candidate is consumed.

    Purpose:
        Capture one structured runtime observation as immutable data rather than
        encoding it as ad-hoc debug text.

    Architectural role:
        Telemetry event in the execution-pipeline observability boundary. It
        tracks candidate consumption from a reusable avoid-probe cache entry.

    Inputs (architectural provenance):
        Populated by the stage or runtime collaborator that owns the observed
        lifecycle transition.

    Outputs (downstream usage):
        Serialized by event sinks and consumed by debugging, telemetry
        snapshots, golden tests, and reproduction artifacts.

    Invariants/constraints:
        Candidate id and cache key must identify the consumed generated
        alternative. The event is data-only: it must not mutate runtime state,
        control stage ordering, or compute follow-up decisions.

    """

    rule_id: str
    cache_key: str
    candidate_id: str


@dataclass(frozen=True, slots=True)
class ProposalAccepted(Event):
    """Event emitted when a proposal is accepted for patch application.

    Purpose:
        Capture one structured runtime observation as immutable data rather than
        encoding it as ad-hoc debug text.

    Architectural role:
        Telemetry event in the execution-pipeline observability boundary. It
        records the selected candidate immediately before patch application.

    Inputs (architectural provenance):
        Populated by the stage or runtime collaborator that owns the observed
        lifecycle transition.

    Outputs (downstream usage):
        Serialized by event sinks and consumed by debugging, telemetry
        snapshots, golden tests, and reproduction artifacts.

    Invariants/constraints:
        Patch hash, bytes length, candidate identity, and label must describe
        the accepted patch. The event is data-only: it must not mutate runtime
        state, control stage ordering, or compute follow-up decisions.

    """

    rule_id: str
    proposal_summary: str
    patch_hash: str
    patch_bytes_len: int
    candidate_kind: str
    candidate_id: str
    candidate_label: str
    candidate_text_excerpt: str | None = None


@dataclass(frozen=True, slots=True)
class ProposalScored(Event):
    """Event emitted once proposal scoring completes.

    Purpose:
        Capture one structured runtime observation as immutable data rather than
        encoding it as ad-hoc debug text.

    Architectural role:
        Telemetry event in the execution-pipeline observability boundary. It
        exposes scoring completion for a proposal without serializing full
        scorer internals.

    Inputs (architectural provenance):
        Populated by the stage or runtime collaborator that owns the observed
        lifecycle transition.

    Outputs (downstream usage):
        Serialized by event sinks and consumed by debugging, telemetry
        snapshots, golden tests, and reproduction artifacts.

    Invariants/constraints:
        Rule id and patch hash must identify the scored candidate or patch
        target. The event is data-only: it must not mutate runtime state,
        control stage ordering, or compute follow-up decisions.

    """

    rule_id: str
    patch_hash: str


@dataclass(frozen=True, slots=True)
class PatchApplied(Event):
    """Event emitted after applying a patch to produce a new doc version.

    Purpose:
        Capture one structured runtime observation as immutable data rather than
        encoding it as ad-hoc debug text.

    Architectural role:
        Telemetry event in the execution-pipeline observability boundary. It
        records the document-version transition caused by an accepted proposal.

    Inputs (architectural provenance):
        Populated by the stage or runtime collaborator that owns the observed
        lifecycle transition.

    Outputs (downstream usage):
        Serialized by event sinks and consumed by debugging, telemetry
        snapshots, golden tests, and reproduction artifacts.

    Invariants/constraints:
        Old/new version ids and patch metadata must describe one successful
        mutation. The event is data-only: it must not mutate runtime state,
        control stage ordering, or compute follow-up decisions.

    """

    rule_id: str
    patch_id: str
    old_version_id: str
    new_version_id: str
    patch_hash: str
    patch_bytes_len: int


@dataclass(frozen=True, slots=True)
class PatchSkipped(Event):
    """Event emitted when a patch/span is corrected or skipped.

    Purpose:
        Capture stable, structured diagnostics for recoverable span or proposal
        drops so normal runtime generation remains observable without relying
        only on human-readable logs.

    """

    rule_id: str | None
    reason: str
    rule_name: str | None = None
    doc_len: int | None = None
    original_span: tuple[int, int] | None = None
    corrected_span: tuple[int, int] | None = None
    span_abs: tuple[int, int] | None = None
    nearby_text: str | None = None
    stage: str | None = None


@dataclass(frozen=True, slots=True)
class DebugEvent(Event):
    """Ad-hoc debug event for structured diagnostic messages.

    Purpose:
        Capture one structured runtime observation as immutable data rather than
        encoding it as ad-hoc debug text.

    Architectural role:
        Telemetry event in the execution-pipeline observability boundary. It
        provides a typed escape hatch for temporary diagnostics that still flow
        through event sinks.

    Inputs (architectural provenance):
        Populated by the stage or runtime collaborator that owns the observed
        lifecycle transition.

    Outputs (downstream usage):
        Serialized by event sinks and consumed by debugging, telemetry
        snapshots, golden tests, and reproduction artifacts.

    Invariants/constraints:
        The message should remain diagnostic text and should not replace durable
        event types. The event is data-only: it must not mutate runtime state,
        control stage ordering, or compute follow-up decisions.

    """

    msg: str
