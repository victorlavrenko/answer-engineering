"""Immutable runtime telemetry value objects.

Purpose:
    Define the frozen telemetry snapshot dataclasses produced by aggregation and
    attached to generation outputs.

Architectural role:
    Value layer for runtime telemetry outputs.

Owns:
    - Immutable per-condition/per-candidate/per-rule snapshot types.
    - The top-level `RuntimeTelemetrySnapshot` transport value.
    - Small normalization helpers needed to build stable snapshot fields.

Does not own:
    - Event replay and counter mutation (`telemetry.aggregation`).
    - Runtime event sink transport and emission (`telemetry.events`).

"""

from __future__ import annotations

from dataclasses import dataclass

from answer_engineering.engine.pipeline import (
    events as runtime_events,
)


@dataclass(frozen=True, slots=True)
class _ConditionIdentity:
    section: str
    operator: str


def _identity_from_marker(marker: str | None) -> _ConditionIdentity | None:
    if marker is None:
        return None
    normalized = marker.strip().casefold()
    if not normalized:
        return None
    if normalized == "connector":
        return _ConditionIdentity(section="connector", operator="any")
    for section in ("prefix", "postfix", "prompt"):
        for operator in ("all", "any", "none", "incomplete"):
            if normalized == f"{section}_{operator}":
                return _ConditionIdentity(section=section, operator=operator)
    return None


def _normalize_condition_section(
    event: runtime_events.GuardConditionEvaluated,
) -> str:
    """Normalize a guard-condition event into the section name used by.

    Purpose:
        Canonicalize optional guard-condition section names so aggregation and
        serialization use stable grouping keys.

    Architectural role:
        Immutable telemetry value object or helper inside the runtime telemetry
        snapshot boundary.

    Inputs:
        Aggregate counters, guard-condition events, and per-rule telemetry
        values from the runtime projection layer.

    Outputs:
        Stable snapshot values and small summary helpers consumed by
        serialization, reporting, and tests.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.snapshots.snapshots`
        within the engine telemetry boundary.

    Invariants:
        Returns a deterministic derived value for the supplied inputs.

    """
    identity = _identity_from_marker(event.marker)
    if identity is not None:
        return identity.section
    return event.node_path


def _normalize_condition_operator(
    event: runtime_events.GuardConditionEvaluated,
) -> str:
    """Normalize a condition event into a canonical operator label."""
    identity = _identity_from_marker(event.marker)
    if identity is not None:
        return identity.operator
    return event.node_type


@dataclass(frozen=True, slots=True)
class ConditionTelemetrySnapshot:
    """Immutable aggregate of one guard-condition identity.

    Purpose:
        Represent how often one condition identity was seen and matched during a
        run after event replay is complete.

    Fields:
        - `condition_id` is the grouping key used by aggregation/serialization.
        - `matched` and `seen` are aggregate counts, not per-event values.

    Invariants:
        This value is immutable and safe to serialize without retaining live
        runtime state.

    """

    condition_id: str
    node_path: str
    node_type: str
    debug_expression: str
    matched: int
    seen: int

    @classmethod
    def from_event(
        cls, event: runtime_events.GuardConditionEvaluated
    ) -> ConditionTelemetrySnapshot:
        """Build a condition snapshot from one guard-condition evaluation event.

        Purpose:
            Offer a convenience construction path that derives this value
            directly from upstream runtime or evaluation inputs.

        Architectural role:
            Immutable telemetry value object or helper inside the runtime
            telemetry snapshot boundary.

        Inputs:
            Aggregate counters, guard-condition events, and per-rule telemetry
            values from the runtime projection layer.

        Outputs:
            Stable snapshot values and small summary helpers consumed by
            serialization, reporting, and tests.

        Ownership:
            Owned by `answer_engineering.engine.telemetry.snapshots.snapshots`
            within the engine telemetry boundary.

        """
        normalized_path = _normalize_condition_section(event)
        normalized_type = _normalize_condition_operator(event)
        condition_id = (
            f"{normalized_path}:{normalized_type}:{event.debug_expression}"
        )
        return cls(
            condition_id=condition_id,
            node_path=normalized_path,
            node_type=normalized_type,
            debug_expression=event.debug_expression,
            matched=int(event.matched),
            seen=1,
        )


@dataclass(frozen=True, slots=True)
class CandidateTelemetrySnapshot:
    """Immutable aggregate for one selected candidate identity.

    Purpose:
        Record how many times a candidate kind/id pair was chosen for one rule
        across the run.

    Invariants:
        `chosen` is an aggregate count emitted by aggregation.

    """

    kind: str
    candidate_id: str
    label: str
    chosen: int


@dataclass(frozen=True, slots=True)
class RuleTelemetrySnapshot:
    """Immutable per-rule telemetry rollup for one execution run.

    Purpose:
        Bundle per-rule counters plus nested condition/candidate aggregates
        emitted by `RuntimeTelemetryAggregator`.

    Key relationships:
        Built from `_RuleMetricsState.to_snapshot()` and consumed by reporting
        and serialization code.

    """

    rule_id: str
    rule_name: str
    evaluations: int
    applied: int
    trigger_firings: int
    proposals_generated: int
    generated_candidates_considered: int
    fallback_candidates_considered: int
    static_candidates_considered: int
    noop_candidates_generated: int
    conditions: tuple[ConditionTelemetrySnapshot, ...]
    candidate_choices: tuple[CandidateTelemetrySnapshot, ...]


@dataclass(frozen=True, slots=True)
class RuntimeTelemetrySnapshot:
    """Immutable top-level telemetry payload attached to one generation result.

    Purpose:
        Carry replayed runtime events, per-rule rollups, and summary counters in
        one transport value.

    Data flow:
        Produced by `RuntimeTelemetryAggregator.build_snapshot(...)` and
        serialized later by `telemetry.representation.telemetry_types`.

    """

    runtime_sec: float | None
    applied_decisions: int
    decision_limit_reached: bool
    rules: tuple[RuleTelemetrySnapshot, ...]
    events: tuple[runtime_events.Event, ...]

    @property
    def rules_triggered_count(self) -> int:
        """Count rules whose trigger fired at least once in this snapshot.

        Purpose:
            Summarize how many rules produced at least one trigger firing across
            the snapshot.

        Architectural role:
            Immutable telemetry value object or helper inside the runtime
            telemetry snapshot boundary.

        Inputs:
            Aggregate counters, guard-condition events, and per-rule telemetry
            values from the runtime projection layer.

        Outputs:
            Stable snapshot values and small summary helpers consumed by
            serialization, reporting, and tests.

        Ownership:
            Owned by `answer_engineering.engine.telemetry.snapshots.snapshots`
            within the engine telemetry boundary.

        Invariants:
            Returns a deterministic derived value for the supplied inputs.

        """
        return sum(1 for rule in self.rules if rule.trigger_firings > 0)

    @property
    def rules_applied_count(self) -> int:
        """Count rules that applied at least one decision in this snapshot.

        Purpose:
            Summarize how many rules were applied at least once across the
            snapshot.

        Architectural role:
            Immutable telemetry value object or helper inside the runtime
            telemetry snapshot boundary.

        Inputs:
            Aggregate counters, guard-condition events, and per-rule telemetry
            values from the runtime projection layer.

        Outputs:
            Stable snapshot values and small summary helpers consumed by
            serialization, reporting, and tests.

        Ownership:
            Owned by `answer_engineering.engine.telemetry.snapshots.snapshots`
            within the engine telemetry boundary.

        Invariants:
            Returns a deterministic derived value for the supplied inputs.

        """
        return sum(1 for rule in self.rules if rule.applied > 0)

    @classmethod
    def empty(
        cls, *, decision_limit_reached: bool = False
    ) -> RuntimeTelemetrySnapshot:
        """Return an empty runtime telemetry snapshot with zero counts and no.

        Purpose:
            Construct the canonical zero-valued telemetry snapshot used when no
            runtime data has been observed.

        Architectural role:
            Immutable telemetry value object or helper inside the runtime
            telemetry snapshot boundary.

        Inputs:
            Aggregate counters, guard-condition events, and per-rule telemetry
            values from the runtime projection layer.

        Outputs:
            Stable snapshot values and small summary helpers consumed by
            serialization, reporting, and tests.

        Ownership:
            Owned by `answer_engineering.engine.telemetry.snapshots.snapshots`
            within the engine telemetry boundary.

        """
        return cls(
            runtime_sec=None,
            applied_decisions=0,
            decision_limit_reached=decision_limit_reached,
            rules=(),
            events=(),
        )
