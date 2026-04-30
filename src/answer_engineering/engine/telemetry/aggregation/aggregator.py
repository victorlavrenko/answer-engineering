"""Project runtime pipeline events into telemetry snapshot state.

Purpose:
    Replay ordered engine events into mutable per-rule counters, then freeze
    that state into immutable runtime telemetry snapshots.

Architectural role:
    Aggregation layer between raw event recording and snapshot values.

Owns:
    - Event replay logic for rule/condition/candidate counters.
    - Reduction of many event kinds into one `RuntimeTelemetrySnapshot`.

Does not own:
    - Event emission or sink behavior while the run is executing.
    - Serialization/reporting formats used outside engine telemetry.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import cast

from answer_engineering.engine.pipeline import (
    events as runtime_events,
)
from answer_engineering.engine.telemetry.snapshots.snapshots import (
    CandidateTelemetrySnapshot,
    ConditionTelemetrySnapshot,
    RuleTelemetrySnapshot,
    RuntimeTelemetrySnapshot,
)

type RuleNameResolver = Callable[[str], str]


@dataclass(slots=True)
class _RuleMetricsState:
    """Mutable per-rule accumulator used while projecting runtime events.

    Purpose:
        Accumulate the mutable counters and per-rule substructures that will
        later be frozen into one `RuleTelemetrySnapshot`.

    Architectural role:
        Projection component that reduces pipeline runtime events into telemetry
        snapshot state.

    Inputs:
        Ordered runtime pipeline events emitted during rule evaluation, proposal
        generation, and acceptance.

    Outputs:
        Mutable aggregate state during replay and immutable telemetry snapshots
        once reduction is complete.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.aggregation.aggregator`
        within the engine telemetry boundary.

    Lifecycle:
        Constructed by runtime or reporting orchestration and reused across the
        local operation scope it serves.

    """

    rule_id: str
    rule_name: str
    evaluations: int = 0
    applied: int = 0
    trigger_firings: int = 0
    proposals_generated: int = 0
    generated_candidates_considered: int = 0
    fallback_candidates_considered: int = 0
    static_candidates_considered: int = 0
    noop_candidates_generated: int = 0
    conditions: dict[str, ConditionTelemetrySnapshot] = field(
        default_factory=lambda: cast(dict[str, ConditionTelemetrySnapshot], {})
    )
    candidate_choices: dict[str, CandidateTelemetrySnapshot] = field(
        default_factory=lambda: cast(dict[str, CandidateTelemetrySnapshot], {})
    )

    def to_snapshot(self) -> RuleTelemetrySnapshot:
        """Convert mutable per-rule aggregate state into an immutable rule.

        Purpose:
            Convert the mutable accumulator into the immutable per-rule snapshot
            consumed by downstream serialization and reporting layers.

        Architectural role:
            Projection component that reduces pipeline runtime events into
            telemetry snapshot state.

        Inputs:
            Ordered runtime pipeline events emitted during rule evaluation,
            proposal generation, and acceptance.

        Outputs:
            Mutable aggregate state during replay and immutable telemetry
            snapshots once reduction is complete.

        Ownership:
            Owned by
            `answer_engineering.engine.telemetry.aggregation.aggregator` within
            the engine telemetry boundary.

        """
        return RuleTelemetrySnapshot(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            evaluations=self.evaluations,
            applied=self.applied,
            trigger_firings=self.trigger_firings,
            proposals_generated=self.proposals_generated,
            generated_candidates_considered=(
                self.generated_candidates_considered
            ),
            fallback_candidates_considered=self.fallback_candidates_considered,
            static_candidates_considered=self.static_candidates_considered,
            noop_candidates_generated=self.noop_candidates_generated,
            conditions=tuple(self.conditions.values()),
            candidate_choices=tuple(self.candidate_choices.values()),
        )


@dataclass(slots=True)
class RuntimeTelemetryAggregator:
    """Accumulate runtime events and build a `RuntimeTelemetrySnapshot`.

    Purpose:
        Provide the structured data and behavior needed for this aggregation
        component without leaking formatting decisions into unrelated code.

    Architectural role:
        Projection component that reduces pipeline runtime events into telemetry
        snapshot state.

    Inputs:
        Ordered runtime pipeline events emitted during rule evaluation, proposal
        generation, and acceptance.

    Outputs:
        Mutable aggregate state during replay and immutable telemetry snapshots
        once reduction is complete.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.aggregation.aggregator`
        within the engine telemetry boundary.

    Lifecycle:
        Constructed by runtime or reporting orchestration and reused across the
        local operation scope it serves.

    """

    rule_name_for: RuleNameResolver
    _events: list[runtime_events.Event] = field(
        default_factory=lambda: cast(list[runtime_events.Event], [])
    )
    _rules: dict[str, _RuleMetricsState] = field(
        default_factory=lambda: cast(dict[str, _RuleMetricsState], {})
    )
    _applied_decisions: int = 0

    def observe_event(self, event: runtime_events.Event) -> None:
        """Observe one runtime event and update projection state.

        Purpose:
            Update in-memory counters and retained raw events from one pipeline
            event while preserving event-order semantics.

        Architectural role:
            Projection component that reduces pipeline runtime events into
            telemetry snapshot state.

        Inputs:
            Ordered runtime pipeline events emitted during rule evaluation,
            proposal generation, and acceptance.

        Outputs:
            Mutable aggregate state during replay and immutable telemetry
            snapshots once reduction is complete.

        Ownership:
            Owned by
            `answer_engineering.engine.telemetry.aggregation.aggregator` within
            the engine telemetry boundary.

        """
        self._events.append(event)

        if isinstance(event, runtime_events.RuleEvaluationStarted):
            rule = self._rule_state(event.rule_id)
            rule.evaluations += 1
            return

        if isinstance(event, runtime_events.GuardConditionEvaluated):
            self._observe_condition_event(event)
            return

        if isinstance(event, runtime_events.ProposalsGenerated):
            rule = self._rule_state(event.rule_id)
            rule.trigger_firings += 1
            rule.proposals_generated += int(event.proposals_count)
            rule.generated_candidates_considered += int(event.generated_count)
            rule.fallback_candidates_considered += int(event.fallback_count)
            rule.static_candidates_considered += int(event.static_count)
            rule.noop_candidates_generated += int(event.noop_count)
            return

        if isinstance(event, runtime_events.ProposalAccepted):
            self._observe_proposal_accepted(event)

    def observe_events(self, events: Iterable[runtime_events.Event]) -> None:
        """Observe an iterable of runtime events in order.

        Purpose:
            Replay a runtime event sequence through `observe_event` so bulk
            callers share the same reduction logic.

        Architectural role:
            Projection component that reduces pipeline runtime events into
            telemetry snapshot state.

        Inputs:
            Ordered runtime pipeline events emitted during rule evaluation,
            proposal generation, and acceptance.

        Outputs:
            Mutable aggregate state during replay and immutable telemetry
            snapshots once reduction is complete.

        Ownership:
            Owned by
            `answer_engineering.engine.telemetry.aggregation.aggregator` within
            the engine telemetry boundary.

        """
        for event in events:
            self.observe_event(event)

    def build_snapshot(
        self, *, decision_limit_reached: bool
    ) -> RuntimeTelemetrySnapshot:
        """Build the immutable runtime telemetry snapshot from current.

        Purpose:
            Freeze the accumulated mutable state into the immutable runtime
            telemetry snapshot returned to downstream reporting code.

        Architectural role:
            Projection component that reduces pipeline runtime events into
            telemetry snapshot state.

        Inputs:
            Ordered runtime pipeline events emitted during rule evaluation,
            proposal generation, and acceptance.

        Outputs:
            Mutable aggregate state during replay and immutable telemetry
            snapshots once reduction is complete.

        Ownership:
            Owned by
            `answer_engineering.engine.telemetry.aggregation.aggregator` within
            the engine telemetry boundary.

        """
        ordered_rules = tuple(
            self._rules[rule_id].to_snapshot()
            for rule_id in sorted(self._rules)
        )
        return RuntimeTelemetrySnapshot(
            runtime_sec=None,
            applied_decisions=self._applied_decisions,
            decision_limit_reached=decision_limit_reached,
            rules=ordered_rules,
            events=tuple(self._events),
        )

    def _rule_state(self, rule_id: str) -> _RuleMetricsState:
        """Return the mutable aggregate state for one rule, creating it on.

        Purpose:
            Return the per-rule accumulator, creating and naming it on first
            touch so later event handling can mutate counters in place.

        Architectural role:
            Projection component that reduces pipeline runtime events into
            telemetry snapshot state.

        Inputs:
            Ordered runtime pipeline events emitted during rule evaluation,
            proposal generation, and acceptance.

        Outputs:
            Mutable aggregate state during replay and immutable telemetry
            snapshots once reduction is complete.

        Ownership:
            Owned by
            `answer_engineering.engine.telemetry.aggregation.aggregator` within
            the engine telemetry boundary.

        """
        existing = self._rules.get(rule_id)
        if existing is not None:
            return existing
        created = _RuleMetricsState(
            rule_id=rule_id,
            rule_name=self.rule_name_for(rule_id),
        )
        self._rules[rule_id] = created
        return created

    def _observe_condition_event(
        self, event: runtime_events.GuardConditionEvaluated
    ) -> None:
        rule = self._rule_state(event.rule_id)
        condition = ConditionTelemetrySnapshot.from_event(event)
        existing = rule.conditions.get(condition.condition_id)
        if existing is None:
            rule.conditions[condition.condition_id] = condition
            return
        rule.conditions[condition.condition_id] = ConditionTelemetrySnapshot(
            condition_id=existing.condition_id,
            node_path=existing.node_path,
            node_type=existing.node_type,
            debug_expression=existing.debug_expression,
            matched=existing.matched + condition.matched,
            seen=existing.seen + condition.seen,
        )

    def _observe_proposal_accepted(
        self, event: runtime_events.ProposalAccepted
    ) -> None:
        rule = self._rule_state(event.rule_id)
        rule.applied += 1
        self._applied_decisions += 1
        candidate_id = event.candidate_id.strip()
        candidate_kind = event.candidate_kind.strip()
        candidate_label = event.candidate_label.strip()
        if not candidate_id:
            raise ValueError("ProposalAccepted.candidate_id must be non-empty")
        if not candidate_kind:
            raise ValueError(
                "ProposalAccepted.candidate_kind must be non-empty"
            )
        if not candidate_label:
            raise ValueError(
                "ProposalAccepted.candidate_label must be non-empty"
            )
        candidate_key = f"{candidate_kind}:{candidate_id}"
        existing = rule.candidate_choices.get(candidate_key)
        if existing is None:
            rule.candidate_choices[candidate_key] = CandidateTelemetrySnapshot(
                kind=candidate_kind,
                candidate_id=candidate_id,
                label=candidate_label,
                chosen=1,
            )
            return
        rule.candidate_choices[candidate_key] = CandidateTelemetrySnapshot(
            kind=existing.kind,
            candidate_id=existing.candidate_id,
            label=candidate_label,
            chosen=existing.chosen + 1,
        )
