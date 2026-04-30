"""Decision-log event shaping for runtime candidate selection.

Purpose:
    Define decision-log records and formatting/emission helpers that explain why
    one proposal candidate won within a grouped decision.

Architectural role:
    Event-layer helper for human inspection of runtime choices.

Owns:
    - Structured decision-log records (`DecisionEvent`, `CandidateRow`).
    - Formatting/grouping helpers for scored candidates.
    - Emission of already-formatted decision-log text via debug sinks.

Does not own:
    - Runtime event sink transport and buffering policy (`event_sink`).
    - Aggregated telemetry counters or snapshot production.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite

from answer_engineering.engine.runtime.runtime_types import (
    PatchOp,
    PatchProposal,
)
from answer_engineering.engine.telemetry.events.event_sink import (
    DebugEventEmitter,
)


@dataclass(frozen=True, slots=True)
class CandidateRow:
    """One formatted candidate row inside a decision log event.

    Purpose:
        Carry one already-shaped reporting row so downstream serializers and
        table renderers can operate on stable field names.

    Architectural role:
        Decision-logging helper inside the engine telemetry observability
        boundary.

    Inputs:
        Scored candidates, patch proposals, and span/context data supplied by
        orchestration and runtime formatting callers.

    Outputs:
        Readable decision-log text, grouped candidate structures, or lightweight
        decision value objects consumed by debug output.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.decision_logging`
        within the engine telemetry boundary.

    Invariants:
        Instances are transport-oriented values and should remain stable after
        construction.

    """

    rank: int
    name: str
    score: float
    ratio_to_best: float
    is_winner: bool
    text_excerpt: str


@dataclass(frozen=True, slots=True, init=False)
class DecisionEvent:
    """Structured record describing one scored decision and the applied winner.

    Purpose:
        Provide the structured data and behavior needed for this decision
        logging component without leaking formatting decisions into unrelated
        code.

    Architectural role:
        Decision-logging helper inside the engine telemetry observability
        boundary.

    Inputs:
        Scored candidates, patch proposals, and span/context data supplied by
        orchestration and runtime formatting callers.

    Outputs:
        Readable decision-log text, grouped candidate structures, or lightweight
        decision value objects consumed by debug output.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.decision_logging`
        within the engine telemetry boundary.

    Invariants:
        Instances are transport-oriented values and should remain stable after
        construction.

    """

    event_id: int
    scope: str
    op: PatchOp
    span_start: int | None
    span_end: int | None
    rule_key: str
    rule_id_full: str
    rule_id_short: str
    priority: int
    repeat: bool
    around_excerpt: str
    old_excerpt: str
    new_excerpt: str
    winner_name: str
    winner_score: float
    gap2: float
    ratio2: float
    candidates: tuple[CandidateRow, ...]

    def __init__(
        self,
        *,
        event_id: int,
        scope: str,
        op: PatchOp,
        span_start: int | None,
        span_end: int | None,
        rule_key: str,
        rule_id_full: str,
        rule_id_short: str,
        priority: int,
        repeat: bool,
        around_excerpt: str,
        old_excerpt: str,
        new_excerpt: str,
        winner_name: str,
        winner_score: float,
        gap2: float,
        ratio2: float,
        candidates: list[CandidateRow] | tuple[CandidateRow, ...],
    ) -> None:
        """Freeze one decision-event payload and normalize candidate records.

        Purpose:
            Construct a stable telemetry event for one decision-log moment while
            normalizing candidate objects into the tuple shape used by renderers
            and tests.

        Architectural role:
            Telemetry-construction boundary between mutable runtime decision
            data and immutable reporting events.

        Inputs (architectural provenance):
            Receives event kind, step/rule metadata, optional document spans,
            selected candidate data, and candidate collections from
            orchestration or selection.

        Outputs (downstream usage):
            Stores an immutable decision event consumed by telemetry snapshots,
            decision-log formatting, golden tests, and debugging output.

        Invariants/constraints:
            Candidate collections are frozen during construction. Stored
            metadata should describe one decision event and should remain
            JSON/reporting friendly.

        """
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "span_start", span_start)
        object.__setattr__(self, "span_end", span_end)
        object.__setattr__(self, "rule_key", rule_key)
        object.__setattr__(self, "rule_id_full", rule_id_full)
        object.__setattr__(self, "rule_id_short", rule_id_short)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "repeat", repeat)
        object.__setattr__(self, "around_excerpt", around_excerpt)
        object.__setattr__(self, "old_excerpt", old_excerpt)
        object.__setattr__(self, "new_excerpt", new_excerpt)
        object.__setattr__(self, "winner_name", winner_name)
        object.__setattr__(self, "winner_score", winner_score)
        object.__setattr__(self, "gap2", gap2)
        object.__setattr__(self, "ratio2", ratio2)
        object.__setattr__(self, "candidates", tuple(candidates))


@dataclass(frozen=True, slots=True)
class DecisionGroupContext:
    """Context describing the text span and source material for one grouped.

    Purpose:
        Bundle the contextual identifiers, paths, and aggregate values that
        multiple reporting or upload steps need to share consistently.

    Architectural role:
        Decision-logging helper inside the engine telemetry observability
        boundary.

    Inputs:
        Scored candidates, patch proposals, and span/context data supplied by
        orchestration and runtime formatting callers.

    Outputs:
        Readable decision-log text, grouped candidate structures, or lightweight
        decision value objects consumed by debug output.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.decision_logging`
        within the engine telemetry boundary.

    Invariants:
        Instances are transport-oriented values and should remain stable after
        construction.

    """

    rule_key: str
    rule_id_full: str
    rule_id_short: str
    priority: int
    repeat: bool
    guard_span: tuple[int, int]
    guard_text: str
    edit_span: tuple[int, int]
    edit_text: str
    doc_text: str


@dataclass(frozen=True, slots=True)
class ScoredCandidateRecord:
    """Carrier for one scored candidate before final decision-log formatting.

    Purpose:
        Provide the structured data and behavior needed for this decision
        logging component without leaking formatting decisions into unrelated
        code.

    Architectural role:
        Decision-logging helper inside the engine telemetry observability
        boundary.

    Inputs:
        Scored candidates, patch proposals, and span/context data supplied by
        orchestration and runtime formatting callers.

    Outputs:
        Readable decision-log text, grouped candidate structures, or lightweight
        decision value objects consumed by debug output.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.decision_logging`
        within the engine telemetry boundary.

    Invariants:
        Instances are transport-oriented values and should remain stable after
        construction.

    """

    proposal: PatchProposal
    score: float
    prob_ratio_to_best: float | None = None


@dataclass(slots=True)
class DecisionEmitter:
    """Emit human-readable decision logs through the debug-text channel.

    Purpose:
        Own the final emission step for already-prepared debug or decision-log
        text.

    Architectural role:
        Decision-logging helper inside the engine telemetry observability
        boundary.

    Inputs:
        Scored candidates, patch proposals, and span/context data supplied by
        orchestration and runtime formatting callers.

    Outputs:
        Readable decision-log text, grouped candidate structures, or lightweight
        decision value objects consumed by debug output.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.decision_logging`
        within the engine telemetry boundary.

    Lifecycle:
        Constructed by runtime or reporting orchestration and reused across the
        local operation scope it serves.

    """

    enabled: bool
    next_event_id: Callable[[], int]
    formatter: DecisionFormatter
    debug_emitter: DebugEventEmitter = field(default_factory=DebugEventEmitter)

    def begin(
        self,
        *,
        context: DecisionGroupContext,
        span: tuple[int, int] | None,
        op: PatchOp,
        n: int,
    ) -> int:
        """Emit the header lines for one decision episode and return the.

        Purpose:
            Start one decision-log episode by emitting the shared header, rule
            metadata, span views, and candidate count before ranked rows are
            written.

        Architectural role:
            Decision-logging helper inside the engine telemetry observability
            boundary.

        Inputs:
            Scored candidates, patch proposals, and span/context data supplied
            by orchestration and runtime formatting callers.

        Outputs:
            Readable decision-log text, grouped candidate structures, or
            lightweight decision value objects consumed by debug output.

        Ownership:
            Owned by
            `answer_engineering.engine.telemetry.events.decision_logging` within
            the engine telemetry boundary.

        """
        event_id = int(self.next_event_id())
        if not self.enabled:
            return event_id
        span_fields = _format_span_fields(span)
        self._emit_core_verbose_line(
            f"DECISION #{event_id} scope=core op={_op_label(op)} {span_fields}",
        )
        repeat_mode = str(context.repeat).lower()
        self._emit_core_verbose_line(
            f"rule: {context.rule_key} "
            f"(id={context.rule_id_short}, full_id={context.rule_id_full}, "
            f"pri={context.priority}, repeat={repeat_mode})",
        )
        self._emit_core_verbose_line(
            (
                "guard_view: "
                f"span={context.guard_span[0]}:{context.guard_span[1]} "
                f"len={context.guard_span[1] - context.guard_span[0]} "
                f'text="{_visible_text(context.guard_text)}"'
            ),
        )
        self._emit_core_verbose_line(
            (
                "edit_view: "
                f"span={context.edit_span[0]}:{context.edit_span[1]} "
                f"len={context.edit_span[1] - context.edit_span[0]} "
                f'text="{_visible_text(context.edit_text)}"'
            ),
        )
        safe_span = span or (0, 0)
        around_excerpt = self.formatter.context_excerpt(
            context.doc_text, safe_span
        )
        self._emit_core_verbose_line(f'around: "{around_excerpt}"')
        self._emit_core_verbose_line(f"candidates ({n}):")
        return event_id

    def row(
        self,
        *,
        event_id: int,
        rank: int,
        candidate_name: str,
        score: float,
        prob_ratio_to_best: float | None,
        is_winner: bool,
        text_excerpt: str,
    ) -> None:
        """Emit one ranked candidate line for an in-progress decision episode.

        Purpose:
            Append one ranked candidate line for the current decision episode,
            including score, probability ratio, winner marker, and formatted
            text excerpt.

        Architectural role:
            Decision-logging helper inside the engine telemetry observability
            boundary.

        Inputs:
            Scored candidates, patch proposals, and span/context data supplied
            by orchestration and runtime formatting callers.

        Outputs:
            Readable decision-log text, grouped candidate structures, or
            lightweight decision value objects consumed by debug output.

        Ownership:
            Owned by
            `answer_engineering.engine.telemetry.events.decision_logging` within
            the engine telemetry boundary.

        """
        del event_id
        if not self.enabled:
            return
        ratio_text = (
            _fmt_float(prob_ratio_to_best)
            if prob_ratio_to_best is not None
            else "n/a"
        )
        winner_tick = "✓" if is_winner else " "
        self._emit_core_verbose_line(
            f"  {rank}) {candidate_name} score={_fmt_float(score)} "
            f'ratio={ratio_text} {winner_tick} text="{text_excerpt}"',
        )

    def end(self, *, event: DecisionEvent) -> None:
        """Emit the winning decision summary and the old/new text excerpts for.

        Purpose:
            Close the decision episode by emitting the winner summary together
            with the old and new excerpts of the applied edit.

        Architectural role:
            Decision-logging helper inside the engine telemetry observability
            boundary.

        Inputs:
            Scored candidates, patch proposals, and span/context data supplied
            by orchestration and runtime formatting callers.

        Outputs:
            Readable decision-log text, grouped candidate structures, or
            lightweight decision value objects consumed by debug output.

        Ownership:
            Owned by
            `answer_engineering.engine.telemetry.events.decision_logging` within
            the engine telemetry boundary.

        """
        if not self.enabled:
            return
        winner_line = (
            f"DECISION #{event.event_id} winner: {event.winner_name} "
            f"score={_fmt_float(event.winner_score)}  "
            f"gap2={_fmt_float(event.gap2)}  ratio2={_fmt_float(event.ratio2)}"
        )
        self._emit_core_verbose_line(winner_line)
        self._emit_core_verbose_line(f"DECISION #{event.event_id} apply:")
        self._emit_core_verbose_line(f'  old: "{event.old_excerpt}"')
        self._emit_core_verbose_line(f'  new: "{event.new_excerpt}"')

    def _emit_core_verbose_line(self, line: str) -> None:
        """Forward one already-formatted decision-log line to the debug emitter.

        Purpose:
            Guard debug-line emission behind the enabled flag so callers can
            format once and delegate the final write step here.

        Architectural role:
            Decision-logging helper inside the engine telemetry observability
            boundary.

        Inputs:
            Scored candidates, patch proposals, and span/context data supplied
            by orchestration and runtime formatting callers.

        Outputs:
            Readable decision-log text, grouped candidate structures, or
            lightweight decision value objects consumed by debug output.

        Ownership:
            Owned by
            `answer_engineering.engine.telemetry.events.decision_logging` within
            the engine telemetry boundary.

        """
        if not self.enabled:
            return
        self.debug_emitter.emit(line)


@dataclass(frozen=True, slots=True)
class DecisionFormatter:
    """Format scored decision data into excerpts, labels, and structured.

    Purpose:
        Centralize the string-shaping rules that keep decision and reporting
        output stable, bounded, and readable.

    Architectural role:
        Decision-logging helper inside the engine telemetry observability
        boundary.

    Inputs:
        Scored candidates, patch proposals, and span/context data supplied by
        orchestration and runtime formatting callers.

    Outputs:
        Readable decision-log text, grouped candidate structures, or lightweight
        decision value objects consumed by debug output.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.decision_logging`
        within the engine telemetry boundary.

    Lifecycle:
        Constructed by runtime or reporting orchestration and reused across the
        local operation scope it serves.

    """

    excerpt_limit: int = 110
    context_limit: int = 140

    def excerpt(self, s: str, limit: int | None = None) -> str:
        """Trim text to a readable excerpt for decision-log presentation.

        Purpose:
            Normalize whitespace and enforce the configured character budget so
            logged text remains readable and bounded.

        Architectural role:
            Decision-logging helper inside the engine telemetry observability
            boundary.

        Inputs:
            Scored candidates, patch proposals, and span/context data supplied
            by orchestration and runtime formatting callers.

        Outputs:
            Readable decision-log text, grouped candidate structures, or
            lightweight decision value objects consumed by debug output.

        Ownership:
            Owned by
            `answer_engineering.engine.telemetry.events.decision_logging` within
            the engine telemetry boundary.

        """
        compact = " ".join(s.split())
        effective_limit = self.excerpt_limit if limit is None else limit
        if len(compact) <= effective_limit:
            return compact
        return f"{compact[: effective_limit - 1]}…"

    def context_excerpt(
        self,
        doc: str,
        span: tuple[int, int],
        limit: int | None = None,
    ) -> str:
        """Extract a bounded context window around one decision span.

        Purpose:
            Build a centered context window around the selected span and mark
            the focal text before truncation.

        Architectural role:
            Decision-logging helper inside the engine telemetry observability
            boundary.

        Inputs:
            Scored candidates, patch proposals, and span/context data supplied
            by orchestration and runtime formatting callers.

        Outputs:
            Readable decision-log text, grouped candidate structures, or
            lightweight decision value objects consumed by debug output.

        Ownership:
            Owned by
            `answer_engineering.engine.telemetry.events.decision_logging` within
            the engine telemetry boundary.

        """
        i0, i1 = span
        effective_limit = self.context_limit if limit is None else limit
        half = max(1, effective_limit // 2)
        left = doc[max(0, i0 - half) : i0]
        middle = doc[i0:i1]
        right = doc[i1 : min(len(doc), i1 + half)]
        return self.excerpt(f"{left}~~{middle}~~{right}", limit=effective_limit)

    def apply_excerpts(
        self, doc_text: str, proposal: PatchProposal
    ) -> tuple[str, str]:
        """Produce old/new text excerpts describing the applied edit.

        Purpose:
            Derive stable old/new snippets for one proposal so the applied edit
            can be shown without dumping the full document.

        Architectural role:
            Decision-logging helper inside the engine telemetry observability
            boundary.

        Inputs:
            Scored candidates, patch proposals, and span/context data supplied
            by orchestration and runtime formatting callers.

        Outputs:
            Readable decision-log text, grouped candidate structures, or
            lightweight decision value objects consumed by debug output.

        Ownership:
            Owned by
            `answer_engineering.engine.telemetry.events.decision_logging` within
            the engine telemetry boundary.

        """
        if proposal.span_abs is None:
            return "", self.excerpt(proposal.payload or "")
        i0, i1 = proposal.span_abs
        old_text = self.excerpt(doc_text[i0:i1])
        payload = (
            proposal.payload_norm
            if proposal.payload_norm is not None
            else (proposal.payload or "")
        )
        if proposal.op == PatchOp.DELETE:
            return old_text, ""
        if proposal.op in (PatchOp.INSERT_AFTER, PatchOp.INSERT_BEFORE):
            return old_text, self.excerpt(doc_text[i0:i1] + payload)
        return old_text, self.excerpt(payload)

    def candidate_text_excerpt(
        self, doc_text: str, proposal: PatchProposal
    ) -> str:
        """Extract the candidate text that should appear in decision logs.

        Purpose:
            Choose the candidate-facing text snippet that best represents the
            proposal in ranked decision output.

        Architectural role:
            Decision-logging helper inside the engine telemetry observability
            boundary.

        Inputs:
            Scored candidates, patch proposals, and span/context data supplied
            by orchestration and runtime formatting callers.

        Outputs:
            Readable decision-log text, grouped candidate structures, or
            lightweight decision value objects consumed by debug output.

        Ownership:
            Owned by
            `answer_engineering.engine.telemetry.events.decision_logging` within
            the engine telemetry boundary.

        """
        if proposal.op == PatchOp.DELETE:
            return "<delete>"
        return self.apply_excerpts(doc_text, proposal)[1]


def short_id(full: str) -> str:
    """Shorten a long identifier for readable decision-log output."""
    if len(full) <= 18:
        return full
    return f"{full[:8]}..{full[-8:]}"


def conflicts(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Apply decision span-overlap rules, including insertion-point cases."""
    a0, a1 = a
    b0, b1 = b
    if a0 == a1 and b0 == b1:
        return a0 == b0
    if a0 == a1:
        return b0 <= a0 <= b1
    if b0 == b1:
        return a0 <= b0 <= a1
    return max(a0, b0) < min(a1, b1)


def group_candidates_by_span(
    candidates: list[ScoredCandidateRecord],
) -> dict[tuple[tuple[int, int] | None, str], list[ScoredCandidateRecord]]:
    """Group scored candidates by absolute span and patch operation."""
    grouped: dict[
        tuple[tuple[int, int] | None, str], list[ScoredCandidateRecord]
    ] = {}
    for item in candidates:
        key = (item.proposal.span_abs, item.proposal.op.value)
        grouped.setdefault(key, []).append(item)
    return grouped


def _format_span_fields(span: tuple[int, int] | None) -> str:
    """Format span coordinates for inclusion in a structured decision log.

    Purpose:
        Render a span tuple into the `start=... end=... len=...` text fragment
        used in decision headers.

    Architectural role:
        Decision-logging helper inside the engine telemetry observability
        boundary.

    Inputs:
        Scored candidates, patch proposals, and span/context data supplied by
        orchestration and runtime formatting callers.

    Outputs:
        Readable decision-log text, grouped candidate structures, or lightweight
        decision value objects consumed by debug output.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.decision_logging`
        within the engine telemetry boundary.

    """
    if span is None:
        return "span=none len=0"
    return f"span={span[0]}:{span[1]} len={span[1] - span[0]}"


def _fmt_float(value: float) -> str:
    """Format a floating-point value for stable human-readable decision logs.

    Purpose:
        Format finite and non-finite numeric values into stable short strings
        for verbose decision output.

    Architectural role:
        Decision-logging helper inside the engine telemetry observability
        boundary.

    Inputs:
        Scored candidates, patch proposals, and span/context data supplied by
        orchestration and runtime formatting callers.

    Outputs:
        Readable decision-log text, grouped candidate structures, or lightweight
        decision value objects consumed by debug output.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.decision_logging`
        within the engine telemetry boundary.

    """
    return f"{value:.4f}" if isfinite(value) else str(value)


def _visible_text(value: str) -> str:
    """Normalize text so decision-log excerpts remain readable in one-line.

    Purpose:
        Escape control-like whitespace so logged snippets stay single-line and
        visually unambiguous.

    Architectural role:
        Decision-logging helper inside the engine telemetry observability
        boundary.

    Inputs:
        Scored candidates, patch proposals, and span/context data supplied by
        orchestration and runtime formatting callers.

    Outputs:
        Readable decision-log text, grouped candidate structures, or lightweight
        decision value objects consumed by debug output.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.decision_logging`
        within the engine telemetry boundary.

    """
    return value.encode("unicode_escape").decode("ascii")


def _op_label(op: PatchOp) -> str:
    """Return the short operation label shown for one candidate/edit kind.

    Purpose:
        Map the patch operation enum to the short label used in decision-log
        headers.

    Architectural role:
        Decision-logging helper inside the engine telemetry observability
        boundary.

    Inputs:
        Scored candidates, patch proposals, and span/context data supplied by
        orchestration and runtime formatting callers.

    Outputs:
        Readable decision-log text, grouped candidate structures, or lightweight
        decision value objects consumed by debug output.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.decision_logging`
        within the engine telemetry boundary.

    """
    if op in (PatchOp.INSERT_AFTER, PatchOp.INSERT_BEFORE):
        return "INSERT"
    return op.value.upper()
