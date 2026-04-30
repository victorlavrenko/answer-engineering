"""Runtime event sink contracts and concrete sink implementations.

Purpose:
    Own transport policy for runtime events and debug lines: buffer, fan out,
    print, discard, or keep in memory for later inspection.

Architectural role:
    Event-ingestion boundary between pipeline execution and telemetry consumers.

Architectural direction:
    Keep transport concerns isolated here while maintaining explicit boundaries
    from decision formatting and telemetry aggregation semantics.

Why this matters:
    Event transport is a central seam, and boundary drift can blur ownership
    between ingestion, formatting, and aggregation logic.

What better would look like:
    Runtime producers depend on stable sink contracts while downstream
    formatting and aggregation evolve independently.

How improvement can be recognized:
    - Clearer separation between event transport, formatting, and aggregation
    - Fewer cross-layer dependencies from sinks into report semantics
    - Stable sink interfaces with predictable behavior across runtime contexts

Open constraint:
    Sink capabilities should remain adaptable to new observability and reporting
    consumers.

"""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from answer_engineering.engine.pipeline.events import (
    DebugEvent,
    Event,
)
from answer_engineering.infra.console.contracts import (
    DebugSafeLineEmitter,
)


class Clock(Protocol):
    """Timestamp provider used when wrapping emitted runtime events.

    Purpose:
        Decouple event-envelope timestamp generation from concrete sink
        implementations so tests and deterministic runs can inject their own
        time source.

    Used by:
        `RuntimeEventSink.with_event_envelope(...)`.

    Not intended for:
        Formatting decision logs or mutating runtime events.

    """

    def now(self) -> str:
        """Return an event-envelope timestamp string."""
        raise NotImplementedError


class IdGenerator(Protocol):
    """Event id provider for runtime-event envelopes.

    Purpose:
        Decouple envelope id generation from sink implementations so tests can
        use deterministic ids and production can use UUIDs.

    Used by:
        `RuntimeEventSink.with_event_envelope(...)`.

    """

    def next_id(self) -> str:
        """Return the next event id string for envelope attachment."""
        raise NotImplementedError


@dataclass(slots=True)
class UtcClock(Clock):
    """Clock implementation returning the current UTC timestamp in ISO-like.

    Purpose:
        Provide the structured data and behavior needed for this event sink
        component without leaking formatting decisions into unrelated code.

    Architectural role:
        Runtime event sink or debug-output helper inside the engine telemetry
        observability boundary.

    Inputs:
        Runtime events or already-formatted debug lines emitted by
        orchestration, decode, or telemetry helpers.

    Outputs:
        Forwarded, buffered, discarded, or displayed runtime events and debug
        lines according to the sink implementation.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.event_sink` within
        the engine telemetry boundary.

    """

    def now(self) -> str:
        """Return the current UTC timestamp string used for runtime event.

        Purpose:
            Supply the primitive value required by sink infrastructure through a
            narrow protocol role instead of hard-coding one implementation.

        Architectural role:
            Runtime event sink or debug-output helper inside the engine
            telemetry observability boundary.

        Inputs:
            Runtime events or already-formatted debug lines emitted by
            orchestration, decode, or telemetry helpers.

        Outputs:
            Forwarded, buffered, discarded, or displayed runtime events and
            debug lines according to the sink implementation.

        Ownership:
            Owned by `answer_engineering.engine.telemetry.events.event_sink`
            within the engine telemetry boundary.

        """
        return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class UuidGenerator(IdGenerator):
    """Id generator implementation returning random UUID-based event ids.

    Purpose:
        Provide the structured data and behavior needed for this event sink
        component without leaking formatting decisions into unrelated code.

    Architectural role:
        Runtime event sink or debug-output helper inside the engine telemetry
        observability boundary.

    Inputs:
        Runtime events or already-formatted debug lines emitted by
        orchestration, decode, or telemetry helpers.

    Outputs:
        Forwarded, buffered, discarded, or displayed runtime events and debug
        lines according to the sink implementation.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.event_sink` within
        the engine telemetry boundary.

    """

    def next_id(self) -> str:
        """Return a fresh UUID-derived event id.

        Purpose:
            Supply the primitive value required by sink infrastructure through a
            narrow protocol role instead of hard-coding one implementation.

        Architectural role:
            Runtime event sink or debug-output helper inside the engine
            telemetry observability boundary.

        Inputs:
            Runtime events or already-formatted debug lines emitted by
            orchestration, decode, or telemetry helpers.

        Outputs:
            Forwarded, buffered, discarded, or displayed runtime events and
            debug lines according to the sink implementation.

        Ownership:
            Owned by `answer_engineering.engine.telemetry.events.event_sink`
            within the engine telemetry boundary.

        """
        return uuid4().hex


class RuntimeEventSink(Protocol):
    """Protocol for components that receive runtime events.

    Purpose:
        Define or implement one event-delivery role used to capture, forward,
        discard, or present runtime events.

    Architectural role:
        Runtime event sink or debug-output helper inside the engine telemetry
        observability boundary.

    Inputs:
        Runtime events or already-formatted debug lines emitted by
        orchestration, decode, or telemetry helpers.

    Outputs:
        Forwarded, buffered, discarded, or displayed runtime events and debug
        lines according to the sink implementation.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.event_sink` within
        the engine telemetry boundary.

    """

    def emit(self, event: Event) -> None:
        """Receive one runtime event from execution code.

        Purpose:
            Perform the concrete event-delivery step for this sink or emitter
            implementation while preserving the runtime-side call contract.

        Architectural role:
            Runtime event sink or debug-output helper inside the engine
            telemetry observability boundary.

        Inputs:
            Runtime events or already-formatted debug lines emitted by
            orchestration, decode, or telemetry helpers.

        Outputs:
            Forwarded, buffered, discarded, or displayed runtime events and
            debug lines according to the sink implementation.

        Ownership:
            Owned by `answer_engineering.engine.telemetry.events.event_sink`
            within the engine telemetry boundary.

        """
        raise NotImplementedError


@dataclass(slots=True)
class CompositeRuntimeEventSink(RuntimeEventSink):
    """Fan out each runtime event to multiple downstream sinks.

    Purpose:
        Define or implement one event-delivery role used to capture, forward,
        discard, or present runtime events.

    Architectural role:
        Runtime event sink or debug-output helper inside the engine telemetry
        observability boundary.

    Inputs:
        Runtime events or already-formatted debug lines emitted by
        orchestration, decode, or telemetry helpers.

    Outputs:
        Forwarded, buffered, discarded, or displayed runtime events and debug
        lines according to the sink implementation.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.event_sink` within
        the engine telemetry boundary.

    """

    sinks: tuple[RuntimeEventSink, ...]

    def emit(self, event: Event) -> None:
        """Forward one runtime event to every configured child sink in order.

        Purpose:
            Perform the concrete event-delivery step for this sink or emitter
            implementation while preserving the runtime-side call contract.

        Architectural role:
            Runtime event sink or debug-output helper inside the engine
            telemetry observability boundary.

        Inputs:
            Runtime events or already-formatted debug lines emitted by
            orchestration, decode, or telemetry helpers.

        Outputs:
            Forwarded, buffered, discarded, or displayed runtime events and
            debug lines according to the sink implementation.

        Ownership:
            Owned by `answer_engineering.engine.telemetry.events.event_sink`
            within the engine telemetry boundary.

        """
        for sink in self.sinks:
            sink.emit(event)


@dataclass(slots=True)
class NullRuntimeEventSink(RuntimeEventSink):
    """Discard all emitted runtime events.

    Purpose:
        Define or implement one event-delivery role used to capture, forward,
        discard, or present runtime events.

    Architectural role:
        Runtime event sink or debug-output helper inside the engine telemetry
        observability boundary.

    Inputs:
        Runtime events or already-formatted debug lines emitted by
        orchestration, decode, or telemetry helpers.

    Outputs:
        Forwarded, buffered, discarded, or displayed runtime events and debug
        lines according to the sink implementation.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.event_sink` within
        the engine telemetry boundary.

    """

    def emit(self, event: Event) -> None:
        """Intentionally ignore one emitted runtime event.

        Purpose:
            Perform the concrete event-delivery step for this sink or emitter
            implementation while preserving the runtime-side call contract.

        Architectural role:
            Runtime event sink or debug-output helper inside the engine
            telemetry observability boundary.

        Inputs:
            Runtime events or already-formatted debug lines emitted by
            orchestration, decode, or telemetry helpers.

        Outputs:
            Forwarded, buffered, discarded, or displayed runtime events and
            debug lines according to the sink implementation.

        Ownership:
            Owned by `answer_engineering.engine.telemetry.events.event_sink`
            within the engine telemetry boundary.

        """
        del event


def _make_event_list() -> list[Event]:
    return list()


@dataclass(slots=True)
class RecordingRuntimeEventSink(RuntimeEventSink):
    """Append each emitted runtime event to an in-memory list without modifying.

    Purpose:
        Define or implement one event-delivery role used to capture, forward,
        discard, or present runtime events.

    Architectural role:
        Runtime event sink or debug-output helper inside the engine telemetry
        observability boundary.

    Inputs:
        Runtime events or already-formatted debug lines emitted by
        orchestration, decode, or telemetry helpers.

    Outputs:
        Forwarded, buffered, discarded, or displayed runtime events and debug
        lines according to the sink implementation.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.event_sink` within
        the engine telemetry boundary.

    """

    delegate: RuntimeEventSink = field(default_factory=NullRuntimeEventSink)
    events: list[Event] = field(default_factory=_make_event_list)

    def emit(self, event: Event) -> None:
        """Append one runtime event to the in-memory recording buffer.

        Purpose:
            Perform the concrete event-delivery step for this sink or emitter
            implementation while preserving the runtime-side call contract.

        Architectural role:
            Runtime event sink or debug-output helper inside the engine
            telemetry observability boundary.

        Inputs:
            Runtime events or already-formatted debug lines emitted by
            orchestration, decode, or telemetry helpers.

        Outputs:
            Forwarded, buffered, discarded, or displayed runtime events and
            debug lines according to the sink implementation.

        Ownership:
            Owned by `answer_engineering.engine.telemetry.events.event_sink`
            within the engine telemetry boundary.

        """
        self.delegate.emit(event)
        self.events.append(event)


@dataclass(slots=True)
class DebugTextEmitter:
    """Emit raw debug text through a callback only when debugging is enabled.

    Purpose:
        Own the final emission step for already-prepared debug or decision-log
        text.

    Architectural role:
        Runtime event sink or debug-output helper inside the engine telemetry
        observability boundary.

    Inputs:
        Runtime events or already-formatted debug lines emitted by
        orchestration, decode, or telemetry helpers.

    Outputs:
        Forwarded, buffered, discarded, or displayed runtime events and debug
        lines according to the sink implementation.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.event_sink` within
        the engine telemetry boundary.

    """

    enabled: bool = True
    debug_line_emitter: DebugSafeLineEmitter | None = None
    prefix: str = "[AE]"

    def emit(self, msg: str) -> None:
        """Forward one debug text line when debugging is enabled.

        Purpose:
            Perform the concrete event-delivery step for this sink or emitter
            implementation while preserving the runtime-side call contract.

        Architectural role:
            Runtime event sink or debug-output helper inside the engine
            telemetry observability boundary.

        Inputs:
            Runtime events or already-formatted debug lines emitted by
            orchestration, decode, or telemetry helpers.

        Outputs:
            Forwarded, buffered, discarded, or displayed runtime events and
            debug lines according to the sink implementation.

        Ownership:
            Owned by `answer_engineering.engine.telemetry.events.event_sink`
            within the engine telemetry boundary.

        """
        if not self.enabled:
            return
        if self.debug_line_emitter is not None:
            self.debug_line_emitter.emit_debug_line(msg)
            return
        builtins.print(f"{self.prefix} {msg}", flush=True)


@dataclass(slots=True)
class ConsoleRuntimeEventSink(RuntimeEventSink):
    """Print runtime events to the console in a human-readable form.

    Purpose:
        Define or implement one event-delivery role used to capture, forward,
        discard, or present runtime events.

    Architectural role:
        Runtime event sink or debug-output helper inside the engine telemetry
        observability boundary.

    Inputs:
        Runtime events or already-formatted debug lines emitted by
        orchestration, decode, or telemetry helpers.

    Outputs:
        Forwarded, buffered, discarded, or displayed runtime events and debug
        lines according to the sink implementation.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.event_sink` within
        the engine telemetry boundary.

    """

    debug_printer: DebugTextEmitter = field(default_factory=DebugTextEmitter)

    @classmethod
    def with_debug_line_emitter(
        cls, debug_line_emitter: DebugSafeLineEmitter
    ) -> ConsoleRuntimeEventSink:
        """Build a sink that emits debug lines through a debug-safe role."""
        return cls(
            debug_printer=DebugTextEmitter(
                debug_line_emitter=debug_line_emitter
            )
        )

    def emit(self, event: Event) -> None:
        """Render one runtime event to the configured console printer.

        Purpose:
            Perform the concrete event-delivery step for this sink or emitter
            implementation while preserving the runtime-side call contract.

        Architectural role:
            Runtime event sink or debug-output helper inside the engine
            telemetry observability boundary.

        Inputs:
            Runtime events or already-formatted debug lines emitted by
            orchestration, decode, or telemetry helpers.

        Outputs:
            Forwarded, buffered, discarded, or displayed runtime events and
            debug lines according to the sink implementation.

        Ownership:
            Owned by `answer_engineering.engine.telemetry.events.event_sink`
            within the engine telemetry boundary.

        """
        if isinstance(event, DebugEvent):
            self.debug_printer.emit(event.msg)


@dataclass(slots=True)
class DebugEventEmitter:
    """Wrap debug text into `DebugEvent` records and send them through a.

    Purpose:
        Own the final emission step for already-prepared debug or decision-log
        text.

    Architectural role:
        Runtime event sink or debug-output helper inside the engine telemetry
        observability boundary.

    Inputs:
        Runtime events or already-formatted debug lines emitted by
        orchestration, decode, or telemetry helpers.

    Outputs:
        Forwarded, buffered, discarded, or displayed runtime events and debug
        lines according to the sink implementation.

    Ownership:
        Owned by `answer_engineering.engine.telemetry.events.event_sink` within
        the engine telemetry boundary.

    """

    event_sink: RuntimeEventSink | None = None

    def emit(self, msg: str) -> None:
        """Wrap one debug text line in a `DebugEvent` and send it through the.

        Purpose:
            Perform the concrete event-delivery step for this sink or emitter
            implementation while preserving the runtime-side call contract.

        Architectural role:
            Runtime event sink or debug-output helper inside the engine
            telemetry observability boundary.

        Inputs:
            Runtime events or already-formatted debug lines emitted by
            orchestration, decode, or telemetry helpers.

        Outputs:
            Forwarded, buffered, discarded, or displayed runtime events and
            debug lines according to the sink implementation.

        Ownership:
            Owned by `answer_engineering.engine.telemetry.events.event_sink`
            within the engine telemetry boundary.

        """
        sink: RuntimeEventSink = (
            ConsoleRuntimeEventSink()
            if self.event_sink is None
            or isinstance(self.event_sink, NullRuntimeEventSink)
            else self.event_sink
        )
        sink.emit(DebugEvent(msg=msg))
