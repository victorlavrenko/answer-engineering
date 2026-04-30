"""Role protocols shared across console rendering collaborators."""

from __future__ import annotations

from typing import Protocol


class DebugSafeLineEmitter(Protocol):
    """Protocol for debug emission that preserves visible output integrity.

    Purpose:
        Define the minimal operation needed by collaborators that emit debug
        lines while assistant-visible text may already be streaming.

    Architectural role:
        Console rendering contract between runtime/debug instrumentation and the
        concrete stream printer.

    Inputs (architectural provenance):
        Implemented by console printers or adapters that understand their own
        buffering and line-replay state.

    Outputs (downstream usage):
        Consumed by runtime components that need debug output without directly
        managing terminal layout.

    Invariants/constraints:
        Implementations must avoid corrupting already emitted assistant text,
        including buffered partial lines and replayable visible output.

    """

    def emit_debug_line(self, msg: str) -> None:
        """Emit one debug line through a layout-safe console boundary.

        Purpose:
            Allow instrumentation code to publish diagnostics without knowing
            how the active printer buffers or replays visible assistant text.

        Architectural role:
            Protocol member implemented by debug-safe console emitters.

        Inputs (architectural provenance):
            Receives a fully formatted diagnostic message from runtime, probing,
            proposal, or printer-debug code.

        Outputs (downstream usage):
            Writes the diagnostic line to the active output stream for human
            debugging and test replay.

        Invariants/constraints:
            The emitted debug line must not merge with, truncate, or otherwise
            corrupt visible assistant output already owned by the stream
            printer.

        """
        raise NotImplementedError


__all__ = ["DebugSafeLineEmitter"]
