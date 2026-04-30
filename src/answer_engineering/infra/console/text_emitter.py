"""Text emission boundary for visible output streams.

Purpose:
    Define the minimal abstraction responsible for emitting already-rendered
    visible text to an external sink. This module establishes a stable boundary
    between the console printer (layout / buffering / replay logic) and the
    concrete output mechanism (e.g., stdout).

Architectural role:
    This module owns the terminal-side emission interface. It does not perform
    formatting, wrapping, buffering, replay, or mutation of visible text. All
    layout and rendering decisions are made upstream by the printer subsystem.
    The emitter is strictly responsible for delivering text bytes to the target
    output stream exactly as instructed.

Boundaries:
    - Upstream: Reactive / layout-aware printer components provide finalized
      visible text segments ready for emission.
    - Downstream: External output sinks (stdout, file descriptors, test doubles,
      or other stream implementations).
    - This module does not depend on printer internals and remains agnostic to
      wrapping width, buffer size, replay triggers, or generation semantics.

Invariants and constraints:
    - Emission is append-only and side-effect-limited to the target stream.
    - The emitter must not transform, split, coalesce, or rewrap text.
    - The emitter must preserve character ordering and content exactly.
    - Flush behavior is implementation-defined but must ensure timely visibility
      of emitted text when required by interactive streaming scenarios.

Extensibility:
    Additional emitters may implement the TextEmitter protocol to support
    alternative sinks (e.g., buffered collectors, logging adapters, network
    streams) without modifying printer or layout logic. This preserves the
    single-responsibility boundary between rendering and delivery.

"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import Protocol


class TextEmitter(Protocol):
    """Boundary for writing already-rendered visible text.

    Purpose:
        Isolate terminal or stream side effects from console layout and printer
        state machines.

    Architectural role:
        Output-port protocol used by reactive console printers. The printer
        decides what text should be emitted; the emitter only performs the
        write.

    Inputs (architectural provenance):
        Receives text fragments that have already been selected and formatted by
        the console printer.

    Outputs (downstream usage):
        Produces visible side effects on the configured output stream.

    Invariants/constraints:
        Implementations must not add separators, wrapping, buffering semantics,
        or extra newlines. Text should be emitted exactly as provided.

    """

    def emit(self, text: str) -> None:
        """Emit visible text exactly as provided."""
        raise NotImplementedError


@dataclass(slots=True)
class StdoutTextEmitter:
    """Text emitter backed by standard output.

    Purpose:
        Provide the default side-effect adapter for console rendering in
        notebooks, scripts, and manual runs.

    Architectural role:
        Concrete output-port implementation for the console infrastructure.

    Inputs (architectural provenance):
        Receives printer-selected text fragments through the `TextEmitter`
        protocol.

    Outputs (downstream usage):
        Writes the fragments to `stdout` so runtime progress and debug text
        become visible to the caller.

    Invariants/constraints:
        The emitter does not own layout decisions and should not transform the
        text beyond the direct standard-output write.

    """

    def emit(self, text: str) -> None:
        """Write text to stdout without adding extra separators."""
        builtins.print(text, end="", flush=True)


__all__ = ["TextEmitter", "StdoutTextEmitter"]
