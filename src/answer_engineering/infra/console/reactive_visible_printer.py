"""Reactive console printer for streaming visible-text snapshots.

Purpose:
    Provide a deterministic, replay-safe printer that consumes successive full
    visible-text snapshots and emits only the minimal terminal updates required
    to keep the display consistent. The printer maintains a retractable tail
    buffer so recent characters remain editable while earlier content becomes
    committed and stable.

Architectural role:
    This module owns the control-plane logic that bridges source-space visible
    text and terminal-space rendering. It coordinates buffering, layout
    comparison, replay decisions, and emission ordering while delegating line
    segmentation to the VisibleTextLayouter and output delivery to a TextEmitter
    implementation.

    The printer operates on complete snapshots rather than incremental tokens.
    Each observation reconciles the new snapshot against previously committed
    content and determines whether to append, replay, or defer emission.

Boundaries:
    - Upstream: A runtime or generator provides full visible-text snapshots in
      monotonic time order.
    - Internal collaborators:
        * VisibleTextLayouter — computes deterministic printed-line layouts.
        * PrintedLayout — represents the current committed terminal structure.
        * TextEmitter — performs side-effectful output to the terminal or sink.
        * Downstream: The terminal or output stream receiving rendered text.
        * This module does not generate content, interpret semantics, or manage
          terminal cursor positioning beyond newline emission.

Core model:
    The printer maintains three logical regions:

        printed_visible_text Stable prefix already rendered to the terminal.

        unprinted_buffer Retractable tail that may still change due to edits.

        previous_visible_text Last observed snapshot used for invariant
        validation.

    At all times:

        printed_visible_text + unprinted_buffer == previous_visible_text

    The buffer never disappears once populated; it is only partially committed
    from the left. Flushing commits the remaining buffer without clearing the
    structural model.

Emission semantics:

    Prefix growth:
        When the new snapshot extends the previous one, additional characters
        accumulate in the buffer. Once the buffer exceeds the retractable tail
        threshold, a safe prefix is committed and emitted.

    Committed edit:
        When the new snapshot diverges from previously committed text, the
        printer flushes the buffer, computes new layouts, and replays terminal
        output starting from the earliest changed printed line.

    Finalization:
        When the caller signals completion, the remaining buffer is committed
        regardless of size constraints.

Replay semantics:
    Layout comparison determines the minimal replay boundary. If the visible
    structure changes (text difference or newline termination change), the
    printer emits a replay notice and re-renders from the first affected line.
    This guarantees that terminal state always reflects the authoritative
    layout.

Buffering rules:
    - The retractable tail defines how many recent characters remain editable.
    - Emission occurs only at safe boundaries, typically whitespace-aligned.
    - Very small overflow segments may be deferred to avoid fragmentary output.
    - Trailing whitespace is trimmed before commitment to maintain stable wraps.

Invariants and constraints:
    - The merged committed text and buffer must exactly match the last observed
      snapshot.
    - Layout computation is deterministic for identical inputs.
    - Emission order strictly follows snapshot progression.
    - Replays always begin at a printed-line boundary.
    - The printer never mutates source text content.

Failure mode:
    Detection of a core invariant violation triggers an assertion, indicating
    corruption of the printer's internal state or incorrect snapshot ordering.

Non-goals:
    - Token-level streaming or incremental diffing.
    - Terminal cursor manipulation or ANSI control logic.
    - Adaptive terminal width detection.
    - Content formatting or semantic interpretation.

Extensibility:
    The printer can support alternative output sinks, debugging emitters, or
    layout strategies by substituting compatible implementations of the
    TextEmitter or VisibleTextLayouter interfaces without modifying replay
    logic.

"""

from __future__ import annotations

from dataclasses import dataclass, field

from answer_engineering.infra.console.contracts import DebugSafeLineEmitter
from answer_engineering.infra.console.text_emitter import TextEmitter
from answer_engineering.infra.console.visible_layout import (
    PrintedLayout,
    VisibleTextLayouter,
)


@dataclass(slots=True)
class ReactiveVisiblePrinter(DebugSafeLineEmitter):
    """Reactive printer for complete visible-text snapshots.

    Purpose:
        Maintain a stable printed prefix plus retractable tail buffer so
        streaming output can absorb recent edits without unnecessary terminal
        replay.

    Architectural role:
        Console output boundary between runtime-visible text snapshots and
        terminal emission. It owns buffering, flushing, layout comparison, and
        replay decisions; it delegates line layout to `VisibleTextLayouter` and
        byte output to `TextEmitter`.

    Inputs (architectural provenance):
        Receives full visible-text snapshots from generation/runtime code in
        observation order, plus occasional debug-line and termination requests.

    Outputs (downstream usage):
        Emits committed visible text, replay banners, and debug-safe line breaks
        to the configured output sink.

    Invariants/constraints:
        The core state invariant is `printed_visible_text + unprinted_buffer ==
        previous_visible_text`. Edits inside the retractable buffer should
        update the buffer without replay. Edits before the committed prefix
        require flushing and replay from the earliest changed printed line.

    """

    emitter: TextEmitter
    layouter: VisibleTextLayouter
    retractable_tail_chars: int = 80
    min_emit_chars: int = 10
    debug_prefix: str = "[AE]"

    _printed_visible_text: str = field(default="", init=False)
    _previous_visible_text: str = field(default="", init=False)
    _unprinted_buffer: str = field(default="", init=False)
    _printed_layout: PrintedLayout = field(init=False)
    _terminal_ends_with_newline: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        """Clamp emit thresholds and initialize replay state.

        Purpose:
            Normalize printer configuration immediately after dataclass
            construction so later append and reprint decisions can rely on valid
            thresholds.

        Architectural role:
            Constructor completion hook for the reactive console renderer. It
            owns the state invariant connecting printed text, retractable buffer
            content, and the previous visible snapshot.

        Inputs (architectural provenance):
            Reads configured tail-buffer, overflow, and layout settings supplied
            by the console-printing caller.

        Outputs (downstream usage):
            Initializes mutable replay state used by `observe`, `reset`, and
            flush operations.

        Invariants/constraints:
            Thresholds must be positive enough to avoid unstable tiny appends,
            and the initial committed text plus buffer must represent an empty
            visible snapshot.

        """
        self.min_emit_chars = min(
            self.min_emit_chars, self.retractable_tail_chars
        )
        self.reset()

    def reset(self) -> None:
        """Reset printed and buffered state to an empty baseline.

        Purpose:
            Clear the renderer after a completed stream or explicit caller reset
            so the next visible-text sequence starts without stale replay state.

        Architectural role:
            Lifecycle boundary for the console printer. It reestablishes the
            same state invariant created by construction.

        Inputs (architectural provenance):
            Uses only the printer's current mutable state and does not depend on
            a new model-visible snapshot.

        Outputs (downstream usage):
            Leaves the next `observe` call behaving like the first snapshot in a
            fresh stream.

        Invariants/constraints:
            The reset must clear committed layout, retractable buffer content,
            and the previous visible text together; partial reset would corrupt
            append/reprint decisions.

        """
        self._printed_visible_text = ""
        self._previous_visible_text = ""
        self._unprinted_buffer = ""
        self._printed_layout = self.layouter.layout("")
        self._terminal_ends_with_newline = True

    def observe_visible_text(
        self,
        visible_text: str,
        *,
        is_final: bool = False,
    ) -> None:
        """Observe visible text and emit newly committed segments.

        Purpose:
            Accept the latest complete visible-text snapshot and update the
            terminal output with the smallest safe append or replay operation.

        Architectural role:
            Reactive console boundary between generation snapshots and
            human-visible streaming output.

        Inputs (architectural provenance):
            Receives the full visible text currently owned by the runtime output
            stream.

        Outputs (downstream usage):
            Emits text through the configured emitter and updates committed,
            buffered, and previous-snapshot state.

        Invariants/constraints:
            The printer owns append-versus-replay decisions. Edits contained
            within the retractable buffer should not force committed-line
            replay, while edits before the buffer require replay from the
            earliest affected rendered line.

        """
        if visible_text.startswith(self._previous_visible_text):
            self._observe_prefix_growth(visible_text, is_final)
        else:
            self._observe_committed_edit(visible_text)

        if is_final:
            self.flush()

        self._previous_visible_text = visible_text
        self._check_invariant()

    def flush(self) -> None:
        """Emit any currently buffered text to the output sink.

        Purpose:
            Force the retractable tail buffer into committed output at lifecycle
            boundaries.

        Architectural role:
            Console-printer state transition used before debug output, replay,
            and final stream completion.

        Inputs (architectural provenance):
            Reads the printer's internal unprinted buffer.

        Outputs (downstream usage):
            Emits buffered text and moves it into the committed visible-text
            state.

        Invariants/constraints:
            After a flush, the buffer is empty and committed text reflects
            everything the printer has emitted as normal visible output.

        """
        if not self._unprinted_buffer:
            return
        self._commit_buffer_prefix(len(self._unprinted_buffer))

    def emit_debug_line(self, msg: str) -> None:
        """Emit a debug line safely.

        Purpose:
            Write diagnostic output without corrupting the currently buffered
            visible generation stream.

        Architectural role:
            Debug-output boundary shared by runtime diagnostics and the reactive
            visible printer.

        Inputs (architectural provenance):
            Receives a preformatted debug message from orchestration or runtime
            support code.

        Outputs (downstream usage):
            Emits a complete line through the configured emitter.

        Invariants/constraints:
            Pending visible text is flushed first so debug lines never appear
            inside a partially buffered generated answer.

        """
        self.flush()
        if not self._terminal_ends_with_newline:
            self._emit_terminal_text("\n")

        self._emit_terminal_text(f"{self.debug_prefix} {msg}\n")

    def terminate_visible_output_line(self) -> None:
        """Terminate any active visible output line.

        Purpose:
            Ensure subsequent output begins on a fresh line after streaming
            visible generation text.

        Architectural role:
            Console lifecycle helper used when leaving the visible-output
            channel.

        Inputs (architectural provenance):
            Reads the committed and buffered visible-output state.

        Outputs (downstream usage):
            Emits a newline when the active output does not already end at a
            line boundary.

        Invariants/constraints:
            The method flushes first, then inspects the committed suffix. It
            must not add duplicate blank lines when output is already
            terminated.

        """
        self.flush()
        if self._terminal_ends_with_newline:
            return
        self._emit_terminal_text("\n")

    def _observe_prefix_growth(
        self, current: str, is_final: bool = False
    ) -> None:
        added_text = current[len(self._previous_visible_text) :]
        self._unprinted_buffer += added_text

        overflow = len(self._unprinted_buffer) - self.retractable_tail_chars

        if overflow <= 0:
            return

        if overflow < self.min_emit_chars and not is_final:
            return

        overflow = self._unprinted_buffer.rfind(" ", 0, overflow)

        if overflow <= 0:
            return

        self._commit_buffer_prefix(overflow)

    def _observe_committed_edit(self, current: str) -> None:
        old_visible_text = self._printed_visible_text + self._unprinted_buffer
        old_printed_text = self._printed_visible_text
        changed_index = _common_prefix_len(old_visible_text, current)

        if changed_index >= len(self._printed_visible_text):
            self._unprinted_buffer = current[len(self._printed_visible_text) :]
            return

        self.flush()
        old_layout = self._printed_layout

        new_printed_text, new_buffer = self._split_new_visible_text(
            current,
            old_printed_text,
            old_layout,
        )
        new_layout = self.layouter.layout(new_printed_text)
        first_changed_line = _first_changed_line(old_layout, new_layout)

        if first_changed_line is not None:
            remaining_lines = len(old_layout.lines) - first_changed_line

            s = "s" if remaining_lines != 1 else ""

            self._emit_terminal_text(
                f"\n\n=== AE: ignore previous "
                f"{remaining_lines} line{s} "
                "due to an edit ===\n"
            )
            self._emit_layout_from_line(new_layout, first_changed_line)

        self._printed_visible_text = new_printed_text
        self._printed_layout = new_layout
        self._unprinted_buffer = new_buffer

    def _split_new_visible_text(
        self,
        current: str,
        old_printed_text: str,
        old_layout: PrintedLayout,
    ) -> tuple[str, str]:
        if self.retractable_tail_chars == 0:
            return current, ""

        if len(current) <= self.retractable_tail_chars:
            return "", current

        split_index = len(current) - self.retractable_tail_chars

        while (
            split_index < len(current)
            and split_index < len(old_printed_text)
            and current[split_index] == old_printed_text[split_index]
        ):
            split_index += 1

        split_index = old_layout.line_start_for_source_index(split_index)

        while split_index > 0 and current[split_index - 1] == " ":
            split_index -= 1

        return current[:split_index], current[split_index:]

    def _commit_buffer_prefix(self, length: int) -> None:
        while length > 0 and self._unprinted_buffer[length - 1] == " ":
            length -= 1

        if length <= 0:
            return

        committed_text = self._unprinted_buffer[:length]
        self._unprinted_buffer = self._unprinted_buffer[length:]

        old_layout = self._printed_layout
        new_printed_text = self._printed_visible_text + committed_text
        new_layout = self.layouter.layout(new_printed_text)

        self._emit_layout_append(old_layout=old_layout, new_layout=new_layout)

        self._printed_visible_text = new_printed_text
        self._printed_layout = new_layout

    def _emit_layout_append(
        self,
        *,
        old_layout: PrintedLayout,
        new_layout: PrintedLayout,
    ) -> None:
        old_text = _render_layout(old_layout)
        new_text = _render_layout(new_layout)

        if old_text == "":
            self._emit_terminal_text(new_text)
            return

        if new_text.startswith(old_text):
            suffix = new_text[len(old_text) :]
            if suffix:
                self._emit_terminal_text(suffix)
            return

        first_changed_line = _first_changed_line(old_layout, new_layout)
        if first_changed_line is None:
            return

        remaining_lines = len(old_layout.lines) - first_changed_line

        s = "s" if remaining_lines != 1 else ""

        self._emit_terminal_text(
            f"\n\n=== AE: ignore previous "
            f"{remaining_lines} line{s} "
            "due to a rewrap ===\n"
        )
        self._emit_layout_from_line(new_layout, first_changed_line)

    def _emit_layout_from_line(
        self,
        layout: PrintedLayout,
        first_line: int,
    ) -> None:
        if first_line >= len(layout.lines):
            return
        self._emit_terminal_text(_render_layout_from_line(layout, first_line))

    def _emit_terminal_text(self, text: str) -> None:
        if not text:
            return
        self.emitter.emit(text)
        self._terminal_ends_with_newline = text.endswith("\n")

    def _check_invariant(self) -> None:
        merged = self._printed_visible_text + self._unprinted_buffer
        if merged != self._previous_visible_text:
            raise AssertionError(
                "Invariant violated:\n"
                f"previous={len(self._previous_visible_text)}\n"
                f"merged={len(merged)}"
            )


def _first_changed_line(
    old_layout: PrintedLayout, new_layout: PrintedLayout
) -> int | None:
    shared = min(len(old_layout.lines), len(new_layout.lines))

    for index in range(shared):
        old_line = old_layout.lines[index]
        new_line = new_layout.lines[index]
        if (
            old_line.text != new_line.text
            or old_line.ends_with_newline != new_line.ends_with_newline
        ):
            return index

    if len(old_layout.lines) != len(new_layout.lines):
        return shared

    return None


def _render_layout(layout: PrintedLayout) -> str:
    return _render_layout_from_line(layout, 0)


def _render_layout_from_line(layout: PrintedLayout, first_line: int) -> str:
    if first_line >= len(layout.lines):
        return ""

    parts: list[str] = []
    last_index = len(layout.lines) - 1
    for index in range(first_line, len(layout.lines)):
        line = layout.lines[index]
        parts.append(line.text)
        if line.ends_with_newline or index < last_index:
            parts.append("\n")

    return "".join(parts)


def _common_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


__all__ = ["ReactiveVisiblePrinter"]
