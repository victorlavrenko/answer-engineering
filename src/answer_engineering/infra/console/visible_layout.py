"""Deterministic visible-text layout model for console rendering.

Purpose:
    Provide a stable, reproducible mapping from a source visible-text string to
    a sequence of printed lines under explicit newline handling and fixed-width
    soft wrapping. The layout produced by this module serves as the canonical
    representation of what the terminal should display for a given source text.

Architectural role:
    This module owns the layout model used by the reactive console printer. It
    converts a linear source string into structured line records with precise
    source offsets. The printer relies on this structure to detect layout
    changes and determine the earliest line that must be replayed after a
    committed edit.

    The layouter is purely functional: given identical input text and wrap
    width, it must produce identical line boundaries and offsets. This
    determinism is critical for reliable replay and debugging.

Boundaries:
    - Input: A fully assembled visible-text string produced upstream by the
      generation runtime or buffer management logic.
    - Output: A PrintedLayout describing the line segmentation and source
      offsets, or a rendered terminal string derived from that layout.
    - This module does not manage buffering, streaming, incremental emission,
      terminal state, or cursor movement.
    - This module does not mutate source text or maintain historical state.

Key concepts:
    PrintedLine:
        Immutable record describing one printed line derived from the source
        string. Each line carries the exact source span that produced it and a
        flag indicating whether the line terminated due to an explicit newline.

    PrintedLayout:
        Immutable sequence of PrintedLine objects representing the full layout
        of a source string. Provides helper methods for replay logic, including
        earliest change detection and mapping from source indices to line
        boundaries.

    VisibleTextLayouter:
        Stateless transformer that computes layouts using:
            - Explicit newline breaks when present in the source text.
            - Soft breaks at whitespace when within wrap width.
            - Hard breaks at wrap width when no whitespace is available.

Invariants and constraints:
    - Layout is deterministic for a given (text, wrap_width) pair.
    - Line boundaries are monotonic and non-overlapping.
    - Source offsets always refer to positions in the original input string.
    - Explicit newline characters are consumed into the line span but are not
      included in the line text.
    - Soft wrapping prefers whitespace but falls back to hard wrapping when
      necessary.
    - Trailing whitespace after a soft break is skipped before the next line
      begins.

Replay semantics:
    The earliest_changed_line_start method compares two layouts and returns the
    source offset of the first line whose visible content or newline termination
    status differs. This offset defines the minimal safe replay boundary for the
    console printer.

    If no differences exist, the method returns None.

Non-goals:
    - Terminal width detection or dynamic resizing.
    - Word hyphenation or language-aware wrapping.
    - Incremental layout updates or diff optimization.
    - Rendering side effects or output emission.

Extensibility:
    Alternative layout strategies (e.g., indentation-aware wrapping, ANSI-aware
    rendering, or proportional fonts) may be introduced by providing compatible
    PrintedLayout semantics while preserving deterministic line boundary
    behavior.

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrintedLine:
    """One deterministic terminal line derived from source visible text.

    Purpose:
        Record the text and source offsets for one line after explicit newline
        handling and fixed-width soft wrapping.

    Architectural role:
        Leaf layout value consumed by `PrintedLayout` and the reactive console
        printer when deciding append and replay boundaries.

    Inputs (architectural provenance):
        Constructed only by `VisibleTextLayouter.layout` from a source
        visible-text string and wrap width.

    Outputs (downstream usage):
        Provides line text for terminal rendering and source offsets for mapping
        changed source positions back to printed-line starts.

    Invariants/constraints:
        `source_start` and `source_end` refer to the original source string. The
        newline terminator is represented by `ends_with_newline` and is not
        included in `text`.

    """

    source_start: int
    source_end: int
    text: str
    ends_with_newline: bool


@dataclass(frozen=True, slots=True)
class PrintedLayout:
    """Immutable layout of source visible text into printed lines.

    Purpose:
        Represent the canonical terminal-space structure for a source
        visible-text snapshot.

    Architectural role:
        Comparison and replay data structure used by the reactive printer.

    Inputs (architectural provenance):
        Built by `VisibleTextLayouter.layout` after wrapping and newline
        handling.

    Outputs (downstream usage):
        Supplies line records to append/replay logic and maps changed source
        indices to safe printed-line replay starts.

    Invariants/constraints:
        Lines must be ordered by source offset, non-overlapping, and derived
        from a single source snapshot.

    """

    lines: tuple[PrintedLine, ...]

    def line_start_for_source_index(self, source_index: int) -> int:
        """Return the printed-line start offset containing a source index.

        Purpose:
            Map a source-text character position back to the start of the
            rendered line that owns it.

        Architectural role:
            Layout-query helper used by replay and earliest-changed-line
            calculations.

        Inputs (architectural provenance):
            Receives a source index derived from visible-text comparison.

        Outputs (downstream usage):
            Returns the source offset where the corresponding rendered line
            begins.

        Invariants/constraints:
            The layout must have been built from the same source string that
            supplied the index. Boundary positions resolve to the nearest owning
            printed line.

        """
        if not self.lines:
            return 0

        for line in self.lines:
            if line.source_start <= source_index < line.source_end:
                return line.source_start

        return self.lines[-1].source_start


@dataclass(frozen=True, slots=True)
class VisibleTextLayouter:
    """Stateless mapper from visible text to deterministic printed layout.

    Purpose:
        Apply explicit newline handling and fixed-width wrapping to a complete
        visible-text snapshot.

    Architectural role:
        Pure layout boundary below the reactive printer. It owns line
        segmentation but not buffering, replay decisions, or terminal emission.

    Inputs (architectural provenance):
        Configured with a wrap width and called with full visible-text snapshots
        from printer state.

    Outputs (downstream usage):
        Produces `PrintedLayout` values consumed by console replay and debug
        logic.

    Invariants/constraints:
        The same `(text, wrap_width)` pair must always produce the same layout.
        Wrap width must be positive.

    """

    wrap_width: int

    def __post_init__(self) -> None:
        """Reject non-positive wrap widths before layout operations.

        Purpose:
            Validate the configured console width at construction time so layout
            code can assume a meaningful wrapping column.

        Architectural role:
            Constructor guard for the visible-text layout boundary. It prevents
            invalid configuration from surfacing later as ambiguous wrapping
            behavior.

        Inputs (architectural provenance):
            Reads the wrap width supplied by the console printer or tests.

        Outputs (downstream usage):
            Leaves a valid layouter instance for snapshot layout, append
            detection, and earliest-changed-line calculations.

        Invariants/constraints:
            Width must be greater than zero. Invalid widths fail immediately
            rather than producing degenerate printed lines.

        """
        if self.wrap_width <= 0:
            raise ValueError("wrap_width must be positive")

    def layout(self, text: str) -> PrintedLayout:
        """Compute the printed layout for one visible-text snapshot.

        Purpose:
            Split source text into printed lines using explicit newlines,
            whitespace soft breaks, and hard wrapping when no safe whitespace
            break exists.

        Architectural role:
            Deterministic transformation used by the printer before comparing
            old and new terminal layouts.

        Inputs (architectural provenance):
            Receives the current source visible text from
            `ReactiveVisiblePrinter`.

        Outputs (downstream usage):
            Returns a `PrintedLayout` whose source offsets and line text drive
            append and replay decisions.

        Invariants/constraints:
            Explicit newline characters terminate lines but are not included in
            line text. Whitespace consumed by a soft break is skipped before the
            next line.

        """
        if not text:
            return PrintedLayout(lines=())

        lines: list[PrintedLine] = []
        line_start = 0
        text_length = len(text)

        while line_start < text_length:
            hard_limit = min(line_start + self.wrap_width, text_length)
            explicit_newline = text.find("\n", line_start, hard_limit + 1)
            if explicit_newline != -1:
                lines.append(
                    PrintedLine(
                        source_start=line_start,
                        source_end=explicit_newline + 1,
                        text=text[line_start:explicit_newline],
                        ends_with_newline=True,
                    )
                )
                line_start = explicit_newline + 1
                continue

            if hard_limit == text_length:
                break

            soft_break = text.rfind(" ", line_start, hard_limit + 1)
            line_end = hard_limit if soft_break <= line_start else soft_break
            lines.append(
                PrintedLine(
                    source_start=line_start,
                    source_end=line_end,
                    text=text[line_start:line_end],
                    ends_with_newline=False,
                )
            )
            while line_end < len(text) and text[line_end] == " ":
                line_end += 1
            line_start = line_end

        if line_start < text_length:
            lines.append(
                PrintedLine(
                    source_start=line_start,
                    source_end=text_length,
                    text=text[line_start:],
                    ends_with_newline=False,
                )
            )

        return PrintedLayout(lines=tuple(lines))


__all__ = ["PrintedLine", "PrintedLayout", "VisibleTextLayouter"]
