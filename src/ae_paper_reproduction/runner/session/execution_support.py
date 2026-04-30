"""Provide progress-display helpers for reproduction sessions.

Purpose:
    Wrap iterable task streams with optional progress reporting and maintain
    lightweight accuracy text that can be shown while a run is executing.

Architectural role:
    Execution-support module used by session orchestration.

Inputs (architectural provenance):
    Consumes iterables of tasks and live accuracy metrics produced during
    execution.

Outputs (downstream usage):
    Progress iterators and postfix text consumed by the session runner.

Invariants/constraints:
    These helpers should remain presentation-oriented and must not own
    evaluation or summary logic.

"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ProgressMetrics:
    """Store the lightweight progress text shown during a running session.

    Purpose:
        Represent the derived accuracy string displayed in the progress bar
        postfix without coupling the rest of the runner to the progress-library
        API.

    Architectural role:
        Presentation value object for session progress display.

    Inputs (architectural provenance):
        Constructed from live accuracy values produced during execution.

    Outputs (downstream usage):
        Progress-bar postfix mappings consumed by the progress wrapper.

    Invariants/constraints:
        This type should remain a small formatting helper rather than an
        evaluation object.

    """

    accuracy_text: str

    def to_postfix(self) -> dict[str, str]:
        """Convert progress metrics into a postfix mapping for the progress.

        Purpose:
            Translate the stored progress text into the shape expected by the
            underlying progress-bar implementation.

        Architectural role:
            Adapter method between session metrics and the progress-library API.

        Inputs (architectural provenance):
            Reads the stored metrics on this object.

        Outputs (downstream usage):
            A postfix mapping consumed by the wrapped progress iterator.

        Invariants/constraints:
            The mapping should remain stable and presentation-focused.

        """
        return {"acc": self.accuracy_text}


class Progress[T]:
    """Progress wrapper for notebook reproduction loops.

    Wrap a tuple of items with a progress display and expose helper methods for
    updating visible metrics such as accuracy. This class keeps runner loops
    readable without embedding progress-bar details in evaluation logic.

    .. note::
        Progress rendering is presentation-only. Reports and summaries should
        use structured results rather than values displayed in the progress bar.

    Examples:
        ```python
        progress = Progress(tasks, desc=subrun.name)
        for task in progress:
            answer = runtime.generate(
                GenerationRequest(question=task.question),
                policy=policy,
                rules=task.compiled_rules,
            )
            progress.accuracy(current_accuracy)
        ```

    Args:
        items: Tuple of items to iterate over.
        desc: Progress-bar description shown to the user.

    Yields:
        T:
            Next item from the wrapped task sequence. The progress wrapper does
            not modify items or evaluation results; it only updates presentation
            state during iteration.

    Methods:
        :meth:`~ae_paper_reproduction.Progress.set_metrics`
            Update displayed runtime metrics.

        :meth:`~ae_paper_reproduction.Progress.accuracy`
            Compute evaluation accuracy statistics.

    Runtime behavior:
        Iteration delegates to the wrapped items and updates presentation state.
        It does not change the items or evaluation results.

    Architectural role:
        Notebook execution-support boundary for progress presentation.

    Consumes:
        A fixed tuple of items and optional display metrics.

    Produces:
        Iterated items plus human-readable progress output.

    Invariants:
        The wrapped item sequence should remain stable for the lifetime of the
        progress object.

    Developer Notes:
        Keep progress display independent from report generation. If richer
        metrics are added, flow them through typed metric objects rather than ad
        hoc strings.

    Todo:
        Improve metric formatting and non-notebook fallback behavior without
        changing runner-loop semantics.

    See Also:
        :class:`~ae_paper_reproduction.EvaluationPrinter`
        :class:`~ae_paper_reproduction.SubrunTask`

    """

    def __init__(self, items: tuple[T, ...], *, desc: str) -> None:
        """Create a progress display around a fixed set of reproduction items.

        Use this in notebook execution loops when iterating over selected tasks.
        The wrapper delegates visual progress rendering to ``tqdm`` while
        preserving the original task objects and their order. It also exposes
        convenience reporting methods used by the reproduction notebook to show
        incremental accuracy.

        Example:
            ```python
            tasks = subrun.select_tasks(n=25)
            for task in Progress(tasks, desc=subrun.name):
                answer = runtime.generate(...)
                ...
            ```

        Args:
            items: Tuple of items to iterate over, usually ``SubrunTask``
                objects returned by ``Subrun.select_tasks``. The tuple is not
                modified; the wrapper only presents iteration progress.
            desc: Human-readable progress label shown next to the progress bar.
                In the reproduction notebook this is normally the subrun name,
                making it clear whether the current loop is baseline, scoped, or
                rule-enabled.

        Notes:
            The progress wrapper is presentation infrastructure. It should not
            decide which tasks are selected, whether answers are correct, or how
            telemetry is aggregated. Those responsibilities belong to the subrun
            planning, evaluation, and summary objects.

        Developer notes:
            Keep this constructor small and predictable. Notebook users should
            be able to replace it with direct iteration or another progress UI
            without changing evaluation semantics.

        """
        self._progress = tqdm(
            items,
            desc=desc,
            unit="case",
            dynamic_ncols=True,
        )

    def __iter__(self) -> Iterator[T]:
        """Iterate over tasks while preserving any configured progress display.

        Purpose:
            Provide iteration over tasks while preserving any configured
            progress display.

        Architectural role:
            Method on the session progress wrapper.

        Inputs (architectural provenance):
            Consumes the wrapped task iterable or live metric values from the
            runner.

        Outputs (downstream usage):
            Progress-wrapper state changes or task iteration consumed by the
            execution loop.

        Invariants/constraints:
            Presentation updates must not change the underlying task stream.

        """
        return iter(self._progress)

    def set_metrics(self, metrics: ProgressMetrics) -> None:
        """Update the stored progress metrics shown by the wrapper.

        Purpose:
            Set the stored progress metrics shown by the wrapper.

        Architectural role:
            Method on the session progress wrapper.

        Inputs (architectural provenance):
            Consumes the wrapped task iterable or live metric values from the
            runner.

        Outputs (downstream usage):
            Progress-wrapper state changes or task iteration consumed by the
            execution loop.

        Invariants/constraints:
            Presentation updates must not change the underlying task stream.

        """
        self._progress.set_postfix(metrics.to_postfix())

    def accuracy(self, value: float) -> None:
        """Update the progress display from one numeric accuracy value.

        Purpose:
            Set the progress display from one numeric accuracy value.

        Architectural role:
            Method on the session progress wrapper.

        Inputs (architectural provenance):
            Consumes the wrapped task iterable or live metric values from the
            runner.

        Outputs (downstream usage):
            Progress-wrapper state changes or task iteration consumed by the
            execution loop.

        Invariants/constraints:
            Presentation updates must not change the underlying task stream.

        """
        self.set_metrics(ProgressMetrics(accuracy_text=f"{value:.3f}"))
