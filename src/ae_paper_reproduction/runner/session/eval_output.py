"""Format human-readable console output for reproduction tasks.

Purpose:
    Render task start and task end information, including correctness and answer
    text, at a caller-selected verbosity level during session execution.

Architectural role:
    UI helper module for the session runner.

Inputs (architectural provenance):
    Consumes tasks and evaluation results emitted by the session runner.

Outputs (downstream usage):
    Console-formatted strings printed by the session orchestration layer.

Invariants/constraints:
    Formatting choices should not affect evaluation semantics or run summaries.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from textwrap import TextWrapper
from typing import Final

from ae_paper_reproduction.core.evaluation.reports import (
    RulesetEvaluationResult,
)
from ae_paper_reproduction.core.planning.subruns import SubrunTask

DEFAULT_EVAL_OUTPUT_WIDTH: Final[int] = 80


def _format_task_label(task: SubrunTask, *, ruleset_name: str) -> str:
    return f"TASK {task.id}, TYPE {task.case_type}, RULESET {ruleset_name}"


@dataclass(frozen=True, slots=True)
class EvaluationPrinter:
    """Console printer for per-task evaluation progress.

    Render task boundaries, generated answers, expected answers, correctness,
    and optional telemetry/debug context during reproduction runs. It is
    intended for human notebook inspection, not for machine-readable reporting.

    .. note::
        Set low verbosity for ordinary runs and higher verbosity when diagnosing
        a specific case or rule behavior.

    Examples:
        ```python
        printer = EvaluationPrinter(verbosity=VERBOSITY)
        printer.task_start(task, ruleset_name=subrun.ruleset_name)
        printer.task_end(task, result=evaluation, answer=answer)
        ```

    Attributes:
        verbosity: Console detail level selected by the notebook or session.
        enabled: Whether this printer should emit output at the current
            verbosity.

    Methods:
        :meth:`~ae_paper_reproduction.EvaluationPrinter.enabled`
            Return whether this printer should emit console output.

        :meth:`~ae_paper_reproduction.EvaluationPrinter.task_start`
            Print the beginning of one task.

        :meth:`~ae_paper_reproduction.EvaluationPrinter.task_end`
            Print generated output and evaluation status for one task.

    Runtime behavior:
        Printing is side-effect-only. It does not affect generation, evaluation,
        metrics, or persisted artifacts.

    Architectural role:
        User-interface boundary for reproduction sessions.

    Consumes:
        Subrun tasks, generation outputs, evaluation rows, and verbosity
        settings.

    Produces:
        Human-readable terminal or notebook output.

    Invariants:
        Output formatting should not become a source of truth for reports or
        paper metrics.

    Developer Notes:
        Keep this printer separate from structured telemetry and artifact
        generation. If console rendering becomes richer, preserve a clean
        boundary between human output and data aggregation.

    Todo:
        Improve formatting for long answers and telemetry snippets without
        coupling the printer to runtime internals.

    See Also:
        :class:`~ae_paper_reproduction.Progress`
        :class:`~ae_paper_reproduction.RulesetEvaluationResult`

    """

    verbosity: int = 0
    width: int = DEFAULT_EVAL_OUTPUT_WIDTH
    break_long_words: bool = True
    break_on_hyphens: bool = True
    _wrap: Callable[[str], str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build and cache the text-wrapping callable from printer settings.

        Purpose:
            Finish printer construction by deriving the callable used for
            formatted evaluation output.

        Architectural role:
            Presentation-layer constructor hook for notebook and CLI evaluation
            output.

        Inputs (architectural provenance):
            Reads width and formatting options supplied to the printer
            dataclass.

        Outputs (downstream usage):
            Stores a reusable wrapper consumed by subsequent print/render
            methods.

        Invariants/constraints:
            Wrapping policy should be fixed after construction so a single
            printer instance produces stable output.

        """
        wrapper = TextWrapper(
            width=self.width,
            break_long_words=self.break_long_words,
            break_on_hyphens=self.break_on_hyphens,
        )
        object.__setattr__(self, "_wrap", wrapper.fill)

    @property
    def enabled(self) -> bool:
        """Return whether console printing is enabled for the configured.

        Purpose:
            Expose whether console printing is enabled for the configured
            verbosity.

        Architectural role:
            Presentation method on the evaluation printer.

        Inputs (architectural provenance):
            Consumes stored verbosity/wrapping settings and, for task methods,
            session task/result inputs.

        Outputs (downstream usage):
            A boolean gate or side-effecting console output used by the runner.

        Invariants/constraints:
            Formatting behavior should remain a pure presentation concern.

        """
        return self.verbosity > 0

    def task_start(self, task: SubrunTask, *, ruleset_name: str) -> None:
        """Print the start-of-task line for one evaluation task.

        Purpose:
            Emit the start-of-task line for one evaluation task.

        Architectural role:
            Presentation method on the evaluation printer.

        Inputs (architectural provenance):
            Consumes stored verbosity/wrapping settings and, for task methods,
            session task/result inputs.

        Outputs (downstream usage):
            A boolean gate or side-effecting console output used by the runner.

        Invariants/constraints:
            Formatting behavior should remain a pure presentation concern.

        """
        if not self.enabled:
            return
        label = _format_task_label(task, ruleset_name=ruleset_name)
        self._print_major_separator()
        self._print_section("QUESTION", label, body=task.question)
        self._print_minor_separator()
        self._print_section("STREAMING", label, body="")

    def task_end(
        self,
        task: SubrunTask,
        *,
        ruleset_name: str,
        task_result: RulesetEvaluationResult,
    ) -> None:
        """Print the end-of-task block for one evaluated task result.

        Purpose:
            Emit the end-of-task block for one evaluated task result.

        Architectural role:
            Presentation method on the evaluation printer.

        Inputs (architectural provenance):
            Consumes stored verbosity/wrapping settings and, for task methods,
            session task/result inputs.

        Outputs (downstream usage):
            A boolean gate or side-effecting console output used by the runner.

        Invariants/constraints:
            Formatting behavior should remain a pure presentation concern.

        """
        if not self.enabled:
            return
        label = _format_task_label(task, ruleset_name=ruleset_name)
        status = "PASS" if task_result.ok else "FAIL"
        self._print_minor_separator()
        self._print_section("ANSWER", label, body=task_result.answer)
        self._print_minor_separator()
        print(f"RESULT: {status}, {label}")

    def _print_section(
        self, title: str, label: str, *, body: str | None = None
    ) -> None:
        print(f"{title}: {label}")
        if body is not None:
            print()
            print(self._wrap(body))

    def _print_major_separator(self) -> None:
        print("\n" + "=" * self.width)

    def _print_minor_separator(self) -> None:
        print("-" * self.width)


__all__ = ["EvaluationPrinter"]
