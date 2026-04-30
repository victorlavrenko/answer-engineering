"""Execute a planned set of reproduction subruns against a runtime.

Purpose:
    Coordinate task selection, generation, answer evaluation, optional console
    output, and summary assembly for one reproduction session.

Architectural role:
    Primary application-service module for reproduction execution.

Inputs (architectural provenance):
    Consumes datasets, a generation runtime, and planned subruns supplied by the
    public API or caller code.

Outputs (downstream usage):
    Subrun results and, optionally, a full run summary consumed by notebooks and
    scripts.

Invariants/constraints:
    Session orchestration should call into domain objects for evaluation and
    reporting rather than embedding those calculations inline.

"""

from __future__ import annotations

from dataclasses import dataclass

from ae_paper_reproduction.core.evaluation.reports import (
    RulesetEvaluationResult,
)
from ae_paper_reproduction.core.planning.subruns import (
    Subrun,
    SubrunResult,
)
from ae_paper_reproduction.infra.datasets.datasets import Dataset
from ae_paper_reproduction.runner.session.summary import Summary
from answer_engineering import (
    GenerationPolicy,
    GenerationRequest,
    GenerationRuntime,
)


@dataclass(frozen=True, slots=True)
class ReproductionSession:
    """Execute planned reproduction subruns against one runtime.

    Coordinate the notebook-facing workflow that selects tasks, calls the Answer
    Engineering runtime, evaluates answers, and optionally packages the
    completed results into a summary object. Use this class when a notebook or
    script already has a materialized dataset, a runtime, and an ordered set of
    planned subruns.

    .. note::
        Creating a session does not run model generation. Generation starts only
        when :meth:`~ae_paper_reproduction.api.ReproductionSession.evaluate` or
        :meth:`~ae_paper_reproduction.api.ReproductionSession.run` is called.

    Examples:
        ```python
        session = ReproductionSession(
            dataset=dataset,
            runtime=runtime,
            subruns=tuple(subruns),
        )

        summary = session.run(n_eval=25, verbosity=1)
        print(summary.reports_table())
        ```

    Attributes:
        dataset: Dataset used to select benchmark rows for every subrun.
        runtime: Materialized generation runtime shared by all subruns.
        subruns: Ordered executable subruns to evaluate.

    Methods:
        :meth:`~ae_paper_reproduction.api.ReproductionSession.evaluate`
            Execute each planned subrun and return ordered subrun results.

        :meth:`~ae_paper_reproduction.api.ReproductionSession.run`
            Execute the session and return a notebook-facing summary.

    Runtime behavior:
        For each subrun, the session selects tasks, builds a matching
        :class:`~answer_engineering.GenerationPolicy`, calls
        :meth:`~answer_engineering.GenerationRuntime.generate`, and wraps each
        generated answer in a
        :class:`~ae_paper_reproduction.RulesetEvaluationResult`.

    Architectural role:
        Application-service boundary for reproduction execution. Dataset
        selection, generation, evaluation rows, and summary construction stay
        connected here, while scoring details and report aggregation remain in
        their domain objects.

    Consumes:
        :class:`~ae_paper_reproduction.Dataset`
            Dataset abstraction that provides rows for selected tasks.

        :class:`~answer_engineering.GenerationRuntime`
            Runtime used for baseline and rule-enabled generation.

        :class:`~ae_paper_reproduction.Subrun`
            Runnable subrun plans extracted from notebooks.

    Produces:
        :class:`~ae_paper_reproduction.SubrunResult`
            Completed result bundles returned by
            :meth:`~ae_paper_reproduction.api.ReproductionSession.evaluate`.

        :class:`~ae_paper_reproduction.Summary`
            Run summary returned by
            :meth:`~ae_paper_reproduction.api.ReproductionSession.run`.

    Invariants:
        Subruns are evaluated in the order stored on the session. Each subrun
        receives its own policy so rules, system prompts, and verbosity remain
        isolated.

    Developer Notes:
        Keep this class thin. It should orchestrate the public workflow but not
        duplicate dataset filtering, answer judgement, pairwise reporting,
        telemetry aggregation, or paper-artifact rendering.

    Todo:
        Preserve the small session API while making failures, skipped rows,
        partial runs, and retry policy explicit. Backward compatibility is not
        guaranteed while the reproduction layer is still converging.

    See Also:
        :class:`~ae_paper_reproduction.NotebookSubruns`
        :class:`~ae_paper_reproduction.SubrunResult`
        :class:`~ae_paper_reproduction.Summary`
        :class:`~answer_engineering.GenerationRuntime`

    """

    dataset: Dataset
    runtime: GenerationRuntime
    subruns: tuple[Subrun, ...]

    def evaluate(
        self, *, n_eval: int, verbosity: int
    ) -> tuple[SubrunResult, ...]:
        """Execute planned subruns and return their ordered evaluation results.

        Purpose:
            Select tasks for each subrun, build the matching generation policy,
            run the model, and wrap generated answers in evaluation result
            objects.

        Architectural role:
            Execution loop at the reproduction-session boundary. It connects
            notebook-derived planning objects to the public generation runtime.

        Inputs (architectural provenance):
            `n_eval` and `verbosity` come from notebook or script callers.
            Subrun definitions, compiled rules, system prompts, and dataset rows
            come from the session's constructed state.

        Outputs (downstream usage):
            Returns one `SubrunResult` per planned subrun in the original
            planning order. `run` and notebook code consume these results for
            summaries and artifact generation.

        Invariants/constraints:
            Subrun order is preserved. The method creates a fresh
            `GenerationPolicy` per subrun so each ruleset and system prompt
            remain isolated.

        """
        subresults: list[SubrunResult] = []
        for subrun in self.subruns:
            tasks = subrun.select_tasks(n=n_eval)
            policy = GenerationPolicy(
                rules=subrun.compiled_rules,
                system_prompt=subrun.system_prompt,
                verbosity=verbosity,
            )
            task_results = tuple(
                RulesetEvaluationResult(
                    task.row,
                    answer=self.runtime.generate(
                        GenerationRequest(question=task.question),
                        policy,
                    ),
                )
                for task in tasks
            )
            subresults.append(
                SubrunResult(
                    subrun,
                    task_results,
                    n_eval_requested=n_eval,
                )
            )
        return tuple(subresults)

    def run(self, *, n_eval: int, verbosity: int) -> Summary:
        """Execute the session and build the notebook-facing summary object.

        Purpose:
            Provide the short public workflow used by notebooks: run all planned
            subruns and package their results as a `Summary`.

        Architectural role:
            Convenience entrypoint above `evaluate` at the reproduction-session
            boundary.

        Inputs (architectural provenance):
            Receives evaluation count and verbosity from notebook or script
            callers and delegates execution to `evaluate`.

        Outputs (downstream usage):
            Returns a `Summary` consumed by notebooks for inspection, reporting,
            and optional artifact publication.

        Invariants/constraints:
            This method should remain thin. Evaluation semantics belong in
            `evaluate`; summary construction belongs in the `Summary` boundary.

        """
        return Summary(self.evaluate(n_eval=n_eval, verbosity=verbosity))


__all__ = ["ReproductionSession"]
