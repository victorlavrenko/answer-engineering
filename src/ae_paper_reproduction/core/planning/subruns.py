"""Turn notebook-extracted rulesets into executable reproduction subrun plans.

Purpose:
    Bind notebook ruleset specs to dataset rows and runtime configuration,
    compile authored rules lazily, and expand planned subruns into executable
    evaluation tasks.

Architectural role:
    Planning boundary between notebook-derived experiment design and execution
    sessions.

Architectural direction:
    Keep planning responsibilities explicit while making them easier to explain
    independently from execution and reporting concerns.

Why this matters:
    The current planning model is correctly tied to notebook-derived rulesets
    and evaluation workflows, but remains experiment-shaped.

What better would look like:
    Planning abstractions stay clear and stable even as execution/reporting
    layers evolve around them.

How improvement can be recognized:
    - Clearer ownership boundaries between planning, execution, and reporting
    - Lower cross-module edits for new experiment planning shapes
    - Simpler explanation of subrun identity, scope, and task expansion

Open constraint:
    Planning abstractions should remain responsive to future experiment shapes.

"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

from ae_paper_reproduction.core.aggregation.rule_stats import TelemetryItem
from ae_paper_reproduction.core.evaluation.reports import (
    AccuracyReport,
    RulesetEvaluationResult,
)
from ae_paper_reproduction.core.evaluation.result_types import DatasetRow
from ae_paper_reproduction.core.evaluation.run_session import SubrunSession
from ae_paper_reproduction.core.planning.notebook_extractor import (
    GenerationMode,
    NotebookRulesetSpec,
    PaperRole,
    extract_answer_engineering_subruns_from_ipynb,
)
from ae_paper_reproduction.infra.datasets.datasets import Dataset
from answer_engineering import (
    CompiledRules,
    GenerationPolicy,
    GenerationRuntime,
)


@dataclass(frozen=True, slots=True)
class SubrunDefinition:
    """Immutable definition of one notebook-authored reproduction subrun.

    Keep the parsed notebook ruleset, optional case-type scope, notebook path,
    and extraction index together as the canonical plan for one runnable subrun.
    This object is the stable source for names, prompts, rules markdown, and
    provenance used later by execution and reporting.

    .. note::
        This is a planning record, not an execution object. It does not select
        dataset rows, compile rules eagerly, or run generation.

    Examples:
        ```python
        definition = SubrunDefinition(
            ruleset=ruleset_spec,
            case_type="ssnhl",
            index=1,
            notebook_path="notebooks/reproduce.ipynb",
        )
        print(definition.name)
        ```

    Attributes:
        ruleset: Parsed notebook ruleset specification.
        case_type: Optional case-type filter for this subrun.
        index: Zero-based extraction order within the notebook plan.
        notebook_path: Notebook path that produced the ruleset.
        ruleset_name: Notebook-authored ruleset name.
        scope_label: Case-type label, or ``"all"`` for unscoped subruns.
        rules_markdown: Authored rules markdown compiled during execution.
        system_prompt: Notebook-authored prompt or generation-policy default.
        name: Composite ruleset/scope name used in reports and progress.

    Runtime behavior:
        All public naming and prompt fields are derived from the stored notebook
        specification and scope. This prevents display ids, reporting ids, and
        rule provenance from drifting across planning and execution.

    Architectural role:
        Value-object boundary between notebook extraction and executable
        :class:`~ae_paper_reproduction.Subrun` construction.

    Consumes:
        :class:`~ae_paper_reproduction.NotebookRulesetSpec`
            Parsed ruleset cell from a notebook.

    Produces:
        Derived scalar metadata consumed by
        :class:`~ae_paper_reproduction.Subrun`, task selection, summaries, and
        artifact export.

    Invariants:
        Derived names and prompts must come from this object rather than being
        recomputed from scattered notebook fields.

    Developer Notes:
        Keep this object immutable and side-effect free. Lazy compilation,
        dataset access, and generation belong to ``Subrun`` and the runtime
        layer, not to the definition record.

    Todo:
        If notebook plans grow richer scoping or sampling options, add those
        fields here explicitly so downstream code keeps one source of truth.

    See Also:
        :class:`~ae_paper_reproduction.NotebookRulesetSpec`
        :class:`~ae_paper_reproduction.Subrun`
        :class:`~ae_paper_reproduction.NotebookSubruns`

    """

    ruleset: NotebookRulesetSpec
    case_type: str | None
    index: int
    notebook_path: str
    mode: GenerationMode
    paper_role: PaperRole | None = None
    paper_variant: str | None = None

    @property
    def ruleset_name(self) -> str:
        """Return the notebook-authored ruleset name.

        Purpose:
            Expose one derived planning field from the canonical subrun
            definition.

        Architectural role:
            Cheap accessor on the notebook-planning value object used by public
            reproduction notebooks and reporting code.

        Inputs (architectural provenance):
            Reads fields captured from notebook extraction and scope assignment
            during `NotebookSubruns` construction.

        Outputs (downstream usage):
            Returns the name consumed by report labels, subrun ids, and progress
            output.

        Invariants/constraints:
            The value must remain derived from the stored definition so
            planning, execution, and reporting agree on one source of truth.

        """
        return self.ruleset.ruleset_name

    @property
    def scope_label(self) -> str:
        """Return the comparable scope label for this subrun.

        Purpose:
            Expose one derived planning field from the canonical subrun
            definition.

        Architectural role:
            Cheap accessor on the notebook-planning value object used by public
            reproduction notebooks and reporting code.

        Inputs (architectural provenance):
            Reads fields captured from notebook extraction and scope assignment
            during `NotebookSubruns` construction.

        Outputs (downstream usage):
            Returns the case-type scope or `all` when the subrun is not narrowed
            to one case type.

        Invariants/constraints:
            The value must remain derived from the stored definition so
            planning, execution, and reporting agree on one source of truth.

        """
        return self.case_type or "all"

    @property
    def rules_markdown(self) -> str:
        """Return the authored rules markdown for this subrun.

        Purpose:
            Expose one derived planning field from the canonical subrun
            definition.

        Architectural role:
            Cheap accessor on the notebook-planning value object used by public
            reproduction notebooks and reporting code.

        Inputs (architectural provenance):
            Reads fields captured from notebook extraction and scope assignment
            during `NotebookSubruns` construction.

        Outputs (downstream usage):
            Returns markdown later compiled into `CompiledRules` and exported in
            reproduction artifacts.

        Invariants/constraints:
            The value must remain derived from the stored definition so
            planning, execution, and reporting agree on one source of truth.

        """
        return self.ruleset.rules_markdown

    @property
    def system_prompt(self) -> str:
        """Return the system prompt selected for this subrun.

        Purpose:
            Expose one derived planning field from the canonical subrun
            definition.

        Architectural role:
            Cheap accessor on the notebook-planning value object used by public
            reproduction notebooks and reporting code.

        Inputs (architectural provenance):
            Reads fields captured from notebook extraction and scope assignment
            during `NotebookSubruns` construction.

        Outputs (downstream usage):
            Returns the notebook-authored prompt or the public generation-
            policy default when omitted.

        Invariants/constraints:
            The value must remain derived from the stored definition so
            planning, execution, and reporting agree on one source of truth.

        """
        return (
            GenerationPolicy.default_system_prompt
            if self.ruleset.system_prompt is None
            else self.ruleset.system_prompt
        )

    @property
    def name(self) -> str:
        """Return the composite display and identity name.

        Purpose:
            Expose one derived planning field from the canonical subrun
            definition.

        Architectural role:
            Cheap accessor on the notebook-planning value object used by public
            reproduction notebooks and reporting code.

        Inputs (architectural provenance):
            Reads fields captured from notebook extraction and scope assignment
            during `NotebookSubruns` construction.

        Outputs (downstream usage):
            Returns the ruleset/scope composite consumed by progress output,
            reports, and stable ids.

        Invariants/constraints:
            The value must remain derived from the stored definition so
            planning, execution, and reporting agree on one source of truth.

        """
        return f"{self.ruleset_name}-{self.scope_label}"


@dataclass(frozen=True, slots=True, init=False)
class SubrunResult:
    """Completed result bundle for one executed reproduction subrun.

    Group the executed subrun, per-case evaluation rows, derived accuracy
    report, requested evaluation count, and actual evaluated row count. This is
    the object summaries and pairwise comparisons consume after a notebook or
    session finishes one subrun.

    .. note::
        ``n_eval_requested`` and ``n_eval_actual`` may differ during exploratory
        work, for example when generation failures are skipped or a saved run is
        imported partially.

    Examples:
        ```python
        task_results = []
        for task in Progress(tasks, desc=subrun.name):
            answer = runtime.generate(
                GenerationRequest(question=task.question),
                policy=policy,
                rules=task.compiled_rules,
            )
            task_results.append(
                RulesetEvaluationResult(task.row, answer=answer)
            )

        subrun_result = SubrunResult(
            subrun,
            task_results,
            n_eval_requested=len(tasks),
        )
        print(subrun_result.report.accuracy)
        ```

    Attributes:
        subrun: Executed subrun descriptor.
        results: Immutable tuple of per-case evaluation rows.
        report: Accuracy report derived from ``results``.
        n_eval_requested: Number of evaluations requested by the caller.
        n_eval_actual: Number of evaluation rows actually stored.
        subrun_id: Stable identifier of the executed subrun.
        ruleset_name: Display name of the ruleset that produced the result.
        scope_label: Scope label used to group comparable subruns.

    Methods:
        :meth:`~ae_paper_reproduction.SubrunResult.subrun_id`
            Return the stable identifier of the executed subrun.

        :meth:`~ae_paper_reproduction.SubrunResult.ruleset_name`
            Return the display name of the ruleset that produced the result.

        :meth:`~ae_paper_reproduction.SubrunResult.scope_label`
            Return the scope label used to group comparable subruns.

        :meth:`~ae_paper_reproduction.SubrunResult.telemetry_items`
            Extract runtime telemetry items from evaluated answers.

    Runtime behavior:
        Construction materializes the result sequence, computes one accuracy
        report from exactly those rows, records requested and actual counts, and
        freezes the completed bundle for downstream reporting.

    Architectural role:
        Execution-to-reporting boundary between runner loops and summaries.

    Consumes:
        :class:`~ae_paper_reproduction.Subrun`
            Executed subrun plan.

        :class:`~ae_paper_reproduction.RulesetEvaluationResult`
            Per-case evaluated answers produced by the subrun.

    Produces:
        Accuracy and telemetry views consumed by
        :class:`~ae_paper_reproduction.Summary`, pairwise reports, and paper
        metric generation.

    Invariants:
        Counts and the aggregate report must reflect the stored immutable result
        tuple. The attached subrun must be the plan that produced those rows.

    Developer Notes:
        Keep this as the canonical result bundle. Summary and paper code should
        not carry parallel lists of rows, counters, and reports that can drift
        from each other.

    Todo:
        Add explicit skipped, failed, imported, or filtered counters if future
        telemetry needs those distinctions.

    See Also:
        :class:`~ae_paper_reproduction.Subrun`
        :class:`~ae_paper_reproduction.SubrunTask`
        :class:`~ae_paper_reproduction.RulesetEvaluationResult`
        :class:`~ae_paper_reproduction.Summary`

    """

    subrun: Subrun
    results: tuple[RulesetEvaluationResult, ...]
    report: AccuracyReport
    n_eval_requested: int
    n_eval_actual: int

    def __init__(
        self,
        subrun: Subrun,
        results: Sequence[RulesetEvaluationResult],
        *,
        n_eval_requested: int | None = None,
    ) -> None:
        """Build the completed result object for one executed subrun.

        Examples:
            ```python
            task_results = []
            for task in Progress(tasks, desc=subrun.name):
                policy = GenerationPolicy(rules=rules)
                result = runtime.generate(request, policy)
                task_results.append(
                    RulesetEvaluationResult(task.row, answer=result)
                )
            subrun_result = SubrunResult(subrun, task_results)
            ```

            ```python
            subrun_result = SubrunResult(
                subrun, imported_results,
                n_eval_requested=100,
            )
            ```

        Use this after a notebook loop has evaluated every selected task for a
        subrun. The constructor freezes the per-case results, computes the
        accuracy report once, records how many cases were requested, and stores
        how many evaluated rows were actually produced.

        This is the object that makes later notebook cells simple: summaries,
        pairwise comparisons, paper metric extraction, and telemetry dumps can
        consume one ``SubrunResult`` instead of separately passing the original
        subrun, rows, and manual counts around the notebook.

        Example:
            ```python
            task_results = []
            for task in Progress(tasks, desc=subrun.name):
                answer = runtime.generate(...)
                task_results.append(
                    RulesetEvaluationResult(task.row, answer=answer)
                )
            subrun_result = SubrunResult(subrun, task_results)
            ```

        Args:
            subrun: Executed subrun plan. The result keeps this object so
                downstream reports can display the subrun name, ruleset name,
                scope label, and requested task count without losing the
                connection to the notebook plan.
            results: Per-case evaluation rows produced for this subrun. The
                constructor materializes the sequence into an immutable tuple
                and derives the stored ``AccuracyReport`` from exactly these
                rows.
            n_eval_requested: Optional requested evaluation count. When omitted,
                the constructor uses ``subrun.last_task_count`` if task
                selection recorded one, otherwise it falls back to the number of
                supplied results. Provide this value for imported, partial, or
                externally filtered runs where the original requested count
                should remain visible.

        Notes:
            ``n_eval_requested`` and ``n_eval_actual`` may differ in exploratory
            reproduction work. For example, a notebook can request 100 questions
            but skip failed generations, filter by a question id, or import a
            subset of saved rows. Keeping both values makes those deviations
            visible in summaries.

        Developer notes:
            This constructor is the execution-to-reporting boundary. It should
            remain a single source of truth for the derived report, rather than
            allowing summary code to recompute accuracy from a separate mutable
            result list.

        Todo:
            If future telemetry reports distinguish skipped, failed, and
            intentionally filtered rows, extend this object with explicit
            counters rather than overloading the requested-versus-actual
            distinction.

        Validation guidance:
            Prefer deriving report data here from the immutable result tuple.
            This keeps summaries, paper metrics, and custom notebooks aligned
            even when exploratory runs request more rows than they actually
            complete.

        """
        requested = n_eval_requested
        if requested is None:
            requested = (
                subrun.last_task_count
                if subrun.last_task_count is not None
                else len(results)
            )
        materialized_results = tuple(results)
        object.__setattr__(self, "subrun", subrun)
        object.__setattr__(self, "results", materialized_results)
        object.__setattr__(self, "report", AccuracyReport(materialized_results))
        object.__setattr__(self, "n_eval_requested", requested)
        object.__setattr__(self, "n_eval_actual", len(materialized_results))

    @property
    def subrun_id(self) -> str:
        """Stable identifier of the executed subrun.

        Purpose:
            Expose the stable subrun identifier for this completed result.

        Architectural role:
            Derived accessor on the completed subrun-result record.

        Inputs (architectural provenance):
            Reads the stored executed subrun attached to this result.

        Outputs (downstream usage):
            A scalar naming value consumed by aggregation and summaries.

        Invariants/constraints:
            The derived value must match the executed subrun represented by this
            result.

        """
        return self.subrun.subrun_id

    @property
    def ruleset_name(self) -> str:
        """Display name of the ruleset that produced this result.

        Purpose:
            Expose the display name of the ruleset that produced this result.

        Architectural role:
            Derived accessor on the completed subrun-result record.

        Inputs (architectural provenance):
            Reads the stored executed subrun attached to this result.

        Outputs (downstream usage):
            A scalar naming value consumed by aggregation and summaries.

        Invariants/constraints:
            The derived value must match the executed subrun represented by this
            result.

        """
        return self.subrun.name

    @property
    def scope_label(self) -> str:
        """Scope label used to group comparable subruns.

        Purpose:
            Expose the scope label shared by this result and its comparable
            subruns.

        Architectural role:
            Derived accessor on the completed subrun-result record.

        Inputs (architectural provenance):
            Reads the stored executed subrun attached to this result.

        Outputs (downstream usage):
            A scalar naming value consumed by aggregation and summaries.

        Invariants/constraints:
            The derived value must match the executed subrun represented by this
            result.

        """
        return self.subrun.scope_label

    @property
    def mode(self) -> GenerationMode:
        """Explicit generation mode for the executed subrun."""
        return self.subrun.mode

    @property
    def paper_role(self) -> PaperRole | None:
        """Paper reporting role for the executed subrun."""
        return self.subrun.paper_role

    @property
    def paper_variant(self) -> str | None:
        """Paper reporting variant for the executed subrun."""
        return self.subrun.paper_variant

    def telemetry_items(self) -> tuple[TelemetryItem, ...]:
        """Extract one telemetry item per evaluated answer that captured.

        Purpose:
            Pull runtime telemetry snapshots from the stored evaluation results
            and normalize them into the item sequence consumed by run-level
            telemetry aggregation.

        Architectural role:
            Adapter method between completed subrun results and telemetry
            aggregation.

        Inputs (architectural provenance):
            Reads the evaluation results stored on this completed result.

        Outputs (downstream usage):
            A tuple of `TelemetryItem` objects consumed by telemetry merging.

        Invariants/constraints:
            Only evaluation results that actually carry runtime telemetry should
            contribute items.

        """
        return tuple(
            TelemetryItem(result.ae_telemetry)
            for result in self.results
            if result.ae_telemetry is not None
        )


@dataclass(frozen=True, slots=True)
class SubrunTask:
    """Executable dataset task selected for one reproduction subrun.

    Bind one dataset row to the subrun identifiers, scope labels, authored
    rules, and compiled rules needed by a notebook runner loop. Users normally
    receive these objects from
    :meth:`~ae_paper_reproduction.Subrun.select_tasks`.

    .. note::
        A task only prepares generation inputs. It does not call the model or
        judge the answer.

    Examples:
        ```python
        for task in subrun.select_tasks(n=10):
            request = GenerationRequest(question=task.question)
            answer = runtime.generate(
                request,
                policy=policy,
                rules=task.compiled_rules,
            )
            evaluated = RulesetEvaluationResult(task.row, answer=answer)
        ```

    Attributes:
        subrun_id: Stable identifier for the subrun that selected the task.
        ruleset_name: Human-readable ruleset or baseline name.
        scope_label: Human-readable scope/case label.
        case_type_filter: Optional case-type filter applied by the subrun.
        row: Original dataset row selected for execution.
        rules_markdown: Authored rules markdown associated with the subrun.
        compiled_rules: Compiled rules object supplied to generation.
        id: Dataset case identifier.
        question: Question text sent to the model.
        gold: Gold/reference expression used for scoring.
        case_type: Dataset case type used for grouping.

    Methods:
        :meth:`~ae_paper_reproduction.SubrunTask.id`
            Return the dataset case identifier for this task.

        :meth:`~ae_paper_reproduction.SubrunTask.question`
            Return the question text sent to the model.

        :meth:`~ae_paper_reproduction.SubrunTask.gold`
            Return the gold/reference answer used for scoring.

        :meth:`~ae_paper_reproduction.SubrunTask.case_type`
            Return the dataset case type used for grouping and filtering.

    Runtime behavior:
        The object exposes row-derived properties so notebook loops can use a
        compact, readable generation pattern without repeatedly reaching into
        the dataset row.

    Architectural role:
        Planning-to-execution value object used by reproduction notebooks and
        session runners.

    Consumes:
        :class:`~ae_paper_reproduction.Subrun` metadata and a dataset row.

    Produces:
        Inputs for :class:`~answer_engineering.GenerationRequest`,
        :class:`~answer_engineering.GenerationRuntime`, and
        :class:`~ae_paper_reproduction.RulesetEvaluationResult`.

    Invariants:
        Task metadata, dataset row, rules markdown, and compiled rules must all
        refer to the same subrun selection.

    Developer Notes:
        Keep this object easy to inspect in notebooks. It is a debugging aid for
        rule behavior, dataset selection, scope filtering, and telemetry
        experiments.

    Todo:
        Add explicit sampling/provenance fields if future task selection gains
        stratification, random seeds, or richer filtering.

    See Also:
        :class:`~ae_paper_reproduction.Subrun`
        :class:`~ae_paper_reproduction.RulesetEvaluationResult`
        :class:`~answer_engineering.GenerationRequest`

    """

    subrun_id: str
    ruleset_name: str
    scope_label: str
    mode: GenerationMode
    paper_role: PaperRole | None
    paper_variant: str | None
    case_type_filter: str | None
    row: DatasetRow
    rules_markdown: str
    compiled_rules: CompiledRules

    @property
    def id(self) -> str:
        """Dataset case identifier for this task.

        Purpose:
            Expose the dataset case identifier attached to this task.

        Architectural role:
            Read-only accessor on one executable subrun task.

        Inputs (architectural provenance):
            Reads the dataset row stored on the task.

        Outputs (downstream usage):
            A scalar task value consumed by session execution, output, or
            evaluation.

        Invariants/constraints:
            Derived values must stay aligned with the dataset row bound to this
            task.

        """
        return self.row.id

    @property
    def question(self) -> str:
        """Question text sent to the model for this task.

        Purpose:
            Expose the question text that should be sent to the model for this
            task.

        Architectural role:
            Read-only accessor on one executable subrun task.

        Inputs (architectural provenance):
            Reads the dataset row stored on the task.

        Outputs (downstream usage):
            A scalar task value consumed by session execution, output, or
            evaluation.

        Invariants/constraints:
            Derived values must stay aligned with the dataset row bound to this
            task.

        """
        return self.row.question

    @property
    def gold(self) -> str:
        """Gold expression used to score the generated answer.

        Purpose:
            Expose the gold expression used to judge the generated answer for
            this task.

        Architectural role:
            Read-only accessor on one executable subrun task.

        Inputs (architectural provenance):
            Reads the dataset row stored on the task.

        Outputs (downstream usage):
            A scalar task value consumed by session execution, output, or
            evaluation.

        Invariants/constraints:
            Derived values must stay aligned with the dataset row bound to this
            task.

        """
        return self.row.gold

    @property
    def case_type(self) -> str:
        """Case-type label taken from the bound dataset row.

        Purpose:
            Expose the case-type label attached to this task's dataset row.

        Architectural role:
            Read-only accessor on one executable subrun task.

        Inputs (architectural provenance):
            Reads the dataset row stored on the task.

        Outputs (downstream usage):
            A scalar task value consumed by session execution, output, or
            evaluation.

        Invariants/constraints:
            Derived values must stay aligned with the dataset row bound to this
            task.

        """
        return self.row.case_type


def _resolve_notebook_fallback_path(fallback: str | Path) -> Path:
    """Resolve a caller-supplied notebook path using the local fallback search.

    Purpose:
        Normalize the caller's fallback value into an existing notebook path,
        applying the module's fallback search rules before notebook extraction
        begins.

    Architectural role:
        Private path-resolution helper at the notebook-extraction edge of
        planning.

    Inputs (architectural provenance):
        Consumes the caller's fallback notebook path argument.

    Outputs (downstream usage):
        A resolved notebook path string or fast failure via exception.

    Invariants/constraints:
        The returned path must point to the notebook that will actually be
        parsed for rulesets.

    """
    path = Path(fallback)
    if path.is_absolute() or path.exists() or "google.colab" in sys.modules:
        return path

    cwd = Path.cwd()
    candidates = [cwd / path, cwd / "notebooks" / path.name]
    for child in cwd.iterdir():
        if not child.is_dir():
            continue
        candidates.append(child / path)
        candidates.append(child / "notebooks" / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


@dataclass(slots=True)
class Subrun:
    """Runnable reproduction subrun extracted from a notebook plan.

    Combine a notebook-derived subrun definition with a dataset and generation
    runtime so notebooks can select executable tasks, inspect rules, and run a
    baseline or rule-enabled configuration against the same benchmark rows.

    .. note::
        Accessing :attr:`compiled_rules` may compile the authored rules lazily.
        Selecting tasks still does not generate answers.

    Examples:
        ```python
        subruns = NotebookSubruns(
            "reproduce",
            dataset=dataset,
            model=runtime,
        )
        subrun = subruns[0]

        tasks = subrun.select_tasks(n=25)
        for task in Progress(tasks, desc=subrun.name):
            answer = runtime.generate(
                GenerationRequest(question=task.question),
                policy=policy,
                rules=task.compiled_rules,
            )
        ```

    Attributes:
        definition: Immutable subrun definition extracted from the notebook.
        dataset: Dataset used to select rows.
        model: Runtime associated with the notebook run.
        ruleset: Raw notebook ruleset specification.
        case_type: Optional case-type scope filter.
        index: Zero-based subrun order from the notebook.
        notebook_path: Source notebook path.
        ruleset_name: Human-readable ruleset name.
        scope_label: Human-readable scope label.
        rules_markdown: Authored Markdown rules for this subrun.
        system_prompt: System prompt selected for this subrun.
        name: Display name combining ruleset and scope context.
        subrun_id: Stable identifier used in reports.
        compiled_rules: Lazily compiled rules object.
        last_task_count: Number of tasks selected by the latest call.
        tasks: Cached full task tuple for this subrun.

    Methods:
        :meth:`~ae_paper_reproduction.api.Subrun.ruleset`
            Return the notebook ruleset specification bound to this subrun.

        :meth:`~ae_paper_reproduction.api.Subrun.case_type`
            Return the optional case-type filter applied by this subrun.

        :meth:`~ae_paper_reproduction.api.Subrun.index`
            Return the positional index of this subrun within the notebook plan.

        :meth:`~ae_paper_reproduction.api.Subrun.notebook_path`
            Return the notebook path that produced this subrun.

        :meth:`~ae_paper_reproduction.api.Subrun.ruleset_name`
            Return the human-readable ruleset name.

        :meth:`~ae_paper_reproduction.api.Subrun.scope_label`
            Return the human-readable scope label.

        :meth:`~ae_paper_reproduction.api.Subrun.rules_markdown`
            Return authored Markdown rules for this subrun.

        :meth:`~ae_paper_reproduction.api.Subrun.system_prompt`
            Return the system prompt selected for this subrun.

        :meth:`~ae_paper_reproduction.api.Subrun.name`
            Return the display name combining ruleset and scope context.

        :meth:`~ae_paper_reproduction.api.Subrun.subrun_id`
            Return the stable identifier used in reports.

        :meth:`~ae_paper_reproduction.api.Subrun.compiled_rules`
            Return the lazily compiled rules object.

        :meth:`~ae_paper_reproduction.api.Subrun.last_task_count`
            Return the number of tasks selected by the latest selection call.

        :meth:`~ae_paper_reproduction.api.Subrun.select_tasks`
            Select dataset rows and bind them into executable tasks.

        :meth:`~ae_paper_reproduction.api.Subrun.tasks`
            Return the cached full task tuple for this subrun.

    Runtime behavior:
        Selection applies the subrun scope to dataset rows and binds each row
        with subrun metadata, authored rules, and compiled rules. Generation is
        performed later by :class:`~answer_engineering.GenerationRuntime`.

    Architectural role:
        Planning-to-execution boundary for notebook reproduction.

    Consumes:
        :class:`~ae_paper_reproduction.SubrunDefinition`,
        :class:`~ae_paper_reproduction.Dataset`, and
        :class:`~answer_engineering.GenerationRuntime`.

    Produces:
        :class:`~ae_paper_reproduction.SubrunTask` objects for runner loops.

    Invariants:
        Subrun metadata must reflect notebook extraction order. Rule compilation
        should be idempotent for the same authored markdown, and task selection
        should preserve dataset/scope consistency.

    Developer Notes:
        Keep generation, answer judgement, and paper aggregation outside this
        class. This boundary should prepare executable tasks, not evaluate model
        behavior.

    Todo:
        Support richer selection strategies explicitly if future notebooks need
        stratified, seeded, or case-balanced sampling.

    See Also:
        :class:`~ae_paper_reproduction.NotebookSubruns`
        :class:`~ae_paper_reproduction.SubrunDefinition`
        :class:`~ae_paper_reproduction.SubrunTask`
        :class:`~answer_engineering.CompiledRules`

    """

    definition: SubrunDefinition
    dataset: Dataset
    model: GenerationRuntime
    _compiled_rules: CompiledRules | None = field(
        default=None, init=False, repr=False
    )
    _tasks: tuple[SubrunTask, ...] | None = field(
        default=None, init=False, repr=False
    )
    _last_task_count: int | None = field(default=None, init=False, repr=False)

    @property
    def ruleset(self) -> NotebookRulesetSpec:
        """Notebook ruleset spec bound to this subrun.

        Purpose:
            Expose the notebook ruleset specification bound to this subrun.

        Architectural role:
            Derived accessor on the runnable subrun plan.

        Inputs (architectural provenance):
            Reads stored subrun state and, for compiled rules, may trigger lazy
            compilation.

        Outputs (downstream usage):
            A scalar planning value or cached compiled-rules object consumed by
            execution and summaries.

        Invariants/constraints:
            Derived values must stay consistent with the subrun definition, and
            lazy compilation must remain idempotent for the same rules markdown.

        """
        return self.definition.ruleset

    @property
    def case_type(self) -> str | None:
        """Optional case-type filter applied when selecting rows.

        Purpose:
            Expose the optional case-type filter applied by this subrun.

        Architectural role:
            Derived accessor on the runnable subrun plan.

        Inputs (architectural provenance):
            Reads stored subrun state and, for compiled rules, may trigger lazy
            compilation.

        Outputs (downstream usage):
            A scalar planning value or cached compiled-rules object consumed by
            execution and summaries.

        Invariants/constraints:
            Derived values must stay consistent with the subrun definition, and
            lazy compilation must remain idempotent for the same rules markdown.

        """
        return self.definition.case_type

    @property
    def index(self) -> int:
        """Positional index of this subrun within the notebook plan.

        Purpose:
            Expose the positional index assigned to this subrun within the
            notebook plan.

        Architectural role:
            Derived accessor on the runnable subrun plan.

        Inputs (architectural provenance):
            Reads stored subrun state and, for compiled rules, may trigger lazy
            compilation.

        Outputs (downstream usage):
            A scalar planning value or cached compiled-rules object consumed by
            execution and summaries.

        Invariants/constraints:
            Derived values must stay consistent with the subrun definition, and
            lazy compilation must remain idempotent for the same rules markdown.

        """
        return self.definition.index

    @property
    def notebook_path(self) -> str:
        """Notebook path that produced this subrun.

        Purpose:
            Expose the notebook path that produced this subrun definition.

        Architectural role:
            Derived accessor on the runnable subrun plan.

        Inputs (architectural provenance):
            Reads stored subrun state and, for compiled rules, may trigger lazy
            compilation.

        Outputs (downstream usage):
            A scalar planning value or cached compiled-rules object consumed by
            execution and summaries.

        Invariants/constraints:
            Derived values must stay consistent with the subrun definition, and
            lazy compilation must remain idempotent for the same rules markdown.

        """
        return self.definition.notebook_path

    @property
    def ruleset_name(self) -> str:
        """Notebook-defined ruleset name for this subrun.

        Purpose:
            Expose the notebook-defined ruleset name for this subrun.

        Architectural role:
            Derived accessor on the runnable subrun plan.

        Inputs (architectural provenance):
            Reads stored subrun state and, for compiled rules, may trigger lazy
            compilation.

        Outputs (downstream usage):
            A scalar planning value or cached compiled-rules object consumed by
            execution and summaries.

        Invariants/constraints:
            Derived values must stay consistent with the subrun definition, and
            lazy compilation must remain idempotent for the same rules markdown.

        """
        return self.definition.ruleset_name

    @property
    def scope_label(self) -> str:
        """Scope label used to group comparable subruns.

        Purpose:
            Expose the normalized scope label used to group comparable subruns.

        Architectural role:
            Derived accessor on the runnable subrun plan.

        Inputs (architectural provenance):
            Reads stored subrun state and, for compiled rules, may trigger lazy
            compilation.

        Outputs (downstream usage):
            A scalar planning value or cached compiled-rules object consumed by
            execution and summaries.

        Invariants/constraints:
            Derived values must stay consistent with the subrun definition, and
            lazy compilation must remain idempotent for the same rules markdown.

        """
        return self.definition.scope_label

    @property
    def rules_markdown(self) -> str:
        """Authored rules markdown compiled and exported for this subrun.

        Purpose:
            Expose the authored rules markdown executed by this subrun.

        Architectural role:
            Derived accessor on the runnable subrun plan.

        Inputs (architectural provenance):
            Reads stored subrun state and, for compiled rules, may trigger lazy
            compilation.

        Outputs (downstream usage):
            A scalar planning value or cached compiled-rules object consumed by
            execution and summaries.

        Invariants/constraints:
            Derived values must stay consistent with the subrun definition, and
            lazy compilation must remain idempotent for the same rules markdown.

        """
        return self.definition.rules_markdown

    @property
    def system_prompt(self) -> str:
        """System prompt used when this subrun executes.

        Purpose:
            Expose the system prompt that should be used when this subrun
            executes.

        Architectural role:
            Derived accessor on the runnable subrun plan.

        Inputs (architectural provenance):
            Reads stored subrun state and, for compiled rules, may trigger lazy
            compilation.

        Outputs (downstream usage):
            A scalar planning value or cached compiled-rules object consumed by
            execution and summaries.

        Invariants/constraints:
            Derived values must stay consistent with the subrun definition, and
            lazy compilation must remain idempotent for the same rules markdown.

        """
        return self.definition.system_prompt

    @property
    def name(self) -> str:
        """Combined ruleset/scope name used in progress output and ids.

        Purpose:
            Expose the combined ruleset/scope display name for this subrun.

        Architectural role:
            Derived accessor on the runnable subrun plan.

        Inputs (architectural provenance):
            Reads stored subrun state and, for compiled rules, may trigger lazy
            compilation.

        Outputs (downstream usage):
            A scalar planning value or cached compiled-rules object consumed by
            execution and summaries.

        Invariants/constraints:
            Derived values must stay consistent with the subrun definition, and
            lazy compilation must remain idempotent for the same rules markdown.

        """
        return self.definition.name

    @property
    def subrun_id(self) -> str:
        """Canonical session identifier for this subrun.

        Purpose:
            Expose the canonical identifier for this subrun.

        Architectural role:
            Derived accessor on the runnable subrun plan.

        Inputs (architectural provenance):
            Reads stored subrun state and, for compiled rules, may trigger lazy
            compilation.

        Outputs (downstream usage):
            A scalar planning value or cached compiled-rules object consumed by
            execution and summaries.

        Invariants/constraints:
            Derived values must stay consistent with the subrun definition, and
            lazy compilation must remain idempotent for the same rules markdown.

        """
        return SubrunSession(
            index=self.index,
            ruleset_name=self.name,
        ).subrun_id

    @property
    def mode(self) -> GenerationMode:
        """Return the explicit generation mode for this subrun."""
        return self.definition.mode

    @property
    def paper_role(self) -> PaperRole | None:
        """Return paper reporting role metadata for this subrun."""
        return self.definition.paper_role

    @property
    def paper_variant(self) -> str | None:
        """Return paper reporting variant metadata for this subrun."""
        return self.definition.paper_variant

    @property
    def compiled_rules(self) -> CompiledRules:
        """Compiled rules for this subrun, materialized lazily on first access.

        Purpose:
            Expose the compiled rules object for this subrun, compiling lazily
            when first needed.

        Architectural role:
            Derived accessor on the runnable subrun plan.

        Inputs (architectural provenance):
            Reads stored subrun state and, for compiled rules, may trigger lazy
            compilation.

        Outputs (downstream usage):
            A scalar planning value or cached compiled-rules object consumed by
            execution and summaries.

        Invariants/constraints:
            Derived values must stay consistent with the subrun definition, and
            lazy compilation must remain idempotent for the same rules markdown.

        """
        if self._compiled_rules is None:
            self._compiled_rules = CompiledRules(self.rules_markdown)
        return self._compiled_rules

    @property
    def last_task_count(self) -> int | None:
        """Most recent materialized task count, if tasks have been selected.

        Purpose:
            Expose the last materialized task count recorded for this subrun, if
            any.

        Architectural role:
            Derived accessor on the runnable subrun plan.

        Inputs (architectural provenance):
            Reads stored subrun state and, for compiled rules, may trigger lazy
            compilation.

        Outputs (downstream usage):
            A scalar planning value or cached compiled-rules object consumed by
            execution and summaries.

        Invariants/constraints:
            Derived values must stay consistent with the subrun definition, and
            lazy compilation must remain idempotent for the same rules markdown.

        """
        return self._last_task_count

    def select_tasks(
        self,
        *,
        n: int | None = None,
        question_id: str | None = None,
    ) -> tuple[SubrunTask, ...]:
        """Select dataset rows and bind them into executable tasks.

        Examples:
            ```python
            tasks = subrun.select_tasks(n=10)
            for task in Progress(tasks, desc=subrun.name):
                policy = GenerationPolicy(rules=subrun.compiled_rules)
                result = runtime.generate(
                    GenerationRequest(question=task.question),
                    policy,
                )
            ```

            ```python
            debug_tasks = subrun.select_tasks(question_id="ssnhl-042")
            assert len(debug_tasks) <= 1
            ```

        Call this when a reproduction notebook wants to decide which cases from
        the materialized dataset will be run for one subrun. The method applies
        the subrun's scope filter, optionally narrows the selection to a count
        or a single question, and returns ``SubrunTask`` objects that already
        carry the compiled rules, ruleset metadata, scope label, and source
        dataset row.

        In normal paper reproduction, the notebook calls ``select_tasks(n=N)``
        for each subrun and then iterates over the returned tasks. In
        exploratory work, the same method is useful for smoke tests,
        single-question debugging, telemetry inspection, and comparing the
        baseline against one rule-enabled subrun without running the full
        dataset.

        Args:
            n: Optional maximum number of dataset rows to select. When provided
                without ``question_id``, the dataset adapter chooses up to this
                many rows within the subrun's case-type scope. Use small values
                for quick notebook checks and larger values for paper-style
                reproduction runs.
            question_id: Optional dataset question identifier to select exactly
                one case for debugging or focused comparison. When this argument
                is used, the method ignores ``n`` and then still applies the
                subrun's case-type scope if the subrun has one.

        Returns:
            Tuple of ``SubrunTask`` objects. Each task contains the original
            dataset row, the question text, subrun identifiers, human-readable
            labels, optional rules markdown, and the compiled rules passed to
            generation.

        Notes:
            Selecting tasks does not generate answers. It only prepares
            executable records for the runner loop. This separation is useful
            when users want to inspect the rule language, print the selected
            questions, or run custom telemetry collection before calling
            ``GenerationRuntime.generate``.

        Rule-language context:
            For rule-enabled subruns, each returned task carries the compiled
            form of the markdown rules authored in the reproduction notebook.
            Those rules can contain replacement or avoidance directives that
            intervene during generation. The notebook usually treats the
            compiled object as opaque, but users can inspect the originating
            rules markdown on the task or subrun when designing additional
            experiments.

        Developer notes:
            This method is the planning-to-execution expansion point. It should
            preserve dataset ordering, scope filtering, and compiled-rule
            binding while keeping generation concerns outside the planning
            layer.

        Todo:
            If future notebooks support stratified sampling, seeded sampling, or
            richer case filters, add explicit parameters here and document how
            they interact with ``n`` and ``question_id`` instead of hiding
            selection behavior in the dataset adapter.

        """
        rows = self.dataset.rows(
            n=n if question_id is None else None,
            question_id=question_id,
            case_type=self.case_type if question_id is None else None,
        )
        if question_id is not None and self.case_type is not None:
            rows = [row for row in rows if row.case_type == self.case_type]
        compiled_rules = self.compiled_rules
        tasks = tuple(
            SubrunTask(
                subrun_id=self.subrun_id,
                ruleset_name=self.name,
                scope_label=self.scope_label,
                mode=self.mode,
                paper_role=self.paper_role,
                paper_variant=self.paper_variant,
                case_type_filter=self.case_type,
                row=row,
                rules_markdown=self.rules_markdown,
                compiled_rules=compiled_rules,
            )
            for row in rows
        )
        self._last_task_count = len(tasks)
        if n is None and question_id is None:
            self._tasks = tasks
        return tasks

    @property
    def tasks(self) -> tuple[SubrunTask, ...]:
        """Cached task tuple for this subrun, materialized lazily on first.

        Purpose:
            Provide stable access to the executable tasks belonging to this
            subrun without forcing callers to manage task caching themselves.

        Architectural role:
            Lazy task-accessor on the runnable subrun plan.

        Inputs (architectural provenance):
            Reads or materializes the subrun's task list from the underlying
            dataset.

        Outputs (downstream usage):
            A tuple of tasks consumed by the session runner or inspection code.

        Invariants/constraints:
            Repeated access should return the same cached task tuple for the
            current subrun instance.

        """
        if self._tasks is None:
            self._tasks = self.select_tasks()
        return self._tasks


@dataclass(slots=True, init=False)
class NotebookSubruns:
    """Collection-like facade exposing runnable notebook subruns.

    Load the reproduction plan from a notebook, extract baseline and
    rule-enabled subruns, bind them to a shared dataset and runtime, and expose
    the result as a small ordered collection. This is the main notebook-facing
    entry point for turning authored rule cells into executable experiments.

    .. note::
        Iterating this object does not execute model generation. It only yields
        runnable subrun plans.

    Examples:
        ```python
        dataset = CachedHFDataset(DATASET_ID, SPLIT).materialize()
        runtime = GenerationRuntime(MODEL_ID).materialize()

        subruns = NotebookSubruns(
            "reproduce",
            dataset=dataset,
            model=runtime,
        )

        for subrun in subruns:
            print(subrun.name)
        ```

    Attributes:
        notebook_path: Resolved notebook path used as the source plan.
        dataset: Dataset abstraction shared by extracted subruns.
        model: Generation runtime shared by extracted subruns.
        subruns: Ordered tuple of extracted ``Subrun`` objects.

    Methods:
        :meth:`~ae_paper_reproduction.NotebookSubruns.__iter__`
            Iterate over available subruns.

        :meth:`~ae_paper_reproduction.NotebookSubruns.__len__`
            Return number of configured subruns.

        :meth:`~ae_paper_reproduction.NotebookSubruns.__getitem__`
            Return a subrun by index or a tuple by slice.

    Runtime behavior:
        Construction reads the notebook plan, extracts marked ruleset cells,
        creates scoped subrun definitions, and exposes ready-to-select
        ``Subrun`` objects. Rule compilation happens behind the subrun facade,
        before tasks are passed to generation.

    Architectural role:
        Public notebook boundary between notebook extraction/planning and
        execution loops.

    Consumes:
        Notebook path or name, a materialized dataset, and a generation runtime.

    Produces:
        Ordered :class:`~ae_paper_reproduction.Subrun` objects for session
        execution, smoke tests, telemetry experiments, and paper reproduction.

    Invariants:
        Collection order must match notebook extraction order. Construction
        should fail before generation if notebook parsing or ruleset planning is
        invalid.

    Developer Notes:
        Keep parser, compiler, and planner internals hidden behind this facade.
        By the time a subrun is yielded, notebook code should not need to
        understand extraction details.

    Todo:
        Make notebook plan diagnostics easier to inspect when users author new
        reproduction notebooks. Keep this facade focused on planning as paper
        metric generation moves toward a single source of truth.

    See Also:
        :class:`~ae_paper_reproduction.Subrun`
        :class:`~ae_paper_reproduction.SubrunTask`
        :class:`~ae_paper_reproduction.ReproductionSession`
        :class:`~answer_engineering.GenerationRuntime`

    """

    notebook_path: str
    dataset: Dataset
    model: GenerationRuntime
    subruns: tuple[Subrun, ...]

    def __init__(
        self,
        fallback: str | Path,
        *,
        dataset: Dataset,
        model: GenerationRuntime,
    ) -> None:
        """Load a notebook reproduction plan and expose executable subruns.

        Construct this object near the top of a reproduction notebook after
        preparing a dataset and model runtime. It reads the configured notebook
        plan from the given fallback path or notebook name, extracts the
        baseline and rule-enabled subruns, compiles rule markdown where needed,
        and binds every subrun to the shared dataset and model runtime.

        The resulting object behaves like a small ordered collection. Notebook
        users can iterate through it for full reproduction, index into it for a
        specific subrun, or inspect names, scopes, rules, and task counts before
        running generation.

        Examples:
            ```python
            dataset = CachedHFDataset(DATASET_ID, SPLIT).materialize()
            runtime = GenerationRuntime(MODEL_ID).materialize()
            subruns = NotebookSubruns(
                "reproduce",
                dataset=dataset,
                model=runtime,
            )

            for subrun in subruns:
                tasks = subrun.select_tasks(n=25)
                ...
            ```

        Args:
            fallback: Notebook name or path used to locate the reproduction
                plan. In the standard paper notebook this is usually
                ``"reproduce"`` or the notebook filename. The loader uses it as
                the user-facing source of subrun definitions when constructing
                the executable collection.
            dataset: Dataset adapter or materialized dataset shared by all
                extracted subruns. Each subrun selects rows from this dataset
                when ``select_tasks`` is called.
            model: Generation runtime or materializable model object shared by
                all extracted subruns. The object is stored with the plan so
                session code can run baseline and rule-enabled variants against
                the same runtime.

        Notes:
            This constructor is intentionally a high-level facade. Users should
            not need to import parser, compiler, or planner internals just to
            reproduce the paper. If construction fails, treat it as an early
            signal that the notebook plan, dataset binding, or rule markdown is
            invalid rather than as a runtime generation failure.

        Rule-language context:
            Rule-enabled subruns are created from markdown-like rule blocks used
            by Answer Engineering. The common notebook pattern is to compare a
            baseline run with one or more rulesets that replace, avoid, or
            redirect model output during generation. ``NotebookSubruns`` owns
            the conversion from authored notebook rules to compiled rules so
            later cells can focus on execution and telemetry.

        Suggested experiments:
            After constructing this object, users can run a single question
            across all subruns, run a small ``n`` for fast smoke testing, run
            only one case type, or add telemetry extraction around the
            generation loop to study how often rules fired and whether
            interventions improved or degraded accuracy.

        Developer notes:
            This constructor is the public notebook boundary for reproduction
            planning. Keep deep notebook parsing, subrun definition loading, and
            rule compilation behind this facade. The constructed object should
            be ready to iterate or index immediately.

        Todo:
            As the reporting model moves toward a single generated metrics file,
            keep this facade focused on execution planning and avoid mixing
            table rendering or paper-output concerns into construction.

        """
        notebook_path = str(_resolve_notebook_fallback_path(fallback))
        specs = extract_answer_engineering_subruns_from_ipynb(notebook_path)
        self.notebook_path = notebook_path
        self.dataset = dataset
        self.model = model
        self.subruns = tuple(
            Subrun(
                definition=SubrunDefinition(
                    ruleset=spec[0],
                    case_type=spec[1],
                    index=index,
                    notebook_path=notebook_path,
                    mode=_require_subrun_mode(spec[0], notebook_path),
                    paper_role=spec[0].paper_role,
                    paper_variant=spec[0].paper_variant,
                ),
                dataset=dataset,
                model=model,
            )
            for index, spec in enumerate(specs)
        )

    def __iter__(self) -> Iterator[Subrun]:
        """Iterate over subruns in notebook extraction order.

        Purpose:
            Provide iteration over the runnable subruns extracted from the
            notebook.

        Architectural role:
            Collection protocol method on the notebook-subruns facade.

        Inputs (architectural provenance):
            Reads the stored tuple of extracted subruns.

        Outputs (downstream usage):
            Subruns or collection protocol values consumed by caller code.

        Invariants/constraints:
            Collection access should reflect the extraction order stored on the
            facade.

        """
        return iter(self.subruns)

    def __len__(self) -> int:
        """Return the number of runnable subruns extracted from the notebook.

        Purpose:
            Expose the number of runnable subruns extracted from the notebook.

        Architectural role:
            Collection protocol method on the notebook-subruns facade.

        Inputs (architectural provenance):
            Reads the stored tuple of extracted subruns.

        Outputs (downstream usage):
            Subruns or collection protocol values consumed by caller code.

        Invariants/constraints:
            Collection access should reflect the extraction order stored on the
            facade.

        """
        return len(self.subruns)

    @overload
    def __getitem__(self, index: int) -> Subrun: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Subrun, ...]: ...

    def __getitem__(self, index: int | slice) -> Subrun | tuple[Subrun, ...]:
        """Return one subrun or a slice of subruns by notebook order.

        Purpose:
            Support notebook ergonomics for selecting a specific extracted
            ruleset or a contiguous group of planned subruns.

        Architectural role:
            Collection protocol method on the notebook-planning facade.

        Inputs (architectural provenance):
            Receives an integer index or slice from caller code and reads the
            stored ordered subrun tuple created during notebook extraction.

        Outputs (downstream usage):
            Returns either one `Subrun` or a tuple of `Subrun` objects for
            direct execution, inspection, or `ReproductionSession` construction.

        Invariants/constraints:
            Indexing must reflect extraction order and must not mutate the
            planned subrun collection. Overload stubs provide the static type
            split while this implementation owns the runtime behavior.

        """
        return self.subruns[index]


__all__ = [
    "NotebookSubruns",
    "Subrun",
    "SubrunDefinition",
    "SubrunResult",
    "SubrunTask",
]


def _require_subrun_mode(
    ruleset: NotebookRulesetSpec,
    notebook_path: str,
) -> GenerationMode:
    """Require explicit mode metadata for notebook-derived subruns."""
    if ruleset.mode is None:
        msg = (
            f"Ruleset {ruleset.ruleset_name!r} in {notebook_path} is missing "
            "an explicit '## Mode: ...' heading."
        )
        raise ValueError(msg)
    return ruleset.mode
