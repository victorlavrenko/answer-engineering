"""Notebook-facing toolkit for reproducing and extending paper experiments.

Import from this package when you want to run the Answer Engineering paper
reproduction, inspect its intermediate outputs, or adapt the same pipeline to a
nearby experiment. The package is intentionally shaped as a small user surface:
it exposes dataset loading, notebook subrun discovery, model execution helpers,
per-task evaluation rows, progress printing, and final summary construction.

Typical use:
    ```python
    from ae_paper_reproduction import CachedHFDataset
    from ae_paper_reproduction import NotebookSubruns
    from ae_paper_reproduction import Progress
    from ae_paper_reproduction import RulesetEvaluationResult
    from ae_paper_reproduction import SubrunResult
    from ae_paper_reproduction import Summary
    from answer_engineering import GenerationRequest
    from answer_engineering import GenerationRuntime

    dataset = CachedHFDataset(DATASET_ID, SPLIT).materialize()
    runtime = GenerationRuntime(MODEL_ID).materialize()
    subruns = NotebookSubruns(NOTEBOOK_NAME, dataset=dataset, model=runtime)

    results = []
    for subrun in subruns:
        task_results = []
        tasks = subrun.select_tasks(n=TASK_LIMIT)
        for task in Progress(tasks, desc=subrun.name):
            request = GenerationRequest(question=task.question)
            policy = GenerationPolicy(rules=subrun.compiled_rules)
            generated = runtime.generate(
                request,
                policy,
            )
            task_results.append(
                RulesetEvaluationResult(task.row, answer=generated)
            )
        results.append(SubrunResult(subrun, task_results))
    summary = Summary(results)
    ```

What users can reproduce:
    The default notebook workflow compares a baseline generation run with one or
    more rule-enabled runs. Each subrun binds a dataset slice, a system prompt,
    and an optional compiled ruleset. The resulting outputs can be used for the
    paper's headline accuracy tables, but they are also useful for custom
    audits: inspecting failures, comparing rule behavior by case type, measuring
    how often interventions fired, or exporting telemetry for a different
    report.

Rule-language context:
    The reproduction notebooks usually author Answer Engineering rules in
    markdown cells. A ruleset is a small intervention program: Replace-style
    rules redirect text the model is about to produce, while Avoid-style rules
    probe alternatives when a forbidden or low-quality continuation is detected.
    Rules may include matching options such as case behavior, block scope, and
    replacement text. Users do not need to write a parser; notebook extraction
    and compilation happen before a subrun is exposed.

Telemetry and reporting context:
    Runtime telemetry is produced while generation runs. Evaluation objects then
    summarize dataset rows, model answers, correctness, and report fields. This
    separation lets users rerun only selected tasks, add additional telemetry
    checks, or produce a private analysis without changing the paper narrative.
    Useful follow-up experiments include running a smaller task limit during
    development, selecting only one case type, comparing baseline and ruleset
    failures side by side, and inspecting intervention events for surprising
    generated answers.

Import guidance:
    Prefer this facade over deep implementation imports in notebooks, examples,
    and documentation. Deep modules exist so maintainers can keep dataset,
    planning, runtime, telemetry, and artifact-writing boundaries clean, but a
    user trying the project should normally start here.

Developer notes:
    Every export here is part of the notebook-facing contract. Keep the first
    half of each exported docstring useful to a reader who has not read the
    paper. Architectural notes and TODOs are valuable, but they should appear
    after the practical workflow, expected behavior, examples, and extension
    points.

Todo:
    Keep this facade aligned with the single-source paper-metrics reporting
    model. The intended end state is that reproduction code writes one metrics
    file, while LaTeX tables and paper prose read from that file instead of
    relying on separate generated table artifacts or manually copied numbers.

"""

from ae_paper_reproduction.api import (
    CachedHFDataset,
    Dataset,
    NotebookSubruns,
    SubrunResult,
    SubrunTask,
    Summary,
)
from ae_paper_reproduction.core.evaluation.reports import (
    RulesetEvaluationResult,
)
from ae_paper_reproduction.runner.session import EvaluationPrinter, Progress

__all__ = [
    "SubrunResult",
    "SubrunTask",
    "Progress",
    "EvaluationPrinter",
    "RulesetEvaluationResult",
    "NotebookSubruns",
    "Dataset",
    "CachedHFDataset",
    "Summary",
]
