"""Stable import facade for paper-reproduction notebooks and scripts.

Use this module when you want one predictable import location for the public
reproduction workflow. It collects the objects needed to load the benchmark
split, prepare notebook-defined subruns, run selected tasks, print progress,
convert generated answers into evaluation rows, and build a final summary.

Typical use:
    ```python
    from ae_paper_reproduction.api import CachedHFDataset
    from ae_paper_reproduction.api import NotebookSubruns
    from ae_paper_reproduction.api import SubrunResult
    from ae_paper_reproduction.api import Summary

    dataset = CachedHFDataset(DATASET_ID, SPLIT)
    subruns = NotebookSubruns(
        NOTEBOOK_NAME,
        dataset=dataset,
        model=runtime,
    )
    summary = Summary([SubrunResult(subrun, rows) for subrun, rows in runs])
    ```

Why this facade exists:
    The reproduction package has implementation modules for dataset access,
    notebook extraction, ruleset compilation, evaluation reports, telemetry
    shaping, and artifact writing. Those modules are intentionally separated so
    maintainers can refactor internals without forcing notebook users to learn
    the whole architecture. This facade is the library-style entry point.

What to expect:
    Objects imported here are safe to use in notebooks and small scripts. They
    are designed to fail early when a dataset, model, notebook plan, or ruleset
    cannot be prepared. They do not hide expensive work once materialization or
    generation starts: dataset loading, model loading, and model-backed
    generation can still take time and resources.

Rule-language context:
    Subruns discovered through this facade may carry compiled Answer Engineering
    rules. These rules come from notebook-authored text blocks and describe
    interventions such as replacing a risky continuation or avoiding a
    disallowed answer pattern. Users can experiment by editing the notebook
    rules, rerunning extraction, and comparing the resulting subruns.

Telemetry and custom analysis:
    The facade supports the paper reproduction path, but it is not limited to
    the exact paper tables. A user can select a small subset of tasks, inspect
    per-task correctness, compare cases by subrun, check generated answers, and
    use runtime telemetry to ask questions that are not in the paper.

Developer notes:
    Keep exports here deliberate. Adding an object to this module makes it
    notebook-facing and should imply a user-quality docstring. Avoid reexporting
    low-level helpers only to make internal code shorter.

Todo:
    Keep the facade synchronized with the reporting simplification where the
    reproduction pipeline emits one paper-metrics source of truth instead of
    multiple generated LaTeX tables.

"""

from ae_paper_reproduction.core.planning import (
    NotebookSubruns,
    Subrun,
    SubrunDefinition,
    SubrunResult,
    SubrunTask,
)
from ae_paper_reproduction.core.planning.notebook_extractor import (
    NotebookRulesetSpec,
)
from ae_paper_reproduction.infra.datasets import CachedHFDataset, Dataset
from ae_paper_reproduction.runner import ReproductionSession, Summary

__all__ = [
    "Dataset",
    "CachedHFDataset",
    "ReproductionSession",
    "NotebookSubruns",
    "Subrun",
    "SubrunDefinition",
    "SubrunResult",
    "SubrunTask",
    "Summary",
    "NotebookRulesetSpec",
]
