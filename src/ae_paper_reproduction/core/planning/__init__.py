"""Define the planning boundary for reproduction runs.

Purpose:
    Collect the types that describe which notebook rulesets, dataset slices, and
    model/runtime settings will become executable subruns.

Architectural role:
    Domain package for turning notebook-extracted specifications into runnable
    subrun plans.

Inputs (architectural provenance):
    Consumes notebook-derived ruleset specifications, datasets, and model
    runtimes.

Outputs (downstream usage):
    Subrun definitions, tasks, and notebook-level subrun collections consumed by
    the reproduction runner.

Invariants/constraints:
    Planning code should describe what will run; it should not perform
    session-level reporting or artifact publishing.

"""

from ae_paper_reproduction.core.planning.subruns import (
    NotebookSubruns,
    Subrun,
    SubrunDefinition,
    SubrunResult,
    SubrunTask,
)

__all__ = [
    "NotebookSubruns",
    "Subrun",
    "SubrunDefinition",
    "SubrunResult",
    "SubrunTask",
]
