"""Define the reproduction application boundary.

Purpose:
    Own the application services that execute planned subruns, present progress,
    and assemble run summaries on top of the pure reproduction domain objects.

Architectural role:
    Top-level application package above `core`.

Inputs (architectural provenance):
    Consumes planned subruns and runtime services produced by the planning
    boundary.

Outputs (downstream usage):
    Runner entrypoints and session services used by public APIs and notebooks.

Invariants/constraints:
    Runner code may orchestrate execution and output, but domain calculations
    should stay in `core`.

"""

from ae_paper_reproduction.runner.session import ReproductionSession, Summary

__all__ = ["ReproductionSession", "Summary"]
