"""Expose session-level reproduction application services.

Purpose:
    Group the concrete session orchestrators, progress helpers, console
    printers, and summary builders that execute one planned reproduction run.

Architectural role:
    Session-oriented application package under the reproduction runner.

Inputs (architectural provenance):
    Consumes planned subruns and runtime services from `core` and infrastructure
    layers.

Outputs (downstream usage):
    Session orchestration and summary components used by the public API.

Invariants/constraints:
    Code here may coordinate execution and output, but report calculations
    should remain in `core`.

"""

from ae_paper_reproduction.runner.session.eval_output import (
    EvaluationPrinter,
)
from ae_paper_reproduction.runner.session.execution_support import Progress
from ae_paper_reproduction.runner.session.reproduction_session import (
    ReproductionSession,
)
from ae_paper_reproduction.runner.session.summary import Summary

__all__ = ["ReproductionSession", "Summary", "EvaluationPrinter", "Progress"]
