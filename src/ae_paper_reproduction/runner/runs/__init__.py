"""Expose run-level application services for reproduction workflows.

Purpose:
    Reserve the run-level package for orchestration helpers that act on whole
    reproduction runs rather than individual domain objects.

Architectural role:
    Application subpackage under the reproduction runner.

Inputs (architectural provenance):
    Imported by session-level or higher-level orchestration code.

Outputs (downstream usage):
    Run-level services or facades consumed by external callers.

Invariants/constraints:
    This package should not become a miscellaneous bucket for unrelated
    utilities.

"""
