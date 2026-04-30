"""Runtime-edit engine package boundary.

Purpose:
    Identify the package that owns orchestration, proposal generation, scoring,
    selection, patching, and execution telemetry.

Architectural role:
    Package scaffold for the runtime control plane. Public runtime construction
    should remain at the package root or explicit subsystem boundaries, not here
    by accident.

Inputs (architectural provenance):
    The package contains subsystem modules rather than executable configuration
    loaded at import time.

Outputs (downstream usage):
    Provides an importable namespace for runtime internals and boundary audits.

Invariants/constraints:
    Avoid eager re-exports that create circular imports or blur ownership among
    runtime subsystems.

"""
