"""Define the reproduction domain boundary.

Purpose:
    Group the data structures and pure transformations that describe experiment
    setup, evaluation results, and downstream aggregations independently of the
    session runner.

Architectural role:
    Domain package for reproduction concepts used by planning, evaluation, and
    aggregation code.

Inputs (architectural provenance):
    Imported by runner orchestration and public APIs that assemble reproducible
    experiment runs.

Outputs (downstream usage):
    Domain objects and pure report-building helpers consumed by the runner and
    reporting layers.

Invariants/constraints:
    Code here should remain free of runtime-side orchestration, progress I/O,
    and external publishing concerns.

"""
