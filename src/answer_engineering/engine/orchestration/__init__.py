"""Runtime-edit orchestration package.

Purpose:
    Expose and group the orchestration modules that assemble stages, drive the
    runtime queue, and coordinate stage execution.

Architectural role:
    Top-level orchestration boundary above proposal, scoring, selection, and
    apply stages.

Owns:
    - Queue-driven control flow across runtime stages.
    - Assembly of stage dependencies and deterministic run wiring.

Does not own:
    - Proposal/scoring/selection/apply policy logic implemented in stage
      boundaries.
    - Telemetry schema/serialization beyond emitting runtime events.

"""
