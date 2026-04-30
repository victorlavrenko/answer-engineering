"""Runtime telemetry event plumbing package.

Purpose:
    Organize the modules that own event-sink behavior and decision-log helpers
    used while a runtime execution is in progress.

Architectural role:
    Internal implementation package beneath ``engine.telemetry``. This package
    exists for code organization, not as a facade import surface.

Owns:
    - event sink protocols and concrete sink implementations (`event_sink`)
    - decision-log records, grouping helpers, and formatting utilities
      (`decision_logging`)

Does not own:
    - event replay/projection into summaries (`telemetry.aggregation`)
    - immutable telemetry snapshot dataclasses (`telemetry.snapshots`)

Import policy:
    Import directly from concrete owning modules such as ``event_sink`` or
    ``decision_logging``.

"""
