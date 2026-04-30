"""Runtime telemetry snapshot package.

Purpose:
    Group immutable telemetry value objects that leave the runtime boundary as
    execution summaries.

Architectural role:
    Internal value-model package for telemetry outputs.

Owns:
    - frozen snapshot dataclasses and normalization helpers (`snapshots`)

Does not own:
    - runtime event sink behavior (`telemetry.events`)
    - replay/projection of events into counters (`telemetry.aggregation`)

Import policy:
    Import directly from concrete owning modules in this package.

"""
