"""Runtime telemetry aggregation package.

Purpose:
    Group reducers that replay ordered runtime events into telemetry counters
    and frozen summary objects.

Architectural role:
    Internal implementation package for telemetry projection logic.

Owns:
    - event replay and counter projection (`aggregator`)

Does not own:
    - live event capture/sink plumbing (`telemetry.events`)
    - snapshot dataclass definitions (`telemetry.snapshots`)

Import policy:
    Import directly from concrete owning modules under this package.

"""
