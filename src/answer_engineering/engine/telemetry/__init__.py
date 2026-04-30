"""Engine runtime telemetry boundary.

Purpose:
    Group the runtime-observability layers used during one engine execution:
    event sinks, decision-log formatting helpers, event aggregation, and
    immutable telemetry snapshots.

Architectural role:
    Namespace seam for telemetry internals under `answer_engineering.engine`.

Owns:
    - Runtime-facing observability code used while the pipeline is running.
    - Aggregation of pipeline events into runtime telemetry summaries.
    - Snapshot value types attached to generation results.

Does not own:
    - Serialization/export schemas used by repro reporting
      (`telemetry.representation`).
    - Core pipeline execution or proposal decisions themselves.

Data flow:
    - Pipeline/decode/orchestration code emits runtime events and debug lines.
    - `telemetry.events` handles sinks and decision-log event shaping.
    - `telemetry.aggregation` replays events into counter state.
    - `telemetry.snapshots` exposes immutable values attached to results.
    - `telemetry.representation` serializes those values later, outside this
      package.

Public surface:
    This package is mainly a namespace; behavior is owned by subpackages.

"""
