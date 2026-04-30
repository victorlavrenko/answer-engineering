"""Runtime pipeline contract seam.

Purpose:
    Group the data contracts shared across orchestration and stages:
    step-context values, queue handoff messages, and runtime event records.

Architectural role:
    Contract boundary for runtime control/data handoffs.

Owns:
    - Immutable execution context values (`context`).
    - Runtime queue message envelopes (`messages`).
    - Runtime event record types for observability (`events`).

Does not own:
    - Stage execution logic (stages/orchestration).
    - Telemetry reduction and snapshot serialization.

Key relationships:
    - Produced by orchestration and stage stages.
    - Consumed by telemetry sinks/aggregation and result assembly.

"""
