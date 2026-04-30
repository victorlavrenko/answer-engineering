# Telemetry Schema

This page summarizes the current stable telemetry payload surfaces used by reproduction/reporting.

## Runtime telemetry source

Runtime telemetry is emitted by engine event sinks, aggregated into immutable snapshots, and exposed through:

- `answer_engineering.telemetry.RuntimeTelemetrySnapshot`
- related snapshot types (`RuleTelemetrySnapshot`, `ConditionTelemetrySnapshot`, `CandidateTelemetrySnapshot`)

## Serialized artifact payload shape

Reproduction/reporting serializes runtime telemetry via:

- `ae_paper_reproduction.telemetry.telemetry_types.serialize_runtime_telemetry(...)`

Current serialized payload keys include:

- `events`
- optional `runtime_sec`
- `rules_triggered_count`
- `rules_applied_count`
- `applied_decisions`
- `decision_limit_reached`
- optional `rules` (per-rule telemetry)

## Schema version for exported telemetry rows

`ae_paper_reproduction.telemetry.telemetry_types` defines:

- `SCHEMA_VERSION = "1"`

This schema version is carried in run/subrun context rows used by reporting artifacts.

## Architectural maturity note

The telemetry schema is stable enough for current downstream reporting consumers.

Reporting and publishing surfaces may continue evolving as telemetry consumers diversify, so schema usage should be treated as stable-by-contract for current outputs rather than frozen for all future reporting shapes.

## Scope of stability

Stable for downstream reporting consumers:

- top-level runtime snapshot transport values in `answer_engineering.telemetry`
- serialized telemetry row payload produced by `serialize_runtime_telemetry`

Not guaranteed stable:

- private internal telemetry aggregation state or event-loop internals.
