# Current Capabilities

This document describes **implemented behavior** only.

## Core runtime

Implemented:

-   rule-based trajectory editing
-   deterministic proposal generation
-   replace operations
-   avoid operations
-   scoped rule execution
-   immutable document snapshots
-   structured telemetry emission
-   reproducible evaluation pipeline

## Determinism

Deterministic:

-   proposal ordering
-   conflict resolution
-   rule execution behavior
-   event sequencing

Potentially non-deterministic:

-   model sampling
-   model scoring

## Observability

Implemented:

-   telemetry events
-   intervention tracking
-   evaluation artifacts

## Known limitations

Not implemented:

-   causal trajectory repair
-   alternative trajectory tracking
-   uncertainty signaling tokens
-   partial-history editing
-   dependency-aware upstream repair
