"""Answer Engineering telemetry public package.

Purpose:
    Expose the narrow stable telemetry surface used by downstream packages that
    consume runtime telemetry artifacts.

Architectural role:
    Public package boundary for telemetry contracts. External packages should
    import from this surface instead of reaching into internal engine modules.

Exports:
    Downstream-consumable telemetry snapshots and selected runtime event record
    types used by reproduction/reporting flows.

Boundary note:
    This package provides one canonical telemetry import surface and keeps
    downstream code decoupled from internal engine layout.

"""

from answer_engineering.engine.pipeline.events import (
    AvoidProbeCacheExhausted,
    AvoidProbeCandidatePopped,
    AvoidProbeEpisodeStarted,
    AvoidProbeSetGenerated,
    PatchSkipped,
    ProposalRejected,
    ProposalsGenerated,
)
from answer_engineering.engine.telemetry.snapshots.snapshots import (
    CandidateTelemetrySnapshot,
    ConditionTelemetrySnapshot,
    RuleTelemetrySnapshot,
    RuntimeTelemetrySnapshot,
)

__all__ = [
    "AvoidProbeCacheExhausted",
    "AvoidProbeCandidatePopped",
    "AvoidProbeEpisodeStarted",
    "AvoidProbeSetGenerated",
    "CandidateTelemetrySnapshot",
    "ConditionTelemetrySnapshot",
    "PatchSkipped",
    "ProposalRejected",
    "ProposalsGenerated",
    "RuleTelemetrySnapshot",
    "RuntimeTelemetrySnapshot",
]
