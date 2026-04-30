"""Inference-side probing boundary.

Owns now:
    - Probe-prefix construction and alignment-aware prefix reconstruction.
    - Probe-generation helpers (grouped-beam invocation + raw probe candidate
      flow).
    - Probe cache key/value records plus runtime cache reuse flow.
    - Probing debug/telemetry emission at runtime.

Does not own:
    - Proposal semantics (guard meaning, final selection policy).
    - A fully probing-native request/result API (not yet converged).

Current boundary leak:
    Runtime probing still depends on proposal-layer shapes (`StepContext`,
    `GenerationPrecheck`, `CandidateSpec`).

Todo:
    Introduce probing-native request/result contracts so probing runtime can
    stay inference-owned while proposal-facing adaptation moves outward.

"""
