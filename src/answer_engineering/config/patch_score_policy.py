"""Patch-score policy knobs for context windows and score weighting.

Purpose:
    Define the default scoring policy used when candidate patches are compared
    with deterministic and model-backed evidence.

Architectural role:
    Configuration boundary between scoring implementation details and runtime
    orchestration. The module names score weights and context windows without
    coupling callers to logits-scoring internals.

Inputs (architectural provenance):
    Values originate from repository defaults or explicit caller overrides
    passed into the scoring pipeline.

Outputs (downstream usage):
    Provides policy objects and constants consumed by score task construction,
    model-score aggregation, and proposal ranking.

Invariants/constraints:
    Defaults must be deterministic and serializable enough for tests and
    telemetry to explain why two candidates received comparable scores.

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PatchScorePolicy:
    """Configure context windows and weighting for patch logprob scoring.

    Purpose:
        Keep the scoring-window and weighting knobs used when evaluating
        candidate patches in one immutable policy value.

    Architectural role:
        Configuration contract between selection/scoring code and caller-facing
        generation policy defaults.

    Inputs (architectural provenance):
        Constructed from package defaults or explicit runtime configuration
        before candidate scoring occurs.

    Outputs (downstream usage):
        Supplies prefix, replacement, and suffix context limits plus scoring
        weights to logprob-based patch evaluation.

    Invariants/constraints:
        The policy should describe scoring behavior only. It must not carry
        runtime tensors, tokenizer state, or mutable candidate-specific data.

    """

    n_left_ctx: int = 3
    n_right_ctx: int = 3
    w_left_ctx: float = 0.25
    w_right_ctx: float = 0.5
    score_left: bool = True
    score_right: bool = True
    score_replacement: bool = True
    continuation_tokens: int = 0
    continuation_weight: float = 1.0
