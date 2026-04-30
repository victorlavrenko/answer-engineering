"""Local winner selection within one scored rule group.

Purpose:
    Pick the best-scoring candidate for a rule-local competition and optionally
    reject it when the winner margin is too small.

Architectural role:
    Local selection helper used before global conflict resolution.

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, init=False)
class RuleWinnerDecision:
    """Outcome of local rule-level winner selection.

    Purpose:
        Record which candidate won within one scored rule group, the
        per-candidate probability ratios relative to the best score, and the
        winner-versus-runner-up margin.

    Architectural role:
        Local selection result object between scoring and later global conflict
        resolution.

    Inputs (architectural provenance):
        Constructed by `decide_rule_winner` after scoring one local proposal
        group.

    Outputs (downstream usage):
        Consumed by scoring stages when annotating scored proposals and deciding
        whether any candidate advances.

    Invariants/constraints:
        `winner_index` must either be `None` or point into `ratios_to_best` for
        the same score vector.

    """

    winner_index: int | None
    ratios_to_best: list[float]
    winner_ratio_to_runner_up: float

    def __init__(
        self,
        scores: Sequence[float],
        *,
        min_prob_ratio_to_best: float | None,
    ) -> None:
        """Select the local winner and enforce the margin threshold.

        Purpose:
            Convert a sequence of candidate scores into the accepted winner
            index, ratios to the best score, and winner-versus-runner-up ratio.

        Architectural role:
            Constructor-first validity boundary for rule-local winner decisions.

        Inputs (architectural provenance):
            Receives candidate scores from local scoring and an optional minimum
            probability-ratio threshold from selection policy.

        Outputs (downstream usage):
            Initializes immutable decision fields consumed by
            scoring/orchestration code when deciding whether a proposal advances
            to global conflict resolution.

        Invariants/constraints:
            Empty score sets and all-negative-infinity score sets produce no
            winner. When a threshold is supplied, the best candidate must clear
            the winner-versus-runner-up ratio to be accepted.

        """
        if not scores:
            object.__setattr__(self, "winner_index", None)
            object.__setattr__(self, "ratios_to_best", [])
            object.__setattr__(self, "winner_ratio_to_runner_up", 0.0)
            return
        winner_index = max(range(len(scores)), key=scores.__getitem__)
        best_score = scores[winner_index]
        if best_score == float("-inf"):
            object.__setattr__(self, "winner_index", None)
            object.__setattr__(self, "ratios_to_best", [0.0 for _ in scores])
            object.__setattr__(self, "winner_ratio_to_runner_up", 0.0)
            return
        ratios_to_best = score_ratios_to_best(scores)
        runner_up_score = max(
            (score for idx, score in enumerate(scores) if idx != winner_index),
            default=float("-inf"),
        )
        winner_ratio_to_runner_up = _ratio_from_gap(best_score, runner_up_score)
        accepted_winner = winner_index
        if (
            min_prob_ratio_to_best is not None
            and winner_ratio_to_runner_up < min_prob_ratio_to_best
        ):
            accepted_winner = None
        object.__setattr__(self, "winner_index", accepted_winner)
        object.__setattr__(self, "ratios_to_best", ratios_to_best)
        object.__setattr__(
            self, "winner_ratio_to_runner_up", winner_ratio_to_runner_up
        )


def score_ratios_to_best(scores: Sequence[float]) -> list[float]:
    """Return probability ratios for scores relative to the best score.

    Purpose:
        Convert local candidate scores into normalized relative ratios without
        changing the winner ordering.

    Architectural role:
        Numeric helper in the rule-local selection boundary.

    Inputs (architectural provenance):
        Receives scored candidates from local proposal scoring.

    Outputs (downstream usage):
        Returns one ratio per input score for `RuleWinnerDecision` and
        downstream telemetry/reporting.

    Invariants/constraints:
        Empty inputs return an empty list. Negative-infinity scores map to zero;
        when every score is negative infinity, every ratio is zero.

    """
    if not scores:
        return list()
    best = max(scores)
    if best == float("-inf"):
        return [0.0 for _ in scores]
    return [
        math.exp(score - best) if score != float("-inf") else 0.0
        for score in scores
    ]


def _ratio_from_gap(best_score: float, runner_up_score: float) -> float:
    """Convert a score gap into an exponential probability-ratio estimate.

    Purpose:
        Translate the best-versus-runner-up score gap into the ratio used by
        local winner-threshold checks.

    Architectural role:
        Numeric helper inside the rule-level selection boundary.

    Inputs (architectural provenance):
        Consumes best and runner-up scores computed during local winner
        selection.

    Outputs (downstream usage):
        Returned ratio is consumed by `decide_rule_winner` and persisted in
        `RuleWinnerDecision`.

    Invariants/constraints:
        Negative-infinity runner-up scores represent no competitor and therefore
        yield an infinite ratio.

    """
    if runner_up_score == float("-inf"):
        return float("inf")
    gap = best_score - runner_up_score
    return float("inf") if gap > 700 else math.exp(gap)
