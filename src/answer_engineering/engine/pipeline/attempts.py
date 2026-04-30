"""Per-run attempt tracking for candidate reuse control.

Purpose:
    Hold deterministic attempt bookkeeping keyed by span, operation, and prompt
    prefix so orchestration can avoid retrying already-used candidates.

Architectural role:
    State-management utility inside runtime orchestration.

Inputs:
    Updated from selection and orchestration flow each time a candidate is
    considered or wins for a particular attempt lane.

Outputs:
    Provides duplicate checks and attempt snapshots that guide bounded retries,
    convergence behavior, and debugging.

"""

from __future__ import annotations

from dataclasses import dataclass, field


def _empty_used_candidate_hashes_by_key() -> dict[AttemptKey, set[str]]:
    return {}


def _empty_attempt_counts_by_key() -> dict[AttemptKey, int]:
    return {}


@dataclass(frozen=True, slots=True)
class AttemptKey:
    """Identity of one attempt lane within a run.

    Purpose:
        Distinguish independent retry histories by target span, operation, and
        the hashed prompt prefix that produced candidates.

    Architectural role:
        Value object shared by attempt tracking and selection logic.

    Inputs:
        Derived from the active step context and candidate-generation prefix.

    Outputs:
        Used as a stable dictionary key for counting attempts and storing
        winning candidate hashes.

    """

    span: tuple[int, int] | None
    op: str
    prefix_hash: str


@dataclass(slots=True)
class AttemptState:
    """Mutable per-run store for attempt counts and winning candidate hashes.

    Purpose:
        Track which candidates were already accepted for each attempt lane and
        how many attempts have been spent there.

    Architectural role:
        Retry-control state service owned by orchestration.

    Inputs:
        Updated by selection and orchestrator code as candidates are evaluated
        and winners recorded.

    Outputs:
        Serves duplicate-rejection checks and attempt-count queries that shape
        later selection decisions.

    """

    _used_candidate_hashes_by_key: dict[AttemptKey, set[str]] = field(
        default_factory=_empty_used_candidate_hashes_by_key
    )
    _attempt_counts_by_key: dict[AttemptKey, int] = field(
        default_factory=_empty_attempt_counts_by_key
    )

    def has_seen(self, key: AttemptKey, candidate_hash: str) -> bool:
        """Return whether this attempt lane has already accepted the hash."""
        return candidate_hash in self._used_candidate_hashes_by_key.get(
            key, set()
        )

    def record_attempt(
        self, key: AttemptKey, *, winning_candidate_hash: str | None = None
    ) -> None:
        """Record one proposal attempt and optionally its winning candidate.

        Purpose:
            Track how many attempts have been made for a rule/span lane and
            remember accepted candidate hashes so retries do not reuse the same
            winner.

        Architectural role:
            Mutable orchestration state behind retry and duplicate-suppression
            policy.

        Inputs (architectural provenance):
            Receives an `AttemptKey` from the proposal lane and, when selection
            succeeds, the hash of the winning candidate.

        Outputs (downstream usage):
            Updates attempt counters and duplicate-detection state consulted by
            later proposal passes.

        Invariants/constraints:
            Attempt counting happens even when no candidate wins. Candidate
            hashes are recorded only after acceptance so rejected alternatives
            remain eligible if later policy allows them.

        """
        self._attempt_counts_by_key[key] = (
            self._attempt_counts_by_key.get(key, 0) + 1
        )
        if winning_candidate_hash is not None:
            used = self._used_candidate_hashes_by_key.setdefault(key, set())
            used.add(winning_candidate_hash)

    def reject_duplicate(self, key: AttemptKey, candidate_hash: str) -> bool:
        """Return whether the candidate hash was already accepted for a lane.

        Purpose:
            Provide the duplicate-rejection predicate used before a proposal is
            allowed to compete again for the same attempt key.

        Architectural role:
            Retry-policy helper on orchestration attempt state.

        Inputs (architectural provenance):
            Receives the attempt lane identity and candidate hash produced by
            proposal construction or selection.

        Outputs (downstream usage):
            Returns `True` when the candidate should be rejected as a duplicate
            of an already accepted winner.

        Invariants/constraints:
            The method is read-only. It delegates to `has_seen` and must not
            increment attempt counters or record new winners.

        """
        return self.has_seen(key, candidate_hash)
