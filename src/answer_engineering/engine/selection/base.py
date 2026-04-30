"""Selection protocols for rule winners and global conflicts.

Purpose:
    Define the scorer-facing and orchestration-facing contracts used after
    proposals have been generated and scored.

Architectural role:
    Selection boundary between local per-rule ranking and document-wide conflict
    resolution. The module keeps selection pluggable without leaking scorer or
    patcher implementation details.

Inputs (architectural provenance):
    Receives proposal groups, scores, patch spans, and document context from
    orchestration stages.

Outputs (downstream usage):
    Produces winner decisions and conflict-filtered proposal sets consumed by
    the apply stage.

Invariants/constraints:
    Selection implementations should be deterministic for equal inputs and
    should not mutate proposal objects while resolving conflicts.

"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from answer_engineering.engine.scoring.base import ScoredProposal


class Selector(Protocol):
    """Protocol for selecting accepted scored proposals.

    Purpose:
        Define the common selector operations used for local winner picking and
        global conflict resolution.

    Architectural role:
        Scoring-to-patching decision boundary in the engine pipeline.

    Inputs (architectural provenance):
        Receives scored proposals from scoring stages or grouped orchestration
        helpers.

    Outputs (downstream usage):
        Produces either one selected proposal or accepted/rejected proposal
        lists for downstream patch application and telemetry.

    Invariants/constraints:
        Selection must be deterministic for equal inputs so tests and
        reproduction runs remain stable.

    """

    def select(self, scored: Iterable[ScoredProposal]) -> ScoredProposal | None:
        """Select one scored proposal from an iterable, or return none.

        Purpose:
            Choose the local winning proposal for a single selection context.

        Architectural role:
            Abstract local-selection boundary after scoring and before global
            conflict resolution.

        Inputs (architectural provenance):
            Receives scored proposals produced by a scorer for one runtime step
            or rule group.

        Outputs (downstream usage):
            Returns the selected score record, or `None` when no proposal should
            advance.

        Invariants/constraints:
            Selection should not mutate proposals or apply patches. It only
            chooses from already scored candidates.

        """
        raise NotImplementedError

    def resolve(
        self, scored: Iterable[ScoredProposal]
    ) -> tuple[list[ScoredProposal], list[ScoredProposal]]:
        """Resolve scored proposals into accepted and rejected lists.

        Purpose:
            Partition scored candidates after local scoring into proposals that
            may be applied and proposals that should be rejected.

        Architectural role:
            Abstract conflict-resolution boundary consumed by orchestration
            stages.

        Inputs (architectural provenance):
            Receives scored proposal records from scoring or local-selection
            stages.

        Outputs (downstream usage):
            Returns accepted and rejected proposal lists used by apply stages
            and telemetry event construction.

        Invariants/constraints:
            Implementations should preserve deterministic ordering and explain
            rejected proposals through their returned records or companion
            events.

        """
        raise NotImplementedError
