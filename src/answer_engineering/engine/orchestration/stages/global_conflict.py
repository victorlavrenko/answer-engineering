"""Stage that performs global conflict resolution on scored proposals.

Purpose:
    Apply selector-based overlap resolution after scoring and translate rejected
    losers into ProposalRejected events.

Architectural role:
    Runtime stage between scoring and patch application.

"""

from __future__ import annotations

from dataclasses import dataclass

from answer_engineering.engine.pipeline.events import (
    ProposalRejected,
)
from answer_engineering.engine.pipeline.messages import (
    AcceptedPatchesReady,
    ScoredProposalsReady,
)
from answer_engineering.engine.selection.base import Selector


@dataclass(slots=True)
class GlobalConflictStage:
    """Run global conflict resolution across scored proposals.

    Purpose:
        Select a document-wide compatible set of local proposal winners before
        any patch is applied.

    Architectural role:
        Orchestration stage between scoring and application. It delegates
        conflict policy to the configured resolver instead of embedding that
        policy in the plan runner.

    Inputs (architectural provenance):
        Receives scored local winners and their candidate patch spans from
        upstream scoring and selection stages.

    Outputs (downstream usage):
        Returns the proposals that may proceed to the apply stage.

    Invariants/constraints:
        Conflict resolution must operate on scored proposal metadata only. It
        should not mutate document text or re-score candidates.

    """

    selector: Selector

    def handle(
        self,
        event: ScoredProposalsReady,
    ) -> GlobalConflictStage.HandleResult:
        """Return accepted proposals and rejection events for conflicts.

        Purpose:
            Run global conflict resolution over scored proposals for the current
            step.

        Architectural role:
            Orchestration stage between scoring/local selection and patch
            application.

        Inputs (architectural provenance):
            Receives scored proposals from the scoring stage and the current
            step context for event construction.

        Outputs (downstream usage):
            Returns accepted proposals, rejected proposals, and conflict events
            consumed by the apply stage and telemetry capture.

        Invariants/constraints:
            The stage must not mutate the document. Its responsibility is to
            make the accept/reject decision explicit before any patch is
            applied.

        """
        accepted_scored, rejected_scored = self.selector.resolve(event.scored)
        rejected_events = [
            ProposalRejected(rule_id=item.proposal.rule_id, reason="conflict")
            for item in rejected_scored
        ]
        return self.HandleResult(
            accepted=AcceptedPatchesReady(
                accepted=[item.proposal for item in accepted_scored]
            ),
            rejected_events=rejected_events,
        )

    @dataclass(frozen=True, slots=True)
    class HandleResult:
        """Return payload from the global-conflict stage.

        Purpose:
            Bundle the accepted-proposals message with the rejection events
            emitted for losers after global overlap arbitration.

        Architectural role:
            Stage-stage result object between global conflict resolution and
            apply-stage orchestration.

        Inputs (architectural provenance):
            Constructed by `GlobalConflictStage.handle()` after selector-based
            conflict resolution.

        Outputs (downstream usage):
            Consumed by orchestration code that forwards accepted patches and
            emits proposal-rejection events.

        Invariants/constraints:
            The accepted message and rejection events must describe the same
            conflict-resolution pass.

        """

        accepted: AcceptedPatchesReady
        rejected_events: list[ProposalRejected]

        def __iter__(self):
            """Yield the accepted message and rejection-event list in stage.

            Purpose:
                Support tuple-style unpacking of the global-conflict stage
                result without exposing its fields positionally at call sites.

            Architectural role:
                Small convenience adapter on the stage result object.

            Inputs (architectural provenance):
                Reads the already-constructed accepted message and
                rejection-event list.

            Outputs (downstream usage):
                Yielded values are consumed by orchestration code unpacking the
                handle result.

            Invariants/constraints:
                Iteration order must remain stable as `(accepted,
                rejected_events)`.

            """
            yield self.accepted
            yield self.rejected_events
