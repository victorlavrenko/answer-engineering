"""Stage that applies accepted proposals and emits apply events.

Purpose:
    Rebase and apply accepted patches in deterministic order and publish
    accepted/applied/skipped events.

Architectural role:
    Final runtime stage before result assembly.

"""

from __future__ import annotations

from dataclasses import dataclass

from answer_engineering.engine.patching import patch_canonical, patcher
from answer_engineering.engine.pipeline.events import (
    Event,
    PatchApplied,
    PatchSkipped,
    ProposalAccepted,
)
from answer_engineering.engine.pipeline.messages import (
    AcceptedPatchesReady,
    PatchAppliedReady,
)
from answer_engineering.engine.runtime.runtime_types import (
    AppliedPatch,
    DocumentState,
    PatchOp,
)


@dataclass(slots=True)
class ApplyStage:
    """Apply accepted proposals and emit patch lifecycle events.

    Purpose:
        Turn selected proposal decisions into document updates through the
        patching boundary.

    Architectural role:
        Orchestration stage after local/global selection. It coordinates patch
        application but does not decide proposal quality or conflict winners.

    Inputs (architectural provenance):
        Receives accepted proposal decisions, document versions, and event sinks
        from the plan runner context.

    Outputs (downstream usage):
        Produces updated document versions and emits accepted, skipped, and
        applied lifecycle events for telemetry.

    Invariants/constraints:
        The stage should preserve patcher ownership of text mutation semantics
        and keep observability separate from selection policy.

    """

    def handle(
        self,
        event: AcceptedPatchesReady,
        *,
        doc: DocumentState,
        applied_count: int,
    ) -> PatchAppliedReady:
        """Apply accepted proposals in deterministic order for the current step.

        Purpose:
            Materialize accepted edits into the step document and emit the
            resulting lifecycle events.

        Architectural role:
            Orchestration stage between conflict resolution and the next
            generation or proposal cycle.

        Inputs (architectural provenance):
            Receives the current step context plus accepted and rejected
            proposals from global conflict resolution.

        Outputs (downstream usage):
            Returns apply-stage output containing the updated context and
            telemetry events consumed by the plan runner.

        Invariants/constraints:
            Accepted proposals are applied in deterministic order. Rejected
            proposals remain observational data and must not modify the
            document.

        """
        current_doc = doc
        applied_patches: list[AppliedPatch] = []
        emitted_events: list[Event] = []

        ordered = sorted(
            event.accepted,
            key=lambda proposal: (
                -(
                    proposal.span_abs[0]
                    if proposal.span_abs is not None
                    else -1
                ),
                proposal.rule_id,
                -proposal.score,
            ),
        )
        for offset, proposal in enumerate(ordered):
            old_version = current_doc.version_id
            try:
                rebased = proposal.with_updates(
                    base_version_id=current_doc.version_id
                )
                if rebased.op in {
                    PatchOp.REPLACE,
                    PatchOp.INSERT_AFTER,
                    PatchOp.INSERT_BEFORE,
                }:
                    assert rebased.payload_norm is not None, (
                        "accepted proposal payload must be canonical"
                    )
                if rebased.patch_bytes:
                    assert rebased.patch_hash == patch_canonical.patch_hash(
                        rebased.patch_bytes
                    )
                next_doc = patcher.apply_patch(current_doc, rebased)
                applied_patch = AppliedPatch(
                    patch_id=f"{proposal.rule_id}:{applied_count + offset}",
                    proposal=proposal,
                    new_version_id=next_doc.version_id,
                )
            except ValueError as exc:
                emitted_events.append(
                    PatchSkipped(rule_id=proposal.rule_id, reason=str(exc))
                )
                continue

            current_doc = next_doc
            patch_id = applied_patch.patch_id
            applied_patches.append(applied_patch)
            candidate_id = (
                proposal.candidate_id
                or f"candidate_{proposal.candidate_index + 1}"
            )
            candidate_label = proposal.candidate_label or candidate_id
            emitted_events.append(
                ProposalAccepted(
                    rule_id=proposal.rule_id,
                    proposal_summary=proposal.reason,
                    patch_hash=proposal.patch_hash,
                    patch_bytes_len=len(proposal.patch_bytes),
                    candidate_kind=proposal.candidate_kind,
                    candidate_id=candidate_id,
                    candidate_label=candidate_label,
                    candidate_text_excerpt=(
                        proposal.payload[:80] if proposal.payload else None
                    ),
                )
            )
            emitted_events.append(
                PatchApplied(
                    rule_id=proposal.rule_id,
                    patch_id=patch_id,
                    old_version_id=old_version,
                    new_version_id=current_doc.version_id,
                    patch_hash=proposal.patch_hash,
                    patch_bytes_len=len(proposal.patch_bytes),
                )
            )

        return PatchAppliedReady(
            doc=current_doc,
            applied_patches=applied_patches,
            emitted_events=emitted_events,
        )
