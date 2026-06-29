from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch

from answer_engineering.engine.orchestration.stages import apply as apply_stage
from answer_engineering.engine.orchestration.stages.apply import ApplyStage
from answer_engineering.engine.patching.proposals import (
    PatchProposal,
    ProposalContext,
)
from answer_engineering.engine.pipeline.events import (
    PatchSkipped,
    ProposalAccepted,
)
from answer_engineering.engine.pipeline.messages import (
    AcceptedPatchesReady,
)
from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    PatchOp,
)


def test_apply_stage_preserves_guard_abs_start_when_rebasing(
    monkeypatch: MonkeyPatch,
) -> None:
    captured: list[PatchProposal] = []

    def _fake_apply_patch(
        doc: DocumentState, proposal: PatchProposal
    ) -> DocumentState:
        captured.append(proposal)
        return DocumentState(text=doc.text, version_id=f"{doc.version_id}.1")

    monkeypatch.setattr(apply_stage.patcher, "apply_patch", _fake_apply_patch)

    proposal = PatchProposal.noop(
        context=ProposalContext(
            base_version_id="v1", rule_id="r1", guard_abs_start=7
        ),
        reason="noop",
    )
    stage = ApplyStage()

    stage.handle(
        AcceptedPatchesReady(accepted=[proposal]),
        doc=DocumentState(text="abc", version_id="v1"),
        applied_count=0,
    )

    assert len(captured) == 1
    assert captured[0].guard_abs_start == proposal.guard_abs_start


def test_apply_stage_emits_authored_label_for_static_candidate_acceptance(
    monkeypatch: MonkeyPatch,
) -> None:
    proposal = PatchProposal(
        op=PatchOp.REPLACE,
        span_abs=(0, 3),
        payload="SSNHL",
        base_version_id="v1",
        rule_id="r1",
        score=1.0,
        reason="valid edit",
        payload_norm="SSNHL",
        patch_bytes=b"",
        patch_hash="",
        candidate_kind="static",
        candidate_id="rewrite_2",
        candidate_label="SSNHL",
    )

    def _fake_apply_patch(
        doc: DocumentState, proposal: PatchProposal
    ) -> DocumentState:
        _ = proposal
        return DocumentState(text=doc.text, version_id=f"{doc.version_id}.1")

    monkeypatch.setattr(apply_stage.patcher, "apply_patch", _fake_apply_patch)

    result = ApplyStage().handle(
        AcceptedPatchesReady(accepted=[proposal]),
        doc=DocumentState(text="abc", version_id="v1"),
        applied_count=0,
    )

    accepted_events = [
        event
        for event in result.emitted_events
        if isinstance(event, ProposalAccepted)
    ]
    assert accepted_events
    accepted = accepted_events[0]
    assert accepted.candidate_id == "rewrite_2"
    assert accepted.candidate_label == "SSNHL"


def test_apply_stage_drops_wildly_invalid_accepted_span() -> None:
    doc = DocumentState(text="abc", version_id="v1")
    proposal = PatchProposal(
        op=PatchOp.REPLACE,
        span_abs=(0, len(doc.text) + 10000),
        payload="whole document replacement should not happen",
        base_version_id=doc.version_id,
        rule_id="r1",
        score=1.0,
        reason="invalid accepted edit",
        payload_norm="whole document replacement should not happen",
    )

    result = ApplyStage().handle(
        AcceptedPatchesReady(accepted=[proposal]),
        doc=doc,
        applied_count=0,
    )

    assert result.doc == doc
    skipped = [
        event
        for event in result.emitted_events
        if isinstance(event, PatchSkipped)
    ]
    assert skipped
    assert skipped[0].reason == "invalid_span_dropped"
    assert skipped[0].doc_len == len(doc.text)
    assert skipped[0].span_abs == proposal.span_abs
    assert skipped[0].stage == "apply"


def test_apply_stage_drops_base_version_mismatch_even_v0() -> None:
    doc = DocumentState(text="abc", version_id="v1")
    proposal = PatchProposal(
        op=PatchOp.REPLACE,
        span_abs=(0, 3),
        payload="xyz",
        base_version_id="v0",
        rule_id="r1",
        score=1.0,
        reason="stale edit",
        payload_norm="xyz",
    )

    result = ApplyStage().handle(
        AcceptedPatchesReady(accepted=[proposal]),
        doc=doc,
        applied_count=0,
    )

    assert result.doc == doc
    skipped = [
        event
        for event in result.emitted_events
        if isinstance(event, PatchSkipped)
    ]
    assert skipped
    assert skipped[0].reason == "proposal_base_version_mismatch"
    assert skipped[0].doc_len == len(doc.text)
    assert skipped[0].span_abs == proposal.span_abs
    assert skipped[0].stage == "apply"
