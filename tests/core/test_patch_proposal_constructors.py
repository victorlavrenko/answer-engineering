from __future__ import annotations

import pytest

from answer_engineering.engine.patching.proposals import (
    PatchProposal,
    ProposalContext,
)
from answer_engineering.engine.runtime.runtime_types import (
    PatchOp,
)
from answer_engineering.rules.compile.plan import (
    CandidateSpec,
)


def test_patch_proposal_from_candidate_populates_identity_and_scoring() -> None:
    proposal = PatchProposal.from_candidate(
        op=PatchOp.REPLACE,
        span_abs=(1, 4),
        payload="beta",
        reason="valid edit",
        context=ProposalContext(
            base_version_id="v1", rule_id="r1", guard_abs_start=0
        ),
        candidate=CandidateSpec(
            op=PatchOp.REPLACE,
            text="beta",
            kind="fallback",
            priority=3,
            candidate_id="c2",
            logprob=-1.25,
        ),
        payload_norm="beta",
        candidate_index=1,
        candidate_hash="hash2",
    )

    assert proposal.base_version_id == "v1"
    assert proposal.rule_id == "r1"
    assert proposal.candidate_id == "c2"
    assert proposal.candidate_kind == "fallback"
    assert proposal.score == 3.0
    assert proposal.cached_score_logprob == -1.25


def test_patch_proposal_noop_named_constructor_enforces_noop_shape() -> None:
    proposal = PatchProposal.noop(
        context=ProposalContext(base_version_id="v1", rule_id="r1"),
        reason="guard failed",
    )
    assert proposal.op == PatchOp.NOOP
    assert proposal.span_abs is None
    assert proposal.payload is None
    assert proposal.score == 0.0


def test_patch_proposal_non_noop_requires_span_and_payload() -> None:
    with pytest.raises(ValueError, match="span_abs is required"):
        PatchProposal(
            op=PatchOp.REPLACE,
            base_version_id="v1",
            rule_id="r1",
            reason="missing span",
            payload="x",
            score=1.0,
        )
