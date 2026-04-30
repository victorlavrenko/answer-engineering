from __future__ import annotations

import pytest

from answer_engineering.engine.patching.patcher import (
    apply_patch,
)
from answer_engineering.engine.patching.proposals import PatchProposal
from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    PatchOp,
)


def test_reject_patch_when_base_version_mismatch() -> None:
    v1 = DocumentState("alpha beta")
    proposal_v1 = PatchProposal(
        op=PatchOp.REPLACE,
        span_abs=(6, 10),
        payload="BETA",
        base_version_id=v1.version_id,
        rule_id="r1",
        score=1.0,
        reason="valid edit",
    )

    v2 = apply_patch(
        v1,
        PatchProposal(
            op=PatchOp.INSERT_AFTER,
            span_abs=(5, 5),
            payload="!",
            base_version_id=v1.version_id,
            rule_id="r2",
            score=1.0,
            reason="valid edit",
        ),
    )

    with pytest.raises(ValueError, match="base version mismatch"):
        apply_patch(v2, proposal_v1)
