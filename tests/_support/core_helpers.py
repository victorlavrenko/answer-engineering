from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from answer_engineering.engine.patching.patcher import (
    apply_patch,
)
from answer_engineering.engine.patching.proposals import PatchProposal
from answer_engineering.engine.pipeline.context import StepSnapshot
from answer_engineering.engine.runtime.runtime_types import (
    Decision,
    DocumentState,
    TokenAlignedTextView,
    TokenCharAlignment,
)
from answer_engineering.inference.decode.session_orchestration import (
    ExecutionSession,
)


@dataclass(frozen=True, slots=True)
class DummyTokenAlignedTextView:
    assistant_visible_text: str
    generated_token_ids: list[int]
    generated_token_alignment: list[TokenCharAlignment]


def apply_proposal_to_text(text: str, proposal: PatchProposal | None) -> str:
    if proposal is None:
        return text
    doc = DocumentState(text)
    p = PatchProposal(
        op=proposal.op,
        span_abs=proposal.span_abs,
        payload=proposal.payload,
        base_version_id=doc.version_id,
        rule_id=proposal.rule_id,
        score=proposal.score,
        reason=proposal.reason,
    )
    return apply_patch(doc, p).text


def create_step_snapshot(
    snapshot_text: str,
    token_index: int,
    generated_ids: tuple[int, ...] | None = None,
    generated_token_alignment: tuple[TokenCharAlignment, ...] | None = None,
    prompt_ids: torch.Tensor | None = None,
    prompt_text: str = "",
) -> StepSnapshot:
    dummy_state = DummyTokenAlignedTextView(
        assistant_visible_text=snapshot_text,
        generated_token_ids=(
            list(generated_ids) if generated_ids is not None else list()
        ),
        generated_token_alignment=(
            list(generated_token_alignment)
            if generated_token_alignment is not None
            else list()
        ),
    )
    return StepSnapshot(
        state=cast(TokenAlignedTextView, dummy_state),
        token_index=token_index,
        prompt_ids=prompt_ids,
        prompt_text=prompt_text,
    )


def step_test(
    engine: ExecutionSession,
    snapshot_text: str,
    token_index: int,
) -> Decision:
    return engine.execute_step(
        create_step_snapshot(
            snapshot_text=snapshot_text, token_index=token_index
        )
    )
