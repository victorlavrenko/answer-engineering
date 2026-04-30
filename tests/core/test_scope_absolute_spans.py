from __future__ import annotations

from answer_engineering.engine.patching.patcher import (
    apply_patch,
)
from answer_engineering.engine.pipeline.context import (
    StepContext,
)
from answer_engineering.engine.proposal.proposal_engine import (
    ProposalPlanner,
)
from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    PatchOp,
    TextView,
)
from answer_engineering.rules.compile.plan import (
    AnchorQuerySpec,
    CandidateSpec,
    EditTargetSpec,
    PlanIR,
    RulePlan,
    ScopeSpec,
)
from tests._support.core_helpers import create_step_snapshot


def _replace_ctx(
    text: str, *, pattern: str, replacement: str, scope: ScopeSpec
) -> StepContext:
    doc = DocumentState(text)
    rule = RulePlan(
        rule_id="r1",
        scope=scope,
        anchors=(
            AnchorQuerySpec(
                anchor_id="m1", match_phrase_any=(pattern,), match_mode="last"
            ),
        ),
        target=EditTargetSpec(kind="match_span", anchor_id="m1"),
        candidates=(CandidateSpec(op=PatchOp.REPLACE, text=replacement),),
    )
    plan = PlanIR(rules=(rule,))
    guard_view = TextView(doc, rule.effective_guard_scope())
    edit_view = TextView(doc, rule.effective_edit_scope())
    return StepContext(
        plan=plan,
        rule=rule,
        doc=doc,
        step=create_step_snapshot(snapshot_text=doc.text, token_index=0),
        guard_view=guard_view,
        edit_view=edit_view,
    )


def test_tail_scope_match_returns_absolute_span_and_patch_applies() -> None:
    text = "prefix " + ("x" * 120) + " suggests diagnosis"
    scope = ScopeSpec(kind="tail_chars", n=40, casefold=True)
    ctx = _replace_ctx(
        text, pattern="suggests", replacement="demonstrates", scope=scope
    )

    proposals = ProposalPlanner().generate(ctx)
    proposal = next(p for p in proposals if p.op == PatchOp.REPLACE)

    assert proposal.span_abs is not None
    start, end = proposal.span_abs
    assert ctx.doc.text[start:end].lower() == "suggests"

    patched = apply_patch(ctx.doc, proposal)
    assert "demonstrates" in patched.text
    assert "suggests" not in patched.text


def test_whole_text_scope_span_matches_view_span() -> None:
    text = "aaa __TARGET__ zzz"
    scope = ScopeSpec(kind="tail_chars", n=len(text), casefold=False)
    ctx = _replace_ctx(
        text, pattern="__TARGET__", replacement="DONE", scope=scope
    )

    proposals = ProposalPlanner().generate(ctx)
    proposal = next(p for p in proposals if p.op == PatchOp.REPLACE)

    assert proposal.span_abs == (4, 14)
    patched = apply_patch(ctx.doc, proposal)
    assert patched.text == "aaa DONE zzz"
