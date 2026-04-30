from __future__ import annotations

import pytest

from answer_engineering.engine.pipeline.context import (
    StepContext,
)
from answer_engineering.engine.proposal.proposal_logic import (
    StandardProposalGenerator,
    _maybe_extend_after_span_until_parenthesis_close,  # pyright: ignore[reportPrivateUsage]
    _normalize_after_parenthetical_insertion_point,  # pyright: ignore[reportPrivateUsage]
)
from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    PatchOp,
    TextView,
    TokenCharAlignment,
)
from answer_engineering.rules.compile.plan import (
    AnchorQuerySpec,
    CandidateSpec,
    DecisionPolicySpec,
    EditTargetSpec,
    FirePolicySpec,
    PlanIR,
    RulePlan,
    ScopeSpec,
)
from tests._support.core_helpers import create_step_snapshot


def _base_ctx(*, candidate: CandidateSpec) -> StepContext:
    doc = DocumentState(text="Alpha Beta", version_id="v1")
    rule = RulePlan(
        rule_id="rid",
        name="replace:test",
        scope=ScopeSpec(kind="whole_doc"),
        target=EditTargetSpec(kind="scope_entire"),
        candidates=(candidate,),
        policy=DecisionPolicySpec(min_prob_ratio_to_best=0.0),
        fire=FirePolicySpec(mode="repeat"),
    )
    return StepContext(
        plan=PlanIR(rules=(rule,)),
        rule=rule,
        doc=doc,
        step=create_step_snapshot(
            snapshot_text=doc.text,
            token_index=0,
            generated_token_alignment=(
                TokenCharAlignment(
                    token_index=0, char_start=0, char_end=5, piece_text="Alpha"
                ),
                TokenCharAlignment(
                    token_index=1, char_start=5, char_end=6, piece_text=" "
                ),
                TokenCharAlignment(
                    token_index=2, char_start=6, char_end=10, piece_text="Beta"
                ),
            ),
        ),
        guard_view=TextView(
            doc=doc,
            abs_start=0,
            abs_end=len(doc.text),
        ),
        edit_view=TextView(
            doc=doc,
            abs_start=1,
            abs_end=len(doc.text),
        ),
    )


def test_replace_span_snaps_to_token_boundaries() -> None:
    ctx = _base_ctx(
        candidate=CandidateSpec(op=PatchOp.REPLACE, text="Z", kind="static")
    )
    proposals, _ = StandardProposalGenerator().generate(
        ctx, ctx.rule.candidates
    )

    assert len(proposals) == 1
    assert proposals[0].span_abs == (0, len(ctx.doc.text))


def test_insert_before_snaps_inside_token_start() -> None:
    ctx = _base_ctx(
        candidate=CandidateSpec(
            op=PatchOp.INSERT_BEFORE, text="Z", kind="static"
        )
    )
    proposals, _ = StandardProposalGenerator().generate(
        ctx, ctx.rule.candidates
    )

    assert len(proposals) == 1
    assert proposals[0].span_abs == (0, 0)


def test_empty_replace_span_is_not_backexpanded_by_alignment() -> None:
    text = (
        "Findings are consistent with sudden sensorineural hearing loss "
        "(SSNHL).\n\n\n"
    )
    doc = DocumentState(text=text, version_id="v1")
    cluster_start = text.index("(")
    rule = RulePlan(
        rule_id="rid-after",
        name="after:ssnhl",
        scope=ScopeSpec(kind="whole_doc"),
        anchors=(
            AnchorQuerySpec(
                anchor_id="a", match_phrase_any=("SSNHL",), match_mode="last"
            ),
        ),
        target=EditTargetSpec(kind="after_anchor_to_scope_end", anchor_id="a"),
        candidates=(
            CandidateSpec(
                op=PatchOp.REPLACE,
                text="This condition requires urgent treatment.",
                kind="static",
            ),
        ),
        policy=DecisionPolicySpec(min_prob_ratio_to_best=0.0),
        fire=FirePolicySpec(mode="repeat"),
    )
    ctx = StepContext(
        plan=PlanIR(rules=(rule,)),
        rule=rule,
        doc=doc,
        step=create_step_snapshot(
            snapshot_text=doc.text,
            token_index=0,
            generated_token_alignment=(
                TokenCharAlignment(
                    token_index=0,
                    char_start=0,
                    char_end=cluster_start,
                    piece_text=text[:cluster_start],
                ),
                TokenCharAlignment(
                    token_index=1,
                    char_start=cluster_start,
                    char_end=len(text),
                    piece_text=text[cluster_start:],
                ),
            ),
        ),
        guard_view=TextView(doc=doc, abs_start=0, abs_end=len(doc.text)),
        edit_view=TextView(doc=doc, abs_start=0, abs_end=len(doc.text)),
    )

    proposals, _ = StandardProposalGenerator().generate(ctx, rule.candidates)
    expected_insert_at = text.index(".\n") + 1
    assert len(proposals) == 1
    assert proposals[0].span_abs == (expected_insert_at, expected_insert_at)


def test_parenthetical_extension_scans_from_span_start() -> None:
    text = "(SSNHL).\n\n"
    doc = DocumentState(text=text, version_id="v1")
    anchor_start = text.index("(")
    anchor_end = text.index(")")
    rule = RulePlan(
        rule_id="rid-after-parenthetical",
        name="after:ssnhl",
        scope=ScopeSpec(kind="whole_doc"),
        target=EditTargetSpec(kind="after_anchor_to_scope_end", anchor_id="a"),
        anchors=(
            AnchorQuerySpec(
                anchor_id="a",
                match_phrase_any=("(SSNHL.",),
                match_mode="last",
            ),
        ),
        candidates=(
            CandidateSpec(
                op=PatchOp.REPLACE,
                text="This condition requires urgent treatment.",
                kind="static",
            ),
        ),
        policy=DecisionPolicySpec(min_prob_ratio_to_best=0.0),
        fire=FirePolicySpec(mode="repeat"),
    )
    ctx = StepContext(
        plan=PlanIR(rules=(rule,)),
        rule=rule,
        doc=doc,
        step=create_step_snapshot(snapshot_text=text, token_index=0),
        guard_view=TextView(doc=doc, abs_start=0, abs_end=len(doc.text)),
        edit_view=TextView(doc=doc, abs_start=0, abs_end=len(doc.text)),
    )
    span, reason = _maybe_extend_after_span_until_parenthesis_close(
        ctx=ctx,
        span=(text.index(")"), len(text)),
        anchor_span=(anchor_start, anchor_end),
    )
    index_after_dot = text.index(".") + 1
    assert reason is None
    assert span == (index_after_dot, index_after_dot)


def test_parenthetical_suffix_whitespace_span_normalizes() -> None:
    text = "(SSNHL).\n\n"
    start = text.index("\n")
    normalized = _normalize_after_parenthetical_insertion_point(
        text=text,
        span=(start, len(text)),
        max_end=len(text),
    )
    assert normalized == text.index(".") + 1


@pytest.mark.parametrize(
    ("text", "span", "expected"),
    [
        ("(SSNHL)\n\n", (7, 9), 7),
        ("(SSNHL).\n\n", (8, 10), 8),
        ("(SSNHL).   \n", (8, 12), 8),
        ("(SSNHL). More text", (8, 18), None),
    ],
)
def test_parenthetical_suffix_normalization_only_for_trailing_whitespace(
    text: str, span: tuple[int, int], expected: int | None
) -> None:
    normalized = _normalize_after_parenthetical_insertion_point(
        text=text,
        span=span,
        max_end=len(text),
    )
    assert normalized == expected
