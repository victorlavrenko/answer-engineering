from __future__ import annotations

import pytest

from answer_engineering.engine.orchestration.orchestrator import (
    PlanRunner,
)
from answer_engineering.engine.patching.patch_canonical import (
    canonicalize_payload,
)
from answer_engineering.engine.patching.patcher import (
    apply_patch,
    normalize_insert_payload,
    normalize_replace_payload,
)
from answer_engineering.engine.patching.proposals import PatchProposal
from answer_engineering.engine.proposal.proposal_engine import (
    ProposalPlanner,
)
from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    PatchOp,
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
from tests._support.core_helpers import (
    create_step_snapshot,
)


def test_insert_spacing_is_normalized_in_patcher() -> None:
    assert normalize_insert_payload(
        "Findings (SSNHL) and", 16, "requires urgent treatment"
    ) == (" requires urgent treatment")


def test_replace_spacing_is_normalized_in_patcher() -> None:
    text = "weber rinne left right positive this suggests conductive findings"
    start = text.index(" conductive")
    end = len(text)
    assert normalize_replace_payload(
        text, start, end, "these findings require further evaluation."
    ) == (" these findings require further evaluation.")


@pytest.mark.parametrize("right_punct", [")", ".", ",", ":", ";", "!", "?"])
def test_replace_spacing_does_not_insert_before_right_tight_punctuation(
    right_punct: str,
) -> None:
    text = f"(indicating sensorineural hearing loss{right_punct}"
    start = text.index("sensorineural")
    end = text.index(right_punct)
    assert normalize_replace_payload(text, start, end, "SSNHL") == "SSNHL"


def test_replace_spacing_no_insert_on_punct_sequence() -> None:
    text = "(indicating sensorineural hearing loss), and additional findings"
    start = text.index("sensorineural")
    end = text.index("),")
    assert normalize_replace_payload(text, start, end, "SSNHL") == "SSNHL"


def test_spacing_helpers_use_boundary_char_for_long_inputs() -> None:
    assert normalize_replace_payload("a bc)", 2, 4, "SSNHL") == "SSNHL"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("*sensorineural hearing loss*", "*SSNHL*"),
        ("**sensorineural hearing loss**", "**SSNHL**"),
    ],
)
def test_replace_spacing_keeps_markdown_emphasis_tight(
    text: str, expected: str
) -> None:
    start = text.index("sensorineural")
    end = text.index("*", start)
    replaced = (
        text[:start]
        + normalize_replace_payload(text, start, end, "SSNHL")
        + text[end:]
    )
    assert replaced == expected


def test_replace_spacing_is_added_when_right_side_is_alphanumeric() -> None:
    text = "(indicating sensorineural hearing lossand findings continue"
    start = text.index("sensorineural")
    end = text.index("and")
    assert normalize_replace_payload(text, start, end, "SSNHL") == "SSNHL "


def test_apply_patch_replace_uses_spacing_normalization() -> None:
    text = "alpha suggests conductive"
    doc = DocumentState(text)
    proposal = PatchProposal(
        op=PatchOp.REPLACE,
        span_abs=(text.index(" conductive"), len(text)),
        payload="this needs follow up",
        base_version_id=doc.version_id,
        rule_id="r1",
        score=1.0,
        reason="test",
    )
    payload_norm = canonicalize_payload(
        op=proposal.op,
        payload=proposal.payload,
        text=doc.text,
        span_abs=proposal.span_abs,
    )
    canonical = ProposalPlanner().freeze_normalized_proposal(
        doc,
        PatchProposal(
            op=proposal.op,
            span_abs=proposal.span_abs,
            payload=proposal.payload,
            payload_norm=payload_norm,
            base_version_id=proposal.base_version_id,
            rule_id=proposal.rule_id,
            score=proposal.score,
            reason=proposal.reason,
        ),
    )
    patched = apply_patch(doc, canonical)
    assert patched.text == "alpha suggests this needs follow up"


def test_fire_once_after_rule_does_not_reinsert_on_already_edited_text() -> (
    None
):
    rule = RulePlan(
        rule_id="r-after",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        anchors=(
            AnchorQuerySpec(
                anchor_id="a", match_phrase_any=("(SSNHL)",), match_mode="last"
            ),
        ),
        target=EditTargetSpec(kind="match_span", anchor_id="a"),
        candidates=(
            CandidateSpec(
                op=PatchOp.INSERT_AFTER,
                text="requires urgent treatment",
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )
    plan = PlanIR(rules=(rule,))

    first = PlanRunner().run(
        plan,
        create_step_snapshot(
            snapshot_text="Findings (SSNHL) and likely severe.", token_index=5
        ),
    )
    assert first.applied_patches
    second = PlanRunner().run(
        plan,
        create_step_snapshot(snapshot_text=first.final_doc.text, token_index=6),
    )
    assert not second.applied_patches


def test_fire_once_after_rule_skips_when_any_candidate_already_in_scope() -> (
    None
):
    rule = RulePlan(
        rule_id="r-after-multi",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=300, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=300, casefold=True),
        anchors=(
            AnchorQuerySpec(
                anchor_id="a", match_phrase_any=("SSNHL",), match_mode="last"
            ),
        ),
        target=EditTargetSpec(kind="match_span", anchor_id="a"),
        candidates=(
            CandidateSpec(
                op=PatchOp.INSERT_AFTER,
                text="Prompt treatment is indicated.",
                priority=10,
            ),
            CandidateSpec(
                op=PatchOp.INSERT_AFTER,
                text="Treatment should be initiated without delay.",
                priority=9,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )
    plan = PlanIR(rules=(rule,))

    text = (
        "This appears to be SSNHL. Treatment should be initiated without "
        "delay. Further recommendations follow."
    )
    result = PlanRunner().run(
        plan, create_step_snapshot(snapshot_text=text, token_index=10)
    )
    assert not result.applied_patches


def test_after_parenthetical_dot_capitalizes_inserted_text() -> None:
    rule = RulePlan(
        rule_id="r-after-dot",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        anchors=(
            AnchorQuerySpec(
                anchor_id="a",
                match_phrase_any=("(SSNHL).", "(SSNHL),", "(SSNHL)"),
                match_mode="last",
            ),
        ),
        target=EditTargetSpec(kind="match_span", anchor_id="a"),
        candidates=(
            CandidateSpec(
                op=PatchOp.INSERT_AFTER,
                text="prompt treatment is indicated.",
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )

    result = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(
            snapshot_text="Findings (SSNHL). Further details.", token_index=1
        ),
    )
    assert (
        result.final_doc.text
        == "Findings (SSNHL). Prompt treatment is indicated. Further details."
    )


def test_after_parenthetical_comma_decapitalizes_inserted_text() -> None:
    rule = RulePlan(
        rule_id="r-after-comma",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        anchors=(
            AnchorQuerySpec(
                anchor_id="a",
                match_phrase_any=("(SSNHL).", "(SSNHL),", "(SSNHL)"),
                match_mode="last",
            ),
        ),
        target=EditTargetSpec(kind="match_span", anchor_id="a"),
        candidates=(
            CandidateSpec(
                op=PatchOp.INSERT_AFTER,
                text="Prompt treatment is indicated.",
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )

    result = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(
            snapshot_text="Findings (SSNHL), with severe loss.", token_index=1
        ),
    )
    assert (
        result.final_doc.text
        == "Findings (SSNHL), prompt treatment is indicated. with severe loss."
    )


def test_after_parenthetical_without_punctuation_adds_dot_and_capitalizes() -> (
    None
):
    rule = RulePlan(
        rule_id="r-after-none",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        anchors=(
            AnchorQuerySpec(
                anchor_id="a",
                match_phrase_any=("(SSNHL).", "(SSNHL),", "(SSNHL)"),
                match_mode="last",
            ),
        ),
        target=EditTargetSpec(kind="match_span", anchor_id="a"),
        candidates=(
            CandidateSpec(
                op=PatchOp.INSERT_AFTER,
                text="treatment should be initiated without delay.",
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )

    result = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(
            snapshot_text="Findings (SSNHL) with severe loss.", token_index=1
        ),
    )
    assert result.final_doc.text == (
        "Findings (SSNHL). Treatment should be initiated without delay. "
        "with severe loss."
    )


def test_after_plain_anchor_without_punctuation_adds_dot_and_capitalizes() -> (
    None
):
    rule = RulePlan(
        rule_id="r-after-plain-none",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        anchors=(
            AnchorQuerySpec(
                anchor_id="a",
                match_phrase_any=("SSNHL.", "SSNHL,", "SSNHL"),
                match_mode="last",
            ),
        ),
        target=EditTargetSpec(kind="match_span", anchor_id="a"),
        candidates=(
            CandidateSpec(
                op=PatchOp.INSERT_AFTER,
                text="prompt treatment is indicated.",
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )

    result = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(
            snapshot_text="Findings SSNHL with severe loss.", token_index=1
        ),
    )
    assert (
        result.final_doc.text
        == "Findings SSNHL. Prompt treatment is indicated. with severe loss."
    )


def test_after_plain_anchor_with_comma_decapitalizes_inserted_text() -> None:
    rule = RulePlan(
        rule_id="r-after-plain-comma",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        anchors=(
            AnchorQuerySpec(
                anchor_id="a",
                match_phrase_any=("SSNHL.", "SSNHL,", "SSNHL"),
                match_mode="last",
            ),
        ),
        target=EditTargetSpec(kind="match_span", anchor_id="a"),
        candidates=(
            CandidateSpec(
                op=PatchOp.INSERT_AFTER,
                text="Prompt treatment is indicated.",
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )

    result = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(
            snapshot_text="Findings SSNHL, with severe loss.", token_index=1
        ),
    )
    assert (
        result.final_doc.text
        == "Findings SSNHL, prompt treatment is indicated. with severe loss."
    )


def test_unclosed_parenthetical_waits_for_closing_by_default() -> None:
    rule = RulePlan(
        rule_id="r-after-unclosed",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        anchors=(
            AnchorQuerySpec(
                anchor_id="a",
                match_phrase_any=("SSNHL.", "SSNHL,", "SSNHL", "(SSNHL"),
                match_mode="last",
            ),
        ),
        target=EditTargetSpec(kind="match_span", anchor_id="a"),
        candidates=(
            CandidateSpec(
                op=PatchOp.INSERT_AFTER,
                text="this condition requires urgent treatment.",
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )

    initial = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(
            snapshot_text="Findings SSNHL (Sudden Sensorineural Hearing Loss",
            token_index=1,
        ),
    )
    assert (
        initial.final_doc.text
        == "Findings SSNHL (Sudden Sensorineural Hearing Loss"
    )
    assert not initial.applied_patches

    closed = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(
            snapshot_text="Findings SSNHL (Sudden Sensorineural Hearing Loss).",
            token_index=2,
        ),
    )
    assert closed.final_doc.text == (
        "Findings SSNHL (Sudden Sensorineural Hearing Loss). "
        "This condition requires urgent treatment."
    )


def test_after_scope_end_replace_waits_for_closing_parenthesis_by_default() -> (
    None
):
    rule = RulePlan(
        rule_id="r-after-scope-end-replace",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=400, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=400, casefold=True),
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
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )

    initial = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(
            snapshot_text="Findings SSNHL (Sudden Sensorineural Hearing Loss",
            token_index=1,
        ),
    )
    assert (
        initial.final_doc.text
        == "Findings SSNHL (Sudden Sensorineural Hearing Loss"
    )
    assert not initial.applied_patches


def _after_scope_end_replace_rule() -> RulePlan:
    return RulePlan(
        rule_id="r-after-scope-end-replace-regression",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=400, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=400, casefold=True),
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
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )


@pytest.mark.parametrize(
    "text",
    [
        (
            "Findings are consistent with sudden sensorineural hearing loss "
            "(SSNHL).\n\n\n"
        ),
        "Findings **(SSNHL).** ",
        "Findings **Sudden Sensorineural Hearing Loss (SSNHL)**?! ",
        "Findings (SSNHL). ",
        "Findings (SSNHL).\n\n",
        "Findings (SSNHL).\n\n\n",
    ],
)
def test_after_scope_end_replace_preserves_parenthetical_suffix_structure(
    text: str,
) -> None:
    result = PlanRunner().run(
        PlanIR(rules=(_after_scope_end_replace_rule(),)),
        create_step_snapshot(snapshot_text=text, token_index=1),
    )
    assert "(SSNHL)" in result.final_doc.text
    assert "SSNHL." not in result.final_doc.text
    assert "This condition requires urgent treatment." in result.final_doc.text


def test_after_scope_end_replace_blank_line_shape_matches_expected_text() -> (
    None
):
    rule = _after_scope_end_replace_rule()
    text = (
        "Findings are consistent with sudden sensorineural hearing loss "
        "(SSNHL).\n\n\n"
    )

    result = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(snapshot_text=text, token_index=1),
    )

    expected = (
        "Findings are consistent with sudden sensorineural hearing loss "
        "(SSNHL). This condition requires urgent treatment.\n\n\n"
    )
    assert result.final_doc.text == expected
    assert "(SSNHL)" in result.final_doc.text
    assert "SSNHL." not in result.final_doc.text


def test_scope_end_replace_consumes_trailing_parenthetical_markers() -> None:
    rule = RulePlan(
        rule_id="r-after-scope-end-suffix-markers",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=400, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=400, casefold=True),
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
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )

    closed = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(
            snapshot_text=(
                "Findings **Sudden Sensorineural Hearing Loss (SSNHL)**?!"
            ),
            token_index=2,
        ),
    )
    assert closed.final_doc.text == (
        "Findings **Sudden Sensorineural Hearing Loss (SSNHL)**?!. "
        "This condition requires urgent treatment."
    )


def test_after_scope_end_replace_honors_comma_lowercasing() -> None:
    rule = RulePlan(
        rule_id="r-after-scope-end-comma",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=400, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=400, casefold=True),
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
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )

    result = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(snapshot_text="Findings SSNHL,", token_index=1),
    )
    assert (
        result.final_doc.text
        == "Findings SSNHL, this condition requires urgent treatment."
    )


def test_after_scope_end_replace_replays_debug_parenthesis_then_closes() -> (
    None
):
    rule = RulePlan(
        rule_id="r-after-scope-end-debug",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=2000, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=2000, casefold=True),
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
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )
    plan = PlanIR(rules=(rule,))

    unclosed = (
        "The Rinne test shows that air conduction is better than bone "
        "conduction in the "
        "right ear, which is consistent with SSNHL ("
    )
    first = PlanRunner().run(
        plan,
        create_step_snapshot(snapshot_text=unclosed, token_index=1),
    )
    assert first.final_doc.text == unclosed
    assert not first.applied_patches

    closed = (
        "The Rinne test shows that air conduction is better than bone "
        "conduction in the "
        "right ear, which is consistent with SSNHL (Sudden Sensorineural "
        "Hearing Loss)."
    )
    second = PlanRunner().run(
        plan, create_step_snapshot(snapshot_text=closed, token_index=2)
    )
    assert second.final_doc.text == (
        "The Rinne test shows that air conduction is better than bone "
        "conduction in the right ear, which is consistent with SSNHL "
        "(Sudden Sensorineural Hearing Loss). This condition requires "
        "urgent treatment."
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "Findings are consistent with SSNHL (Sudden Sensorineural "
            "Hearing Loss).",
            "Findings are consistent with SSNHL (Sudden Sensorineural "
            "Hearing Loss). "
            "This condition requires urgent treatment.",
        ),
        (
            "Findings are consistent with SSNHL (Sudden Sensorineural "
            "Hearing Loss), with concern.",
            "Findings are consistent with SSNHL (Sudden Sensorineural "
            "Hearing Loss), "
            "this condition requires urgent treatment. with concern.",
        ),
    ],
)
def test_after_scope_end_replace_preserves_parenthetical_text_and_punctuation(
    source: str, expected: str
) -> None:
    rule = RulePlan(
        rule_id="r-after-scope-end-preserve-parenthetical",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=400, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=400, casefold=True),
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
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )

    result = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(snapshot_text=source, token_index=1),
    )
    assert result.final_doc.text == expected


def test_scope_end_replace_empty_region_adds_period() -> None:
    rule = RulePlan(
        rule_id="r-after-empty-capital",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=300, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=300, casefold=True),
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
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )

    result = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(snapshot_text="Findings SSNHL", token_index=1),
    )
    assert (
        result.final_doc.text
        == "Findings SSNHL. This condition requires urgent treatment."
    )


def test_after_unclosed_parenthesis_can_opt_out_of_waiting() -> None:
    rule = RulePlan(
        rule_id="r-after-unclosed-opt-out",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        anchors=(
            AnchorQuerySpec(
                anchor_id="a", match_phrase_any=("SSNHL",), match_mode="last"
            ),
        ),
        target=EditTargetSpec(kind="match_span", anchor_id="a"),
        candidates=(
            CandidateSpec(
                op=PatchOp.INSERT_AFTER,
                text="this condition requires urgent treatment.",
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
        wait_for_closing_parenthesis=False,
    )

    result = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(
            snapshot_text="Findings SSNHL (Sudden Sensorineural Hearing Loss",
            token_index=1,
        ),
    )
    assert result.final_doc.text == (
        "Findings SSNHL. This condition requires urgent treatment. "
        "(Sudden Sensorineural Hearing Loss"
    )


def test_non_acronym_anchor_without_punct_adds_dot_caps() -> None:
    rule = RulePlan(
        rule_id="r-after-word-none",
        name="after:diagnosis",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=200, casefold=True),
        anchors=(
            AnchorQuerySpec(
                anchor_id="a",
                match_phrase_any=("diagnosis",),
                match_mode="last",
            ),
        ),
        target=EditTargetSpec(kind="match_span", anchor_id="a"),
        candidates=(
            CandidateSpec(
                op=PatchOp.INSERT_AFTER,
                text="prompt treatment is indicated.",
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )

    result = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(
            snapshot_text="Working diagnosis remains uncertain.", token_index=1
        ),
    )
    assert result.final_doc.text == (
        "Working diagnosis. Prompt treatment is indicated. remains uncertain."
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "Findings SSNHL with severe loss.",
            "Findings SSNHL. Prompt treatment is indicated. with severe loss.",
        ),
        (
            "Findings (SSNHL with severe loss.",
            "Findings (SSNHL with severe loss.",
        ),
        (
            "Findings (SSNHL) with severe loss.",
            "Findings (SSNHL). Prompt treatment is indicated. "
            "with severe loss.",
        ),
        (
            "Findings SSNHL, with severe loss.",
            "Findings SSNHL, prompt treatment is indicated. with severe loss.",
        ),
        (
            "Findings SSNHL. with severe loss.",
            "Findings SSNHL. Prompt treatment is indicated. with severe loss.",
        ),
        (
            "Findings (SSNHL), with severe loss.",
            "Findings (SSNHL), prompt treatment is indicated. "
            "with severe loss.",
        ),
        (
            "Findings (SSNHL). with severe loss.",
            "Findings (SSNHL). Prompt treatment is indicated. "
            "with severe loss.",
        ),
    ],
)
def test_after_ssnhl_anchor_variants_cover_plain_parenthetical_and_punctuation(
    source: str, expected: str
) -> None:
    rule = RulePlan(
        rule_id="r-after-variants",
        name="after:ssnhl",
        guard_scope=ScopeSpec(kind="tail_chars", max_chars=400, casefold=True),
        edit_scope=ScopeSpec(kind="tail_chars", max_chars=400, casefold=True),
        anchors=(
            AnchorQuerySpec(
                anchor_id="a",
                match_phrase_any=(
                    "SSNHL.",
                    "SSNHL,",
                    "SSNHL",
                    "(SSNHL).",
                    "(SSNHL),",
                    "(SSNHL)",
                    "(SSNHL.",
                    "(SSNHL,",
                    "(SSNHL",
                ),
                match_mode="last",
            ),
        ),
        target=EditTargetSpec(kind="match_span", anchor_id="a"),
        candidates=(
            CandidateSpec(
                op=PatchOp.INSERT_AFTER,
                text="Prompt treatment is indicated.",
                priority=10,
            ),
        ),
        policy=DecisionPolicySpec(skip_tokens=0),
        fire=FirePolicySpec(mode="once"),
    )

    result = PlanRunner().run(
        PlanIR(rules=(rule,)),
        create_step_snapshot(snapshot_text=source, token_index=1),
    )
    assert result.final_doc.text == expected
