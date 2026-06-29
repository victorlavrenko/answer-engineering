from __future__ import annotations

# pyright: reportPrivateUsage=false
import logging
import random
from typing import Any, cast

import pytest

from answer_engineering.engine.orchestration.stages.scoring import (
    _editable_proposals,
)
from answer_engineering.engine.patching import patcher
from answer_engineering.engine.patching.proposals import PatchProposal
from answer_engineering.engine.pipeline import text_bounds
from answer_engineering.engine.pipeline.context import StepContext
from answer_engineering.engine.pipeline.events import PatchSkipped
from answer_engineering.engine.pipeline.messages import ProposalsReady
from answer_engineering.engine.proposal.proposal_logic import (
    _snap_span_to_token_boundaries,
)
from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    PatchOp,
    TextView,
    TokenCharAlignment,
)
from answer_engineering.engine.span_utils import (
    is_valid_span,
    normalize_span,
    validate_token_alignment_detailed,
)
from answer_engineering.engine.telemetry.events.event_sink import (
    RecordingRuntimeEventSink,
)
from answer_engineering.inference.decode.session_orchestration import (
    RetokenizedAssistant,
)
from answer_engineering.inference.probing.prefix.request_prefix import (
    _build_probe_prefix_ids,
)
from answer_engineering.rules.compile.plan import (
    DecisionPolicySpec,
    EditTargetSpec,
    FirePolicySpec,
    PlanIR,
    RulePlan,
    ScopeSpec,
)
from tests._support.core_helpers import create_step_snapshot


def _ctx(text: str, alignment: tuple[TokenCharAlignment, ...]) -> StepContext:
    doc = DocumentState(text=text, version_id="v1")
    rule = RulePlan(
        rule_id="rid",
        name="replace:test",
        scope=ScopeSpec(kind="whole_doc"),
        target=EditTargetSpec(kind="scope_entire"),
        candidates=(),
        policy=DecisionPolicySpec(min_prob_ratio_to_best=0.0),
        fire=FirePolicySpec(mode="repeat"),
    )
    return StepContext(
        plan=PlanIR(rules=(rule,)),
        rule=rule,
        doc=doc,
        step=create_step_snapshot(text, 0, generated_token_alignment=alignment),
        guard_view=TextView(doc=doc, abs_start=0, abs_end=len(text)),
        edit_view=TextView(doc=doc, abs_start=0, abs_end=len(text)),
    )


def test_normalize_span_prefers_valid_fallback() -> None:
    result = normalize_span((1, 99), "abc", fallback=(1, 3))
    assert result.span == (1, 3)
    assert result.fallback_used


def test_overlap_alignment_diagnostic_is_detailed() -> None:
    text = "a–b"
    alignment = [
        TokenCharAlignment(
            token_index=0,
            token_id=10,
            char_start=0,
            char_end=1,
            piece_text="a",
        ),
        TokenCharAlignment(
            token_index=1,
            token_id=11,
            char_start=1,
            char_end=2,
            piece_text="–",
        ),
        TokenCharAlignment(
            token_index=2,
            token_id=12,
            char_start=1,
            char_end=2,
            piece_text="–",
        ),
        TokenCharAlignment(
            token_index=3,
            token_id=13,
            char_start=2,
            char_end=3,
            piece_text="b",
        ),
    ]

    error = validate_token_alignment_detailed(alignment, text)

    assert error is not None
    assert error.kind == "char_span_overlap"
    assert error.token_index == 2
    assert error.previous_token_index == 1
    assert error.token_id == 12
    assert error.previous_token_id == 11
    assert error.char_start == 1
    assert error.char_end == 2
    assert error.previous_char_start == 1
    assert error.previous_char_end == 2
    assert error.text_slice == "–"
    assert error.previous_text_slice == "–"
    assert error.doc_len == 3
    assert error.overlap_kind == "identical_span"
    compact = error.compact()
    for expected in (
        "char_span_overlap",
        "token_index=2",
        "token_id=12",
        "previous_token_index=1",
        "previous_token_id=11",
        "span=(1, 2)",
        "previous_span=(1, 2)",
        "overlap_kind=identical_span",
    ):
        assert expected in compact


def test_token_index_non_monotonic_diagnostic_kind() -> None:
    text = "ab"
    alignment = [
        TokenCharAlignment(
            token_index=1,
            token_id=10,
            char_start=0,
            char_end=1,
            piece_text="a",
        ),
        TokenCharAlignment(
            token_index=1,
            token_id=11,
            char_start=1,
            char_end=2,
            piece_text="b",
        ),
    ]

    error = validate_token_alignment_detailed(alignment, text)

    assert error is not None
    assert error.kind == "token_index_not_monotonic"
    assert error.token_index == 1
    assert error.previous_token_index == 1
    assert "token_index_not_monotonic" in error.compact()


def test_invalid_incremental_alignment_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    text = "- No obstruction, normal"
    alignment = (
        TokenCharAlignment(0, 0, 17, text[:17]),
        TokenCharAlignment(1, 17, len(text) + 1, " normal"),
    )
    with caplog.at_level(logging.WARNING):
        result = _snap_span_to_token_boundaries(
            ctx=_ctx(text, alignment), span=(17, len(text)), op=PatchOp.REPLACE
        )
    assert result is not None
    assert is_valid_span(result, text)
    assert result[1] <= len(text)
    assert "invalid_incremental_snap_fallback_tokenizer" in caplog.text


def test_unicode_medical_text_boundaries_and_snapping() -> None:
    text = (
        "Sudden onset (≤72 h) → likely SSNHL. "
        "No obstruction, normal tympanic membranes. "
        "Air conduction > bone conduction. -"
    )
    for start in (
        -5,
        0,
        text.index("≤72 h"),
        text.index("→ likely"),
        text.index("normal tympanic membranes"),
        len(text) + 9,
    ):
        for limit in (-3, 0, len(text) // 2, len(text), len(text) + 12):
            assert (
                0
                <= text_bounds.find_sentence_end(text, start, limit)
                <= len(text)
            )
            assert (
                0
                <= text_bounds.find_clause_end(text, start, limit)
                <= len(text)
            )
            assert (
                0
                <= text_bounds.find_clause_start(text, start, limit)
                <= len(text)
            )


class DriftTokenizer:
    bos_token_id: int | None = None
    pad_token_id: int | None = None
    eos_token_id: int | None = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return [1, 2, 3]

    def decode(
        self, ids: list[int], *, skip_special_tokens: bool = True
    ) -> str:
        del skip_special_tokens
        return {1: "A", 2: "<=", 3: "B"}[ids[0]]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        del text, add_special_tokens, return_offsets_mapping
        return {
            "input_ids": [1, 2, 3],
            "offset_mapping": [(0, 1), (1, 2), (2, 3)],
        }


def test_retokenized_assistant_uses_offsets_for_decode_drift() -> None:
    assistant = RetokenizedAssistant(cast(Any, DriftTokenizer()), "A≤B")
    assert [a.piece_text for a in assistant.alignment] == ["A", "≤", "B"]
    assert assistant.alignment[-1].char_end == 3


class BadFallbackTokenizer(DriftTokenizer):
    def __call__(
        self, *args: object, **kwargs: object
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        del args, kwargs
        raise TypeError("no offsets")


class OverlapOffsetTokenizer(DriftTokenizer):
    def __call__(
        self, *args: object, **kwargs: object
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        del args, kwargs
        return {
            "input_ids": [10, 11, 12, 13],
            "offset_mapping": [(0, 1), (1, 2), (1, 2), (2, 3)],
        }


class OutOfBoundsOffsetTokenizer(DriftTokenizer):
    def __call__(
        self, *args: object, **kwargs: object
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        del args, kwargs
        return {
            "input_ids": [1],
            "offset_mapping": [(0, 453)],
        }


class PrefixEncodeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]


def test_retokenized_assistant_drops_invalid_per_token_fallback() -> None:
    assistant = RetokenizedAssistant(cast(Any, BadFallbackTokenizer()), "A≤B")
    assert assistant.token_ids
    assert assistant.alignment == []


def test_overlap_alignment_fallback_logs_debug_not_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        assistant = RetokenizedAssistant(
            cast(Any, OverlapOffsetTokenizer()), "a–b"
        )
    assert assistant.token_ids
    assert not any(
        record.levelno >= logging.WARNING for record in caplog.records
    )
    assert any(
        "char_span_overlap" in record.message for record in caplog.records
    )
    assert any(
        "overlap_kind=identical_span" in record.message
        for record in caplog.records
    )


def test_optional_unavailable_alignment_is_not_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        assistant = RetokenizedAssistant(
            cast(Any, BadFallbackTokenizer()), "A≤B"
        )
    assert assistant.token_ids
    assert assistant.alignment == []
    assert not any(
        record.levelno >= logging.WARNING for record in caplog.records
    )
    assert "invalid_generated_alignment_unavailable" in caplog.text


def test_out_of_bounds_generated_alignment_still_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    text = "x" * 452
    with caplog.at_level(logging.WARNING):
        assistant = RetokenizedAssistant(
            cast(Any, OutOfBoundsOffsetTokenizer()), text
        )
    assert assistant.token_ids
    assert any(record.levelno >= logging.WARNING for record in caplog.records)
    assert "char_end_out_of_bounds" in caplog.text
    assert "doc_len=452" in caplog.text
    assert "char_end=453" in caplog.text


def test_probe_prefix_out_of_bounds_alignment_still_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    text = "x" * 452
    alignment = (
        TokenCharAlignment(
            token_index=0,
            token_id=99,
            char_start=0,
            char_end=453,
            piece_text="",
        ),
    )

    with caplog.at_level(logging.WARNING):
        prefix_ids, used_alignment = _build_probe_prefix_ids(
            tok=cast(Any, PrefixEncodeTokenizer()),
            prompt_ids=[1, 2],
            doc_text=text,
            abs_start=3,
            generated_ids=[99],
            generated_token_alignment=alignment,
        )

    assert prefix_ids == [1, 2, *[ord("x")] * 3]
    assert not used_alignment
    assert any(record.levelno >= logging.WARNING for record in caplog.records)
    assert "invalid_generated_alignment_fallback_reencode" in caplog.text
    assert "char_end_out_of_bounds" in caplog.text
    assert "doc_len=452" in caplog.text
    assert "char_end=453" in caplog.text


def test_probe_prefix_overlap_alignment_logs_debug_not_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    text = "a–b"
    alignment = (
        TokenCharAlignment(0, 0, 1, "a", token_id=10),
        TokenCharAlignment(1, 1, 2, "–", token_id=11),
        TokenCharAlignment(2, 1, 2, "–", token_id=12),
        TokenCharAlignment(3, 2, 3, "b", token_id=13),
    )

    with caplog.at_level(logging.DEBUG):
        prefix_ids, used_alignment = _build_probe_prefix_ids(
            tok=cast(Any, PrefixEncodeTokenizer()),
            prompt_ids=[1, 2],
            doc_text=text,
            abs_start=2,
            generated_ids=[10, 11, 12, 13],
            generated_token_alignment=alignment,
        )

    assert prefix_ids == [1, 2, ord("a"), ord("–")]
    assert not used_alignment
    assert not any(
        record.levelno >= logging.WARNING for record in caplog.records
    )
    assert "invalid_generated_alignment_fallback_reencode" in caplog.text
    assert "char_span_overlap" in caplog.text
    assert "overlap_kind=identical_span" in caplog.text


def test_scoring_filters_invalid_proposal() -> None:
    text = "abc"
    ctx = _ctx(text, tuple())
    proposal = PatchProposal(
        op=PatchOp.REPLACE,
        span_abs=(0, len(text) + 99),
        payload="x",
        base_version_id=ctx.doc.version_id,
        rule_id="rid",
        reason="test",
    )
    sink = RecordingRuntimeEventSink()
    assert (
        _editable_proposals(
            ProposalsReady(ctx=ctx, proposals=(proposal,)), event_sink=sink
        )
        == []
    )
    skipped = [
        event for event in sink.events if isinstance(event, PatchSkipped)
    ]
    assert skipped
    assert skipped[0].reason == "invalid_span_dropped"
    assert skipped[0].rule_name == ctx.rule.name
    assert skipped[0].doc_len == len(text)
    assert skipped[0].original_span == proposal.span_abs
    assert skipped[0].span_abs == proposal.span_abs
    assert skipped[0].stage == "scoring"


def test_patcher_remains_strict_with_context() -> None:
    doc = DocumentState("abc", version_id="v1")
    proposal = PatchProposal(
        op=PatchOp.REPLACE,
        span_abs=(0, 4),
        payload="x",
        base_version_id=doc.version_id,
        rule_id="rid",
        reason="test",
    )
    with pytest.raises(ValueError) as exc:
        patcher.apply_patch(doc, proposal)
    msg = str(exc.value)
    assert "doc_len" in msg and "span_abs" in msg and "rule_id" in msg


def test_b4563fa9_regression_rejects_bad_incremental_snap() -> None:
    text = (
        "The Weber and Rinne results should be analyzed first.\n"
        "- Weber lateralizes to the left ear → the left ear is the better "
        "hearing ear.\n"
        "- Rinne is positive on the right ear → bone conduction is greater "
        "than air conduction on the right side. -\n"
        "These findings are classic for a **sudden sensorineural hearing "
        "loss (SSNHL)**.\n\n"
        "**Key points that support this diagnosis:**\n"
        "- Sudden onset (≤72 h) of unilateral hearing loss.\n"
        "- No obstruction, normal"
    )
    span = (text.index(", normal"), len(text))
    bad = (433, len(text) + 1)
    alignment = (TokenCharAlignment(0, bad[0], bad[1], text[bad[0] :]),)
    final = _snap_span_to_token_boundaries(
        ctx=_ctx(text, alignment), span=span, op=PatchOp.REPLACE
    )
    assert final is not None
    assert is_valid_span(final, text)
    assert final[1] <= len(text)
    assert final != bad


def test_b4563fa9_literal_doc_len_452_regression(
    caplog: pytest.LogCaptureFixture,
) -> None:
    text = "x" * 444 + ", normal"
    assert len(text) == 452
    span = (444, 452)
    bad = (433, 453)
    alignment = (TokenCharAlignment(0, bad[0], bad[1], text[bad[0] :]),)
    with caplog.at_level(logging.WARNING):
        final = _snap_span_to_token_boundaries(
            ctx=_ctx(text, alignment), span=span, op=PatchOp.REPLACE
        )
    assert final is None or is_valid_span(final, text)
    assert final != bad
    assert "invalid_incremental_alignment" in caplog.text


def test_lightweight_unicode_fuzz_span_and_boundaries() -> None:
    chars = (
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        + [
            " ",
            "\n",
            ",",
            ".",
            "(",
            ")",
            ";",
            ":",
            "≤",
            "≥",
            "→",
            "←",
            "-",
            "–",
            "—",
            "•",
            "\u00a0",
            "\u202f",
        ]
        + list("אבגדהוזחטיכלמנסעפצקרשת")
        + list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
        + list("абвгдеёжзийклмнопрстуфхцчшщьыъэюя")
        + ["🙂", "🧠", "\u0301"]
    )
    rng = random.Random(7)
    for _ in range(120):
        text = "".join(rng.choice(chars) for _ in range(rng.randrange(0, 80)))
        span = (
            rng.randrange(-20, len(text) + 20),
            rng.randrange(-20, len(text) + 20),
        )
        result = normalize_span(span, text, mode="fallback_then_clamp")
        assert result.span is None or is_valid_span(result.span, text)
        assert (
            0
            <= text_bounds.find_sentence_end(text, span[0], span[1])
            <= len(text)
        )
        assert (
            0
            <= text_bounds.find_clause_end(text, span[0], span[1])
            <= len(text)
        )
        assert (
            0
            <= text_bounds.find_clause_start(text, span[0], span[1])
            <= len(text)
        )
