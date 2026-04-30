from __future__ import annotations

from answer_engineering.engine.proposal.guards.guard_matching import (
    evaluate_guard,
)
from answer_engineering.engine.proposal.match_tree.match_tree import (
    MatchAndThen,
    MatchAny,
    MatchTerm,
)
from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    TextView,
)
from answer_engineering.rules.compile.plan import (
    GuardSpec,
)


def _view(text: str) -> TextView:
    doc = DocumentState(text=text, version_id="v")
    return TextView(
        doc=doc,
        abs_start=0,
        abs_end=len(text),
    )


def test_evaluate_guard_uses_prompt_text_for_left_side() -> None:
    guard = GuardSpec(
        expression=MatchAndThen(
            MatchTerm("left", marker="prompt_all"),
            MatchAny((MatchTerm("SSNHL", marker="postfix_any"),)),
            marker="prompt_answer_boundary",
        )
    )
    ok, observations, _ = evaluate_guard(
        _view("findings are consistent with SSNHL"),
        guard,
        anchors={},
        prompt_text="The prompt says left ear.",
        casefold=True,
    )
    assert ok is True
    assert any(ob.marker == "prompt_all" and ob.matched for ob in observations)


def test_evaluate_guard_fails_when_prompt_side_missing() -> None:
    guard = GuardSpec(
        expression=MatchAndThen(
            MatchTerm("left", marker="prompt_all"),
            MatchAny((MatchTerm("SSNHL", marker="postfix_any"),)),
            marker="prompt_answer_boundary",
        )
    )
    ok, _observations, _ = evaluate_guard(
        _view("findings are consistent with SSNHL"),
        guard,
        anchors={},
        prompt_text="The prompt says right ear.",
        casefold=True,
    )
    assert ok is False
