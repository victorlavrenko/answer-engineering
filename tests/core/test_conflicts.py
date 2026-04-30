from __future__ import annotations

import pytest

from answer_engineering.engine.orchestration.orchestrator import (
    PlanRunner,
)
from answer_engineering.engine.patching.proposals import PatchProposal
from answer_engineering.engine.runtime.runtime_types import (
    PatchOp,
)
from answer_engineering.inference.decode.session_orchestration import (
    ExecutionSession,
)
from answer_engineering.rules.compile.compiled_rules import (
    CompiledRules,
)
from tests._support.core_helpers import step_test


def _proposal(
    rule_id: str, span: tuple[int, int], logprob: float, *, op: PatchOp
) -> PatchProposal:
    return PatchProposal(
        op=op,
        span_abs=span,
        payload="x",
        base_version_id="v0",
        rule_id=rule_id,
        score=logprob,
        reason="valid edit",
        cached_score_logprob=logprob,
    )


def test_insert_vs_insert_same_position_highest_logprob_wins() -> None:
    runner = PlanRunner(verbose=False)
    accepted, rejected = runner._resolve_overlaps(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        [
            _proposal("r1", (5, 5), -0.3, op=PatchOp.INSERT_AFTER),
            _proposal("r2", (5, 5), -0.1, op=PatchOp.INSERT_AFTER),
        ]
    )
    assert [p.rule_id for p in accepted] == ["r2"]
    assert [p.rule_id for p in rejected] == ["r1"]


def test_insert_inside_replace_conflicts_and_logprob_decides() -> None:
    runner = PlanRunner(verbose=False)
    replace_low = _proposal("replace", (5, 10), -0.5, op=PatchOp.REPLACE)
    insert_high = _proposal("insert", (7, 7), -0.1, op=PatchOp.INSERT_AFTER)
    accepted, rejected = runner._resolve_overlaps([replace_low, insert_high])  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert [p.rule_id for p in accepted] == ["insert"]
    assert [p.rule_id for p in rejected] == ["replace"]


def test_insert_at_replace_boundary_conflicts() -> None:
    runner = PlanRunner(verbose=False)
    replace_high = _proposal("replace", (5, 10), -0.1, op=PatchOp.REPLACE)
    insert_low = _proposal("insert", (10, 10), -0.2, op=PatchOp.INSERT_AFTER)
    accepted, rejected = runner._resolve_overlaps([replace_high, insert_low])  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert [p.rule_id for p in accepted] == ["replace"]
    assert [p.rule_id for p in rejected] == ["insert"]


def test_three_way_mixed_overlap_keeps_only_best() -> None:
    runner = PlanRunner(verbose=False)
    a = _proposal("a", (0, 5), -0.1, op=PatchOp.REPLACE)
    b = _proposal("b", (4, 4), -0.2, op=PatchOp.INSERT_AFTER)
    c = _proposal("c", (4, 8), -0.05, op=PatchOp.REPLACE)
    accepted, rejected = runner._resolve_overlaps([a, b, c])  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert [p.rule_id for p in accepted] == ["c"]
    assert sorted(p.rule_id for p in rejected) == ["a", "b"]


def test_verbose_guard_failed_noop_does_not_print(
    capsys: pytest.CaptureFixture[str],
) -> None:
    md = """
## Avoid (postfix, repeat): conductive

Scope:
* 1 sentence

Prefix (all):
* absent_token

Postfix (any):
* conductive

Fallback:
* fallback text
"""
    engine = ExecutionSession(
        plan=CompiledRules(md).plan, runner=PlanRunner(verbose=True)
    )
    _ = step_test(
        engine, "this suggests conductive hearing loss", token_index=0
    )
    assert capsys.readouterr().out == ""
