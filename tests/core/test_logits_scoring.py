from __future__ import annotations

import math
from dataclasses import replace

import pytest
from _pytest.monkeypatch import MonkeyPatch

from answer_engineering.engine.orchestration.orchestrator import (
    PlanRunner,
)
from answer_engineering.engine.patching.patcher import (
    apply_patch,
)
from answer_engineering.engine.patching.proposals import PatchProposal
from answer_engineering.engine.pipeline.context import (
    StepContext,
)
from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    PatchOp,
    TextView,
)
from answer_engineering.engine.scoring.base import ScoreResult
from answer_engineering.engine.selection.rule_winner import (
    RuleWinnerDecision,
)
from answer_engineering.rules.compile.plan import (
    DecisionPolicySpec,
    PlanIR,
    RulePlan,
)
from tests._support.core_helpers import create_step_snapshot
from tests.core._scoring_stubs import GenerationRuntimeStub


def _proposal(rule: str, span: tuple[int, int], payload: str) -> PatchProposal:
    return PatchProposal(
        op=PatchOp.REPLACE,
        span_abs=span,
        payload=payload,
        payload_norm=payload,
        base_version_id="v1",
        rule_id=rule,
        score=0.0,
        reason="test",
    )


def test_rule_winner_decision_uses_best_vs_second_best_gap() -> None:
    decision = RuleWinnerDecision(
        [0.0, -0.1, -4.0], min_prob_ratio_to_best=1.05
    )
    assert decision.winner_index == 0
    assert decision.winner_ratio_to_runner_up > 1.05
    assert math.isclose(decision.ratios_to_best[0], 1.0)

    decision2 = RuleWinnerDecision([0.0, -0.1], min_prob_ratio_to_best=1.2)
    assert decision2.winner_index is None
    assert decision2.winner_ratio_to_runner_up < 1.2


def test_rule_winner_decision_returns_none_for_all_negative_infinity() -> None:
    decision = RuleWinnerDecision(
        [float("-inf"), float("-inf")], min_prob_ratio_to_best=None
    )
    assert decision.winner_index is None
    assert decision.ratios_to_best == [0.0, 0.0]
    assert decision.winner_ratio_to_runner_up == 0.0


def test_rule_winner_decision_handles_very_large_score_gaps() -> None:
    decision = RuleWinnerDecision([0.0, -10000.0], min_prob_ratio_to_best=None)
    assert decision.winner_index == 0
    assert decision.ratios_to_best[0] == 1.0
    assert decision.winner_ratio_to_runner_up == float("inf")


def test_rule_winner_caches_normalized_payload_and_skips_re_normalize() -> None:
    base = DocumentState(text="Hello,world", version_id="v1")
    raw = PatchProposal(
        op=PatchOp.INSERT_AFTER,
        span_abs=(5, 5),
        payload="doctor",
        payload_norm=" doctor ",
        base_version_id="v1",
        rule_id="r1",
        score=0.0,
        reason="test",
    )
    runner = PlanRunner(
        verbose=False,
        runtime=GenerationRuntimeStub.loaded_runtime(),
    )
    rule = RulePlan(
        rule_id="r1",
        policy=DecisionPolicySpec(min_prob_ratio_to_best=None),
    )

    ctx, scored = _score_proposals_for_rule(
        runner,
        rule=rule,
        base_doc=base,
        proposals=[raw],
    )

    decision = RuleWinnerDecision(
        [item.score for item in scored.scored],
        min_prob_ratio_to_best=ctx.rule.policy.min_prob_ratio_to_best,
    )
    winner = (
        None
        if decision.winner_index is None
        else scored.scored[decision.winner_index].proposal
    )

    assert winner is not None
    assert winner.payload_norm == " doctor "

    mutated = apply_patch(base, winner).text
    assert mutated == "Hello doctor ,world"

    forced = PatchProposal(
        op=PatchOp.INSERT_AFTER,
        span_abs=(5, 5),
        payload="doctor",
        payload_norm="doctor",
        base_version_id="v1",
        rule_id="r1",
        score=0.0,
        reason="test",
    )
    # if apply_patch re-normalized this payload it would become " doctor "
    assert apply_patch(base, forced).text == "Hellodoctor,world"


def _score_proposals_for_rule(
    runner: PlanRunner,
    *,
    rule: RulePlan,
    base_doc: DocumentState,
    proposals: list[PatchProposal],
) -> tuple[StepContext, ScoreResult]:
    step = create_step_snapshot(
        snapshot_text=base_doc.text,
        token_index=0,
    )

    ctx = StepContext(
        plan=PlanIR(rules=(rule,), plan_version="runner"),
        rule=rule,
        doc=base_doc,
        step=step,
        guard_view=TextView(
            doc=base_doc,
            abs_start=0,
            abs_end=len(base_doc.text),
        ),
        edit_view=TextView(
            doc=base_doc,
            abs_start=0,
            abs_end=len(base_doc.text),
        ),
        trajectory_debug=runner.trajectory_debug,
        event_sink=runner.event_sink,
    )

    runner._configure_scorer()  # pyright: ignore[reportPrivateUsage]

    normalized = [
        runner.proposal_engine.freeze_normalized_proposal(ctx.doc, proposal)
        for proposal in proposals
        if proposal.op != PatchOp.NOOP
    ]

    scored = runner.scorer_component.score(ctx, normalized)
    return ctx, scored


def test_choose_rule_winner_respects_min_prob_ratio_to_best() -> None:
    base = DocumentState(text="abcdef", version_id="v1")
    runner = PlanRunner(
        verbose=False,
        runtime=GenerationRuntimeStub.loaded_runtime(),
    )
    rule = RulePlan(
        rule_id="r1",
        policy=DecisionPolicySpec(min_prob_ratio_to_best=100.0),
    )

    ctx, scored = _score_proposals_for_rule(
        runner,
        rule=rule,
        base_doc=base,
        proposals=[
            _proposal("r1", (2, 3), "x"),
            _proposal("r1", (2, 3), "y"),
        ],
    )

    decision = RuleWinnerDecision(
        [item.score for item in scored.scored],
        min_prob_ratio_to_best=ctx.rule.policy.min_prob_ratio_to_best,
    )
    winner = (
        None
        if decision.winner_index is None
        else scored.scored[decision.winner_index].proposal
    )

    assert winner is None


def test_overlap_resolution_chooses_highest_logprob_three_way(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = PlanRunner(verbose=True)
    a = _proposal("rA", (0, 5), "x")
    b = _proposal("rB", (2, 6), "y")
    c = _proposal("rC", (4, 8), "z")
    proposals = [
        replace(a, cached_score_logprob=-0.1),
        replace(b, cached_score_logprob=-0.3),
        replace(c, cached_score_logprob=-0.2),
    ]
    accepted, rejected = runner._resolve_overlaps(proposals)  # pyright: ignore[reportPrivateUsage]
    assert [p.rule_id for p in accepted] == ["rA"]
    assert sorted(p.rule_id for p in rejected) == ["rB", "rC"]
    out = capsys.readouterr().out
    assert "winner=rA" in out


def test_require_model_scoring_requires_runtime() -> None:
    runner = PlanRunner(verbose=False, require_model_scoring=True)

    base = DocumentState(text="abc", version_id="v1")
    rule = RulePlan(
        rule_id="r1",
        policy=DecisionPolicySpec(min_prob_ratio_to_best=None),
    )

    step = create_step_snapshot(
        snapshot_text=base.text,
        token_index=0,
    )

    ctx = StepContext(
        plan=PlanIR(rules=(rule,), plan_version="runner"),
        rule=rule,
        doc=base,
        step=step,
        guard_view=TextView(
            doc=base,
            abs_start=0,
            abs_end=len(base.text),
        ),
        edit_view=TextView(
            doc=base,
            abs_start=0,
            abs_end=len(base.text),
        ),
        trajectory_debug=runner.trajectory_debug,
        event_sink=runner.event_sink,
    )

    proposals = [
        _proposal("r1", (0, 1), "x"),
    ]

    normalized = [
        runner.proposal_engine.freeze_normalized_proposal(ctx.doc, proposal)
        for proposal in proposals
        if proposal.op != PatchOp.NOOP
    ]

    with pytest.raises(ValueError, match="runtime required"):
        runner._configure_scorer()  # pyright: ignore[reportPrivateUsage]
        runner.scorer_component.score(ctx, normalized)


def test_avoid_fallback_candidates_use_same_batch_logprob_path(
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[int] = []

    def _fake_batch(
        *_: object, continuation_ids_list: list[list[int]], **__: object
    ) -> list[object]:
        calls.append(len(continuation_ids_list))
        return [
            type("R", (), {"logprob_sum": -float(len(ids))})()
            for ids in continuation_ids_list
        ]

    monkeypatch.setattr(
        "answer_engineering.engine.scoring.logits.scorer.RuntimeLogprobScorer.score_continuations_batch",
        _fake_batch,
    )

    base = DocumentState(text="abcdef", version_id="v1")
    rule = RulePlan(
        rule_id="r_avoid",
        policy=DecisionPolicySpec(min_prob_ratio_to_best=None),
        name="avoid:test",
    )
    proposals = [
        _proposal("r_avoid", (2, 3), "x"),
        _proposal("r_avoid", (2, 3), "y"),
    ]

    runner = PlanRunner(
        verbose=False,
        runtime=GenerationRuntimeStub.loaded_runtime(),
    )

    ctx, scored = _score_proposals_for_rule(
        runner,
        rule=rule,
        base_doc=base,
        proposals=proposals,
    )

    decision = RuleWinnerDecision(
        [item.score for item in scored.scored],
        min_prob_ratio_to_best=ctx.rule.policy.min_prob_ratio_to_best,
    )
    winner = (
        None
        if decision.winner_index is None
        else scored.scored[decision.winner_index].proposal
    )

    assert winner is not None
    assert calls
    assert max(calls) >= 2


def test_scoring_includes_replacement_segment_even_with_empty_prefix(
    monkeypatch: MonkeyPatch,
) -> None:
    captured_prefixes: list[list[int]] = []

    def _fake_batch(
        *_: object,
        prefix_ids: list[int],
        continuation_ids_list: list[list[int]],
        **__: object,
    ) -> list[object]:
        captured_prefixes.append(list(prefix_ids))
        return [
            type("R", (), {"logprob_sum": -1.0})()
            for _ in continuation_ids_list
        ]

    monkeypatch.setattr(
        "answer_engineering.engine.scoring.logits.scorer.RuntimeLogprobScorer.score_continuations_batch",
        _fake_batch,
    )

    base = DocumentState(text="abcdef", version_id="v1")
    rule = RulePlan(
        rule_id="r1",
        policy=DecisionPolicySpec(min_prob_ratio_to_best=None),
    )
    runner = PlanRunner(
        verbose=False,
        runtime=GenerationRuntimeStub.loaded_runtime(),
    )

    ctx, scored = _score_proposals_for_rule(
        runner,
        rule=rule,
        base_doc=base,
        proposals=[_proposal("r1", (0, 1), "x")],
    )

    decision = RuleWinnerDecision(
        [item.score for item in scored.scored],
        min_prob_ratio_to_best=ctx.rule.policy.min_prob_ratio_to_best,
    )
    winner = (
        None
        if decision.winner_index is None
        else scored.scored[decision.winner_index].proposal
    )

    assert winner is not None
    assert [] in captured_prefixes


def test_continuation_scoring_is_batched(monkeypatch: MonkeyPatch) -> None:
    batch_calls: list[int] = []

    def _fake_batch(
        *_: object,
        prefix_ids: list[int],
        continuation_ids_list: list[list[int]],
        **__: object,
    ) -> list[object]:
        del prefix_ids
        batch_calls.append(len(continuation_ids_list))
        return [
            type("R", (), {"logprob_sum": -1.0})()
            for _ in continuation_ids_list
        ]

    monkeypatch.setattr(
        "answer_engineering.engine.scoring.logits.scorer.RuntimeLogprobScorer.score_continuations_batch",
        _fake_batch,
    )

    base = DocumentState(text="abcdef", version_id="v1")
    rule = RulePlan(
        rule_id="r1",
        policy=DecisionPolicySpec(min_prob_ratio_to_best=None),
    )
    runner = PlanRunner(
        verbose=False,
        runtime=GenerationRuntimeStub.loaded_runtime(),
    )

    from answer_engineering.config.patch_score_policy import PatchScorePolicy

    runner.patch_score_policy = PatchScorePolicy(continuation_tokens=2)

    ctx, scored = _score_proposals_for_rule(
        runner,
        rule=rule,
        base_doc=base,
        proposals=[
            _proposal("r1", (2, 3), "x"),
            _proposal("r1", (2, 3), "y"),
        ],
    )

    decision = RuleWinnerDecision(
        [item.score for item in scored.scored],
        min_prob_ratio_to_best=ctx.rule.policy.min_prob_ratio_to_best,
    )
    winner = (
        None
        if decision.winner_index is None
        else scored.scored[decision.winner_index].proposal
    )

    assert winner is not None
    assert batch_calls
    assert max(batch_calls) >= 2
