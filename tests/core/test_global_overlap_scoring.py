from __future__ import annotations

from answer_engineering.config.patch_score_policy import PatchScorePolicy
from answer_engineering.engine.patching.proposals import PatchProposal
from answer_engineering.engine.runtime.runtime_types import (
    PatchOp,
)
from answer_engineering.engine.scoring.base import ScoredProposal
from answer_engineering.engine.scoring.logits.scorer import LogitsScorer
from answer_engineering.engine.selection.conflict_resolver import (
    ConflictResolverSelector,
)
from answer_engineering.inference.decode.session_orchestration import (
    ExecutionSession,
)
from answer_engineering.rules.compile.compiled_rules import (
    CompiledRules,
)
from tests._support.runtime_harness import configure_runtime_scoring


def _scored(
    rule_id: str, candidate_index: int, span: tuple[int, int], score: float
) -> ScoredProposal:
    return ScoredProposal(
        proposal=PatchProposal(
            op=PatchOp.REPLACE,
            span_abs=span,
            payload="x",
            base_version_id="v0",
            rule_id=rule_id,
            score=score,
            reason="candidate",
            candidate_index=candidate_index,
            cached_score_logprob=score,
        ),
        score=score,
    )


def test_global_greedy_overlap_keeps_global_best() -> None:
    resolver = ConflictResolverSelector()
    scored = [
        _scored("r1", 0, (0, 5), -0.2),
        _scored("r1", 1, (0, 5), -0.1),
        _scored("r1", 2, (0, 5), -0.3),
        _scored("r2", 0, (3, 7), -0.05),
        _scored("r2", 1, (3, 7), -0.4),
        _scored("r2", 2, (3, 7), -0.6),
        _scored("r3", 0, (4, 6), -0.15),
        _scored("r3", 1, (4, 6), -0.25),
        _scored("r3", 2, (4, 6), -0.35),
    ]

    accepted, rejected = resolver.resolve(scored)
    assert len(accepted) == 1
    assert accepted[0].proposal.rule_id == "r2"
    assert accepted[0].proposal.candidate_index == 0
    assert len(rejected) == 8


def test_global_greedy_accepts_disjoint_spans() -> None:
    resolver = ConflictResolverSelector()
    scored = [
        _scored("r1", 0, (0, 2), -0.3),
        _scored("r2", 0, (3, 5), -0.1),
        _scored("r3", 0, (6, 8), -0.2),
    ]
    accepted, rejected = resolver.resolve(scored)
    assert [item.proposal.rule_id for item in accepted] == ["r2", "r3", "r1"]
    assert rejected == []


def test_patch_score_policy_is_wired_globally() -> None:
    engine = ExecutionSession(
        plan=CompiledRules("## Replace: x\n\nWith:\n\n- y").plan
    )
    policy = PatchScorePolicy(
        n_left_ctx=1, n_right_ctx=5, w_left_ctx=0.1, w_right_ctx=0.8
    )

    configure_runtime_scoring(
        engine,
        generation_runtime=None,
        require_model_scoring=False,
        patch_score_policy=policy,
    )

    runner = engine.runner
    assert runner.patch_score_policy == policy
    assert isinstance(runner.scorer_component, LogitsScorer)
    runner._configure_scorer()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert runner.scorer_component.policy == policy
