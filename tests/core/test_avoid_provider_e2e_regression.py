from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch

from answer_engineering.engine.pipeline.context import StepContext
from answer_engineering.engine.pipeline.events import (
    AvoidProbeCacheExhausted,
    AvoidProbeCandidatePopped,
    AvoidProbeEpisodeStarted,
    ProposalsGenerated,
)
from answer_engineering.engine.proposal.candidates import (
    avoid as avoid_candidates,
)
from answer_engineering.engine.runtime.runtime_types import PatchOp
from answer_engineering.inference.decode.session_orchestration import (
    ExecutionSession,
)
from answer_engineering.inference.probing.runtime.probe_runtime import (
    ProbeRuntime,
)
from answer_engineering.rules.compile.compiled_rules import (
    CompiledRules,
)
from answer_engineering.rules.compile.plan import CandidateSpec
from tests._support.core_helpers import step_test
from tests._support.runtime_harness import configure_runtime_scoring


def _avoid_engine() -> ExecutionSession:
    runtime = ExecutionSession(
        plan=CompiledRules(
            """## Avoid (repeat): alpha

Postfix (any):

- alpha

Fallback:

- fallback
"""
        ).plan
    )
    configure_runtime_scoring(
        runtime, generation_runtime=None, require_model_scoring=False
    )
    return runtime


def _generated_spec(text: str, *, idx: int) -> CandidateSpec:
    return CandidateSpec(
        op=PatchOp.REPLACE,
        text=text,
        kind="generated",
        priority=100 - idx,
        label=f"probe_{idx}",
        candidate_id=f"probe_{idx}",
        logprob=-0.01,
    )


def test_avoid_generated_candidates_are_consumed_one_per_step(
    monkeypatch: MonkeyPatch,
) -> None:
    calls = 0

    def _generated(
        _self: ProbeRuntime, *_args: object, **_kwargs: object
    ) -> list[CandidateSpec]:
        nonlocal calls
        calls += 1
        return [
            _generated_spec("GEN1", idx=1),
            _generated_spec("GEN2", idx=2),
        ]

    monkeypatch.setattr(ProbeRuntime, "generate", _generated)
    engine = _avoid_engine()

    first = step_test(engine, "alpha beta", token_index=0)
    second = step_test(engine, "alpha beta", token_index=1)

    assert first.final_text == "GEN1"
    assert second.final_text == "GEN2"
    assert calls == 1


def test_avoid_same_step_does_not_consume_multiple_generated_candidates(
    monkeypatch: MonkeyPatch,
) -> None:
    def _generated(
        _self: ProbeRuntime, *_args: object, **_kwargs: object
    ) -> list[CandidateSpec]:
        return [
            _generated_spec("GEN1", idx=1),
            _generated_spec("GEN2", idx=2),
        ]

    monkeypatch.setattr(ProbeRuntime, "generate", _generated)
    engine = _avoid_engine()

    first = step_test(engine, "alpha beta", token_index=0)
    second = step_test(engine, "alpha beta", token_index=0)

    assert first.final_text == "GEN1"
    assert second.final_text != "GEN2"


def test_avoid_already_satisfied_skips_probe_generation(
    monkeypatch: MonkeyPatch,
) -> None:
    calls = 0

    def _generated(
        _self: ProbeRuntime, *_args: object, **_kwargs: object
    ) -> list[CandidateSpec]:
        nonlocal calls
        calls += 1
        return [_generated_spec("GEN1", idx=1)]

    monkeypatch.setattr(ProbeRuntime, "generate", _generated)
    engine = _avoid_engine()

    out = step_test(engine, "alpha fallback", token_index=0)

    assert calls == 0
    assert out.final_text == "alpha fallback"


def test_avoid_scope_floor_activates_only_after_repeated_empty_stream(
    monkeypatch: MonkeyPatch,
) -> None:
    floor_starts: list[int | None] = []

    def _none_generated(
        _self: ProbeRuntime, ctx: StepContext, **_kwargs: object
    ) -> list[CandidateSpec]:
        floor_starts.append(ctx.avoid_edit_floor_abs_start)
        return list()

    def _always_allow_floor(_ctx: StepContext) -> bool:
        return True

    def _forced_sentence_start(*, text: str, span_abs: tuple[int, int]) -> int:
        del text, span_abs  # Unused in this forced-floor test.
        return 2

    monkeypatch.setattr(ProbeRuntime, "generate", _none_generated)
    monkeypatch.setattr(
        avoid_candidates,
        "_allow_avoid_scope_floor",
        _always_allow_floor,
    )
    monkeypatch.setattr(
        avoid_candidates.scope,
        "sentence_floor_start",
        _forced_sentence_start,
    )
    engine = _avoid_engine()

    first = step_test(engine, "alpha. beta", token_index=0)
    second = step_test(engine, "alpha. beta", token_index=1)
    third = step_test(engine, "alpha. beta", token_index=2)

    assert first.final_text == "fallback"
    assert "fallback" in second.final_text
    assert "fallback" in third.final_text
    assert floor_starts[0] is None
    assert floor_starts[1] is not None


def test_avoid_rule_does_not_surface_duplicate_fallback_candidates(
    monkeypatch: MonkeyPatch,
) -> None:
    def _none_generated(
        _self: ProbeRuntime, *_args: object, **_kwargs: object
    ) -> list[CandidateSpec]:
        return list()

    monkeypatch.setattr(ProbeRuntime, "generate", _none_generated)
    engine = _avoid_engine()

    first = step_test(engine, "alpha beta", token_index=0)
    second = step_test(engine, "alpha beta", token_index=1)

    assert first.final_text == "fallback"
    assert second.final_text == "fallback"
    first_proposals = [
        event for event in first.events if isinstance(event, ProposalsGenerated)
    ]
    second_proposals = [
        event
        for event in second.events
        if isinstance(event, ProposalsGenerated)
    ]
    assert len(first_proposals) == 1
    assert len(second_proposals) == 1
    assert first_proposals[0].proposals_count == 1
    assert second_proposals[0].proposals_count == 1
    assert first_proposals[0].fallback_count == 1
    assert second_proposals[0].fallback_count == 1


def test_avoid_exhaustion_event_emitted_once_per_stream(
    monkeypatch: MonkeyPatch,
) -> None:
    def _none_generated(
        _self: ProbeRuntime, *_args: object, **_kwargs: object
    ) -> list[CandidateSpec]:
        return list()

    monkeypatch.setattr(ProbeRuntime, "generate", _none_generated)
    engine = _avoid_engine()

    first = step_test(engine, "alpha beta", token_index=0)
    second = step_test(engine, "alpha beta", token_index=1)

    events = [*first.events, *second.events]
    exhausted = [
        event for event in events if isinstance(event, AvoidProbeCacheExhausted)
    ]
    assert len(exhausted) == 1


def test_avoid_event_lifecycle_generated_then_exhausted_stream(
    monkeypatch: MonkeyPatch,
) -> None:
    calls = 0

    def _generated(
        _self: ProbeRuntime, *_args: object, **_kwargs: object
    ) -> list[CandidateSpec]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [_generated_spec("GEN1", idx=1)]
        return list()

    monkeypatch.setattr(ProbeRuntime, "generate", _generated)
    engine = _avoid_engine()

    first = step_test(engine, "alpha beta", token_index=0)
    second = step_test(engine, "alpha beta", token_index=1)
    third = step_test(engine, "alpha beta", token_index=2)
    events = [*first.events, *second.events, *third.events]

    episode = [
        event for event in events if isinstance(event, AvoidProbeEpisodeStarted)
    ]
    popped = [
        event
        for event in events
        if isinstance(event, AvoidProbeCandidatePopped)
    ]
    exhausted = [
        event for event in events if isinstance(event, AvoidProbeCacheExhausted)
    ]
    assert len(episode) == 1
    assert len(popped) == 1
    assert len(exhausted) == 1


def test_avoid_empty_probe_preserves_final_winner_behavior(
    monkeypatch: MonkeyPatch,
) -> None:
    def _none_generated(
        _self: ProbeRuntime, *_args: object, **_kwargs: object
    ) -> list[CandidateSpec]:
        return list()

    monkeypatch.setattr(ProbeRuntime, "generate", _none_generated)
    engine = _avoid_engine()

    out = step_test(engine, "alpha beta", token_index=0)

    assert out.final_text == "fallback"
