from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch

from answer_engineering.engine.pipeline.context import (
    StepContext,
)
from answer_engineering.engine.pipeline.events import (
    AvoidProbeCacheExhausted,
)
from answer_engineering.engine.proposal.candidates.avoid import (
    AvoidCandidatesProvider,
)
from answer_engineering.engine.proposal.candidates.base import (
    CandidateRequest,
)
from answer_engineering.engine.proposal.proposal_logic import (
    GenerationPrecheck,
)
from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    PatchOp,
    TextView,
)
from answer_engineering.engine.telemetry.events.event_sink import (
    RecordingRuntimeEventSink,
    RuntimeEventSink,
)
from answer_engineering.inference.probing.runtime.probe_runtime import (
    ProbeRuntime,
)
from answer_engineering.rules.compile.plan import (
    CandidateSpec,
    DecisionPolicySpec,
    EditTargetSpec,
    FirePolicySpec,
    PlanIR,
    RulePlan,
    ScopeSpec,
)
from tests._support.core_helpers import create_step_snapshot


def _avoid_ctx(
    text: str,
    *,
    token_index: int,
    fallback_text: str = "fallback",
    validation_for_all: bool = False,
    event_sink: RuntimeEventSink | None = None,
) -> StepContext:
    doc = DocumentState(text=text)
    rule = RulePlan(
        rule_id="avoid-r1",
        name="avoid:test",
        scope=ScopeSpec(kind="whole_doc"),
        target=EditTargetSpec(kind="scope_entire"),
        candidates=(
            CandidateSpec(
                op=PatchOp.REPLACE,
                text=fallback_text,
                kind="static",
                priority=5,
                label="fallback",
            ),
        ),
        policy=DecisionPolicySpec(validation_for_all=validation_for_all),
        fire=FirePolicySpec(mode="repeat"),
    )
    view = TextView(
        doc=doc,
        abs_start=0,
        abs_end=len(doc.text),
    )
    return StepContext(
        plan=PlanIR(rules=(rule,)),
        rule=rule,
        doc=doc,
        step=create_step_snapshot(
            snapshot_text=doc.text, token_index=token_index
        ),
        guard_view=view,
        edit_view=view,
        event_sink=event_sink or RecordingRuntimeEventSink(),
    )


def test_avoid_provider_provide_consumes_generated_candidates_sequentially(
    monkeypatch: MonkeyPatch,
) -> None:
    provider = AvoidCandidatesProvider(runtime=None, trajectory_debug=False)
    generate_calls = 0

    def _generated(
        _self: ProbeRuntime, *_args: object, **_kwargs: object
    ) -> list[CandidateSpec]:
        nonlocal generate_calls
        generate_calls += 1
        return [
            CandidateSpec(
                op=PatchOp.REPLACE,
                text="g1",
                kind="generated",
                priority=100,
                label="probe_1",
                candidate_id="probe_1",
            ),
            CandidateSpec(
                op=PatchOp.REPLACE,
                text="g2",
                kind="generated",
                priority=99,
                label="probe_2",
                candidate_id="probe_2",
            ),
            CandidateSpec(
                op=PatchOp.REPLACE,
                text="g3",
                kind="generated",
                priority=98,
                label="probe_3",
                candidate_id="probe_3",
            ),
        ]

    monkeypatch.setattr(ProbeRuntime, "generate", _generated)
    observed_generated: list[str] = []
    for token_index in range(4):
        ctx = _avoid_ctx(
            "First sentence. Second sentence.", token_index=token_index
        )
        provision = provider.provide(
            CandidateRequest(ctx=ctx, precheck=GenerationPrecheck(ctx))
        )
        generated = [
            c.text for c in provision.candidates if c.kind == "generated"
        ]
        fallback = [
            c.text for c in provision.candidates if c.kind == "fallback"
        ]
        observed_generated.extend(generated)
        assert fallback == ["fallback"]
    assert observed_generated == ["g1", "g2", "g3"]
    assert generate_calls == 1


def test_avoid_provider_provide_preserves_flooring_timing_atomically(
    monkeypatch: MonkeyPatch,
) -> None:
    provider = AvoidCandidatesProvider(runtime=None, trajectory_debug=False)

    def _none_generated(
        _self: ProbeRuntime, *_args: object, **_kwargs: object
    ) -> list[CandidateSpec]:
        return list()

    monkeypatch.setattr(ProbeRuntime, "generate", _none_generated)
    first_ctx = _avoid_ctx("Clause one. Clause two tail", token_index=0)
    first = provider.provide(
        CandidateRequest(ctx=first_ctx, precheck=GenerationPrecheck(first_ctx))
    )
    assert first.ctx.avoid_edit_floor_abs_start is None
    assert [c.kind for c in first.candidates] == ["fallback"]

    second_ctx = _avoid_ctx("Clause one. Clause two tail", token_index=1)
    second = provider.provide(
        CandidateRequest(
            ctx=second_ctx, precheck=GenerationPrecheck(second_ctx)
        )
    )
    assert second.ctx.avoid_edit_floor_abs_start is not None
    assert [c.kind for c in second.candidates] == ["fallback"]


def test_avoid_provider_provide_same_step_does_not_overconsume(
    monkeypatch: MonkeyPatch,
) -> None:
    provider = AvoidCandidatesProvider(runtime=None, trajectory_debug=False)

    def _generated(
        _self: ProbeRuntime, *_args: object, **_kwargs: object
    ) -> list[CandidateSpec]:
        return [
            CandidateSpec(
                op=PatchOp.REPLACE,
                text="g1",
                kind="generated",
                priority=100,
                label="probe_1",
                candidate_id="probe_1",
            ),
            CandidateSpec(
                op=PatchOp.REPLACE,
                text="g2",
                kind="generated",
                priority=99,
                label="probe_2",
                candidate_id="probe_2",
            ),
        ]

    monkeypatch.setattr(ProbeRuntime, "generate", _generated)
    ctx_same_step = _avoid_ctx(
        "First sentence. Second sentence.", token_index=0
    )
    first = provider.provide(
        CandidateRequest(
            ctx=ctx_same_step,
            precheck=GenerationPrecheck(ctx_same_step),
        )
    )
    second = provider.provide(
        CandidateRequest(
            ctx=ctx_same_step,
            precheck=GenerationPrecheck(ctx_same_step),
        )
    )
    next_step_ctx = _avoid_ctx(
        "First sentence. Second sentence.",
        token_index=1,
    )
    third = provider.provide(
        CandidateRequest(
            ctx=next_step_ctx,
            precheck=GenerationPrecheck(next_step_ctx),
        )
    )
    assert [c.text for c in first.candidates if c.kind == "generated"] == ["g1"]
    assert [c.text for c in second.candidates if c.kind == "generated"] == []
    assert [c.text for c in third.candidates if c.kind == "generated"] == ["g2"]


def test_avoid_provider_provide_already_satisfied_skips_probe_generation(
    monkeypatch: MonkeyPatch,
) -> None:
    provider = AvoidCandidatesProvider(runtime=None, trajectory_debug=False)
    calls: list[int] = []

    def _generated(
        _self: ProbeRuntime, *_args: object, **_kwargs: object
    ) -> list[CandidateSpec]:
        calls.append(1)
        return list()

    monkeypatch.setattr(ProbeRuntime, "generate", _generated)
    ctx = _avoid_ctx(
        "already contains fallback here",
        token_index=0,
        fallback_text="fallback",
    )
    provision = provider.provide(
        CandidateRequest(ctx=ctx, precheck=GenerationPrecheck(ctx))
    )
    assert calls == []
    assert [c.kind for c in provision.candidates] == ["fallback"]


def test_avoid_provider_provide_exhaustion_event_emitted_once_per_stream(
    monkeypatch: MonkeyPatch,
) -> None:
    provider = AvoidCandidatesProvider(runtime=None, trajectory_debug=False)
    generate_calls = 0

    def _none_generated(
        _self: ProbeRuntime, *_args: object, **_kwargs: object
    ) -> list[CandidateSpec]:
        nonlocal generate_calls
        generate_calls += 1
        return list()

    monkeypatch.setattr(ProbeRuntime, "generate", _none_generated)
    sink = RecordingRuntimeEventSink()
    for token_index in range(3):
        ctx = _avoid_ctx(
            "no generated candidates here",
            token_index=token_index,
            validation_for_all=True,
            event_sink=sink,
        )
        provider.provide(
            CandidateRequest(
                ctx=ctx,
                precheck=GenerationPrecheck(ctx),
                event_sink=sink,
            )
        )
    exhausted_events = [
        event
        for event in sink.events
        if isinstance(event, AvoidProbeCacheExhausted)
    ]
    assert generate_calls == 1
    assert len(exhausted_events) == 1


def test_avoid_provider_provide_keeps_base_and_floored_streams_isolated(
    monkeypatch: MonkeyPatch,
) -> None:
    provider = AvoidCandidatesProvider(runtime=None, trajectory_debug=False)
    generated_for_ctx: list[str] = []

    def _generated(
        _self: ProbeRuntime, ctx: StepContext, **_kwargs: object
    ) -> list[CandidateSpec]:
        if ctx.avoid_edit_floor_abs_start is None:
            generated_for_ctx.append("base")
            return [
                CandidateSpec(
                    op=PatchOp.REPLACE,
                    text="base1",
                    kind="generated",
                    priority=100,
                    label="probe_1",
                    candidate_id="probe_1",
                )
            ]
        generated_for_ctx.append("floored")
        return [
            CandidateSpec(
                op=PatchOp.REPLACE,
                text="floor1",
                kind="generated",
                priority=100,
                label="probe_1",
                candidate_id="probe_1",
            )
        ]

    monkeypatch.setattr(ProbeRuntime, "generate", _generated)
    text = "Sentence one. Sentence two trailing"
    first_ctx = _avoid_ctx(text, token_index=0)
    second_ctx = _avoid_ctx(text, token_index=1)
    third_ctx = _avoid_ctx(text, token_index=2)
    first = provider.provide(
        CandidateRequest(ctx=first_ctx, precheck=GenerationPrecheck(first_ctx))
    )
    second = provider.provide(
        CandidateRequest(
            ctx=second_ctx, precheck=GenerationPrecheck(second_ctx)
        )
    )
    third = provider.provide(
        CandidateRequest(ctx=third_ctx, precheck=GenerationPrecheck(third_ctx))
    )
    assert [c.text for c in first.candidates if c.kind == "generated"] == [
        "base1"
    ]
    assert [c.text for c in second.candidates if c.kind == "generated"] == []
    assert [c.text for c in third.candidates if c.kind == "generated"] == [
        "floor1"
    ]
    assert generated_for_ctx == ["base", "floored"]
