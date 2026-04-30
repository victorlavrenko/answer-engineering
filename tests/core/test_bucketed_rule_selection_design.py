from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch
from pytest import CaptureFixture

from answer_engineering.engine.orchestration.orchestrator import (
    PlanRunner,
)
from answer_engineering.engine.proposal.candidates.avoid import (
    AvoidCandidatesProvider,
)
from answer_engineering.engine.proposal.candidates.base import (
    CandidateProvision,
    CandidateRequest,
)
from answer_engineering.engine.runtime.runtime_types import (
    PatchOp,
)
from answer_engineering.inference.decode.session_orchestration import (
    ExecutionSession,
)
from answer_engineering.rules.compile.compiled_rules import (
    CompiledRules,
)
from answer_engineering.rules.compile.plan import (
    AnchorQuerySpec,
    CandidateSpec,
    DecisionPolicySpec,
    EditTargetSpec,
    FirePolicySpec,
    GuardSpec,
    PlanIR,
    RulePlan,
    ScopeSpec,
)
from tests._support.core_helpers import (
    create_step_snapshot,
    step_test,
)
from tests._support.runtime_harness import configure_runtime_scoring
from tests.core.match_tree_guard_factory import build_guard_expression


def _guard(**legacy_fields: object) -> GuardSpec:
    return GuardSpec(expression=build_guard_expression(**legacy_fields))


def _engine(md: str) -> ExecutionSession:
    runtime = ExecutionSession(plan=CompiledRules(md).plan)
    configure_runtime_scoring(
        runtime, generation_runtime=None, require_model_scoring=False
    )
    return runtime


def test_bucket_selection_applies_only_earliest_edit_start_group() -> None:
    engine = _engine(
        """## Replace: alpha

With:

- __R1__

---

## Replace: beta

With:

- __R2__
"""
    )

    result = step_test(engine, "alpha beta", token_index=0)

    # Desired bucketed behavior: only the earliest edit-start bucket is allowed
    # to produce edits in this pass.
    assert result.final_text == "__R1__ beta"


def test_bucket_conflict_resolution_runs_within_same_earliest_bucket() -> None:
    engine = _engine(
        """## Replace: alpha

With:

- LONG

---

## Replace: alpha

With:

- S
"""
    )

    result = step_test(engine, "alpha beta", token_index=0)

    # Both edits target the same start bucket;
    # conflict/scoring should choose one.
    assert result.final_text == "S beta"


def test_bucketing_advances_when_earliest_bucket_only_noops() -> None:
    engine = _engine(
        """## Replace: alpha

With:

- alpha

---

## Replace: beta

With:

- __R2__
"""
    )

    result = step_test(engine, "alpha beta", token_index=0)

    # Desired pragmatic behavior: if earliest bucket only yields noops,
    # advance to the next bucket.
    assert result.final_text == "alpha __R2__"


def test_bucket_selection_supports_fake_generated_candidates(
    monkeypatch: MonkeyPatch,
) -> None:
    engine = _engine(
        """## Avoid (repeat): alpha

Postfix (any):

- alpha

Fallback:

- fallback

---

## Replace: beta

With:

- __R2__
"""
    )

    runner = engine.runner
    if runner.proposal_engine.candidates_providers is None:
        raise ValueError("Proposal engine candidates providers not configured.")
    avoid_provider = next(
        provider
        for provider in runner.proposal_engine.candidates_providers
        if isinstance(provider, AvoidCandidatesProvider)
    )

    def _fake_provide(request: CandidateRequest) -> CandidateProvision:
        return CandidateProvision(
            ctx=request.ctx,
            candidates=(
                CandidateSpec(
                    op=PatchOp.REPLACE,
                    text="GEN",
                    kind="generated",
                    priority=100,
                    label="probe_1",
                    candidate_id="probe_1",
                    logprob=-0.01,
                ),
            ),
        )

    monkeypatch.setattr(avoid_provider, "provide", _fake_provide)

    result = step_test(engine, "alpha beta", token_index=0)

    # Fake generated candidate should be consumable by avoid path and win over
    # fallback candidates for its span.
    assert result.final_text == "GEN"


def test_bucket_selection_inserts_after_anchor_end() -> None:
    engine = _engine(
        """## After: alpha

With:

- alpha inserted

---

## Replace: alpha beta

With:

- replaced
"""
    )

    result = step_test(engine, "alpha beta", token_index=0)

    # Insert-after is bucketed at anchor end,
    # so earlier replace bucket wins first.
    assert result.final_text == "replaced"


def test_bucket_debug_logs_include_precheck_spans_and_bucket_map(
    capsys: CaptureFixture[str],
) -> None:
    engine = ExecutionSession(
        plan=CompiledRules(
            """## Replace: alpha

With:

- R

---

## After: alpha

With:

- a
"""
        ).plan
    )
    configure_runtime_scoring(
        engine, generation_runtime=None, require_model_scoring=False
    )
    runner = engine.runner
    runner.trajectory_debug = True
    runner.verbose = False

    _ = step_test(engine, "alpha beta", token_index=0)
    out = capsys.readouterr().out

    assert "[AE] PRECHECK_TRIGGER" in out
    assert "bucket_start=" in out
    assert "[AE] PRECHECK_BUCKETS" in out


def test_bucket_debug_skips_precheck_when_non_generated_satisfied(
    capsys: CaptureFixture[str],
) -> None:
    engine = ExecutionSession(
        plan=CompiledRules(
            """## Replace: sudden sensorineural hearing loss

With:

- sudden sensorineural hearing loss
"""
        ).plan
    )
    configure_runtime_scoring(
        engine, generation_runtime=None, require_model_scoring=False
    )
    runner = engine.runner
    runner.trajectory_debug = True
    runner.verbose = False

    _ = step_test(
        engine,
        "The diagnosis is sudden sensorineural hearing loss.",
        token_index=0,
    )
    out = capsys.readouterr().out

    assert "[AE] PRECHECK_TRIGGER" not in out


def test_bucket_advances_on_insufficient_postfix_overlap() -> None:
    plan = PlanIR(
        rules=(
            RulePlan(
                rule_id="r-avoid-earliest",
                name="avoid:insufficient overlap",
                guard_scope=ScopeSpec(kind="whole_doc", casefold=True),
                edit_scope=ScopeSpec(kind="tail_sentences", n=1, casefold=True),
                guard=_guard(
                    required_before_all=("left",),
                    required_after_any=("sensorineural",),
                ),
                target=EditTargetSpec(kind="scope_entire"),
                candidates=(
                    CandidateSpec(op=PatchOp.REPLACE, text="SAFE", priority=10),
                ),
                policy=DecisionPolicySpec(skip_tokens=0),
                fire=FirePolicySpec(mode="repeat"),
            ),
            RulePlan(
                rule_id="r-replace-later",
                name="replace:beta",
                guard_scope=ScopeSpec(kind="whole_doc", casefold=True),
                edit_scope=ScopeSpec(kind="whole_doc", casefold=True),
                anchors=(
                    AnchorQuerySpec(
                        anchor_id="replace_anchor",
                        match_phrase_any=("Final",),
                        match_mode="first",
                    ),
                ),
                target=EditTargetSpec(
                    kind="match_span", anchor_id="replace_anchor"
                ),
                candidates=(
                    CandidateSpec(op=PatchOp.REPLACE, text="DONE", priority=5),
                ),
                policy=DecisionPolicySpec(skip_tokens=0),
                fire=FirePolicySpec(mode="repeat"),
            ),
        )
    )
    text = "Left ear appears sensorineural. Final sentence has no diagnosis."

    out = PlanRunner().run(
        plan, create_step_snapshot(snapshot_text=text, token_index=10)
    )

    assert out.final_doc.text.startswith("Left ear appears sensorineural. DONE")


def test_bucket_stops_when_earliest_avoid_has_sufficient_postfix_overlap() -> (
    None
):
    plan = PlanIR(
        rules=(
            RulePlan(
                rule_id="r-avoid-earliest",
                name="avoid:sufficient overlap",
                guard_scope=ScopeSpec(kind="whole_doc", casefold=True),
                edit_scope=ScopeSpec(kind="tail_sentences", n=1, casefold=True),
                guard=_guard(
                    required_before_all=("left",),
                    required_after_any=("sensorineural",),
                ),
                target=EditTargetSpec(kind="scope_entire"),
                candidates=(
                    CandidateSpec(op=PatchOp.REPLACE, text="SAFE", priority=10),
                ),
                policy=DecisionPolicySpec(skip_tokens=0),
                fire=FirePolicySpec(mode="repeat"),
            ),
            RulePlan(
                rule_id="r-replace-later",
                name="replace:beta",
                guard_scope=ScopeSpec(kind="whole_doc", casefold=True),
                edit_scope=ScopeSpec(kind="whole_doc", casefold=True),
                anchors=(
                    AnchorQuerySpec(
                        anchor_id="replace_anchor",
                        match_phrase_any=("Final",),
                        match_mode="first",
                    ),
                ),
                target=EditTargetSpec(
                    kind="match_span", anchor_id="replace_anchor"
                ),
                candidates=(
                    CandidateSpec(op=PatchOp.REPLACE, text="DONE", priority=5),
                ),
                policy=DecisionPolicySpec(skip_tokens=0),
                fire=FirePolicySpec(mode="repeat"),
            ),
        )
    )
    text = "Left ear appears baseline. Final sentence is sensorineural."

    out = PlanRunner().run(
        plan, create_step_snapshot(snapshot_text=text, token_index=10)
    )

    assert out.final_doc.text == "Left ear appears baseline. SAFE"
