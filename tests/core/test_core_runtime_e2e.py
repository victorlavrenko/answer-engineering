from __future__ import annotations

import torch
from pytest import CaptureFixture

from answer_engineering.engine.orchestration.orchestrator import (
    PlanRunner,
)
from answer_engineering.inference.decode.session_orchestration import (
    ExecutionSession,
)
from answer_engineering.inference.decode.state import (
    StreamingDecodeState,
)
from answer_engineering.rules.compile.compiled_rules import (
    CompiledRules,
)
from tests._support.core_helpers import step_test
from tests._support.runtime_harness import configure_runtime_scoring
from tests.core._scoring_stubs import GenerationRuntimeStub

RULES_MD = """
## Replace (once): sensorineural hearing loss

With:

* sudden sensorineural hearing loss
* SSNHL
"""


def test_core_engine_step_edits_with_scoring() -> None:
    engine = ExecutionSession(plan=CompiledRules(RULES_MD).plan)
    configure_runtime_scoring(
        engine,
        generation_runtime=GenerationRuntimeStub.loaded_runtime(),
        require_model_scoring=True,
    )

    out = step_test(
        engine, "Findings support sensorineural hearing loss.", token_index=0
    )
    assert out.changed
    assert "SSNHL" in out.final_text


def test_execution_session_assigns_core_text() -> None:
    engine = ExecutionSession(
        plan=CompiledRules("## Replace: orig\n\nWith:\n\n- edited").plan
    )
    configure_runtime_scoring(
        engine, generation_runtime=None, require_model_scoring=False
    )
    state = StreamingDecodeState(
        past_key_values=None,
        next_input=None,
        assistant_visible_text="orig",
        generated_token_ids=[],
        eos_ids=set(),
        device=torch.device("cpu"),
    )
    changed = engine.apply_step(state=state, tick_index=7)
    assert changed is True
    assert state.assistant_visible_text == "edited"


def test_verbose_noop_is_silent(capsys: CaptureFixture[str]) -> None:
    runner = PlanRunner(verbose=True)
    engine = ExecutionSession(plan=CompiledRules(RULES_MD).plan, runner=runner)
    _ = step_test(engine, "No matching trigger here.", token_index=0)
    out = capsys.readouterr()
    assert out.out == ""


def test_engine_verbose_includes_guard_and_edit_views_with_visible_whitespace(
    capsys: CaptureFixture[str],
) -> None:
    runner = PlanRunner(verbose=True)
    engine = ExecutionSession(plan=CompiledRules(RULES_MD).plan, runner=runner)
    configure_runtime_scoring(
        engine,
        generation_runtime=GenerationRuntimeStub.loaded_runtime(),
        require_model_scoring=True,
    )

    _ = step_test(
        engine, "Findings support\n\tsensorineural hearing loss.", token_index=0
    )
    out = capsys.readouterr().out

    assert "guard_view: span=" in out
    assert "edit_view: span=" in out
    assert "\\n\\t" in out


def test_engine_verbose_emits_single_decision_block(
    capsys: CaptureFixture[str],
) -> None:
    runner = PlanRunner(verbose=True)
    engine = ExecutionSession(plan=CompiledRules(RULES_MD).plan, runner=runner)
    configure_runtime_scoring(
        engine,
        generation_runtime=GenerationRuntimeStub.loaded_runtime(),
        require_model_scoring=True,
    )

    _ = step_test(
        engine, "Findings support sensorineural hearing loss.", token_index=0
    )
    out = capsys.readouterr().out

    assert "[AE] DECISION #" in out
    assert "candidates (" in out
    assert "winner:" in out
    assert "gap2=" in out
    assert "ratio2=" in out
    assert "apply:" in out
    assert 'old: "sensorineural hearing loss"' in out
    assert "new:" in out
    assert "SCORE_ROW" not in out


def test_engine_verbose_has_one_decision_block_per_applied_patch(
    capsys: CaptureFixture[str],
) -> None:
    runner = PlanRunner(verbose=True)
    engine = ExecutionSession(plan=CompiledRules(RULES_MD).plan, runner=runner)
    configure_runtime_scoring(
        engine,
        generation_runtime=GenerationRuntimeStub.loaded_runtime(),
        require_model_scoring=True,
    )

    decision = step_test(
        engine, "Findings support sensorineural hearing loss.", token_index=0
    )
    out = capsys.readouterr().out

    assert out.count("[AE] DECISION #") >= len(decision.applied_patches)
    assert out.count("scope=core") == len(decision.applied_patches)


def test_engine_verbose_streams_rows_before_footer(
    capsys: CaptureFixture[str],
) -> None:
    runner = PlanRunner(verbose=True)
    engine = ExecutionSession(plan=CompiledRules(RULES_MD).plan, runner=runner)
    configure_runtime_scoring(
        engine,
        generation_runtime=GenerationRuntimeStub.loaded_runtime(),
        require_model_scoring=True,
    )

    _ = step_test(
        engine, "Findings support sensorineural hearing loss.", token_index=0
    )
    out = capsys.readouterr().out

    assert out.index("candidates (") < out.index("winner:")
    assert "  1) " in out
    assert 'text="' in out


def test_engine_verbose_decision_ids_do_not_skip_across_steps(
    capsys: CaptureFixture[str],
) -> None:
    runner = PlanRunner(verbose=True)
    engine = ExecutionSession(plan=CompiledRules(RULES_MD).plan, runner=runner)
    configure_runtime_scoring(
        engine,
        generation_runtime=GenerationRuntimeStub.loaded_runtime(),
        require_model_scoring=True,
    )

    _ = step_test(
        engine, "Findings support sensorineural hearing loss.", token_index=0
    )
    out1 = capsys.readouterr().out
    _ = step_test(
        engine, "Findings support sensorineural hearing loss.", token_index=0
    )
    out2 = capsys.readouterr().out

    id1 = int(out1.split("[AE] DECISION #", 1)[1].split(" ", 1)[0])
    id2 = int(out2.split("[AE] DECISION #", 1)[1].split(" ", 1)[0])
    assert id2 == id1 + 1
