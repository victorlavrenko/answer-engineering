from __future__ import annotations

from datetime import datetime

import pytest

from ae_paper_reproduction.core.evaluation.run_session import (
    RunSession,
    SubrunSession,
)
from answer_engineering import (
    CompiledRules,
    GenerationPolicy,
    GenerationRequest,
    GenerationRuntime,
)
from tests._support.core_helpers import (
    create_step_snapshot,
)


def test_application_session_contract_ids_are_stable() -> None:
    run = RunSession(now=datetime(2026, 4, 6), run_tag=None)
    tagged_run = RunSession(now=datetime(2026, 4, 6), run_tag="phase-b")

    assert run.run_id == "20260406T000000Z"
    assert tagged_run.run_id == "20260406T000000Z-phase-b"
    assert (
        SubrunSession(index=3, ruleset_name="Paper SSNHL / demo").subrun_id
        == "003-paper-ssnhl-demo"
    )


def test_generation_runtime_contract_enforces_constructor_invariants() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        GenerationRequest(question="")

    request = GenerationRequest(question="What is SSNHL?")
    policy = GenerationPolicy(rules="## Replace: foo\n\nWith:\n\n- bar\n")
    runtime = GenerationRuntime(model_id="demo/model")

    assert request.question == "What is SSNHL?"
    assert isinstance(policy.compiled_rules, CompiledRules)
    assert runtime.model_id == "demo/model"
    assert policy.max_new_tokens > 0


def test_rule_engine_step_contract_preserves_payload_semantics() -> None:
    step = create_step_snapshot(
        snapshot_text="hello",
        token_index=2,
        prompt_text="prompt",
        generated_ids=(10, 11),
    )

    assert step.snapshot_text == "hello"
    assert step.token_index == 2
    assert step.prompt_text == "prompt"
    assert step.generated_ids == (10, 11)
