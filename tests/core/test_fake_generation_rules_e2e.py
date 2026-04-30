from __future__ import annotations

from pathlib import Path

from answer_engineering.inference.decode.session_orchestration import (
    ExecutionSession,
)
from answer_engineering.rules.compile.compiled_rules import (
    CompiledRules,
)
from tests._support.core_helpers import step_test
from tests._support.runtime_harness import configure_runtime_scoring
from tests.core._scoring_stubs import GenerationRuntimeStub


def _build_engine(
    rules_md: str, *, with_model_scoring: bool
) -> ExecutionSession:
    engine = ExecutionSession(plan=CompiledRules(rules_md).plan)
    if with_model_scoring:
        configure_runtime_scoring(
            engine,
            generation_runtime=GenerationRuntimeStub.loaded_runtime(),
            require_model_scoring=True,
        )
    else:
        configure_runtime_scoring(
            engine,
            generation_runtime=None,
            require_model_scoring=False,
        )
    return engine


def test_fake_generation_applies_replace_after_and_avoid_rules() -> None:
    replace_engine = _build_engine(
        """
## Replace (once): sensorineural hearing loss
With:
- SSNHL
""",
        with_model_scoring=True,
    )
    replace_decision = step_test(
        replace_engine,
        "The findings are consistent with sensorineural hearing loss.",
        token_index=0,
    )
    assert replace_decision.changed
    assert "SSNHL" in replace_decision.final_text

    after_engine = _build_engine(
        """
## After (once): SSNHL
Add:
- Prompt steroid treatment is indicated.
""",
        with_model_scoring=False,
    )
    after_decision = step_test(
        after_engine, "The findings are consistent with SSNHL.", token_index=1
    )
    assert after_decision.changed
    assert "Prompt steroid treatment is indicated." in after_decision.final_text

    avoid_engine = _build_engine(
        Path("tests/fixtures/rules_full_syntax.md").read_text(encoding="utf-8"),
        with_model_scoring=False,
    )
    avoid_decision = step_test(
        avoid_engine,
        (
            "weber rinne left right positive this suggests "
            "conductive hearing loss."
        ),
        token_index=2,
    )
    assert avoid_decision.changed
    assert (
        "these findings require further evaluation."
        in avoid_decision.final_text
    )
