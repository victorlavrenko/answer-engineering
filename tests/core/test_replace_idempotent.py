from __future__ import annotations

from answer_engineering.engine.orchestration.orchestrator import (
    PlanRunner,
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
from answer_engineering.rules.compile.plan import PlanIR
from tests._support.core_helpers import (
    create_step_snapshot,
)
from tests._support.runtime_harness import configure_runtime_scoring

RULES_MD = """
## Replace (once): sensorineural hearing loss

With:

* sudden sensorineural hearing loss
* SSNHL

Scope:

* 800 chars, casefold
""".strip()


def _load_plan() -> tuple[PlanRunner, PlanIR]:
    engine = ExecutionSession(plan=CompiledRules(RULES_MD).plan)
    plan = engine.plan
    configure_runtime_scoring(
        engine, generation_runtime=None, require_model_scoring=False
    )
    return engine.runner, plan


def test_replace_skips_when_scope_already_satisfies_with_phrase() -> None:
    runner, plan = _load_plan()
    text = "The patient has sudden sensorineural hearing loss."
    result = runner.run(
        plan,
        create_step_snapshot(snapshot_text=text, token_index=0),
    )

    assert result.applied_patches == []
    assert result.proposals == []
    assert result.events == []


def test_replace_proposes_edits_when_with_phrase_not_in_scope() -> None:
    runner, plan = _load_plan()
    text = "The patient has sensorineural hearing loss."
    result = runner.run(
        plan,
        create_step_snapshot(snapshot_text=text, token_index=0),
    )

    assert any(p.op == PatchOp.REPLACE for p in result.proposals)
