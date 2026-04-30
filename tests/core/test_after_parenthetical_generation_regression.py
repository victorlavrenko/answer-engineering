from __future__ import annotations

from answer_engineering.engine.runtime.runtime_types import TokenCharAlignment
from answer_engineering.inference.decode.session_orchestration import (
    ExecutionSession,
)
from answer_engineering.rules.compile.compiled_rules import CompiledRules
from tests._support.core_helpers import create_step_snapshot, step_test
from tests._support.runtime_harness import configure_runtime_scoring

_RULES = """
## After: SSNHL
Add:
- This condition requires urgent treatment.
"""


def _build_engine() -> ExecutionSession:
    engine = ExecutionSession(plan=CompiledRules(_RULES).plan)
    configure_runtime_scoring(
        engine,
        generation_runtime=None,
        require_model_scoring=False,
    )
    return engine


def test_after_ssnhl_waits_until_parenthetical_is_closed() -> None:
    engine = _build_engine()

    open_parenthetical = (
        "These findings are consistent with a diagnosis of sudden "
        "sensorineural hearing loss (SSNHL"
    )

    premature = step_test(engine, open_parenthetical, token_index=0)

    assert not premature.changed
    assert premature.final_text == open_parenthetical


def test_after_ssnhl_appends_after_closing_parenthesis_and_punctuation() -> (
    None
):
    engine = _build_engine()

    closed_parenthetical = (
        "These findings are consistent with a diagnosis of sudden "
        "sensorineural hearing loss (SSNHL)."
    )

    decision = step_test(engine, closed_parenthetical, token_index=1)

    assert decision.changed
    assert decision.final_text == (
        "These findings are consistent with a diagnosis of sudden "
        "sensorineural hearing loss (SSNHL). "
        "This condition requires urgent treatment."
    )
    assert "(SSNHL. This condition" not in decision.final_text
    assert "(SSNHL). This condition" in decision.final_text


def test_after_does_not_snap_empty_after_span_back_into_parenthetical() -> None:
    engine = _build_engine()

    text = (
        "These findings are consistent with a diagnosis of sudden "
        "sensorineural hearing loss (SSNHL).\n\n"
    )

    # Simulates the notebook case where the closing suffix is inside/near one
    # generated token span. The bug is that replace-span snapping expands the
    # after-rule replacement backwards into the already-complete parenthetical.
    alignment = (
        TokenCharAlignment(
            token_index=0,
            char_start=text.index(")."),
            char_end=len(text),
            piece_text=").\n\n",
        ),
    )

    decision = engine.execute_step(
        create_step_snapshot(
            snapshot_text=text,
            token_index=1,
            generated_token_alignment=alignment,
        )
    )

    assert decision.changed
    assert (
        "(SSNHL). This condition requires urgent treatment."
        in decision.final_text
    )
    assert "(SSNHL. This condition" not in decision.final_text


def test_after_production_shape_does_not_snap_newline_target_into_suffix() -> (
    None
):
    engine = _build_engine()

    text = (
        "The patient presents with sudden onset hearing loss in the right "
        "ear, which is a concerning symptom. The otoscopic examination is "
        "normal, which rules out any obvious external or middle ear "
        "pathology. \n\n"
        "Next, we look at the tuning fork testing results. The Weber test "
        "indicates that sound is heard more prominently in the left ear "
        "when the fork is placed on the forehead and air conduction is "
        "better than bone conduction in the right ear. These findings are "
        "consistent with a diagnosis of sudden sensorineural hearing loss "
        "(SSNHL).\n\n"
    )

    assert text[522:530] == "(SSNHL)."
    assert text[530:532] == "\n\n"
    assert text[528:532] == ").\n\n"

    decision = engine.execute_step(
        create_step_snapshot(
            snapshot_text=text,
            token_index=1,
            generated_token_alignment=(
                TokenCharAlignment(
                    token_index=0,
                    char_start=528,
                    char_end=532,
                    piece_text=").\n\n",
                ),
            ),
        )
    )

    assert decision.changed
    applied = decision.applied_patches[0]
    assert applied.proposal.span_abs == (530, 530)

    assert (
        "(SSNHL). This condition requires urgent treatment."
        in decision.final_text
    )
    assert "(SSNHL. This condition" not in decision.final_text
