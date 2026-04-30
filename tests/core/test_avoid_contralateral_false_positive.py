from __future__ import annotations

from answer_engineering.inference.decode.session_orchestration import (
    ExecutionSession,
)
from answer_engineering.rules.compile.compiled_rules import (
    CompiledRules,
)
from tests._support.core_helpers import step_test
from tests._support.runtime_harness import configure_runtime_scoring


def _engine(md: str) -> ExecutionSession:
    runtime = ExecutionSession(plan=CompiledRules(md).plan)
    configure_runtime_scoring(
        runtime, generation_runtime=None, require_model_scoring=False
    )
    return runtime


def test_contralateral_r_to_l_requires_required_after_all_terms() -> None:
    engine = _engine(
        """## Avoid (repeat): contralateral conductive inference (R->L)

Scope:

* 2 sentences

Prefix (all):

* Weber
* right

Postfix (all):

* left
* conductive

Fallback:

* The test results shall be analyzed carefully.
"""
    )

    early = step_test(
        engine,
        "The Weber test lateralizing to the right.",
        token_index=0,
    )
    assert early.changed is False

    offending = step_test(
        engine,
        (
            "The Weber test lateralizing to the right is consistent with"
            "conductive hearing loss on the left."
        ),
        token_index=1,
    )
    assert offending.changed is True
    assert (
        "The test results shall be analyzed carefully." in offending.final_text
    )


def test_contralateral_l_to_r_requires_required_after_all_terms() -> None:
    engine = _engine(
        """## Avoid (repeat): contralateral conductive inference (L->R)

Prefix (all):

* Weber
* left

Postfix (all):

* right
* conductive

Fallback:

* The test results shall be analyzed carefully.
"""
    )

    early = step_test(
        engine,
        "The Weber test lateralizing to the left.",
        token_index=0,
    )
    assert early.changed is False

    offending = step_test(
        engine,
        (
            "The Weber test lateralizing to the left supports conductive"
            "hearing loss on the right."
        ),
        token_index=1,
    )
    assert offending.changed is True
    assert (
        "The test results shall be analyzed carefully." in offending.final_text
    )


def test_diag_then_tests_no_early_contralateral() -> None:
    engine = _engine(
        """## Avoid (repeat): diagnosis then tests

Scope:

* 2 sentences

Prefix (any):

* conductive

Postfix (any):

* test

Fallback:

* The Weber test lateralizing to the right.

---

## Avoid (repeat): contralateral conductive inference (R->L)

Scope:

* 2 sentences

Prefix (all):

* Weber
* right

Postfix (all):

* left
* conductive

Fallback:

* The test results shall be analyzed carefully.
"""
    )

    first = step_test(
        engine,
        (
            "Conductive hearing loss is suspected, and test findings are still"
            "preliminary."
        ),
        token_index=0,
    )
    assert first.changed is True
    assert first.final_text == "The Weber test lateralizing to the right."

    second = step_test(engine, first.final_text, token_index=1)
    assert second.changed is False

    third = step_test(
        engine,
        first.final_text + " conductive hearing loss on the left is inferred.",
        token_index=2,
    )
    assert third.changed is True
    assert "The test results shall be analyzed carefully." in third.final_text


def test_bidirectional_rules_select_matching_direction_without_connector() -> (
    None
):
    engine = _engine(
        """## Avoid (repeat): contralateral conductive inference (L->R)

Prefix (all):

* Weber
* left

Postfix (all):

* right
* conductive

Fallback:

* fallback-l2r

---

## Avoid (repeat): contralateral conductive inference (R->L)

Prefix (all):

* Weber
* right

Postfix (all):

* left
* conductive

Fallback:

* fallback-r2l
"""
    )

    result = step_test(
        engine,
        (
            "The Weber test lateralizing to the right is consistent with"
            "conductive hearing loss on the left."
        ),
        token_index=0,
    )

    assert result.changed is True
    assert result.final_text == "fallback-r2l"


def test_bidirectional_rules_select_opposite_direction_when_text_reversed() -> (
    None
):
    engine = _engine(
        """## Avoid (repeat): contralateral conductive inference (L->R)

Prefix (all):

* Weber
* left

Postfix (all):

* right
* conductive

Fallback:

* fallback-l2r

---

## Avoid (repeat): contralateral conductive inference (R->L)

Prefix (all):

* Weber
* right

Postfix (all):

* left
* conductive

Fallback:

* fallback-r2l
"""
    )

    result = step_test(
        engine,
        (
            "The Weber test lateralizing to the left is consistent with"
            "conductive hearing loss on the right."
        ),
        token_index=0,
    )

    assert result.changed is True
    assert result.final_text == "fallback-l2r"


def test_scope_beginning_prefix_first_postfix_last() -> None:
    engine = _engine(
        """## Avoid (repeat): conductive without sided context

Scope:

* all

Prefix (all):

* right

Postfix (any):

* conductive

Fallback:

* The findings require complete bilateral assessment.
"""
    )

    result = step_test(
        engine,
        "The right ear findings were noted during bedside testing. "
        "Otoscopic examination is normal. "
        "These results suggest conductive hearing loss.",
        token_index=0,
    )

    assert result.changed is True
    assert (
        result.final_text
        == "The findings require complete bilateral assessment."
    )


def test_connector_requires_postfix_after_connector_when_present() -> None:
    engine = _engine(
        """## Avoid (repeat): connector ordered

Prefix (all):

* Weber

Connector:

* consistent with

Postfix (all):

* conductive
* left

Fallback:

* ordered-fallback
"""
    )

    wrong_order = step_test(
        engine,
        (
            "The Weber test suggests conductive hearing loss on the left, which"
            "is consistent with exam findings."
        ),
        token_index=0,
    )
    assert wrong_order.changed is False

    correct_order = step_test(
        engine,
        (
            "The Weber test is consistent with conductive hearing loss on the"
            "left."
        ),
        token_index=1,
    )
    assert correct_order.changed is True
    assert correct_order.final_text == "ordered-fallback"


def test_without_connector_requires_prefix_before_postfix() -> None:
    engine = _engine(
        """## Avoid (repeat): direction-only ordered

Prefix (all):

* Weber
* right

Postfix (all):

* left
* conductive

Fallback:

* ordered-fallback
"""
    )

    wrong_order = step_test(
        engine,
        (
            "Conductive hearing loss on the left is considered, and only later"
            "Weber lateralizes to the right."
        ),
        token_index=0,
    )
    assert wrong_order.changed is False

    correct_order = step_test(
        engine,
        (
            "Weber lateralizes to the right, supporting conductive hearing loss"
            "on the left."
        ),
        token_index=1,
    )
    assert correct_order.changed is True
    assert correct_order.final_text == "ordered-fallback"


def test_no_connector_prefix_repeats_after_postfix() -> None:
    engine = _engine(
        """## Avoid (repeat): subset ordering with duplicate prefix

Prefix (all):

* left

Postfix (all):

* right

Fallback:

* ordered-fallback
"""
    )

    result = step_test(
        engine,
        "left right something left",
        token_index=0,
    )

    assert result.changed is True
    assert result.final_text == "ordered-fallback"


def test_no_connector_postfix_repeats_before_prefix() -> None:
    engine = _engine(
        """## Avoid (repeat): subset ordering with duplicate postfix

Prefix (all):

* left

Postfix (all):

* right

Fallback:

* ordered-fallback
"""
    )

    result = step_test(
        engine,
        "right mention first, then left appears, and right appears again",
        token_index=0,
    )

    assert result.changed is True
    assert result.final_text == "ordered-fallback"


def test_without_connector_rejects_when_only_postfix_is_before_prefix() -> None:
    engine = _engine(
        """## Avoid (repeat): subset ordering with no valid postfix-after-prefix

Prefix (all):

* left

Postfix (all):

* right

Fallback:

* ordered-fallback
"""
    )

    result = step_test(
        engine,
        "right is noted first and only then left is mentioned",
        token_index=0,
    )

    assert result.changed is False


def test_required_before_all_duplicate_sides_valid_subset() -> None:
    engine = _engine(
        """## Avoid (repeat): contralateral conductive inference (R->L)

Prefix (all):

* Weber
* right

Postfix (all):

* left
* conductive

Fallback:

* ordered-fallback
"""
    )

    result = step_test(
        engine,
        (
            "The Weber test lateralizes to the right, then text references"
            "conductive hearing loss on the left, and later says right again."
        ),
        token_index=0,
    )

    assert result.changed is True
    assert result.final_text == "ordered-fallback"


def test_connector_allows_early_postfix_when_valid_ordered_subset_exists() -> (
    None
):
    engine = _engine(
        """## Avoid (repeat): connector subset ordering

Prefix (all):

* Weber

Connector:

* consistent with

Postfix (all):

* conductive
* left

Fallback:

* ordered-fallback
"""
    )

    result = step_test(
        engine,
        (
            "Early left conductive wording appears, but the Weber finding is"
            "consistent with conductive hearing loss on the left."
        ),
        token_index=0,
    )

    assert result.changed is True
    assert result.final_text == "ordered-fallback"


def test_required_before_any_one_candidate_precedes_postfix() -> None:
    engine = _engine(
        """## Avoid (repeat): prefix-any ordering

Prefix (any):

* left
* right

Postfix (all):

* conductive

Fallback:

* ordered-fallback
"""
    )

    result = step_test(
        engine,
        (
            "The note mentions right first, then conductive findings, then left"
            "later."
        ),
        token_index=0,
    )

    assert result.changed is True
    assert result.final_text == "ordered-fallback"


def test_required_before_any_rejects_all_after_postfix() -> None:
    engine = _engine(
        """## Avoid (repeat): prefix-any invalid ordering

Prefix (any):

* left
* right

Postfix (all):

* conductive

Fallback:

* ordered-fallback
"""
    )

    result = step_test(
        engine,
        (
            "Conductive findings are described first; left and right are only"
            "mentioned later."
        ),
        token_index=0,
    )

    assert result.changed is False


def test_required_before_incomplete_triggers_one_side_before() -> None:
    engine = _engine(
        """## Avoid (repeat): incomplete ordering

Prefix (incomplete):

* left
* right

Postfix (all):

* conductive

Fallback:

* ordered-fallback
"""
    )

    result = step_test(
        engine,
        (
            "Left-sided findings are noted, followed by conductive hearing loss"
            "without bilateral context."
        ),
        token_index=0,
    )

    assert result.changed is True
    assert result.final_text == "ordered-fallback"


def test_prefix_none_blocks_avoid_when_forbidden_prefix_term_is_present() -> (
    None
):
    engine = _engine(
        """## Avoid (repeat): prefix none passthrough

Prefix (none):

* left
* right

Postfix (all):

* conductive

Fallback:

* ordered-fallback
"""
    )

    allowed_result = step_test(
        engine,
        "Conductive hearing loss is present without sided context.",
        token_index=0,
    )

    assert allowed_result.changed is True
    assert allowed_result.final_text == "ordered-fallback"

    blocked_result = step_test(
        engine,
        "Left-sided findings suggest conductive hearing loss.",
        token_index=1,
    )
    assert blocked_result.changed is False


def test_avoid_edit_scope_last_sentence_rewrites_only_last_sentence() -> None:
    engine = _engine(
        """## Avoid (1 sentence): conductive without sided context

Scope:

* all

Prefix (all):

* right

Postfix (any):

* conductive

Fallback:

* SAFE
"""
    )

    source = (
        "The right ear findings were noted during bedside testing. "
        "Otoscopic examination is normal. "
        "These results suggest conductive hearing loss."
    )
    result = step_test(engine, source, token_index=0)

    assert result.changed is True
    assert result.final_text == (
        "The right ear findings were noted during bedside testing. "
        "Otoscopic examination is normal. SAFE"
    )


def test_last_sentence_required_before_incomplete_any_missing() -> None:
    engine = _engine(
        """## Avoid (last sentence): conductive

Scope: all

Prefix (incomplete):

* left
* right

Postfix:

* conductive

Fallback:

* SAFE
"""
    )

    missing_right = step_test(
        engine,
        (
            "Left-sided findings are documented. These findings suggest"
            "conductive hearing loss."
        ),
        token_index=0,
    )
    assert missing_right.changed is True
    assert (
        missing_right.final_text == "Left-sided findings are documented. SAFE"
    )

    missing_both = step_test(
        engine,
        (
            "No side is documented. These findings suggest conductive hearing"
            "loss."
        ),
        token_index=1,
    )
    assert missing_both.changed is True

    assert missing_both.final_text == "No side is documented. SAFE"

    both_present = step_test(
        engine,
        (
            "Left and right findings are documented. These findings suggest"
            "conductive hearing loss."
        ),
        token_index=2,
    )
    assert both_present.changed is False


def test_prefix_clause_rewrites_diagnosis_to_generation_end() -> None:
    engine = _engine(
        """## Avoid (prefix clause): diagnosis with tests

Scope: all

Prefix (any):

* conductive
* sensorineural

Postfix (any):

* test
* testing

Fallback:

* The test results shall be analyzed carefully.
"""
    )

    source = (
        "The patient has intermittent tinnitus. "
        "This is likely conductive hearing loss. This condition is concerning. "
        "Fork test shows something. "
        "Another sentence follows."
    )
    result = step_test(engine, source, token_index=0)

    assert result.changed is True
    assert len(result.applied_patches) == 1
    span = result.applied_patches[0].proposal.span_abs
    assert span is not None
    replaced_text = source[span[0] : span[1]]
    assert replaced_text == (
        " This is likely conductive hearing loss. This condition is "
        "concerning. Fork test shows something. Another sentence follows."
    )
    assert replaced_text.lstrip() == (
        "This is likely conductive hearing loss. This condition is concerning. "
        "Fork test shows something. Another sentence follows."
    )
    assert result.final_text == (
        "The patient has intermittent tinnitus. "
        "The test results shall be analyzed carefully."
    )


def test_avoid_postfix_rewrites_from_clause_containing_postfix_match() -> None:
    engine = _engine(
        """## Avoid (postfix): diagnosis with tests

Scope: all

Prefix (any):

* conductive

Postfix (any):

* testing

Fallback:

* The test results shall be analyzed carefully.
"""
    )

    source = (
        "a b c d. something. some clause, e f conductive g h i. j. k l. m n "
        "testing"
    )
    result = step_test(engine, source, token_index=0)

    assert result.changed is True
    assert len(result.applied_patches) == 1
    span = result.applied_patches[0].proposal.span_abs
    assert span is not None
    replaced_text = source[span[0] : span[1]]
    assert replaced_text == " m n testing"
    assert replaced_text.lstrip() == "m n testing"
    assert result.final_text == (
        "a b c d. something. some clause, "
        "e f conductive g h i. j. k l. The test results shall be analyzed "
        "carefully."
    )


def test_bidirectional_no_double_match_when_both_sides() -> None:
    engine = _engine(
        """## Avoid (once): contralateral conductive inference Weber

Prefix:

- Weber
- left | right

Postfix:

- right | left
- conductive

Fallback:

- fallback
"""
    )

    text = (
        "On tuning fork testing, the Weber test indicates that sound is heard "
        "more prominently in the left ear, suggesting a conductive hearing "
        "loss in the right."
    )

    result = step_test(engine, text, token_index=0)

    assert result.changed is True
    assert result.final_text == "fallback"
