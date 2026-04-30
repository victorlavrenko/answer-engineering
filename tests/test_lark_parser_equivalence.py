from __future__ import annotations

import pytest

from answer_engineering.config.engine_defaults import ScopeDefaults
from answer_engineering.rules.parse.ast import (
    AfterRuleAST,
    AvoidRuleAST,
    ReplaceRuleAST,
)
from answer_engineering.rules.parse.errors import (
    RulesSyntaxError,
)
from answer_engineering.rules.parse.parser import (
    MarkdownRulesParser,
)


def test_lark_parser_produces_expected_ast_shape() -> None:
    text = """
## Replace (once): sensorineural hearing loss

With:

* sudden sensorineural hearing loss
* SSNHL

---

## After (repeat): (SSNHL)

Add:

* This condition requires urgent treatment.

---

## Avoid (postfix, repeat): conductive

Prefix (all):

* weber
* rinne

Postfix (any):

* conductive
"""
    ruleset = MarkdownRulesParser().parse(text)

    assert len(ruleset.rules) == 3
    assert isinstance(ruleset.rules[0], ReplaceRuleAST)
    assert ruleset.rules[0].candidates == (
        "sudden sensorineural hearing loss",
        "SSNHL",
    )
    assert isinstance(ruleset.rules[1], AfterRuleAST)
    assert ruleset.rules[1].fire == "repeat"
    assert isinstance(ruleset.rules[2], AvoidRuleAST)
    assert ruleset.rules[2].edit.kind == "postfix"
    assert tuple(ruleset.rules[2].required_before_all) == ("weber", "rinne")


def test_lark_parser_is_deterministic() -> None:
    text = "## Replace (once): x\n\nWith:\n\n* y\n"
    parser = MarkdownRulesParser()
    assert parser.parse(text) == parser.parse(text)


def test_lark_parser_uses_central_scope_defaults() -> None:
    text = "## Replace: x\n\nWith:\n\n* y\n"
    ruleset = MarkdownRulesParser().parse(text)

    replace = ruleset.rules[0]
    assert isinstance(replace, ReplaceRuleAST)
    assert replace.scope is not None
    assert replace.scope.kind == "whole_doc"
    assert replace.scope.n == 0
    assert replace.scope.casefold is ScopeDefaults().casefold


def test_lark_parser_reports_line_and_column() -> None:
    text = "## Replace (once): target\n\nWith:\n\n* ok\n%%%"
    parser = MarkdownRulesParser()

    with pytest.raises(RulesSyntaxError) as ei:
        parser.parse(text)

    exc = ei.value
    assert exc.line == 6
    assert exc.column == 1
    assert "%%%" in exc.snippet


def test_lark_parser_rejects_stray_prose_lines() -> None:
    text = "## Replace (once): x\n\nWith:\n\n* y\n\nNote: extra prose"
    parser = MarkdownRulesParser()

    with pytest.raises(RulesSyntaxError) as ei:
        parser.parse(text)

    exc = ei.value
    assert exc.line == 7
    assert exc.column == 1
    assert "extra prose" in exc.snippet


def test_lark_parser_defaults_rule_modifiers_by_kind() -> None:
    text = """
## Replace: hearing loss

With:

* HL

---

## Avoid: diagnosis then tests

Prefix (any):

* conductive

Postfix (any):

* test

Fallback:

* The test results shall be analyzed carefully.
"""
    ruleset = MarkdownRulesParser().parse(text)

    replace = ruleset.rules[0]
    assert isinstance(replace, ReplaceRuleAST)
    assert replace.fire == "once"

    avoid = ruleset.rules[1]
    assert isinstance(avoid, AvoidRuleAST)
    assert avoid.fire == "repeat"
    assert avoid.edit.kind == "everything"


def test_lark_parser_defaults_unqualified_prefix_to_any_for_replace() -> None:
    text = """
## Replace: hearing loss

Prefix:

* sudden
* acute

With:

* SSNHL
"""
    ruleset = MarkdownRulesParser().parse(text)

    replace = ruleset.rules[0]
    assert isinstance(replace, ReplaceRuleAST)
    assert replace.gate is not None
    assert tuple(replace.gate.any_of) == ("sudden", "acute")
    assert tuple(replace.gate.all_of) == ()
    assert tuple(replace.gate.none_of) == ()


def test_lark_parser_defaults_unqualified_prefix_to_any_for_after() -> None:
    text = """
## After: SSNHL

Prefix:

* sudden

Add:

* Prompt treatment is indicated.
"""
    ruleset = MarkdownRulesParser().parse(text)

    after = ruleset.rules[0]
    assert isinstance(after, AfterRuleAST)
    assert after.gate is not None
    assert tuple(after.gate.any_of) == ("sudden",)
    assert tuple(after.gate.all_of) == ()
    assert tuple(after.gate.none_of) == ()


def test_lark_parser_defaults_unqualified_avoid_prefix_and_postfix() -> None:
    text = """
## Avoid: diagnosis then tests

Prefix:

* conductive
* SSNHL

Postfix:

* test

Fallback:

* The test results shall be analyzed carefully.
"""
    ruleset = MarkdownRulesParser().parse(text)

    avoid = ruleset.rules[0]
    assert isinstance(avoid, AvoidRuleAST)
    assert tuple(avoid.required_before_any) == ()
    assert tuple(avoid.required_before_all) == ("conductive", "SSNHL")
    assert tuple(avoid.required_before_incomplete) == ()
    assert tuple(avoid.required_after_any) == ()
    assert tuple(avoid.required_after_all) == ("test",)


@pytest.mark.parametrize(
    "text",
    [
        """
## Avoid: diagnosis then tests

Scope: 2 sentences

Prefix:

* conductive

Postfix:

* test

Fallback:

* The reported findings should be reviewed together before concluding \
the mechanism of hearing loss.
""",
        """
## Avoid: diagnosis then tests
scope: 2 sentences
prefix: conductive
postfix: test
fallback: The reported findings should be reviewed together before \
concluding the mechanism of hearing loss.
""",
    ],
)
def test_lark_parser_supports_bullet_and_inline_single_value_sections_for_avoid(
    text: str,
) -> None:
    ruleset = MarkdownRulesParser().parse(text)

    avoid = ruleset.rules[0]
    assert isinstance(avoid, AvoidRuleAST)
    assert avoid.scope is not None
    assert avoid.scope.kind == "tail_sentences"
    assert avoid.scope.n == 2
    assert tuple(avoid.required_before_all) == ("conductive",)
    assert tuple(avoid.required_after_all) == ("test",)
    assert tuple(avoid.fallback) == (
        (
            "The reported findings should be reviewed together before "
            "concluding the mechanism of hearing loss."
        ),
    )


def test_lark_parser_defaults_avoid_scope_to_all() -> None:
    text = (
        "## Avoid: diagnosis then tests\n\n"
        "Fallback:\n\n"
        "* Use diagnosis-first ordering.\n"
    )
    ruleset = MarkdownRulesParser().parse(text)

    avoid = ruleset.rules[0]
    assert isinstance(avoid, AvoidRuleAST)
    assert avoid.scope is not None
    assert avoid.scope.kind == "whole_doc"
    assert avoid.scope.n == 0
    assert avoid.scope.casefold is ScopeDefaults().casefold


def test_lark_parser_preserves_explicit_avoid_scope() -> None:
    text = "## Avoid: diagnosis then tests\n\nScope:\n\n* 250 chars\n"
    ruleset = MarkdownRulesParser().parse(text)

    avoid = ruleset.rules[0]
    assert isinstance(avoid, AvoidRuleAST)
    assert avoid.scope is not None
    assert avoid.scope.kind == "tail_chars"
    assert avoid.scope.n == 250


def test_lark_parser_expands_pipe_templates_for_avoid() -> None:
    text = """
## Avoid: contralateral conductive inference

Prefix (all):

* Weber
* right | left

Postfix (all):

* left | right
* conductive

Fallback:

* The test results shall be analyzed carefully.
"""
    ruleset = MarkdownRulesParser().parse(text)

    assert len(ruleset.rules) == 2
    first = ruleset.rules[0]
    second = ruleset.rules[1]
    assert isinstance(first, AvoidRuleAST)
    assert isinstance(second, AvoidRuleAST)
    assert tuple(first.required_before_all) == ("Weber", "right")
    assert tuple(first.required_after_all) == ("left", "conductive")
    assert tuple(second.required_before_all) == ("Weber", "left")
    assert tuple(second.required_after_all) == ("right", "conductive")


def test_lark_parser_expands_pipe_templates_with_three_variants() -> None:
    text = """
## Replace: term

Prefix (all):

* A | B | C
* C | B | A

With:

* E | F | G
"""
    ruleset = MarkdownRulesParser().parse(text)

    assert len(ruleset.rules) == 3
    first = ruleset.rules[0]
    second = ruleset.rules[1]
    third = ruleset.rules[2]
    assert isinstance(first, ReplaceRuleAST)
    assert isinstance(second, ReplaceRuleAST)
    assert isinstance(third, ReplaceRuleAST)
    assert first.gate is not None
    assert second.gate is not None
    assert third.gate is not None
    assert tuple(first.gate.all_of) == ("A", "C")
    assert tuple(second.gate.all_of) == ("B", "B")
    assert tuple(third.gate.all_of) == ("C", "A")
    assert first.candidates == ("E",)
    assert second.candidates == ("F",)
    assert third.candidates == ("G",)


def test_lark_parser_rejects_template_variant_count_mismatch() -> None:
    text = """
## Replace: term

Prefix (all):

* A | B
* C | D | E

With:

* fix
"""
    with pytest.raises(RulesSyntaxError):
        MarkdownRulesParser().parse(text)


def test_lark_parser_expands_multidimensional_pipe_templates() -> None:
    text = """
## Avoid (once): contralateral conductive inference Weber

Prefix:

* Weber | forehead
* left || right

Postfix:

* right || left
* conductive

Fallback:

* The test results shall be analyzed carefully.
"""
    ruleset = MarkdownRulesParser().parse(text)

    assert len(ruleset.rules) == 4
    pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for rule in ruleset.rules:
        assert isinstance(rule, AvoidRuleAST)
        pairs.append(
            (tuple(rule.required_before_all), tuple(rule.required_after_all))
        )

    assert pairs == [
        (("Weber", "left"), ("right", "conductive")),
        (("Weber", "right"), ("left", "conductive")),
        (("forehead", "left"), ("right", "conductive")),
        (("forehead", "right"), ("left", "conductive")),
    ]


def test_lark_parser_rejects_mixed_pipe_dimensions_in_single_bullet() -> None:
    text = """
## Replace: term

With:

* A | B || C
"""
    with pytest.raises(RulesSyntaxError):
        MarkdownRulesParser().parse(text)


def test_lark_parser_rejects_multidimensional_variant_count_mismatch() -> None:
    text = """
## Replace: term

Prefix (all):

* A || B
* C || D || E

With:

* fixed
"""
    with pytest.raises(RulesSyntaxError):
        MarkdownRulesParser().parse(text)


def test_lark_parser_allows_escaped_pipe_in_bullets() -> None:
    text = """
## Replace: term

With:

* literal \\| pipe
"""
    ruleset = MarkdownRulesParser().parse(text)
    replace = ruleset.rules[0]
    assert isinstance(replace, ReplaceRuleAST)
    assert replace.candidates == ("literal | pipe",)
