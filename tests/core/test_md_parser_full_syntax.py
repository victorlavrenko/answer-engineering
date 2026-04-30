from __future__ import annotations

from pathlib import Path

import pytest

from answer_engineering.engine.proposal.match_tree.match_tree import (
    MatchAll,
    MatchAndThen,
    MatchAny,
    MatchNot,
    MatchTerm,
    MatchTree,
)
from answer_engineering.rules.parse.ast import (
    AfterRuleAST,
    AvoidRuleAST,
    ReplaceRuleAST,
)
from answer_engineering.rules.parse.parser import (
    ConditionedTerms,
    MarkdownRulesParser,
    Mode,
)


def test_full_syntax_fixture_parses() -> None:
    md = Path("tests/fixtures/rules_full_syntax.md").read_text(encoding="utf-8")
    ast = MarkdownRulesParser().parse(md)

    assert len(ast.rules) == 3
    assert isinstance(ast.rules[0], ReplaceRuleAST)
    assert isinstance(ast.rules[1], AfterRuleAST)
    assert isinstance(ast.rules[2], AvoidRuleAST)
    avoid = ast.rules[2]
    assert avoid.edit.kind == "postfix"
    assert tuple(avoid.required_before_all) == (
        "weber",
        "rinne",
        "left",
        "right",
        "positive",
    )
    assert tuple(avoid.required_after_any) == ("conductive",)


def test_md_parser_accepts_dash_bullets() -> None:
    md = """
## Replace (once): sensorineural hearing loss

With:

- SSNHL
"""
    ast = MarkdownRulesParser().parse(md)
    assert len(ast.rules) == 1
    assert isinstance(ast.rules[0], ReplaceRuleAST)
    assert ast.rules[0].candidates == ("SSNHL",)


def test_md_parser_parses_avoid_required_after_all_section() -> None:
    md = """## Avoid: contralateral conductive inference (R->L)

Prefix (all):

* Weber
* right

Postfix (all):

* left
* conductive

Fallback:

* The test results shall be analyzed carefully.
"""
    ast = MarkdownRulesParser().parse(md)

    avoid = ast.rules[0]
    assert isinstance(avoid, AvoidRuleAST)
    assert avoid.required_after_any == ()
    assert avoid.required_after_all == ("left", "conductive")


def test_md_parser_parses_prompt_section_for_avoid_boundary_matching() -> None:
    md = """## Avoid (repeat): conductive

Prompt (all):

* left
* right

Prefix (incomplete):

* left
* right

Postfix (any):

* sensorineural
* SSNHL

Fallback:

* fallback
"""
    ast = MarkdownRulesParser().parse(md)
    avoid = ast.rules[0]
    assert isinstance(avoid, AvoidRuleAST)
    assert isinstance(avoid.guard_expression, MatchAndThen)
    assert avoid.guard_expression.marker == "prompt_answer_boundary"
    markers = {
        marker
        for marker in _collect_markers(avoid.guard_expression)
        if marker is not None
    }
    assert "prompt_all" in markers
    assert "prefix_incomplete" in markers
    assert "postfix_any" in markers
    assert all(
        marker.startswith(("prefix_", "postfix_", "prompt_"))
        or marker in {"connector", "prompt_answer_boundary"}
        for marker in markers
    )
    assert all(not marker.startswith(("pre_", "post_")) for marker in markers)


@pytest.mark.parametrize(
    ("mod", "expected_kind", "expected_count"),
    [
        ("last sentence", "tail_sentences", 1),
        ("1 sentence", "tail_sentences", 1),
        ("1 last sentence", "tail_sentences", 1),
        ("2 last sentences", "tail_sentences", 2),
        ("3 sentences", "tail_sentences", 3),
        ("last clause", "tail_clauses", 1),
        ("1 clause", "tail_clauses", 1),
        ("2 last clauses", "tail_clauses", 2),
        ("prefix clause", "prefix_clause", 0),
        ("matched prefix clause", "prefix_clause", 0),
        ("clause containing anchor to scope end", "prefix_clause", 0),
        ("clause_containing_anchor_to_scope_end", "prefix_clause", 0),
    ],
)
def test_md_parser_parses_avoid_edit_scope_last_sentences_mod(
    mod: str, expected_kind: str, expected_count: int
) -> None:
    md = f"""## Avoid ({mod}): conductive

Prefix (all):

* weber

Postfix (any):

* conductive

Fallback:

* fallback
"""
    ast = MarkdownRulesParser().parse(md)

    avoid = ast.rules[0]
    assert isinstance(avoid, AvoidRuleAST)
    assert avoid.edit.kind == expected_kind
    if expected_kind == "tail_sentences":
        assert avoid.edit.n_sentences == expected_count
    elif expected_kind == "tail_clauses":
        assert avoid.edit.n_clauses == expected_count


def test_md_parser_parses_explicit_avoid_everything_mod() -> None:
    md = """## Avoid (everything): conductive

Prefix (all):

* weber

Postfix (any):

* conductive

Fallback:

* fallback
"""
    ast = MarkdownRulesParser().parse(md)

    avoid = ast.rules[0]
    assert isinstance(avoid, AvoidRuleAST)
    assert avoid.edit.kind == "everything"


def test_md_parser_parses_all_as_avoid_everything_mod() -> None:
    md = """## Avoid (all): conductive

Prefix (all):

* weber

Postfix (any):

* conductive

Fallback:

* fallback
"""
    ast = MarkdownRulesParser().parse(md)

    avoid = ast.rules[0]
    assert isinstance(avoid, AvoidRuleAST)
    assert avoid.edit.kind == "everything"


def _collect_markers(node: MatchTree | None) -> tuple[str | None, ...]:
    if node is None:
        return tuple()
    markers: list[str | None] = [node.marker]
    if isinstance(node, MatchTerm):
        return tuple(markers)
    if isinstance(node, MatchAndThen):
        return tuple(
            [
                *markers,
                *_collect_markers(node.left),
                *_collect_markers(node.right),
            ]
        )
    if isinstance(node, MatchNot):
        return tuple([*markers, *_collect_markers(node.child)])
    if isinstance(node, MatchAll | MatchAny):
        for child in node.children:
            markers.extend(_collect_markers(child))
    return tuple(markers)


@pytest.mark.parametrize("operator", ["incomplete", "partial", "missing"])
def test_md_parser_parses_avoid_required_before_incomplete_synonyms(
    operator: str,
) -> None:
    md = f"""## Avoid: conductive

Prefix ({operator}):

* left
* right

Postfix:

* conductive

Fallback:

* fallback
"""
    ast = MarkdownRulesParser().parse(md)

    avoid = ast.rules[0]
    assert isinstance(avoid, AvoidRuleAST)
    assert avoid.required_before_incomplete == ("left", "right")


@pytest.mark.parametrize("kind", ["Replace", "After", "Avoid", "Force"])
def test_md_parser_supports_inline_scope_value(kind: str) -> None:
    tail = {
        "Replace": """With:

* replacement""",
        "After": """Add:

* inserted""",
        "Avoid": """Postfix:

* banned

Fallback:

* fallback""",
        "Force": """Add:

* force text""",
    }[kind]
    md = f"""## {kind}: token

Scope: all

{tail}
"""

    ast = MarkdownRulesParser().parse(md)
    assert ast.rules[0].scope is not None
    assert ast.rules[0].scope.kind == "whole_doc"


def test_conditioned_terms_builds_modes_with_default_routing() -> None:
    terms = ConditionedTerms(
        section_values={
            "prefix": ["a", "b"],
            "prefix:any": ["c"],
            "prefix:all": ["d"],
            "prefix:none": ["e"],
            "prefix:incomplete": ["f"],
        },
        label="prefix",
        default_op="all",
    )

    assert dict(terms.items()) == {
        "any": ("c",),
        "all": ("a", "b", "d"),
        "none": ("e",),
        "incomplete": ("f",),
    }


def test_conditioned_terms_preserves_constructor_provided_terms() -> None:
    terms = ConditionedTerms(
        terms_by_mode={
            "any": ("base-any",),
            "all": ("base-all",),
            "none": ("base-none",),
            "incomplete": ("base-inc",),
        }
    )

    assert dict(terms.items()) == {
        "any": ("base-any",),
        "all": ("base-all",),
        "none": ("base-none",),
        "incomplete": ("base-inc",),
    }


def test_conditioned_terms_mode_alias_is_reusable_for_mode_iteration() -> None:
    ordered_modes: tuple[Mode, ...] = ConditionedTerms.MODES
    assert ordered_modes == ("any", "all", "none", "incomplete")
