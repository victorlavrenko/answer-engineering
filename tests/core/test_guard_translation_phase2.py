from __future__ import annotations

from answer_engineering.engine.proposal.match_tree.match_tree import (
    MatchAll,
    MatchAndThen,
    MatchAny,
    MatchNot,
    MatchTerm,
)
from answer_engineering.rules.compile.compiler import (
    FullPlanCompiler,
)
from answer_engineering.rules.parse.ast import (
    ReplaceRuleAST,
    RulesetAST,
    SetMatchAST,
)
from answer_engineering.rules.parse.parser import (
    MarkdownRulesParser,
)
from tests.core.match_tree_guard_factory import build_guard_expression


def test_build_expression_single_any_requirement() -> None:
    tree = build_guard_expression(required_before_any=("weber",))
    assert isinstance(tree, MatchAny)
    assert tree.children == (MatchTerm("weber"),)


def test_build_expression_ordered_before_and_after() -> None:
    tree = build_guard_expression(
        required_before_all=("left",),
        required_after_any=("sensorineural",),
        ordered=True,
    )
    assert isinstance(tree, MatchAndThen)
    assert tree.left == MatchTerm("left")
    assert isinstance(tree.right, MatchAny)
    assert tree.right.children == (MatchTerm("sensorineural"),)


def test_build_expression_after_all_is_all_node() -> None:
    tree = build_guard_expression(required_after_all=("left", "conductive"))
    assert isinstance(tree, MatchAll)
    assert tree.children == (MatchTerm("left"), MatchTerm("conductive"))


def test_build_expression_empty_guard_is_none() -> None:
    assert build_guard_expression() is None


def test_build_expression_incomplete_lowers_to_not_all() -> None:
    tree = build_guard_expression(required_before_incomplete=("left", "right"))
    assert isinstance(tree, MatchNot)
    assert isinstance(tree.child, MatchAll)
    assert tree.child.children == (MatchTerm("left"), MatchTerm("right"))


def test_build_expression_ordered_connector_is_between_left_and_right() -> None:
    tree = build_guard_expression(
        required_before_all=("left",),
        connectors=("suggests", "consistent with"),
        required_after_any=("sensorineural",),
        ordered=True,
    )
    assert isinstance(tree, MatchAndThen)
    assert tree.left == MatchTerm("left")
    assert isinstance(tree.right, MatchAndThen)
    assert isinstance(tree.right.left, MatchAny)
    assert tree.right.left.children == (
        MatchTerm("suggests"),
        MatchTerm("consistent with"),
    )
    assert isinstance(tree.right.right, MatchAny)
    assert tree.right.right.children == (MatchTerm("sensorineural"),)


def test_build_expression_mixed_shape_is_structural() -> None:
    tree = build_guard_expression(
        required_before_all=("weber",),
        required_before_any=("left", "right"),
        required_before_incomplete=("conductive", "sensorineural"),
        connectors=("is consistent with",),
        required_after_all=("left", "conductive"),
        required_after_any=("urgent",),
        ordered=True,
    )
    assert isinstance(tree, MatchAll)
    assert any(isinstance(child, MatchNot) for child in tree.children)
    assert any(isinstance(child, MatchAndThen) for child in tree.children)


def test_compiler_emits_expression_only_guard_for_avoid() -> None:
    md = """## Avoid (repeat): conductive

Prefix (all):

* weber

Postfix (any):

* conductive

Fallback:

* fallback
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    guard = plan.rules[0].guard
    assert guard is not None
    assert isinstance(guard.expression, MatchAndThen)
    assert isinstance(guard.expression.right, MatchAny)


def test_compiler_emits_expression_only_guard_for_gate() -> None:
    ast = RulesetAST(
        [
            ReplaceRuleAST(
                rule_id="1",
                fire="once",
                target="old",
                candidates=("new",),
                gate=SetMatchAST(expression=MatchAny((MatchTerm("trigger"),))),
            )
        ]
    )
    plan = FullPlanCompiler().compile(ast)

    guard = plan.rules[0].guard
    assert guard is not None
    assert isinstance(guard.expression, MatchAny)
    assert guard.expression.children == (MatchTerm("trigger"),)
