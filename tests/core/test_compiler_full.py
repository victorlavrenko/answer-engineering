from __future__ import annotations

from pathlib import Path

from answer_engineering.engine.proposal.match_tree.match_tree import (
    MatchAll,
    MatchAndThen,
    MatchAny,
    MatchNot,
    MatchTerm,
    MatchTree,
)
from answer_engineering.engine.runtime.runtime_types import (
    PatchOp,
)
from answer_engineering.rules.compile.compiler import (
    FullPlanCompiler,
)
from answer_engineering.rules.parse.ast import (
    AfterRuleAST,
    ReplaceRuleAST,
    SetMatchAST,
)
from answer_engineering.rules.parse.parser import (
    MarkdownRulesParser,
)


def test_full_compiler_lowers_avoid_postfix() -> None:
    md = Path("tests/fixtures/rules_full_syntax.md").read_text(encoding="utf-8")
    ast = MarkdownRulesParser().parse(md)
    plan = FullPlanCompiler().compile(ast)

    assert len(plan.rules) == 3
    avoid = plan.rules[2]
    assert avoid.guard_scope is not None
    assert avoid.edit_scope is not None
    assert avoid.guard_scope.kind == "tail_clauses"
    assert avoid.guard_scope.n == 1
    assert avoid.edit_scope.kind == "tail_clauses"
    assert avoid.edit_scope.n == 1
    assert avoid.guard is not None
    assert avoid.guard.expression is not None
    assert avoid.target.kind == "clause_containing_anchor_to_scope_end"
    assert avoid.target.anchor_id == "postfix_match"
    assert avoid.anchors
    assert avoid.anchors[0].anchor_id == "postfix_match"
    assert avoid.policy.probe_num_beams == 7
    assert avoid.policy.probe_max_new_tokens == 6
    assert avoid.policy.skip_tokens == 2
    candidates = list(avoid.candidates)
    assert candidates[0].op == PatchOp.REPLACE


def test_full_compiler_lowers_avoid_everything_to_scope_entire() -> None:
    md = """## Avoid (repeat): conductive

Prefix (all):

* weber

Postfix (any):

* conductive

Scope:

* 800 chars

Fallback:

* these findings require further evaluation.
"""
    ast = MarkdownRulesParser().parse(md)
    plan = FullPlanCompiler().compile(ast)

    avoid = plan.rules[0]
    assert avoid.anchors == ()
    assert avoid.target.kind == "scope_entire"
    assert avoid.target.anchor_id is None
    assert avoid.scope.kind == "tail_chars"
    assert avoid.scope.max_chars == 800
    assert avoid.guard_scope is not None
    assert avoid.edit_scope is not None
    assert avoid.guard_scope == avoid.scope
    assert avoid.edit_scope == avoid.scope
    assert avoid.guard is not None
    assert avoid.guard.expression is not None


def test_full_compiler_expands_parenthetical_after_anchor_punctuation() -> None:
    md = """## After (once): (SSNHL)

Add:

* This condition requires urgent treatment.
"""
    ast = MarkdownRulesParser().parse(md)
    plan = FullPlanCompiler().compile(ast)

    after = plan.rules[0]
    assert after.anchors
    assert after.anchors[0].match_phrase_any == (
        "(SSNHL).",
        "(SSNHL),",
        "(SSNHL)",
    )


def test_full_compiler_expands_after_anchor_parenthetical_variants() -> None:
    md = """## After: SSNHL

Add:

* This condition requires urgent treatment.
"""
    ast = MarkdownRulesParser().parse(md)
    plan = FullPlanCompiler().compile(ast)

    after = plan.rules[0]
    assert after.anchors
    assert after.anchors[0].match_phrase_any == (
        "SSNHL.",
        "SSNHL,",
        "SSNHL",
        "(SSNHL).",
        "(SSNHL),",
        "(SSNHL)",
        "(SSNHL.",
        "(SSNHL,",
        "(SSNHL",
    )


def test_full_compiler_maps_avoid_probe_option_aliases() -> None:
    md = """## Avoid (repeat): conductive

Prefix (all):

* weber

Connector:

* this suggests

Postfix (any):

* conductive

Options:

* Width: 3
* Tokens: 2

Fallback:

* fallback
"""
    ast = MarkdownRulesParser().parse(md)
    plan = FullPlanCompiler().compile(ast)

    avoid = plan.rules[0]
    assert avoid.policy.probe_num_beams == 3
    assert avoid.policy.probe_max_new_tokens == 2


def test_full_compiler_defaults_after_scope_to_all() -> None:
    md = """## After (once): SSNHL

Add:

* This condition requires urgent treatment.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    after = plan.rules[0]
    assert after.scope.kind == "whole_doc"


def test_full_compiler_after_anchor_scope_end_replace_target() -> None:
    md = """## After (once): SSNHL

Add:

* This condition requires urgent treatment.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    after = plan.rules[0]
    assert after.target.kind == "after_anchor_to_scope_end"
    assert all(
        candidate.op.value == "replace" for candidate in after.candidates
    )


def test_full_compiler_respects_explicit_after_scope_override() -> None:
    md = """## After (once): SSNHL

Scope:

* 120 chars

Add:

* This condition requires urgent treatment.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    after = plan.rules[0]
    assert after.scope.kind == "tail_chars"
    assert after.scope.max_chars == 120


def test_full_compiler_defaults_avoid_scope_to_all() -> None:
    md = """## Avoid (repeat): conductive

Prefix (all):

* weber

Postfix (any):

* conductive

Fallback:

* these findings require further evaluation.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    avoid = plan.rules[0]
    assert avoid.scope.kind == "whole_doc"


def test_full_compiler_respects_explicit_avoid_scope_override() -> None:
    md = """## Avoid (repeat): conductive

Prefix (all):

* weber

Postfix (any):

* conductive

Scope:

* 2 sentences

Fallback:

* these findings require further evaluation.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    avoid = plan.rules[0]
    assert avoid.scope.kind == "tail_sentences"
    assert avoid.scope.n == 2


def test_full_compiler_parses_scope_from_beginning_aliases_to_whole_doc() -> (
    None
):
    md = """## Avoid (repeat): conductive

Scope:

* from the beginning

Prefix (all):

* weber

Postfix (any):

* conductive

Fallback:

* these findings require further evaluation.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    avoid = plan.rules[0]
    assert avoid.scope.kind == "whole_doc"


def test_full_compiler_defaults_after_wait_for_closing_parenthesis() -> None:
    md = """## After (once): SSNHL

Add:

* This condition requires urgent treatment.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    after = plan.rules[0]
    assert after.wait_for_closing_parenthesis is True


def test_full_compiler_parses_after_wait_for_closing_override() -> None:
    md = """## After (once): SSNHL

Add:

* This condition requires urgent treatment.

Options:

* Fire regime: don't wait for closing
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    after = plan.rules[0]
    assert after.wait_for_closing_parenthesis is False


def test_full_compiler_maps_avoid_required_after_all_to_guard_all() -> None:
    md = """## Avoid (repeat): contralateral conductive inference (R->L)

Prefix (all):

* Weber
* right

Postfix (all):

* left
* conductive

Fallback:

* The test results shall be analyzed carefully.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    avoid = plan.rules[0]
    assert avoid.guard is not None
    assert avoid.guard.expression is not None


def test_full_compiler_avoid_postfix_rewrite_keeps_require_order_enabled() -> (
    None
):
    md = """## Avoid (repeat, postfix): conductive inference

Prefix (all):

* Weber

Connector:

* is consistent with

Postfix (all):

* conductive

Fallback:

* fallback text
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    avoid = plan.rules[0]
    assert avoid.guard is not None
    assert avoid.guard.expression is not None


def test_full_compiler_parses_avoid_edit_scope_last_sentences_mod() -> None:
    md = """## Avoid (2 last sentences): conductive

Scope:

* all

Prefix (all):

* weber

Postfix (any):

* conductive

Fallback:

* these findings require further evaluation.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    avoid = plan.rules[0]
    assert avoid.guard_scope is not None
    assert avoid.edit_scope is not None
    assert avoid.guard_scope.kind == "whole_doc"
    assert avoid.edit_scope.kind == "tail_sentences"
    assert avoid.edit_scope.n == 2


def test_full_compiler_parses_avoid_edit_scope_last_clause_mod() -> None:
    md = """## Avoid (last clause): conductive

Scope:

* all

Prefix (all):

* weber

Postfix (any):

* conductive

Fallback:

* these findings require further evaluation.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    avoid = plan.rules[0]
    assert avoid.guard_scope is not None
    assert avoid.edit_scope is not None
    assert avoid.guard_scope.kind == "whole_doc"
    assert avoid.edit_scope.kind == "tail_clauses"
    assert avoid.edit_scope.n == 1
    assert avoid.edit_scope.include_leading_delimiter is True
    assert avoid.target.kind == "scope_entire"


def test_full_compiler_parses_avoid_edit_scope_prefix_clause_mod() -> None:
    md = """## Avoid (prefix clause): diagnosis

Scope:

* all

Prefix (any):

* conductive
* sensorineural

Postfix (any):

* test

Fallback:

* fallback text
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    avoid = plan.rules[0]
    assert avoid.guard_scope is not None
    assert avoid.edit_scope is not None
    assert avoid.guard_scope.kind == "whole_doc"
    assert avoid.edit_scope.kind == "whole_doc"
    assert avoid.target.kind == "clause_containing_anchor_to_scope_end"
    assert avoid.target.anchor_id == "prefix_match"
    assert avoid.target.include_anchor is True
    assert avoid.anchors
    anchor = avoid.anchors[0]
    assert anchor.anchor_id == "prefix_match"
    assert anchor.match_mode == "last"
    assert anchor.match_phrase_any == ("conductive", "sensorineural")


def test_full_compiler_parses_avoid_edit_scope_anchor_scope_end_mod() -> None:
    md = """## Avoid (clause_containing_anchor_to_scope_end): diagnosis

Scope:

* all

Prefix (any):

* conductive
* sensorineural

Postfix (any):

* test

Fallback:

* fallback text
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    avoid = plan.rules[0]
    assert avoid.target.kind == "clause_containing_anchor_to_scope_end"
    assert avoid.target.anchor_id == "prefix_match"


def test_full_compiler_parses_explicit_avoid_everything_mod() -> None:
    md = """## Avoid (everything): conductive

Scope:

* all

Prefix (all):

* weber

Postfix (any):

* conductive

Fallback:

* fallback text
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    avoid = plan.rules[0]
    assert avoid.guard_scope is not None
    assert avoid.edit_scope is not None
    assert avoid.guard_scope.kind == "whole_doc"
    assert avoid.edit_scope.kind == "whole_doc"
    assert avoid.target.kind == "scope_entire"
    assert avoid.anchors == ()


def test_full_compiler_parses_all_as_avoid_everything_mod() -> None:
    md = """## Avoid (all): conductive

Scope:

* all

Prefix (all):

* weber

Postfix (any):

* conductive

Fallback:

* fallback text
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    avoid = plan.rules[0]
    assert avoid.guard_scope is not None
    assert avoid.edit_scope is not None
    assert avoid.guard_scope.kind == "whole_doc"
    assert avoid.edit_scope.kind == "whole_doc"
    assert avoid.target.kind == "scope_entire"
    assert avoid.anchors == ()


def test_full_compiler_supports_inline_scope_value_for_all_rules() -> None:
    md = """## Replace: a

Scope: all

With:

* b

---

## After: marker

Scope: all

Add:

* note

---

## Avoid: bad

Scope: all

Prefix (incomplete):

* left
* right

Postfix:

* conductive

Fallback:

* fallback

---

## Force: marker

Scope: all

Add:

* force
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    assert len(plan.rules) == 4
    assert all(rule.scope.kind == "whole_doc" for rule in plan.rules)
    avoid = plan.rules[2]
    assert avoid.guard is not None
    assert avoid.guard.expression is not None
    assert "left" in _leaf_terms(avoid.guard.expression)
    assert "right" in _leaf_terms(avoid.guard.expression)


def test_compiler_expands_templates_into_cross_product() -> None:
    md = """## Avoid (once): contralateral conductive inference Weber

Prefix:

* Weber | forehead
* left || right

Postfix:

* right || left
* conductive

Fallback:

* The test results shall be analyzed carefully.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))

    assert len(plan.rules) == 4
    prefix_pairs: list[tuple[str, ...]] = []
    postfix_pairs: list[tuple[str, ...]] = []
    for rule in plan.rules:
        if (
            rule.guard is None
            or rule.guard.expression is None
            or not isinstance(rule.guard.expression, MatchAll)
        ):
            continue
        ordered_children = [
            child
            for child in rule.guard.expression.children
            if isinstance(child, MatchAndThen)
        ]
        if len(ordered_children) != 2:
            continue
        left_terms = [
            child.left.expression
            for child in ordered_children
            if isinstance(child.left, MatchTerm)
        ]
        prefix_pairs.append(tuple(left_terms))
        postfix_pairs.append(tuple(_leaf_terms(ordered_children[0].right)))
    assert prefix_pairs == [
        ("Weber", "left"),
        ("Weber", "right"),
        ("forehead", "left"),
        ("forehead", "right"),
    ]
    assert postfix_pairs == [
        ("right", "conductive"),
        ("left", "conductive"),
        ("right", "conductive"),
        ("left", "conductive"),
    ]


def test_full_compiler_lowers_ordered_avoid_tail_andthen_markers() -> None:
    md = """## Avoid (once): ordered tail

Prefix (all):

* alpha

Connector:

* mid

Postfix (all):

* beta
* gamma
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))
    guard = plan.rules[0].guard
    assert guard is not None
    assert isinstance(guard.expression, MatchAndThen)
    assert isinstance(guard.expression.right, MatchAndThen)
    assert isinstance(guard.expression.right.left, MatchAny)
    connector_group = guard.expression.right.left
    assert all(
        isinstance(node, MatchTerm) and node.marker == "connector"
        for node in connector_group.children
    )
    postfix_group = guard.expression.right.right
    assert isinstance(postfix_group, MatchAll)
    assert tuple(
        child.expression
        for child in postfix_group.children
        if isinstance(child, MatchTerm)
    ) == ("beta", "gamma")
    assert all(
        isinstance(child, MatchTerm) and child.marker == "postfix_all"
        for child in postfix_group.children
    )


def test_compile_replace_candidates_use_authored_text_as_label() -> None:
    md = """## Replace (once): sensorineural hearing loss

With:

* sudden sensorineural hearing loss
* SSNHL
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))
    replace = plan.rules[0]

    assert tuple(c.candidate_id for c in replace.candidates) == (
        "rewrite_1",
        "rewrite_2",
    )
    assert tuple(c.label for c in replace.candidates) == (
        "sudden sensorineural hearing loss",
        "SSNHL",
    )


def test_compile_after_candidates_use_authored_text_as_label() -> None:
    md = """## After (once): SSNHL

Add:

* This condition requires urgent treatment.
* Recommend ENT follow-up.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))
    after = plan.rules[0]

    assert tuple(c.candidate_id for c in after.candidates) == (
        "insert_1",
        "insert_2",
    )
    assert tuple(c.label for c in after.candidates) == (
        "This condition requires urgent treatment.",
        "Recommend ENT follow-up.",
    )


def test_compile_force_candidates_use_authored_text_as_label() -> None:
    md = """## Force (repeat): hearing loss counseling

Add:

* Counsel patient about symptom monitoring.
* Document red-flag return precautions.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))
    force = plan.rules[0]

    assert tuple(c.candidate_id for c in force.candidates) == (
        "force_1",
        "force_2",
    )
    assert tuple(c.label for c in force.candidates) == (
        "Counsel patient about symptom monitoring.",
        "Document red-flag return precautions.",
    )


def _leaf_terms(node: MatchTree) -> list[str]:
    if isinstance(node, MatchTerm):
        return [node.expression]
    if isinstance(node, MatchNot):
        return _leaf_terms(node.child)
    if isinstance(node, MatchAndThen):
        return _leaf_terms(node.left) + _leaf_terms(node.right)
    if isinstance(node, MatchAll | MatchAny):
        out: list[str] = []
        for child in node.children:
            out.extend(_leaf_terms(child))
        return out
    return list()


def test_stable_replace_rule_id_ignores_gate_markers() -> None:
    compiler = FullPlanCompiler()
    marked_prefix_any = ReplaceRuleAST(
        rule_id="rule-1",
        fire="once",
        target="target",
        candidates=("candidate",),
        gate=SetMatchAST(
            expression=MatchAny(
                (
                    MatchTerm("sudden", marker="prefix_any"),
                    MatchTerm("abrupt", marker="prefix_any"),
                ),
                marker="prefix_any_group",
            )
        ),
    )
    marked_prefix_all = ReplaceRuleAST(
        rule_id="rule-1",
        fire="once",
        target="target",
        candidates=("candidate",),
        gate=SetMatchAST(
            expression=MatchAny(
                (
                    MatchTerm("sudden", marker="prefix_all"),
                    MatchTerm("abrupt", marker="prefix_all"),
                ),
                marker="prefix_all_group",
            )
        ),
    )

    assert compiler.stable_rule_id(
        marked_prefix_any
    ) == compiler.stable_rule_id(marked_prefix_all)


def test_stable_after_rule_id_ignores_gate_markers() -> None:
    compiler = FullPlanCompiler()
    prefix_any_gate = SetMatchAST(
        expression=MatchAndThen(
            MatchTerm("left", marker="prefix_any"),
            MatchTerm("right", marker="postfix_any"),
            marker="sequence_any",
        )
    )
    prefix_none_gate = SetMatchAST(
        expression=MatchAndThen(
            MatchTerm("left", marker="prefix_none"),
            MatchTerm("right", marker="postfix_none"),
            marker="sequence_none",
        )
    )
    before = AfterRuleAST(
        rule_id="rule-after",
        fire="repeat",
        target="anchor",
        candidates=("c",),
        gate=prefix_any_gate,
    )
    after = AfterRuleAST(
        rule_id="rule-after",
        fire="repeat",
        target="anchor",
        candidates=("c",),
        gate=prefix_none_gate,
    )

    assert repr(before) != repr(after)
    assert compiler.stable_rule_id(before) == compiler.stable_rule_id(after)


def test_stable_rule_id_changes_when_replace_gate_semantics_change() -> None:
    compiler = FullPlanCompiler()
    sudden_gate = ReplaceRuleAST(
        rule_id="rule-semantics",
        fire="once",
        target="target",
        candidates=("candidate",),
        gate=SetMatchAST(expression=MatchTerm("sudden")),
    )
    abrupt_gate = ReplaceRuleAST(
        rule_id="rule-semantics",
        fire="once",
        target="target",
        candidates=("candidate",),
        gate=SetMatchAST(expression=MatchTerm("abrupt")),
    )

    assert compiler.stable_rule_id(sudden_gate) != compiler.stable_rule_id(
        abrupt_gate
    )


def test_stable_rule_id_changes_when_structural_any_semantics_change() -> None:
    compiler = FullPlanCompiler()
    any_a_b = AfterRuleAST(
        rule_id="rule-struct",
        fire="once",
        target="anchor",
        candidates=("candidate",),
        gate=SetMatchAST(expression=MatchAny((MatchTerm("a"), MatchTerm("b")))),
    )
    any_a_c = AfterRuleAST(
        rule_id="rule-struct",
        fire="once",
        target="anchor",
        candidates=("candidate",),
        gate=SetMatchAST(expression=MatchAny((MatchTerm("a"), MatchTerm("c")))),
    )

    assert compiler.stable_rule_id(any_a_b) != compiler.stable_rule_id(any_a_c)
