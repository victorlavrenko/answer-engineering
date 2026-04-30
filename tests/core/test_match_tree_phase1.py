from __future__ import annotations

import pytest

from answer_engineering.engine.proposal.match_tree.match_tree import (
    MatchAll,
    MatchAndThen,
    MatchAny,
    MatchNot,
    MatchTerm,
    MatchTree,
    Span,
)


def test_all_and_any_reject_empty_children() -> None:
    with pytest.raises(ValueError, match="MatchAll nodes"):
        MatchAll(())
    with pytest.raises(ValueError, match="MatchAny nodes"):
        MatchAny(())


def test_constructor_normalization_flattens_nested_all_and_any() -> None:
    tree = MatchAll(
        (MatchTerm("a"), MatchAll((MatchTerm("b"), MatchTerm("c"))))
    )
    assert isinstance(tree, MatchAll)
    assert tuple(
        child.expression
        for child in tree.children
        if isinstance(child, MatchTerm)
    ) == (
        "a",
        "b",
        "c",
    )

    any_tree = MatchAny(
        (MatchTerm("x"), MatchAny((MatchTerm("y"), MatchTerm("z"))))
    )
    assert isinstance(any_tree, MatchAny)
    assert tuple(
        child.expression
        for child in any_tree.children
        if isinstance(child, MatchTerm)
    ) == (
        "x",
        "y",
        "z",
    )


def test_normalize_policy_is_constructor_owned() -> None:
    nested = MatchAll(
        (MatchTerm("a"), MatchAll((MatchTerm("b"), MatchTerm("c"))))
    )
    normalized = nested.normalize()
    assert normalized is nested
    assert isinstance(normalized, MatchAll)
    assert tuple(
        child.expression
        for child in normalized.children
        if isinstance(child, MatchTerm)
    ) == (
        "a",
        "b",
        "c",
    )


def test_term_all_any_not_evaluation_basics() -> None:
    text = "alpha beta"

    term_result = MatchTerm("alpha").evaluate(text)
    assert term_result.matched
    assert term_result.spans

    all_result = MatchAll((MatchTerm("alpha"), MatchTerm("beta"))).evaluate(
        text
    )
    assert all_result.matched
    assert all_result.telemetry.node_type == "MatchAll"
    assert len(all_result.telemetry.children) == 2

    any_result = MatchAny((MatchTerm("gamma"), MatchTerm("beta"))).evaluate(
        text
    )
    assert any_result.matched
    assert any_result.telemetry.node_type == "MatchAny"

    not_result = MatchNot(MatchTerm("gamma")).evaluate(text)
    assert not_result.matched
    assert not_result.telemetry.node_type == "MatchNot"
    assert len(not_result.telemetry.children) == 1


def test_not_has_empty_spans_and_preserves_child_telemetry() -> None:
    result = MatchNot(
        MatchTerm("alpha", marker="child"), marker="not-node"
    ).evaluate("alpha beta")
    assert not result.matched
    assert result.spans == ()
    assert result.telemetry.marker == "not-node"
    assert result.telemetry.children[0].marker == "child"
    assert result.telemetry.children[0].matched


def test_and_then_ordered_semantics_allow_adjacency() -> None:
    tree = MatchAndThen(MatchTerm("a"), MatchTerm("b"))
    assert tree.evaluate("a b").matched
    assert tree.evaluate("ab").matched
    assert not tree.evaluate("b a").matched


def test_and_then_adjacency_uses_left_end_lte_right_start() -> None:
    tree = MatchAndThen(MatchTerm("a"), MatchTerm("b"))
    result = tree.evaluate("ab")
    assert result.spans == (Span(0, 2),)


def test_and_then_searches_all_pairs_not_first_found_only() -> None:
    tree = MatchAndThen(MatchTerm("a"), MatchTerm("b"))
    assert tree.evaluate("b a b").matched
    assert tree.evaluate("a b a").matched


def test_disjunction_ordering_examples() -> None:
    tree = MatchAndThen(
        MatchAny((MatchTerm("a"), MatchTerm("b"))),
        MatchAny((MatchTerm("c"), MatchTerm("d"))),
    )

    positives = (
        "a d",
        "c b c",
        "c b d",
        "d c a c",
        "d a b a d",
    )
    for text in positives:
        assert tree.evaluate(text).matched, text

    negatives = (
        "d c b",
        "c d b a",
        "d c b a",
    )
    for text in negatives:
        assert not tree.evaluate(text).matched, text


def test_disjunction_ordering_corner_cases_regression() -> None:
    tree = MatchAndThen(
        MatchAny((MatchTerm("a"), MatchTerm("b"))),
        MatchAny((MatchTerm("c"), MatchTerm("d"))),
    )

    negatives = (
        "b a",
        "c a",
        "d b",
        "c b",
        "c b a b",
    )
    for text in negatives:
        assert not tree.evaluate(text).matched, text

    positives = (
        "a c a",
        "b c a",
        "b c b",
        "a c b",
        "a d b",
        "d a c",
        "c b c",
        "c b d",
    )
    for text in positives:
        assert tree.evaluate(text).matched, text


def test_marker_telemetry_is_preserved_on_nodes() -> None:
    left: MatchTree = MatchAny(
        (MatchTerm("a", marker="term-a"), MatchTerm("b", marker="term-b")),
        marker="left",
    )
    right: MatchTree = MatchAny(
        (MatchTerm("c", marker="term-c"), MatchTerm("d", marker="term-d")),
        marker="right",
    )
    tree = MatchAndThen(left, right, marker="root")

    result = tree.evaluate("a d")
    assert result.matched
    assert result.telemetry.marker == "root"
    assert result.telemetry.children[0].marker == "left"
    assert result.telemetry.children[1].marker == "right"
    assert result.telemetry.children[0].children[0].marker == "term-a"
    assert result.telemetry.children[1].children[1].marker == "term-d"


def test_node_owned_debug_string_rendering() -> None:
    tree = MatchAndThen(
        MatchAll((MatchTerm("a"), MatchAny((MatchTerm("b"), MatchTerm("c"))))),
        MatchNot(MatchTerm("d")),
    )
    assert tree.to_debug_string() == "AND_THEN(ALL(a, ANY(b, c)) -> NOT(d))"


def test_ordered_overlap_subtree_is_node_owned_and_topology_aware() -> None:
    ordered = MatchAndThen(
        MatchTerm("left"),
        MatchAndThen(MatchAny((MatchTerm("connector"),)), MatchTerm("right")),
    )
    assert ordered.ordered_overlap_subtree() == ordered.right

    mixed = MatchAll((MatchTerm("extra"), ordered))
    subtree = mixed.ordered_overlap_subtree()
    assert subtree is not None
    assert subtree.to_debug_string() == "AND_THEN(ANY(connector) -> right)"


def test_nested_and_then_with_disjunction_accepts_expected_in_order() -> None:
    tree = MatchAndThen(
        MatchAndThen(
            MatchTerm("a"), MatchAny((MatchTerm("b1"), MatchTerm("b2")))
        ),
        MatchTerm("c"),
    )

    positives = (
        "a b1 c",
        "a b2 c",
        "a b1 c a",
        "a b1 a c",
        "c a b2 a c",
    )
    for text in positives:
        assert tree.evaluate(text).matched, text


def test_nested_and_then_with_disjunction_rejects_out_of_order_overlap() -> (
    None
):
    tree = MatchAndThen(
        MatchAndThen(
            MatchTerm("a"), MatchAny((MatchTerm("b1"), MatchTerm("b2")))
        ),
        MatchTerm("c"),
    )

    assert not tree.evaluate("b2 a b1 c").matched


def test_match_term_fingerprint_ignores_marker() -> None:
    left = MatchTerm("sudden", marker="prefix_any")
    right = MatchTerm("sudden", marker="prefix_all")
    assert left.fingerprint() == right.fingerprint()


def test_structural_fingerprints_are_semantic_and_ordered() -> None:
    any_a_b = MatchAny((MatchTerm("a"), MatchTerm("b")))
    any_a_b_marked = MatchAny(
        (
            MatchTerm("a", marker="prefix_any"),
            MatchTerm("b", marker="prefix_any"),
        ),
        marker="prefix_any_group",
    )
    any_b_a = MatchAny((MatchTerm("b"), MatchTerm("a")))
    any_a_c = MatchAny((MatchTerm("a"), MatchTerm("c")))
    assert any_a_b.fingerprint() == any_a_b_marked.fingerprint()
    assert any_a_b.fingerprint() != any_b_a.fingerprint()
    assert any_a_b.fingerprint() != any_a_c.fingerprint()

    all_a_b = MatchAll((MatchTerm("a"), MatchTerm("b")))
    all_a_b_marked = MatchAll(
        (
            MatchTerm("a", marker="prefix_all"),
            MatchTerm("b", marker="prefix_all"),
        ),
        marker="prefix_all_group",
    )
    assert all_a_b.fingerprint() == all_a_b_marked.fingerprint()
    assert all_a_b.fingerprint() != any_a_b.fingerprint()

    not_a = MatchNot(MatchTerm("a"))
    not_a_marked = MatchNot(MatchTerm("a", marker="prefix_none"), marker="none")
    not_b = MatchNot(MatchTerm("b"))
    assert not_a.fingerprint() == not_a_marked.fingerprint()
    assert not_a.fingerprint() != not_b.fingerprint()

    and_then_a_b = MatchAndThen(MatchTerm("a"), MatchTerm("b"))
    and_then_a_b_marked = MatchAndThen(
        MatchTerm("a", marker="prefix_all"),
        MatchTerm("b", marker="postfix_all"),
        marker="ordered",
    )
    and_then_b_a = MatchAndThen(MatchTerm("b"), MatchTerm("a"))
    assert and_then_a_b.fingerprint() == and_then_a_b_marked.fingerprint()
    assert and_then_a_b.fingerprint() != and_then_b_a.fingerprint()
