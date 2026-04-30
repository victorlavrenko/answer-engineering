from __future__ import annotations

from typing import cast

from answer_engineering.engine.proposal.match_tree.match_tree import (
    MatchAll,
    MatchAndThen,
    MatchAny,
    MatchNot,
    MatchTerm,
    MatchTree,
)


def build_guard_expression(
    **kwargs: object,
) -> MatchTree | None:
    required_before_all = _as_terms(kwargs.get("required_before_all"))
    required_before_any = _as_terms(kwargs.get("required_before_any"))
    required_before_incomplete = _as_terms(
        kwargs.get("required_before_incomplete")
    )
    connectors = _as_terms(
        kwargs.get("connectors", kwargs.get("connector_terms"))
    )
    required_after_any = _as_terms(kwargs.get("required_after_any"))
    required_after_all = _as_terms(kwargs.get("required_after_all"))
    ordered = bool(kwargs.get("ordered")) or bool(kwargs.get("require_order"))
    if not ordered and (
        (required_before_all or required_before_any or connectors)
        and (required_after_any or required_after_all)
    ):
        ordered = True

    before_all_terms = tuple(MatchTerm(expr) for expr in required_before_all)
    before_any_terms = tuple(MatchTerm(expr) for expr in required_before_any)
    incomplete_guard = (
        MatchNot(
            MatchAll(
                tuple(MatchTerm(expr) for expr in required_before_incomplete)
            )
        )
        if required_before_incomplete
        else None
    )

    after_children: list[MatchTree] = []
    if required_after_any:
        after_children.append(
            MatchAny(tuple(MatchTerm(expr) for expr in required_after_any))
        )
    after_children.extend(MatchTerm(expr) for expr in required_after_all)

    left = _combine(
        [
            *before_all_terms,
            *([MatchAny(before_any_terms)] if before_any_terms else []),
        ]
    )
    connector = (
        MatchAny(tuple(MatchTerm(expr) for expr in connectors))
        if connectors
        else None
    )
    right = _combine(after_children)

    if ordered:
        ordered_right = _ordered_connector_right(
            connector=connector, right=right
        )
        ordered_requirements: list[MatchTree] = []
        if ordered_right is not None:
            ordered_requirements.extend(
                MatchAndThen(term, ordered_right) for term in before_all_terms
            )
            if before_any_terms:
                ordered_requirements.append(
                    MatchAny(
                        tuple(
                            MatchAndThen(term, ordered_right)
                            for term in before_any_terms
                        )
                    )
                )
            if not ordered_requirements:
                ordered_requirements.append(ordered_right)
        return _combine(
            [
                node
                for node in (incomplete_guard, _combine(ordered_requirements))
                if node is not None
            ]
        )

    return _combine(
        [
            node
            for node in (incomplete_guard, left, connector, right)
            if node is not None
        ]
    )


def _combine(nodes: list[MatchTree]) -> MatchTree | None:
    if not nodes:
        return None
    if len(nodes) == 1:
        return nodes[0]
    return MatchAll(tuple(nodes))


def _ordered_connector_right(
    *, connector: MatchTree | None, right: MatchTree | None
) -> MatchTree | None:
    if connector is None:
        return right
    if right is None:
        return connector
    return MatchAndThen(connector, right)


def _as_terms(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, tuple):
        out: list[str] = []
        for item in cast(tuple[object, ...], value):
            if isinstance(item, str):
                out.append(item)
        return tuple(out)
    if isinstance(value, list):
        out: list[str] = []
        for item in cast(list[object], value):
            if isinstance(item, str):
                out.append(item)
        return tuple(out)
    return ()
