from __future__ import annotations

from answer_engineering.engine.pipeline.events import (
    GuardConditionEvaluated,
)


def test_guard_condition_event_serialization_omits_legacy_field_names() -> None:
    payload = GuardConditionEvaluated(
        rule_id="r1",
        node_id="n1",
        node_path="guard.prefix",
        node_type="all",
        marker="prefix_all",
        debug_expression="x",
        matched=True,
        spans=((0, 1),),
    ).serialize()

    banned = {
        "condition_id",
        "section",
        "operator",
        "expression",
        "event.condition_id",
        "event.section",
        "event.operator",
        "event.expression",
    }
    for token in banned:
        assert token not in payload
    assert payload["debug_expression"] == "x"
    assert payload["node_path"] == "guard.prefix"
