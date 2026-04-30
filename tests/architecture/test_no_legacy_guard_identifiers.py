from __future__ import annotations

from dataclasses import fields

from answer_engineering.engine.pipeline.events import (
    GuardConditionEvaluated,
)
from answer_engineering.rules.compile.plan import (
    GuardSpec,
)


def test_guard_spec_exposes_expression_only() -> None:
    names = {item.name for item in fields(GuardSpec)}
    assert names == {"expression"}


def test_guard_condition_event_has_only_current_schema_fields() -> None:
    names = {item.name for item in fields(GuardConditionEvaluated)}
    assert names.issuperset(
        {
            "rule_id",
            "node_id",
            "node_path",
            "node_type",
            "marker",
            "debug_expression",
            "matched",
        }
    )
    for legacy_name in ("condition_id", "section", "operator", "expression"):
        assert legacy_name not in names
