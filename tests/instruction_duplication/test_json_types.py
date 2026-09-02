from __future__ import annotations

from collections.abc import Mapping

from instruction_duplication.json_types import (
    freeze_json_object,
    json_object,
    object_value,
)
from instruction_duplication.records import AnalysisRow, JudgmentWrite


def test_json_domain_values_are_deeply_immutable() -> None:
    original: dict[str, object] = {"nested": {"values": [1, 2]}}
    frozen = freeze_json_object(original)
    nested_original = original["nested"]
    assert isinstance(nested_original, dict)
    values_original = nested_original["values"]
    assert isinstance(values_original, list)
    values_original.append(3)

    assert json_object(frozen) == {"nested": {"values": [1, 2]}}
    nested = frozen["nested"]
    assert isinstance(nested, Mapping)
    values = nested["values"]
    assert isinstance(values, tuple)


def test_typed_records_freeze_json_at_construction() -> None:
    judgment: dict[str, object] = {"metrics": {"accuracy": 1.0}}
    write = JudgmentWrite("cell", "hash", "2026-08-06T00:00:00+00:00", judgment, "q")
    row = AnalysisRow("q", "model", "before", "completed", "dataset", judgment)
    metrics = judgment["metrics"]
    assert isinstance(metrics, dict)
    metrics["accuracy"] = 0.0

    write_metrics = object_value(write.judgment["metrics"], name="write.metrics")
    assert write_metrics["accuracy"] == 1.0
    assert row.judgment is not None
    row_metrics = object_value(row.judgment["metrics"], name="row.metrics")
    assert row_metrics["accuracy"] == 1.0
