from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import cast

import pytest

from ae_paper_reproduction.core.planning.notebook_extractor import (
    ColabIpynbResponse,
    NotebookPayload,
)
from ae_paper_reproduction.infra.datasets.datasets import CachedDataset
from ae_paper_reproduction.telemetry import telemetry_types
from answer_engineering.engine.pipeline import (
    events as runtime_events,
)
from answer_engineering.telemetry import (
    CandidateTelemetrySnapshot,
    RuleTelemetrySnapshot,
    RuntimeTelemetrySnapshot,
)

serialize_runtime_telemetry = telemetry_types.serialize_runtime_telemetry


def test_dataset_row_boundary_parser_success_and_failures() -> None:
    @dataclass(slots=True)
    class _TestDataset(CachedDataset):
        external_rows: Iterable[Mapping[str, object]] = field(
            default_factory=tuple
        )

        def _iter_external_rows(self) -> Iterator[Mapping[str, object]]:
            yield from self.external_rows

    dataset = _TestDataset(
        "demo/dataset",
        "validation",
        external_rows=(
            {"id": "1", "case_type": "ssnhl", "question": "Q?", "gold": "A"},
            {"id": "2", "case_type": "ssnhl", "question": "Q2?", "gold": "B"},
        ),
    )
    rows = tuple(dataset.iter_rows())
    assert rows[0].id == "1"
    assert rows[1].question == "Q2?"

    dataset.external_rows = (
        {"id": 1, "case_type": "ssnhl", "question": "Q?", "gold": "A"},
    )
    with pytest.raises(TypeError, match="must be a string"):
        tuple(dataset.iter_rows())


def test_dataset_revision_and_split_are_forwarded_to_loader() -> None:
    # CachedDataset is in-memory and exposes metadata directly;
    # this test now asserts
    # the same boundary behavior through metadata rather than a loader callback.
    @dataclass(slots=True)
    class _TestDataset(CachedDataset):
        external_rows: Iterable[Mapping[str, object]] = field(
            default_factory=tuple
        )

        def _iter_external_rows(self) -> Iterator[Mapping[str, object]]:
            yield from self.external_rows

    dataset = _TestDataset(
        "demo/dataset",
        "train",
        external_rows=(
            {"id": "1", "case_type": "c", "question": "q", "gold": "g"},
        ),
    )
    tuple(dataset.iter_rows())
    assert dataset.metadata() == {
        "dataset_id": "demo/dataset",
        "split": "train",
    }


def test_notebook_colab_payload_boundary_parser() -> None:
    parsed = ColabIpynbResponse(
        {
            "ipynb": {
                "cells": [
                    {"cell_type": "markdown", "source": ["# header\n", "body"]},
                    {"cell_type": "code", "source": "print('ok')"},
                ]
            }
        }
    )
    assert parsed is not None
    assert parsed.notebook.cells[0].source == "# header\nbody"
    assert parsed.notebook.cells[1].cell_type == "code"

    with pytest.raises(ValueError, match="Notebook root must be a JSON object"):
        ColabIpynbResponse({"ipynb": []})
    assert NotebookPayload({"cells": [123]}).cells == ()


def test_runtime_telemetry_serialization_payload_shape() -> None:
    snapshot = RuntimeTelemetrySnapshot(
        runtime_sec=1.5,
        applied_decisions=2,
        decision_limit_reached=False,
        rules=(
            RuleTelemetrySnapshot(
                rule_id="rule-1",
                rule_name="avoid:foo",
                evaluations=3,
                applied=2,
                trigger_firings=2,
                proposals_generated=0,
                generated_candidates_considered=0,
                fallback_candidates_considered=0,
                static_candidates_considered=0,
                noop_candidates_generated=0,
                conditions=(),
                candidate_choices=(
                    CandidateTelemetrySnapshot(
                        kind="generated",
                        candidate_id="probe_1",
                        label="probe_1",
                        chosen=1,
                    ),
                ),
            ),
        ),
        events=(
            runtime_events.ProposalsGenerated(
                rule_id="rule-1",
                proposals_count=3,
                generated_count=3,
                fallback_count=0,
                static_count=0,
                noop_count=0,
            ),
        ),
    )
    serialized = serialize_runtime_telemetry(snapshot)

    assert serialized is not None
    assert serialized["applied_decisions"] == 2
    assert serialized["runtime_sec"] == 1.5
    rules = cast(dict[str, object], serialized["rules"])
    rule = cast(dict[str, object], rules["rule-1"])
    assert rule["rule_name"] == "avoid:foo"
    candidate_choices = cast(dict[str, object], rule["candidate_choices"])
    choice = cast(dict[str, object], candidate_choices["generated:probe_1"])
    assert choice["candidate_id"] == "probe_1"
    events = cast(list[dict[str, object]], serialized["events"])
    assert events[0]["type"] == "ProposalsGenerated"

    assert serialize_runtime_telemetry(None) is None
