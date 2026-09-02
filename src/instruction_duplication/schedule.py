"""Deterministic pseudo-random generation schedule."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

from .io_utils import sha256_json
from .json_types import JsonObject, json_object
from .protocol import CONDITIONS
from .types import Question

SCHEDULE_VERSION = "sha256-balanced-cell-order-v1"


class CellIdFactory(Protocol):
    def __call__(self, question_id: str, model_id: str, condition_id: str) -> str: ...


def schedule_key(cell_id: str) -> str:
    """Return a stable opaque ordering key independent of condition position."""
    payload = f"{SCHEDULE_VERSION}\0{cell_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def build_generation_schedule(
    questions: Sequence[Question],
    model_ids: Sequence[str],
    *,
    cell_id: CellIdFactory,
) -> JsonObject:
    """Freeze the exact cell order used within each model's fair worker queue."""
    rows: list[JsonObject] = []
    for question in questions:
        for model_id in model_ids:
            for condition in CONDITIONS:
                identifier = str(cell_id(question.id, model_id, condition.id))
                rows.append(
                    {
                        "cell_id": identifier,
                        "question_id": question.id,
                        "model_id": model_id,
                        "condition_id": condition.id,
                        "schedule_key": schedule_key(identifier),
                    }
                )
    rows.sort(key=lambda row: str(row["schedule_key"]))
    for index, row in enumerate(rows):
        row["schedule_index"] = index
    schedule_hash = sha256_json(rows)
    return json_object(
        {
            "schedule_version": SCHEDULE_VERSION,
            "assignment": (
                "cells are ordered by SHA-256 of the versioned cell id; fair per-model workers "
                "preserve this randomized within-model order"
            ),
            "cell_count": len(rows),
            "schedule_hash": schedule_hash,
            "cells": rows,
        },
        path="generation schedule",
    )
