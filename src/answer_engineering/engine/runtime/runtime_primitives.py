"""Shared runtime primitive types for patch operations.

Purpose:
    Define low-level operation enums and span aliases used by patch proposals
    and patch-application logic.

Architectural role:
    Foundation types imported by runtime, proposal, and orchestration modules to
    keep patch semantics consistent.

Inputs:
    Referenced by proposal builders and parser/compiler integration when
    constructing operation intents.

Outputs:
    Consumed by patch application, conflict handling, and telemetry formatting.

"""

from __future__ import annotations

from enum import StrEnum

SpanAbs = tuple[int, int]


class PatchOp(StrEnum):
    """Canonical edit operation vocabulary shared by proposal and patch."""

    NOOP = "noop"
    REPLACE = "replace"
    INSERT_BEFORE = "insert_before"
    INSERT_AFTER = "insert_after"
    DELETE = "delete"
