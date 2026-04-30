"""Define canonical row types for reproduction datasets.

Purpose:
    Normalize external dataset rows into the small internal schema used by
    planning and evaluation so downstream code can rely on stable field names
    and types.

Architectural role:
    Boundary module between external dataset adapters and reproduction-domain
    evaluation code.

Inputs (architectural provenance):
    Consumes raw mapping-like rows provided by dataset adapters.

Outputs (downstream usage):
    Typed dataset rows consumed by subrun planning, task creation, and
    evaluation.

Invariants/constraints:
    All required dataset fields must be present as strings before a row can
    enter reproduction planning.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

type ExternalRow = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DatasetRow:
    """Represent one canonical dataset row used by reproduction planning and.

    Purpose:
        Store the normalized id, case type, question text, and gold expression
        fields expected by the rest of the reproduction pipeline.

    Architectural role:
        Boundary value object between external datasets and internal
        planning/evaluation code.

    Inputs (architectural provenance):
        Constructed from raw dataset rows after field validation.

    Outputs (downstream usage):
        Typed dataset rows consumed by subrun planning and answer evaluation.

    Invariants/constraints:
        All four required fields must be present and typed before a row is
        accepted.

    """

    id: str
    case_type: str
    question: str
    gold: str

    @classmethod
    def from_external_row(
        cls,
        row: ExternalRow,
        *,
        id_field: str,
        case_type_field: str,
        question_field: str,
        gold_field: str,
    ) -> DatasetRow:
        """Build a canonical dataset row from an external mapping.

        Purpose:
            Read the configured field names from one external dataset record,
            validate that they are present as strings, and normalize them into
            the internal row schema.

        Architectural role:
            Factory method at the edge between dataset adapters and the
            reproduction domain.

        Inputs (architectural provenance):
            Consumes one mapping-like external row together with the caller's
            field-name configuration.

        Outputs (downstream usage):
            A validated `DatasetRow` ready for planning and evaluation.

        Invariants/constraints:
            Missing or non-string required fields should fail fast instead of
            propagating ambiguous data downstream.

        """
        return cls(
            id=_require_string_field(row, id_field),
            case_type=_require_string_field(row, case_type_field),
            question=_require_string_field(row, question_field),
            gold=_require_string_field(row, gold_field),
        )


def _require_string_field(row: ExternalRow, field_name: str) -> str:
    """Require that one dataset field exists and is a string.

    Purpose:
        Validate one external row field at the boundary so later planning and
        evaluation code can rely on a typed internal schema.

    Architectural role:
        Private validation helper used during dataset-row normalization.

    Inputs (architectural provenance):
        Consumes one mapping-like row and the name of the required field.

    Outputs (downstream usage):
        A validated string value or a fast failure via exception.

    Invariants/constraints:
        This helper should raise immediately on missing fields or unexpected
        types.

    """
    if field_name not in row:
        msg = f"Dataset row missing required field: {field_name}"
        raise KeyError(msg)

    value = row[field_name]
    if isinstance(value, str):
        return value

    value_type = type(value).__name__
    msg = f"Dataset row field {field_name!r} must be a string, got {value_type}"
    raise TypeError(msg)
