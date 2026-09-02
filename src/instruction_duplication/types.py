"""Validated domain types used by the experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from .json_types import JsonObject, string_mapping


class CellStatus(StrEnum):
    """Lifecycle state for a planned generation cell."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    RETRYABLE = "retryable"
    FAILED = "failed"
    TRUNCATED = "truncated"
    REFUSED = "refused"
    BUDGET_BLOCKED = "budget_blocked"


class AttemptStatus(StrEnum):
    """Terminal status of one provider request attempt."""

    COMPLETED = "completed"
    HTTP_ERROR = "http_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    TRUNCATED = "truncated"
    REFUSED = "refused"
    INVALID_RESPONSE = "invalid_response"
    REASONING_EXPOSED = "reasoning_exposed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Question:
    """A validated, immutable multiple-choice question."""

    id: str
    dataset: str
    stem: str
    choices: Mapping[str, str]
    gold: str
    gold_text: str
    gold_source: str
    gold_raw: str
    source_split: str

    def __post_init__(self) -> None:
        identifier = self.id.strip()
        dataset = self.dataset.strip()
        stem = self.stem.strip()
        split = self.source_split.strip()
        if not identifier or not dataset or not stem or not split:
            raise ValueError("question id, dataset, stem, and source_split must be non-empty")

        normalized_choices: dict[str, str] = {}
        for raw_label, raw_text in self.choices.items():
            label = str(raw_label).strip().upper()
            text = " ".join(str(raw_text).split())
            if len(label) != 1 or not label.isascii() or not label.isalpha():
                raise ValueError(f"invalid choice label: {raw_label!r}")
            if not text:
                raise ValueError(f"choice {label} has empty text")
            if label in normalized_choices:
                raise ValueError(f"duplicate choice label: {label}")
            normalized_choices[label] = text
        if not 2 <= len(normalized_choices) <= 26:
            raise ValueError("a question must have between 2 and 26 choices")
        labels = tuple(normalized_choices)
        expected = tuple(chr(ord("A") + index) for index in range(len(labels)))
        if labels != expected:
            raise ValueError(f"choice labels must be contiguous from A; got {labels!r}")
        folded = [text.casefold() for text in normalized_choices.values()]
        if len(folded) != len(set(folded)):
            raise ValueError("choice texts must be unique")

        gold = self.gold.strip().upper()
        if gold not in normalized_choices:
            raise ValueError(f"gold label {gold!r} is absent from choices")
        if " ".join(self.gold_text.split()) != normalized_choices[gold]:
            raise ValueError("gold_text must equal choices[gold]")

        object.__setattr__(self, "id", identifier)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "stem", stem)
        object.__setattr__(self, "source_split", split)
        object.__setattr__(self, "gold", gold)
        object.__setattr__(self, "gold_text", normalized_choices[gold])
        object.__setattr__(self, "choices", MappingProxyType(normalized_choices))

    def to_dict(self) -> JsonObject:
        """Serialize the question as plain JSON-compatible data."""
        return {
            "id": self.id,
            "dataset": self.dataset,
            "stem": self.stem,
            "choices": dict(self.choices),
            "gold": self.gold,
            "gold_text": self.gold_text,
            "gold_source": self.gold_source,
            "gold_raw": self.gold_raw,
            "source_split": self.source_split,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> Question:
        """Validate and deserialize a normalized question."""
        required = {
            "id",
            "dataset",
            "stem",
            "choices",
            "gold",
            "gold_text",
            "gold_source",
            "gold_raw",
            "source_split",
        }
        missing = required - set(row)
        if missing:
            raise ValueError("question is missing fields: " + ", ".join(sorted(missing)))
        choices = string_mapping(row["choices"], name="question.choices")
        return cls(
            id=str(row["id"]),
            dataset=str(row["dataset"]),
            stem=str(row["stem"]),
            choices=choices,
            gold=str(row["gold"]),
            gold_text=str(row["gold_text"]),
            gold_source=str(row["gold_source"]),
            gold_raw=str(row["gold_raw"]),
            source_split=str(row["source_split"]),
        )
