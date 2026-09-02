"""Typed domain records shared by generation, persistence, and analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .json_types import (
    FrozenJsonObject,
    JsonObject,
    freeze_json_object,
    string_mapping,
)
from .types import AttemptStatus, CellStatus

type Phase = Literal["preflight", "generation"]
type Backend = Literal["huggingface", "openrouter", "fake"]


def _nonempty(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _nonnegative(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _optional_nonnegative(value: float | None, name: str) -> float | None:
    return None if value is None else _nonnegative(value, name)


def _optional_token_count(value: int | None, name: str) -> int | None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class GenerationCell:
    """The exact question/model/condition data required by a provider request."""

    cell_id: str
    question_id: str
    model_id: str
    condition_id: str
    copies: int
    dataset: str
    stem: str
    choices: Mapping[str, str]
    gold: str

    def __post_init__(self) -> None:
        normalized_choices = {
            _nonempty(label, "choice label"): _nonempty(text, f"choice {label}")
            for label, text in self.choices.items()
        }
        if len(normalized_choices) < 2:
            raise ValueError("a generation cell must have at least two choices")
        gold = _nonempty(self.gold, "gold")
        if gold not in normalized_choices:
            raise ValueError("gold must identify one of the choices")
        if self.copies < 0:
            raise ValueError("copies must be non-negative")
        object.__setattr__(self, "cell_id", _nonempty(self.cell_id, "cell_id"))
        object.__setattr__(self, "question_id", _nonempty(self.question_id, "question_id"))
        object.__setattr__(self, "model_id", _nonempty(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "condition_id",
            _nonempty(self.condition_id, "condition_id"),
        )
        object.__setattr__(self, "dataset", _nonempty(self.dataset, "dataset"))
        object.__setattr__(self, "stem", _nonempty(self.stem, "stem"))
        object.__setattr__(self, "choices", MappingProxyType(normalized_choices))
        object.__setattr__(self, "gold", gold)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GenerationCell:
        """Validate and deserialize a persisted generation cell."""
        choices = string_mapping(value["choices"], name="cell.choices")
        return cls(
            cell_id=str(value["cell_id"]),
            question_id=str(value["question_id"]),
            model_id=str(value["model_id"]),
            condition_id=str(value["condition_id"]),
            copies=int(str(value["copies"])),
            dataset=str(value["dataset"]),
            stem=str(value["stem"]),
            choices=choices,
            gold=str(value["gold"]),
        )

    def to_json(self) -> JsonObject:
        """Serialize the record as JSON data."""
        return {
            "cell_id": self.cell_id,
            "question_id": self.question_id,
            "model_id": self.model_id,
            "condition_id": self.condition_id,
            "copies": self.copies,
            "dataset": self.dataset,
            "stem": self.stem,
            "choices": dict(self.choices),
            "gold": self.gold,
        }


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One provider request, including failures and conservative cost accounting."""

    request_key: str
    phase: Phase
    cell_id: str | None
    model_id: str
    attempt_number: int
    backend: Backend
    provider: str
    routed_model: str
    status: AttemptStatus
    idempotency_key: str
    requested_max_tokens: int
    reservation_usd: float
    reported_cost_usd: float | None
    accounted_cost_usd: float
    input_tokens: int | None
    output_tokens: int | None
    http_status: int | None
    finish_reason: str | None
    latency_seconds: float
    error: str | None
    raw_response_json: str | None
    started_at: str
    completed_at: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.request_key, "request_key"),
            (self.model_id, "model_id"),
            (self.provider, "provider"),
            (self.routed_model, "routed_model"),
            (self.idempotency_key, "idempotency_key"),
            (self.started_at, "started_at"),
            (self.completed_at, "completed_at"),
        ):
            _nonempty(value, name)
        if self.phase == "generation" and self.cell_id is None:
            raise ValueError("generation attempts require a cell_id")
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be positive")
        if self.requested_max_tokens < 1:
            raise ValueError("requested_max_tokens must be positive")
        _nonnegative(self.reservation_usd, "reservation_usd")
        _optional_nonnegative(self.reported_cost_usd, "reported_cost_usd")
        _nonnegative(self.accounted_cost_usd, "accounted_cost_usd")
        _optional_token_count(self.input_tokens, "input_tokens")
        _optional_token_count(self.output_tokens, "output_tokens")
        _nonnegative(self.latency_seconds, "latency_seconds")

    def sql_values(self) -> tuple[object, ...]:
        """Return values in the canonical attempts-table column order."""
        return (
            self.request_key,
            self.phase,
            self.cell_id,
            self.model_id,
            self.attempt_number,
            self.backend,
            self.provider,
            self.routed_model,
            self.status.value,
            self.idempotency_key,
            self.requested_max_tokens,
            self.reservation_usd,
            self.reported_cost_usd,
            self.accounted_cost_usd,
            self.input_tokens,
            self.output_tokens,
            self.http_status,
            self.finish_reason,
            self.latency_seconds,
            self.error,
            self.raw_response_json,
            self.started_at,
            self.completed_at,
        )


@dataclass(frozen=True, slots=True)
class CellCompletion:
    """Terminal generation state applied together with its attempt record."""

    cell_id: str
    status: CellStatus
    provider: str | None
    content: str | None
    error: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float | None
    raw_response_json: str | None
    completed_at: str

    def __post_init__(self) -> None:
        _nonempty(self.cell_id, "cell_id")
        _nonempty(self.completed_at, "completed_at")
        _optional_token_count(self.input_tokens, "input_tokens")
        _optional_token_count(self.output_tokens, "output_tokens")
        _optional_nonnegative(self.latency_seconds, "latency_seconds")
        if self.status is CellStatus.COMPLETED and not (self.content or "").strip():
            raise ValueError("completed cells require visible content")


@dataclass(frozen=True, slots=True)
class CellStarted:
    """Event clearing stale terminal state before a fresh attempt wave."""

    cell_id: str
    started_at: str


@dataclass(frozen=True, slots=True)
class BudgetBlocked:
    """Event marking a cell undispatched because the cumulative cap was reached."""

    cell_id: str
    completed_at: str


@dataclass(frozen=True, slots=True)
class AttemptFinished:
    """Atomic attempt event with an optional terminal cell update."""

    attempt: AttemptRecord
    final: CellCompletion | None


type GenerationEvent = CellStarted | BudgetBlocked | AttemptFinished


@dataclass(frozen=True, slots=True, init=False)
class JudgmentWrite:
    """One versioned judgment upsert."""

    cell_id: str
    content_hash: str
    judged_at: str
    judgment: FrozenJsonObject
    question_id: str

    def __init__(
        self,
        cell_id: str,
        content_hash: str,
        judged_at: str,
        judgment: object,
        question_id: str,
    ) -> None:
        object.__setattr__(self, "cell_id", _nonempty(cell_id, "cell_id"))
        object.__setattr__(self, "content_hash", _nonempty(content_hash, "content_hash"))
        object.__setattr__(self, "judged_at", _nonempty(judged_at, "judged_at"))
        object.__setattr__(
            self,
            "judgment",
            freeze_json_object(judgment, path="judgment"),
        )
        object.__setattr__(self, "question_id", _nonempty(question_id, "question_id"))


@dataclass(frozen=True, slots=True, init=False)
class AnalysisRow:
    """Compact typed statistical input loaded from SQLite."""

    question_id: str
    model_id: str
    condition_id: str
    status: str
    dataset: str
    judgment: FrozenJsonObject | None

    def __init__(
        self,
        question_id: str,
        model_id: str,
        condition_id: str,
        status: str,
        dataset: str,
        judgment: object | None,
    ) -> None:
        object.__setattr__(self, "question_id", _nonempty(question_id, "question_id"))
        object.__setattr__(self, "model_id", _nonempty(model_id, "model_id"))
        object.__setattr__(
            self,
            "condition_id",
            _nonempty(condition_id, "condition_id"),
        )
        object.__setattr__(self, "status", _nonempty(status, "status"))
        object.__setattr__(self, "dataset", _nonempty(dataset, "dataset"))
        object.__setattr__(
            self,
            "judgment",
            None if judgment is None else freeze_json_object(judgment, path="analysis.judgment"),
        )
