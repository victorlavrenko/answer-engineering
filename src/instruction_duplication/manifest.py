"""Versioned experiment manifest construction and validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from . import __version__
from .io_utils import sha256_json
from .json_types import (
    FrozenJsonObject,
    JsonObject,
    freeze_json_object,
    integer_value,
    is_object_sequence,
    json_object,
)
from .lexical import LEXICAL_VERSION
from .models import SMOKE_PROFILE_ID, Model
from .protocol import CONDITIONS, PROTOCOL_HASH

MANIFEST_SCHEMA_VERSION = 2
JUDGE_VERSION = __version__
ANALYSIS_VERSION = __version__


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """Immutable identity of one prepared experiment."""

    schema_version: int
    package_version: str
    created_at: str
    source: FrozenJsonObject
    selection: FrozenJsonObject
    question_ids: tuple[str, ...]
    questions_hash: str
    models: tuple[FrozenJsonObject, ...]
    models_hash: str
    environment_hash: str
    protocol_hash: str
    condition_hash: str
    lexical_version: str
    judge_version: str
    analysis_version: str
    smoke_profile_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source",
            freeze_json_object(self.source, path="manifest.source"),
        )
        object.__setattr__(
            self,
            "selection",
            freeze_json_object(self.selection, path="manifest.selection"),
        )
        object.__setattr__(
            self,
            "models",
            tuple(
                freeze_json_object(model, path=f"manifest.models[{index}]")
                for index, model in enumerate(self.models)
            ),
        )

    def to_dict(self) -> JsonObject:
        """Serialize the manifest."""
        return {
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "created_at": self.created_at,
            "source": json_object(self.source, path="manifest.source"),
            "selection": json_object(self.selection, path="manifest.selection"),
            "question_ids": list(self.question_ids),
            "questions_hash": self.questions_hash,
            "models": [
                json_object(item, path=f"manifest.models[{index}]")
                for index, item in enumerate(self.models)
            ],
            "models_hash": self.models_hash,
            "environment_hash": self.environment_hash,
            "protocol_hash": self.protocol_hash,
            "condition_hash": self.condition_hash,
            "lexical_version": self.lexical_version,
            "judge_version": self.judge_version,
            "analysis_version": self.analysis_version,
            "smoke_profile_id": self.smoke_profile_id,
        }

    @property
    def identity_hash(self) -> str:
        """Return a hash excluding the creation timestamp."""
        value = self.to_dict()
        value.pop("created_at", None)
        return sha256_json(value)

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> ExperimentManifest:
        """Load and reject incompatible manifest schemas."""
        version = integer_value(row.get("schema_version", -1), name="manifest.schema_version")
        if version != MANIFEST_SCHEMA_VERSION:
            raise RuntimeError(
                f"workspace manifest schema {version} is incompatible with "
                f"schema {MANIFEST_SCHEMA_VERSION}; use a new workspace"
            )
        models = row.get("models")
        question_ids = row.get("question_ids")
        if not is_object_sequence(models):
            raise ValueError("manifest models must be a list")
        if not is_object_sequence(question_ids):
            raise ValueError("manifest question_ids must be a list")
        source = freeze_json_object(row["source"], path="$.source")
        selection = freeze_json_object(row["selection"], path="$.selection")
        model_rows = tuple(
            freeze_json_object(item, path=f"$.models[{index}]") for index, item in enumerate(models)
        )
        return cls(
            schema_version=version,
            package_version=str(row["package_version"]),
            created_at=str(row["created_at"]),
            source=source,
            selection=selection,
            question_ids=tuple(str(item) for item in question_ids),
            questions_hash=str(row["questions_hash"]),
            models=model_rows,
            models_hash=str(row["models_hash"]),
            environment_hash=str(row["environment_hash"]),
            protocol_hash=str(row["protocol_hash"]),
            condition_hash=str(row["condition_hash"]),
            lexical_version=str(row["lexical_version"]),
            judge_version=str(row["judge_version"]),
            analysis_version=str(row["analysis_version"]),
            smoke_profile_id=str(row["smoke_profile_id"]),
        )


def build_manifest(
    *,
    source: JsonObject,
    selection: JsonObject,
    questions: Sequence[JsonObject],
    models: Sequence[Model],
    environment: JsonObject,
) -> ExperimentManifest:
    """Build a complete manifest from prepared inputs."""
    question_ids = tuple(str(question["id"]) for question in questions)
    model_rows = tuple(model.to_dict() for model in models)
    return ExperimentManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        package_version=__version__,
        created_at=datetime.now(UTC).isoformat(),
        source=freeze_json_object(source, path="manifest.source"),
        selection=freeze_json_object(selection, path="manifest.selection"),
        question_ids=question_ids,
        questions_hash=sha256_json(list(questions)),
        models=tuple(
            freeze_json_object(model, path=f"manifest.models[{index}]")
            for index, model in enumerate(model_rows)
        ),
        models_hash=sha256_json(list(model_rows)),
        environment_hash=sha256_json(environment),
        protocol_hash=PROTOCOL_HASH,
        condition_hash=sha256_json([condition.to_dict() for condition in CONDITIONS]),
        lexical_version=LEXICAL_VERSION,
        judge_version=JUDGE_VERSION,
        analysis_version=ANALYSIS_VERSION,
        smoke_profile_id=SMOKE_PROFILE_ID,
    )
