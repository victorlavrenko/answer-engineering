"""Fixed model panel and smoke-calibrated generation limits."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from .json_types import JsonObject, integer_value, is_object_sequence, number_value

type PreferredBackend = Literal["huggingface", "openrouter"]


@dataclass(frozen=True, slots=True)
class Model:
    """Validated execution and pricing configuration for one model."""

    id: str
    hf_id: str
    openrouter_id: str
    openrouter_providers: tuple[str, ...]
    preferred_backend: PreferredBackend
    input_usd_per_million: float
    output_usd_per_million: float
    max_concurrency: int
    request_output_tokens: int
    provider_output_capability: int
    smoke_observed_max_output_tokens: int

    def __post_init__(self) -> None:
        if not all((self.id.strip(), self.hf_id.strip(), self.openrouter_id.strip())):
            raise ValueError("model identifiers must be non-empty")
        if self.preferred_backend not in {"huggingface", "openrouter"}:
            raise ValueError("preferred_backend must be huggingface or openrouter")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if self.request_output_tokens < self.smoke_observed_max_output_tokens:
            raise ValueError("request_output_tokens must cover the observed smoke-run maximum")
        if self.request_output_tokens > self.provider_output_capability:
            raise ValueError("request_output_tokens exceeds provider capability")
        for value in (self.input_usd_per_million, self.output_usd_per_million):
            if not math.isfinite(value) or value < 0:
                raise ValueError("model prices must be finite and non-negative")

    def to_dict(self) -> JsonObject:
        """Serialize the model configuration."""
        return {
            "id": self.id,
            "hf_id": self.hf_id,
            "openrouter_id": self.openrouter_id,
            "openrouter_providers": list(self.openrouter_providers),
            "preferred_backend": self.preferred_backend,
            "input_usd_per_million": self.input_usd_per_million,
            "output_usd_per_million": self.output_usd_per_million,
            "max_concurrency": self.max_concurrency,
            "request_output_tokens": self.request_output_tokens,
            "provider_output_capability": self.provider_output_capability,
            "smoke_observed_max_output_tokens": self.smoke_observed_max_output_tokens,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> Model:
        """Load only the current versioned model schema."""
        required = {
            "id",
            "hf_id",
            "openrouter_id",
            "openrouter_providers",
            "preferred_backend",
            "input_usd_per_million",
            "output_usd_per_million",
            "max_concurrency",
            "request_output_tokens",
            "provider_output_capability",
            "smoke_observed_max_output_tokens",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(
                "model configuration is from an incompatible package version; missing "
                + ", ".join(sorted(missing))
            )
        providers = row["openrouter_providers"]
        if not is_object_sequence(providers):
            raise ValueError("openrouter_providers must be a list")
        return cls(
            id=str(row["id"]),
            hf_id=str(row["hf_id"]),
            openrouter_id=str(row["openrouter_id"]),
            openrouter_providers=tuple(str(item) for item in providers),
            preferred_backend=_preferred_backend(row["preferred_backend"]),
            input_usd_per_million=number_value(
                row["input_usd_per_million"], name="model.input_usd_per_million"
            ),
            output_usd_per_million=number_value(
                row["output_usd_per_million"], name="model.output_usd_per_million"
            ),
            max_concurrency=integer_value(row["max_concurrency"], name="model.max_concurrency"),
            request_output_tokens=integer_value(
                row["request_output_tokens"], name="model.request_output_tokens"
            ),
            provider_output_capability=integer_value(
                row["provider_output_capability"], name="model.provider_output_capability"
            ),
            smoke_observed_max_output_tokens=integer_value(
                row["smoke_observed_max_output_tokens"],
                name="model.smoke_observed_max_output_tokens",
            ),
        )


def _preferred_backend(value: object) -> PreferredBackend:
    if value == "huggingface":
        return "huggingface"
    if value == "openrouter":
        return "openrouter"
    raise ValueError(f"invalid preferred backend: {value!r}")


# The request ceilings retain the loop-bounding 30-question / 1,680-cell smoke calibration.
# Mistral Large is the one targeted exception: a later 240-cell current-protocol smoke observed
# an ordinary stop completion at 1,663 tokens, so its 1,792-token legacy ceiling is restored to
# the previously validated 2,304-token ceiling. Other ceilings remain bounded rather than being
# raised toward provider maxima. The packaged profile records the supplemental evidence.
# Provider capability is retained as metadata and is not used for budget estimation.
MODELS: tuple[Model, ...] = (
    Model(
        "gemma-3-12b",
        "google/gemma-3-12b-it",
        "google/gemma-3-12b-it",
        ("deepinfra",),
        "huggingface",
        0.05,
        0.15,
        16,
        1280,
        14_000,
        919,
    ),
    Model(
        "llama-3.3-70b-instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/llama-3.3-70b-instruct",
        ("deepinfra", "novita", "crusoe", "groq"),
        "openrouter",
        0.10,
        0.32,
        16,
        1280,
        14_000,
        896,
    ),
    Model(
        "llama-4-scout",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/llama-4-scout",
        ("deepinfra", "groq", "novita"),
        "huggingface",
        0.10,
        0.30,
        16,
        1280,
        14_000,
        962,
    ),
    Model(
        "ministral-3-14b-2512",
        "mistralai/Ministral-3-14B-Instruct-2512",
        "mistralai/ministral-14b-2512",
        ("mistral", "nextbit"),
        "huggingface",
        0.20,
        0.20,
        16,
        4352,
        14_000,
        3375,
    ),
    Model(
        "mistral-large-3-2512",
        "mistralai/Mistral-Large-3-675B-Instruct-2512",
        "mistralai/mistral-large-2512",
        ("mistral",),
        "huggingface",
        0.50,
        1.50,
        16,
        2304,
        14_000,
        1309,
    ),
    Model(
        "qwen3-30b-a3b-instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "qwen/qwen3-30b-a3b-instruct-2507",
        ("wandb", "nebius", "alibaba", "siliconflow"),
        "openrouter",
        0.10,
        0.30,
        16,
        2304,
        14_000,
        1698,
    ),
    Model(
        "qwen3-235b-a22b-instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "qwen/qwen3-235b-a22b-2507",
        ("alibaba", "parasail", "novita", "deepinfra"),
        "openrouter",
        0.10,
        0.62,
        16,
        2816,
        14_000,
        2214,
    ),
)

MODEL_BY_ID = {model.id: model for model in MODELS}
SMOKE_PROFILE_SCHEMA_VERSION = 2
SMOKE_PROFILE_ID = "run.tgz-2026-08-05-plus-2026-08-10-ceiling-fix-v1"


def select_models(ids: Iterable[str] | None = None) -> list[Model]:
    """Return the default panel or a validated subset in caller-specified order."""
    requested = list(ids or ())
    if not requested:
        return list(MODELS)
    if len(requested) != len(set(requested)):
        raise ValueError("model ids must not be repeated")
    unknown = [model_id for model_id in requested if model_id not in MODEL_BY_ID]
    if unknown:
        raise ValueError("unknown model ids: " + ", ".join(unknown))
    return [MODEL_BY_ID[model_id] for model_id in requested]
