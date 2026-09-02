"""Typed request, response, retry, and cost handling for inference providers."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypedDict

import httpx
from huggingface_hub import get_token

from . import __version__
from .json_types import (
    JsonObject,
    boolean_value,
    integer_value,
    is_object_sequence,
    is_string_mapping,
    json_object,
    number_value,
    object_value,
    string_mapping,
)
from .models import Model
from .protocol import ChatMessage, render_messages
from .records import Backend, GenerationCell


class ProviderPreference(TypedDict):
    """OpenRouter provider-routing preference."""

    order: list[str]
    allow_fallbacks: bool


class RequestPayloadBase(TypedDict):
    """Fields shared by the current provider request schemas."""

    model: str
    messages: list[ChatMessage]
    temperature: int
    seed: int


class HuggingFaceRequestPayload(RequestPayloadBase):
    """Current Hugging Face Router chat-completion request payload."""

    max_tokens: int


class OpenRouterRequestPayload(RequestPayloadBase):
    """Current OpenRouter chat-completion request payload."""

    max_completion_tokens: int
    provider: ProviderPreference


class FakeRequestPayload(RequestPayloadBase):
    """Internal deterministic fake-run payload used only for cost planning."""

    max_completion_tokens: int


type RequestPayload = HuggingFaceRequestPayload | OpenRouterRequestPayload | FakeRequestPayload


class FakeMessageRequired(TypedDict):
    """Fields always present in a fake-provider message."""

    content: str


class FakeMessage(FakeMessageRequired, total=False):
    """Mutable fake-provider message with optional reasoning indicators."""

    reasoning: str
    reasoning_content: str
    thinking: str


class FakeChoice(TypedDict):
    """One fake provider choice."""

    message: FakeMessage
    finish_reason: str


class FakeUsage(TypedDict):
    """Fake provider usage data."""

    prompt_tokens: int
    completion_tokens: int
    cost: float


class FakeResponse(TypedDict):
    """Typed mutable fake provider response."""

    id: str
    provider: str
    choices: list[FakeChoice]
    usage: FakeUsage


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HUGGINGFACE_URL = "https://router.huggingface.co/v1/chat/completions"
TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
TRUNCATION_REASONS = {"length", "max_tokens", "max_output_tokens"}
SAFETY_REASONS = {"content_filter", "safety", "blocked"}


def _backend(value: object) -> Backend:
    if value == "huggingface":
        return "huggingface"
    if value == "openrouter":
        return "openrouter"
    if value == "fake":
        return "fake"
    raise ValueError(f"invalid route backend: {value}")


@dataclass(frozen=True, slots=True)
class Route:
    """A fixed provider route with route-specific pricing and capacity."""

    backend: Backend
    provider: str
    model: str
    input_usd_per_million: float
    output_usd_per_million: float
    max_concurrency: int = 8
    pricing_source: str = "model-panel"
    determinism_verified: bool = False

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("route provider and model must be non-empty")
        if self.max_concurrency < 1:
            raise ValueError("route max_concurrency must be positive")
        for price in (self.input_usd_per_million, self.output_usd_per_million):
            if not math.isfinite(price) or price < 0:
                raise ValueError("route prices must be finite and non-negative")

    def to_dict(self) -> JsonObject:
        """Serialize a route."""
        return {
            "backend": self.backend,
            "provider": self.provider,
            "model": self.model,
            "input_usd_per_million": self.input_usd_per_million,
            "output_usd_per_million": self.output_usd_per_million,
            "max_concurrency": self.max_concurrency,
            "pricing_source": self.pricing_source,
            "determinism_verified": self.determinism_verified,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> Route:
        """Load only the current route schema."""
        required = {
            "backend",
            "provider",
            "model",
            "input_usd_per_million",
            "output_usd_per_million",
            "max_concurrency",
            "pricing_source",
            "determinism_verified",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(
                "route data is incompatible with version 2; missing " + ", ".join(sorted(missing))
            )
        return cls(
            backend=_backend(row["backend"]),
            provider=str(row["provider"]),
            model=str(row["model"]),
            input_usd_per_million=number_value(
                row["input_usd_per_million"], name="route.input_usd_per_million"
            ),
            output_usd_per_million=number_value(
                row["output_usd_per_million"], name="route.output_usd_per_million"
            ),
            max_concurrency=integer_value(row["max_concurrency"], name="route.max_concurrency"),
            pricing_source=str(row["pricing_source"]),
            determinism_verified=boolean_value(
                row["determinism_verified"], name="route.determinism_verified"
            ),
        )

    @property
    def label(self) -> str:
        """Return a compact route label."""
        if self.backend == "huggingface":
            return f"Hugging Face / {self.provider}"
        if self.backend == "openrouter":
            return f"OpenRouter / {self.provider}"
        return "fake"


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Validated provider response fields used by the experiment."""

    content: str
    input_tokens: int | None
    output_tokens: int | None
    reported_cost_usd: float | None
    provider: str | None
    finish_reason: str | None
    refusal: str | None


def http_error_payload(response: httpx.Response) -> JsonObject | None:
    """Return a structured provider error body when the response exposes one."""
    try:
        decoded: object = response.json()
    except (ValueError, TypeError):
        return None
    try:
        return json_object(decoded, path="provider error response")
    except ValueError:
        return None


def rate_limit_source(
    route: Route,
    response: httpx.Response,
    payload: Mapping[str, object] | None,
) -> str:
    """Classify a 429 conservatively from explicit response provenance only."""
    if response.status_code != 429:
        raise ValueError("rate-limit provenance requires an HTTP 429 response")
    if route.backend != "openrouter":
        return f"unknown-via-{route.backend}:{route.provider}"

    error = payload.get("error") if payload is not None else None
    error_object: Mapping[str, object]
    metadata: Mapping[str, object] = {}
    message = ""
    try:
        error_object = object_value(error, name="provider error")
    except ValueError:
        error_object = {}
    raw_metadata = error_object.get("metadata")
    try:
        metadata = object_value(raw_metadata, name="provider error metadata")
    except ValueError:
        metadata = {}
    raw_message = error_object.get("message")
    if isinstance(raw_message, str):
        message = raw_message.casefold()

    for key in ("provider_name", "provider", "provider_slug"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return f"upstream:{value.strip()}"

    header_names = {name.casefold() for name in response.headers}
    if any(name.startswith("x-openrouter") for name in header_names) or "openrouter" in message:
        return "openrouter"
    return "unknown-via-openrouter"


def hf_token() -> str | None:
    """Return an HF token from environment variables or the standard login cache."""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or get_token()
    )


def openrouter_token() -> str | None:
    """Return the configured OpenRouter API key."""
    return os.environ.get("OPENROUTER_API_KEY")


def normalize_routes(raw: Mapping[str, object], models: Sequence[Model]) -> dict[str, Route]:
    """Load current-schema routes for every selected model."""
    expected = {model.id for model in models}
    if set(raw) != expected:
        missing = sorted(expected - set(raw))
        extra = sorted(set(raw) - expected)
        raise ValueError(f"route/model mismatch; missing={missing}, extra={extra}")
    return {
        model.id: Route.from_dict(object_value(raw[model.id], name=f"routes.{model.id}"))
        for model in models
    }


def route_url(route: Route) -> str:
    """Return the OpenAI-compatible endpoint URL."""
    if route.backend == "huggingface":
        return HUGGINGFACE_URL
    if route.backend == "openrouter":
        return OPENROUTER_URL
    raise ValueError("fake routes have no URL")


def route_api_key(route: Route) -> str | None:
    """Return the credential for a route."""
    if route.backend == "huggingface":
        return hf_token()
    if route.backend == "openrouter":
        return openrouter_token()
    return "fake"


def headers(
    backend: Backend,
    api_key: str,
    *,
    idempotency_key: str | None = None,
) -> dict[str, str]:
    """Build request headers without project-external attribution."""
    result = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": f"instruction-duplication/{__version__}",
    }
    if idempotency_key:
        result["Idempotency-Key"] = idempotency_key
    if backend == "openrouter":
        result.update(
            {
                "HTTP-Referer": os.environ.get(
                    "INSTRUCTION_DUPLICATION_REFERER",
                    "https://github.com/victorlavrenko/answer-engineering",
                ),
                "X-Title": "Instruction Duplication",
            }
        )
    return result


def request_payload(
    model: Model,
    cell: GenerationCell,
    route: Route,
    *,
    max_output_tokens: int | None = None,
) -> RequestPayload:
    """Build a deterministic request using each provider's current schema."""
    ceiling = model.request_output_tokens if max_output_tokens is None else int(max_output_tokens)
    if not 1 <= ceiling <= model.provider_output_capability:
        raise ValueError("requested output-token limit is outside the model capability")
    payload: RequestPayloadBase = {
        "model": route.model,
        "messages": render_messages(
            cell.stem,
            cell.choices,
            cell.condition_id,
        ),
        "temperature": 0,
        "seed": int(cell.cell_id[:8], 16),
    }
    if route.backend == "openrouter":
        return {
            **payload,
            "max_completion_tokens": ceiling,
            "provider": {"order": [route.provider], "allow_fallbacks": False},
        }
    if route.backend == "huggingface":
        return {**payload, "max_tokens": ceiling}
    if route.backend == "fake":
        return {**payload, "max_completion_tokens": ceiling}
    raise AssertionError(f"unreachable backend: {route.backend}")


def _nonnegative_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid token count: {value!r}") from exc
    if result < 0:
        raise ValueError(f"negative token count: {result}")
    return result


def _nonnegative_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = number_value(value, name="provider cost")
    except ValueError as exc:
        raise ValueError(f"invalid cost: {value!r}") from exc
    if result < 0:
        raise ValueError(f"invalid cost: {result}")
    return result


def parse_response(raw: Mapping[str, object]) -> ProviderResponse:
    """Validate visible text, usage, finish reason, refusal, and cost."""
    choices = raw.get("choices")
    if not is_object_sequence(choices) or not choices:
        raise ValueError("response has no choices array")
    first = object_value(choices[0], name="provider response choices[0]")
    message = object_value(first.get("message"), name="provider response choices[0].message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response has empty visible content")
    usage_value = raw.get("usage")
    usage: Mapping[str, object] = (
        {} if usage_value is None else object_value(usage_value, name="provider response usage")
    )
    provider = raw.get("provider")
    finish = first.get("finish_reason")
    refusal_value = message.get("refusal")
    return ProviderResponse(
        content=content,
        input_tokens=_nonnegative_int(usage.get("prompt_tokens")),
        output_tokens=_nonnegative_int(usage.get("completion_tokens")),
        reported_cost_usd=_nonnegative_float(usage.get("cost")),
        provider=str(provider) if provider else None,
        finish_reason=str(finish).casefold() if finish is not None else None,
        refusal=str(refusal_value) if refusal_value not in (None, "", [], {}) else None,
    )


def reasoning_field(raw: Mapping[str, object], content: str) -> str | None:
    """Return the first exposed reasoning-channel indicator."""
    choices = raw.get("choices")
    if is_object_sequence(choices) and choices:
        first = choices[0]
        if is_string_mapping(first):
            message = first.get("message")
            if is_string_mapping(message):
                for key in ("reasoning", "reasoning_content", "thinking", "reasoning_details"):
                    if message.get(key) not in (None, "", [], {}):
                        return key
    if "<think>" in content.casefold() or "</think>" in content.casefold():
        return "inline think tag"
    usage = raw.get("usage")
    if is_string_mapping(usage):
        details = usage.get("completion_tokens_details")
        if is_string_mapping(details):
            reasoning_tokens = _nonnegative_int(details.get("reasoning_tokens")) or 0
            if reasoning_tokens > 0:
                return "reasoning tokens"
    return None


_HTTP_DATE_FORMATS = (
    "%a, %d %b %Y %H:%M:%S GMT",
    "%A, %d-%b-%y %H:%M:%S GMT",
    "%a %b %d %H:%M:%S %Y",
)


def _parse_http_date(value: str) -> datetime:
    """Parse the three HTTP-date forms accepted by RFC 9110."""
    for date_format in _HTTP_DATE_FORMATS:
        try:
            return datetime.strptime(value, date_format).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"invalid HTTP date: {value!r}")


def retry_delay(exc: Exception, attempt: int, request_key: str) -> float:
    """Return a server delay bounded below by deterministic exponential backoff."""
    digest = hashlib.sha256(f"{request_key}\0{attempt}".encode()).digest()
    jitter = int.from_bytes(digest[:2], "big") / 65535 * 0.5
    backoff = min(60.0, 2.0**attempt + jitter)
    if isinstance(exc, httpx.HTTPStatusError):
        value = exc.response.headers.get("Retry-After")
        if value:
            try:
                server_delay = min(120.0, max(0.0, float(value)))
                return max(backoff, server_delay)
            except ValueError:
                try:
                    retry_at = _parse_http_date(value)
                    server_delay = min(
                        120.0,
                        max(0.0, (retry_at - datetime.now(UTC)).total_seconds()),
                    )
                    return max(backoff, server_delay)
                except ValueError:
                    pass
    return backoff


def estimate_prompt_tokens(messages: Sequence[ChatMessage | Mapping[str, str]]) -> int:
    """Conservatively estimate tokens from the attached smoke-run character ratios."""
    characters = sum(len(message.get("content", "")) for message in messages)
    # Smoke maximum was 0.342 tokens/character. 1/2.75 plus 64 overhead is conservative.
    return max(1, math.ceil(characters / 2.75) + 64)


def _chat_messages(value: object) -> list[dict[str, str]]:
    if not is_object_sequence(value):
        raise ValueError("payload messages must be a sequence")
    messages: list[dict[str, str]] = []
    for index, item in enumerate(value):
        messages.append(string_mapping(item, name=f"payload.messages[{index}]"))
    return messages


def estimate_cost(
    route: Route,
    payload: Mapping[str, object],
    output_tokens: int | None = None,
) -> float:
    """Calculate a route-specific worst-case or realized request cost."""
    messages = _chat_messages(payload.get("messages"))
    input_tokens = estimate_prompt_tokens(messages)
    configured_limit = payload.get("max_completion_tokens", payload.get("max_tokens"))
    if configured_limit is None:
        raise ValueError("payload is missing its provider output-token limit")
    completion_tokens = int(str(configured_limit)) if output_tokens is None else int(output_tokens)
    return (
        input_tokens * route.input_usd_per_million
        + completion_tokens * route.output_usd_per_million
    ) / 1_000_000


def realized_cost(route: Route, response: ProviderResponse, reservation: float) -> float:
    """Conservatively combine provider-reported and route-derived realized cost."""
    derived: float | None = None
    if response.input_tokens is not None or response.output_tokens is not None:
        derived = (
            (response.input_tokens or 0) * route.input_usd_per_million
            + (response.output_tokens or 0) * route.output_usd_per_million
        ) / 1_000_000
    if response.reported_cost_usd is not None and derived is not None:
        return max(response.reported_cost_usd, derived)
    if response.reported_cost_usd is not None:
        return response.reported_cost_usd
    if derived is not None:
        return derived
    return reservation


def idempotency_key(cell_id: str, phase: str = "generation") -> str:
    """Return one stable idempotency key reused across transport retries."""
    return hashlib.sha256(f"instruction-duplication-v3\0{phase}\0{cell_id}".encode()).hexdigest()


def fake_response(cell: GenerationCell) -> FakeResponse:
    """Return a deterministic, format-neutral response for tests."""
    labels = list(cell.choices)
    second = next(label for label in labels if label != cell.gold)
    if cell.condition_id == "zero":
        content = f"Final answer: {cell.gold}."
    else:
        content = "\n".join(
            (
                "1. Facts",
                cell.stem,
                "2. Implications",
                f"The stated findings support {cell.choices[cell.gold]} and argue against alternatives.",
                "3. Provisional answer",
                f"Option {cell.gold}, {cell.choices[cell.gold]}, is provisional because it best fits the facts.",
                "4. Best alternative",
                f"Option {second}, {cell.choices[second]}, is the best alternative but loses on the decisive finding.",
                "5. Decisive distinction",
                cell.stem,
                "6. What would change the answer",
                f"If the decisive finding changed to support {cell.choices[second]}, option {second} would become best.",
                "7. Reconsideration",
                f"Retain option {cell.gold} after checking the stem again; the decisive evidence remains: {cell.stem}",
                "8. Final answer",
                f"Option {cell.gold}: {cell.choices[cell.gold]}",
            )
        )
    return {
        "id": f"fake-{cell.cell_id[:12]}",
        "provider": "fake",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 400,
            "completion_tokens": max(20, len(content) // 4),
            "cost": 0.0,
        },
    }
