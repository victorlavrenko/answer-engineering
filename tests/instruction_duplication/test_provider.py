from __future__ import annotations

import math

import httpx
import pytest

from instruction_duplication.models import MODEL_BY_ID
from instruction_duplication.provider import (
    Route,
    estimate_cost,
    estimate_prompt_tokens,
    fake_response,
    headers,
    http_error_payload,
    idempotency_key,
    normalize_routes,
    parse_response,
    rate_limit_source,
    realized_cost,
    reasoning_field,
    request_payload,
    retry_delay,
    route_api_key,
    route_url,
)
from instruction_duplication.records import GenerationCell


def route(model, backend="openrouter"):
    routed_model = model.openrouter_id if backend == "openrouter" else model.hf_id
    return Route(
        backend,
        "Provider",
        routed_model,
        model.input_usd_per_million,
        model.output_usd_per_million,
        4,
        "test",
        True,
    )


def cell(question) -> GenerationCell:
    return GenerationCell(
        cell_id="1" * 64,
        question_id=question.id,
        model_id="test-model",
        condition_id="system_before_after",
        copies=2,
        dataset=question.dataset,
        stem=question.stem,
        choices=question.choices,
        gold=question.gold,
    )


def test_payload_uses_current_provider_specific_token_fields(question):
    model = MODEL_BY_ID["ministral-3-14b-2512"]
    openrouter_payload = request_payload(model, cell(question), route(model))
    assert openrouter_payload["max_completion_tokens"] == 4352
    assert "max_tokens" not in openrouter_payload
    assert openrouter_payload["max_completion_tokens"] < model.provider_output_capability

    huggingface_payload = request_payload(model, cell(question), route(model, "huggingface"))
    assert huggingface_payload["max_tokens"] == 4352
    assert "max_completion_tokens" not in huggingface_payload


def test_cost_reservation_uses_payload_max_and_route_prices(question):
    model = MODEL_BY_ID["gemma-3-12b"]
    selected = route(model)
    payload = request_payload(model, cell(question), selected)
    expected = (
        estimate_prompt_tokens(payload["messages"]) * selected.input_usd_per_million
        + payload["max_completion_tokens"] * selected.output_usd_per_million
    ) / 1_000_000
    assert estimate_cost(selected, payload) == pytest.approx(expected)


def test_idempotency_key_is_stable_across_transport_retries():
    assert idempotency_key("cell") == idempotency_key("cell")
    assert idempotency_key("cell") != idempotency_key("other")


def test_response_schema_validation():
    with pytest.raises(ValueError, match="choices"):
        parse_response({})
    parsed = parse_response(
        {
            "provider": "P",
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.01},
        }
    )
    assert parsed.provider == "P"
    assert parsed.reported_cost_usd == 0.01


def test_retry_jitter_is_deterministic():
    exc = httpx.ReadTimeout("timeout")
    assert retry_delay(exc, 2, "key") == retry_delay(exc, 2, "key")


def test_retry_after_http_date_or_seconds():
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(429, request=request, headers={"Retry-After": "7"})
    exc = httpx.HTTPStatusError("rate", request=request, response=response)
    assert retry_delay(exc, 0, "key") == 7

    response = httpx.Response(
        429,
        request=request,
        headers={"Retry-After": "Sun, 06 Nov 1994 08:49:37 GMT"},
    )
    exc = httpx.HTTPStatusError("rate", request=request, response=response)
    assert 1 <= retry_delay(exc, 0, "key") <= 1.5

    response = httpx.Response(429, request=request, headers={"Retry-After": "not-a-date"})
    exc = httpx.HTTPStatusError("rate", request=request, response=response)
    assert retry_delay(exc, 0, "key") == retry_delay(exc, 0, "key")


def test_rate_limit_provenance_is_explicit_and_conservative():
    model = MODEL_BY_ID["gemma-3-12b"]
    selected = route(model)
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    upstream = httpx.Response(
        429,
        request=request,
        json={"error": {"metadata": {"provider_name": "DeepInfra"}}},
    )
    payload = http_error_payload(upstream)
    assert payload == {"error": {"metadata": {"provider_name": "DeepInfra"}}}
    assert rate_limit_source(selected, upstream, payload) == "upstream:DeepInfra"

    router = httpx.Response(429, request=request, headers={"X-OpenRouter-Error": "rate-limit"})
    assert rate_limit_source(selected, router, None) == "openrouter"

    unknown = httpx.Response(429, request=request, text="too many requests")
    assert http_error_payload(unknown) is None
    assert rate_limit_source(selected, unknown, None) == "unknown-via-openrouter"


def test_headers_have_configurable_project_attribution(monkeypatch):
    monkeypatch.setenv("INSTRUCTION_DUPLICATION_REFERER", "https://example.test/repo")
    result = headers("openrouter", "secret", idempotency_key="id")
    assert result["HTTP-Referer"] == "https://example.test/repo"
    assert result["Idempotency-Key"] == "id"
    assert all(math.isfinite(value) for value in (1.0,))


def test_route_schema_and_model_set_are_strict(monkeypatch):
    model = MODEL_BY_ID["gemma-3-12b"]
    selected = route(model)
    assert Route.from_dict(selected.to_dict()) == selected
    with pytest.raises(ValueError, match="missing"):
        Route.from_dict({"backend": "openrouter"})
    with pytest.raises(ValueError, match="mismatch"):
        normalize_routes({}, [model])
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    assert route_api_key(selected) == "secret"
    assert route_url(selected).startswith("https://openrouter.ai/")
    with pytest.raises(ValueError, match="no URL"):
        route_url(Route("fake", "fake", "fake", 0, 0, 1, "fake", True))


def test_reasoning_indicators_and_cost_fallback(question):
    model = MODEL_BY_ID["gemma-3-12b"]
    selected = route(model)
    raw = fake_response(cell(question))
    raw["choices"][0]["message"]["reasoning"] = "private"
    parsed = parse_response(raw)
    assert reasoning_field(raw, parsed.content) == "reasoning"
    no_usage = type(parsed)(parsed.content, None, None, None, None, "stop", None)
    assert realized_cost(selected, no_usage, 0.25) == 0.25
    usage = type(parsed)(parsed.content, 100, 50, None, None, "stop", None)
    expected = (
        100 * selected.input_usd_per_million + 50 * selected.output_usd_per_million
    ) / 1_000_000
    assert realized_cost(selected, usage, 0.25) == pytest.approx(expected)
    zero_reported = type(parsed)(parsed.content, 100, 50, 0.0, None, "stop", None)
    assert realized_cost(selected, zero_reported, 0.25) == pytest.approx(expected)


def test_response_rejects_invalid_usage_and_empty_content():
    with pytest.raises(ValueError, match="empty visible"):
        parse_response({"choices": [{"message": {"content": ""}}]})
    with pytest.raises(ValueError, match="negative token"):
        parse_response({"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": -1}})
    with pytest.raises(ValueError, match="invalid cost"):
        parse_response(
            {
                "choices": [{"message": {"content": "x"}}],
                "usage": {"cost": float("nan")},
            }
        )
