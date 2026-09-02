from __future__ import annotations

import asyncio
import types
from pathlib import Path

import httpx
import pytest

from instruction_duplication.json_types import is_object_sequence, object_value
from instruction_duplication.models import MODEL_BY_ID
from instruction_duplication.preflight import (
    PreflightBudget,
    PreflightBudgetExceeded,
    ProbeFailure,
    ProbeResult,
    _probe_route,
    check_routes,
    discover_hf_providers,
    discover_openrouter_providers,
)
from instruction_duplication.provider import Route, fake_response
from instruction_duplication.records import AttemptRecord, GenerationCell
from instruction_duplication.storage import Database
from instruction_duplication.types import AttemptStatus


@pytest.fixture(autouse=True)
def isolate_preflight_tests_from_live_provider_discovery(monkeypatch):
    """Keep route-selection tests offline even when the host exports proxy settings."""

    async def no_live_openrouter_discovery(selected_model, client):
        del selected_model, client
        return []

    monkeypatch.delenv("ALL_PROXY", raising=False)
    monkeypatch.delenv("all_proxy", raising=False)
    monkeypatch.setattr(
        "instruction_duplication.preflight.discover_openrouter_providers",
        no_live_openrouter_discovery,
    )


def test_discovery_uses_current_hub_schema_and_sorts(monkeypatch):
    info = types.SimpleNamespace(
        inference_provider_mapping=[
            types.SimpleNamespace(provider="Zulu", status="live", task="text-generation"),
            types.SimpleNamespace(provider="alpha", status="live", task="conversational"),
            types.SimpleNamespace(provider="dead", status="staging", task="text-generation"),
        ]
    )
    monkeypatch.setattr("instruction_duplication.preflight.model_info", lambda *a, **k: info)
    providers = discover_hf_providers(MODEL_BY_ID["gemma-3-12b"], "token", 1)
    assert providers == ["alpha", "Zulu"]


@pytest.mark.asyncio
async def test_openrouter_discovery_uses_current_endpoint_names():
    model = MODEL_BY_ID["qwen3-30b-a3b-instruct-2507"]

    async def handler(request):
        assert request.url.path.endswith(
            "/api/v1/models/qwen/qwen3-30b-a3b-instruct-2507/endpoints"
        )
        return httpx.Response(
            200,
            request=request,
            json={
                "data": {
                    "endpoints": [
                        {"provider_name": "Weights & Biases", "tag": "wandb"},
                        {"provider_name": "Nebius Token Factory", "tag": "nebius"},
                        {"provider_name": "SiliconFlow", "tag": "siliconflow"},
                    ]
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        providers = await discover_openrouter_providers(model, client)
    assert providers == ["wandb", "nebius", "siliconflow"]


@pytest.mark.asyncio
async def test_auto_preflight_falls_back_from_huggingface_to_openrouter(monkeypatch, question):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    calls = []

    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-token")
    monkeypatch.setattr(
        "instruction_duplication.preflight.discover_hf_providers",
        lambda *args: ["deepinfra"],
    )

    async def openrouter_providers(selected_model, client):
        del selected_model, client
        return ["deepinfra"]

    async def fake_probe(client, selected_model, selected_cell, route, **kwargs):
        del client, selected_model, selected_cell
        calls.append((route.backend, route.provider, kwargs["probe_index"]))
        if route.backend == "huggingface":
            raise ProbeFailure("403 Forbidden", [attempt_record("hf-rejected")])
        return ProbeResult(
            "stable",
            (attempt_record(f"or-{kwargs['probe_index']}"),),
            route.provider,
        )

    monkeypatch.setattr(
        "instruction_duplication.preflight.discover_openrouter_providers",
        openrouter_providers,
    )
    monkeypatch.setattr("instruction_duplication.preflight._probe_route", fake_probe)

    progress = []
    result = await check_routes(
        [model],
        cell,
        backend="auto",
        progress_sink=progress.append,
        load_concurrency=2,
    )
    assert result.routes[model.id].backend == "openrouter"
    assert result.routes[model.id].provider == "deepinfra"
    assert [(event.backend, event.provider, event.status) for event in progress] == [
        ("huggingface", None, "discovering"),
        ("huggingface", "deepinfra", "probing"),
        ("huggingface", "deepinfra", "rejected"),
        ("openrouter", None, "discovering"),
        ("openrouter", "deepinfra", "probing"),
        ("openrouter", "deepinfra", "selected"),
    ]
    assert calls == [
        ("huggingface", "deepinfra", 1),
        ("openrouter", "deepinfra", 1),
        ("openrouter", "deepinfra", 2),
        ("openrouter", "deepinfra", 3),
        ("openrouter", "deepinfra", 4),
    ]


@pytest.mark.asyncio
async def test_auto_preflight_uses_model_openrouter_preference_first(monkeypatch, question):
    model = MODEL_BY_ID["llama-3.3-70b-instruct"]
    cell = probe_cell(question)
    calls = []

    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-token")

    async def openrouter_providers(selected_model, client):
        del selected_model, client
        return ["groq"]

    async def fake_probe(client, selected_model, selected_cell, route, **kwargs):
        del client, selected_model, selected_cell
        calls.append((route.backend, route.provider, kwargs["probe_index"]))
        return ProbeResult(
            "stable",
            (attempt_record(f"or-{kwargs['probe_index']}"),),
            route.provider,
        )

    monkeypatch.setattr(
        "instruction_duplication.preflight.discover_openrouter_providers",
        openrouter_providers,
    )
    monkeypatch.setattr("instruction_duplication.preflight._probe_route", fake_probe)

    progress = []
    result = await check_routes(
        [model],
        cell,
        backend="auto",
        progress_sink=progress.append,
        load_concurrency=2,
    )
    assert result.routes[model.id].backend == "openrouter"
    assert result.routes[model.id].provider == "groq"
    assert calls == [
        ("openrouter", "groq", 1),
        ("openrouter", "groq", 2),
        ("openrouter", "groq", 3),
        ("openrouter", "groq", 4),
    ]
    assert progress[0].backend == "openrouter"
    assert result.provenance["backend_order"][model.id] == ("openrouter", "huggingface")


@pytest.mark.asyncio
async def test_per_model_backend_override_changes_auto_order(monkeypatch, question):
    model = MODEL_BY_ID["llama-3.3-70b-instruct"]
    cell = probe_cell(question)
    calls = []

    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-token")
    monkeypatch.setattr(
        "instruction_duplication.preflight.discover_hf_providers",
        lambda *args: ["deepinfra"],
    )

    async def fake_probe(client, selected_model, selected_cell, route, **kwargs):
        del client, selected_model, selected_cell
        calls.append((route.backend, route.provider, kwargs["probe_index"]))
        return ProbeResult(
            "stable",
            (attempt_record(f"hf-{kwargs['probe_index']}"),),
            route.provider,
        )

    monkeypatch.setattr("instruction_duplication.preflight._probe_route", fake_probe)
    result = await check_routes(
        [model],
        cell,
        backend="auto",
        model_backends={model.id: "huggingface"},
        load_concurrency=2,
    )
    assert result.routes[model.id].backend == "huggingface"
    assert calls == [
        ("huggingface", "deepinfra", 1),
        ("huggingface", "deepinfra", 2),
        ("huggingface", "deepinfra", 3),
        ("huggingface", "deepinfra", 4),
    ]


@pytest.mark.asyncio
async def test_auto_preflight_reports_missing_openrouter_fallback(monkeypatch, question):
    from instruction_duplication.preflight import PreflightRunError

    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "instruction_duplication.preflight.discover_hf_providers",
        lambda *args: ["deepinfra"],
    )

    async def fake_probe(client, selected_model, selected_cell, route, **kwargs):
        del client, selected_model, selected_cell, route, kwargs
        raise ProbeFailure("403 Forbidden", [attempt_record("hf-rejected")])

    monkeypatch.setattr("instruction_duplication.preflight._probe_route", fake_probe)
    with pytest.raises(PreflightRunError) as captured:
        await check_routes([model], cell, backend="auto")
    models = object_value(captured.value.provenance["models"], name="provenance.models")
    candidates = models[model.id]
    assert is_object_sequence(candidates)
    assert any(
        object_value(row, name="candidate").get("backend") == "openrouter"
        and "OPENROUTER_API_KEY" in str(object_value(row, name="candidate").get("error"))
        for row in candidates
    )


def attempt_record(
    key: str,
    *,
    status: AttemptStatus = AttemptStatus.COMPLETED,
    reported_cost_usd: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> AttemptRecord:
    return AttemptRecord(
        request_key=key,
        phase="preflight",
        cell_id=None,
        model_id="test-model",
        attempt_number=1,
        backend="openrouter",
        provider="test-provider",
        routed_model="test-model",
        status=status,
        idempotency_key="test-key",
        requested_max_tokens=1280,
        reservation_usd=0.01,
        reported_cost_usd=reported_cost_usd,
        accounted_cost_usd=reported_cost_usd or 0.01,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        http_status=200,
        finish_reason="stop",
        latency_seconds=0.1,
        error=None,
        raw_response_json="{}",
        started_at="2026-08-05T00:00:00+00:00",
        completed_at="2026-08-05T00:00:01+00:00",
    )


@pytest.mark.asyncio
async def test_fake_preflight_returns_current_routes(question):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = GenerationCell(
        cell_id="1" * 64,
        question_id=question.id,
        model_id=model.id,
        condition_id="system_before_after",
        copies=2,
        dataset=question.dataset,
        stem=question.stem,
        choices=question.choices,
        gold=question.gold,
    )
    result = await check_routes([model], cell, fake=True)
    route = result.routes[model.id]
    assert route.backend == "fake"
    assert route.determinism_verified is True
    assert result.attempts == ()


def probe_cell(question) -> GenerationCell:
    return GenerationCell(
        cell_id="2" * 64,
        question_id=question.id,
        model_id="test-model",
        condition_id="system_before_after",
        copies=2,
        dataset=question.dataset,
        stem=question.stem,
        choices=question.choices,
        gold=question.gold,
    )


def probe_route(model, provider="DeepInfra"):
    return Route(
        "openrouter",
        provider,
        model.openrouter_id,
        model.input_usd_per_million,
        model.output_usd_per_million,
        model.max_concurrency,
        "test",
        False,
    )


@pytest.mark.asyncio
async def test_probe_route_success_records_cost(question):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    raw = fake_response(cell)
    raw["provider"] = "DeepInfra"

    async def handler(request):
        assert request.headers["Idempotency-Key"]
        assert request.url.path.endswith("chat/completions")
        return httpx.Response(200, request=request, json=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _probe_route(client, model, cell, probe_route(model))
    assert result.content.startswith("1. Facts")
    assert result.returned_provider == "DeepInfra"
    assert len(result.attempts) == 1
    assert result.attempts[0].status is AttemptStatus.COMPLETED
    attempt = result.attempts[0]
    expected_cost = (
        int(attempt.input_tokens or 0) * probe_route(model).input_usd_per_million
        + int(attempt.output_tokens or 0) * probe_route(model).output_usd_per_million
    ) / 1_000_000
    assert attempt.accounted_cost_usd == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_probe_route_accepts_model_noncompliance_as_route_success(question):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    raw = fake_response(cell)
    raw["provider"] = "DeepInfra"
    raw["choices"][0]["message"]["content"] = (
        "<facts>The stem mentions a patient.</facts>\n"
        "<implications>I am not going to choose an option.</implications>"
    )

    async def handler(request):
        return httpx.Response(200, request=request, json=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _probe_route(client, model, cell, probe_route(model))

    assert result.attempts[0].status is AttemptStatus.COMPLETED
    assert "not going to choose" in result.content


@pytest.mark.asyncio
async def test_probe_route_accepts_single_xml_markdown_fence(question):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    raw = fake_response(cell)
    message = raw["choices"][0]["message"]
    message["content"] = "```xml\n" + message["content"] + "\n```"
    raw["provider"] = "DeepInfra"

    async def handler(request):
        return httpx.Response(200, request=request, json=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _probe_route(client, model, cell, probe_route(model))
    assert result.content.startswith("```xml\n1. Facts")
    assert result.attempts[0].status is AttemptStatus.COMPLETED


@pytest.mark.asyncio
async def test_probe_route_accepts_unusual_section_punctuation_when_answer_is_parseable(question):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    raw = fake_response(cell)
    message = raw["choices"][0]["message"]
    message["content"] = message["content"].replace("1. Facts", "1. Facts — R&D evidence:", 1)
    raw["provider"] = "DeepInfra"

    async def handler(request):
        return httpx.Response(200, request=request, json=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _probe_route(client, model, cell, probe_route(model))

    assert result.content.startswith("1. Facts — R&D evidence:")
    assert result.attempts[0].status is AttemptStatus.COMPLETED


@pytest.mark.asyncio
async def test_probe_retries_transient_http_with_same_idempotency(monkeypatch, question):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    raw = fake_response(cell)
    raw["provider"] = "DeepInfra"
    calls = []

    async def handler(request):
        calls.append(request.headers["Idempotency-Key"])
        if len(calls) == 1:
            return httpx.Response(429, request=request, headers={"Retry-After": "0"})
        return httpx.Response(200, request=request, json=raw)

    monkeypatch.setattr("instruction_duplication.preflight.retry_delay", lambda *args: 0.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _probe_route(client, model, cell, probe_route(model), transport_retries=1)
    assert len(result.attempts) == 2
    assert [attempt.status for attempt in result.attempts] == [
        AttemptStatus.HTTP_ERROR,
        AttemptStatus.COMPLETED,
    ]
    assert result.attempts[0].accounted_cost_usd == 0.0
    assert len(set(calls)) == 1


@pytest.mark.asyncio
async def test_probe_default_rejects_sustained_rate_limit_after_three_attempts(
    monkeypatch, question
):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    calls = []

    async def handler(request):
        calls.append(request.headers["Idempotency-Key"])
        return httpx.Response(429, request=request, headers={"Retry-After": "0"})

    monkeypatch.setattr("instruction_duplication.preflight.retry_delay", lambda *args: 0.0)
    budget = PreflightBudget(cap=1.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProbeFailure) as captured:
            await _probe_route(client, model, cell, probe_route(model), budget=budget)

    assert len(captured.value.attempts) == 3
    assert all(attempt.accounted_cost_usd == 0.0 for attempt in captured.value.attempts)
    assert budget.committed == 0.0
    assert budget.reserved == 0.0
    assert len(set(calls)) == 1


@pytest.mark.asyncio
async def test_probe_truncation_is_terminal_and_retains_attempt(question):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    raw = fake_response(cell)
    raw["provider"] = "DeepInfra"
    raw["choices"][0]["finish_reason"] = "length"

    async def handler(request):
        return httpx.Response(200, request=request, json=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProbeFailure) as captured:
            await _probe_route(client, model, cell, probe_route(model), transport_retries=5)
    assert len(captured.value.attempts) == 1
    assert captured.value.attempts[0].status is AttemptStatus.TRUNCATED


@pytest.mark.asyncio
async def test_probe_rejects_unexpected_finish_reason_without_retry(question):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    raw = fake_response(cell)
    raw["provider"] = "DeepInfra"
    raw["choices"][0]["finish_reason"] = "error"

    async def handler(request):
        return httpx.Response(200, request=request, json=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProbeFailure) as captured:
            await _probe_route(client, model, cell, probe_route(model), transport_retries=5)
    assert len(captured.value.attempts) == 1
    assert captured.value.attempts[0].status is AttemptStatus.INVALID_RESPONSE


@pytest.mark.asyncio
async def test_probe_retains_complete_stop_when_reported_usage_exceeds_ceiling(question):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    raw = fake_response(cell)
    raw["provider"] = "DeepInfra"
    raw["usage"]["completion_tokens"] = model.request_output_tokens + 1

    async def handler(request):
        return httpx.Response(200, request=request, json=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _probe_route(client, model, cell, probe_route(model))
    assert result.attempts[0].status is AttemptStatus.COMPLETED
    assert result.attempts[0].output_tokens == model.request_output_tokens + 1


@pytest.mark.asyncio
async def test_probe_rerun_uses_fresh_physical_and_provider_request_ids(tmp_path: Path, question):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    raw = fake_response(cell)
    raw["provider"] = "DeepInfra"
    idempotency_keys: list[str] = []

    async def handler(request):
        idempotency_keys.append(request.headers["Idempotency-Key"])
        return httpx.Response(200, request=request, json=raw)

    with Database(tmp_path / "run.sqlite3") as db:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            first = await _probe_route(
                client,
                model,
                cell,
                probe_route(model),
                probe_index=1,
                attempt_sink=db.record_attempts,
            )
            second = await _probe_route(
                client,
                model,
                cell,
                probe_route(model),
                probe_index=1,
                attempt_sink=db.record_attempts,
            )
        stored = list(db.attempts())

    assert len(stored) == 2
    assert first.attempts[0].request_key != second.attempts[0].request_key
    assert idempotency_keys[0] != idempotency_keys[1]


@pytest.mark.asyncio
async def test_route_selection_preserves_rejected_probe_attempts(monkeypatch, question):
    model = MODEL_BY_ID["llama-3.3-70b-instruct"]
    cell = probe_cell(question)
    rejected = attempt_record("rejected")
    calls = []

    async def fake_probe(client, selected_model, selected_cell, route, **kwargs):
        del client, selected_model, selected_cell
        calls.append((route.provider, kwargs["probe_index"]))
        if route.provider == model.openrouter_providers[0]:
            raise ProbeFailure("bad first route", [rejected])
        return ProbeResult(
            "same",
            (attempt_record(f"ok-{kwargs['probe_index']}"),),
            route.provider,
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "token")
    monkeypatch.setattr("instruction_duplication.preflight._probe_route", fake_probe)
    result = await check_routes([model], cell, backend="openrouter", load_concurrency=2)
    assert result.routes[model.id].provider == model.openrouter_providers[1]
    assert [attempt.request_key for attempt in result.attempts] == [
        "rejected",
        "ok-1",
        "ok-2",
        "ok-3",
        "ok-4",
    ]
    assert calls[:2] == [(model.openrouter_providers[0], 1), (model.openrouter_providers[1], 1)]


@pytest.mark.asyncio
async def test_second_probe_failure_keeps_first_and_second_attempt_once(monkeypatch, question):
    model = MODEL_BY_ID["llama-3.3-70b-instruct"]
    cell = probe_cell(question)

    async def fake_probe(client, selected_model, selected_cell, route, **kwargs):
        del client, selected_model, selected_cell
        probe_index = kwargs["probe_index"]
        if route.provider == model.openrouter_providers[0] and probe_index == 2:
            raise ProbeFailure(
                "second probe failed",
                [attempt_record("first-provider-bad-2")],
            )
        return ProbeResult(
            "stable",
            (attempt_record(f"{route.provider}-{probe_index}"),),
            route.provider,
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "token")
    monkeypatch.setattr("instruction_duplication.preflight._probe_route", fake_probe)
    result = await check_routes([model], cell, backend="openrouter", load_concurrency=2)
    keys = [attempt.request_key for attempt in result.attempts]
    assert keys == [
        f"{model.openrouter_providers[0]}-1",
        "first-provider-bad-2",
        f"{model.openrouter_providers[1]}-1",
        f"{model.openrouter_providers[1]}-2",
        f"{model.openrouter_providers[1]}-3",
        f"{model.openrouter_providers[1]}-4",
    ]


@pytest.mark.asyncio
async def test_capacity_probe_batch_reaches_requested_simultaneous_load(monkeypatch, question):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    raw = fake_response(cell)
    raw["provider"] = "DeepInfra"
    calls = 0
    active = 0
    maximum = 0
    release = asyncio.Event()

    async def handler(request):
        nonlocal calls, active, maximum
        calls += 1
        if calls > 2:
            active += 1
            maximum = max(maximum, active)
            if active == 4:
                release.set()
            try:
                await asyncio.wait_for(release.wait(), timeout=1.0)
            finally:
                active -= 1
        return httpx.Response(200, request=request, json=raw)

    monkeypatch.setenv("OPENROUTER_API_KEY", "token")
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)

    async def client_for(self, route):
        del self, route
        return client

    monkeypatch.setattr(
        "instruction_duplication.preflight.PreflightSession.client_for",
        client_for,
    )
    try:
        result = await check_routes(
            [model],
            cell,
            backend="openrouter",
            load_concurrency=4,
        )
    finally:
        await client.aclose()

    assert maximum == 4
    assert len(result.attempts) == 6
    models = object_value(result.provenance["models"], name="preflight.models")
    candidates = models[model.id]
    assert is_object_sequence(candidates)
    selected = object_value(candidates[-1], name="selected candidate")
    assert selected["capacity_test_concurrency"] == 4
    assert selected["capacity_test_succeeded"] == 4


@pytest.mark.asyncio
async def test_capacity_failure_rejects_candidate_and_tries_next_provider(monkeypatch, question):
    model = MODEL_BY_ID["llama-3.3-70b-instruct"]
    cell = probe_cell(question)

    async def fake_probe(client, selected_model, selected_cell, route, **kwargs):
        del client, selected_model, selected_cell
        probe_index = kwargs["probe_index"]
        record = attempt_record(f"{route.provider}-{probe_index}")
        if route.provider == model.openrouter_providers[0] and probe_index == 3:
            raise ProbeFailure("HTTP 429 during capacity test", [record])
        return ProbeResult("stable", (record,), route.provider)

    monkeypatch.setenv("OPENROUTER_API_KEY", "token")
    monkeypatch.setattr("instruction_duplication.preflight._probe_route", fake_probe)
    result = await check_routes(
        [model],
        cell,
        backend="openrouter",
        load_concurrency=2,
    )

    assert result.routes[model.id].provider == model.openrouter_providers[1]
    models = object_value(result.provenance["models"], name="preflight.models")
    candidates = models[model.id]
    assert is_object_sequence(candidates)
    first = object_value(candidates[0], name="first candidate")
    assert first["result"] == "rejected"
    assert "capacity test failed" in str(first["error"])


@pytest.mark.asyncio
async def test_preflight_budget_blocks_before_dispatch(question):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(500, request=request)

    budget = PreflightBudget(cap=0.000000001)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PreflightBudgetExceeded):
            await _probe_route(client, model, cell, probe_route(model), budget=budget)
    assert calls == 0
    assert budget.committed == 0.0


@pytest.mark.asyncio
async def test_probe_sink_receives_terminal_attempt_before_failure(question):
    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    raw = fake_response(cell)
    raw["provider"] = "DeepInfra"
    raw["choices"][0]["finish_reason"] = "length"
    persisted = []

    async def handler(request):
        return httpx.Response(200, request=request, json=raw)

    def sink(attempts):
        persisted.extend(attempts)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProbeFailure):
            await _probe_route(
                client,
                model,
                cell,
                probe_route(model),
                attempt_sink=sink,
            )
    assert len(persisted) == 1
    assert persisted[0].status is AttemptStatus.TRUNCATED
    assert persisted[0].output_tokens is not None


def test_route_calibration_uses_provider_reported_probe_cost():
    from instruction_duplication.preflight import _calibrate_route

    model = MODEL_BY_ID["gemma-3-12b"]
    candidate = probe_route(model)
    baseline = (
        1000 * candidate.input_usd_per_million + 1000 * candidate.output_usd_per_million
    ) / 1_000_000
    attempts = (
        attempt_record(
            "calibration",
            reported_cost_usd=baseline * 2,
            input_tokens=1000,
            output_tokens=1000,
        ),
    )
    calibrated = _calibrate_route(candidate, attempts)
    assert calibrated.input_usd_per_million == pytest.approx(candidate.input_usd_per_million * 2.2)
    assert calibrated.output_usd_per_million == pytest.approx(
        candidate.output_usd_per_million * 2.2
    )
    assert calibrated.pricing_source.startswith("provider-reported-probe-calibration-v1")


@pytest.mark.asyncio
async def test_total_route_failure_exposes_partial_provenance(monkeypatch, question):
    from instruction_duplication.preflight import PreflightRunError

    model = MODEL_BY_ID["gemma-3-12b"]
    cell = probe_cell(question)
    persisted = []

    async def fake_probe(client, selected_model, selected_cell, route, **kwargs):
        del client, selected_model, selected_cell
        record = attempt_record(f"{route.provider}-{kwargs['probe_index']}")
        kwargs["attempt_sink"]((record,))
        raise ProbeFailure("terminal probe failure", [record])

    monkeypatch.setenv("OPENROUTER_API_KEY", "token")
    monkeypatch.setattr("instruction_duplication.preflight._probe_route", fake_probe)
    with pytest.raises(PreflightRunError) as captured:
        await check_routes(
            [model],
            cell,
            backend="openrouter",
            attempt_sink=lambda rows: persisted.extend(rows),
        )
    assert captured.value.provenance["status"] == "failed"
    models = object_value(captured.value.provenance["models"], name="provenance.models")
    candidates = models[model.id]
    assert is_object_sequence(candidates)
    first = object_value(candidates[0], name="provenance candidate")
    assert first["result"] == "rejected"
    assert persisted
