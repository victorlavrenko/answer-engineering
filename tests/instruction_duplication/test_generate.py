from __future__ import annotations

import asyncio
import math
from pathlib import Path

import httpx
import pytest

from instruction_duplication.generate import (
    Budget,
    GenerationProgress,
    GenerationResponseError,
    GenerationRuntime,
    parallelism_summary,
    route_parallelism_summary,
    run_pending,
)
from instruction_duplication.models import MODEL_BY_ID
from instruction_duplication.provider import Route, fake_response
from instruction_duplication.records import GenerationCell
from instruction_duplication.storage import Database
from instruction_duplication.types import AttemptStatus


@pytest.mark.asyncio
async def test_budget_rejects_reservation_that_would_cross_cap():
    budget = Budget(1.0, 0.4)
    lock = asyncio.Lock()
    assert await budget.reserve(0.5, lock)
    assert not await budget.reserve(0.2, lock)
    await budget.reconcile(0.5, 0.45, lock)
    assert budget.committed == pytest.approx(0.85)
    assert budget.reserved == 0


@pytest.mark.asyncio
async def test_budget_keeps_a_realized_overrun_and_blocks_later_dispatch() -> None:
    budget = Budget(1.0, 0.8)
    lock = asyncio.Lock()
    assert await budget.reserve(0.1, lock)
    await budget.reconcile(0.1, 0.3, lock)
    assert budget.committed == pytest.approx(1.1)
    assert not await budget.reserve(0.001, lock)


def test_unexpected_finish_reason_is_a_retryable_provider_response_error() -> None:
    model = MODEL_BY_ID["gemma-3-12b"]
    route = Route("openrouter", "Provider", model.openrouter_id, 0.05, 0.15, 4, "test", True)
    raw = {
        "choices": [{"message": {"content": "visible answer"}, "finish_reason": "error"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    with pytest.raises(GenerationResponseError) as captured:
        GenerationRuntime._validate_result(raw, route, 0.01, model.request_output_tokens)
    assert captured.value.status is AttemptStatus.INVALID_RESPONSE
    assert captured.value.retryable


def test_complete_stop_is_retained_when_provider_usage_exceeds_requested_ceiling() -> None:
    model = MODEL_BY_ID["gemma-3-12b"]
    route = Route("openrouter", "Provider", model.openrouter_id, 0.05, 0.15, 4, "test", True)
    raw = {
        "choices": [{"message": {"content": "visible answer"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": model.request_output_tokens + 1,
        },
    }
    parsed, actual = GenerationRuntime._validate_result(
        raw, route, 0.01, model.request_output_tokens
    )
    assert parsed.finish_reason == "stop"
    assert parsed.output_tokens == model.request_output_tokens + 1
    assert actual > 0


def test_route_parallelism_summary_accounts_for_shared_provider_caps():
    models = [MODEL_BY_ID["gemma-3-12b"], MODEL_BY_ID["llama-4-scout"]]
    routes = {
        model.id: Route("huggingface", "deepinfra", model.hf_id, 0, 0, 3, "test", True)
        for model in models
    }
    effective, model_limits, route_limits = route_parallelism_summary(
        models, routes, concurrency=8, per_model_concurrency=2
    )
    assert model_limits == {model.id: 2 for model in models}
    assert route_limits == {("huggingface", "deepinfra"): 3}
    assert effective == 3


def test_nonfinite_budget_and_concurrency_are_rejected():
    with pytest.raises(ValueError):
        Budget(float("nan"), 0)
    model = MODEL_BY_ID["gemma-3-12b"]
    with pytest.raises(ValueError):
        parallelism_summary([model], 0, 1)


def test_progress_eta_uses_recent_throughput(monkeypatch, tmp_path: Path):
    model = MODEL_BY_ID["gemma-3-12b"]
    route = Route("fake", "fake", model.id, 0, 0, 4, "fake", True)
    emitted: list[GenerationProgress] = []
    now = 1_000.0
    monkeypatch.setattr("instruction_duplication.generate.time.monotonic", lambda: now)
    with Database(tmp_path / "run.sqlite3") as db:
        runtime = GenerationRuntime(
            db=db,
            models={model.id: model},
            routes={model.id: route},
            global_limit=4,
            model_limits={model.id: 4},
            budget=Budget(1, 0),
            timeout_seconds=5,
            fake=True,
            progress_sink=emitted.append,
        )
        runtime.counters["completed"] = 100
        runtime.progress_samples.clear()
        runtime.progress_samples.extend(((900.0, 80), (990.0, 100)))
        throttle = next(iter(runtime.route_throttles.values()))
        throttle.limit = 1
        throttle.active = 1
        circuit = next(iter(runtime.circuits.values()))
        circuit.retry_after = now + 30
        event = runtime._progress_event(200, now)
        assert event.rate_per_second == pytest.approx(0.2)
        assert event.eta_seconds == pytest.approx(500)
        assert len(event.bottlenecks) == 1
        bottleneck = event.bottlenecks[0]
        assert bottleneck.model_id == model.id
        assert bottleneck.provider == "fake"
        assert bottleneck.current_limit == 1
        assert bottleneck.unconstrained_limit == 4
        assert bottleneck.active == 1
        assert bottleneck.cooldown_seconds == pytest.approx(30)

        throttle.active = 0
        circuit.retry_after = now
        assert runtime._progress_event(200, now).bottlenecks == ()

        circuit.retry_after = now + 30
        cooldown_only = runtime._progress_event(200, now).bottlenecks
        assert len(cooldown_only) == 1
        assert cooldown_only[0].active == 0
        assert cooldown_only[0].cooldown_seconds == pytest.approx(30)


@pytest.mark.asyncio
async def test_fake_generation_records_one_attempt_per_cell(tmp_path: Path, question):
    model = MODEL_BY_ID["gemma-3-12b"]
    route = Route("fake", "fake", model.id, 0, 0, 4, "fake", True)
    with Database(tmp_path / "run.sqlite3") as db:
        db.prepare([question], [model.id])
        summary = await run_pending(
            db,
            [model],
            {model.id: route},
            concurrency=4,
            per_model_concurrency=4,
            max_cost=1,
            timeout_seconds=5,
            retries=5,
            fake=True,
        )
        assert summary["completed"] == 8
        assert len(list(db.attempts())) == 8
        assert db.counts() == {"completed": 8}


class AlwaysTruncateClient:
    def __init__(self) -> None:
        self.calls = 0

    async def post(self, *args: object, **kwargs: object) -> httpx.Response:
        self.calls += 1
        payload = kwargs["json"]
        assert isinstance(payload, dict)
        ceiling = payload["max_completion_tokens"]
        assert isinstance(ceiling, int)
        request = httpx.Request("POST", "https://example.test")
        assert ceiling == MODEL_BY_ID["gemma-3-12b"].request_output_tokens
        finish_reason = "length"
        return httpx.Response(
            200,
            request=request,
            json={
                "provider": "Provider",
                "choices": [
                    {
                        "message": {"content": "partial"},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 100, "cost": 0.0001},
            },
        )

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_truncation_is_terminal_and_is_not_conditionally_resampled(
    monkeypatch, tmp_path: Path, question
):
    model = MODEL_BY_ID["gemma-3-12b"]
    route = Route(
        "openrouter",
        "Provider",
        model.openrouter_id,
        0.05,
        0.15,
        4,
        "test",
        True,
    )
    client = AlwaysTruncateClient()

    monkeypatch.setattr(
        "instruction_duplication.generate.httpx.AsyncClient",
        lambda *args, **kwargs: client,
    )
    monkeypatch.setattr("instruction_duplication.generate.route_api_key", lambda route: "key")
    monkeypatch.setattr(
        "instruction_duplication.generate.route_url",
        lambda route: "https://example.test",
    )
    with Database(tmp_path / "run.sqlite3") as db:
        db.prepare([question], [model.id])
        summary = await run_pending(
            db,
            [model],
            {model.id: route},
            concurrency=2,
            per_model_concurrency=2,
            max_cost=10,
            timeout_seconds=5,
            retries=3,
        )
        assert summary["completed"] == 0
        assert summary["failed"] == 8
        attempts = list(db.attempts())
        assert len(attempts) == 8
        assert {attempt["status"] for attempt in attempts} == {"truncated"}
        assert {int(attempt["requested_max_tokens"]) for attempt in attempts} == {
            model.request_output_tokens
        }
        assert db.counts() == {"truncated": 8}
        assert db.spend() == pytest.approx(0.0008)
        assert client.calls == 8


@pytest.mark.asyncio
async def test_budget_blocking_does_not_dispatch(monkeypatch, tmp_path: Path, question):
    model = MODEL_BY_ID["mistral-large-3-2512"]
    route = Route(
        "openrouter",
        "Provider",
        model.openrouter_id,
        1_000_000,
        1_000_000,
        2,
        "test",
        True,
    )
    called = False

    class NeverClient:
        def __init__(self, *args, **kwargs):
            nonlocal called
            called = True

    monkeypatch.setattr("instruction_duplication.generate.httpx.AsyncClient", NeverClient)
    with Database(tmp_path / "run.sqlite3") as db:
        db.prepare([question], [model.id])
        summary = await run_pending(
            db,
            [model],
            {model.id: route},
            concurrency=2,
            per_model_concurrency=2,
            max_cost=0.01,
            timeout_seconds=5,
            retries=0,
        )
        assert summary["budget_blocked"] == 8
        assert not called
        assert db.counts() == {"budget_blocked": 8}
        assert math.isfinite(float(summary["spend"]))


class TimeoutThenSuccessClient:
    def __init__(self, raw):
        self.raw = raw
        self.calls: dict[str, int] = {}

    async def post(self, *args, **kwargs):
        key = kwargs["headers"]["Idempotency-Key"]
        self.calls[key] = self.calls.get(key, 0) + 1
        if self.calls[key] == 1:
            raise httpx.ReadTimeout("uncertain first attempt")
        request = httpx.Request("POST", "https://example.test")
        return httpx.Response(200, request=request, json=self.raw)

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_timeout_attempts_are_accounted_and_retried_idempotently(
    monkeypatch, tmp_path: Path, question
):
    model = MODEL_BY_ID["gemma-3-12b"]
    route = Route("openrouter", "DeepInfra", model.openrouter_id, 0.05, 0.15, 4, "test", True)
    raw = fake_response(
        GenerationCell(
            cell_id="f" * 64,
            question_id=question.id,
            model_id=model.id,
            condition_id="system",
            copies=1,
            dataset=question.dataset,
            stem=question.stem,
            choices=question.choices,
            gold=question.gold,
        )
    )
    raw["provider"] = "DeepInfra"
    client = TimeoutThenSuccessClient(raw)
    monkeypatch.setattr(
        "instruction_duplication.generate.httpx.AsyncClient",
        lambda *args, **kwargs: client,
    )
    monkeypatch.setattr("instruction_duplication.generate.route_api_key", lambda route: "key")
    monkeypatch.setattr(
        "instruction_duplication.generate.route_url",
        lambda route: "https://example.test",
    )
    monkeypatch.setattr("instruction_duplication.generate.retry_delay", lambda *args: 0.0)
    with Database(tmp_path / "run.sqlite3") as db:
        db.prepare([question], [model.id])
        summary = await run_pending(
            db,
            [model],
            {model.id: route},
            concurrency=2,
            per_model_concurrency=2,
            max_cost=10,
            timeout_seconds=5,
            retries=1,
        )
        attempts = list(db.attempts())
        assert summary["completed"] == 8
        assert len(attempts) == 16
        assert {attempt["status"] for attempt in attempts} == {"timeout", "completed"}
        by_cell: dict[str, set[str]] = {}
        for attempt in attempts:
            cell_id = attempt["cell_id"]
            assert cell_id is not None
            by_cell.setdefault(cell_id, set()).add(attempt["idempotency_key"])
        assert all(len(keys) == 1 for keys in by_cell.values())
        assert db.spend() > 0


class RateLimitThenSuccessClient:
    def __init__(self, raw: dict[str, object]) -> None:
        self.raw = raw
        self.calls: dict[str, int] = {}
        self.rate_limits_remaining = 1

    async def post(self, *args: object, **kwargs: object) -> httpx.Response:
        headers = kwargs["headers"]
        assert isinstance(headers, dict)
        key = headers["Idempotency-Key"]
        assert isinstance(key, str)
        self.calls[key] = self.calls.get(key, 0) + 1
        request = httpx.Request("POST", "https://example.test")
        if self.rate_limits_remaining:
            self.rate_limits_remaining -= 1
            return httpx.Response(429, request=request, headers={"Retry-After": "0"})
        return httpx.Response(200, request=request, json=self.raw)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_rate_limits_are_requeued_without_counting_generation_spend(
    monkeypatch, tmp_path: Path, question, caplog
):
    model = MODEL_BY_ID["gemma-3-12b"]
    route = Route(
        "openrouter",
        "DeepInfra",
        model.openrouter_id,
        0.05,
        0.15,
        4,
        "test",
        True,
    )
    raw = fake_response(
        GenerationCell(
            cell_id="e" * 64,
            question_id=question.id,
            model_id=model.id,
            condition_id="system",
            copies=1,
            dataset=question.dataset,
            stem=question.stem,
            choices=question.choices,
            gold=question.gold,
        )
    )
    raw["provider"] = "DeepInfra"
    client = RateLimitThenSuccessClient(raw)
    monkeypatch.setattr(
        "instruction_duplication.generate.httpx.AsyncClient",
        lambda *args, **kwargs: client,
    )
    monkeypatch.setattr("instruction_duplication.generate.route_api_key", lambda route: "key")
    monkeypatch.setattr(
        "instruction_duplication.generate.route_url",
        lambda route: "https://example.test",
    )
    monkeypatch.setattr("instruction_duplication.generate.retry_delay", lambda *args: 0.0)

    with Database(tmp_path / "run.sqlite3") as db:
        db.prepare([question], [model.id])
        summary = await run_pending(
            db,
            [model],
            {model.id: route},
            concurrency=4,
            per_model_concurrency=4,
            max_cost=10,
            timeout_seconds=5,
            retries=1,
        )
        attempts = list(db.attempts())
        assert summary["completed"] == 8
        assert len(attempts) == 9
        rate_limited = [attempt for attempt in attempts if attempt["http_status"] == 429]
        assert len(rate_limited) == 1
        assert {attempt["accounted_cost_usd"] for attempt in rate_limited} == {0.0}
        by_cell: dict[str, set[str]] = {}
        for attempt in attempts:
            cell_id = attempt["cell_id"]
            assert isinstance(cell_id, str)
            by_cell.setdefault(cell_id, set()).add(str(attempt["idempotency_key"]))
        assert all(len(keys) == 1 for keys in by_cell.values())
        warnings = [record.message for record in caplog.records if record.levelno >= 30]
        assert any(
            f"{model.id} via openrouter/DeepInfra" in message and "HTTP 429" in message
            for message in warnings
        )


class BurstRateLimitedClient:
    def __init__(self) -> None:
        self.calls = 0

    async def post(self, *args: object, **kwargs: object) -> httpx.Response:
        self.calls += 1
        request = httpx.Request("POST", "https://example.test")
        if self.calls <= 4:
            return httpx.Response(
                429,
                request=request,
                json={
                    "error": {
                        "message": "upstream capacity unavailable",
                        "metadata": {"provider_name": "Mistral"},
                    }
                },
            )
        return httpx.Response(
            200,
            request=request,
            json={
                "provider": "Mistral",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            },
        )

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_concentrated_rate_limit_burst_recovers_without_aborting_run(
    monkeypatch, tmp_path: Path, question
):
    model = MODEL_BY_ID["mistral-large-3-2512"]
    route = Route(
        "openrouter",
        "mistral",
        model.openrouter_id,
        0.5,
        1.5,
        4,
        "test",
        True,
    )
    client = BurstRateLimitedClient()
    monkeypatch.setattr(
        "instruction_duplication.generate.httpx.AsyncClient",
        lambda *args, **kwargs: client,
    )
    monkeypatch.setattr("instruction_duplication.generate.route_api_key", lambda route: "key")
    monkeypatch.setattr(
        "instruction_duplication.generate.route_url",
        lambda route: "https://example.test",
    )
    monkeypatch.setattr("instruction_duplication.generate.retry_delay", lambda *args: 0.0)

    with Database(tmp_path / "run.sqlite3") as db:
        db.prepare([question], [model.id])
        summary = await run_pending(
            db,
            [model],
            {model.id: route},
            concurrency=4,
            per_model_concurrency=4,
            max_cost=10,
            timeout_seconds=5,
            retries=3,
        )
        attempts = list(db.attempts())
        counts = db.counts()

    rate_limited = [attempt for attempt in attempts if attempt["http_status"] == 429]
    assert summary["completed"] == 8
    assert summary["failed"] == 0
    assert counts == {"completed": 8}
    assert client.calls == 12
    assert len(rate_limited) == 4
    assert all(attempt["accounted_cost_usd"] == 0.0 for attempt in rate_limited)
    assert all("rate_limit_source=upstream:Mistral" in attempt["error"] for attempt in rate_limited)


class ConcurrencyTracker:
    def __init__(self, target: int):
        self.target = target
        self.active = 0
        self.maximum = 0
        self.release = asyncio.Event()

    async def enter(self) -> None:
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        if self.active >= self.target:
            self.release.set()
        try:
            await asyncio.wait_for(self.release.wait(), timeout=1.0)
        finally:
            self.active -= 1


class CoordinatedClient:
    def __init__(self, tracker: ConcurrencyTracker):
        self.tracker = tracker

    async def post(self, *args: object, **kwargs: object) -> httpx.Response:
        await self.tracker.enter()
        request = httpx.Request("POST", "https://example.test")
        return httpx.Response(
            200,
            request=request,
            json={
                "provider": "test",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            },
        )

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_generation_fills_per_model_concurrency_without_head_of_line_blocking(
    monkeypatch, tmp_path: Path, question
):
    models = [MODEL_BY_ID["gemma-3-12b"], MODEL_BY_ID["llama-3.3-70b-instruct"]]
    routes = {
        models[0].id: Route(
            "openrouter", "provider-a", models[0].openrouter_id, 0, 0, 2, "test", True
        ),
        models[1].id: Route(
            "openrouter", "provider-b", models[1].openrouter_id, 0, 0, 2, "test", True
        ),
    }
    tracker = ConcurrencyTracker(target=4)
    monkeypatch.setattr(
        "instruction_duplication.generate.httpx.AsyncClient",
        lambda *args, **kwargs: CoordinatedClient(tracker),
    )
    monkeypatch.setattr("instruction_duplication.generate.route_api_key", lambda route: "key")
    monkeypatch.setattr(
        "instruction_duplication.generate.route_url",
        lambda route: "https://example.test",
    )

    with Database(tmp_path / "run.sqlite3") as db:
        db.prepare([question], [model.id for model in models])
        summary = await run_pending(
            db,
            models,
            routes,
            concurrency=4,
            per_model_concurrency=2,
            max_cost=1,
            timeout_seconds=5,
            retries=0,
        )

    assert summary["completed"] == 16
    assert tracker.maximum == 4


class BlankNetworkErrorThenSuccessClient:
    """Simulate the empty-message transport failures observed in the 100-question run."""

    def __init__(self, raw: dict[str, object]) -> None:
        self.raw = raw
        self.calls: dict[str, int] = {}

    async def post(self, *args: object, **kwargs: object) -> httpx.Response:
        headers = kwargs["headers"]
        assert isinstance(headers, dict)
        key = headers["Idempotency-Key"]
        assert isinstance(key, str)
        self.calls[key] = self.calls.get(key, 0) + 1
        if self.calls[key] == 1:
            # Some real httpx/anyio network exceptions stringify to "".
            raise httpx.NetworkError("")
        request = httpx.Request("POST", "https://example.test")
        return httpx.Response(200, request=request, json=self.raw)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_blank_message_network_outage_is_persisted_and_retried_without_crash(
    monkeypatch, tmp_path: Path, question
):
    model = MODEL_BY_ID["gemma-3-12b"]
    route = Route("openrouter", "DeepInfra", model.openrouter_id, 0.05, 0.15, 8, "test", True)
    raw = fake_response(
        GenerationCell(
            cell_id="d" * 64,
            question_id=question.id,
            model_id=model.id,
            condition_id="system",
            copies=1,
            dataset=question.dataset,
            stem=question.stem,
            choices=question.choices,
            gold=question.gold,
        )
    )
    raw["provider"] = "DeepInfra"
    client = BlankNetworkErrorThenSuccessClient(raw)
    monkeypatch.setattr(
        "instruction_duplication.generate.httpx.AsyncClient",
        lambda *args, **kwargs: client,
    )
    monkeypatch.setattr("instruction_duplication.generate.route_api_key", lambda route: "key")
    monkeypatch.setattr(
        "instruction_duplication.generate.route_url",
        lambda route: "https://example.test",
    )
    monkeypatch.setattr("instruction_duplication.generate.retry_delay", lambda *args: 0.0)

    with Database(tmp_path / "run.sqlite3") as db:
        db.prepare([question], [model.id])
        summary = await run_pending(
            db,
            [model],
            {model.id: route},
            concurrency=8,
            per_model_concurrency=8,
            max_cost=10,
            timeout_seconds=5,
            retries=1,
        )
        attempts = list(db.attempts())
        counts = db.counts()

    network = [attempt for attempt in attempts if attempt["status"] == "network_error"]
    assert summary["completed"] == 8
    assert summary["failed"] == 0
    assert counts == {"completed": 8}
    assert len(network) == 8
    assert all(attempt["error"] == "NetworkError" for attempt in network)
    assert len(attempts) == 16
