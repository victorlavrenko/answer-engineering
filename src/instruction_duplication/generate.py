"""Bounded concurrent generation with strict budgets and attempt provenance."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import httpx

from .io_utils import canonical_json
from .json_types import JsonObject, json_object
from .models import Model
from .provider import (
    SAFETY_REASONS,
    TRANSIENT_STATUS,
    TRUNCATION_REASONS,
    ProviderResponse,
    RequestPayload,
    Route,
    estimate_cost,
    fake_response,
    headers,
    http_error_payload,
    idempotency_key,
    parse_response,
    rate_limit_source,
    realized_cost,
    reasoning_field,
    request_payload,
    retry_delay,
    route_api_key,
    route_url,
)
from .records import (
    AttemptFinished,
    AttemptRecord,
    BudgetBlocked,
    CellCompletion,
    CellStarted,
    GenerationCell,
    GenerationEvent,
)
from .schedule import schedule_key
from .storage import Database
from .types import AttemptStatus, CellStatus

LOGGER = logging.getLogger(__name__)
BUDGET_SAFETY_FACTOR = 1.10
ETA_WINDOW_SECONDS = 120.0
PROGRESS_HEARTBEAT_SECONDS = 30.0
type ProcessDisposition = Literal["finished", "retryable"]


@dataclass(frozen=True, slots=True)
class RouteBottleneck:
    """One exact route whose adaptive limit or cooldown is constraining generation."""

    model_id: str
    backend: str
    provider: str
    routed_model: str
    current_limit: int
    unconstrained_limit: int
    active: int
    cooldown_seconds: float


@dataclass(frozen=True, slots=True)
class GenerationProgress:
    """A throttled snapshot of generation completion counters and route pressure."""

    processed: int
    total: int
    completed: int
    failed: int
    budget_blocked: int
    rate_per_second: float | None
    eta_seconds: float | None
    bottlenecks: tuple[RouteBottleneck, ...]


type GenerationProgressSink = Callable[[GenerationProgress], None]


def utcnow() -> str:
    """Return one timezone-aware UTC timestamp."""
    return dt.datetime.now(dt.UTC).isoformat()


def _positive_finite(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def parallelism_summary(
    models: Sequence[Model], concurrency: int, per_model_concurrency: int
) -> tuple[int, dict[str, int]]:
    if concurrency < 1 or per_model_concurrency < 1:
        raise ValueError("concurrency limits must be positive")
    limits = {model.id: min(per_model_concurrency, model.max_concurrency) for model in models}
    return min(concurrency, sum(limits.values())), limits


def _provider_gate_key(model_id: str, route: Route) -> tuple[str, str]:
    """Return the shared provider-capacity key for one pinned route."""
    if route.backend == "fake":
        return route.backend, model_id
    return route.backend, route.provider


def _route_throttle_key(model_id: str, route: Route) -> tuple[str, str, str]:
    """Return the exact route key used for adaptive throttling."""
    if route.backend == "fake":
        return route.backend, model_id, route.model
    return route.backend, route.provider, route.model


def route_parallelism_summary(
    models: Sequence[Model],
    routes: Mapping[str, Route],
    concurrency: int,
    per_model_concurrency: int,
) -> tuple[int, dict[str, int], dict[tuple[str, str], int]]:
    """Return the real request ceiling after model and shared-route limits."""
    _, model_limits = parallelism_summary(models, concurrency, per_model_concurrency)
    route_demands: dict[tuple[str, str], int] = {}
    route_limits: dict[tuple[str, str], int] = {}
    for model in models:
        route = routes.get(model.id)
        if route is None:
            continue
        key = _provider_gate_key(model.id, route)
        route_demands[key] = route_demands.get(key, 0) + model_limits[model.id]
        previous = route_limits.get(key)
        route_limits[key] = (
            route.max_concurrency if previous is None else min(previous, route.max_concurrency)
        )
    route_capacity = sum(min(route_demands[key], limit) for key, limit in route_limits.items())
    if not route_limits:
        route_capacity = sum(model_limits.values())
    return min(concurrency, route_capacity), model_limits, route_limits


@dataclass(slots=True)
class Budget:
    """Concurrency-safe cumulative cost accounting."""

    cap: float
    committed: float
    reserved: float = 0.0

    def __post_init__(self) -> None:
        _positive_finite(self.cap, "max_cost")
        if self.committed < 0 or not math.isfinite(self.committed):
            raise ValueError("recorded spend is invalid")

    async def reserve(self, amount: float, lock: asyncio.Lock) -> bool:
        """Reserve worst-case request cost without crossing the cap."""
        if amount < 0 or not math.isfinite(amount):
            raise ValueError("reservation must be finite and non-negative")
        async with lock:
            if self.committed + self.reserved + amount > self.cap + 1e-12:
                return False
            self.reserved += amount
            return True

    async def reconcile(self, reservation: float, actual: float, lock: asyncio.Lock) -> None:
        """Commit actual cost; later reservations stop naturally if the cap was exceeded."""
        if actual < 0 or not math.isfinite(actual):
            actual = reservation
        async with lock:
            self.reserved = max(0.0, self.reserved - reservation)
            self.committed += actual


class AsyncDatabaseWriter:
    """Batch SQLite writes on one background thread-facing coroutine."""

    def __init__(self, db: Database, *, batch_size: int = 100, flush_seconds: float = 0.05):
        self.db = db
        self.batch_size = batch_size
        self.flush_seconds = flush_seconds
        self.queue: asyncio.Queue[tuple[GenerationEvent | None, asyncio.Future[None] | None]] = (
            asyncio.Queue()
        )
        self.task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> AsyncDatabaseWriter:
        self.task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.queue.put((None, None))
        if self.task is not None:
            await self.task

    async def submit(self, event: GenerationEvent) -> None:
        """Persist an event and return only after the batch commit succeeds."""
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self.queue.put((event, future))
        await future

    async def _next_batch(
        self,
        first: tuple[GenerationEvent, asyncio.Future[None] | None],
    ) -> tuple[list[tuple[GenerationEvent, asyncio.Future[None] | None]], bool]:
        batch = [first]
        stopping = False
        deadline = asyncio.get_running_loop().time() + self.flush_seconds
        while len(batch) < self.batch_size:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                event, future = await asyncio.wait_for(self.queue.get(), timeout)
            except TimeoutError:
                break
            if event is None:
                stopping = True
                break
            batch.append((event, future))
        return batch, stopping

    async def _run(self) -> None:
        stopping = False
        while not stopping:
            event, future = await self.queue.get()
            if event is None:
                break
            batch, stopping = await self._next_batch((event, future))
            try:
                await asyncio.to_thread(
                    self.db.apply_generation_events,
                    [item for item, _ in batch],
                )
            except Exception as exc:
                for _, pending in batch:
                    if pending is not None and not pending.done():
                        pending.set_exception(exc)
            else:
                for _, pending in batch:
                    if pending is not None and not pending.done():
                        pending.set_result(None)


@dataclass(slots=True)
class ProviderCircuit:
    """Share provider-wide cooldown state without releasing a burst after one success."""

    failures: int = 0
    retry_after: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def wait(self) -> None:
        """Wait until the provider-wide cooldown expires."""
        while True:
            async with self.lock:
                delay = self.retry_after - asyncio.get_running_loop().time()
            if delay <= 0:
                return
            await asyncio.sleep(delay)

    async def success(self) -> None:
        """Decay failure pressure without cancelling an active cooldown."""
        async with self.lock:
            self.failures = max(0, self.failures - 1)

    async def transient_failure(self, delay: float) -> None:
        """Extend the shared provider cooldown after a transient error."""
        async with self.lock:
            self.failures += 1
            adaptive = min(120.0, delay * min(4.0, 1.0 + self.failures / 4))
            self.retry_after = max(
                self.retry_after,
                asyncio.get_running_loop().time() + adaptive,
            )


@dataclass(slots=True)
class RouteThrottle:
    """Adapt exact-route concurrency downward on transient provider failures."""

    maximum: int
    limit: int = field(init=False)
    active: int = 0
    successful_requests: int = 0
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    def __post_init__(self) -> None:
        if self.maximum < 1:
            raise ValueError("route concurrency must be positive")
        self.limit = self.maximum

    async def acquire(self) -> None:
        """Wait for one slot under the current adaptive route limit."""
        async with self.condition:
            await self.condition.wait_for(lambda: self.active < self.limit)
            self.active += 1

    async def release(self) -> None:
        """Release one active route slot."""
        async with self.condition:
            self.active -= 1
            if self.active < 0:
                raise RuntimeError("route throttle released without a matching acquisition")
            self.condition.notify_all()

    async def transient_failure(self) -> tuple[int, int]:
        """Halve route concurrency and return the previous and current limits."""
        async with self.condition:
            previous = self.limit
            self.limit = max(1, math.ceil(self.limit / 2))
            self.successful_requests = 0
            self.condition.notify_all()
            return previous, self.limit

    async def success(self) -> tuple[int, int] | None:
        """Restore route concurrency gradually after sustained successful requests."""
        async with self.condition:
            if self.limit >= self.maximum:
                self.successful_requests = 0
                return None
            self.successful_requests += 1
            threshold = max(8, self.limit * 4)
            if self.successful_requests < threshold:
                return None
            previous = self.limit
            self.limit += 1
            self.successful_requests = 0
            self.condition.notify_all()
            return previous, self.limit


def _attempt_status(exc: Exception, status_code: int | None) -> AttemptStatus:
    if isinstance(exc, httpx.TimeoutException):
        return AttemptStatus.TIMEOUT
    if isinstance(exc, httpx.NetworkError):
        return AttemptStatus.NETWORK_ERROR
    if isinstance(exc, httpx.HTTPStatusError) or status_code is not None:
        return AttemptStatus.HTTP_ERROR
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return AttemptStatus.INVALID_RESPONSE
    return AttemptStatus.FAILED


def _terminal_cell_status(attempt_status: AttemptStatus) -> CellStatus:
    if attempt_status is AttemptStatus.TRUNCATED:
        return CellStatus.TRUNCATED
    if attempt_status is AttemptStatus.REFUSED:
        return CellStatus.REFUSED
    return CellStatus.FAILED


def _transport_error_cost(exc: Exception, reservation: float) -> float:
    """Release reservations for explicit rate-limit rejections; keep uncertain failures."""
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return 0.0
    return reservation


@dataclass(slots=True)
class AttemptOutcome:
    """Everything needed to persist and act on one provider attempt."""

    status: AttemptStatus
    final_status: CellStatus | None
    parsed: ProviderResponse | None
    raw: JsonObject | None
    http_status: int | None
    error: str | None
    actual_cost: float
    started_at: str
    latency_seconds: float
    retryable: bool = False
    cancelled: bool = False


def _empty_provider_gates() -> dict[tuple[str, str], asyncio.Semaphore]:
    return {}


def _empty_circuits() -> dict[tuple[str, str], ProviderCircuit]:
    return {}


def _empty_route_throttles() -> dict[tuple[str, str, str], RouteThrottle]:
    return {}


def _empty_clients() -> dict[tuple[str, str], httpx.AsyncClient]:
    return {}


def _empty_progress_samples() -> deque[tuple[float, int]]:
    return deque()


@dataclass(slots=True)
class GenerationRuntime:
    """Stateful service for one generation wave."""

    db: Database
    models: Mapping[str, Model]
    routes: Mapping[str, Route]
    global_limit: int
    model_limits: Mapping[str, int]
    budget: Budget
    timeout_seconds: float
    fake: bool
    progress_sink: GenerationProgressSink | None = None
    last_progress_report: int = 0
    started_monotonic: float = field(default_factory=time.monotonic)
    last_progress_report_monotonic: float = field(init=False)
    progress_samples: deque[tuple[float, int]] = field(default_factory=_empty_progress_samples)
    global_gate: asyncio.Semaphore = field(init=False)
    provider_gates: dict[tuple[str, str], asyncio.Semaphore] = field(
        default_factory=_empty_provider_gates
    )
    circuits: dict[tuple[str, str], ProviderCircuit] = field(default_factory=_empty_circuits)
    route_throttles: dict[tuple[str, str, str], RouteThrottle] = field(
        default_factory=_empty_route_throttles
    )
    clients: dict[tuple[str, str], httpx.AsyncClient] = field(default_factory=_empty_clients)
    client_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    budget_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    counter_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    counters: dict[str, int] = field(
        default_factory=lambda: {"completed": 0, "failed": 0, "budget_blocked": 0}
    )

    def __post_init__(self) -> None:
        self.last_progress_report_monotonic = self.started_monotonic
        self.progress_samples.append((self.started_monotonic, 0))
        self.global_gate = asyncio.Semaphore(self.global_limit)
        provider_limits: dict[tuple[str, str], int] = {}
        route_limits: dict[tuple[str, str, str], int] = {}
        for model_id, route in self.routes.items():
            provider_key = _provider_gate_key(model_id, route)
            current = provider_limits.get(provider_key)
            provider_limits[provider_key] = (
                route.max_concurrency if current is None else min(current, route.max_concurrency)
            )
            route_limits[_route_throttle_key(model_id, route)] = min(
                route.max_concurrency,
                self.model_limits.get(model_id, route.max_concurrency),
            )
        self.provider_gates = {
            key: asyncio.Semaphore(limit) for key, limit in provider_limits.items()
        }
        self.circuits = {key: ProviderCircuit() for key in provider_limits}
        self.route_throttles = {key: RouteThrottle(limit) for key, limit in route_limits.items()}

    def route_for(self, model: Model) -> Route:
        """Return a real pinned route or a zero-cost deterministic fake route."""
        if self.fake:
            return Route(
                "fake",
                "fake",
                model.id,
                0.0,
                0.0,
                model.max_concurrency,
                "fake",
                True,
            )
        return self.routes[model.id]

    async def client_for(self, route: Route) -> httpx.AsyncClient:
        """Return one shared client per backend/provider route."""
        key = (route.backend, route.provider)
        if key in self.clients:
            return self.clients[key]
        async with self.client_lock:
            if key in self.clients:
                return self.clients[key]
            api_key = route_api_key(route)
            if not api_key:
                variable = "HF_TOKEN" if route.backend == "huggingface" else "OPENROUTER_API_KEY"
                raise RuntimeError(f"{variable} is not set")
            self.clients[key] = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                headers=headers(route.backend, api_key),
                limits=httpx.Limits(
                    max_connections=max(4, route.max_concurrency),
                    max_keepalive_connections=max(2, route.max_concurrency),
                ),
            )
            return self.clients[key]

    def _trim_progress_samples(self, now: float) -> None:
        """Keep one boundary sample plus recent samples for a responsive ETA."""
        cutoff = now - ETA_WINDOW_SECONDS
        while len(self.progress_samples) > 1 and self.progress_samples[1][0] <= cutoff:
            self.progress_samples.popleft()

    def _route_bottlenecks(self, now: float) -> tuple[RouteBottleneck, ...]:
        """Return exact routes currently constrained by adaptive throttling or cooldown."""
        bottlenecks: list[RouteBottleneck] = []
        for model_id, route in self.routes.items():
            throttle = self.route_throttles.get(_route_throttle_key(model_id, route))
            if throttle is None:
                continue
            unconstrained = min(
                route.max_concurrency,
                self.model_limits.get(model_id, route.max_concurrency),
            )
            circuit = self.circuits.get(_provider_gate_key(model_id, route))
            cooldown = max(0.0, circuit.retry_after - now) if circuit is not None else 0.0
            current_limit = min(throttle.limit, unconstrained)
            throttled_and_saturated = (
                current_limit < unconstrained and throttle.active >= current_limit
            )
            if not throttled_and_saturated and cooldown <= 0:
                continue
            bottlenecks.append(
                RouteBottleneck(
                    model_id=model_id,
                    backend=route.backend,
                    provider=route.provider,
                    routed_model=route.model,
                    current_limit=current_limit,
                    unconstrained_limit=unconstrained,
                    active=throttle.active,
                    cooldown_seconds=cooldown,
                )
            )
        bottlenecks.sort(
            key=lambda item: (
                item.current_limit / item.unconstrained_limit,
                -item.cooldown_seconds,
                item.model_id,
            )
        )
        return tuple(bottlenecks)

    def _progress_event(self, total: int, now: float) -> GenerationProgress:
        """Build a progress event whose ETA follows recent observed throughput."""
        processed = sum(self.counters.values())
        self._trim_progress_samples(now)
        first_time, first_processed = self.progress_samples[0]
        delta = processed - first_processed
        span = now - first_time
        rate = delta / span if delta > 0 and span > 0 else None
        if rate is None and processed > 0:
            lifetime = now - self.started_monotonic
            if lifetime > 0:
                rate = processed / lifetime
        remaining = max(0, total - processed)
        eta = remaining / rate if rate is not None and rate > 0 else None
        return GenerationProgress(
            processed=processed,
            total=total,
            completed=self.counters["completed"],
            failed=self.counters["failed"],
            budget_blocked=self.counters["budget_blocked"],
            rate_per_second=rate,
            eta_seconds=eta,
            bottlenecks=self._route_bottlenecks(now),
        )

    def _emit_progress(self, total: int, now: float) -> None:
        """Emit one progress snapshot to logging and the optional CLI sink."""
        event = self._progress_event(total, now)
        LOGGER.info(
            "generation progress %s remaining=%d rate=%s eta=%s",
            self.counters,
            total - event.processed,
            event.rate_per_second,
            event.eta_seconds,
        )
        if self.progress_sink is not None:
            self.progress_sink(event)

    async def count(self, key: str, total: int) -> None:
        """Update counters and emit milestone progress with a rolling ETA."""
        async with self.counter_lock:
            self.counters[key] += 1
            processed = sum(self.counters.values())
            now = time.monotonic()
            self.progress_samples.append((now, processed))
            report_step = max(25, math.ceil(total / 20))
            should_report = (
                processed == total or processed - self.last_progress_report >= report_step
            )
            if should_report:
                self.last_progress_report = processed
                self.last_progress_report_monotonic = now
                self._emit_progress(total, now)

    async def progress_heartbeat(
        self,
        total: int,
        stop: asyncio.Event,
    ) -> None:
        """Refresh ETA during long cooldowns even when no milestone is completed."""
        while not stop.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=PROGRESS_HEARTBEAT_SECONDS)
            if stop.is_set():
                break
            async with self.counter_lock:
                processed = sum(self.counters.values())
                if processed >= total:
                    continue
                now = time.monotonic()
                if now - self.last_progress_report_monotonic < PROGRESS_HEARTBEAT_SECONDS:
                    continue
                self.last_progress_report_monotonic = now
                self._emit_progress(total, now)

    async def _request(
        self,
        cell: GenerationCell,
        route: Route,
        payload: RequestPayload,
        logical_idempotency: str,
    ) -> tuple[JsonObject, int | None]:
        if self.fake:
            return json_object(fake_response(cell), path="fake provider response"), None
        response = await (await self.client_for(route)).post(
            route_url(route),
            json=payload,
            headers={"Idempotency-Key": logical_idempotency},
        )
        response.raise_for_status()
        decoded: object = response.json()
        return json_object(decoded, path="provider response"), response.status_code

    @staticmethod
    def _validate_result(
        raw: JsonObject,
        route: Route,
        reservation: float,
        requested_max_tokens: int,
    ) -> tuple[ProviderResponse, float]:
        parsed = parse_response(raw)
        exposed = reasoning_field(raw, parsed.content)
        if exposed:
            raise GenerationResponseError(
                AttemptStatus.REASONING_EXPOSED,
                f"endpoint exposed reasoning: {exposed}",
                parsed=parsed,
                actual_cost=realized_cost(route, parsed, reservation),
            )
        if parsed.finish_reason in TRUNCATION_REASONS:
            raise GenerationResponseError(
                AttemptStatus.TRUNCATED,
                f"finish_reason={parsed.finish_reason}; requested output ceiling="
                f"{requested_max_tokens}",
                parsed=parsed,
                actual_cost=realized_cost(route, parsed, reservation),
            )
        if parsed.finish_reason in SAFETY_REASONS or parsed.refusal:
            raise GenerationResponseError(
                AttemptStatus.REFUSED,
                parsed.refusal or f"blocked: {parsed.finish_reason}",
                parsed=parsed,
                actual_cost=realized_cost(route, parsed, reservation),
            )
        if parsed.finish_reason != "stop":
            raise GenerationResponseError(
                AttemptStatus.INVALID_RESPONSE,
                f"provider returned non-success finish_reason={parsed.finish_reason!r}",
                parsed=parsed,
                actual_cost=realized_cost(route, parsed, reservation),
                retryable=True,
            )
        actual = realized_cost(route, parsed, reservation)
        return parsed, actual

    async def execute_attempt(
        self,
        *,
        cell: GenerationCell,
        model: Model,
        route: Route,
        payload: RequestPayload,
        requested_max_tokens: int,
        reservation: float,
        logical_idempotency: str,
        request_key: str,
        attempt_index: int,
    ) -> AttemptOutcome:
        """Execute, classify, and cost one request without writing database state."""
        started_at = utcnow()
        started = time.monotonic()
        raw: JsonObject | None = None
        parsed: ProviderResponse | None = None
        http_status: int | None = None
        actual_cost = reservation
        provider_key = _provider_gate_key(model.id, route)
        throttle_key = _route_throttle_key(model.id, route)
        circuit = self.circuits.setdefault(provider_key, ProviderCircuit())
        throttle = self.route_throttles.setdefault(
            throttle_key,
            RouteThrottle(
                min(
                    route.max_concurrency,
                    self.model_limits.get(model.id, route.max_concurrency),
                )
            ),
        )
        try:
            raw, http_status = await self._request(cell, route, payload, logical_idempotency)
            parsed, actual_cost = self._validate_result(
                raw,
                route,
                reservation,
                requested_max_tokens,
            )
            await circuit.success()
            recovered = await throttle.success()
            if recovered is not None:
                previous, current = recovered
                LOGGER.info(
                    "%s via %s/%s (%s): route concurrency increased from %d to %d "
                    "after sustained success",
                    model.id,
                    route.backend,
                    route.provider,
                    route.model,
                    previous,
                    current,
                )
            return AttemptOutcome(
                AttemptStatus.COMPLETED,
                CellStatus.COMPLETED,
                parsed,
                raw,
                http_status,
                None,
                actual_cost,
                started_at,
                time.monotonic() - started,
            )
        except asyncio.CancelledError:
            return AttemptOutcome(
                AttemptStatus.CANCELLED,
                CellStatus.RETRYABLE,
                parsed,
                raw,
                http_status,
                "task cancelled",
                actual_cost,
                started_at,
                time.monotonic() - started,
                retryable=True,
                cancelled=True,
            )
        except GenerationResponseError as exc:
            parsed = exc.parsed or parsed
            actual_cost = (
                exc.actual_cost
                if exc.actual_cost is not None
                else (realized_cost(route, parsed, reservation) if parsed else reservation)
            )
            final_status = (
                CellStatus.RETRYABLE if exc.retryable else _terminal_cell_status(exc.status)
            )
            return AttemptOutcome(
                exc.status,
                final_status,
                parsed,
                raw,
                http_status,
                str(exc),
                actual_cost,
                started_at,
                time.monotonic() - started,
                retryable=exc.retryable,
            )
        except Exception as exc:
            status = _attempt_status(exc, http_status)
            retryable = isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))
            # httpx/anyio transport exceptions are allowed to carry an empty message
            # (this is common for connection resets and some Windows socket failures).
            # A retryable cell must still have a durable non-empty diagnostic because the
            # SQLite schema deliberately rejects opaque failure states.
            detail = str(exc).strip()
            error = detail or type(exc).__name__
            if isinstance(exc, httpx.HTTPStatusError):
                http_status = exc.response.status_code
                retryable = http_status in TRANSIENT_STATUS
                raw = http_error_payload(exc.response)
                if http_status == 429:
                    source = rate_limit_source(route, exc.response, raw)
                    error = f"{error}; rate_limit_source={source}"
            if retryable:
                delay = retry_delay(exc, attempt_index, request_key)
                await circuit.transient_failure(delay)
                previous, current = await throttle.transient_failure()
                if current != previous:
                    reason = f"HTTP {http_status}" if http_status is not None else status.value
                    LOGGER.warning(
                        "%s via %s/%s (%s): route concurrency reduced from %d to %d after %s",
                        model.id,
                        route.backend,
                        route.provider,
                        route.model,
                        previous,
                        current,
                        reason,
                    )
            return AttemptOutcome(
                status,
                CellStatus.RETRYABLE if retryable else _terminal_cell_status(status),
                parsed,
                raw,
                http_status,
                error,
                _transport_error_cost(exc, reservation),
                started_at,
                time.monotonic() - started,
                retryable=retryable,
            )

    @staticmethod
    def attempt_event(
        *,
        cell: GenerationCell,
        model: Model,
        route: Route,
        request_key: str,
        attempt_number: int,
        logical_idempotency: str,
        requested_max_tokens: int,
        reservation: float,
        outcome: AttemptOutcome,
    ) -> AttemptFinished:
        """Build one typed attempt row and optional terminal update."""
        parsed = outcome.parsed
        completed_at = utcnow()
        raw_response_json = canonical_json(outcome.raw) if outcome.raw is not None else None
        attempt = AttemptRecord(
            request_key=request_key,
            phase="generation",
            cell_id=cell.cell_id,
            model_id=model.id,
            attempt_number=attempt_number,
            backend=route.backend,
            provider=route.provider,
            routed_model=route.model,
            status=outcome.status,
            idempotency_key=logical_idempotency,
            requested_max_tokens=requested_max_tokens,
            reservation_usd=reservation,
            reported_cost_usd=parsed.reported_cost_usd if parsed else None,
            accounted_cost_usd=outcome.actual_cost,
            input_tokens=parsed.input_tokens if parsed else None,
            output_tokens=parsed.output_tokens if parsed else None,
            http_status=outcome.http_status,
            finish_reason=parsed.finish_reason if parsed else None,
            latency_seconds=outcome.latency_seconds,
            error=outcome.error,
            raw_response_json=raw_response_json,
            started_at=outcome.started_at,
            completed_at=completed_at,
        )
        final = None
        if outcome.final_status is not None:
            content = (
                parsed.content if outcome.final_status is CellStatus.COMPLETED and parsed else None
            )
            final = CellCompletion(
                cell_id=cell.cell_id,
                status=outcome.final_status,
                provider=(parsed.provider or route.label) if parsed else route.label,
                content=content,
                error=outcome.error,
                input_tokens=parsed.input_tokens if parsed else None,
                output_tokens=parsed.output_tokens if parsed else None,
                latency_seconds=outcome.latency_seconds,
                raw_response_json=raw_response_json,
                completed_at=completed_at,
            )
        return AttemptFinished(attempt=attempt, final=final)

    async def process(
        self,
        cell: GenerationCell,
        writer: AsyncDatabaseWriter,
        total: int,
    ) -> ProcessDisposition:
        """Run one cell while preserving pinned-route retry semantics."""
        model = self.models[cell.model_id]
        route = self.route_for(model)
        provider_key = _provider_gate_key(model.id, route)
        throttle_key = _route_throttle_key(model.id, route)
        provider_gate = self.provider_gates.setdefault(
            provider_key,
            asyncio.Semaphore(route.max_concurrency),
        )
        circuit = self.circuits.setdefault(provider_key, ProviderCircuit())
        throttle = self.route_throttles.setdefault(
            throttle_key,
            RouteThrottle(
                min(
                    route.max_concurrency,
                    self.model_limits.get(model.id, route.max_concurrency),
                )
            ),
        )
        requested_max_tokens = model.request_output_tokens
        payload = request_payload(
            model,
            cell,
            route,
            max_output_tokens=requested_max_tokens,
        )
        reservation = estimate_cost(route, payload) * BUDGET_SAFETY_FACTOR
        await circuit.wait()
        await throttle.acquire()
        try:
            async with provider_gate, self.global_gate:
                if not await self.budget.reserve(reservation, self.budget_lock):
                    await writer.submit(BudgetBlocked(cell.cell_id, utcnow()))
                    await self.count("budget_blocked", total)
                    return "finished"
                attempt_count, _last_status, _last_max_tokens = await asyncio.to_thread(
                    self.db.generation_attempt_state,
                    str(cell.cell_id),
                )
                attempt_number = attempt_count + 1
                request_key = f"generation:{cell.cell_id}:{attempt_number}"
                logical_idempotency = idempotency_key(f"{cell.cell_id}:{requested_max_tokens}")
                await writer.submit(CellStarted(cell.cell_id, utcnow()))
                outcome = await self.execute_attempt(
                    cell=cell,
                    model=model,
                    route=route,
                    payload=payload,
                    requested_max_tokens=requested_max_tokens,
                    reservation=reservation,
                    logical_idempotency=logical_idempotency,
                    request_key=request_key,
                    attempt_index=attempt_number - 1,
                )
        finally:
            await throttle.release()

        await writer.submit(
            self.attempt_event(
                cell=cell,
                model=model,
                route=route,
                request_key=request_key,
                attempt_number=attempt_number,
                logical_idempotency=logical_idempotency,
                requested_max_tokens=requested_max_tokens,
                reservation=reservation,
                outcome=outcome,
            )
        )
        await self.budget.reconcile(reservation, outcome.actual_cost, self.budget_lock)
        if outcome.cancelled:
            raise asyncio.CancelledError()
        if outcome.retryable:
            return "retryable"
        if outcome.final_status is None:
            raise RuntimeError("non-transient generation attempt has no terminal status")
        key = "completed" if outcome.final_status is CellStatus.COMPLETED else "failed"
        await self.count(key, total)
        return "finished"

    async def close_clients(self) -> None:
        """Close every lazily created HTTP client."""
        await asyncio.gather(*(client.aclose() for client in self.clients.values()))


class GenerationResponseError(RuntimeError):
    """A terminal parsed-response defect with structured attempt metadata."""

    def __init__(
        self,
        status: AttemptStatus,
        message: str,
        *,
        parsed: ProviderResponse | None = None,
        actual_cost: float | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.parsed = parsed
        self.actual_cost = actual_cost
        self.retryable = retryable


async def _run_workers(
    runtime: GenerationRuntime,
    cells: Sequence[GenerationCell],
    writer: AsyncDatabaseWriter,
    *,
    total: int,
    final_round: bool,
) -> list[GenerationCell]:
    """Run one sweep and return transiently failed cells for a later sweep."""
    pending_by_model: dict[str, deque[GenerationCell]] = {
        model_id: deque() for model_id in runtime.model_limits
    }
    for cell in cells:
        pending_by_model[cell.model_id].append(cell)
    retryable: list[GenerationCell] = []

    async def worker(pending: deque[GenerationCell]) -> None:
        while pending:
            cell = pending.popleft()
            disposition = await runtime.process(cell, writer, total)
            if disposition == "retryable":
                if final_round:
                    await runtime.count("failed", total)
                else:
                    retryable.append(cell)

    max_workers = max(runtime.model_limits.values(), default=0)
    async with asyncio.TaskGroup() as tasks:
        for worker_index in range(max_workers):
            for model_id, limit in runtime.model_limits.items():
                pending = pending_by_model[model_id]
                if worker_index < limit and worker_index < len(pending):
                    tasks.create_task(worker(pending))
    retryable.sort(key=lambda cell: schedule_key(cell.cell_id))
    return retryable


async def run_pending(
    db: Database,
    models: Sequence[Model],
    routes: Mapping[str, Route],
    *,
    concurrency: int,
    per_model_concurrency: int,
    max_cost: float,
    timeout_seconds: float,
    retries: int,
    fake: bool = False,
    progress_sink: GenerationProgressSink | None = None,
) -> dict[str, int | float]:
    """Run pending cells with fair per-model workers and return a concise summary."""
    _positive_finite(max_cost, "max_cost")
    _positive_finite(timeout_seconds, "timeout")
    if retries < 0:
        raise ValueError("retries must be non-negative")
    model_by_id = {model.id: model for model in models}
    await asyncio.to_thread(db.reset_interrupted, utcnow())
    cells = await asyncio.to_thread(db.pending, list(model_by_id))
    if not cells:
        return {"completed": 0, "failed": 0, "budget_blocked": 0, "spend": db.spend()}
    missing_routes = sorted({cell.model_id for cell in cells} - set(routes))
    if missing_routes and not fake:
        raise RuntimeError("missing pinned routes for: " + ", ".join(missing_routes))
    effective, model_limits, route_limits = route_parallelism_summary(
        models, routes, concurrency, per_model_concurrency
    )
    LOGGER.info(
        "generation parallelism=%d model_limits=%s route_limits=%s",
        effective,
        model_limits,
        route_limits,
    )
    runtime = GenerationRuntime(
        db=db,
        models=model_by_id,
        routes=routes,
        global_limit=concurrency,
        model_limits=model_limits,
        budget=Budget(max_cost, await asyncio.to_thread(db.spend)),
        timeout_seconds=timeout_seconds,
        fake=fake,
        progress_sink=progress_sink,
    )
    heartbeat_stop = asyncio.Event()
    heartbeat = asyncio.create_task(runtime.progress_heartbeat(len(cells), heartbeat_stop))
    try:
        async with AsyncDatabaseWriter(db) as writer:
            remaining = list(cells)
            for retry_round in range(retries + 1):
                final_round = retry_round == retries
                remaining = await _run_workers(
                    runtime,
                    remaining,
                    writer,
                    total=len(cells),
                    final_round=final_round,
                )
                if not remaining:
                    break
                if not final_round:
                    LOGGER.warning(
                        "requeueing %d transient cells for retry round %d/%d",
                        len(remaining),
                        retry_round + 1,
                        retries,
                    )
    finally:
        heartbeat_stop.set()
        await heartbeat
        await runtime.close_clients()
    summary: dict[str, int | float] = {
        **runtime.counters,
        "spend": runtime.budget.committed,
        "reserved": runtime.budget.reserved,
    }
    return summary
