"""Versioned SQLite persistence with attempt-level provenance."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import TypedDict

from .io_utils import canonical_json, sha256_bytes
from .json_types import JsonObject, json_object, object_value
from .manifest import JUDGE_VERSION, MANIFEST_SCHEMA_VERSION
from .protocol import CONDITIONS
from .records import (
    AnalysisRow,
    AttemptFinished,
    AttemptRecord,
    CellStarted,
    GenerationCell,
    GenerationEvent,
    JudgmentWrite,
)
from .schedule import schedule_key
from .types import AttemptStatus, CellStatus, Question


class JudgmentTarget(TypedDict):
    """A fully typed cell selected for deterministic judging."""

    cell_id: str
    question_id: str
    condition_id: str
    status: str
    content: str | None
    dataset: str
    stem: str
    choices: dict[str, str]
    gold: str
    gold_text: str
    gold_source: str
    gold_raw: str
    source_split: str
    content_hash: str


class AttemptExport(TypedDict):
    """Typed attempt row exposed by the export API."""

    attempt_id: int
    request_key: str
    phase: str
    cell_id: str | None
    model_id: str
    attempt_number: int
    backend: str
    provider: str
    routed_model: str
    status: str
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
    started_at: str
    completed_at: str
    raw_response: JsonObject | None


DATABASE_SCHEMA_VERSION = 2
_CELL_STATUSES = tuple(status.value for status in CellStatus)
_ATTEMPT_STATUSES = tuple(status.value for status in AttemptStatus)
_CONDITION_COPIES = {condition.id: condition.copies for condition in CONDITIONS}
_CONDITION_CHECK = " OR ".join(
    f"(condition_id='{condition_id}' AND copies={copies})"
    for condition_id, copies in _CONDITION_COPIES.items()
)


def _quoted(values: Sequence[str]) -> str:
    return ",".join(f"'{value}'" for value in values)


DDL = f"""
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS questions (
  id TEXT PRIMARY KEY CHECK(length(id) > 0),
  dataset TEXT NOT NULL CHECK(length(dataset) > 0),
  stem TEXT NOT NULL CHECK(length(stem) > 0),
  choices_json TEXT NOT NULL,
  gold TEXT NOT NULL CHECK(length(gold) = 1),
  gold_text TEXT NOT NULL CHECK(length(gold_text) > 0),
  gold_source TEXT NOT NULL,
  gold_raw TEXT NOT NULL,
  source_split TEXT NOT NULL CHECK(length(source_split) > 0)
);
CREATE TABLE IF NOT EXISTS cells (
  cell_id TEXT PRIMARY KEY,
  question_id TEXT NOT NULL REFERENCES questions(id),
  model_id TEXT NOT NULL CHECK(length(model_id) > 0),
  condition_id TEXT NOT NULL CHECK(condition_id IN ({_quoted(tuple(_CONDITION_COPIES))})),
  copies INTEGER NOT NULL CHECK(copies BETWEEN 0 AND 3),
  status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ({_quoted(_CELL_STATUSES)})),
  provider TEXT,
  content TEXT,
  error TEXT,
  input_tokens INTEGER CHECK(input_tokens IS NULL OR input_tokens >= 0),
  output_tokens INTEGER CHECK(output_tokens IS NULL OR output_tokens >= 0),
  latency_seconds REAL CHECK(latency_seconds IS NULL OR latency_seconds >= 0),
  raw_response_json TEXT,
  started_at TEXT,
  completed_at TEXT,
  UNIQUE(question_id, model_id, condition_id),
  CHECK({_CONDITION_CHECK}),
  CHECK(status != 'completed' OR (content IS NOT NULL AND length(content) > 0)),
  CHECK(status NOT IN ('failed','retryable','truncated','refused') OR error IS NOT NULL)
);
CREATE TABLE IF NOT EXISTS attempts (
  attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_key TEXT NOT NULL UNIQUE,
  phase TEXT NOT NULL CHECK(phase IN ('preflight','generation')),
  cell_id TEXT REFERENCES cells(cell_id),
  model_id TEXT NOT NULL,
  attempt_number INTEGER NOT NULL CHECK(attempt_number >= 1),
  backend TEXT NOT NULL,
  provider TEXT NOT NULL,
  routed_model TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ({_quoted(_ATTEMPT_STATUSES)})),
  idempotency_key TEXT NOT NULL,
  requested_max_tokens INTEGER NOT NULL CHECK(requested_max_tokens > 0),
  reservation_usd REAL NOT NULL CHECK(reservation_usd >= 0),
  reported_cost_usd REAL CHECK(reported_cost_usd IS NULL OR reported_cost_usd >= 0),
  accounted_cost_usd REAL NOT NULL CHECK(accounted_cost_usd >= 0),
  input_tokens INTEGER CHECK(input_tokens IS NULL OR input_tokens >= 0),
  output_tokens INTEGER CHECK(output_tokens IS NULL OR output_tokens >= 0),
  http_status INTEGER,
  finish_reason TEXT,
  latency_seconds REAL NOT NULL CHECK(latency_seconds >= 0),
  error TEXT,
  raw_response_json TEXT,
  started_at TEXT NOT NULL,
  completed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS judgments (
  cell_id TEXT PRIMARY KEY REFERENCES cells(cell_id),
  judge_version TEXT NOT NULL,
  protocol_hash TEXT NOT NULL,
  lexical_version TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  judgment_json TEXT NOT NULL,
  judged_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS cells_status_idx ON cells(status);
CREATE INDEX IF NOT EXISTS attempts_cell_idx ON attempts(cell_id, attempt_number);
CREATE INDEX IF NOT EXISTS attempts_phase_idx ON attempts(phase);
"""


class Database:
    """Thread-safe, schema-versioned storage for one workspace."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        try:
            with self._lock:
                self._initialize_or_validate()
        except BaseException:
            self._conn.close()
            raise

    def _initialize_or_validate(self) -> None:
        tables = {
            str(row["name"])
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if tables and "metadata" not in tables:
            raise RuntimeError(
                "database predates schema versioning and cannot be resumed safely; "
                "use a new workspace"
            )
        self._conn.executescript(DDL)
        version_row = self._conn.execute(
            "SELECT value FROM metadata WHERE key='database_schema_version'"
        ).fetchone()
        if version_row is None:
            self._conn.executemany(
                "INSERT INTO metadata(key,value) VALUES(?,?)",
                [
                    ("database_schema_version", str(DATABASE_SCHEMA_VERSION)),
                    ("manifest_schema_version", str(MANIFEST_SCHEMA_VERSION)),
                ],
            )
            self._conn.commit()
            return
        if int(version_row["value"]) != DATABASE_SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema {version_row['value']} is incompatible with "
                f"schema {DATABASE_SCHEMA_VERSION}; use a new workspace"
            )

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Run a group of statements atomically with rollback on failure."""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield self._conn
            except BaseException:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()

    @staticmethod
    def cell_id(question_id: str, model_id: str, condition_id: str) -> str:
        """Return a stable current-schema identifier for one factorial cell."""
        value = f"cell-v2\0{question_id}\0{model_id}\0{condition_id}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def prepare(self, questions: Iterable[Question], model_ids: Iterable[str]) -> int:
        """Insert questions and the complete model-condition plan atomically."""
        question_list = list(questions)
        model_list = list(model_ids)
        if len({question.id for question in question_list}) != len(question_list):
            raise ValueError("question ids must be unique")
        if len(set(model_list)) != len(model_list):
            raise ValueError("model ids must be unique")
        with self.transaction() as connection:
            existing = connection.execute("SELECT COUNT(*) AS n FROM cells").fetchone()["n"]
            if existing:
                raise RuntimeError("workspace already contains planned cells")
            connection.executemany(
                """
                INSERT INTO questions(
                  id,dataset,stem,choices_json,gold,gold_text,gold_source,gold_raw,source_split
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        question.id,
                        question.dataset,
                        question.stem,
                        canonical_json(dict(question.choices)),
                        question.gold,
                        question.gold_text,
                        question.gold_source,
                        question.gold_raw,
                        question.source_split,
                    )
                    for question in question_list
                ],
            )
            rows = [
                (
                    self.cell_id(question.id, model_id, condition.id),
                    question.id,
                    model_id,
                    condition.id,
                    condition.copies,
                )
                for question in question_list
                for model_id in model_list
                for condition in CONDITIONS
            ]
            connection.executemany(
                """
                INSERT INTO cells(cell_id,question_id,model_id,condition_id,copies)
                VALUES(?,?,?,?,?)
                """,
                rows,
            )
        return len(rows)

    def validate_plan(self, question_ids: Sequence[str], model_ids: Sequence[str]) -> None:
        """Check database plan consistency against the workspace manifest."""
        expected = len(question_ids) * len(model_ids) * len(CONDITIONS)
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, COUNT(DISTINCT question_id) AS q, "
                "COUNT(DISTINCT model_id) AS m FROM cells"
            ).fetchone()
            stored_questions = [
                str(item["id"])
                for item in self._conn.execute("SELECT id FROM questions ORDER BY rowid")
            ]
            stored_models = [
                str(item["model_id"])
                for item in self._conn.execute(
                    "SELECT model_id FROM cells GROUP BY model_id ORDER BY MIN(rowid)"
                )
            ]
        if (
            int(row["n"]) != expected
            or stored_questions != list(question_ids)
            or stored_models != list(model_ids)
        ):
            raise RuntimeError("database cell plan does not match the workspace manifest")

    def reset_interrupted(self, completed_at: str) -> int:
        """Return cells left running by a terminated process to retryable."""
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE cells SET status='retryable', error='interrupted previous run',
                  provider=NULL,content=NULL,input_tokens=NULL,output_tokens=NULL,
                  latency_seconds=NULL,raw_response_json=NULL,completed_at=?
                WHERE status='running'
                """,
                (completed_at,),
            )
            connection.execute(
                "DELETE FROM judgments WHERE cell_id IN "
                "(SELECT cell_id FROM cells WHERE status='retryable')"
            )
        return int(cursor.rowcount)

    def pending(self, model_ids: Sequence[str]) -> list[GenerationCell]:
        """Load only fields needed to generate pending or retryable cells."""
        if not model_ids:
            return []
        placeholders = ",".join("?" for _ in model_ids)
        query = f"""
          SELECT c.cell_id,c.question_id,c.model_id,c.condition_id,c.copies,
                 q.dataset,q.stem,q.choices_json,q.gold
          FROM cells c JOIN questions q ON q.id=c.question_id
          WHERE c.status IN ('pending','retryable','budget_blocked')
            AND c.model_id IN ({placeholders})
        """
        with self._lock:
            rows = self._conn.execute(query, tuple(model_ids)).fetchall()
        cells = [self._decode_generation_row(row) for row in rows]
        cells.sort(key=lambda cell: schedule_key(cell.cell_id))
        return cells

    def representative_cell(self) -> GenerationCell:
        """Load the longest instructed prompt candidate for preflight."""
        query = """
          SELECT c.cell_id,c.question_id,c.model_id,c.condition_id,c.copies,
                 q.dataset,q.stem,q.choices_json,q.gold
          FROM cells c JOIN questions q ON q.id=c.question_id
          WHERE c.condition_id='system_before_after'
          ORDER BY length(q.stem) DESC LIMIT 1
        """
        with self._lock:
            row = self._conn.execute(query).fetchone()
        if row is None:
            raise RuntimeError("no planned cell is available for preflight")
        return self._decode_generation_row(row)

    def mark_running(self, cell_id: str, started_at: str) -> None:
        """Start a fresh cell attempt and clear all stale terminal data."""
        with self.transaction() as connection:
            connection.execute("DELETE FROM judgments WHERE cell_id=?", (cell_id,))
            cursor = connection.execute(
                """
                UPDATE cells SET status='running',provider=NULL,content=NULL,error=NULL,
                  input_tokens=NULL,output_tokens=NULL,latency_seconds=NULL,
                  raw_response_json=NULL,started_at=?,completed_at=NULL
                WHERE cell_id=?
                """,
                (started_at, cell_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown cell: {cell_id}")

    @staticmethod
    def _insert_attempt(connection: sqlite3.Connection, attempt: AttemptRecord) -> None:
        fields = (
            "request_key",
            "phase",
            "cell_id",
            "model_id",
            "attempt_number",
            "backend",
            "provider",
            "routed_model",
            "status",
            "idempotency_key",
            "requested_max_tokens",
            "reservation_usd",
            "reported_cost_usd",
            "accounted_cost_usd",
            "input_tokens",
            "output_tokens",
            "http_status",
            "finish_reason",
            "latency_seconds",
            "error",
            "raw_response_json",
            "started_at",
            "completed_at",
        )
        connection.execute(
            f"INSERT INTO attempts({','.join(fields)}) VALUES({','.join('?' for _ in fields)})",
            attempt.sql_values(),
        )

    def record_attempts(self, attempts: Sequence[AttemptRecord]) -> None:
        """Persist many provider attempts in one transaction."""
        if not attempts:
            return
        with self.transaction() as connection:
            for attempt in attempts:
                self._insert_attempt(connection, attempt)

    def apply_generation_events(self, events: Sequence[GenerationEvent]) -> None:
        """Apply a writer-queue batch atomically without blocking the event loop."""
        if not events:
            return
        with self.transaction() as connection:
            for event in events:
                if isinstance(event, CellStarted):
                    connection.execute("DELETE FROM judgments WHERE cell_id=?", (event.cell_id,))
                    cursor = connection.execute(
                        """
                        UPDATE cells SET status='running',provider=NULL,content=NULL,error=NULL,
                          input_tokens=NULL,output_tokens=NULL,latency_seconds=NULL,
                          raw_response_json=NULL,started_at=?,completed_at=NULL
                        WHERE cell_id=?
                        """,
                        (event.started_at, event.cell_id),
                    )
                    if cursor.rowcount != 1:
                        raise KeyError(f"unknown cell: {event.cell_id}")
                elif isinstance(event, AttemptFinished):
                    self._insert_attempt(connection, event.attempt)
                    final = event.final
                    if final is not None:
                        connection.execute(
                            "DELETE FROM judgments WHERE cell_id=?",
                            (final.cell_id,),
                        )
                        # Never let an empty provider/transport exception string turn a
                        # retryable or failed cell into an invalid SQLite row.  The normal
                        # generation path supplies a useful diagnostic; this fallback is the
                        # persistence boundary's last line of defence against opaque third-
                        # party exceptions and future classifier regressions.
                        final_error = (final.error or "").strip()
                        if (
                            final.status
                            in {
                                CellStatus.FAILED,
                                CellStatus.RETRYABLE,
                                CellStatus.TRUNCATED,
                                CellStatus.REFUSED,
                            }
                            and not final_error
                        ):
                            final_error = f"{final.status.value}: no error detail supplied"
                        cursor = connection.execute(
                            """
                            UPDATE cells SET status=?,provider=?,content=?,error=?,input_tokens=?,
                              output_tokens=?,latency_seconds=?,raw_response_json=?,completed_at=?
                            WHERE cell_id=?
                            """,
                            (
                                final.status.value,
                                final.provider,
                                final.content,
                                final_error[:4000] if final_error else None,
                                final.input_tokens,
                                final.output_tokens,
                                final.latency_seconds,
                                final.raw_response_json,
                                final.completed_at,
                                final.cell_id,
                            ),
                        )
                        if cursor.rowcount != 1:
                            raise KeyError(f"unknown cell: {final.cell_id}")
                else:
                    connection.execute(
                        """
                        UPDATE cells SET status='budget_blocked',provider=NULL,content=NULL,
                          error='cumulative cost cap reached',input_tokens=NULL,output_tokens=NULL,
                          latency_seconds=NULL,raw_response_json=NULL,completed_at=? WHERE cell_id=?
                        """,
                        (event.completed_at, event.cell_id),
                    )

    def judgment_targets(
        self,
        *,
        lexical_version: str,
        generation_protocol_hash: str,
    ) -> list[JudgmentTarget]:
        """Return cells whose stored judgment is missing or version/content-stale."""
        query = """
          SELECT c.cell_id,c.question_id,c.condition_id,c.status,c.content,
                 q.dataset,q.stem,q.choices_json,q.gold,q.gold_text,q.gold_source,
                 q.gold_raw,q.source_split,
                 j.judge_version,j.protocol_hash,j.lexical_version,j.content_hash
          FROM cells c JOIN questions q ON q.id=c.question_id
          LEFT JOIN judgments j ON j.cell_id=c.cell_id
          ORDER BY c.model_id,c.question_id,c.condition_id
        """
        with self._lock:
            rows = self._conn.execute(query).fetchall()
        result: list[JudgmentTarget] = []
        for row in rows:
            content = str(row["content"] or "")
            content_hash = sha256_bytes((str(row["status"]) + "\0" + content).encode("utf-8"))
            current = (
                row["judge_version"] == JUDGE_VERSION
                and row["protocol_hash"] == generation_protocol_hash
                and row["lexical_version"] == lexical_version
                and row["content_hash"] == content_hash
            )
            if not current:
                choices_json = row["choices_json"]
                if not isinstance(choices_json, str):
                    raise RuntimeError("database choices_json is not text")
                decoded_choices: object = json.loads(choices_json)
                choices_object = json_object(decoded_choices, path="database.choices")
                choices = {
                    label: value
                    for label, value in choices_object.items()
                    if isinstance(value, str)
                }
                if len(choices) != len(choices_object):
                    raise RuntimeError("database choices are not a string mapping")
                result.append(
                    {
                        "cell_id": str(row["cell_id"]),
                        "question_id": str(row["question_id"]),
                        "condition_id": str(row["condition_id"]),
                        "status": str(row["status"]),
                        "content": None if row["content"] is None else str(row["content"]),
                        "dataset": str(row["dataset"]),
                        "stem": str(row["stem"]),
                        "choices": choices,
                        "gold": str(row["gold"]),
                        "gold_text": str(row["gold_text"]),
                        "gold_source": str(row["gold_source"]),
                        "gold_raw": str(row["gold_raw"]),
                        "source_split": str(row["source_split"]),
                        "content_hash": content_hash,
                    }
                )
        return result

    def put_judgments(
        self,
        rows: Sequence[JudgmentWrite],
        *,
        lexical_version: str,
        generation_protocol_hash: str,
    ) -> None:
        """Batch upsert current-version judgments in one transaction."""
        if not rows:
            return
        with self.transaction() as connection:
            connection.executemany(
                """
                INSERT INTO judgments(
                  cell_id,judge_version,protocol_hash,lexical_version,
                  content_hash,judgment_json,judged_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(cell_id) DO UPDATE SET
                  judge_version=excluded.judge_version,
                  protocol_hash=excluded.protocol_hash,
                  lexical_version=excluded.lexical_version,
                  content_hash=excluded.content_hash,
                  judgment_json=excluded.judgment_json,
                  judged_at=excluded.judged_at
                """,
                [
                    (
                        item.cell_id,
                        JUDGE_VERSION,
                        generation_protocol_hash,
                        lexical_version,
                        item.content_hash,
                        canonical_json(item.judgment),
                        item.judged_at,
                    )
                    for item in rows
                ],
            )

    def iter_analysis_rows(self) -> Iterator[AnalysisRow]:
        """Stream compact statistical rows without completion or question text."""
        query = """
          SELECT c.question_id,c.model_id,c.condition_id,c.status,q.dataset,j.judgment_json
          FROM cells c JOIN questions q ON q.id=c.question_id
          LEFT JOIN judgments j ON j.cell_id=c.cell_id
          ORDER BY c.model_id,c.question_id,c.condition_id
        """
        with self._lock:
            cursor = self._conn.execute(query)
            while True:
                batch = cursor.fetchmany(512)
                if not batch:
                    break
                for row in batch:
                    judgment_text = row["judgment_json"]
                    judgment = None
                    if judgment_text:
                        decoded: object = json.loads(str(judgment_text))
                        judgment = json_object(decoded, path="database.judgment")
                    yield AnalysisRow(
                        question_id=str(row["question_id"]),
                        model_id=str(row["model_id"]),
                        condition_id=str(row["condition_id"]),
                        status=str(row["status"]),
                        dataset=str(row["dataset"]),
                        judgment=judgment,
                    )

    def iter_rows(self, *, include_raw_response: bool = False) -> Iterator[JsonObject]:
        """Stream cells, questions, and judgments without loading all responses at once."""
        raw_column = ",c.raw_response_json" if include_raw_response else ""
        query = f"""
          SELECT c.cell_id,c.question_id,c.model_id,c.condition_id,c.copies,c.status,
                 c.provider,c.content,c.error,c.input_tokens,c.output_tokens,
                 c.latency_seconds,c.started_at,c.completed_at{raw_column},
                 q.dataset,q.stem,q.choices_json,q.gold,q.gold_text,q.gold_source,
                 q.gold_raw,q.source_split,j.judgment_json
          FROM cells c JOIN questions q ON q.id=c.question_id
          LEFT JOIN judgments j ON j.cell_id=c.cell_id
          ORDER BY c.model_id,c.question_id,c.condition_id
        """
        with self._lock:
            cursor = self._conn.execute(query)
            while True:
                batch = cursor.fetchmany(256)
                if not batch:
                    break
                decoded = [self._decode_analysis_row(row, include_raw_response) for row in batch]
                yield from decoded

    def attempts(self) -> Iterator[AttemptExport]:
        """Stream attempt-level provenance."""
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM attempts ORDER BY attempt_id")
            while True:
                batch = cursor.fetchmany(256)
                if not batch:
                    break
                for row in batch:
                    raw = row["raw_response_json"]
                    raw_response: JsonObject | None = None
                    if raw:
                        decoded: object = json.loads(str(raw))
                        raw_response = json_object(decoded, path="database.raw_response")
                    yield AttemptExport(
                        attempt_id=int(row["attempt_id"]),
                        request_key=str(row["request_key"]),
                        phase=str(row["phase"]),
                        cell_id=None if row["cell_id"] is None else str(row["cell_id"]),
                        model_id=str(row["model_id"]),
                        attempt_number=int(row["attempt_number"]),
                        backend=str(row["backend"]),
                        provider=str(row["provider"]),
                        routed_model=str(row["routed_model"]),
                        status=str(row["status"]),
                        idempotency_key=str(row["idempotency_key"]),
                        requested_max_tokens=int(row["requested_max_tokens"]),
                        reservation_usd=float(row["reservation_usd"]),
                        reported_cost_usd=(
                            None
                            if row["reported_cost_usd"] is None
                            else float(row["reported_cost_usd"])
                        ),
                        accounted_cost_usd=float(row["accounted_cost_usd"]),
                        input_tokens=(
                            None if row["input_tokens"] is None else int(row["input_tokens"])
                        ),
                        output_tokens=(
                            None if row["output_tokens"] is None else int(row["output_tokens"])
                        ),
                        http_status=(
                            None if row["http_status"] is None else int(row["http_status"])
                        ),
                        finish_reason=(
                            None if row["finish_reason"] is None else str(row["finish_reason"])
                        ),
                        latency_seconds=float(row["latency_seconds"]),
                        error=None if row["error"] is None else str(row["error"]),
                        started_at=str(row["started_at"]),
                        completed_at=str(row["completed_at"]),
                        raw_response=raw_response,
                    )

    def counts(self) -> dict[str, int]:
        """Return cell counts grouped by status."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT status,COUNT(*) AS n FROM cells GROUP BY status"
            ).fetchall()
        return {str(row["status"]): int(row["n"]) for row in rows}

    def spend(self) -> float:
        """Return conservative cumulative accounted provider spend."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(accounted_cost_usd),0) AS total FROM attempts"
            ).fetchone()
        return float(row["total"])

    def reported_spend(self) -> float:
        """Return sum of provider-reported costs when available."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(reported_cost_usd),0) AS total FROM attempts"
            ).fetchone()
        return float(row["total"])

    def generation_attempt_state(self, cell_id: str) -> tuple[int, str | None, int | None]:
        """Return count and most recent status/output ceiling for generation attempts."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS n,
                       (SELECT status FROM attempts a2
                        WHERE a2.cell_id=? AND a2.phase='generation'
                        ORDER BY attempt_id DESC LIMIT 1) AS last_status,
                       (SELECT requested_max_tokens FROM attempts a3
                        WHERE a3.cell_id=? AND a3.phase='generation'
                        ORDER BY attempt_id DESC LIMIT 1) AS last_max_tokens
                FROM attempts a1 WHERE a1.cell_id=? AND a1.phase='generation'
                """,
                (cell_id, cell_id, cell_id),
            ).fetchone()
        last_status = None if row["last_status"] is None else str(row["last_status"])
        last_max_tokens = None if row["last_max_tokens"] is None else int(row["last_max_tokens"])
        return int(row["n"]), last_status, last_max_tokens

    @staticmethod
    def _decode_generation_row(row: sqlite3.Row) -> GenerationCell:
        choices_raw: object = json.loads(str(row["choices_json"]))
        choices_object = object_value(choices_raw, name="database.choices")
        return GenerationCell.from_mapping(
            {
                "cell_id": row["cell_id"],
                "question_id": row["question_id"],
                "model_id": row["model_id"],
                "condition_id": row["condition_id"],
                "copies": row["copies"],
                "dataset": row["dataset"],
                "stem": row["stem"],
                "choices": choices_object,
                "gold": row["gold"],
            }
        )

    @staticmethod
    def _decode_analysis_row(row: sqlite3.Row, include_raw_response: bool) -> JsonObject:
        item = json_object(dict(row), path="database.analysis_row")
        choices_text = item.pop("choices_json")
        decoded_choices: object = json.loads(str(choices_text))
        item["choices"] = json_object(decoded_choices, path="database.choices")
        judgment_text = item.pop("judgment_json")
        if judgment_text:
            decoded_judgment: object = json.loads(str(judgment_text))
            item["judgment"] = json_object(decoded_judgment, path="database.judgment")
        else:
            item["judgment"] = None
        if include_raw_response:
            raw_text = item.pop("raw_response_json")
            if raw_text:
                decoded_raw: object = json.loads(str(raw_text))
                item["raw_response"] = json_object(decoded_raw, path="database.raw_response")
            else:
                item["raw_response"] = None
        return item
