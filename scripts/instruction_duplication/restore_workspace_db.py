#!/usr/bin/env python3
"""Restore the paper SQLite workspace from the frozen all-cell JSONL export."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from instruction_duplication.io_utils import canonical_json, sha256_bytes, sha256_json
from instruction_duplication.lexical import LEXICAL_VERSION
from instruction_duplication.protocol import CONDITIONS
from instruction_duplication.storage import Database
from instruction_duplication.types import Question
from instruction_duplication.workspace import Workspace

TERMINAL_WITH_ERROR = {"failed", "retryable", "truncated", "refused"}
CONDITION_COPIES = {condition.id: condition.copies for condition in CONDITIONS}


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def require_string(row: Mapping[str, object], key: str, *, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: {key} must be a non-empty string")
    return value


def optional_string(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


def optional_int(row: Mapping[str, object], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer or null")
    return value


def optional_float(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric or null")
    return float(value)


def model_ids_from_document(value: object, *, path: Path) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON array")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: model {index} is not an object")
        result.append(require_string(item, "id", context=f"{path}: model {index}"))
    if len(result) != len(set(result)):
        raise ValueError(f"{path}: duplicate model ids")
    return result


def restore_database(
    workspace: Path,
    *,
    source: Path | None = None,
    overwrite: bool = False,
    include_judgments: bool = False,
) -> dict[str, object]:
    ws = Workspace(workspace.resolve())
    required = (ws.manifest, ws.questions, ws.models, ws.dataset_audit, ws.environment)
    missing = [str(path.relative_to(ws.root)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("workspace is missing frozen artifacts: " + ", ".join(missing))
    manifest = ws.load_manifest()
    question_rows = list(read_jsonl(ws.questions))
    models_document: object = json.loads(ws.models.read_text(encoding="utf-8"))
    environment_document: object = json.loads(ws.environment.read_text(encoding="utf-8"))
    integrity = {
        "questions_hash": sha256_json(question_rows) == manifest.questions_hash,
        "question_ids": tuple(str(row.get("id")) for row in question_rows) == manifest.question_ids,
        "models_hash": sha256_json(models_document) == manifest.models_hash,
        "environment_hash": sha256_json(environment_document) == manifest.environment_hash,
    }
    failed = [name for name, passed in integrity.items() if not passed]
    if failed:
        raise RuntimeError(
            "workspace frozen artifacts failed integrity validation: " + ", ".join(failed)
        )
    export_path = (source or ws.cells_export).resolve()
    if not export_path.is_file():
        raise FileNotFoundError(export_path)

    questions = [Question.from_dict(row) for row in question_rows]
    model_ids = model_ids_from_document(models_document, path=ws.models)
    expected_ids = {
        Database.cell_id(question.id, model_id, condition.id)
        for question in questions
        for model_id in model_ids
        for condition in CONDITIONS
    }

    destination = ws.database
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists; pass --overwrite to replace only this database"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".restore-tmp")
    for sidecar in (temporary, Path(str(temporary) + "-wal"), Path(str(temporary) + "-shm")):
        sidecar.unlink(missing_ok=True)

    imported_judgments = 0
    seen: set[str] = set()
    statuses: Counter[str] = Counter()
    try:
        with Database(temporary) as database:
            database.prepare(questions, model_ids)
            database.validate_plan(manifest.question_ids, tuple(model_ids))

        connection = sqlite3.connect(temporary)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            cell_rows: list[tuple[object, ...]] = []
            judgment_rows: list[tuple[object, ...]] = []
            for line_number, row in enumerate(read_jsonl(export_path), start=1):
                context = f"{export_path}:{line_number}"
                cell_id = require_string(row, "cell_id", context=context)
                if cell_id in seen:
                    raise ValueError(f"{context}: duplicate cell_id {cell_id}")
                if cell_id not in expected_ids:
                    raise ValueError(f"{context}: unexpected cell_id {cell_id}")
                question_id = require_string(row, "question_id", context=context)
                model_id = require_string(row, "model_id", context=context)
                condition_id = require_string(row, "condition_id", context=context)
                if Database.cell_id(question_id, model_id, condition_id) != cell_id:
                    raise ValueError(f"{context}: cell_id does not match its plan coordinates")
                copies = row.get("copies")
                if copies != CONDITION_COPIES.get(condition_id):
                    raise ValueError(f"{context}: copy count does not match {condition_id}")

                status = require_string(row, "status", context=context)
                content = optional_string(row, "content")
                error = optional_string(row, "error")
                if status == "completed" and not (content or "").strip():
                    raise ValueError(f"{context}: completed cell has no response text")
                if status in TERMINAL_WITH_ERROR and not (error or "").strip():
                    raise ValueError(f"{context}: {status} cell has no error detail")

                cell_rows.append(
                    (
                        status,
                        optional_string(row, "provider"),
                        content,
                        error,
                        optional_int(row, "input_tokens"),
                        optional_int(row, "output_tokens"),
                        optional_float(row, "latency_seconds"),
                        optional_string(row, "started_at"),
                        optional_string(row, "completed_at"),
                        cell_id,
                    )
                )
                if include_judgments:
                    judgment = row.get("judgment")
                    if not isinstance(judgment, dict):
                        raise ValueError(f"{context}: judgment is missing or not an object")
                    judge_version = require_string(
                        judgment, "judge_version", context=f"{context}: judgment"
                    )
                    content_hash = sha256_bytes((status + "\0" + (content or "")).encode("utf-8"))
                    judgment_rows.append(
                        (
                            cell_id,
                            judge_version,
                            manifest.protocol_hash,
                            LEXICAL_VERSION,
                            content_hash,
                            canonical_json(judgment),
                            optional_string(row, "completed_at") or "1970-01-01T00:00:00+00:00",
                        )
                    )

                seen.add(cell_id)
                statuses[status] += 1
                if len(cell_rows) >= 500:
                    connection.executemany(
                        """
                        UPDATE cells SET status=?,provider=?,content=?,error=?,input_tokens=?,
                          output_tokens=?,latency_seconds=?,raw_response_json=NULL,
                          started_at=?,completed_at=? WHERE cell_id=?
                        """,
                        cell_rows,
                    )
                    if include_judgments:
                        connection.executemany(
                            """
                            INSERT INTO judgments(
                              cell_id,judge_version,protocol_hash,lexical_version,
                              content_hash,judgment_json,judged_at
                            ) VALUES(?,?,?,?,?,?,?)
                            """,
                            judgment_rows,
                        )
                        imported_judgments += len(judgment_rows)
                    cell_rows.clear()
                    judgment_rows.clear()

            if cell_rows:
                connection.executemany(
                    """
                    UPDATE cells SET status=?,provider=?,content=?,error=?,input_tokens=?,
                      output_tokens=?,latency_seconds=?,raw_response_json=NULL,
                      started_at=?,completed_at=? WHERE cell_id=?
                    """,
                    cell_rows,
                )
                if include_judgments:
                    connection.executemany(
                        """
                        INSERT INTO judgments(
                          cell_id,judge_version,protocol_hash,lexical_version,
                          content_hash,judgment_json,judged_at
                        ) VALUES(?,?,?,?,?,?,?)
                        """,
                        judgment_rows,
                    )
                    imported_judgments += len(judgment_rows)

            missing_ids = expected_ids - seen
            if missing_ids:
                raise ValueError(
                    f"export is incomplete: expected {len(expected_ids)} cells, "
                    f"found {len(seen)}; first missing id is {min(missing_ids)}"
                )
            pending = connection.execute(
                "SELECT COUNT(*) FROM cells WHERE status='pending'"
            ).fetchone()[0]
            if pending:
                raise RuntimeError(f"restoration left {pending} cells pending")
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            connection.close()

        with Database(temporary) as database:
            database.validate_plan(manifest.question_ids, tuple(model_ids))
            restored_counts = database.counts()
            if restored_counts != dict(statuses):
                raise RuntimeError(
                    f"restored status counts {restored_counts} differ from export {dict(statuses)}"
                )

        connection = sqlite3.connect(temporary)
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("PRAGMA journal_mode=DELETE")
        finally:
            connection.close()

        destination.unlink(missing_ok=True)
        temporary.replace(destination)
        Path(str(temporary) + "-wal").unlink(missing_ok=True)
        Path(str(temporary) + "-shm").unlink(missing_ok=True)
    except BaseException:
        for sidecar in (
            temporary,
            Path(str(temporary) + "-wal"),
            Path(str(temporary) + "-shm"),
        ):
            sidecar.unlink(missing_ok=True)
        raise

    return {
        "database": str(destination),
        "cells_restored": len(seen),
        "status_counts": dict(sorted(statuses.items())),
        "judgments_imported": imported_judgments,
        "attempts_restored": 0,
        "raw_provider_envelopes_restored": 0,
        "source": str(export_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("paper-run"))
    parser.add_argument(
        "--source",
        type=Path,
        help="all-cell JSONL export; defaults to WORKSPACE/results/cells-and-judgments.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only WORKSPACE/state/run.sqlite3 if it already exists",
    )
    parser.add_argument(
        "--include-judgments",
        action="store_true",
        help="also import frozen judgments; omit this to rejudge every restored cell",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = restore_database(
        args.workspace,
        source=args.source,
        overwrite=args.overwrite,
        include_judgments=args.include_judgments,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not args.include_judgments:
        print("Next: instruction-duplication judge --workspace " + str(args.workspace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
