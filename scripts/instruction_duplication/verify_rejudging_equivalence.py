#!/usr/bin/env python3
"""Verify fresh deterministic judging against the frozen all-cell judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from instruction_duplication.io_utils import canonical_json


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number}: expected JSON object")
            yield value


def judgment_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def frozen_digests(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, row in enumerate(read_jsonl(path), start=1):
        cell_id = row.get("cell_id")
        judgment = row.get("judgment")
        if not isinstance(cell_id, str) or not cell_id:
            raise RuntimeError(f"{path}:{line_number}: missing cell_id")
        if cell_id in result:
            raise RuntimeError(f"{path}:{line_number}: duplicate cell_id {cell_id}")
        if not isinstance(judgment, dict):
            raise RuntimeError(f"{path}:{line_number}: missing judgment object")
        result[cell_id] = judgment_digest(judgment)
    return result


def database_digests(path: Path) -> dict[str, str]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT cell_id, judgment_json FROM judgments ORDER BY cell_id"
        ).fetchall()
    finally:
        connection.close()
    result: dict[str, str] = {}
    for cell_id, judgment_text in rows:
        if not isinstance(cell_id, str) or not isinstance(judgment_text, str):
            raise RuntimeError("invalid judgment row in restored database")
        value: object = json.loads(judgment_text)
        result[cell_id] = judgment_digest(value)
    return result


def run_python(*args: str, cwd: Path) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) if not existing else str(SRC) + os.pathsep + existing
    subprocess.run([sys.executable, *args], cwd=cwd, env=env, check=True)


def copy_frozen_workspace(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for relative in ("manifest.json",):
        shutil.copy2(source / relative, destination / relative)
    for relative in ("config", "data"):
        shutil.copytree(source / relative, destination / relative)
    (destination / "results").mkdir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("paper-run"))
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="retain the temporary restored workspace for inspection",
    )
    return parser.parse_args()


def verify(workspace: Path, *, keep_temp: bool = False) -> tuple[int, Path | None]:
    source_workspace = workspace.resolve()
    export = source_workspace / "results" / "cells-and-judgments.jsonl"
    if not export.is_file():
        raise FileNotFoundError(export)
    frozen = frozen_digests(export)

    temp_context: tempfile.TemporaryDirectory[str] | None = None
    if keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="instruction-duplication-rejudge-"))
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="instruction-duplication-rejudge-")
        temp_root = Path(temp_context.name)
    try:
        restored = temp_root / "paper-run"
        copy_frozen_workspace(source_workspace, restored)
        run_python(
            "scripts/instruction_duplication/restore_workspace_db.py",
            "--workspace",
            str(restored),
            "--source",
            str(export),
            cwd=ROOT,
        )
        run_python(
            "-m",
            "instruction_duplication",
            "judge",
            "--workspace",
            str(restored),
            cwd=ROOT,
        )
        fresh = database_digests(restored / "state" / "run.sqlite3")
        if set(fresh) != set(frozen):
            missing = sorted(set(frozen) - set(fresh))
            extra = sorted(set(fresh) - set(frozen))
            raise RuntimeError(
                "rejudging cell set differs from frozen export: "
                f"missing={len(missing)} extra={len(extra)}"
            )
        mismatches = [cell_id for cell_id in frozen if frozen[cell_id] != fresh[cell_id]]
        if mismatches:
            preview = ", ".join(mismatches[:5])
            raise RuntimeError(
                f"rejudging equivalence failed: {len(mismatches)} mismatches; first: {preview}"
            )
        retained = temp_root if keep_temp else None
        return len(frozen), retained
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def main() -> int:
    args = parse_args()
    count, retained = verify(args.workspace, keep_temp=args.keep_temp)
    print(f"rejudging_equivalence=ok cells={count} mismatches=0")
    if retained is not None:
        print(f"temporary_workspace={retained}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
