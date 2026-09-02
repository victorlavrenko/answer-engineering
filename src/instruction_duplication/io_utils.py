"""Strict JSON, hashing, and atomic filesystem helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from .json_types import JsonObject, JsonValue, is_object_sequence, is_string_mapping, json_object


def _strict_value(value: object, *, path: str = "$") -> JsonValue:
    """Normalize a JSON-compatible value and replace non-finite floats with null."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if is_string_mapping(value):
        return {key: _strict_value(item, path=f"{path}.{key}") for key, item in value.items()}
    if is_object_sequence(value):
        return [_strict_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} is not JSON-compatible: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Serialize a value to deterministic, standards-compliant JSON."""
    return json.dumps(
        _strict_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    """Return a hexadecimal SHA-256 digest."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    """Hash a value after canonical JSON serialization."""
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def read_json(path: Path) -> JsonValue:
    """Read and recursively validate a UTF-8 JSON document."""
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read valid JSON from {path}: {exc}") from exc
    return _strict_value(decoded)


def read_json_object(path: Path) -> JsonObject:
    """Read a UTF-8 JSON document that must contain an object."""
    return json_object(read_json(path), path=str(path))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_text(path: Path, value: str) -> None:
    """Atomically write UTF-8 text."""
    _atomic_write(path, value)


def write_json(path: Path, value: object) -> None:
    """Atomically write an indented strict JSON document."""
    content = json.dumps(
        _strict_value(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    _atomic_write(path, content + "\n")


def read_jsonl(path: Path) -> list[JsonObject]:
    """Read recursively validated objects from a UTF-8 JSON Lines file."""
    rows: list[JsonObject] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not open {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                decoded: object = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            try:
                rows.append(json_object(decoded, path=f"{path}:{line_number}"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"JSONL row at {path}:{line_number} is not a valid object: {exc}"
                ) from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    """Atomically stream JSON objects as UTF-8 JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(rows, 1):
                normalized = _strict_value(row, path=f"row[{index}]")
                if not isinstance(normalized, dict):
                    raise TypeError(f"row[{index}] must be a JSON object")
                handle.write(canonical_json(normalized) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
