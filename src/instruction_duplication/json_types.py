"""Validated JSON-compatible recursive types and conversion helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TypeGuard

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type JsonArray = list[JsonValue]
type ReadonlyJsonObject = Mapping[str, JsonValue]
type FrozenJsonValue = JsonScalar | tuple[FrozenJsonValue, ...] | Mapping[str, FrozenJsonValue]
type FrozenJsonObject = Mapping[str, FrozenJsonValue]


def is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    """Narrow an object to a mapping with fully known key and value types."""
    return isinstance(value, Mapping)


def object_mapping(value: object) -> Mapping[object, object] | None:
    """Return a typed mapping view after the runtime mapping check."""
    return value if is_object_mapping(value) else None


def is_string_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    """Narrow an object to a string-keyed mapping."""
    mapping = object_mapping(value)
    return mapping is not None and all(isinstance(key, str) for key in mapping)


def is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    """Narrow an object to a sequence with a fully known element type."""
    return isinstance(value, Sequence)


def object_sequence(value: object) -> Sequence[object] | None:
    """Return a typed sequence view after excluding text and byte strings."""
    if isinstance(value, (str, bytes, bytearray)) or not is_sequence(value):
        return None
    return value


def is_object_sequence(value: object) -> TypeGuard[Sequence[object]]:
    """Narrow an object to a non-string sequence."""
    return object_sequence(value) is not None


def object_value(value: object, *, name: str) -> Mapping[str, object]:
    """Validate a string-keyed mapping at an external-data boundary."""
    if not is_string_mapping(value):
        raise ValueError(f"{name} must be a JSON object")
    return value


def string_mapping(value: object, *, name: str) -> dict[str, str]:
    """Validate and copy a string-to-string mapping."""
    row = object_value(value, name=name)
    if not all(isinstance(item, str) for item in row.values()):
        raise ValueError(f"{name} must map strings to strings")
    return {key: item for key, item in row.items() if isinstance(item, str)}


def integer_value(value: object, *, name: str) -> int:
    """Validate an integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def number_value(value: object, *, name: str) -> float:
    """Validate a finite JSON number while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def boolean_value(value: object, *, name: str) -> bool:
    """Validate a JSON boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def json_value(value: object, *, path: str = "value") -> JsonValue:
    """Validate and copy one mutable JSON-compatible value."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    mapping = object_mapping(value)
    if mapping is not None:
        result: JsonObject = {}
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            result[key] = json_value(item, path=f"{path}.{key}")
        return result
    if is_object_sequence(value):
        return [json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{path} is not JSON-compatible")


def json_object(value: object, *, path: str = "value") -> JsonObject:
    """Validate and copy one mutable JSON object."""
    result = json_value(value, path=path)
    if not isinstance(result, dict):
        raise ValueError(f"{path} must be a JSON object")
    return result


def freeze_json(value: object, *, path: str = "value") -> FrozenJsonValue:
    """Validate and recursively freeze one JSON-compatible value."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    mapping = object_mapping(value)
    if mapping is not None:
        result: dict[str, FrozenJsonValue] = {}
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            result[key] = freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(result)
    if is_object_sequence(value):
        return tuple(freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    raise ValueError(f"{path} is not JSON-compatible")


def freeze_json_object(value: object, *, path: str = "value") -> FrozenJsonObject:
    """Validate and recursively freeze one JSON object."""
    result = freeze_json(value, path=path)
    if not isinstance(result, Mapping):
        raise ValueError(f"{path} must be a JSON object")
    return result


def thaw_json(value: FrozenJsonValue) -> JsonValue:
    """Convert an immutable JSON value back to plain mutable containers."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value
