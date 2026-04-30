from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from answer_engineering.engine.pipeline.text_patterns import (
    search_span,
)
from answer_engineering.rules.matching.options import ResolvedMatchOptions

SNAPSHOT_PATH = Path("tests/golden/text_patterns_matching_golden.json")


def _compute_actual(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case in cases:
        span = search_span(
            cast(str, case["text"]),
            cast(str, case["expression"]),
            match_options=ResolvedMatchOptions(
                casefold=cast(bool, case["casefold"]),
                word=cast(bool, case.get("word", False)),
            ),
        )
        out.append(
            {
                "text": case["text"],
                "expression": case["expression"],
                "casefold": case["casefold"],
                "word": case.get("word", False),
                "span": list(span) if span is not None else None,
            }
        )
    return out


def test_text_pattern_matching_matches_golden_snapshot() -> None:
    assert SNAPSHOT_PATH.exists(), (
        f"Golden snapshot missing: {SNAPSHOT_PATH}. "
        "Add snapshot and inspect the diff."
    )
    expected_raw = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    expected = cast(list[dict[str, Any]], expected_raw)
    actual = _compute_actual(expected)
    assert actual == expected
