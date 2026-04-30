from __future__ import annotations

import answer_engineering

FORBIDDEN = [
    "Semantic" + "Rewrite" + "RuntimeEngine",
    "Semantic" + "Runtime" + "Adapter",
    "Rank" + "Override" + "Controller",
    "old_edit_request_to" + "_patch",
    "run_old_pipeline_" + "propose",
    "Noop" + "Patch",
    "ReplaceAbs" + "Patch",
    "InsertAbs" + "Patch",
    "core pipeline",
]


def test_legacy_runtime_symbols_are_not_public_api() -> None:
    exported = set(answer_engineering.__all__)
    for symbol in FORBIDDEN:
        assert symbol not in exported
        assert not hasattr(answer_engineering, symbol)
