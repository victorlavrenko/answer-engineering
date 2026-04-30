from __future__ import annotations

import importlib

import answer_engineering as ae


def test_reproduce_entrypoints_import_cleanly() -> None:
    exported = (
        ae.CompiledRules,
        ae.GenerationPolicy,
        ae.GenerationRequest,
        ae.GenerationResult,
        ae.GenerationRuntime,
    )
    assert all(symbol is not None for symbol in exported)


def test_internal_boundaries_import_without_facade_side_effects() -> None:
    modules = [
        "answer_engineering.engine.proposal",
        "answer_engineering.engine.proposal.candidates",
        "answer_engineering.engine.proposal.guards",
        "answer_engineering.engine.proposal.match_tree",
        "answer_engineering.engine.scoring",
        "answer_engineering.engine.selection",
        "answer_engineering.engine.telemetry.events",
        "answer_engineering.engine.telemetry.aggregation",
        "answer_engineering.engine.telemetry.snapshots",
        "answer_engineering.inference",
        "answer_engineering.inference.decode",
    ]
    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module is not None
