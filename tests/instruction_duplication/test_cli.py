from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from instruction_duplication import cli
from instruction_duplication.cli import main
from instruction_duplication.preflight import PreflightResult
from instruction_duplication.provider import Route


def run_args(workspace: Path, input_path: Path, seed: int = 7):
    return [
        "reproduce",
        "1",
        "--fake",
        "--input-jsonl",
        str(input_path),
        "--datasets",
        "test",
        "other",
        "--models",
        "gemma-3-12b",
        "--workspace",
        str(workspace),
        "--seed",
        str(seed),
        "--permutations",
        "20",
        "--bootstraps",
        "20",
    ]


def test_network_free_end_to_end_and_resume(tmp_path: Path, local_jsonl: Path):
    workspace = tmp_path / "run"
    main(run_args(workspace, local_jsonl))
    expected = {
        workspace / "manifest.json",
        workspace / "config" / "models.json",
        workspace / "config" / "model-eligibility.json",
        workspace / "config" / "environment.json",
        workspace / "config" / "routes.json",
        workspace / "config" / "preflight.json",
        workspace / "data" / "questions.jsonl",
        workspace / "data" / "dataset-audit.json",
        workspace / "data" / "lexical-reference.json",
        workspace / "data" / "fact-inventory.jsonl",
        workspace / "data" / "question-qc.jsonl",
        workspace / "data" / "generation-schedule.json",
        workspace / "state" / "run.sqlite3",
        workspace / "results" / "analysis.json",
        workspace / "results" / "report.txt",
        workspace / "results" / "paper-report.txt",
        workspace / "results" / "model-effect-summary.csv",
        workspace / "results" / "model-effects.csv",
        workspace / "results" / "cells-and-judgments.jsonl",
        workspace / "results" / "attempts.jsonl",
        workspace / "results" / "blinded-matched-pair-audit.jsonl",
        workspace / "results" / "blinded-matched-pair-key.jsonl",
        workspace / "results" / "human-audit-schema.json",
    }
    assert all(path.is_file() for path in expected)
    routes = json.loads((workspace / "config" / "routes.json").read_text())
    assert routes["schema_version"] == 3
    assert routes["preflight_policy"] == "concurrent-load-v1"
    assert routes["load_concurrency"] == 8
    attempts_path = workspace / "results" / "attempts.jsonl"
    attempts_before = attempts_path.read_text().count("\n")
    assert attempts_before == 16
    main(run_args(workspace, local_jsonl))
    attempts_after = attempts_path.read_text().count("\n")
    assert attempts_after == attempts_before
    analysis_path = workspace / "results" / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    assert analysis["sample"]["cells"] == 16
    assert "NaN" not in analysis_path.read_text()


def test_changed_seed_is_rejected_on_resume(tmp_path: Path, local_jsonl: Path):
    workspace = tmp_path / "run"
    main(run_args(workspace, local_jsonl, seed=7))
    with pytest.raises(SystemExit) as exc:
        main(run_args(workspace, local_jsonl, seed=8))
    assert exc.value.code == 3


def test_question_count_is_a_reproduce_shorthand():
    with pytest.raises(SystemExit) as exc:
        main(["1", "--help"])
    assert exc.value.code == 0


def test_cli_rejects_nan_cost(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["run", "--workspace", str(tmp_path), "--max-cost", "nan"])


def test_packaged_smoke_example_command(tmp_path: Path):
    fixture = Path(__file__).parents[1] / "examples" / "questions-smoke.jsonl"
    workspace = tmp_path / "packaged-smoke"
    args = [
        "reproduce",
        "1",
        "--fake",
        "--input-jsonl",
        str(fixture),
        "--datasets",
        "medqa",
        "medxpertqa",
        "afrimedqa",
        "--models",
        "ministral-3-14b-2512",
        "--workspace",
        str(workspace),
        "--permutations",
        "20",
        "--bootstraps",
        "20",
    ]
    main(args)
    analysis = json.loads((workspace / "results" / "analysis.json").read_text())
    assert analysis["sample"]["questions"] == 3
    assert analysis["sample"]["cells"] == 24


def test_reproduce_reports_concise_pipeline_progress(tmp_path: Path, local_jsonl: Path, capsys):
    workspace = tmp_path / "progress"
    main(run_args(workspace, local_jsonl))
    captured = capsys.readouterr()
    assert "Instruction Duplication" in captured.err
    assert "[1/5] preparing the workspace" in captured.err
    assert "[2/5] using deterministic fake routes" in captured.err
    assert "[3/5] generating or resuming responses" in captured.err
    assert "[generation] parallelism:" in captured.err
    assert "[generation] 16/16 (100%), completed=16" in captured.err
    assert "cells/min, ETA=0s" in captured.err
    assert "[4/5] applying deterministic judgments" in captured.err
    assert "[judge]" in captured.err
    assert "judgments persisted" in captured.err
    assert "[5/5] analyzing and exporting results" in captured.err
    assert "[analysis]" in captured.err
    assert "analysis complete" in captured.err


def test_reproduce_analyzes_exhausted_retryable_cells(
    tmp_path: Path, local_jsonl: Path, capsys, monkeypatch
):
    workspace = tmp_path / "retryable-analysis"
    args = run_args(workspace, local_jsonl)
    main(args)
    capsys.readouterr()

    def leave_one_retryable(*_args: object, **_kwargs: object) -> dict[str, int | float]:
        with sqlite3.connect(workspace / "state" / "run.sqlite3") as connection:
            connection.execute(
                """
                UPDATE cells
                   SET status='retryable', content=NULL, error='HTTP 429 after retry rounds'
                 WHERE cell_id=(SELECT cell_id FROM cells ORDER BY cell_id LIMIT 1)
                """
            )
        return {"completed": 0, "failed": 1, "budget_blocked": 0, "spend": 0.0}

    monkeypatch.setattr(cli, "_run_generation", leave_one_retryable)
    main(args)

    captured = capsys.readouterr()
    assert "1 cell remains retryable" in captured.err
    assert "proceeding with ITT analysis" in captured.err
    assert "[4/5] applying deterministic judgments" in captured.err
    assert "[5/5] analyzing and exporting results" in captured.err

    analysis = json.loads((workspace / "results" / "analysis.json").read_text())
    assert analysis["sample"]["status_counts"]["retryable"] == 1
    assert analysis["sample"]["usable_cells"] == 15


def test_reproduce_reports_pinned_routes_on_resume(tmp_path: Path, local_jsonl: Path, capsys):
    workspace = tmp_path / "resume-progress"
    args = run_args(workspace, local_jsonl)
    main(args)
    capsys.readouterr()
    main(args)
    captured = capsys.readouterr()
    assert "[1/5] using the existing prepared workspace" in captured.err
    assert "[2/5] using pinned provider routes" in captured.err
    assert "[routes] gemma-3-12b: fake/fake" in captured.err


def test_preflight_cannot_change_provider_after_generation(
    monkeypatch, tmp_path: Path, local_jsonl: Path
):
    workspace = tmp_path / "provider-continuity"
    main(run_args(workspace, local_jsonl))

    async def changed_route(*args: object, **kwargs: object) -> PreflightResult:
        del args, kwargs
        return PreflightResult(
            {
                "gemma-3-12b": Route(
                    "fake",
                    "different-provider",
                    "gemma-3-12b",
                    0.0,
                    0.0,
                    16,
                    "test",
                    True,
                )
            },
            {"status": "completed"},
            (),
        )

    monkeypatch.setattr(cli, "check_routes", changed_route)
    with pytest.raises(SystemExit) as captured:
        main(["preflight", "--workspace", str(workspace), "--fake"])
    assert captured.value.code == 3
