from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from instruction_duplication.cli import main
from instruction_duplication.io_utils import sha256_json
from instruction_duplication.workspace import Workspace


def prepare(workspace: Path, local_jsonl: Path):
    main(
        [
            "prepare",
            "1",
            "--input-jsonl",
            str(local_jsonl),
            "--datasets",
            "test",
            "other",
            "--models",
            "gemma-3-12b",
            "--fake",
            "--workspace",
            str(workspace),
        ]
    )


def test_manifest_detects_question_tampering(tmp_path: Path, local_jsonl: Path):
    root = tmp_path / "run"
    prepare(root, local_jsonl)
    ws = Workspace(root)
    lines = ws.questions.read_text().splitlines()
    row = json.loads(lines[0])
    row["stem"] += " tampered"
    lines[0] = json.dumps(row)
    ws.questions.write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError, match="questions_hash"):
        ws.require_prepared()


def test_workspace_lock_rejects_live_process(tmp_path: Path):
    ws = Workspace(tmp_path)
    with ws.lock(), pytest.raises(RuntimeError, match="locked"), ws.lock():
        pass


def test_stale_lock_is_recovered(tmp_path: Path):
    ws = Workspace(tmp_path)
    ws.root.mkdir(parents=True, exist_ok=True)
    ws.lock_path.write_text("pid=999999999\n")
    with ws.lock():
        assert ws.lock_path.exists()
    assert not ws.lock_path.exists()


def test_windows_pid_probe_reads_tasklist_csv(monkeypatch):
    result = subprocess.CompletedProcess(
        args=["tasklist.exe"],
        returncode=0,
        stdout='"python.exe","424242","Console","1","12,345 K"\n',
        stderr="",
    )
    monkeypatch.setattr(
        "instruction_duplication.workspace.subprocess.run",
        lambda *args, **kwargs: result,
    )

    assert Workspace._windows_pid_alive(424242)
    assert not Workspace._windows_pid_alive(424243)


def test_windows_pid_probe_preserves_lock_when_tasklist_fails(monkeypatch):
    result = subprocess.CompletedProcess(
        args=["tasklist.exe"], returncode=1, stdout="", stderr="access denied"
    )
    monkeypatch.setattr(
        "instruction_duplication.workspace.subprocess.run",
        lambda *args, **kwargs: result,
    )

    assert Workspace._windows_pid_alive(424242)


def test_manifest_detects_package_version_change(tmp_path: Path, local_jsonl: Path):
    root = tmp_path / "run"
    prepare(root, local_jsonl)
    ws = Workspace(root)
    manifest = json.loads(ws.manifest.read_text())
    manifest["package_version"] = "1.0.0"
    ws.manifest.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="package_version"):
        ws.require_prepared()


def test_generation_integrity_allows_new_measurement_version(tmp_path: Path, local_jsonl: Path):
    root = tmp_path / "run"
    prepare(root, local_jsonl)
    ws = Workspace(root)
    manifest = json.loads(ws.manifest.read_text())
    manifest["package_version"] = "2.2.3"
    manifest["lexical_version"] = "older-lexical-measurement"
    manifest["judge_version"] = "older-judge"
    manifest["analysis_version"] = "older-analysis"
    ws.manifest.write_text(json.dumps(manifest))

    ws.require_generation_integrity()
    with pytest.raises(RuntimeError, match="package_version"):
        ws.require_prepared(require_runtime_environment=False)


def test_transport_patch_can_resume_302_workspace(monkeypatch, tmp_path: Path, local_jsonl: Path):
    root = tmp_path / "run"
    prepare(root, local_jsonl)
    ws = Workspace(root)
    environment = json.loads(ws.environment.read_text())
    distributions = environment["distributions"]
    distributions["instruction-duplication"] = "3.0.2"
    ws.environment.write_text(json.dumps(environment))
    manifest = json.loads(ws.manifest.read_text())
    manifest["package_version"] = "3.0.2"
    manifest["environment_hash"] = sha256_json(environment)
    ws.manifest.write_text(json.dumps(manifest))

    current = json.loads(json.dumps(environment))
    current["distributions"]["instruction-duplication"] = "3.0.3"
    monkeypatch.setattr(
        "instruction_duplication.workspace.runtime_environment",
        lambda: current,
    )
    monkeypatch.setattr("instruction_duplication.workspace.__version__", "3.0.3")

    ws.require_prepared()


def test_transport_compatible_workspace_can_resume_after_measurement_version_change(
    monkeypatch, tmp_path: Path, local_jsonl: Path
):
    root = tmp_path / "run"
    prepare(root, local_jsonl)
    ws = Workspace(root)

    environment = json.loads(ws.environment.read_text())
    environment["distributions"]["instruction-duplication"] = "3.0.6"
    ws.environment.write_text(json.dumps(environment))
    manifest = json.loads(ws.manifest.read_text())
    manifest["package_version"] = "3.0.6"
    manifest["judge_version"] = "format-neutral-semantic-role-judge-v2.8"
    manifest["analysis_version"] = "paired-semantic-role-analysis-v2.8"
    manifest["environment_hash"] = sha256_json(environment)
    ws.manifest.write_text(json.dumps(manifest))

    current = json.loads(json.dumps(environment))
    current["distributions"]["instruction-duplication"] = "3.0.9"
    monkeypatch.setattr(
        "instruction_duplication.workspace.runtime_environment",
        lambda: current,
    )
    monkeypatch.setattr("instruction_duplication.workspace.__version__", "3.0.9")

    ws.require_prepared()


def test_305_does_not_claim_generation_compatibility_with_303(tmp_path: Path, local_jsonl: Path):
    root = tmp_path / "run"
    prepare(root, local_jsonl)
    ws = Workspace(root)
    manifest = json.loads(ws.manifest.read_text())
    manifest["package_version"] = "3.0.3"
    ws.manifest.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="package_version"):
        ws.require_prepared(require_runtime_environment=False)


def test_305_can_resume_304_transport_compatible_workspace(
    monkeypatch, tmp_path: Path, local_jsonl: Path
):
    root = tmp_path / "run"
    prepare(root, local_jsonl)
    ws = Workspace(root)

    environment = json.loads(ws.environment.read_text())
    environment["distributions"]["instruction-duplication"] = "3.0.4"
    ws.environment.write_text(json.dumps(environment))
    manifest = json.loads(ws.manifest.read_text())
    manifest["package_version"] = "3.0.4"
    manifest["environment_hash"] = sha256_json(environment)
    ws.manifest.write_text(json.dumps(manifest))

    current = json.loads(json.dumps(environment))
    current["distributions"]["instruction-duplication"] = "3.0.5"
    monkeypatch.setattr(
        "instruction_duplication.workspace.runtime_environment",
        lambda: current,
    )
    monkeypatch.setattr("instruction_duplication.workspace.__version__", "3.0.5")

    ws.require_prepared()


def test_manifest_detects_environment_change(tmp_path: Path, local_jsonl: Path):
    root = tmp_path / "run"
    prepare(root, local_jsonl)
    ws = Workspace(root)
    environment = json.loads(ws.environment.read_text())
    environment["python"] = "0.0.0"
    ws.environment.write_text(json.dumps(environment))
    with pytest.raises(RuntimeError, match="environment_hash"):
        ws.require_prepared()


def test_read_only_validation_does_not_require_generation_machine(
    monkeypatch, tmp_path: Path, local_jsonl: Path
):
    root = tmp_path / "run"
    prepare(root, local_jsonl)
    ws = Workspace(root)
    monkeypatch.setattr(
        "instruction_duplication.workspace.runtime_environment",
        lambda: {"python": "different"},
    )
    with pytest.raises(RuntimeError, match="runtime_environment"):
        ws.require_prepared()
    ws.require_prepared(require_runtime_environment=False)
