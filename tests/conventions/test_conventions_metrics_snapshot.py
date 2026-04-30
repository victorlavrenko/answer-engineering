from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, TypedDict, cast


class MetricEntry(TypedDict):
    metric: str
    value: int | float | str | None
    target: str
    status: str
    notes: str


class MetricsSnapshot(TypedDict):
    repo_root: str
    metrics: list[MetricEntry]
    summary: dict[str, int]


def _run_metrics_snapshot(repo_root: Path) -> MetricsSnapshot:
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "metrics.json"
        cmd = [
            "python",
            "conventions/enforcement/measure_conventions_metrics.py",
            "--repo-root",
            str(repo_root),
            "--output-json",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, cwd=repo_root)
        payload = cast(
            dict[str, Any], json.loads(out_path.read_text(encoding="utf-8"))
        )
        return MetricsSnapshot(
            repo_root=cast(str, payload["repo_root"]),
            metrics=cast(list[MetricEntry], payload["metrics"]),
            summary=cast(dict[str, int], payload["summary"]),
        )


def _metrics_by_name(
    snapshot: MetricsSnapshot,
) -> dict[str, MetricEntry]:
    return {item["metric"]: item for item in snapshot["metrics"]}


def _count_type_ignores_in_tests(repo_root: Path) -> int:
    type_ignore_re = re.compile(r"#\s*type:\s*ignore")
    count = 0
    tests_root = repo_root / "tests"
    for path in tests_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if type_ignore_re.search(line):
                count += 1
    return count


def test_conventions_metrics_snapshot_schema_and_basics() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    snapshot = _run_metrics_snapshot(repo_root)

    assert snapshot["repo_root"] == str(repo_root)
    assert isinstance(snapshot["summary"], dict)
    assert isinstance(snapshot["metrics"], list)
    assert len(snapshot["metrics"]) >= 10


def test_conventions_metrics_dynamic_expectations() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    snapshot = _run_metrics_snapshot(repo_root)

    metrics = _metrics_by_name(snapshot)

    not_automated = {
        k for k, v in metrics.items() if v["status"] == "not_automated"
    }
    assert not_automated == {
        "gate_pass_rate",
        "boundary_api_adoption",
        "typed_public_api_coverage",
        "phase_validation_record_completeness",
    }

    expected_legacy_count = sum(
        1
        for name in ("engine", "inference", "reproduction")
        if (repo_root / "src" / "answer_engineering" / name).exists()
    )
    legacy_metric = metrics["parallel_internal_paths_per_capability_proxy"]
    assert legacy_metric["value"] == expected_legacy_count
    expected_legacy_status = "pass" if expected_legacy_count == 0 else "fail"
    assert legacy_metric["status"] == expected_legacy_status

    test_type_ignores = _count_type_ignores_in_tests(repo_root)
    type_ignore_metric = metrics["type_ignore_count_tests"]
    assert type_ignore_metric["value"] == test_type_ignores
    expected_type_ignore_status = "warn" if test_type_ignores > 0 else "pass"
    assert type_ignore_metric["status"] == expected_type_ignore_status
