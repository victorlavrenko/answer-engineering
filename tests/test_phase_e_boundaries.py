from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from huggingface_hub import CommitOperationAdd

from ae_paper_reproduction.infra.datasets.datasets import CachedDataset
from ae_paper_reproduction.telemetry.paper_metrics import (
    write_paper_metrics_file,
)
from ae_paper_reproduction.telemetry.reporting import append_rows_to_config


class _TimeoutPublisher:
    def ensure_dataset_repo(
        self, *, dataset_id: str, private: bool, token: str
    ) -> None:
        del dataset_id, private, token

    def commit(
        self,
        *,
        dataset_id: str,
        operations: Iterable[CommitOperationAdd],
        message: str,
        token: str,
    ) -> None:
        del dataset_id, operations, message, token
        raise TimeoutError("network timeout")

    def upload_file(
        self,
        *,
        path_or_fileobj: str | bytes | Path,
        path_in_repo: str,
        dataset_id: str,
        token: str,
    ) -> None:
        del path_or_fileobj, path_in_repo, dataset_id, token
        raise TimeoutError("network timeout")


def test_artifact_publisher_timeout_bubbles_from_boundary() -> None:
    with pytest.raises(TimeoutError, match="network timeout"):
        append_rows_to_config(
            publisher=_TimeoutPublisher(),
            dataset_id="demo/dataset",
            config_name="runs",
            run_id="run-1",
            rows=[{"run_id": "run-1"}],
            token="token",
        )


def test_dataset_loader_revision_mismatch_bubbles_from_boundary() -> None:
    @dataclass(slots=True)
    class _FailingDataset(CachedDataset):
        revision: str | None = None

        def _iter_external_rows(self) -> Iterator[Mapping[str, object]]:
            raise ValueError(f"revision not found: {self.revision}")

    dataset = _FailingDataset(
        "demo/dataset", "validation", revision="missing-rev"
    )
    with pytest.raises(ValueError, match="revision not found: missing-rev"):
        tuple(dataset.iter_rows())


def test_write_paper_metrics_file_writes_expected_filename(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Missing required mode/scope"):
        write_paper_metrics_file(
            subrun_telemetry=tuple(),
            paper_generated_dir=tmp_path,
        )


def test_main_tex_uses_only_generated_paper_metrics_input() -> None:
    main_tex = Path("docs/paper/main.tex").read_text(encoding="utf-8")
    assert r"\input{generated/paper-metrics.tex}" in main_tex
    assert r"\input{generated/overall-results.tex}" not in main_tex
    assert r"\input{generated/runtime-telemetry.tex}" not in main_tex
    assert r"\input{generated/degradation-summary.tex}" not in main_tex
