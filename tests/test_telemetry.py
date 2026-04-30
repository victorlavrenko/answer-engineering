from __future__ import annotations

import sys
import types
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from huggingface_hub import CommitOperationAdd

from ae_paper_reproduction.core.aggregation.rule_stats import (
    AggregatedRunStats,
    TelemetryItem,
)
from ae_paper_reproduction.core.evaluation.reports import (
    AccuracyReport,
    RulesetEvaluationResult,
    RunOutcomeTransitions,
)
from ae_paper_reproduction.core.evaluation.result_types import DatasetRow
from ae_paper_reproduction.core.planning.subruns import SubrunResult
from ae_paper_reproduction.telemetry import telemetry_types
from ae_paper_reproduction.telemetry.reporting import (
    GroupRunContext,
    GroupSubrunReportRow,
    RunContext,
    RunSummaryRow,
    RuntimeSummary,
    SubrunContext,
    append_rows_to_config,
    render_group_report,
    render_run_report,
    require_hf_token,
    update_reports_index,
    write_local_run_artifacts,
)
from answer_engineering.telemetry import RuntimeTelemetrySnapshot

AnswerTelemetryRow = telemetry_types.AnswerTelemetryRow
RunCaseTypeStatsRow = telemetry_types.RunCaseTypeStatsRow


class _FakeApi:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str]] = []

    def ensure_dataset_repo(
        self, *, dataset_id: str, private: bool, token: str
    ) -> None:
        del dataset_id, private, token

    def upload_file(
        self,
        *,
        path_or_fileobj: str | bytes | Path,
        path_in_repo: str,
        dataset_id: str,
        token: str,
    ) -> None:
        if isinstance(path_or_fileobj, bytes):
            payload = path_or_fileobj.decode("utf-8")
        else:
            payload = str(path_or_fileobj)
        self.uploads.append((path_in_repo, dataset_id, payload))
        assert token == "token"

    def commit(
        self,
        *,
        dataset_id: str,
        operations: Iterable[CommitOperationAdd],
        message: str,
        token: str,
    ) -> None:
        del message
        assert token == "token"
        for operation in operations:
            path_or_fileobj = operation.path_or_fileobj
            if isinstance(path_or_fileobj, bytes):
                payload = path_or_fileobj.decode("utf-8")
            else:
                payload = str(path_or_fileobj)
            self.uploads.append((operation.path_in_repo, dataset_id, payload))


def _ctx() -> RunContext:
    return RunContext(
        run_id="20260308T120000Z-test",
        created_at_utc="2026-03-08T12:00:00+00:00",
        code_commit_sha="abc123",
        dataset_id="d",
        split="train",
        case_type_filter="orl",
        n_eval_requested=10,
        n_eval_actual=8,
        model_id="m",
        max_new_tokens=128,
        compute_baseline=True,
        baseline_accuracy=0.5,
        edited_accuracy=0.75,
        delta_accuracy=0.25,
        applied_decisions_total=4,
        decision_limit_reached=False,
        rules_triggered_count=1,
        rules_applied_count=1,
        run_tag="tag",
    )


def test_append_rows_to_config_is_deterministic_per_run_id() -> None:
    api = _FakeApi()
    path = append_rows_to_config(
        publisher=api,
        dataset_id="lavrenko/answer-engineering",
        config_name="runs",
        run_id="run-a",
        rows=[{"run_id": "run-a", "x": 1}],
        token="token",
    )
    assert path == "runs/run-a.jsonl"
    assert api.uploads[0][0] == "runs/run-a.jsonl"
    assert '{"run_id": "run-a", "x": 1}' in api.uploads[0][2]


def test_write_local_artifacts_and_index(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    run_stats = AggregatedRunStats(
        [
            TelemetryItem(
                RuntimeTelemetrySnapshot(
                    runtime_sec=None,
                    applied_decisions=1,
                    decision_limit_reached=False,
                    rules=tuple(),
                    events=tuple(),
                )
            )
        ]
    )
    artifacts = write_local_run_artifacts(
        run_reports_dir="reports/runs",
        ctx=_ctx(),
        rules_markdown="## rule one\n\nWith:\n\n- x",
        run_stats=run_stats,
        case_type_stats_rows=[
            RunCaseTypeStatsRow(
                case_type="orl",
                n_cases=8,
                baseline_accuracy=0.5,
                edited_accuracy=0.75,
                delta_accuracy=0.25,
            )
        ],
        outcome_transitions=RunOutcomeTransitions(
            baseline_correct_to_edited_correct=3,
            baseline_correct_to_edited_incorrect=1,
            baseline_incorrect_to_edited_correct=2,
            baseline_incorrect_to_edited_incorrect=2,
        ),
        answer_rows=[
            AnswerTelemetryRow(
                ctx=SubrunContext(
                    group_context=cast(
                        GroupRunContext, SimpleNamespace(group_run_id="group-a")
                    ),
                    subresult=cast(
                        SubrunResult,
                        SimpleNamespace(
                            subrun=SimpleNamespace(
                                subrun_id="000-rules",
                                name="rules",
                                mode="reasoning",
                                paper_role="primary",
                                paper_variant="reasoning",
                                system_prompt="sys",
                                case_type="orl",
                                dataset=SimpleNamespace(
                                    metadata=lambda: {
                                        "dataset_id": "owner/dataset",
                                        "split": "validation",
                                    }
                                ),
                                model=SimpleNamespace(model_id="openai/test"),
                            ),
                            n_eval_requested=1,
                            n_eval_actual=1,
                            report=AccuracyReport(
                                [
                                    RulesetEvaluationResult(
                                        DatasetRow(
                                            id="1",
                                            case_type="orl",
                                            question="q?",
                                            gold="g",
                                        ),
                                        answer="e",
                                        ok=False,
                                    )
                                ]
                            ),
                        ),
                    ),
                    anchor_subrun=cast(
                        SubrunResult,
                        SimpleNamespace(
                            subrun=SimpleNamespace(subrun_id="000-rules"),
                            report=AccuracyReport(
                                [
                                    RulesetEvaluationResult(
                                        DatasetRow(
                                            id="1",
                                            case_type="orl",
                                            question="q?",
                                            gold="g",
                                        ),
                                        answer="e",
                                        ok=False,
                                    )
                                ]
                            ),
                        ),
                    ),
                    run_stats=AggregatedRunStats([]),
                    default_max_new_tokens=64,
                    created_at_utc="2026-03-18T00:00:00+00:00",
                    code_commit_sha="abc123",
                ),
                result=RulesetEvaluationResult(
                    DatasetRow(
                        id="1", case_type="orl", question="q?", gold="g"
                    ),
                    answer="e",
                    ok=False,
                ),
            )
        ],
        gpu_name="cpu",
    )
    assert artifacts.run_report_md.exists()
    assert artifacts.answers_json.exists()
    ctx = _ctx()
    summaries = [
        RunSummaryRow(
            created_at_utc=ctx.created_at_utc,
            model_id=ctx.model_id,
            case_type_filter=ctx.case_type_filter,
            accuracy=ctx.edited_accuracy,
            delta_accuracy=ctx.delta_accuracy,
            report_md_path=artifacts.run_report_md.as_posix(),
        )
    ]
    index = update_reports_index(
        reports_dir=Path("reports/runs"), run_summaries=summaries
    )
    assert index.exists()
    assert "Latest 10 runs" in index.read_text(encoding="utf-8")


def test_render_run_report_has_required_sections() -> None:
    report = render_run_report(
        ctx=_ctx(),
        gpu_name="cpu",
        runtime_summary=RuntimeSummary(
            (
                cast(
                    AnswerTelemetryRow,
                    SimpleNamespace(
                        baseline_runtime_sec=0.5,
                        edited_runtime_sec=1.25,
                    ),
                ),
            )
        ),
        case_type_stats_rows=[
            RunCaseTypeStatsRow(
                case_type="orl",
                n_cases=8,
                baseline_accuracy=0.5,
                edited_accuracy=0.75,
                delta_accuracy=0.25,
            )
        ],
        outcome_transitions=RunOutcomeTransitions(
            baseline_correct_to_edited_correct=3,
            baseline_correct_to_edited_incorrect=1,
            baseline_incorrect_to_edited_correct=2,
            baseline_incorrect_to_edited_incorrect=2,
        ),
        rules_original_markdown="## rule one",
        rules_with_stats_markdown=(
            "## rule one\n// ae-stats: evaluations=2 applied=1"
        ),
    )
    assert "# Run 20260308T120000Z-test" in report
    assert "## Aggregate telemetry" in report
    assert "## Annotated rules" in report
    assert "## Outcome transition summary" in report
    assert "## Runtime discussion" in report
    assert "1.2 sec/case (2.50× baseline)" in report
    assert "baseline correct → edited incorrect" in report


def test_require_hf_token_prefers_colab_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    google_module = types.ModuleType("google")
    colab_module = types.ModuleType("google.colab")

    class _FakeUserData:
        @staticmethod
        def get(name: str) -> str | None:
            return "secret-token" if name == "HF_TOKEN" else None

    colab_module.userdata = _FakeUserData  # pyright: ignore[reportAttributeAccessIssue]
    google_module.colab = colab_module  # pyright: ignore[reportAttributeAccessIssue]

    monkeypatch.setenv("HF_TOKEN", "env-token")
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)

    assert require_hf_token("HF_TOKEN") == "secret-token"


def test_require_hf_token_falls_back_to_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    monkeypatch.setitem(sys.modules, "google.colab", None)

    assert require_hf_token("HF_TOKEN") == "env-token"


def test_require_hf_token_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "google.colab", None)

    with pytest.raises(RuntimeError, match="Missing Hugging Face token"):
        require_hf_token("HF_TOKEN")


def test_render_group_report_includes_subrun_table() -> None:
    report = render_group_report(
        ctx=GroupRunContext(
            group_run_id="group-a",
            created_at_utc="2026-03-18T00:00:00+00:00",
            code_commit_sha="abc123",
            subresults=(
                cast(
                    SubrunResult,
                    SimpleNamespace(
                        subrun=SimpleNamespace(
                            dataset=SimpleNamespace(
                                metadata=lambda: {
                                    "dataset_id": "owner/dataset",
                                    "split": "validation",
                                }
                            ),
                            model=SimpleNamespace(model_id="openai/test-model"),
                        ),
                        n_eval_requested=4,
                        n_eval_actual=4,
                    ),
                ),
            ),
            default_max_new_tokens=64,
            run_tag="test",
        ),
        subrun_rows=[
            GroupSubrunReportRow(
                subrun_id="000-baseline",
                ruleset_name="baseline",
                accuracy=0.5,
                delta_accuracy=0.0,
                report_md_path=(
                    "reports/runs/run-group-a/subrun-000-baseline/run_report.md"
                ),
            )
        ],
        comparison_rows=[],
    )

    assert "## Subruns" in report
    assert "000-baseline" in report
