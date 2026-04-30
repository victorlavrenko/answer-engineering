from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from _pytest.monkeypatch import MonkeyPatch
from huggingface_hub import CommitOperationAdd

from ae_paper_reproduction.core.aggregation.rule_stats import (
    AggregatedRunStats,
    TelemetryItem,
)
from ae_paper_reproduction.core.evaluation.report_types import (
    CaseTypeAccuracyRow,
)
from ae_paper_reproduction.core.evaluation.reports import (
    AccuracyReport,
    PairwiseComparisonReport,
    PairwiseOutcomeTransitions,
    RulesetEvaluationResult,
    pair_results_by_case_id,
)
from ae_paper_reproduction.core.evaluation.result_types import DatasetRow
from ae_paper_reproduction.core.planning.subruns import SubrunResult
from ae_paper_reproduction.telemetry import telemetry_types
from ae_paper_reproduction.telemetry.artifacts import (
    ArtifactMaterializer,
)
from ae_paper_reproduction.telemetry.reporting import (
    EvaluationArtifactFiles,
    GroupComparisonRow,
    GroupRunContext,
    GroupTelemetry,
    SubrunComparisonResult,
    SubrunContext,
    SubrunEvaluationResult,
    SubrunTelemetry,
    build_subrun_id,
    push_telemetry_bundle,
)
from answer_engineering.telemetry import (
    CandidateTelemetrySnapshot,
    RuleTelemetrySnapshot,
    RuntimeTelemetrySnapshot,
)

AnswerTelemetryRow = telemetry_types.AnswerTelemetryRow
ArtifactManifest = telemetry_types.ArtifactManifest
GroupArtifactFiles = telemetry_types.GroupArtifactFiles
GroupSummaryRow = telemetry_types.GroupSummaryRow
SubrunCaseTypeStatsRow = telemetry_types.SubrunCaseTypeStatsRow


def _build_group_comparison_output(
    *, delta_accuracy: float = 0.1
) -> SubrunComparisonResult:
    return cast(
        SubrunComparisonResult,
        SimpleNamespace(
            anchor_subrun_id="000-empty-rules",
            candidate_subrun_id="001-notebook-rules",
            anchor_ruleset_name="empty-rules",
            candidate_ruleset_name="notebook-rules",
            report=SimpleNamespace(
                delta_overall=delta_accuracy,
                outcome_transitions=PairwiseOutcomeTransitions(
                    [
                        RulesetEvaluationResult(
                            DatasetRow(
                                id="0",
                                question="",
                                case_type="synthetic",
                                gold="",
                            ),
                            answer="",
                            ok=False,
                        ),
                        RulesetEvaluationResult(
                            DatasetRow(
                                id="1",
                                question="",
                                case_type="synthetic",
                                gold="",
                            ),
                            answer="",
                            ok=True,
                        ),
                    ],
                    [
                        RulesetEvaluationResult(
                            DatasetRow(
                                id="0",
                                question="",
                                case_type="synthetic",
                                gold="",
                            ),
                            answer="",
                            ok=True,
                        ),
                        RulesetEvaluationResult(
                            DatasetRow(
                                id="1",
                                question="",
                                case_type="synthetic",
                                gold="",
                            ),
                            answer="",
                            ok=True,
                        ),
                    ],
                ),
            ),
        ),
    )


class _FakeApi:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []
        self.commit_calls = 0

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        private: bool,
        token: str,
        exist_ok: bool,
    ) -> None:
        del repo_id, repo_type, private, token, exist_ok

    def upload_file(
        self,
        *,
        path_or_fileobj: str | bytes,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        token: str,
    ) -> None:
        del path_or_fileobj, repo_type, token
        self.uploads.append((path_in_repo, repo_id))

    def create_commit(
        self,
        repo_id: str,
        operations: Iterable[CommitOperationAdd],
        *,
        commit_message: str,
        token: str,
        repo_type: str,
    ) -> None:
        del commit_message, repo_type, token
        self.commit_calls += 1
        for operation in operations:
            self.uploads.append((operation.path_in_repo, repo_id))


def _build_subrun_result(
    *,
    subrun_id: str,
    case_type: str | None = "orl",
    ok: bool = True,
    mode: str = "reasoning",
    paper_role: str | None = "primary",
    paper_variant: str | None = "reasoning",
) -> object:
    result = RulesetEvaluationResult(
        DatasetRow(id="1", case_type=case_type or "", question="q", gold="g"),
        answer="a",
        ok=ok,
    )
    return SimpleNamespace(
        subrun=SimpleNamespace(
            subrun_id=subrun_id,
            name=subrun_id,
            mode=mode,
            paper_role=paper_role,
            paper_variant=paper_variant,
            case_type=case_type,
            system_prompt="sys",
            dataset=SimpleNamespace(
                metadata=lambda: {"dataset_id": "d", "split": "validation"}
            ),
            model=SimpleNamespace(model_id="m"),
        ),
        n_eval_requested=1,
        n_eval_actual=1,
        report=AccuracyReport([result]),
    )


def _build_subrun_context(
    *, group_run_id: str, subrun_id: str
) -> SubrunContext:
    subresult = cast(SubrunResult, _build_subrun_result(subrun_id=subrun_id))
    return SubrunContext(
        group_context=cast(
            GroupRunContext, SimpleNamespace(group_run_id=group_run_id)
        ),
        subresult=subresult,
        anchor_subrun=subresult,
        run_stats=AggregatedRunStats([]),
        default_max_new_tokens=128,
        created_at_utc="2026-03-18T00:00:00+00:00",
        code_commit_sha="abc123",
    )


def _build_group_context(group_run_id: str) -> GroupRunContext:
    return GroupRunContext(
        group_run_id=group_run_id,
        created_at_utc="2026-03-18T00:00:00+00:00",
        code_commit_sha="abc123",
        subresults=(
            cast(SubrunResult, _build_subrun_result(subrun_id="000-base")),
        ),
        default_max_new_tokens=128,
    )


def test_compare_ruleset_results_counts_improvements() -> None:
    def _result(
        case_id: str, *, answer: str, ok: bool
    ) -> RulesetEvaluationResult:
        return RulesetEvaluationResult(
            DatasetRow(
                id=case_id, case_type="a", question=f"q-{case_id}", gold="g"
            ),
            answer=answer,
            ok=ok,
        )

    anchor_results = [
        _result("1", answer="a", ok=True),
        _result("2", answer="a", ok=False),
    ]
    candidate_results = [
        _result("1", answer="b", ok=False),
        _result("2", answer="b", ok=True),
    ]
    anchor_report = AccuracyReport(total=2, correct=1, by_case={"a": (1, 2)})
    candidate_report = AccuracyReport(total=2, correct=1, by_case={"a": (1, 2)})

    transitions = PairwiseOutcomeTransitions(anchor_results, candidate_results)
    assert transitions.anchor_correct_to_candidate_correct == 0
    assert transitions.anchor_correct_to_candidate_incorrect == 1
    assert transitions.anchor_incorrect_to_candidate_correct == 1
    assert transitions.anchor_incorrect_to_candidate_incorrect == 0

    comparison = PairwiseComparisonReport(
        anchor_results=anchor_results,
        anchor_report=anchor_report,
        candidate_results=candidate_results,
        candidate_report=candidate_report,
    )
    assert comparison.delta_overall == 0.0
    assert comparison.outcome_transitions == transitions


def test_build_subrun_id_slugifies() -> None:
    assert (
        build_subrun_id(index=2, ruleset_name="Notebook Rules")
        == "002-notebook-rules"
    )


def test_grouped_artifacts_and_uploads_use_nested_paths(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def _fake_write_paper_metrics_file(**kwargs: object) -> Path:
        output_dir = cast(Path, kwargs["paper_generated_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "paper-metrics.tex"
        output_path.write_text(
            "\\newcommand{\\PaperEvalN}{1}\\n"
            "\\newcommand{\\AvgInterventionsPerCase}{0.0}\\n",
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr(
        "ae_paper_reproduction.telemetry.paper_metrics.write_paper_metrics_file",
        _fake_write_paper_metrics_file,
    )
    run_stats = AggregatedRunStats(
        [
            TelemetryItem(
                RuntimeTelemetrySnapshot(
                    runtime_sec=None,
                    applied_decisions=1,
                    decision_limit_reached=False,
                    rules=(
                        RuleTelemetrySnapshot(
                            rule_id="r1",
                            rule_name="rule one",
                            evaluations=1,
                            applied=1,
                            trigger_firings=1,
                            proposals_generated=1,
                            generated_candidates_considered=1,
                            fallback_candidates_considered=0,
                            static_candidates_considered=0,
                            noop_candidates_generated=0,
                            conditions=tuple(),
                            candidate_choices=tuple(),
                        ),
                    ),
                    events=tuple(),
                )
            )
        ]
    )
    subrun_ctx = _build_subrun_context(
        group_run_id="group-a", subrun_id="000-empty-rules"
    )
    answer_rows = [
        AnswerTelemetryRow(
            ctx=subrun_ctx,
            result=RulesetEvaluationResult(
                DatasetRow(
                    id="1",
                    case_type="orl-ssnhl-acute",
                    question="",
                    gold="",
                ),
                answer="",
                ok=True,
                runtime_sec=0.5,
                ae_telemetry=RuntimeTelemetrySnapshot(
                    runtime_sec=0.5,
                    applied_decisions=1,
                    decision_limit_reached=False,
                    rules=(
                        RuleTelemetrySnapshot(
                            rule_id="r1",
                            rule_name="avoid:rule one",
                            evaluations=1,
                            applied=1,
                            trigger_firings=1,
                            proposals_generated=2,
                            generated_candidates_considered=2,
                            fallback_candidates_considered=0,
                            static_candidates_considered=0,
                            noop_candidates_generated=0,
                            conditions=(),
                            candidate_choices=(
                                CandidateTelemetrySnapshot(
                                    kind="generated",
                                    candidate_id="probe_1",
                                    label="probe_1",
                                    chosen=1,
                                ),
                            ),
                        ),
                    ),
                    events=(),
                ),
            ),
        )
    ]
    artifact_materializer = ArtifactMaterializer(run_reports_dir="reports/runs")
    subrun_artifacts = artifact_materializer.write_subrun(
        ctx=subrun_ctx,
        rules_markdown="## rule one",
        run_stats=run_stats,
        case_type_stats_rows=(
            SubrunCaseTypeStatsRow(
                case_type="orl",
                n_cases=2,
                accuracy=0.5,
                delta_accuracy_vs_anchor=0.0,
            ),
        ),
        answer_rows=answer_rows,
        gpu_name="cpu",
    )
    assert subrun_artifacts.run_report_md.as_posix().endswith(
        "reports/runs/run-group-a/subrun-000-empty-rules/run_report.md"
    )

    group_ctx = _build_group_context("group-a")
    group_artifacts = artifact_materializer.write_group(
        ctx=group_ctx,
        subrun_telemetry=(
            SubrunTelemetry(
                ctx=subrun_ctx,
                run_stats=run_stats,
                case_type_stats_rows=tuple(),
                artifact_manifest=ArtifactManifest(subrun_artifacts),
                eval_results=tuple(),
            ),
        ),
        comparison_rows=(
            GroupComparisonRow(
                group_context=group_ctx,
                comparison_output=_build_group_comparison_output(
                    delta_accuracy=0.1
                ),
            ),
        ),
    )
    assert group_artifacts.group_report_md.as_posix().endswith(
        "reports/runs/run-group-a/group_report.md"
    )
    assert group_artifacts.paper_metrics_json.exists()
    paper_metrics = json.loads(
        group_artifacts.paper_metrics_json.read_text(encoding="utf-8")
    )
    assert "paper_metrics_tex" in paper_metrics
    metrics_fragment = (
        tmp_path / "docs/paper/generated/paper-metrics.tex"
    ).read_text(encoding="utf-8")
    assert "\\newcommand{\\PaperEvalN}" in metrics_fragment
    assert "\\newcommand{\\AvgInterventionsPerCase}" in metrics_fragment
    assert "\\providecommand" not in metrics_fragment
    assert not (
        tmp_path / "docs/paper/generated/runtime-telemetry.tex"
    ).exists()
    assert not (
        tmp_path / "docs/paper/generated/degradation-summary.tex"
    ).exists()
    assert not (tmp_path / "docs/paper/generated/overall-results.tex").exists()

    fake_api = _FakeApi()
    monkeypatch.setattr(
        "ae_paper_reproduction.telemetry.reporting.HfApi",
        lambda: fake_api,
    )

    built_rows = SubrunTelemetry(
        ctx=subrun_ctx,
        run_stats=run_stats,
        case_type_stats_rows=[
            SubrunCaseTypeStatsRow(
                case_type="orl",
                n_cases=2,
                accuracy=0.5,
                delta_accuracy_vs_anchor=0.0,
            )
        ],
        artifact_manifest=ArtifactManifest(subrun_artifacts),
        eval_results=cast(
            list[RulesetEvaluationResult],
            [
                SimpleNamespace(
                    id="1",
                    case_type="orl",
                    question="q",
                    gold="g",
                    ok=True,
                    answer="a",
                    runtime_sec=None,
                    ae_telemetry=None,
                )
            ],
        ),
    )
    uploaded_paths = push_telemetry_bundle(
        dataset_id="owner/repo",
        private=False,
        token="token",
        prepared_subruns=[(built_rows, subrun_artifacts)],
        payload=GroupTelemetry(
            group_run_id="group-a",
            group_row=GroupSummaryRow(
                group_ctx,
                group_report_md_path=str(group_artifacts.group_report_md),
            ),
            comparison_rows=(
                GroupComparisonRow(
                    group_context=group_ctx,
                    comparison_output=_build_group_comparison_output(
                        delta_accuracy=0.1
                    ),
                ),
            ),
        ),
        artifact_files=group_artifacts,
    )
    assert uploaded_paths["subruns"] == {
        "000-empty-rules": "run-group-a/subrun-000-empty-rules/subrun.jsonl"
    }
    group_paths = uploaded_paths["group"]
    assert group_paths["group"] == "run-group-a/group.jsonl"
    assert group_paths["comparisons"] == "run-group-a/data/comparisons.jsonl"
    assert (
        group_paths["generated:paper-metrics.tex"]
        == "run-group-a/generated/paper-metrics.tex"
    )
    assert any(
        path == "run-group-a/subrun-000-empty-rules/data/rule_stats.jsonl"
        for path, _ in fake_api.uploads
    )
    assert any(
        path == "run-group-a/artifacts/group_report.md"
        for path, _ in fake_api.uploads
    )
    assert any(
        path == "run-group-a/generated/paper-metrics.tex"
        for path, _ in fake_api.uploads
    )
    assert not any(
        path.endswith("runtime-telemetry.tex") for path, _ in fake_api.uploads
    )
    assert not any(
        path.endswith("degradation-summary.tex") for path, _ in fake_api.uploads
    )


def test_write_group_generates_paper_metrics_tex_from_complete_fixture(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def _build_fixture_subrun(
        *,
        subrun_id: str,
        mode: str,
        scope: str,
        ok_values: tuple[bool, bool],
        runtime_sec: float,
    ) -> SubrunTelemetry:
        subresult = cast(
            SubrunResult,
            _build_subrun_result(
                subrun_id=subrun_id,
                case_type=scope,
                mode=mode,
                paper_role="primary",
                paper_variant=mode,
            ),
        )
        ctx = SubrunContext(
            group_context=cast(
                GroupRunContext,
                SimpleNamespace(group_run_id="group-metrics"),
            ),
            subresult=subresult,
            anchor_subrun=subresult,
            run_stats=AggregatedRunStats([]),
            default_max_new_tokens=128,
            created_at_utc="2026-03-18T00:00:00+00:00",
            code_commit_sha="abc123",
        )
        artifact_manifest = ArtifactManifest(
            report_md_path=(
                "reports/runs/run-group-metrics/"
                f"subrun-{subrun_id}/run_report.md"
            ),
            rules_original_md_path=(
                "reports/runs/run-group-metrics/"
                f"subrun-{subrun_id}/rules_original.md"
            ),
            rules_with_stats_md_path=(
                "reports/runs/run-group-metrics/"
                f"subrun-{subrun_id}/rules_with_stats.md"
            ),
        )
        eval_results = (
            RulesetEvaluationResult(
                DatasetRow(
                    id=f"{subrun_id}-1",
                    case_type=scope,
                    question="q1",
                    gold="g1",
                ),
                answer="a1",
                ok=ok_values[0],
                runtime_sec=runtime_sec,
            ),
            RulesetEvaluationResult(
                DatasetRow(
                    id=f"{subrun_id}-2",
                    case_type=scope,
                    question="q2",
                    gold="g2",
                ),
                answer="a2",
                ok=ok_values[1],
                runtime_sec=runtime_sec,
            ),
        )
        return SubrunTelemetry(
            ctx=ctx,
            run_stats=AggregatedRunStats([]),
            case_type_stats_rows=tuple(),
            artifact_manifest=artifact_manifest,
            eval_results=eval_results,
        )

    subrun_telemetry = (
        _build_fixture_subrun(
            subrun_id="000-baseline-ssnhl",
            mode="baseline",
            scope="orl-ssnhl-acute",
            ok_values=(False, True),
            runtime_sec=2.0,
        ),
        _build_fixture_subrun(
            subrun_id="001-baseline-conductive",
            mode="baseline",
            scope="orl-conductive-acute",
            ok_values=(False, False),
            runtime_sec=2.0,
        ),
        _build_fixture_subrun(
            subrun_id="002-reasoning-ssnhl",
            mode="reasoning",
            scope="orl-ssnhl-acute",
            ok_values=(True, False),
            runtime_sec=4.0,
        ),
        _build_fixture_subrun(
            subrun_id="003-reasoning-conductive",
            mode="reasoning",
            scope="orl-conductive-acute",
            ok_values=(True, False),
            runtime_sec=4.0,
        ),
        _build_fixture_subrun(
            subrun_id="004-trajectory-ssnhl",
            mode="trajectory",
            scope="orl-ssnhl-acute",
            ok_values=(True, True),
            runtime_sec=3.0,
        ),
        _build_fixture_subrun(
            subrun_id="005-trajectory-conductive",
            mode="trajectory",
            scope="orl-conductive-acute",
            ok_values=(False, True),
            runtime_sec=3.0,
        ),
    )

    artifact_materializer = ArtifactMaterializer(run_reports_dir="reports/runs")
    group_artifacts = artifact_materializer.write_group(
        ctx=_build_group_context("group-metrics"),
        subrun_telemetry=subrun_telemetry,
        comparison_rows=tuple(),
    )

    metrics_path = tmp_path / "docs/paper/generated/paper-metrics.tex"
    assert group_artifacts.paper_metrics_json.exists()
    assert metrics_path.exists()
    metrics_tex = metrics_path.read_text(encoding="utf-8")

    assert (
        "\\newcommand{\\CombinedBaselineBalancedAccuracyPct}{25.0\\%}"
        in metrics_tex
    )
    assert (
        "\\newcommand{\\CombinedBaselineBalancedAccuracyRaw}{25.0}"
        in metrics_tex
    )

    assert (
        "\\newcommand{\\CombinedReasoningBalancedAccuracyPct}{50.0\\%}"
        in metrics_tex
    )
    assert (
        "\\newcommand{\\CombinedReasoningBalancedAccuracyRaw}{50.0}"
        in metrics_tex
    )

    assert (
        "\\newcommand{\\CombinedTrajectoryBalancedAccuracyPct}{75.0\\%}"
        in metrics_tex
    )
    assert (
        "\\newcommand{\\CombinedTrajectoryBalancedAccuracyRaw}{75.0}"
        in metrics_tex
    )


def test_push_telemetry_bundle_uses_single_commit(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def _fake_write_paper_metrics_file(**kwargs: object) -> Path:
        output_dir = cast(Path, kwargs["paper_generated_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "paper-metrics.tex"
        output_path.write_text(
            "\\newcommand{\\PaperEvalN}{1}\\n", encoding="utf-8"
        )
        return output_path

    monkeypatch.setattr(
        "ae_paper_reproduction.telemetry.paper_metrics.write_paper_metrics_file",
        _fake_write_paper_metrics_file,
    )
    fake_api = _FakeApi()
    monkeypatch.setattr(
        "ae_paper_reproduction.telemetry.reporting.HfApi",
        lambda: fake_api,
    )

    subrun_ctx = _build_subrun_context(
        group_run_id="group-a", subrun_id="000-empty-rules"
    )
    group_ctx = _build_group_context("group-a")
    run_stats = AggregatedRunStats(
        [
            TelemetryItem(
                RuntimeTelemetrySnapshot(
                    runtime_sec=None,
                    applied_decisions=0,
                    decision_limit_reached=False,
                    rules=tuple(),
                    events=tuple(),
                )
            )
        ]
    )
    artifact_materializer = ArtifactMaterializer(run_reports_dir="reports/runs")
    subrun_artifacts = artifact_materializer.write_subrun(
        ctx=subrun_ctx,
        rules_markdown="## Empty\n",
        run_stats=run_stats,
        case_type_stats_rows=(),
        answer_rows=[],
        gpu_name="cpu",
    )
    built_rows = SubrunTelemetry(
        ctx=subrun_ctx,
        run_stats=run_stats,
        case_type_stats_rows=[],
        artifact_manifest=ArtifactManifest(subrun_artifacts),
        eval_results=[],
    )
    assert built_rows.ctx.system_prompt == "sys"
    group_artifacts = artifact_materializer.write_group(
        ctx=group_ctx,
        subrun_telemetry=(built_rows,),
        comparison_rows=(),
    )

    uploaded = push_telemetry_bundle(
        dataset_id="owner/repo",
        private=False,
        token="token",
        prepared_subruns=[(built_rows, subrun_artifacts)],
        payload=GroupTelemetry(
            group_run_id="group-a",
            group_row=GroupSummaryRow(
                group_ctx,
                group_report_md_path=str(group_artifacts.group_report_md),
            ),
            comparison_rows=tuple(),
        ),
        artifact_files=group_artifacts,
    )

    assert fake_api.commit_calls == 1
    assert uploaded["subruns"] == {
        "000-empty-rules": "run-group-a/subrun-000-empty-rules/subrun.jsonl"
    }
    assert uploaded["group"]["group"] == "run-group-a/group.jsonl"
    assert any(
        path == "run-group-a/subrun-000-empty-rules/data/answers.jsonl"
        for path, _ in fake_api.uploads
    )
    assert any(
        path == "run-group-a/group.jsonl" for path, _ in fake_api.uploads
    )


def test_subrun_value_objects_capture_stable_shapes() -> None:
    subrun_output = SubrunEvaluationResult(
        ruleset_name="rules",
        rules_markdown="## rules",
        subrun_id="000-rules",
        case_type_filter=None,
        scope_label="all",
        results=[
            RulesetEvaluationResult(
                DatasetRow(id="1", case_type="orl", question="q1", gold="g1"),
                answer="a1",
                ok=True,
                ae_telemetry=RuntimeTelemetrySnapshot(
                    runtime_sec=None,
                    applied_decisions=1,
                    decision_limit_reached=False,
                    rules=tuple(),
                    events=tuple(),
                ),
            ),
            RulesetEvaluationResult(
                DatasetRow(id="2", case_type="orl", question="q2", gold="g2"),
                answer="a2",
                ok=False,
                ae_telemetry=None,
            ),
        ],
        report=AccuracyReport(total=1, correct=1, by_case={"orl": (1, 1)}),
        n_eval_actual=1,
    )
    comparison_output = SubrunComparisonResult(
        anchor_subrun=SubrunEvaluationResult(
            ruleset_name="rules",
            rules_markdown="## rules",
            subrun_id="000-rules",
            case_type_filter=None,
            scope_label="all",
            results=[
                RulesetEvaluationResult(
                    DatasetRow(id="1", case_type="orl", question="q", gold="g"),
                    answer="a",
                    ok=True,
                )
            ],
            report=AccuracyReport(total=1, correct=1, by_case={"orl": (1, 1)}),
            n_eval_actual=1,
        ),
        candidate_subrun=SubrunEvaluationResult(
            ruleset_name="alt",
            rules_markdown="## alt",
            subrun_id="001-alt",
            case_type_filter=None,
            scope_label="all",
            results=[
                RulesetEvaluationResult(
                    DatasetRow(id="1", case_type="orl", question="q", gold="g"),
                    answer="b",
                    ok=False,
                )
            ],
            report=AccuracyReport(total=1, correct=0, by_case={"orl": (0, 1)}),
            n_eval_actual=1,
        ),
    )

    telemetry_items = subrun_output.telemetry_items()
    assert len(telemetry_items) == 1
    assert telemetry_items[0].applied_decisions == 1
    assert comparison_output.candidate_subrun_id == "001-alt"


def test_subrun_comparison_result_can_build_from_subrun_outputs() -> None:
    anchor_subrun = SubrunEvaluationResult(
        ruleset_name="anchor",
        rules_markdown="## anchor",
        subrun_id="000-anchor",
        case_type_filter=None,
        scope_label="all",
        results=[
            RulesetEvaluationResult(
                DatasetRow(id="1", case_type="orl", question="q1", gold="g"),
                answer="a1",
                ok=True,
            ),
            RulesetEvaluationResult(
                DatasetRow(id="2", case_type="orl", question="q2", gold="g"),
                answer="a2",
                ok=False,
            ),
        ],
        report=AccuracyReport(total=2, correct=1, by_case={"orl": (1, 2)}),
        n_eval_actual=2,
    )
    candidate_subrun = SubrunEvaluationResult(
        ruleset_name="candidate",
        rules_markdown="## candidate",
        subrun_id="001-candidate",
        case_type_filter=None,
        scope_label="all",
        results=[
            RulesetEvaluationResult(
                DatasetRow(id="1", case_type="orl", question="q1", gold="g"),
                answer="b1",
                ok=True,
            ),
            RulesetEvaluationResult(
                DatasetRow(id="2", case_type="orl", question="q2", gold="g"),
                answer="b2",
                ok=True,
            ),
        ],
        report=AccuracyReport(total=2, correct=2, by_case={"orl": (2, 2)}),
        n_eval_actual=2,
    )

    comparison_output = SubrunComparisonResult(
        anchor_subrun=anchor_subrun,
        candidate_subrun=candidate_subrun,
    )

    assert comparison_output.anchor_subrun_id == "000-anchor"
    assert comparison_output.candidate_subrun_id == "001-candidate"
    assert comparison_output.scope_label == "all"
    assert comparison_output.report.delta_overall == 0.5
    assert comparison_output.case_type_rows.rows == (
        CaseTypeAccuracyRow(
            case_type="orl",
            n_cases=2,
            anchor_accuracy=0.5,
            candidate_accuracy=1.0,
            delta_accuracy=0.5,
        ),
    )


def test_pair_results_by_case_id_returns_stable_join() -> None:
    paired = pair_results_by_case_id(
        [
            RulesetEvaluationResult(
                DatasetRow(id="b", case_type="orl", question="q2", gold="g"),
                answer="a2",
                ok=False,
            ),
            RulesetEvaluationResult(
                DatasetRow(id="a", case_type="orl", question="q1", gold="g"),
                answer="a1",
                ok=True,
            ),
        ],
        [
            RulesetEvaluationResult(
                DatasetRow(id="a", case_type="orl", question="q1", gold="g"),
                answer="b1",
                ok=False,
            ),
            RulesetEvaluationResult(
                DatasetRow(id="b", case_type="orl", question="q2", gold="g"),
                answer="b2",
                ok=True,
            ),
        ],
    )

    assert [
        (case_id, anchor.id, candidate.id)
        for case_id, anchor, candidate in paired
    ] == [
        ("a", "a", "a"),
        ("b", "b", "b"),
    ]


def test_group_comparison_row_can_build_from_comparison_output() -> None:
    comparison_output = SubrunComparisonResult(
        anchor_subrun=SubrunEvaluationResult(
            ruleset_name="rules",
            rules_markdown="## rules",
            subrun_id="000-rules",
            case_type_filter=None,
            scope_label="all",
            results=[
                RulesetEvaluationResult(
                    DatasetRow(
                        id="1", case_type="orl", question="q1", gold="g"
                    ),
                    answer="a",
                    ok=True,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="2", case_type="orl", question="q2", gold="g"
                    ),
                    answer="a",
                    ok=True,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="3", case_type="orl", question="q3", gold="g"
                    ),
                    answer="a",
                    ok=True,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="4", case_type="orl", question="q4", gold="g"
                    ),
                    answer="a",
                    ok=True,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="5", case_type="orl", question="q5", gold="g"
                    ),
                    answer="a",
                    ok=False,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="6", case_type="orl", question="q6", gold="g"
                    ),
                    answer="a",
                    ok=False,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="7", case_type="orl", question="q7", gold="g"
                    ),
                    answer="a",
                    ok=False,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="8", case_type="orl", question="q8", gold="g"
                    ),
                    answer="a",
                    ok=False,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="9", case_type="orl", question="q9", gold="g"
                    ),
                    answer="a",
                    ok=False,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="10", case_type="orl", question="q10", gold="g"
                    ),
                    answer="a",
                    ok=False,
                ),
            ],
            report=AccuracyReport(
                total=10, correct=4, by_case={"orl": (4, 10)}
            ),
            n_eval_actual=10,
        ),
        candidate_subrun=SubrunEvaluationResult(
            ruleset_name="alt",
            rules_markdown="## alt",
            subrun_id="001-alt",
            case_type_filter=None,
            scope_label="all",
            results=[
                RulesetEvaluationResult(
                    DatasetRow(
                        id="1", case_type="orl", question="q1", gold="g"
                    ),
                    answer="b",
                    ok=True,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="2", case_type="orl", question="q2", gold="g"
                    ),
                    answer="b",
                    ok=True,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="3", case_type="orl", question="q3", gold="g"
                    ),
                    answer="b",
                    ok=True,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="4", case_type="orl", question="q4", gold="g"
                    ),
                    answer="b",
                    ok=False,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="5", case_type="orl", question="q5", gold="g"
                    ),
                    answer="b",
                    ok=True,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="6", case_type="orl", question="q6", gold="g"
                    ),
                    answer="b",
                    ok=True,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="7", case_type="orl", question="q7", gold="g"
                    ),
                    answer="b",
                    ok=False,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="8", case_type="orl", question="q8", gold="g"
                    ),
                    answer="b",
                    ok=False,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="9", case_type="orl", question="q9", gold="g"
                    ),
                    answer="b",
                    ok=False,
                ),
                RulesetEvaluationResult(
                    DatasetRow(
                        id="10", case_type="orl", question="q10", gold="g"
                    ),
                    answer="b",
                    ok=False,
                ),
            ],
            report=AccuracyReport(
                total=10, correct=6, by_case={"orl": (6, 10)}
            ),
            n_eval_actual=10,
        ),
    )

    row = GroupComparisonRow(
        group_context=_build_group_context("group-a"),
        comparison_output=comparison_output,
    )

    assert row.anchor_subrun_id == "000-rules"
    assert row.candidate_subrun_id == "001-alt"
    assert abs(row.delta_accuracy - 0.2) < 1e-9
    assert row.improved == 2
    assert row.degraded == 1
    assert row.unchanged_correct == 3
    assert row.unchanged_incorrect == 4


def test_eval_result_helpers_can_build_from_dataset_rows() -> None:
    row = DatasetRow(id="1", case_type="orl", question="question", gold="gold")

    ruleset_result = RulesetEvaluationResult(
        row,
        answer="answer",
        ok=True,
        ae_telemetry=RuntimeTelemetrySnapshot(
            runtime_sec=None,
            applied_decisions=1,
            decision_limit_reached=False,
            rules=tuple(),
            events=tuple(),
        ),
    )
    assert ruleset_result.question == "question"
    assert ruleset_result.answer == "answer"


def test_push_telemetry_bundle_requires_grouped_identifiers(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_api = _FakeApi()
    monkeypatch.setattr(
        "ae_paper_reproduction.telemetry.reporting.HfApi",
        lambda: fake_api,
    )

    with pytest.raises(ValueError, match="group_run_id"):

        def _empty_upload_files() -> dict[str, Path]:
            return {}

        push_telemetry_bundle(
            dataset_id="owner/repo",
            private=False,
            token="token",
            prepared_subruns=[
                (
                    SubrunTelemetry(
                        ctx=_build_subrun_context(
                            group_run_id="", subrun_id="000-rules"
                        ),
                        run_stats=AggregatedRunStats([]),
                        case_type_stats_rows=tuple(),
                        artifact_manifest=ArtifactManifest(
                            report_md_path="report.md",
                            rules_original_md_path="rules.md",
                            rules_with_stats_md_path="rules_stats.md",
                        ),
                        eval_results=tuple(),
                    ),
                    cast(
                        EvaluationArtifactFiles,
                        SimpleNamespace(upload_files=_empty_upload_files),
                    ),
                )
            ],
            payload=GroupTelemetry(
                group_run_id="group-a",
                group_row=GroupSummaryRow(
                    **asdict(_build_group_context("group-a")),
                    group_report_md_path="reports/runs/group-a/group_report.md",
                ),
                comparison_rows=tuple(),
            ),
            artifact_files=cast(
                GroupArtifactFiles,
                SimpleNamespace(
                    upload_files=_empty_upload_files,
                    generated_files=_empty_upload_files,
                ),
            ),
        )
