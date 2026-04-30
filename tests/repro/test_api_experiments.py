from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import datasets as hf_datasets
import pytest

from ae_paper_reproduction.api import (
    CachedHFDataset,
    Dataset,
    NotebookSubruns,
    Subrun,
    SubrunDefinition,
    SubrunResult,
    Summary,
)
from ae_paper_reproduction.core.evaluation.reports import (
    RulesetEvaluationResult,
)
from ae_paper_reproduction.core.evaluation.result_types import DatasetRow
from ae_paper_reproduction.core.planning.notebook_extractor import (
    NotebookRulesetSpec,
)
from ae_paper_reproduction.telemetry.reporting import (
    SubrunTelemetry,
)
from answer_engineering import (
    CompiledRules,
    GenerationPolicy,
    GenerationRequest,
    GenerationResult,
    GenerationRuntime,
)
from answer_engineering.telemetry import RuntimeTelemetrySnapshot


def test_dataset_rows_supports_n_and_question_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"id": "1", "case_type": "orl", "question": "q1", "gold": "g1"},
        {"id": "2", "case_type": "cardio", "question": "q2", "gold": "g2"},
        {"id": "3", "case_type": "orl", "question": "q3", "gold": "g3"},
    ]

    def _mock_load_dataset(
        *args: object, **kwargs: object
    ) -> hf_datasets.Dataset:
        del args, kwargs
        return hf_datasets.Dataset.from_list(  # pyright: ignore[reportUnknownMemberType]
            cast(list[dict[str, object]], rows)
        )

    monkeypatch.setattr("datasets.load_dataset", _mock_load_dataset)

    dataset = CachedHFDataset("demo/dataset", "validation")

    assert [row.id for row in dataset.rows(n=2)] == ["1", "2"]
    assert [row.id for row in dataset.rows(n=1, case_type="orl")] == ["1"]
    assert [row.id for row in dataset.rows(question_id="2")] == ["2"]
    with pytest.raises(ValueError, match="QUESTION_ID='9' not found"):
        dataset.rows(question_id="9")


def test_dataset_load_caches_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"id": "1", "case_type": "orl", "question": "q1", "gold": "g1"},
        {"id": "2", "case_type": "cardio", "question": "q2", "gold": "g2"},
    ]
    calls = 0

    def _iter_hf_dataset(
        *args: object, **kwargs: object
    ) -> hf_datasets.Dataset:
        del args, kwargs
        nonlocal calls
        calls += 1
        return hf_datasets.Dataset.from_list(  # pyright: ignore[reportUnknownMemberType]
            cast(list[dict[str, object]], rows)
        )

    monkeypatch.setattr("datasets.load_dataset", _iter_hf_dataset)

    dataset = CachedHFDataset("demo/dataset", "validation")
    loaded = dataset.materialize()

    assert loaded is dataset
    assert [row.id for row in dataset.rows()] == ["1", "2"]
    assert [row.id for row in dataset.rows(n=1)] == ["1"]
    assert calls == 1


def test_load_notebook_subruns_resolves_notebook_filename_from_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook_dir = tmp_path / "notebooks"
    notebook_dir.mkdir()
    (notebook_dir / "demo.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": [
                            "# Answer Engineering Rules\n",
                            "\n",
                            "## Run: demo\n",
                            "\n",
                            "## Mode: reasoning\n",
                            "\n",
                            "## Replace: hearing loss\n",
                            "\n",
                            "With:\n",
                            "\n",
                            "- HL\n",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    def _empty_rows(
        *,
        n: int | None = None,
        question_id: str | None = None,
        case_type: str | None = None,
    ) -> list[DatasetRow]:
        del n, question_id, case_type
        return list()

    dataset = cast(Dataset, SimpleNamespace(rows=_empty_rows))
    model = cast(
        GenerationRuntime,
        SimpleNamespace(model_id="demo-model", max_new_tokens=64),
    )
    [subrun] = NotebookSubruns("demo.ipynb", dataset=dataset, model=model)

    assert subrun.notebook_path == str(notebook_dir / "demo.ipynb")


def test_subrun_iter_tasks_compiles_once_and_builds_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Answer Engineering Rules\n",
                    "\n",
                    "## Run: demo\n",
                    "\n",
                    "## Mode: reasoning\n",
                    "\n",
                    "- orl\n",
                    "\n",
                    "## System Prompt\n",
                    "\n",
                    "Custom prompt from notebook.\n",
                    "\n",
                    "## Replace: hearing loss\n",
                    "\n",
                    "With:\n",
                    "\n",
                    "- HL\n",
                ],
            }
        ]
    }
    ipynb_path = tmp_path / "demo.ipynb"
    ipynb_path.write_text(json.dumps(notebook), encoding="utf-8")

    dataset_rows = [
        DatasetRow(id="1", case_type="orl", question="q1", gold="answer"),
        DatasetRow(id="2", case_type="other", question="q2", gold="answer"),
        DatasetRow(id="3", case_type="orl", question="q3", gold="answer"),
    ]

    def _iter_dataset_rows(
        *args: object, **kwargs: object
    ) -> Iterator[dict[str, str]]:
        del args, kwargs
        return iter(
            {
                "id": row.id,
                "case_type": row.case_type,
                "question": row.question,
                "gold": row.gold,
            }
            for row in dataset_rows
        )

    def _mock_load_dataset(
        *args: object, **kwargs: object
    ) -> hf_datasets.Dataset:
        del args, kwargs
        return hf_datasets.Dataset.from_list(  # pyright: ignore[reportUnknownMemberType]
            cast(list[dict[str, object]], list(_iter_dataset_rows()))
        )

    monkeypatch.setattr("datasets.load_dataset", _mock_load_dataset)

    compile_calls = 0

    class _FakeCompiledRules(CompiledRules):
        def __init__(self, text: str) -> None:
            nonlocal compile_calls
            compile_calls += 1
            object.__setattr__(self, "rules_markdown", text)
            object.__setattr__(self, "plan", object())

    monkeypatch.setattr(
        "ae_paper_reproduction.core.planning.subruns.CompiledRules",
        _FakeCompiledRules,
    )

    class _FakeModel:
        model_id = "demo-model"
        max_new_tokens = 64

        def generate(
            self, request: GenerationRequest, policy: GenerationPolicy
        ) -> GenerationResult:
            assert policy.system_prompt == "Custom prompt from notebook.\n"
            assert policy.verbosity == 1
            assert isinstance(policy.compiled_rules, CompiledRules)
            return GenerationResult(
                text=f"answer:{request.question}",
                ae_telemetry=RuntimeTelemetrySnapshot(
                    runtime_sec=0.25,
                    applied_decisions=0,
                    decision_limit_reached=False,
                    rules=tuple(),
                    events=tuple(),
                ),
                full_ids=None,
                prompt_ids=None,
                runtime_sec=0.25,
            )

    dataset = CachedHFDataset("demo/dataset", "validation")
    model = cast(GenerationRuntime, _FakeModel())
    [subrun] = NotebookSubruns(ipynb_path, dataset=dataset, model=model)

    tasks = subrun.select_tasks(n=2)
    assert [task.id for task in tasks] == ["1", "3"]
    assert compile_calls == 1
    assert all(task.compiled_rules is tasks[0].compiled_rules for task in tasks)

    task_results = [
        RulesetEvaluationResult(
            task.row,
            answer=model.generate(
                GenerationRequest(question=task.question),
                GenerationPolicy(
                    rules=task.compiled_rules,
                    system_prompt=subrun.system_prompt,
                    verbosity=1,
                ),
            ),
        )
        for task in tasks
    ]
    result = SubrunResult(subrun, task_results)

    assert result.subrun_id == "000-demo-orl"
    assert result.report.total == 2
    assert result.n_eval_requested == 2


def test_low_level_question_debug_flow_stays_minimal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Answer Engineering Rules\n",
                    "\n",
                    "## Run: demo\n",
                    "\n",
                    "## Mode: reasoning\n",
                    "\n",
                    "- orl\n",
                    "\n",
                    "## System Prompt\n",
                    "\n",
                    "Custom prompt from notebook.\n",
                    "\n",
                    "## Replace: hearing loss\n",
                    "\n",
                    "With:\n",
                    "\n",
                    "- HL\n",
                ],
            }
        ]
    }
    ipynb_path = tmp_path / "demo.ipynb"
    ipynb_path.write_text(json.dumps(notebook), encoding="utf-8")

    dataset_rows = [
        DatasetRow(id="1", case_type="orl", question="q1", gold="wrong"),
        DatasetRow(id="2", case_type="orl", question="q2", gold="answer:q2"),
    ]

    def _iter_dataset_rows(
        *args: object, **kwargs: object
    ) -> Iterator[dict[str, str]]:
        del args, kwargs
        return iter(
            {
                "id": row.id,
                "case_type": row.case_type,
                "question": row.question,
                "gold": row.gold,
            }
            for row in dataset_rows
        )

    def _mock_load_dataset(
        *args: object, **kwargs: object
    ) -> hf_datasets.Dataset:
        del args, kwargs
        return hf_datasets.Dataset.from_list(  # pyright: ignore[reportUnknownMemberType]
            cast(list[dict[str, object]], list(_iter_dataset_rows()))
        )

    monkeypatch.setattr("datasets.load_dataset", _mock_load_dataset)

    class _FakeModel:
        model_id = "demo-model"
        max_new_tokens = 64

        def generate(
            self, request: GenerationRequest, policy: GenerationPolicy
        ) -> GenerationResult:
            del policy
            return GenerationResult(
                text=f"answer:{request.question}",
                ae_telemetry=RuntimeTelemetrySnapshot(
                    runtime_sec=0.25,
                    applied_decisions=0,
                    decision_limit_reached=False,
                    rules=tuple(),
                    events=tuple(),
                ),
                full_ids=None,
                prompt_ids=None,
                runtime_sec=0.25,
            )

    dataset = CachedHFDataset("demo/dataset", "validation")
    model = cast(GenerationRuntime, _FakeModel())
    [subrun] = NotebookSubruns(ipynb_path, dataset=dataset, model=model)

    [task] = subrun.select_tasks(question_id="1")
    answer = model.generate(
        GenerationRequest(question=task.question),
        GenerationPolicy(
            rules=task.compiled_rules,
            system_prompt=subrun.system_prompt,
            verbosity=0,
        ),
    )
    from ae_paper_reproduction.core.evaluation.gold_checks import check_gold

    ok, _ = check_gold(answer.text, task.gold)
    single_row = {
        "subrun_id": subrun.subrun_id,
        "ruleset_name": subrun.name,
        "id": task.id,
        "case_type": task.case_type,
        "question": task.question,
        "gold": task.gold,
        "ok": ok,
        "answer": answer.text,
    }

    assert single_row == {
        "subrun_id": "000-demo-orl",
        "ruleset_name": "demo-orl",
        "id": "1",
        "case_type": "orl",
        "question": "q1",
        "gold": "wrong",
        "ok": False,
        "answer": "answer:q1",
    }


def test_build_summary_and_push_summary_to_hub_use_fixed_hf_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def _cuda_unavailable() -> bool:
        return False

    monkeypatch.setattr("torch.cuda.is_available", _cuda_unavailable)

    dataset = cast(
        Dataset,
        SimpleNamespace(
            dataset_id="demo/dataset",
            split="validation",
            metadata=lambda: {
                "dataset_id": "demo/dataset",
                "split": "validation",
            },
        ),
    )
    model = cast(
        GenerationRuntime,
        SimpleNamespace(model_id="demo-model", max_new_tokens=64),
    )
    ruleset = NotebookRulesetSpec(
        (
            "# Answer Engineering Rules\n\n## Run: demo\n\n"
            "## Mode: reasoning\n\n- orl\n\n"
            "## System Prompt\n\nsys\n\n## Replace: hearing loss\n\n"
            "With:\n\n- HL\n"
        ),
        cell_index=0,
        source_hint="demo.ipynb",
    )
    subrun = Subrun(
        definition=SubrunDefinition(
            ruleset=ruleset,
            case_type="orl",
            index=0,
            notebook_path="demo.ipynb",
            mode="reasoning",
        ),
        dataset=dataset,
        model=model,
    )
    subresult = SubrunResult(subrun, [], n_eval_requested=4)

    def _fake_write_paper_metrics_file(
        *,
        subrun_telemetry: tuple[SubrunTelemetry, ...],
        paper_generated_dir: Path,
    ) -> Path:
        del subrun_telemetry
        return paper_generated_dir / "paper-metrics.tex"

    monkeypatch.setattr(
        "ae_paper_reproduction.telemetry.paper_metrics.write_paper_metrics_file",
        _fake_write_paper_metrics_file,
    )

    summary = Summary([subresult])
    assert isinstance(summary, Summary)
    assert summary.group_context.n_eval_requested == 4

    calls: dict[str, object] = {}

    def _require_hf_token(env_name: str) -> str:
        calls["env_name"] = env_name
        return "token"

    def _push_telemetry_bundle(**kwargs: object) -> dict[str, dict[str, str]]:
        calls["private"] = kwargs["private"]
        calls["token"] = kwargs["token"]
        prepared_subruns = cast(
            list[tuple[SubrunTelemetry, object]], kwargs["prepared_subruns"]
        )
        return {
            "subruns": {
                built_rows.subrun_id: f"uploaded/{built_rows.subrun_id}"
                for built_rows, _ in prepared_subruns
            },
            "group": {"group": "uploaded/group"},
        }

    monkeypatch.setattr(
        "ae_paper_reproduction.runner.session.summary.require_hf_token",
        _require_hf_token,
    )
    monkeypatch.setattr(
        "ae_paper_reproduction.telemetry.reporting.push_telemetry_bundle",
        _push_telemetry_bundle,
    )

    uploaded = summary.push_to_hub("demo/telemetry")

    assert calls == {"env_name": "HF_TOKEN", "private": False, "token": "token"}
    assert uploaded == {
        "subruns": {"000-demo-orl": "uploaded/000-demo-orl"},
        "group": {"group": "uploaded/group"},
    }
