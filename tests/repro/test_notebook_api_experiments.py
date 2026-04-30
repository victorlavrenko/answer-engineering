from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import ae_paper_reproduction as reproduction_api
from ae_paper_reproduction.core.evaluation.result_types import DatasetRow
from answer_engineering import (
    GenerationPolicy,
    GenerationRequest,
    GenerationResult,
)
from answer_engineering.telemetry import RuntimeTelemetrySnapshot


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _notebooks_dir() -> Path:
    return _repo_root() / "notebooks"


def test_repro_notebook_code_cells_execute_with_answering_and_repro_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook = json.loads(
        (_notebooks_dir() / "reproduce.ipynb").read_text(encoding="utf-8")
    )
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]

    pushed: list[tuple[str, object, list[object]]] = []

    class _FakeDataset:
        def __init__(self, dataset_id: str, split: str) -> None:
            self.dataset_id = dataset_id
            self.split = split
            self.loaded = False

        def load(self) -> _FakeDataset:
            self.loaded = True
            return self

        def materialize(self) -> _FakeDataset:
            self.loaded = True
            return self

    class _FakeModel:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.loaded = False

        def load(self) -> _FakeModel:
            self.loaded = True
            return self

        def materialize(self) -> _FakeModel:
            self.loaded = True
            return self

        def generate(
            self,
            request: GenerationRequest,
            policy: GenerationPolicy,
        ) -> GenerationResult:
            assert policy.compiled_rules is not None
            assert (
                policy.system_prompt == GenerationPolicy.default_system_prompt
            )
            assert policy.verbosity == 0
            return GenerationResult(
                text=f"answer:{request.question}",
                ae_telemetry=RuntimeTelemetrySnapshot(
                    runtime_sec=0.0,
                    applied_decisions=0,
                    decision_limit_reached=False,
                    rules=tuple(),
                    events=tuple(),
                ),
                full_ids=None,
                prompt_ids=None,
                runtime_sec=0.0,
            )

    class _FakeTask:
        def __init__(
            self,
            question: str,
            *,
            question_id: str = "1",
            gold: str | None = None,
        ) -> None:
            resolved_gold = gold or f"answer:{question}"
            self.row = DatasetRow(
                id=question_id,
                case_type="all",
                question=question,
                gold=resolved_gold,
            )
            self.id = question_id
            self.case_type = "all"
            self.question = question
            self.gold = resolved_gold
            self.compiled_rules = "## Replace: hearing loss\n\nWith:\n\n- HL\n"

    class _FakeSubrun:
        def __init__(self, name: str) -> None:
            self.name = name
            self.subrun_id = f"id-{name}"
            self.ruleset_name = name
            self.scope_label = "all"
            self.system_prompt = GenerationPolicy.default_system_prompt
            self.compiled_rules = "## Replace: hearing loss\n\nWith:\n\n- HL\n"

        def select_tasks(
            self, *, n: int | None = None, question_id: str | None = None
        ) -> list[_FakeTask]:
            if question_id is not None:
                assert question_id == "q-1"
                return [
                    _FakeTask(
                        "demo-question",
                        question_id=question_id,
                        gold="answer:demo-question",
                    )
                ]
            assert n in {20, 400, 1000}
            return [_FakeTask("demo-question")]

    @dataclass
    class _FakeSubrunResult:
        subrun_id: str
        ruleset_name: str
        scope_label: str
        report: SimpleNamespace

        def __init__(
            self, subrun: _FakeSubrun, task_results: list[object]
        ) -> None:
            assert len(task_results) == 1
            self.subrun_id = subrun.subrun_id
            self.ruleset_name = subrun.ruleset_name
            self.scope_label = subrun.scope_label
            self.report = SimpleNamespace(accuracy=1.0)

    @dataclass
    class _FakeSummary:
        artifact_files: SimpleNamespace
        payload: SimpleNamespace

        def __init__(self, subresults: list[object]) -> None:
            assert len(subresults) == 1
            self.artifact_files = SimpleNamespace(
                group_report_md="reports/runs/group.md"
            )
            self.payload = SimpleNamespace(
                group_row=SimpleNamespace(
                    to_row=lambda: {"group_run_id": "demo-group"}
                )
            )

        def push_to_hub(self, dataset_id: str) -> None:
            pushed.append((dataset_id, self, []))

    class _FakeNotebookSubruns:
        def __init__(
            self, notebook_name: str, *, dataset: object, model: object
        ) -> None:
            assert notebook_name == "reproduce.ipynb"
            assert isinstance(dataset, _FakeDataset) and dataset.loaded is True
            assert isinstance(model, _FakeModel) and model.loaded is True
            self._subruns = [_FakeSubrun("demo-subrun")]

        def __iter__(self):
            return iter(self._subruns)

        def __len__(self) -> int:
            return len(self._subruns)

        def __getitem__(self, index: int) -> _FakeSubrun:
            return self._subruns[index]

    monkeypatch.setattr(reproduction_api, "CachedHFDataset", _FakeDataset)
    monkeypatch.setattr(
        reproduction_api, "NotebookSubruns", _FakeNotebookSubruns
    )
    monkeypatch.setattr(reproduction_api, "SubrunResult", _FakeSubrunResult)
    monkeypatch.setattr(reproduction_api, "Summary", _FakeSummary)
    monkeypatch.setattr("answer_engineering.GenerationRuntime", _FakeModel)

    globals_dict: dict[str, object] = {"__name__": "__notebook_smoke__"}
    exec(code_cells[0], globals_dict)
    globals_dict["QUESTION_ID"] = "q-1"
    globals_dict["QUESTION_SUBRUN"] = 0
    globals_dict["PUSH_TELEMETRY_TO_HF"] = True

    for cell in code_cells[1:]:
        if cell.lstrip().startswith("!"):
            continue
        try:
            exec(cell, globals_dict)
        except SyntaxError:
            continue

    assert cast(_FakeDataset, globals_dict["dataset"]).loaded is True
    assert cast(_FakeModel, globals_dict["runtime"]).loaded is True
    assert len(cast(list[object], globals_dict["subruns"])) == 1
    assert len(cast(list[object], globals_dict["subresults"])) == 1
    assert globals_dict["single_row"] == {
        "subrun_id": "id-demo-subrun",
        "ruleset_name": "demo-subrun",
        "id": "q-1",
        "case_type": "all",
        "question": "demo-question",
        "gold": "answer:demo-question",
        "ok": True,
        "answer": "answer:demo-question",
    }
    assert pushed and pushed[0][0] == "lavrenko/answer-engineering"
