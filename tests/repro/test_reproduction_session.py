from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest

from ae_paper_reproduction.api import Dataset, Subrun
from ae_paper_reproduction.runner.session.reproduction_session import (
    ReproductionSession,
)
from answer_engineering import (
    GenerationPolicy,
    GenerationRequest,
    GenerationResult,
    GenerationRuntime,
)
from answer_engineering.telemetry import RuntimeTelemetrySnapshot


def test_reproduction_session_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class _FakeDataset:
        dataset_id = "demo/dataset"
        split = "validation"

        def materialize(self) -> _FakeDataset:
            return self

        def iter_rows(
            self,
        ):  # pragma: no cover - test double for protocol conformance
            return iter(())

        def row(
            self, question_id: str
        ):  # pragma: no cover - test double for protocol conformance
            del question_id
            raise ValueError

        def rows(
            self,
            *,
            n: int | None = None,
            question_id: str | None = None,
            case_type: str | None = None,
        ):
            del n, question_id, case_type
            return ()

        def metadata(self) -> dict[str, str]:
            return {"dataset_id": self.dataset_id, "split": self.split}

    class _FakeRuntime:
        def __init__(self, model_id: str) -> None:
            calls["runtime"] = model_id

        def materialize(self) -> _FakeRuntime:
            calls["runtime_materialized"] = True
            return self

        def generate(
            self, request: GenerationRequest, policy: GenerationPolicy
        ) -> GenerationResult:
            calls["last_policy"] = policy
            return GenerationResult(
                text=f"ans:{request.question}",
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

    @dataclass(frozen=True, slots=True)
    class _FakeTask:
        question: str
        row: object

    class _FakeSubrun:
        def __init__(self) -> None:
            self.name = "demo"
            self.subrun_id = "000-demo"
            self.system_prompt = "sys"
            self.compiled_rules = "## Replace: hearing loss\n\nWith:\n\n- HL\n"
            self.ruleset_name = "demo"
            self.scope_label = "all"
            self.case_type = None
            self.rules_markdown = self.compiled_rules
            self.dataset = SimpleNamespace(
                dataset_id="demo/dataset", split="validation"
            )
            self.model = SimpleNamespace(
                model_id="demo/model", max_new_tokens=64
            )

        def select_tasks(
            self, *, n: int | None = None, question_id: str | None = None
        ) -> list[_FakeTask]:
            del question_id
            assert n == 2
            return [
                _FakeTask(
                    question="q1",
                    row=SimpleNamespace(
                        id="1", case_type="all", question="q1", gold="ans:q1"
                    ),
                ),
                _FakeTask(
                    question="q2",
                    row=SimpleNamespace(
                        id="2", case_type="all", question="q2", gold="ans:q2"
                    ),
                ),
            ]

    def _summary_from_results(subresults: object) -> SimpleNamespace:
        return SimpleNamespace(subresults=subresults)

    monkeypatch.setattr(
        "ae_paper_reproduction.runner.session.reproduction_session.Summary",
        _summary_from_results,
    )

    runtime = _FakeRuntime(model_id="demo/model")
    session = ReproductionSession(
        dataset=cast(Dataset, _FakeDataset()),
        runtime=cast(GenerationRuntime, runtime),
        subruns=cast(tuple[Subrun, ...], (_FakeSubrun(),)),
    )
    summary = session.run(n_eval=2, verbosity=0)

    assert calls["runtime"] == "demo/model"
    assert len(summary.subresults) == 1
    assert summary.subresults[0].n_eval_requested == 2
