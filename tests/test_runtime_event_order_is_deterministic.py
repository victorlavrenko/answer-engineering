from __future__ import annotations

from dataclasses import asdict

from answer_engineering.engine.orchestration.orchestrator import (
    OrchestratorResult,
    PlanRunner,
)
from answer_engineering.rules.compile.compiler import (
    FullPlanCompiler,
)
from answer_engineering.rules.parse.parser import (
    MarkdownRulesParser,
)
from tests._support.core_helpers import (
    create_step_snapshot,
)


def _event_trace(
    result: OrchestratorResult,
) -> list[tuple[str, dict[str, object]]]:
    traces: list[tuple[str, dict[str, object]]] = []
    for event in result.events:
        payload = asdict(event)
        payload.pop("event_id", None)
        payload.pop("ts", None)
        traces.append((type(event).__name__, payload))
    return traces


def test_runtime_event_order_is_deterministic() -> None:
    md = """## Avoid (once): conductive

Connector:

* this suggests

Postfix (any):

* conductive

Fallback:

* these findings require further evaluation.
"""
    plan = FullPlanCompiler().compile(MarkdownRulesParser().parse(md))
    text = "this suggests conductive hearing loss."

    first = PlanRunner(verbose=False).run(
        plan, create_step_snapshot(snapshot_text=text, token_index=10)
    )
    second = PlanRunner(verbose=False).run(
        plan, create_step_snapshot(snapshot_text=text, token_index=10)
    )

    assert first.final_doc.text == second.final_doc.text
    assert _event_trace(first) == _event_trace(second)
