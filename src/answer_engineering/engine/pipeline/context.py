"""Execution-context contracts and immutable payload carriers.

Purpose:
    Define the canonical data objects that describe one runtime execution step
    and one rule-evaluation context without encoding orchestration flow.

Architectural role:
    Shared execution boundary between orchestration, stages, proposal, scoring,
    and other runtime stages that need per-step or per-rule context.

Contents:
    - immutable concrete request carriers for one execution step
    - immutable rule-evaluation context objects consumed by runtime stages

Invariants:
    Objects in this module describe execution state, not pipeline sequencing.
    They may be created by orchestration, but they are not owned by the
    orchestration control plane and should remain usable by downstream stages
    without importing orchestration internals.

"""

from dataclasses import dataclass

import torch

from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    TextView,
    TokenAlignedTextView,
    TokenCharAlignment,
)
from answer_engineering.engine.telemetry.events.event_sink import (
    RuntimeEventSink,
)
from answer_engineering.inference.model_types import TextCodec
from answer_engineering.rules.compile.plan import PlanIR, RulePlan


@dataclass(frozen=True, slots=True)
class StepSnapshot:
    """Immutable decode snapshot for one engine step before rule-level.

    Purpose:
        Capture assistant-visible text, generated-token ids/alignment, and
        prompt context at a single decode index so downstream orchestration and
        stages read a stable per-step view.

    Architectural role:
        Snapshot value object in the engine pipeline boundary, passed from
        decode orchestration into proposal/scoring/apply flow.

    Inputs:
        Built from ``TokenAlignedTextView`` decode state plus the current token
        index and optional prompt artifacts.

    Outputs:
        Used as the ``step`` portion of ``StepContext`` and by queue/message
        payloads that describe work for this decode step.

    Invariants:
        ``snapshot_text``, ``generated_ids``, and ``generated_token_alignment``
        all describe the same decode-state moment.

    Ownership:
        Owned by ``answer_engineering.engine.pipeline``.

    """

    snapshot_text: str
    generated_ids: tuple[int, ...]
    generated_token_alignment: tuple[TokenCharAlignment, ...]
    token_index: int
    prompt_ids: torch.Tensor | None
    prompt_text: str

    def __init__(
        self,
        *,
        state: TokenAlignedTextView,
        token_index: int,
        prompt_ids: torch.Tensor | None = None,
        prompt_text: str = "",
    ) -> None:
        """Project decode-state fields into an immutable per-step snapshot.

        Purpose:
            Freeze assistant-visible text, generated token ids/alignment, and
            prompt context so downstream stages read stable step data rather
            than live decode-state structures.

        """
        object.__setattr__(
            self,
            "snapshot_text",
            state.assistant_visible_text,
        )

        object.__setattr__(
            self,
            "token_index",
            token_index,
        )

        object.__setattr__(
            self,
            "prompt_ids",
            prompt_ids,
        )

        object.__setattr__(
            self,
            "prompt_text",
            prompt_text,
        )

        object.__setattr__(
            self,
            "generated_ids",
            tuple(state.generated_token_ids),
        )

        object.__setattr__(
            self,
            "generated_token_alignment",
            tuple(state.generated_token_alignment),
        )


@dataclass(frozen=True, slots=True)
class StepContext:
    """Rule-evaluation context for one rule against one decode-step document.

    Purpose:
        Bundle the active plan/rule, document snapshot, scoped guard/edit views,
        and runtime collaborators needed while proposal, scoring, and apply
        stages process that rule.

    Architectural role:
        Cross-stage context carrier inside the engine pipeline boundary.

    Inputs:
        Constructed by proposal-stage orchestration after trigger checks and
        scope-view construction.

    Outputs:
        Consumed by proposal providers, scoring stages, telemetry emission, and
        apply logic during the same decode step.

    Lifecycle:
        Created per eligible rule per step and discarded after that rule's stage
        path completes.

    State:
        The dataclass is frozen and carries read-only references to step-local
        artifacts and collaborators needed for rule evaluation.

    Non-ownership:
        Not a persisted run report and not the mutable decode loop state itself;
        that state lives in decode/orchestration components that construct this
        context.

    Ownership:
        Owned by ``answer_engineering.engine.pipeline``.

    """

    plan: PlanIR
    rule: RulePlan
    doc: DocumentState
    step: StepSnapshot
    guard_view: TextView
    edit_view: TextView
    trajectory_debug: bool = False
    event_sink: RuntimeEventSink | None = None
    tokenizer: TextCodec | None = None
    avoid_edit_floor_abs_start: int | None = None
