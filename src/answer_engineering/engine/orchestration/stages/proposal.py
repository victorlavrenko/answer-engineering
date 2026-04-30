"""Stage that expands step requests into proposal batches.

Purpose:
    Build ``StepContext`` values per eligible rule and invoke proposal
    generation for each rule in plan order.

Architectural role:
    First runtime stage stage after step request.

"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from answer_engineering.engine.pipeline.context import (
    StepContext,
)
from answer_engineering.engine.pipeline.messages import (
    ProposalsReady,
    StepRequested,
)
from answer_engineering.engine.proposal import proposal_logic
from answer_engineering.engine.proposal.proposal_engine import (
    ProposalPlanner,
)
from answer_engineering.engine.runtime.runtime_types import (
    DocumentState,
    PatchOp,
    TextView,
)
from answer_engineering.engine.telemetry.events.event_sink import (
    DebugEventEmitter,
    RuntimeEventSink,
)
from answer_engineering.rules.compile.plan import RulePlan


def _precheck_span_end(
    triggered_ctx: ProposalStage.TriggeredRuleContext,
) -> int:
    """Return the end offset used for trigger-bucket ordering.

    Purpose:
        Extract a stable span-end sort key from a proposal precheck when a rule
        has already passed trigger-stage evaluation.

    """
    if triggered_ctx.precheck.span is not None:
        return triggered_ctx.precheck.span[1]
    return triggered_ctx.edit_start


def _bucket_by_edit_start(
    triggered: list[ProposalStage.TriggeredRuleContext],
) -> dict[int, list[ProposalStage.TriggeredRuleContext]]:
    buckets: dict[int, list[ProposalStage.TriggeredRuleContext]] = defaultdict(
        list
    )
    for item in triggered:
        buckets[item.edit_start].append(item)
    return dict(buckets)


@dataclass(slots=True)
class ProposalStage:
    """Stage that expands one step request into proposal batches.

    Purpose:
        Build per-rule StepContext values, run trigger-stage filtering, bucket
        triggered rules by edit start, and invoke ProposalPlanner for the
        relevant bucket.

    Architectural role:
        First runtime stage stage after step request and before scoring.

    Outputs (downstream usage):
        Emits the proposal batch that downstream scoring and selection stages
        evaluate.

    """

    proposal_engine: ProposalPlanner
    trajectory_debug: bool = False
    event_sink: RuntimeEventSink | None = None
    debug_emitter: DebugEventEmitter = field(default_factory=DebugEventEmitter)

    @dataclass(frozen=True, slots=True, init=False)
    class TriggeredRuleContext:
        """Rule context that passed trigger-stage precheck.

        Purpose:
            Pair a StepContext with its successful GenerationPrecheck and expose
            the edit-start key used for bucket ordering.

        Architectural role:
            Internal transport record inside ProposalStage's two-stage pipeline.

        """

        ctx: StepContext
        precheck: proposal_logic.GenerationPrecheck
        edit_start: int
        is_triggered: bool

        def __init__(self, ctx: StepContext) -> None:
            """Initialize one triggered-rule context and derive its bucket key.

            Purpose:
                Cache the rule's step context together with its proposal
                precheck so the stage can group triggered rules by effective
                edit start before proposal generation.

            Architectural role:
                Construction-time setup for the stage's trigger-first scheduling
                record.

            Inputs (architectural provenance):
                Receives the per-rule ``StepContext`` built by the proposal
                stage.

            Outputs (downstream usage):
                Stores the precheck result and exposes the edit-start key used
                for bucketed proposal scheduling.

            """
            precheck = proposal_logic.GenerationPrecheck(ctx)
            has_generated_candidates = any(
                candidate.kind == "generated"
                for candidate in ctx.rule.candidates
            )
            already_satisfied_for_static = (
                not has_generated_candidates
            ) and proposal_logic.already_satisfied(
                ctx.edit_view,
                ctx.rule.candidates,
                casefold_compare=ctx.rule.effective_edit_scope().casefold,
            )
            is_triggered = (
                precheck.noop_reason is None
                and precheck.span is not None
                and not already_satisfied_for_static
            )
            object.__setattr__(self, "ctx", ctx)
            object.__setattr__(self, "precheck", precheck)
            object.__setattr__(
                self, "edit_start", precheck.span[0] if precheck.span else -1
            )
            object.__setattr__(self, "is_triggered", is_triggered)

    def handle(self, event: StepRequested) -> list[ProposalsReady]:
        """Generate proposals using a trigger-first, bucketed proposal pipeline.

        Purpose:
            First keep only rules whose trigger-stage precheck passes, then
            group the triggered rules by edit start and process buckets from
            earliest start.

        Architectural role:
            Main scheduling method of ProposalStage.

        Outputs (downstream usage):
            Returns the first bucket that yields at least one editable proposal.
            Buckets that produce only noop proposals are skipped in favor of
            later buckets.

        """
        self.debug_emitter.event_sink = self.event_sink
        triggered_contexts = self._collect_triggered_contexts(event)
        if not triggered_contexts:
            return list()

        bucketed = _bucket_by_edit_start(triggered_contexts)
        self._debug(
            "PRECHECK_BUCKETS "
            + ", ".join(
                f"{edit_start}:"
                + "["
                + ",".join(
                    item.ctx.rule.rule_id for item in bucketed[edit_start]
                )
                + "]"
                for edit_start in sorted(bucketed)
            ),
        )
        out: list[ProposalsReady] = []
        for edit_start in sorted(bucketed):
            bucket_groups: list[ProposalsReady] = []
            bucket_has_editable = False
            for triggered in bucketed[edit_start]:
                proposals = self.proposal_engine.generate(triggered.ctx)
                bucket_groups.append(
                    ProposalsReady(ctx=triggered.ctx, proposals=proposals)
                )
                if any(proposal.op != PatchOp.NOOP for proposal in proposals):
                    bucket_has_editable = True
            out.extend(bucket_groups)
            if bucket_has_editable:
                return out
        return out

    def _collect_triggered_contexts(
        self, event: StepRequested
    ) -> list[TriggeredRuleContext]:
        triggered: list[ProposalStage.TriggeredRuleContext] = []
        for rule in event.plan.rules:
            if event.execution.token_index < rule.policy.skip_tokens:
                continue
            ctx = self._build_context(event, rule=rule)
            triggered_ctx = self.TriggeredRuleContext(ctx)
            if not triggered_ctx.is_triggered:
                continue
            self._debug(
                "PRECHECK_TRIGGER "
                f"rule_id={rule.rule_id} "
                f"span={triggered_ctx.edit_start}:"
                f"{_precheck_span_end(triggered_ctx)} "
                f"edit_view={ctx.edit_view.abs_start}:{ctx.edit_view.abs_end} "
                f"bucket_start={triggered_ctx.edit_start}",
            )
            triggered.append(triggered_ctx)
        return triggered

    def _debug(self, msg: str) -> None:
        """Emit a stage debug message when trajectory debugging is enabled.

        Purpose:
            Keep ProposalStage-specific debug routing local so scheduling code
            can emit diagnostics without repeating the trajectory-debug guard.

        Architectural role:
            Small telemetry helper owned by the proposal stage.

        Inputs (architectural provenance):
            Receives the active step context and a stage-generated debug
            message.

        Outputs (downstream usage):
            May forward the message to the configured debug emitter; otherwise
            produces no effect.

        """
        if not self.trajectory_debug:
            return
        self.debug_emitter.emit(msg)

    def _build_context(
        self, event: StepRequested, *, rule: RulePlan
    ) -> StepContext:
        guard_view, edit_view = self._build_views(doc=event.doc, rule=rule)
        return StepContext(
            plan=event.plan,
            rule=rule,
            doc=event.doc,
            step=event.execution,
            guard_view=guard_view,
            edit_view=edit_view,
            trajectory_debug=self.trajectory_debug,
            event_sink=self.event_sink,
            tokenizer=event.tokenizer,
        )

    def _build_views(
        self, *, doc: DocumentState, rule: RulePlan
    ) -> tuple[TextView, TextView]:
        guard_scope = rule.effective_guard_scope()
        edit_scope = rule.effective_edit_scope()
        guard_view = TextView(doc, guard_scope)
        edit_view = TextView(doc, edit_scope)
        return guard_view, edit_view
