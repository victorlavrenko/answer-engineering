"""Guard-expression evaluation and avoid-overlap actionability analysis.

Purpose:
    Evaluate compiled match-tree guards against a scoped view, flatten telemetry
    observations, and decide whether avoid-rule overlap is actionable enough to
    edit.

Architectural role:
    Guard-semantics layer between compiled rule expressions and proposal
    prechecks.

Inputs (architectural provenance):
    Receives scoped text views, compiled ``MatchTree`` guards, anchor spans, and
    proposed edit spans from proposal logic.

Outputs (downstream usage):
    Produces guard pass/fail results, structured observations, and overlap
    diagnostics consumed by ``GenerationPrecheck`` and telemetry.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from answer_engineering.engine.proposal.match_tree.match_tree import (
    MatchAndThen,
    MatchResult,
    MatchTree,
    NodeTelemetry,
)
from answer_engineering.engine.runtime.runtime_types import (
    Anchor,
    TextView,
)
from answer_engineering.rules.compile.plan import (
    GuardSpec,
)


@dataclass(frozen=True, slots=True)
class GuardNodeObservation:
    """Flattened observation for one guard node during evaluation.

    Purpose:
        Preserve a telemetry-friendly record of node identity, expression,
        spans, and match outcome.

    Architectural role:
        Diagnostic value object emitted by guard evaluation.

    Inputs (architectural provenance):
        Constructed from ``NodeTelemetry`` trees produced by ``MatchTree``
        nodes.

    Outputs (downstream usage):
        Consumed by telemetry events, debugging, and precheck reporting.

    """

    node_id: str
    node_path: str
    node_type: str
    marker: str | None
    debug_expression: str
    matched: bool
    spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class GuardMatchState:
    """Structured result of evaluating a guard against one text view.

    Purpose:
        Carry the top-level boolean result together with detailed node
        observations and raw match telemetry.

    Architectural role:
        Internal result record between guard evaluation and proposal prechecks.

    Inputs (architectural provenance):
        Built by ``evaluate_guard`` from match-tree evaluation over the guard
        view.

    Outputs (downstream usage):
        Consumed by overlap analysis and telemetry emission.

    """

    overlap_telemetry: NodeTelemetry | None


@dataclass(frozen=True, slots=True, init=False)
class AvoidOverlapEvaluation:
    """Assessment of whether an avoid-rule overlap is actionable enough to edit.

    Purpose:
        Explain whether the proposed span overlaps the positive part of the
        guard in a way that justifies an avoid edit.

    Architectural role:
        Specialized diagnostic record for avoid-rule prechecks.

    Inputs (architectural provenance):
        Built from guard telemetry and the candidate edit span during precheck.

    Outputs (downstream usage):
        Consumed by ``GenerationPrecheck`` to keep or suppress avoid proposals.

    """

    has_actionable_overlap: bool

    def __init__(
        self,
        *,
        overlap_telemetry: NodeTelemetry | None,
        span: tuple[int, int],
    ) -> None:
        """Assess overlap actionability from canonical overlap inputs.

        Purpose:
            Decide whether an avoid-rule overlap is actionable and preserve the
            coordinates needed to explain that decision.

        Architectural role:
            Constructor for a proposal-guard value object. It centralizes
            overlap normalization so downstream guard logic reads a complete,
            valid result.

        Inputs (architectural provenance):
            Receives canonical overlap spans and match context produced by avoid
            guard matching.

        Outputs (downstream usage):
            Initializes fields consumed by proposal filtering, debug messages,
            and guard telemetry.

        Invariants/constraints:
            Construction should leave no half-normalized overlap state.
            Actionability and span provenance must stay consistent for later
            diagnostics.

        """
        has_actionable_overlap = span[0] < span[1] and _has_actionable_overlap(
            overlap_telemetry, span
        )
        object.__setattr__(
            self, "has_actionable_overlap", has_actionable_overlap
        )

    @property
    def sufficient(self) -> bool:
        """Return whether the overlap analysis permits an avoid edit to proceed.

        Purpose:
            Provide the boolean policy surface over the richer overlap
            diagnostic data.

        Architectural role:
            Readable convenience property on ``AvoidOverlapEvaluation``.

        Inputs (architectural provenance):
            Reads fields already computed by avoid-overlap analysis.

        Outputs (downstream usage):
            Boolean is consumed by precheck logic when deciding to emit a noop
            reason.

        """
        return self.has_actionable_overlap


def evaluate_guard(
    guard_view: TextView,
    guard: GuardSpec | None,
    anchors: dict[str, Anchor],
    *,
    prompt_text: str = "",
    casefold: bool,
) -> tuple[bool, tuple[GuardNodeObservation, ...], GuardMatchState | None]:
    """Evaluate a compiled guard against the current guard view and collect.

    Purpose:
        Resolve whether the rule guard matches, while preserving node-level
        telemetry and anchor-aware diagnostics.

    Architectural role:
        Main entry point from proposal precheck into guard semantics.

    Inputs (architectural provenance):
        Receives the scoped ``TextView``, compiled ``MatchTree``, resolved
        anchors, and prompt text from ``StepContext``.

    Outputs (downstream usage):
        Returns the top-level boolean, flattened observations, and detailed
        match state for later overlap checks.

    """
    del anchors
    if guard is None or guard.expression is None:
        return True, tuple(), None

    tree_result = _evaluate_guard_expression(
        guard.expression,
        guard_text=guard_view.text,
        prompt_text=prompt_text,
        casefold=casefold,
    )
    observations = tuple(
        _flatten_observations(tree_result.telemetry, path="guard")
    )
    overlap_subtree = (
        guard.expression.ordered_overlap_subtree() or guard.expression
    )
    match_state = GuardMatchState(
        overlap_telemetry=overlap_subtree.evaluate(
            guard_view.text, casefold=casefold
        ).telemetry
    )
    return tree_result.matched, observations, match_state


def _evaluate_guard_expression(
    expression: MatchTree,
    *,
    guard_text: str,
    prompt_text: str,
    casefold: bool,
) -> MatchResult:
    """Evaluate one guard expression node and wrap its telemetry into match.

    Purpose:
        Isolate the raw ``MatchTree.evaluate`` call from the surrounding
        flattening and result-packaging logic.

    Architectural role:
        Internal helper inside guard evaluation.

    Inputs (architectural provenance):
        Receives a compiled guard expression and the scoped text on which to run
        it.

    Outputs (downstream usage):
        Returns a ``GuardMatchState`` used by higher-level evaluation helpers.

    """
    if (
        isinstance(expression, MatchAndThen)
        and expression.marker == "prompt_answer_boundary"
    ):
        left = expression.left.evaluate(prompt_text, casefold=casefold)
        right = expression.right.evaluate(guard_text, casefold=casefold)
        matched = left.matched and right.matched
        telemetry = NodeTelemetry(
            node_type="MatchAndThen",
            marker=expression.marker,
            matched=matched,
            spans=right.spans if matched else (),
            children=(left.telemetry, right.telemetry),
        )
        return MatchResult(
            matched=matched,
            spans=(right.spans if matched else ()),
            telemetry=telemetry,
        )
    return expression.evaluate(guard_text, casefold=casefold)


def spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    """Return whether two half-open spans overlap.

    Purpose:
        Provide a single overlap primitive reused by guard and overlap-analysis
        helpers.

    Architectural role:
        Low-level span utility inside guard matching.

    Inputs (architectural provenance):
        Receives two half-open spans expressed in the same coordinate system.

    Outputs (downstream usage):
        Boolean feeds overlap-analysis decisions.

    """
    return left[0] < right[1] and right[0] < left[1]


def _flatten_observations(
    telemetry: NodeTelemetry,
    *,
    path: str,
) -> Sequence[GuardNodeObservation]:
    """Flatten recursive guard telemetry into a linear list of node.

    Purpose:
        Convert the tree-shaped ``NodeTelemetry`` emitted by match-tree
        evaluation into path-addressed records suitable for telemetry events.

    Architectural role:
        Telemetry adapter inside guard evaluation.

    Inputs (architectural provenance):
        Receives the root telemetry tree and the current path prefix.

    Outputs (downstream usage):
        Produces ordered ``GuardNodeObservation`` records for runtime events and
        debugging.

    """
    node_id = path
    expression = telemetry.expression or telemetry.node_type
    observations = [
        GuardNodeObservation(
            node_id=node_id,
            node_path=path,
            node_type=telemetry.node_type,
            marker=telemetry.marker,
            debug_expression=expression,
            matched=telemetry.matched,
            spans=tuple((s.start, s.end) for s in telemetry.spans),
        )
    ]
    for idx, child in enumerate(telemetry.children):
        observations.extend(_flatten_observations(child, path=f"{path}.{idx}"))
    return observations


def _has_actionable_overlap(
    telemetry: NodeTelemetry | None, span: tuple[int, int]
) -> bool:
    """Return whether telemetry contains positive evidence overlapping the.

    Purpose:
        Walk the guard telemetry tree while honoring special handling for
        negation and conjunction nodes in avoid-overlap analysis.

    Architectural role:
        Recursive helper owned by avoid-overlap evaluation.

    Inputs (architectural provenance):
        Receives a telemetry subtree and the candidate span being tested.

    Outputs (downstream usage):
        Boolean contributes to ``AvoidOverlapEvaluation`` construction.

    """
    if telemetry is None:
        return False
    if telemetry.node_type == "MatchAll":
        positive_children = [
            child
            for child in telemetry.children
            if child.node_type != "MatchNot"
        ]
        if not positive_children:
            return False
        return all(
            _has_actionable_overlap(child, span) for child in positive_children
        )
    if telemetry.node_type == "MatchAny":
        positive_children = [
            child
            for child in telemetry.children
            if child.node_type != "MatchNot"
        ]
        return any(
            _has_actionable_overlap(child, span) for child in positive_children
        )
    if telemetry.node_type == "MatchNot":
        return False
    if telemetry.node_type == "MatchAndThen":
        if len(telemetry.children) < 2:
            return False
        return _has_actionable_overlap(telemetry.children[1], span)
    return any(spans_overlap((s.start, s.end), span) for s in telemetry.spans)
