"""Shared proposal semantics used across rule families.

Purpose:
    Resolve anchors, evaluate guards, compute target spans, detect no-op cases,
    and build proposal candidates with deterministic ordering.

Architectural role:
    Core proposal-semantics module invoked by ProposalPlanner and candidate
    providers.

Current architecture notes:
    This module centralizes the real semantics of proposal generation. It is
    also imported more widely than an ideal closed proposal boundary would
    expose.

Architectural TODO:
    Keep this logic authoritative inside proposal while reducing the amount of
    cross-subsystem code that imports its internal helpers and precheck record
    directly.

"""

from __future__ import annotations

import hashlib
import logging
import re
import string
from collections.abc import Sequence
from dataclasses import dataclass

from answer_engineering.engine.patching import patch_canonical
from answer_engineering.engine.patching.proposals import (
    PatchProposal,
    ProposalContext,
)
from answer_engineering.engine.pipeline import text_bounds, text_patterns
from answer_engineering.engine.pipeline.context import (
    StepContext,
)
from answer_engineering.engine.proposal.guards import guard_matching
from answer_engineering.engine.proposal.guards.guard_matching import (
    GuardNodeObservation,
)
from answer_engineering.engine.runtime import text_alignment
from answer_engineering.engine.runtime.runtime_types import (
    Anchor,
    PatchOp,
    TextView,
    TokenCharAlignment,
)
from answer_engineering.engine.span_utils import (
    describe_span,
    is_valid_span,
    normalize_span,
    validate_token_alignment,
)
from answer_engineering.inference.model_types import TextCodec
from answer_engineering.rules.compile.plan import (
    AnchorQuerySpec,
    CandidateSpec,
    EditTargetSpec,
)

_PARENTHETICAL_SUFFIX_PATTERN = re.compile(r"[,.!?*]+")
_LOG = logging.getLogger(__name__)


def _candidate_hash(spec: CandidateSpec) -> str:
    """Compute a stable hash for one candidate specification.

    Purpose:
        Give candidate identity a deterministic fingerprint based on operation,
        payload text, kind, label, and explicit candidate id.

    Architectural role:
        Identity helper shared by proposal generation and attempt tracking.

    Inputs (architectural provenance):
        Receives a ``CandidateSpec`` selected by a provider.

    Outputs (downstream usage):
        Returns a SHA-1 digest used for duplicate detection and proposal
        metadata.

    """
    raw = (
        f"{spec.op.value}|{spec.text}|{spec.kind}|"
        f"{spec.label}|{spec.candidate_id}"
    ).encode()
    return hashlib.sha1(raw).hexdigest()


def already_satisfied(
    view: TextView,
    candidates: Sequence[CandidateSpec],
    *,
    casefold_compare: bool,
) -> bool:
    """Return whether scoped text already contains one candidate payload."""
    scope_text = view.text.casefold() if casefold_compare else view.text
    replace_payloads = [
        candidate.text
        for candidate in candidates
        if candidate.op == PatchOp.REPLACE and candidate.text
    ]
    if casefold_compare:
        replace_payloads = [payload.casefold() for payload in replace_payloads]
    if replace_payloads and any(
        payload in scope_text for payload in replace_payloads
    ):
        return True
    insert_payloads = [
        candidate.text
        for candidate in candidates
        if candidate.text
        and candidate.op in {PatchOp.INSERT_AFTER, PatchOp.INSERT_BEFORE}
    ]
    if casefold_compare:
        insert_payloads = [payload.casefold() for payload in insert_payloads]
    return bool(
        insert_payloads
        and any(payload in scope_text for payload in insert_payloads)
    )


@dataclass(frozen=True, slots=True, init=False)
class GenerationPrecheck:
    """Immutable summary of proposal prechecks for one rule evaluation.

    Purpose:
        Resolve anchors, evaluate the guard, compute the target span, and decide
        whether proposal generation may proceed.

    Architectural role:
        Gatekeeping record at the front of shared proposal logic.

    Inputs (architectural provenance):
        Built from StepContext by running proposal-side prechecks.

    Outputs (downstream usage):
        Consumed by ProposalPlanner and candidate providers to either
        short-circuit generation or continue with a resolved span and guard
        observations.

    Invariants/constraints:
        When noop_reason is not None, span is absent and proposal generation
        must short-circuit.

    Architectural TODO:
        Keep this record proposal-owned and avoid leaking it as a long-lived
        dependency into probing-facing APIs.

    """

    anchors: dict[str, Anchor]
    span: tuple[int, int] | None
    noop_reason: str | None
    guard_observations: tuple[GuardNodeObservation, ...] = tuple()

    def __init__(self, ctx: StepContext) -> None:
        """Run guard, anchor, and target-span prechecks for one step context.

        Purpose:
            Construct a complete GenerationPrecheck while enforcing the record's
            short-circuit invariants.

        """
        rule = ctx.rule
        guard_view = ctx.guard_view

        anchors = _resolve_anchors(guard_view, rule.anchors)
        guard_ok, guard_observations, match_state = (
            guard_matching.evaluate_guard(
                guard_view,
                rule.guard,
                anchors,
                prompt_text=ctx.step.prompt_text,
                casefold=rule.effective_guard_scope().casefold,
            )
        )
        if not guard_ok:
            object.__setattr__(self, "anchors", anchors)
            object.__setattr__(self, "span", None)
            object.__setattr__(self, "noop_reason", "guard failed")
            object.__setattr__(self, "guard_observations", guard_observations)
            return

        span = _compute_target_span(ctx, rule.target, anchors)
        if span is None:
            object.__setattr__(self, "anchors", anchors)
            object.__setattr__(self, "span", None)
            object.__setattr__(self, "noop_reason", "no target span")
            object.__setattr__(self, "guard_observations", guard_observations)
            return

        anchor_span: tuple[int, int] | None = None
        if rule.target.anchor_id:
            anchor = anchors.get(rule.target.anchor_id)
            if anchor is not None:
                anchor_span = (anchor.abs_start, anchor.abs_end)
        span, deferred_reason = (
            _maybe_extend_after_span_until_parenthesis_close(
                ctx=ctx,
                span=span,
                anchor_span=anchor_span,
            )
        )
        if deferred_reason is not None:
            object.__setattr__(self, "anchors", anchors)
            object.__setattr__(self, "span", None)
            object.__setattr__(self, "noop_reason", deferred_reason)
            object.__setattr__(self, "guard_observations", guard_observations)
            return

        local_span = (
            span[0] - guard_view.abs_start,
            span[1] - guard_view.abs_start,
        )
        overlap_eval: guard_matching.AvoidOverlapEvaluation | None = None
        if (
            rule.name.startswith("avoid:")
            and rule.guard is not None
            and rule.guard.expression is not None
            and match_state is not None
        ):
            overlap_eval = guard_matching.AvoidOverlapEvaluation(
                overlap_telemetry=match_state.overlap_telemetry,
                span=local_span,
            )
        if overlap_eval is not None and not overlap_eval.sufficient:
            object.__setattr__(self, "anchors", anchors)
            object.__setattr__(self, "span", None)
            object.__setattr__(
                self, "noop_reason", "insufficient overlap actionability"
            )
            object.__setattr__(self, "guard_observations", guard_observations)
            return

        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "span", span)
        object.__setattr__(self, "noop_reason", None)
        object.__setattr__(self, "guard_observations", guard_observations)


@dataclass(frozen=True, slots=True)
class StandardProposalGenerator:
    """Standard proposal generator following shared semantics for one rule.

    Purpose:
        Generate patch proposals for one rule evaluation by applying shared
        semantics around candidate sorting, span snapping, payload
        normalization, and no-op filtering.

    Architectural role:
        Default proposal generator used by all rule families that do not require
        custom generation logic.

    Inputs (architectural provenance):
        Receives the StepContext and candidate specifications for one rule
        evaluation.

    Outputs (downstream usage):
        Returns the generated patch proposals and an optional generation status
        string used for tracking and logging.

    Architectural TODO:
        Keep this generator proposal-owned and avoid leaking it as a long-lived
        dependency into probing-facing APIs.

    """

    def generate(
        self, ctx: StepContext, candidates: Sequence[CandidateSpec]
    ) -> tuple[list[PatchProposal], str | None]:
        """Build canonical patch proposals from candidate specifications.

        Purpose:
            Apply shared proposal semantics for one rule: prechecks,
            already-satisfied detection, span flooring, candidate ordering, span
            snapping, no-op filtering, and proposal construction.

        Architectural role:
            Core generation method of the standard proposal boundary.

        Inputs (architectural provenance):
            Receives the current `StepContext` from orchestration and candidate
            specs from compiled rule plans or candidate providers.

        Outputs (downstream usage):
            Returns patch proposals for scoring plus an optional status string
            used by planner telemetry and debug output.

        Invariants/constraints:
            Returned proposals must all share the same proposal context and must
            refer to spans valid for the current document version. Candidate
            ordering is stable so equal inputs produce reproducible proposal
            sequences.

        """
        rule = ctx.rule
        doc = ctx.doc
        edit_view = ctx.edit_view
        precheck = GenerationPrecheck(ctx)
        proposal_context = ProposalContext(
            base_version_id=ctx.doc.version_id,
            rule_id=ctx.rule.rule_id,
            guard_abs_start=ctx.guard_view.abs_start,
        )
        if precheck.noop_reason is not None:
            return [
                PatchProposal.noop(
                    context=proposal_context, reason=precheck.noop_reason
                )
            ], None
        assert precheck.span is not None
        if not candidates:
            return [
                PatchProposal.noop(
                    context=proposal_context, reason="no candidates"
                )
            ], None
        casefold_compare = ctx.rule.effective_edit_scope().casefold
        if already_satisfied(
            edit_view,
            candidates,
            casefold_compare=casefold_compare,
        ):
            return [
                PatchProposal.noop(
                    context=proposal_context, reason="already_satisfied"
                )
            ], "already_satisfied"
        span = precheck.span
        if ctx.avoid_edit_floor_abs_start is not None:
            floored_start = max(span[0], ctx.avoid_edit_floor_abs_start)
            if floored_start < span[1]:
                span = (floored_start, span[1])
        proposals: list[PatchProposal] = []
        for candidate_index, chosen in enumerate(
            sorted(candidates, key=lambda c: (-c.priority, c.label, c.text))
        ):
            span_for_candidate = _snap_span_to_token_boundaries(
                ctx=ctx,
                span=span,
                op=chosen.op,
            )
            if span_for_candidate is None:
                _LOG.warning(
                    "invalid_span_dropped rule_id=%s rule_name=%r candidate=%r",
                    rule.rule_id,
                    rule.name,
                    chosen.label,
                )
                continue
            if not is_valid_span(span_for_candidate, doc.text):
                fixed = normalize_span(
                    span_for_candidate,
                    doc.text,
                    fallback=span,
                    mode="fallback_then_clamp",
                )
                if fixed.span is None:
                    _LOG.warning(
                        (
                            "invalid_span_dropped rule_id=%s "
                            "rule_name=%r candidate=%r span=%s"
                        ),
                        rule.rule_id,
                        rule.name,
                        chosen.label,
                        span_for_candidate,
                    )
                    continue
                _LOG.warning(
                    "%s rule_id=%s original_span=%s corrected_span=%s",
                    fixed.reason or "invalid_span_clamped",
                    rule.rule_id,
                    span_for_candidate,
                    fixed.span,
                )
                span_for_candidate = fixed.span
            candidate_text = _maybe_adjust_after_parenthetical_candidate(
                text=doc.text,
                span=span_for_candidate,
                candidate_text=chosen.text,
                op=chosen.op,
                rule_name=rule.name,
            )
            payload_norm = patch_canonical.canonicalize_payload(
                op=chosen.op,
                payload=candidate_text,
                text=doc.text,
                span_abs=span_for_candidate,
                apply_spacing=chosen.kind != "generated",
            )
            if not _would_change(
                text=doc.text,
                op=chosen.op,
                span=span_for_candidate,
                payload_norm=payload_norm,
                casefold_compare=casefold_compare,
            ):
                continue
            candidate_id = (
                chosen.candidate_id or f"candidate_{candidate_index + 1}"
            )
            proposals.append(
                PatchProposal.from_candidate(
                    op=chosen.op,
                    candidate=chosen,
                    context=proposal_context,
                    span_abs=span_for_candidate,
                    payload=candidate_text,
                    payload_norm=payload_norm,
                    reason=(
                        f"valid edit name={rule.name} "
                        f"candidate={chosen.label} "
                        f"candidate_id={candidate_id} "
                        f"target={rule.target.kind} "
                        f"guard_scope={rule.effective_guard_scope().kind} "
                        f"edit_scope={rule.effective_edit_scope().kind}"
                    ),
                    candidate_index=candidate_index,
                    candidate_hash=_candidate_hash(chosen),
                )
            )
        return proposals or [
            PatchProposal.noop(context=proposal_context, reason="already fired")
        ], None


def _resolve_anchors(
    guard_view: TextView, specs: Sequence[AnchorQuerySpec]
) -> dict[str, Anchor]:
    """Resolve anchor queries against the current guard view.

    Purpose:
        Find the best span for each anchor id using the configured first or last
        match policy across the allowed phrases.

    Architectural role:
        Anchor-resolution helper inside proposal precheck.

    Inputs (architectural provenance):
        Receives the scoped guard ``TextView`` and compiled ``AnchorQuerySpec``
        objects from the active rule.

    Outputs (downstream usage):
        Returns resolved absolute ``Anchor`` records consumed by guard
        evaluation and target-span computation.

    """
    out: dict[str, Anchor] = {}
    for spec in specs:
        best: tuple[int, int] | None = None
        phrase_options = (
            spec.match_phrase_options
            if spec.match_phrase_options
            else tuple(
                (phrase, spec.match_options) for phrase in spec.match_phrase_any
            )
        )
        for phrase, match_options in phrase_options:
            match = text_patterns.search_span(
                guard_view.text,
                phrase,
                match_options=match_options,
            )
            if match is None:
                continue
            match_start, match_end = match
            if spec.match_mode == "first":
                if best is None:
                    best = match
                    continue
                best_start, best_end = best
                if (match_start, -match_end) < (best_start, -best_end):
                    best = match
            else:
                if best is None:
                    best = match
                    continue
                best_start, best_end = best
                if (match_end, -match_start) > (best_end, -best_start):
                    best = match
        if best is None:
            continue
        best_start, best_end = best
        span = guard_view.to_abs_span(best_start, best_end)
        out[spec.anchor_id] = Anchor(
            anchor_id=spec.anchor_id, abs_start=span[0], abs_end=span[1]
        )
    return out


def _compute_target_span(
    ctx: StepContext, target: EditTargetSpec, anchors: dict[str, Anchor]
) -> tuple[int, int] | None:
    """Compute the absolute edit span implied by the resolved target.

    Purpose:
        Translate target semantics such as whole-scope, anchor span, sentence
        end, or clause end into one concrete absolute span.

    """
    doc = ctx.doc
    edit_view = ctx.edit_view
    anchor = anchors.get(target.anchor_id) if target.anchor_id else None
    if (
        target.kind
        in {
            "match_span",
            "after_anchor_to_scope_end",
            "clause_containing_anchor_to_scope_end",
            "after_anchor_to_sentence_end",
            "after_anchor_to_clause_end",
        }
        and anchor is None
    ):
        return None
    if anchor is not None and (
        anchor.abs_end <= edit_view.abs_start
        or anchor.abs_start >= edit_view.abs_end
    ):
        return None

    if target.kind == "scope_entire":
        start, end = edit_view.abs_start, edit_view.abs_end
    elif target.kind == "match_span":
        assert anchor is not None
        start, end = anchor.abs_start, anchor.abs_end
    elif target.kind == "after_anchor_to_scope_end":
        assert anchor is not None
        start = anchor.abs_start if target.include_anchor else anchor.abs_end
        end = edit_view.abs_end
    elif target.kind == "clause_containing_anchor_to_scope_end":
        assert anchor is not None
        start = text_bounds.find_clause_start(
            doc.text, anchor.abs_start, edit_view.abs_start
        )
        end = edit_view.abs_end
    elif target.kind == "after_anchor_to_sentence_end":
        assert anchor is not None
        start = anchor.abs_start if target.include_anchor else anchor.abs_end
        end = text_bounds.find_sentence_end(doc.text, start, edit_view.abs_end)
    elif target.kind == "after_anchor_to_clause_end":
        assert anchor is not None
        start = anchor.abs_start if target.include_anchor else anchor.abs_end
        end = text_bounds.find_clause_end(doc.text, start, edit_view.abs_end)
    else:
        return None

    start = max(start, edit_view.abs_start)
    end = min(end, edit_view.abs_end)
    if start > end:
        return None
    return (start, end)


def _snap_span_to_token_boundaries(
    *, ctx: StepContext, span: tuple[int, int], op: PatchOp
) -> tuple[int, int] | None:
    """Snap an edit span to safe token boundaries, or drop if unsafe."""
    text = ctx.doc.text
    original_result = normalize_span(span, text, mode="clamp")
    if original_result.span is None:
        _LOG.warning(
            "invalid_span_dropped rule_id=%s rule_name=%r %s",
            ctx.rule.rule_id,
            ctx.rule.name,
            describe_span(span, text),
        )
        return None
    if original_result.changed:
        _LOG.warning(
            "%s rule_id=%s rule_name=%r original_span=%s corrected_span=%s "
            "doc_len=%s %s",
            original_result.reason or "invalid_span_clamped",
            ctx.rule.rule_id,
            ctx.rule.name,
            original_result.original,
            original_result.span,
            len(text),
            describe_span(original_result.original, text),
        )
    original = original_result.span

    if op == PatchOp.REPLACE and original[0] == original[1]:
        return original
    if not text:
        return original

    aligned: tuple[int, int] | None = None
    alignment_error = validate_token_alignment(
        ctx.step.generated_token_alignment, text
    )
    if alignment_error is not None:
        _LOG.warning(
            "invalid_incremental_alignment "
            "invalid_incremental_snap_fallback_tokenizer "
            "rule_id=%s rule_name=%r "
            "original_span=%s doc_len=%s error=%s %s",
            ctx.rule.rule_id,
            ctx.rule.name,
            span,
            len(text),
            alignment_error,
            describe_span(span, text),
        )
    else:
        aligned = _snap_span_with_incremental_alignment(
            span=original,
            alignment=ctx.step.generated_token_alignment,
            op=op,
        )
    if aligned is not None:
        if is_valid_span(aligned, text):
            return aligned
        _LOG.warning(
            "invalid_incremental_snap_fallback_tokenizer rule_id=%s "
            "rule_name=%r original_span=%s incremental_span=%s doc_len=%s %s",
            ctx.rule.rule_id,
            ctx.rule.name,
            span,
            aligned,
            len(text),
            describe_span(aligned, text),
        )

    tokenizer_span = _snap_span_with_tokenizer_offsets(
        text=text, span=original, tokenizer=ctx.tokenizer, op=op
    )
    if is_valid_span(tokenizer_span, text):
        return tokenizer_span
    _LOG.warning(
        "invalid_tokenizer_snap_fallback_original rule_id=%s "
        "tokenizer_span=%s doc_len=%s",
        ctx.rule.rule_id,
        tokenizer_span,
        len(text),
    )

    if is_valid_span(original, text):
        return original
    fallback_result = normalize_span(original, text, mode="clamp")
    if fallback_result.span is not None:
        _LOG.warning(
            (
                "invalid_span_clamped rule_id=%s "
                "original_span=%s corrected_span=%s"
            ),
            ctx.rule.rule_id,
            original,
            fallback_result.span,
        )
        return fallback_result.span
    _LOG.warning(
        "invalid_span_dropped rule_id=%s %s",
        ctx.rule.rule_id,
        describe_span(original, text),
    )
    return None


def _snap_span_with_incremental_alignment(
    *,
    span: tuple[int, int],
    alignment: tuple[TokenCharAlignment, ...],
    op: PatchOp,
) -> tuple[int, int] | None:
    """Try to snap a span using incremental generated-token alignment.

    Purpose:
        Prefer already-available alignment data before falling back to tokenizer
        offset recomputation.

    """
    start, end = span
    if not alignment:
        return None
    if start > end:
        return span

    def _contains(pos: int) -> tuple[int, int] | None:
        """Check whether one token-alignment segment lies fully inside a span.

        Purpose:
            Support incremental-alignment span snapping by testing whether one
            token's character coverage is contained within the candidate target
            span.

        Architectural role:
            Small local helper inside token-boundary reconciliation in proposal
            logic.

        Inputs (architectural provenance):
            Receives one token-alignment segment and the candidate half-open
            character span being snapped.

        Outputs (downstream usage):
            Returns a boolean consumed by incremental-alignment snapping when
            selecting token-aligned span boundaries.

        """
        for item in alignment:
            if item.char_start < pos < item.char_end:
                return (item.char_start, item.char_end)
        return None

    left = _contains(start)
    right = _contains(end)

    if op == PatchOp.INSERT_BEFORE and left is not None:
        snapped = (left[0], left[0])
        return snapped
    if op == PatchOp.INSERT_AFTER and right is not None:
        snapped = (right[1], right[1])
        return snapped

    snapped_start = left[0] if left is not None else start
    snapped_end = right[1] if right is not None else end
    snapped = (snapped_start, snapped_end)
    if snapped_start == start and snapped_end == end:
        return None
    return snapped


def _snap_span_with_tokenizer_offsets(
    *,
    text: str,
    span: tuple[int, int],
    tokenizer: TextCodec | None,
    op: PatchOp,
) -> tuple[int, int]:
    """Snap a span using tokenizer offset mappings.

    Purpose:
        Recompute token-boundary-safe spans when incremental alignment is
        unavailable or insufficient.

    """
    start, end = span
    if tokenizer is None:
        return span
    try:
        offsets = text_alignment.TokenizedTextWithOffsets(
            tokenizer, text
        ).offsets
    except (TypeError, ValueError):
        return span
    if not offsets:
        return span

    try:
        tok_start, tok_end = text_alignment.char_span_to_token_span(
            offsets, start, end
        )
    except ValueError:
        return span

    snapped = span
    if op == PatchOp.INSERT_BEFORE:
        if tok_start < len(offsets):
            token_char_start, _ = offsets[tok_start]
            if token_char_start < start:
                snapped = (token_char_start, token_char_start)
    elif op == PatchOp.INSERT_AFTER:
        if tok_end > 0:
            _, token_char_end = offsets[tok_end - 1]
            if token_char_end > end:
                snapped = (token_char_end, token_char_end)
    elif tok_start < len(offsets):
        snapped_start, _ = offsets[tok_start]
        if tok_end > 0:
            _, snapped_end = offsets[tok_end - 1]
        else:
            snapped_end = snapped_start
        if snapped_end >= snapped_start:
            snapped = (snapped_start, snapped_end)

    return snapped


def _would_change(
    *,
    text: str,
    op: PatchOp,
    span: tuple[int, int],
    payload_norm: str | None,
    casefold_compare: bool,
) -> bool:
    """Return whether applying the proposed edit would change document text.

    Purpose:
        Filter redundant replace/insert/delete proposals before they leave
        proposal generation.

    """
    start, end = span
    fixed = payload_norm or ""
    if op == PatchOp.REPLACE:
        before = text[start:end]
        return (
            before.casefold() != fixed.casefold()
            if casefold_compare
            else before != fixed
        )
    if op == PatchOp.INSERT_AFTER:
        return text[end : end + len(fixed)] != fixed
    if op == PatchOp.INSERT_BEFORE:
        return (
            text[start - len(fixed) : start] != fixed
            if start >= len(fixed)
            else True
        )
    if op == PatchOp.DELETE:
        return start != end
    return False


def _maybe_adjust_after_parenthetical_candidate(
    *,
    text: str,
    span: tuple[int, int],
    candidate_text: str,
    op: PatchOp,
    rule_name: str,
) -> str:
    """Adjust an after-candidate around nearby punctuation and parenthetical.

    Purpose:
        Choose a punctuation/casing prefix for after-style candidate text so the
        inserted text reads naturally in context.

    """
    if not candidate_text or not rule_name.startswith("after:"):
        return candidate_text
    if op not in {PatchOp.INSERT_AFTER, PatchOp.REPLACE}:
        return candidate_text

    insert_point = span[1] if op == PatchOp.INSERT_AFTER else span[0]
    leading_punct = (
        _leading_punctuation_char(text, span) if op == PatchOp.REPLACE else None
    )
    punct = leading_punct or _left_punctuation_char(text, insert_point)

    if punct == ",":
        prefix = ", " if leading_punct else " "
        return prefix + _with_first_alpha_case(candidate_text, upper=False)
    if punct == ".":
        prefix = ". " if leading_punct else " "
        return prefix + _with_first_alpha_case(candidate_text, upper=True)

    return ". " + _with_first_alpha_case(candidate_text, upper=True)


def _left_punctuation_char(text: str, index: int) -> str | None:
    """Return the punctuation character immediately to the left of a span.

    Purpose:
        Inspect the source text just before the candidate edit span so proposal
        normalization can preserve or adjust punctuation-sensitive replacements.

    Architectural role:
        Text-normalization helper inside proposal payload adjustment logic.

    Inputs (architectural provenance):
        Receives the full visible text and the absolute start offset of the
        current candidate span.

    Outputs (downstream usage):
        Returns the single left-adjacent punctuation character, or ``None`` when
        no relevant punctuation is present.

    """
    idx = index - 1
    while idx >= 0 and text[idx].isspace():
        idx -= 1
    if idx < 0:
        return None
    return text[idx] if text[idx] in {",", "."} else None


def _leading_punctuation_char(text: str, span: tuple[int, int]) -> str | None:
    """Return the first punctuation character at the start of a string.

    Purpose:
        Detect whether a candidate payload begins with punctuation so proposal
        normalization can avoid duplicating or misplacing punctuation around
        edits.

    Architectural role:
        Payload-inspection helper inside proposal text-adjustment logic.

    Inputs (architectural provenance):
        Receives one candidate payload string during proposal normalization.

    Outputs (downstream usage):
        Returns the leading punctuation character, or ``None`` when the payload
        does not start with punctuation.

    """
    start, end = span
    idx = start
    while idx < end and text[idx].isspace():
        idx += 1
    if idx >= end:
        return None
    return text[idx] if text[idx] in {",", "."} else None


def _with_first_alpha_case(text: str, *, upper: bool) -> str:
    """Apply the case of the first alphabetic source character to new text.

    Purpose:
        Preserve simple casing conventions when proposal normalization rewrites
        text but wants the replacement to follow the original span's initial
        alphabetic character.

    Architectural role:
        Text-normalization helper inside proposal payload adjustment.

    Inputs (architectural provenance):
        Receives source text that provides the reference casing and candidate
        text whose first alphabetic character may need adjustment.

    Outputs (downstream usage):
        Returns the adjusted candidate text used by downstream proposal
        materialization.

    """
    for idx, char in enumerate(text):
        if char in string.ascii_letters:
            repl = char.upper() if upper else char.lower()
            return text[:idx] + repl + text[idx + 1 :]
    return text


def _maybe_extend_after_span_until_parenthesis_close(
    *,
    ctx: StepContext,
    span: tuple[int, int],
    anchor_span: tuple[int, int] | None,
) -> tuple[tuple[int, int], str | None]:
    """Extend an after-span until the relevant closing parenthesis when.

    Purpose:
        Support rules that wait for parenthetical closure before emitting an
        after- style edit.

    """

    def _defer() -> tuple[tuple[int, int], str]:
        return span, "waiting_for_closing_parenthesis"

    def _changed(
        new_span: tuple[int, int], reason: str
    ) -> tuple[tuple[int, int], None]:
        del reason
        return new_span, None

    if not ctx.rule.wait_for_closing_parenthesis:
        return span, None

    if not ctx.rule.name.startswith("after:"):
        return span, None

    if not any(
        candidate.op in {PatchOp.INSERT_AFTER, PatchOp.REPLACE}
        for candidate in ctx.rule.candidates
    ):
        return span, None

    text = ctx.doc.text
    source_start, source_end = anchor_span or span
    anchor_text = text[source_start:source_end]
    depth = max(anchor_text.count("(") - anchor_text.count(")"), 0)

    span_text = text[span[0] : span[1]]
    idx = span[1]
    while idx < ctx.edit_view.abs_end and text[idx].isspace():
        idx += 1

    span_depth = max(span_text.count("(") - span_text.count(")"), 0)

    if depth == 0:
        if idx >= ctx.edit_view.abs_end and span_depth > 0:
            return _defer()
        if idx >= ctx.edit_view.abs_end or text[idx] != "(":
            rel_idx = 0
            while rel_idx < len(span_text) and span_text[rel_idx].isspace():
                rel_idx += 1
            if rel_idx < len(span_text) and span_text[rel_idx] == ")":
                end = _consume_trailing_parenthetical_suffix(
                    text=text,
                    index=span[0] + rel_idx + 1,
                    max_end=ctx.edit_view.abs_end,
                )
                return _changed((end, end), "leading_closing_parenthesis")

            open_offset = span_text.find("(")
            if open_offset < 0:
                normalized_insert_at = (
                    _normalize_after_parenthetical_insertion_point(
                        text=text,
                        span=span,
                        max_end=ctx.edit_view.abs_end,
                    )
                )
                if normalized_insert_at is not None:
                    return _changed(
                        (normalized_insert_at, normalized_insert_at),
                        "normalized_after_parenthetical_insertion_point",
                    )
                return span, None

            local_depth = 0
            close_offset: int | None = None
            for rel_pos in range(open_offset, len(span_text)):
                char = span_text[rel_pos]
                if char == "(":
                    local_depth += 1
                elif char == ")":
                    if local_depth > 0:
                        local_depth -= 1
                    if local_depth == 0:
                        close_offset = rel_pos
                        break

            if close_offset is None:
                return _defer()

            end = _consume_trailing_parenthetical_suffix(
                text=text,
                index=span[0] + close_offset + 1,
                max_end=ctx.edit_view.abs_end,
            )
            return _changed((end, end), "closed_parenthetical_in_span")
    search_start = span[0]
    while search_start < ctx.edit_view.abs_end and text[search_start].isspace():
        search_start += 1
    if search_start >= ctx.edit_view.abs_end:
        return _defer()

    close_idx: int | None = None
    for pos in range(search_start, ctx.edit_view.abs_end):
        char = text[pos]
        if char == "(":
            depth += 1
        elif char == ")":
            if depth > 0:
                depth -= 1
            if depth == 0:
                close_idx = pos
                break

    if close_idx is None:
        return _defer()

    end = _consume_trailing_parenthetical_suffix(
        text=text,
        index=close_idx + 1,
        max_end=ctx.edit_view.abs_end,
    )
    return _changed((end, end), "closed_parenthetical_after_span")


def _consume_trailing_parenthetical_suffix(
    *, text: str, index: int, max_end: int
) -> int:
    """Consume the trailing suffix after a closing parenthesis when it matches.

    Purpose:
        Extend parenthetical handling past the closing delimiter when the
        recognized suffix pattern should remain attached.

    """
    match = _PARENTHETICAL_SUFFIX_PATTERN.match(text, index, max_end)
    return match.end() if match is not None else index


def _normalize_after_parenthetical_insertion_point(
    *,
    text: str,
    span: tuple[int, int],
    max_end: int,
) -> int | None:
    """Normalize an after-rule target span to a safe insertion point.

    Purpose:
        Preserve completed ``)`` + punctuation suffixes when an after-rule
        target lands on a trailing suffix/newline region that should behave as
        an insertion point rather than a replacement span.

    """
    start, end = span
    if start < 0 or end > max_end or start > end:
        return None

    idx = start
    while idx < end and text[idx].isspace():
        idx += 1
    if idx < end and text[idx] == ")":
        suffix_end = _consume_trailing_parenthetical_suffix(
            text=text,
            index=idx + 1,
            max_end=max_end,
        )
        if text[suffix_end:end].isspace():
            return suffix_end
        return None

    if not text[start:end].isspace():
        return None
    left = start - 1
    while left >= 0 and text[left].isspace():
        left -= 1
    if left < 0:
        return None
    if text[left] == ")":
        return _consume_trailing_parenthetical_suffix(
            text=text,
            index=left + 1,
            max_end=max_end,
        )
    if text[left] in {".", ",", "!", "?", "*"}:
        prev = left - 1
        while prev >= 0 and text[prev].isspace():
            prev -= 1
        if prev >= 0 and text[prev] == ")":
            return left + 1
    return None
