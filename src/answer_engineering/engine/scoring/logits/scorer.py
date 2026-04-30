"""Model-backed scoring for patch proposals.

Purpose:
    Build and batch scoring tasks that estimate proposal quality from model
    continuation log-probabilities.

Architectural role:
    LLM-backed implementation of the engine scoring boundary.

Inputs:
    Step-local document state, patch proposals, tokenizer/runtime access, and
    patch-scoring policy.

Outputs:
    Score batches and helper groupings consumed by LogitsScorer.

Ownership:
    Owned by the engine scoring boundary.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from answer_engineering.config.patch_score_policy import PatchScorePolicy
from answer_engineering.engine.patching import patcher
from answer_engineering.engine.patching.proposals import (
    PatchProposal,
)
from answer_engineering.engine.pipeline.context import (
    StepContext,
)
from answer_engineering.engine.runtime import text_alignment
from answer_engineering.engine.scoring.base import (
    ConfigurableScorer,
    ScoredProposal,
    Scorer,
    ScoreResult,
    ScoringDiagnostics,
)
from answer_engineering.inference.model_types import (
    TokenGenerationRuntime,
)
from answer_engineering.inference.runtime_adapter.logprob_scoring import (
    RuntimeLogprobScorer,
)


class _TextCodec(Protocol):
    """Protocol required by logits scorer for tokenizer-like accessors."""

    @property
    def bos_token_id(self) -> int | None:
        """Return the BOS token id used to normalize scoring prefixes."""
        raise NotImplementedError

    def encode(
        self, text: str, *, add_special_tokens: bool = False
    ) -> list[int]:
        """Encode text into token ids for scorer task construction without."""
        raise NotImplementedError


@dataclass(slots=True)
class LogitsScorer(Scorer, ConfigurableScorer):
    """Score patch proposals via model logits or deterministic fallback.

    Determinism:
        With no model, uses deterministic length-based fallback. With model
        configured, determinism depends on model/provider behavior though
        selection remains stable for equal score vectors.

    Notes:
        Uses batched prefix-grouped logprob calls to reduce redundant inference
        work.

    """

    runtime: TokenGenerationRuntime | None = None
    require_model_scoring: bool = False
    policy: PatchScorePolicy = field(default_factory=PatchScorePolicy)

    def configure(
        self,
        *,
        runtime: TokenGenerationRuntime | None,
        require_model_scoring: bool,
    ) -> None:
        """Store runtime and model-scoring policy for later scoring.

        Purpose:
            Attach the token-generation runtime and strictness flag needed
            before proposals can be scored with optional model evidence.

        Architectural role:
            Lifecycle method for the logits scorer. It separates scorer
            construction from runtime injection so orchestration can build
            components before the model-backed runtime is available.

        Inputs (architectural provenance):
            Receives the runtime implementing token scoring and the flag that
            decides whether missing model scoring should fail or fall back.

        Outputs (downstream usage):
            Updates scorer state read by `score` when proposals are evaluated.

        Invariants/constraints:
            The method does not perform scoring or validate proposal content. It
            only records dependencies and scoring strictness for subsequent
            calls.

        """
        self.runtime = runtime
        self.require_model_scoring = require_model_scoring

    def score(
        self, ctx: StepContext, proposals: list[PatchProposal]
    ) -> ScoreResult:
        """Score proposals by applying each patch and evaluating model evidence.

        Purpose:
            Convert candidate patches into comparable scores using deterministic
            patch features and optional model log-probability signals.

        Architectural role:
            Concrete scorer implementation between proposal construction and
            selection.

        Inputs (architectural provenance):
            Receives the current step context and proposal iterable from
            orchestration.

        Outputs (downstream usage):
            Returns scored proposals consumed by selection and telemetry
            emission.

        Invariants/constraints:
            The scorer evaluates hypothetical patched text; it must not mutate
            the live document. Model-backed scoring is used only when runtime
            configuration and scorer policy allow it.

        """
        if not proposals:
            return ScoreResult(
                scored=[],
                diagnostics=ScoringDiagnostics(model_scored=False, num_calls=0),
            )

        candidate_texts = [
            patcher.apply_patch(ctx.doc, proposal).text
            for proposal in proposals
        ]
        if self.runtime is None:
            if self.require_model_scoring:
                raise ValueError("runtime required for core logits scoring")
            scores = [-(float(len(text))) for text in candidate_texts]
            model_scored = False
            num_calls = 0
        else:
            self.runtime.ensure_eval_mode()
            score_batch = _score_patch_candidates_batch(
                runtime=self.runtime,
                original_text=ctx.doc.text,
                proposals=proposals,
                policy=self.policy,
            )
            scores = score_batch.scores
            num_calls = score_batch.num_calls
            model_scored = True

        out: list[ScoredProposal] = []
        for idx, proposal in enumerate(proposals):
            score = scores[idx]
            out.append(
                ScoredProposal(
                    proposal=proposal.with_updates(
                        score=score,
                        cached_score_logprob=score,
                        cached_final_text=candidate_texts[idx],
                    ),
                    score=score,
                )
            )
        return ScoreResult(
            scored=out,
            diagnostics=ScoringDiagnostics(
                model_scored=model_scored,
                num_calls=(num_calls if model_scored else 0),
            ),
        )


def _tokenize_replacement(
    tokenizer: _TextCodec, proposal: PatchProposal
) -> list[int]:
    """Encode the proposal replacement payload into token ids for scorer task.

    Delete proposals contribute an empty replacement.

    """
    if proposal.payload_norm is not None:
        replacement = proposal.payload_norm
    else:
        replacement = proposal.payload or ""
    if proposal.op.value == "delete":
        replacement = ""
    return tokenizer.encode(replacement, add_special_tokens=False)


@dataclass(frozen=True, slots=True)
class _ScoreTaskGroups:
    """Partition of proposal scoring work into cached, model-scored, and.

    Purpose:
        Organize one scoring batch into the subsets that can reuse cached
        values, require model scoring, or need deterministic fallback handling.

    Architectural role:
        Internal planning record inside the logits-based scoring subsystem.

    Inputs (architectural provenance):
        Built by the logits scorer while preparing one batch of proposals for
        evaluation.

    Outputs (downstream usage):
        Consumed by the scorer's batch execution logic to choose the correct
        scoring path for each proposal.

    Invariants/constraints:
        Every proposal in the original batch must appear in exactly one task
        group.

    """

    left_context: list[tuple[int, list[int], list[int]]]
    replacement: list[tuple[int, list[int], list[int]]]
    right_context: list[tuple[int, list[int], list[int]]]
    continuation: list[tuple[int, list[int], list[int]]]


@dataclass(frozen=True, slots=True)
class ScoreBatchResult:
    """Result container for one logits-scoring batch.

    Purpose:
        Carry the scored proposals and batch-level diagnostics produced by one
        model-backed scoring execution.

    Architectural role:
        Internal return object inside the logits scorer before values are
        adapted to the public `ScoreResult`.

    Inputs (architectural provenance):
        Constructed after cached scores, model calls, and fallback paths have
        been merged for one batch.

    Outputs (downstream usage):
        Consumed by the logits scorer when producing the final canonical scoring
        result.

    Invariants/constraints:
        The scored proposals and diagnostics must describe the same batch
        execution.

    """

    scores: list[float]
    num_calls: int


def _score_patch_candidates_batch(
    *,
    runtime: TokenGenerationRuntime,
    original_text: str,
    proposals: list[PatchProposal],
    policy: PatchScorePolicy,
) -> ScoreBatchResult:
    """Batch score patch candidates by grouping shared prefixes.

    Purpose:
        Evaluate many candidate patches with fewer model calls by batching
        scoring tasks that share prefix context.

    Architectural role:
        Model-scoring workhorse behind `LogitsScorer.score`. It bridges
        text-level proposals, tokenizer offsets, score-task construction, and
        weighted score aggregation.

    Inputs (architectural provenance):
        Receives the scoring runtime, original document text, candidate
        proposals, and patch-score policy.

    Outputs (downstream usage):
        Returns per-proposal score totals plus the number of model scoring calls
        used to compute them.

    Invariants/constraints:
        Proposal order is preserved in the returned score list. Score components
        must be accumulated into the matching proposal index after batching.

    """
    tok = runtime.text_codec()
    tokenized = text_alignment.TokenizedTextWithOffsets(tok, original_text)
    orig_ids = tokenized.token_ids
    offsets = tokenized.offsets
    task_groups = _build_score_task_groups(
        proposals=proposals,
        policy=policy,
        tokenizer=tok,
        original_ids=orig_ids,
        offsets=offsets,
    )
    totals = [0.0 for _ in proposals]
    num_calls = 0
    num_calls += _accumulate_batched_scores(
        runtime=runtime,
        tasks=task_groups.left_context,
        totals=totals,
        weight=policy.w_left_ctx,
    )
    num_calls += _accumulate_batched_scores(
        runtime=runtime,
        tasks=task_groups.replacement,
        totals=totals,
    )
    num_calls += _accumulate_batched_scores(
        runtime=runtime,
        tasks=task_groups.right_context,
        totals=totals,
        weight=policy.w_right_ctx,
    )
    num_calls += _accumulate_batched_scores(
        runtime=runtime,
        tasks=task_groups.continuation,
        totals=totals,
        weight=policy.continuation_weight,
    )
    return ScoreBatchResult(scores=totals, num_calls=num_calls)


def _build_score_task_groups(
    *,
    proposals: list[PatchProposal],
    policy: PatchScorePolicy,
    tokenizer: _TextCodec,
    original_ids: list[int],
    offsets: list[tuple[int, int]],
) -> _ScoreTaskGroups:
    """Build the grouped scoring tasks for a proposal batch.

    Each task group captures one scoring component so later batching can reuse
    shared prefixes efficiently.

    """
    left_context: list[tuple[int, list[int], list[int]]] = []
    replacement: list[tuple[int, list[int], list[int]]] = []
    right_context: list[tuple[int, list[int], list[int]]] = []
    continuation: list[tuple[int, list[int], list[int]]] = []

    for proposal_index, proposal in enumerate(proposals):
        if proposal.span_abs is None:
            raise ValueError("span_abs required for scoring proposals")
        span_start, span_end = proposal.span_abs
        token_start, token_end = text_alignment.char_span_to_token_span(
            offsets, span_start, span_end
        )
        replacement_ids = _tokenize_replacement(tokenizer, proposal)
        patched_ids = [
            *original_ids[:token_start],
            *replacement_ids,
            *original_ids[token_end:],
        ]
        _append_left_context_tasks(
            left_context,
            proposal_index=proposal_index,
            original_ids=original_ids,
            token_start=token_start,
            policy=policy,
        )
        _append_replacement_tasks(
            replacement,
            proposal_index=proposal_index,
            patched_ids=patched_ids,
            token_start=token_start,
            replacement_ids=replacement_ids,
            policy=policy,
        )
        _append_right_context_tasks(
            right_context,
            proposal_index=proposal_index,
            original_ids=original_ids,
            patched_ids=patched_ids,
            token_start=token_start,
            token_end=token_end,
            replacement_ids=replacement_ids,
            policy=policy,
        )
        _append_continuation_tasks(
            continuation,
            proposal_index=proposal_index,
            patched_ids=patched_ids,
            token_start=token_start,
            replacement_ids=replacement_ids,
            policy=policy,
        )
    return _ScoreTaskGroups(
        left_context=left_context,
        replacement=replacement,
        right_context=right_context,
        continuation=continuation,
    )


def _append_left_context_tasks(
    tasks: list[tuple[int, list[int], list[int]]],
    *,
    proposal_index: int,
    original_ids: list[int],
    token_start: int,
    policy: PatchScorePolicy,
) -> None:
    """Append left-context scoring tasks for one proposal when enabled."""
    left_start = max(0, token_start - policy.n_left_ctx)
    if not policy.score_left or left_start >= token_start:
        return
    left_continuation = original_ids[left_start:token_start]
    if not left_continuation:
        return
    tasks.append((proposal_index, original_ids[:left_start], left_continuation))


def _append_replacement_tasks(
    tasks: list[tuple[int, list[int], list[int]]],
    *,
    proposal_index: int,
    patched_ids: list[int],
    token_start: int,
    replacement_ids: list[int],
    policy: PatchScorePolicy,
) -> None:
    """Append replacement-token scoring tasks for one proposal when."""
    if not policy.score_replacement or not replacement_ids:
        return
    replacement_end = token_start + len(replacement_ids)
    tasks.append(
        (
            proposal_index,
            patched_ids[:token_start],
            patched_ids[token_start:replacement_end],
        )
    )


def _append_right_context_tasks(
    tasks: list[tuple[int, list[int], list[int]]],
    *,
    proposal_index: int,
    original_ids: list[int],
    patched_ids: list[int],
    token_start: int,
    token_end: int,
    replacement_ids: list[int],
    policy: PatchScorePolicy,
) -> None:
    """Append right-context scoring tasks for one patched proposal when."""
    right_start_orig = token_end
    right_end_orig = min(len(original_ids), token_end + policy.n_right_ctx)
    if not policy.score_right or right_start_orig >= right_end_orig:
        return
    right_start = token_start + len(replacement_ids)
    right_end = right_start + (right_end_orig - right_start_orig)
    right_continuation = patched_ids[right_start:right_end]
    if not right_continuation:
        return
    tasks.append(
        (proposal_index, patched_ids[:right_start], right_continuation)
    )


def _append_continuation_tasks(
    tasks: list[tuple[int, list[int], list[int]]],
    *,
    proposal_index: int,
    patched_ids: list[int],
    token_start: int,
    replacement_ids: list[int],
    policy: PatchScorePolicy,
) -> None:
    """Append continuation scoring tasks for one patched proposal when."""
    if policy.continuation_tokens <= 0:
        return
    continuation_start = token_start + len(replacement_ids)
    continuation_end = min(
        len(patched_ids), continuation_start + policy.continuation_tokens
    )
    continuation_ids = patched_ids[continuation_start:continuation_end]
    if not continuation_ids:
        return
    tasks.append(
        (proposal_index, patched_ids[:continuation_start], continuation_ids)
    )


def _accumulate_batched_scores(
    *,
    runtime: TokenGenerationRuntime,
    tasks: list[tuple[int, list[int], list[int]]],
    totals: list[float],
    weight: float = 1.0,
) -> int:
    """Accumulate weighted batched logprob scores into the proposal totals.

    Tasks are grouped by shared prefix so one runtime call can score multiple
    continuations.

    """
    if not tasks:
        return 0

    grouped_tasks: dict[tuple[int, ...], list[tuple[int, list[int]]]] = {}
    scorer = RuntimeLogprobScorer(runtime)
    for proposal_index, prefix_ids, continuation_ids in tasks:
        grouped_tasks.setdefault(tuple(prefix_ids), []).append(
            (proposal_index, continuation_ids)
        )

    for prefix_ids, grouped_continuations in grouped_tasks.items():
        results = scorer.score_continuations_batch(
            prefix_ids=list(prefix_ids),
            continuation_ids_list=[
                continuation for _, continuation in grouped_continuations
            ],
            return_token_details=False,
        )
        for (proposal_index, _), result in zip(
            grouped_continuations, results, strict=True
        ):
            totals[proposal_index] += weight * result.logprob_sum
    return len(grouped_tasks)
