"""Core-step orchestration and rebuild support for streaming decode.

Purpose:
    Execute core Answer Engineering steps against visible assistant text and
    rebuild decode state when edits change generated output.

Architectural role:
    Bridge between decode execution state and the core plan runner.

Architectural direction:
    This seam should evolve toward clearer separation of decode state
    management, rebuild policy, and core-execution integration.

Why this matters:
    The current shape is functional but concentrates transitional integration
    logic required by local-edit intervention during streaming decode.

What better would look like:
    Decode-state updates, rebuild behavior, and core-step execution become
    easier to extend independently without broad cross-layer edits.

How improvement can be recognized:
    - Narrower surface between decode state and core execution
    - Reduced coupling between rebuild mechanics and orchestration calls
    - Clearer ownership of state mutation vs. execution decisions

Open constraint:
    The exact seam should remain sensitive to future runtime and intervention
    experiments.

"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from answer_engineering.engine.orchestration.orchestrator import (
    PlanRunner,
)
from answer_engineering.engine.pipeline.context import (
    StepSnapshot,
)
from answer_engineering.engine.runtime.runtime_types import (
    Decision,
    TokenCharAlignment,
)
from answer_engineering.inference.decode.state import (
    RebuiltPrefixState,
    StreamingDecodeState,
)
from answer_engineering.inference.model_types import (
    ChatGenerationRuntime,
    TextCodec,
)
from answer_engineering.inference.prompting.prompt_prefix import (
    PrefixExpansion,
)
from answer_engineering.rules.compile.plan import PlanIR


def _retokenize_assistant(tok: TextCodec, text: str) -> list[int]:
    """Encode current assistant text into plain generated token ids.

    Purpose:
        Rebuild the generated-token sequence after the assistant text has
        changed because a runtime edit was applied.

    Architectural role:
        Decode-session rebuild helper. It separates tokenizer re-encoding from
        the higher-level session object that also reconstructs character
        alignment.

    Inputs (architectural provenance):
        Receives the runtime text codec and the current assistant-visible text.

    Outputs (downstream usage):
        Returns token ids without special tokens for cache rebuild, prefix
        expansion, and subsequent decode continuation.

    Invariants/constraints:
        Empty assistant text maps to an empty list. The helper must not add
        chat, BOS, EOS, or other prompt-level tokens.

    """
    if not text:
        return list()
    return list(tok.encode(text, add_special_tokens=False))


@dataclass(frozen=True, slots=True, init=False)
class RetokenizedAssistant:
    """Retokenized assistant text plus reconstructed token-to-character.

    Purpose:
        Hold the replacement token stream and per-token character spans produced
        after the visible assistant text changes and the incremental decode
        state must be rebuilt.

    Architectural role:
        Small decode-side value object used only during rebuild.

    """

    token_ids: list[int]
    alignment: list[TokenCharAlignment]

    def __init__(self, tok: TextCodec, text: str) -> None:
        """Retokenize assistant text and rebuild token-to-character alignment.

        Purpose:
            Reconstruct decode-session state after the visible assistant text
            has changed and the original incremental token stream can no longer
            be trusted.

        Architectural role:
            Constructor for a decode-side value object used during prefix/cache
            rebuild after runtime edits.

        Inputs (architectural provenance):
            Receives the tokenizer/text codec and the current assistant text
            from session-orchestration rebuild logic.

        Outputs (downstream usage):
            Stores replacement token ids and alignment spans consumed by
            subsequent generation and probing steps.

        Invariants/constraints:
            Alignment must be rebuilt from the same tokenization as `token_ids`
            so character spans and token positions remain synchronized.

        """
        token_ids = _retokenize_assistant(tok, text)
        alignment: list[TokenCharAlignment] = []
        cursor = 0
        for token_id in token_ids:
            piece = tok.decode([token_id], skip_special_tokens=True)
            start = cursor
            cursor += len(piece)
            alignment.append(
                TokenCharAlignment(
                    token_index=len(alignment),
                    char_start=start,
                    char_end=cursor,
                    piece_text=piece,
                )
            )
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "alignment", alignment)

    def __iter__(self):
        """Yield ``token_ids`` and ``alignment`` in the unpacking order."""
        yield self.token_ids
        yield self.alignment


@dataclass(slots=True)
class ExecutionSession:
    """Run core Answer Engineering steps from a decode snapshot and rebuild.

    Purpose:
        Wrap the compiled plan and ``PlanRunner`` used by the streaming decode
        loop so one decode step can execute core rules, inspect the resulting
        decision, and rebuild prefix state when the assistant text changes.

    Architectural role:
        Inference-side coordinator between decode state management and the core
        orchestration engine.

    """

    plan: PlanIR
    runner: PlanRunner = field(default_factory=PlanRunner)

    def rule_name(self, rule_id: str) -> str:
        """Return the display name for one compiled rule id, or ``""`` if."""
        for rule in self.plan.rules:
            if rule.rule_id == rule_id:
                return rule.name or ""
        return ""

    def execute_step(self, request: StepSnapshot) -> Decision:
        """Execute step.

        Purpose:
            Implement the operation performed by this decode orchestration
            component.

        Architectural role:
            Inference-side bridge between mutable decode state and engine
            orchestration.

        Inputs:
            Consumes model-runtime state, token ids, prefixes, or probe requests
            from inference callers.

        Outputs:
            Consumed by scoring, probing, or decode-session callers in the
            inference layer.

        """
        result = self.runner.run(self.plan, request)
        return Decision(result)

    def apply_step(
        self,
        *,
        state: StreamingDecodeState,
        tick_index: int,
        prompt_ids: torch.Tensor | None = None,
        prompt_text: str = "",
    ) -> bool:
        """Apply one core decision to the mutable streaming decode state.

        Purpose:
            Update visible text and mark the state for runtime-prefix rebuild
            when a change occurred.

        Architectural role:
            Inference-side bridge between mutable decode state and engine
            orchestration.

        Inputs:
            Consumes model-runtime state, token ids, prefixes, or probe requests
            from inference callers.

        Outputs:
            Consumed by scoring, probing, or decode-session callers in the
            inference layer.

        """
        old_text = state.assistant_visible_text
        request = StepSnapshot(
            state=state,
            token_index=tick_index,
            prompt_ids=prompt_ids,
            prompt_text=prompt_text,
        )
        decision = self.execute_step(request)
        state.last_core_snapshot_text = old_text
        state.last_core_decision = decision
        if not decision.changed:
            return False
        state.assistant_visible_text = decision.final_text
        applied_count = len(decision.applied_patches)
        state.needs_rebuild = True
        new_text = state.assistant_visible_text
        state.last_core_rebuild_reason = (
            f"cause=core_edit old_len={len(old_text)} new_len={len(new_text)} "
            f"applied_patches={applied_count}"
        )
        state.prefill_next_logits = None
        state.next_input = None
        return True

    def rebuild_decode_state(
        self,
        *,
        state: StreamingDecodeState,
        tokenizer: TextCodec,
        runtime: ChatGenerationRuntime,
        prepared_input_ids: torch.Tensor,
    ) -> None:
        """Rebuild token ids, alignment, and cached forward state after a core.

        Purpose:
            Retokenize the visible assistant text, reconstruct the full prefix
            tensor, and prefill past_key_values and next logits for resumed
            decoding.

        Architectural role:
            Inference-side bridge between mutable decode state and engine
            orchestration.

        Inputs:
            Consumes model-runtime state, token ids, prefixes, or probe requests
            from inference callers.

        Outputs:
            Consumed by scoring, probing, or decode-session callers in the
            inference layer.

        """
        retokenized = RetokenizedAssistant(
            tokenizer, state.assistant_visible_text
        )
        state.generated_token_ids = retokenized.token_ids
        state.generated_token_alignment = retokenized.alignment
        full_prefix_ids = PrefixExpansion(
            prompt_ids=prepared_input_ids,
            generated_ids=state.generated_token_ids,
            device=runtime.execution_device(),
        ).full_prefix_ids()
        rebuilt = RebuiltPrefixState(
            runtime=runtime,
            full_prefix_ids=full_prefix_ids,
        )
        state.past_key_values = rebuilt.past_key_values
        state.prefill_next_logits = rebuilt.next_logits
        state.next_input = None
        state.needs_rebuild = False
