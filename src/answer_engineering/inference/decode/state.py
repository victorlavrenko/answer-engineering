"""Mutable state carriers used by the streaming decode loop.

Purpose:
    Hold visible text, token alignment, cached forward state, and rebuild
    bookkeeping across greedy decoding.

Architectural role:
    Decode-internal state module.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from answer_engineering.engine.runtime.runtime_types import (
    Decision,
    TokenAlignedTextView,
    TokenCharAlignment,
)
from answer_engineering.inference.model_types import (
    PastKeyValues,
    ScoringRuntime,
)

type NestedTokenIds = int | Sequence["NestedTokenIds"] | None
type OverrideDebugValue = str | int | float | bool | None
type OverrideDebugPayload = dict[str, OverrideDebugValue]


@dataclass
class StreamingDecodeState(TokenAlignedTextView):
    """Mutable state carried across one streaming decode run.

    Purpose:
        Hold visible text, generated token ids, token alignment, cached runtime
        state, and core-edit rebuild flags.

    Architectural role:
        Central mutable state object for the greedy decode loop.

    """

    past_key_values: PastKeyValues
    next_input: torch.Tensor | None
    assistant_visible_text: str
    generated_token_ids: list[int]
    eos_ids: set[int]
    device: torch.device
    prefill_next_logits: torch.Tensor | None = None
    prefill_done: bool = False
    needs_rebuild: bool = False
    last_override_dbg: OverrideDebugPayload | None = None
    last_override_was_active: bool = False
    token_char_ends: list[int] = field(default_factory=lambda: [])
    generated_token_alignment: list[TokenCharAlignment] = field(
        default_factory=lambda: []
    )
    last_core_decision: Decision | None = None
    last_core_snapshot_text: str = ""
    last_core_rebuild_reason: str = ""
    applied_decisions: int = 0
    decision_limit_reached: bool = False

    def current_visible_text(self) -> str:
        """Return the canonical assistant-visible text for this decode run.

        Purpose:
            Expose the decode-owned visible text snapshot for reactive console
            printing.

        """
        return self.assistant_visible_text


@dataclass(frozen=True, slots=True, init=False)
class RebuiltPrefixState:
    """Forward-pass snapshot for a rebuilt full prefix.

    Purpose:
        Materialize past_key_values and next logits for the entire rebuilt
        prefix after a core edit changes the visible assistant text.

    Architectural role:
        Decode-rebuild helper object.

    """

    past_key_values: PastKeyValues
    next_logits: torch.Tensor

    def __init__(
        self, runtime: ScoringRuntime, full_prefix_ids: torch.Tensor
    ) -> None:
        """Run one cached forward pass for a rebuilt full-prefix tensor.

        Purpose:
            Reconstruct decode state after the visible document prefix has
            changed and the model cache must be refreshed from the canonical
            token prefix.

        Architectural role:
            Constructor boundary between document edits and model-backed greedy
            decode state. It owns the initial forward pass needed to seed logits
            and cache for subsequent token generation.

        Inputs (architectural provenance):
            Receives the scoring/generation runtime, prepared prefix tensor,
            attention-mask tensor, and existing generated-token bookkeeping.

        Outputs (downstream usage):
            Stores refreshed model outputs, cache state, attention mask,
            generated ids, and next-token logits for the decode loop.

        Invariants/constraints:
            The prefix tensor and attention mask must describe the same rebuilt
            prefix. The resulting cache belongs to that prefix and must not be
            reused after a later committed edit invalidates it.

        """
        with torch.inference_mode():
            out = runtime.forward(input_ids=full_prefix_ids, use_cache=True)
        object.__setattr__(self, "past_key_values", out.past_key_values)
        object.__setattr__(self, "next_logits", out.logits[:, -1, :])

    def __iter__(self):
        """Yield rebuilt-prefix forward state in positional order.

        Purpose:
            Support tuple-style unpacking of `RebuiltPrefixState` into cached
            past-key-values and next-token logits after a full-prefix rebuild.

        Architectural role:
            Convenience iterator on the decode-rebuild helper object.

        Inputs (architectural provenance):
            Reads the forward-pass artifacts materialized during
            `RebuiltPrefixState` construction.

        Outputs (downstream usage):
            Yielded values are consumed by decode code that unpacks
            rebuilt-prefix runtime state.

        Invariants/constraints:
            Iteration order must remain stable as `(past_key_values,
            next_logits)`.

        """
        yield self.past_key_values
        yield self.next_logits
