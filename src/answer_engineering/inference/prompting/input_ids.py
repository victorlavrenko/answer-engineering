"""Utilities for building input ids for chat scoring.

This module is responsible for constructing the input ids tensor for a chat
scoring request, given the chat messages and an optional partial assistant
response. It uses the chat text codec from the runtime to encode the messages
according to the chat template, and concatenates the partial assistant text if
provided.

"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from answer_engineering.inference.model_types import (
    ChatMessage,
    ChatScoringRuntime,
    ChatTextCodec,
)


def build_input_ids(
    runtime: ChatScoringRuntime,
    messages: Sequence[ChatMessage],
    partial_assistant_text: str = "",
) -> torch.Tensor:
    """Build prompt ids for a chat transcript and optional assistant prefix.

    Purpose:
        Convert structured chat messages into the token-id prefix used by
        decoding and optional continuation from a partial assistant answer.

    Architectural role:
        Prompt-assembly boundary between public request/policy objects and the
        lower-level token generation runtime.

    Inputs (architectural provenance):
        Receives a tokenizer-like codec, chat messages assembled by the caller,
        and an optional generated prefix supplied by request continuation logic.

    Outputs (downstream usage):
        Returns `PrefixExpansion` with prompt ids, generated-prefix ids, and the
        combined token sequence consumed by decode setup.

    Invariants/constraints:
        Chat-template rendering is delegated to the codec. Prefix normalization
        is centralized in `PrefixExpansion` so downstream stages see immutable
        token tuples.

    """
    tok: ChatTextCodec = runtime.text_codec()
    templated_messages = list(messages)
    prompt_text = str(
        tok.apply_chat_template(
            templated_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    prompt_tensor: torch.Tensor
    try:
        prompt_ids = tok.apply_chat_template(
            templated_messages, tokenize=True, add_generation_prompt=True
        )
        if isinstance(prompt_ids, torch.Tensor):
            prompt_tensor = prompt_ids.to(
                device=runtime.execution_device(), dtype=torch.long
            )
        else:
            prompt_tensor = torch.tensor(
                prompt_ids, dtype=torch.long, device=runtime.execution_device()
            )
            if prompt_tensor.dim() == 1:
                prompt_tensor = prompt_tensor.unsqueeze(0)
    except TypeError:
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        prompt_tensor = torch.tensor(
            [prompt_ids], dtype=torch.long, device=runtime.execution_device()
        )

    if partial_assistant_text:
        continuation_ids = tok.encode(
            partial_assistant_text, add_special_tokens=False
        )
        continuation_tensor = torch.tensor(
            [continuation_ids],
            dtype=torch.long,
            device=runtime.execution_device(),
        )
        prompt_tensor = torch.cat([prompt_tensor, continuation_tensor], dim=1)

    return prompt_tensor


__all__ = ["build_input_ids"]
