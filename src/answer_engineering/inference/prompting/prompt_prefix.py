"""Token-prefix helpers shared by decode, probing, and scoring.

Purpose:
    Normalize empty prefixes, rebuild full prompt-plus-generated prefixes,
    prepare prefix tensors for runtime calls, and compute stable prefix
    fingerprints.

Architectural role:
    Shared inference utility module for token-prefix handling.

"""

from __future__ import annotations

import hashlib
from array import array
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from answer_engineering.inference.model_types import TextCodec


def normalize_prefix_ids(
    *,
    tokenizer: TextCodec,
    prefix_ids: list[int],
    fallback_to_zero: bool = False,
) -> list[int]:
    """Normalize an empty prefix to a safe BOS/EOS fallback sequence.

    Purpose:
        Produce a non-empty token prefix for scoring and generation paths that
        need at least one token of context.

    Architectural role:
        Token-prefix normalization boundary shared by prompt assembly and
        scoring.

    Inputs (architectural provenance):
        Receives caller-provided token ids plus optional BOS and EOS ids exposed
        by the tokenizer.

    Outputs (downstream usage):
        Returns an immutable tuple of token ids suitable for tensor preparation
        or fingerprinting.

    Invariants/constraints:
        Existing prefix ids are preserved exactly. Empty prefixes fall back to
        BOS when available, then EOS when available, and otherwise remain empty
        so the caller can decide whether that runtime path is valid.

    """
    if prefix_ids:
        return list(prefix_ids)

    bos_id = tokenizer.bos_token_id
    if bos_id is not None:
        return [int(bos_id)]
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, list):
        return [int(eos_id[0])] if eos_id else ([0] if fallback_to_zero else [])
    if eos_id is not None:
        return [int(eos_id)]
    if fallback_to_zero:
        return [0]
    return list()


@dataclass(frozen=True, slots=True, init=False)
class PrefixExpansion:
    """Prompt-plus-generated prefix expansion.

    Purpose:
        Combine prepared prompt ids with generated token ids into the full
        prefix tensor used by rebuild, probing, and other runtime calls.

    Architectural role:
        Shared token-prefix value object inside inference.

    """

    prompt_ids: torch.Tensor
    generated_ids: tuple[int, ...]
    device: torch.device

    def __init__(
        self,
        *,
        prompt_ids: torch.Tensor,
        generated_ids: list[int] | tuple[int, ...],
        device: torch.device,
    ) -> None:
        """Store prompt ids and normalize generated ids to an immutable form.

        Purpose:
            Construct the canonical prefix expansion used when prompt tokens and
            generated assistant tokens must be reasoned about together.

        Architectural role:
            Prompting value object at the boundary between rendered prompt text,
            tokenization, and runtime prefix/cache reuse.

        Inputs (architectural provenance):
            Receives prompt token ids and optional generated token ids from
            prompting or session-rebuild code.

        Outputs (downstream usage):
            Exposes immutable token-id sequences consumed by generation,
            probing, and cache-fingerprint helpers.

        Invariants/constraints:
            Stored token ids must be normalized once at construction so
            downstream code does not need to defend against mutable caller-owned
            sequences.

        """
        object.__setattr__(self, "prompt_ids", prompt_ids)
        object.__setattr__(
            self, "generated_ids", tuple(int(value) for value in generated_ids)
        )
        object.__setattr__(self, "device", device)

    def full_prefix_ids(self) -> torch.Tensor:
        """Build the full prompt-plus-generated prefix tensor.

        Purpose:
            Concatenate prepared prompt ids with generated token ids so rebuild,
            probing, and scoring helpers can run on one complete prefix tensor.

        Architectural role:
            Main materialization method on the shared prefix-expansion value
            object.

        Inputs (architectural provenance):
            Reads prompt ids and generated ids stored on `PrefixExpansion`.

        Outputs (downstream usage):
            Returns the rank-2 token tensor consumed by runtime forward passes
            and related inference helpers.

        Invariants/constraints:
            The returned tensor must preserve prompt tokens first and generated
            tokens second.

        """
        if not self.generated_ids:
            return self.prompt_ids
        generated = torch.tensor(
            [list(self.generated_ids)], dtype=torch.long, device=self.device
        )
        return torch.cat([self.prompt_ids, generated], dim=1)


__all__ = ["PrefixExpansion", "normalize_prefix_ids"]


def prepare_prefix_input_ids(
    *, input_ids: torch.Tensor, tokenizer: TextCodec, device: torch.device
) -> torch.Tensor:
    """Validate and normalize prefix ids into a rank-2 long tensor.

    Purpose:
        Convert a Python sequence of prefix token ids into the tensor shape
        expected by model forward and generation calls.

    Architectural role:
        Tensor-materialization boundary between token planning and backend
        runtime execution.

    Inputs (architectural provenance):
        Receives normalized or caller-provided prefix ids and the execution
        device selected by the materialized runtime.

    Outputs (downstream usage):
        Returns a `torch.long` tensor with shape `(1, sequence_length)` consumed
        by backend calls.

    Invariants/constraints:
        Empty prefixes are rejected here because backend execution needs
        concrete input ids. Device placement is performed exactly once at this
        boundary.

    """
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("input_ids must be shape (1, T)")

    if torch.numel(input_ids) > 0:
        return input_ids.to(device=device, dtype=torch.long)

    bos_id = tokenizer.bos_token_id
    if bos_id is not None:
        token_id = int(bos_id)
    else:
        eos_id = tokenizer.eos_token_id
        if isinstance(eos_id, list):
            token_id = int(eos_id[0]) if eos_id else 0
        elif eos_id is not None:
            token_id = int(eos_id)
        else:
            token_id = 0

    return torch.tensor([[token_id]], device=device, dtype=torch.long)


def stable_prefix_fingerprint(prefix_ids: Sequence[int] | str) -> str:
    """Return a stable SHA1 fingerprint for text or token-id prefixes.

    Purpose:
        Produce compact identities for prompt or prefix material that may be
        stored in runtime snapshots, cache keys, or debug traces.

    Architectural role:
        Prompting utility at the boundary between potentially large prefix
        content and small telemetry/cache identifiers.

    Inputs (architectural provenance):
        Receives either raw prefix text or the token ids representing a prompt
        or generated prefix.

    Outputs (downstream usage):
        Returns a deterministic hexadecimal SHA1 digest suitable for equality
        checks and human-readable diagnostics.

    Invariants/constraints:
        Text prefixes are encoded as UTF-8. Token-id prefixes are serialized as
        unsigned integer array bytes, so callers must pass stable token ids
        rather than tokenizer-dependent objects.

    """
    if isinstance(prefix_ids, str):
        return hashlib.sha1(prefix_ids.encode("utf-8")).hexdigest()
    if not prefix_ids:
        return hashlib.sha1(b"").hexdigest()
    token_bytes = array(
        "I", (int(token_id) for token_id in prefix_ids)
    ).tobytes()
    return hashlib.sha1(token_bytes).hexdigest()
