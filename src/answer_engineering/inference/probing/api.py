"""Probing-internal value contracts.

Purpose:
    Define lightweight probing values shared by probing generation helpers and
    probing runtime internals.

Architectural role:
    Internal probing value module, not a closed external probing API.

"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from answer_engineering.inference.model_types import PastKeyValues


@dataclass(slots=True)
class ProbeContext:
    """Runtime context needed to continue generation from a probe prefix.

    Purpose:
        Bundle cached generation state and the tensors required to continue
        probing without reconstructing the full model state each time.

    Architectural role:
        Low-level probing runtime record.

    Inputs (architectural provenance):
        Produced by probing/runtime code after prefix preparation and model
        forward passes.

    Outputs (downstream usage):
        Consumed by probing generation helpers that continue beam search from
        the prepared probe prefix.

    """

    past_key_values: PastKeyValues
    next_logits: torch.Tensor
    device: torch.device
    prefix_input_ids: torch.Tensor


@dataclass(slots=True)
class ProbeCandidate:
    """Raw probing candidate value used by generation helpers.

    Purpose:
        Represent one generated continuation with token ids and summed logprob
        in probing generation flow before higher-level runtime adaptation.

    Architectural role:
        Probing-facing candidate contract.

    Notes:
        This is a generation-facing candidate carrier and is not yet the single
        canonical probing candidate record across the probing tree.

    """

    token_ids: list[int]
    logprob_sum: float
    text: str


__all__ = ["ProbeContext", "ProbeCandidate"]
