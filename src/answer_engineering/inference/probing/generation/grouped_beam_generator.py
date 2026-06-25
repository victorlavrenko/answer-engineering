"""Grouped-beam generation adapter used by probing.

Owns:
    - Adapting probing generation settings into ``GenerationCall`` and
      ``GenerationControl``.
    - Preparing validated prefix ``input_ids`` and attention masks.
    - Invoking ``runtime.generate_tokens(...)`` for probe expansion.

Does not own:
    - Candidate normalization/filtering policy.
    - Probe cache lifecycle.
    - Proposal-facing candidate adaptation.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from huggingface_hub import snapshot_download

from answer_engineering.config.inference_defaults import ProbeDefaults
from answer_engineering.inference.model_types import (
    CausalLMGenerationOutput,
    GenerationCall,
    GenerationControl,
    GenerationRuntimeProtocol,
    TextCodec,
)
from answer_engineering.inference.prompting import prompt_prefix


@dataclass(frozen=True, slots=True, init=False)
class GroupBeamCustomGenerateConfig:
    """Configuration for grouped-beam custom generate() loading.

    Purpose:
        Carry the repository/path, revision, trust_remote_code flag, and
        optional preload behavior for the grouped-beam search implementation
        used by probing.

    Architectural role:
        Probing generation configuration value.

    """

    custom_generate: str
    trust_remote_code: bool
    revision: str | None
    use_cache: bool | None

    def __init__(
        self,
        *,
        custom_generate: str,
        trust_remote_code: bool,
        revision: str | None,
        preload_group_beam_search: bool = False,
        use_cache: bool | None = None,
    ) -> None:
        """Resolve custom-generation source and optional preload behavior.

        Purpose:
            Normalize local or repository-based custom generation configuration
            before grouped probe generation starts.

        Architectural role:
            Adapter constructor between high-level probing configuration and the
            custom generation loader used by Hugging Face generation code.

        Inputs (architectural provenance):
            Receives a source path or repository id plus optional revision,
            trust, and preload flags from probing configuration.

        Outputs (downstream usage):
            Stores the resolved custom-generate settings consumed by grouped
            beam generation.

        Invariants/constraints:
            Resolution should be deterministic for a given source and revision.
            Loading behavior belongs here rather than being scattered across
            probe callers.

        """
        if preload_group_beam_search:
            try:
                local_repo_path = snapshot_download(
                    repo_id=custom_generate,
                    revision=revision or "main",
                    allow_patterns=["custom_generate/*", "*.py"],
                )
            except (OSError, RuntimeError, ValueError):
                object.__setattr__(self, "custom_generate", custom_generate)
                object.__setattr__(self, "trust_remote_code", trust_remote_code)
                object.__setattr__(self, "revision", revision)
                object.__setattr__(self, "use_cache", use_cache)
                return

            object.__setattr__(self, "custom_generate", local_repo_path)
            object.__setattr__(self, "trust_remote_code", False)
            object.__setattr__(self, "revision", None)
            object.__setattr__(self, "use_cache", use_cache)
            return

        object.__setattr__(self, "custom_generate", custom_generate)
        object.__setattr__(self, "trust_remote_code", trust_remote_code)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "use_cache", use_cache)


def _first_token_id(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and value
    ):
        head = cast(Sequence[object], value)[0]
        return int(head) if isinstance(head, int) else None
    return None


def _resolve_pad_token_id(*, tokenizer: TextCodec) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)

    tok_eos = _first_token_id(tokenizer.eos_token_id)
    if tok_eos is not None:
        return tok_eos

    raise ValueError(
        "Could not resolve pad_token_id: tokenizer.pad_token_id "
        "and eos_token_id are unset."
    )


def _build_attention_mask_unpadded(
    *, input_ids: torch.Tensor, pad_token_id: int | None = None
) -> torch.Tensor:
    assert input_ids.ndim == 2, "input_ids must be rank-2"
    if pad_token_id is not None:
        assert not bool(torch.any(input_ids == pad_token_id).item()), (
            "_build_attention_mask_unpadded assumes no padding "
            "tokens in input_ids"
        )
    return torch.ones_like(input_ids)


@dataclass(frozen=True, slots=True)
class GroupedBeamGenerator:
    """Generator adapter for ordinary or grouped beam probing.

    Purpose:
        Adapt probing generation parameters into runtime generation contracts
        and execute ``runtime.generate_tokens(...)`` for probe expansion.

    Owns:
        - Adapting probing generation settings into ``GenerationCall`` and
          ``GenerationControl``.
        - Preparing validated prefix ``input_ids`` and attention masks.
        - Invoking ``runtime.generate_tokens(...)`` for probe expansion.

    Does not own:
        - Candidate normalization/filtering policy.
        - Probe cache lifecycle.
        - Proposal-facing candidate adaptation.

    """

    runtime: GenerationRuntimeProtocol

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        num_beams: int,
        num_return_sequences: int,
        max_new_tokens: int,
        beams_per_group: int = ProbeDefaults().beams_per_group,
        num_beam_groups: int | None = None,
        diversity_penalty: float = ProbeDefaults().diversity_penalty,
        use_group_beam_search: bool = ProbeDefaults().use_group_beam_search,
        group_beam_search_revision: str
        | None = ProbeDefaults().group_beam_search_revision,
        preload_group_beam_search: bool = (
            ProbeDefaults().preload_group_beam_search
        ),
        early_stopping: bool = True,
        output_scores: bool = True,
    ) -> CausalLMGenerationOutput:
        """Generate probe continuations with ordinary or grouped beam search.

        Purpose:
            Adapt probing generation parameters into runtime generation
            contracts and execute ``runtime.generate_tokens(...)`` for probe
            expansion.

        """
        runtime = self.runtime
        tokenizer = runtime.text_codec()

        if beams_per_group < 1:
            raise ValueError("beams_per_group must be >= 1")
        if num_return_sequences < 1:
            raise ValueError("num_return_sequences must be >= 1")
        if diversity_penalty < 0:
            raise ValueError("diversity_penalty must be >= 0")

        if use_group_beam_search:
            resolved_num_beam_groups = (
                num_return_sequences
                if num_beam_groups is None
                else num_beam_groups
            )
            resolved_num_beams = beams_per_group * num_return_sequences
        else:
            resolved_num_beam_groups = num_beam_groups
            resolved_num_beams = num_beams

        if resolved_num_beams < 1:
            raise ValueError("num_beams must be >= 1")
        if num_return_sequences > resolved_num_beams:
            raise ValueError("num_return_sequences must be <= num_beams")

        if resolved_num_beam_groups is not None:
            if resolved_num_beam_groups < 1:
                raise ValueError("num_beam_groups must be >= 1")
            if resolved_num_beam_groups > resolved_num_beams:
                raise ValueError("num_beam_groups must be <= num_beams")
            if resolved_num_beams % resolved_num_beam_groups != 0:
                raise ValueError(
                    "num_beams must be divisible by num_beam_groups"
                )

        prepared_input_ids = prompt_prefix.prepare_prefix_input_ids(
            input_ids=input_ids,
            tokenizer=tokenizer,
            device=runtime.execution_device(),
        )

        pad_token_id = _resolve_pad_token_id(tokenizer=tokenizer)
        attention_mask = _build_attention_mask_unpadded(
            input_ids=prepared_input_ids,
            pad_token_id=pad_token_id,
        )

        custom_generate_config: GroupBeamCustomGenerateConfig | None = None
        if use_group_beam_search:
            custom_generate_config = GroupBeamCustomGenerateConfig(
                custom_generate=ProbeDefaults().group_beam_search_repo_id,
                revision=group_beam_search_revision,
                trust_remote_code=True,
                preload_group_beam_search=preload_group_beam_search,
                use_cache=True,
            )

        generation_control = GenerationControl(
            max_new_tokens=max_new_tokens,
            num_beams=resolved_num_beams,
            num_beam_groups=resolved_num_beam_groups,
            diversity_penalty=diversity_penalty,
            num_return_sequences=num_return_sequences,
            output_scores=output_scores,
            early_stopping=early_stopping,
        )
        generation_call = GenerationCall(
            input_ids=prepared_input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            control=generation_control,
            custom_generate_config=custom_generate_config,
        )
        with torch.inference_mode():
            return runtime.generate_tokens(generation_call)


__all__ = [
    "GroupBeamCustomGenerateConfig",
    "GroupedBeamGenerator",
]
