"""Probe-prefix preparation helpers for alignment-aware probing requests.

Purpose:
    Build probe prefix ids by preferring alignment-based generated-id reuse when
    trustworthy, otherwise falling back to encoding visible assistant prefix
    text from the document snapshot.

Architectural role:
    Probing-owned prefix-preparation module between decode state and probe
    execution.

Boundary note:
    Fingerprinting, cache identity, and cache lifecycle are owned outside this
    module (runtime cache helpers and ``ProbeRuntime``).

"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from answer_engineering.engine.runtime.runtime_types import TokenCharAlignment
from answer_engineering.engine.span_utils import (
    clamp_index,
    validate_token_alignment_detailed,
)
from answer_engineering.inference.model_types import TextCodec

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, init=False)
class ProbeRequestPrefix:
    """Prepared token prefix for one probe request.

    Purpose:
        Hold the probe prefix token ids together with a flag indicating whether
        generated-token alignment was used to construct them.

    Architectural role:
        Probing-owned value object for prefix preparation.

    Inputs (architectural provenance):
        Built from prompt ids, document text, absolute start offset, and
        optionally generated-token alignment.

    Outputs (downstream usage):
        Consumed by ProbeRuntime when constructing a probe prefix snapshot.

    """

    prefix_ids: list[int]
    used_generated_alignment: bool

    def __init__(
        self,
        *,
        tok: TextCodec,
        prompt_ids: list[int],
        doc_text: str,
        abs_start: int,
        generated_ids: Sequence[int] | None = None,
        generated_token_alignment: tuple[TokenCharAlignment, ...] = tuple(),
    ) -> None:
        """Build probe prefix ids and track generated-prefix alignment usage.

        Purpose:
            Assemble the token prefix used for a probe request while preserving
            whether the prefix was sliced from generated-token alignment or
            re-encoded from text.

        Architectural role:
            Construction boundary between runtime text views and model probing.
            It centralizes prefix assembly so probing code can reason about
            cache identity and alignment provenance.

        Inputs (architectural provenance):
            Receives tokenizer access, prompt ids, generated ids with alignment
            data, and the absolute probe start position.

        Outputs (downstream usage):
            Stores prompt ids, prefix ids, and alignment-use metadata consumed
            by probe generation and debug reporting.

        Invariants/constraints:
            Prefix ids must correspond to the document content before the probe
            span. Alignment metadata should be used only when it can faithfully
            represent the requested absolute boundary.

        """
        prefix_ids, used_generated_alignment = _build_probe_prefix_ids(
            tok=tok,
            prompt_ids=prompt_ids,
            doc_text=doc_text,
            abs_start=abs_start,
            generated_ids=generated_ids,
            generated_token_alignment=generated_token_alignment,
        )
        object.__setattr__(self, "prefix_ids", prefix_ids)
        object.__setattr__(
            self, "used_generated_alignment", used_generated_alignment
        )


def _slice_generated_prefix_ids(
    *,
    generated_ids: Sequence[int],
    generated_token_alignment: tuple[TokenCharAlignment, ...],
    abs_start: int,
) -> tuple[int, ...] | None:
    """Slice generated ids up to the token ending before an edit start.

    Purpose:
        Reuse already generated token ids when constructing a probe prefix for
        an edit that begins inside the assistant text.

    Architectural role:
        Probe-prefix alignment helper. It avoids unnecessary text re-encoding
        when generated ids and character alignment still describe the current
        document.

    Inputs (architectural provenance):
        Receives generated ids, per-token character alignment, and the absolute
        edit start that should bound the reusable prefix.

    Outputs (downstream usage):
        Returns the generated-id prefix ending at or before `abs_start`, or
        `None` when alignment and token ids are inconsistent.

    Invariants/constraints:
        The helper never includes a token whose character span crosses the edit
        start. Token and alignment lengths must match exactly to use the fast
        path.

    """
    if len(generated_ids) != len(generated_token_alignment):
        return None

    cutoff = 0
    for item in generated_token_alignment:
        if item.char_end <= abs_start:
            cutoff = item.token_index + 1
            continue
        break
    return tuple(generated_ids[:cutoff])


def _build_probe_prefix_ids(
    *,
    tok: TextCodec,
    prompt_ids: list[int],
    doc_text: str,
    abs_start: int,
    generated_ids: Sequence[int] | None = None,
    generated_token_alignment: tuple[TokenCharAlignment, ...] = tuple(),
) -> tuple[list[int], bool]:
    """Build probe prefix ids from alignment when possible, else text encoding.

    Purpose:
        Prefer alignment-based slicing when generated ids and alignment are
        trustworthy; otherwise encode the document prefix text directly.

    """
    abs_start = clamp_index(abs_start, doc_text)
    if generated_ids and generated_token_alignment:
        err = validate_token_alignment_detailed(
            generated_token_alignment, doc_text
        )
        if err is not None:
            log = (
                _LOG.warning
                if err.kind
                in {
                    "char_end_out_of_bounds",
                    "negative_char_start",
                    "char_end_before_start",
                }
                else _LOG.debug
            )
            log(
                "invalid_generated_alignment_fallback_reencode %s",
                err.compact(),
            )
        else:
            aligned_ids = _slice_generated_prefix_ids(
                generated_ids=generated_ids,
                generated_token_alignment=generated_token_alignment,
                abs_start=abs_start,
            )
            if aligned_ids is not None:
                return [*prompt_ids, *aligned_ids], True

    assistant_prefix_text = doc_text[:abs_start]
    assistant_prefix_ids = tok.encode(
        assistant_prefix_text, add_special_tokens=False
    )
    return [*prompt_ids, *assistant_prefix_ids], False
