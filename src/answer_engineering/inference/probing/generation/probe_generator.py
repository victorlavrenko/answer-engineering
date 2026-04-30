"""Probe-generation flow and first-stage candidate normalization.

Owns now:
    - Raw probe generation flow from prepared prefixes.
    - Oversampled beam probing and conversion of generated sequences into
      probing candidates.
    - First-stage suffix normalization and trajectory-prefix filtering before
      runtime adaptation.

Candidate-shape distinctions:
    - Raw generation candidates: ``inference.probing.api.ProbeCandidate``.
    - Runtime cache replay payloads: ``ProbeCacheCandidate`` in runtime cache
      modules.
    - Final proposal-facing outputs: ``CandidateSpec`` (adapted later by
      ``ProbeRuntime``).

"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from answer_engineering.config.inference_defaults import ProbeDefaults
from answer_engineering.engine.pipeline.events import (
    DebugEvent,
)
from answer_engineering.engine.telemetry.events.event_sink import (
    NullRuntimeEventSink,
    RuntimeEventSink,
)
from answer_engineering.inference.model_types import (
    GenerationRuntimeProtocol,
    TextCodec,
)
from answer_engineering.inference.probing.api import ProbeCandidate
from answer_engineering.inference.probing.generation import (
    grouped_beam_generator,
)
from answer_engineering.inference.prompting import prompt_prefix


def _tail(text: str, limit: int = 120) -> str:
    return text[-limit:]


def _single_line(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def _norm_ws(text: str) -> str:
    return " ".join(text.lstrip().split())


def _is_on_trajectory_prefix(existing_head: str, cand_suffix: str) -> bool:
    existing_norm = _norm_ws(existing_head)
    cand_norm = _norm_ws(cand_suffix)
    if not cand_norm:
        return True
    return existing_norm.startswith(cand_norm)


def _debug_input_ids(*, tok: TextCodec, input_ids: torch.Tensor) -> str:
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        return f"input_ids shape={tuple(input_ids.shape)} (expected (1,T))"
    ids = _tensor_to_int_list(input_ids[0])
    decoded = tok.decode(ids, skip_special_tokens=True)
    return f"input_ids shape=(1,{len(ids)}) decoded={decoded!r}"


def _tensor_to_int_list(tokens: torch.Tensor) -> list[int]:
    vector = tokens.detach().to(device="cpu").reshape(-1)
    return [int(value.item()) for value in vector]


def _probe_debug(
    event_sink: RuntimeEventSink | None, enabled: bool, *, msg: str
) -> None:
    if not enabled or event_sink is None:
        return
    event_sink.emit(DebugEvent(msg=msg))


@dataclass(frozen=True, slots=True)
class ProbeGenerator:
    """Generator for probing generation candidates from prepared prefix inputs.

    Purpose:
        Run generation with probing-specific adaptations to produce raw probe
        candidates for later normalization and runtime adaptation.

    Owns:
        - Running generation with oversampled beam search for probe candidate
          generation.
        - Converting raw generated sequences into ``ProbeCandidate`` values.
        - First-stage probe candidate normalization and trajectory-prefix
          filtering.

    Does not own:
        - Candidate normalization/filtering policy beyond basic
          trajectory-prefix filtering.
        - Probe cache lifecycle.
        - Proposal-facing candidate adaptation.

    """

    runtime: GenerationRuntimeProtocol

    def run_pivot_probes_unsorted(
        self,
        *,
        tok: TextCodec,
        input_ids: torch.Tensor,
        pivot_token_idx: int,
        num_beams: int,
        max_new_tokens: int | None = None,
        event_sink: RuntimeEventSink,
        trajectory_debug: bool,
    ) -> tuple[list[ProbeCandidate], ProbeCandidate]:
        """Run raw probing generation at one pivot position.

        Purpose:
            Generate beam candidates from an already prepared prefix tensor
            without applying later probe-result normalization or proposal-
            facing adaptation.

        Architectural role:
            Low-level generation operation inside the probing subsystem. It is
            the boundary between prepared probe prefixes and backend sequence
            generation.

        Inputs (architectural provenance):
            Receives tokenizer/runtime collaborators, a single-row prefix
            tensor, pivot index, beam count, generation depth, event sink, and
            trajectory-debug policy from higher-level probing runtime code.

        Outputs (downstream usage):
            Returns raw `ProbeCandidate` beams plus an empty fallback candidate
            consumed by `generate_beams_then_fallback` and probe cache
            population.

        Invariants/constraints:
            `input_ids` must be shape `(1, T)` and `pivot_token_idx` must equal
            the prefix length. The method preserves backend generation order and
            does not perform global candidate selection.

        """
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("input_ids must be shape (1, T)")
        expected = int(input_ids.shape[1])
        if pivot_token_idx != expected:
            raise ValueError(
                f"pivot_token_idx={pivot_token_idx} must equal "
                f"input_ids length={expected}"
            )
        generation_depth = (
            ProbeDefaults().max_new_tokens
            if max_new_tokens is None
            else max(0, max_new_tokens)
        )
        requested_num_beams = max(0, num_beams)
        if requested_num_beams == 0:
            fallback = ProbeCandidate(
                token_ids=[], logprob_sum=float("-inf"), text=""
            )
            return list(), fallback
        _probe_debug(
            event_sink,
            trajectory_debug,
            msg=(
                "PROBE_CALL "
                f"pivot_token_idx={pivot_token_idx} num_beams={num_beams} "
                f"max_new_tokens={generation_depth}"
            ),
        )
        _probe_debug(
            event_sink,
            trajectory_debug,
            msg=(
                f"PROBE_START pivot_token_idx={pivot_token_idx} "
                f"num_beams={num_beams} "
                f"max_new_tokens={generation_depth}"
            ),
        )
        _probe_debug(
            event_sink,
            trajectory_debug,
            msg="PROBE_INPUT " + _debug_input_ids(tok=tok, input_ids=input_ids),
        )
        prefix_tail = tok.decode(
            _tensor_to_int_list(input_ids[0, -80:]),
            skip_special_tokens=True,
        )
        _probe_debug(
            event_sink,
            trajectory_debug,
            msg=(
                f"PROBE_INPUT_SHAPE prefix_ids_shape={tuple(input_ids.shape)} "
                f'prefix_tail="{_single_line(prefix_tail)}"'
            ),
        )
        generation_num_beams = max(1, requested_num_beams)
        num_return_sequences = min(requested_num_beams, generation_num_beams)
        gen = grouped_beam_generator.GroupedBeamGenerator(
            self.runtime
        ).generate(
            input_ids=input_ids,
            num_beams=generation_num_beams,
            num_return_sequences=num_return_sequences,
            max_new_tokens=generation_depth,
            early_stopping=True,
            output_scores=True,
        )
        sequences = gen.sequences
        sequence_scores = getattr(gen, "sequences_scores", None)
        prefix_len = int(input_ids.shape[1])
        beams: list[ProbeCandidate] = []
        for i in range(int(sequences.shape[0])):
            seq = sequences[i]
            suffix_ids = _tensor_to_int_list(seq[prefix_len:])
            suffix_text = tok.decode(suffix_ids, skip_special_tokens=True)
            score = (
                float(sequence_scores[i].item())
                if sequence_scores is not None
                else float("nan")
            )
            _probe_debug(
                event_sink,
                trajectory_debug,
                msg=(
                    f"[AE] BEAM i={i} score={score:.6f} "
                    f"len_tokens={len(suffix_ids)} "
                    f'suffix="{_single_line(_tail(suffix_text, 200))}" '
                    f"token_ids={suffix_ids}"
                ),
            )
            beams.append(
                ProbeCandidate(
                    token_ids=suffix_ids, logprob_sum=score, text=suffix_text
                )
            )
        fallback = ProbeCandidate(
            token_ids=[], logprob_sum=float("-inf"), text=""
        )
        return beams[:requested_num_beams], fallback

    def generate_beams_then_fallback(
        self,
        *,
        prefix_ids: list[int],
        num_beams: int,
        max_new_tokens: int,
        abs_start: int | None = None,
        doc_text: str | None = None,
        trajectory_debug: bool = False,
        debug_replay_check: bool = False,
        event_sink: RuntimeEventSink | None = None,
    ) -> tuple[list[ProbeCandidate], ProbeCandidate]:
        """Generate normalized probe beams for a prefix with fallback.

        Purpose:
            Normalize a probe prefix, oversample backend generation, drop
            candidates that merely continue the existing trajectory, and return
            ranked usable probe candidates with an empty fallback.

        Architectural role:
            First-stage probing generation pipeline between prefix
            assembly/cache logic and proposal-facing candidate adaptation.

        Inputs (architectural provenance):
            Receives prefix token ids, beam/depth controls, optional document
            context, debug controls, and an optional event sink from
            `ProbeRuntime` or tests.

        Outputs (downstream usage):
            Returns filtered `ProbeCandidate` beams and a fallback candidate
            consumed by probe cache population and avoid-candidate serving.

        Invariants/constraints:
            Prefix ids are normalized before tensor construction. Generated
            candidates that are just on-trajectory continuations of the existing
            document head are discarded before ranking.

        """
        del debug_replay_check
        requested_num_beams = max(1, num_beams)
        oversampled_num_beams = requested_num_beams + 1
        effective_prefix_ids = prompt_prefix.normalize_prefix_ids(
            tokenizer=self.runtime.text_codec(),
            prefix_ids=prefix_ids,
            fallback_to_zero=True,
        )
        tokenizer = self.runtime.text_codec()
        prefix_tail = tokenizer.decode(
            effective_prefix_ids, skip_special_tokens=True
        )
        _probe_debug(
            event_sink,
            trajectory_debug,
            msg=(
                f"PROBE_START pivot_token_idx={len(effective_prefix_ids)} "
                f"num_beams={num_beams} max_new_tokens={max_new_tokens} "
                "prefix_ids_len="
                f"{len(effective_prefix_ids)} abs_start={abs_start}"
            ),
        )
        _probe_debug(
            event_sink,
            trajectory_debug,
            msg=f'PROBE_PREFIX tail="{_single_line(_tail(prefix_tail))}"',
        )
        if doc_text is not None and abs_start is not None:
            lo = max(0, abs_start - 40)
            hi = min(len(doc_text), abs_start + 40)
            around = doc_text[lo:hi]
            _probe_debug(
                event_sink,
                trajectory_debug,
                msg=f'PROBE_ABS_START around="{_single_line(around)}"',
            )
        sink = event_sink if event_sink is not None else NullRuntimeEventSink()
        input_ids = torch.tensor(
            [effective_prefix_ids],
            device=self.runtime.execution_device(),
            dtype=torch.long,
        )
        beams, _ = self.run_pivot_probes_unsorted(
            tok=tokenizer,
            input_ids=input_ids,
            pivot_token_idx=len(effective_prefix_ids),
            num_beams=oversampled_num_beams,
            max_new_tokens=max_new_tokens,
            event_sink=sink,
            trajectory_debug=trajectory_debug,
        )
        prefix_str = tokenizer.decode(
            effective_prefix_ids, skip_special_tokens=True
        )
        contextual_texts: list[str] = []
        for cand in beams:
            full_str = tokenizer.decode(
                [*effective_prefix_ids, *cand.token_ids],
                skip_special_tokens=True,
            )
            suffix = (
                full_str[len(prefix_str) :]
                if full_str.startswith(prefix_str)
                else full_str
            )
            contextual_texts.append(suffix)
        existing_head = ""
        if doc_text is not None and abs_start is not None:
            head_start = max(0, abs_start)
            head_end = min(len(doc_text), head_start + 300)
            existing_head = doc_text[head_start:head_end]
        paired = tuple(zip(beams, contextual_texts, strict=False))
        filtered_pairs: list[tuple[ProbeCandidate, str]] = []
        for idx, (cand, suffix_text) in enumerate(paired):
            if existing_head and _is_on_trajectory_prefix(
                existing_head, suffix_text
            ):
                _probe_debug(
                    event_sink,
                    trajectory_debug,
                    msg=(
                        f"PROBE_DROP on_trajectory_prefix i={idx} "
                        f'suffix="{_single_line(_tail(suffix_text, 200))}" '
                        "existing_head="
                        f'"{_single_line(_tail(existing_head, 200))}"'
                    ),
                )
                continue
            filtered_pairs.append((cand, suffix_text))
        filtered_pairs.sort(key=lambda item: item[0].logprob_sum, reverse=True)
        filtered_pairs = filtered_pairs[:requested_num_beams]
        _probe_debug(
            event_sink,
            trajectory_debug,
            msg=f"PROBE_RESULT num_probes={len(filtered_pairs)}",
        )
        for idx, (cand, suffix_text) in enumerate(filtered_pairs):
            full_text = tokenizer.decode(
                [*effective_prefix_ids, *cand.token_ids],
                skip_special_tokens=True,
            )
            full_tail = _single_line(_tail(full_text))
            _probe_debug(
                event_sink,
                trajectory_debug,
                msg=(
                    f"PROBE_CAND {idx} score={cand.logprob_sum:.6f} "
                    f"len_chars={len(suffix_text)} "
                    "token_ids="
                    f'{cand.token_ids} raw="{_single_line(suffix_text)}" '
                    f'suffix="{_single_line(suffix_text)}" '
                    f'full_tail="{full_tail}"'
                ),
            )
        resolved_beams = [
            ProbeCandidate(
                token_ids=c.token_ids,
                logprob_sum=c.logprob_sum,
                text=suffix_text,
            )
            for c, suffix_text in filtered_pairs
        ]
        fallback = ProbeCandidate(
            token_ids=[], logprob_sum=float("-inf"), text=""
        )
        return resolved_beams, fallback


@dataclass(frozen=True, slots=True, init=False)
class ProbeResult:
    """Container for one probing run's generated candidates and fallback.

    Purpose:
        Hold the ranked probe beams and the empty fallback candidate returned
        when probing produces no usable suffix.

    Architectural role:
        Probing result record between generation helpers, cache population, and
        higher-level probing runtime code.

    Inputs (architectural provenance):
        Built either directly from precomputed beams/fallback values or by
        running probe_beams_then_fallback_impl.

    Outputs (downstream usage):
        Consumed by ProbeRuntime when populating cache entries and adapting
        results into proposal-facing candidates.

    """

    beams: list[ProbeCandidate]
    fallback: ProbeCandidate

    def __init__(
        self,
        runtime: GenerationRuntimeProtocol | None = None,
        *,
        prefix_ids: list[int] | None = None,
        num_beams: int = 0,
        max_new_tokens: int = 0,
        abs_start: int | None = None,
        doc_text: str | None = None,
        trajectory_debug: bool = False,
        debug_replay_check: bool = False,
        event_sink: RuntimeEventSink | None = None,
        beams: list[ProbeCandidate] | None = None,
        fallback: ProbeCandidate | None = None,
    ) -> None:
        """Use supplied beams and fallback text or run probing to compute them.

        Purpose:
            Support explicit result construction for tests and cached paths
            while also offering the normal construction path that invokes the
            probe generator.

        Architectural role:
            Rich constructor for the probing result boundary. It packages
            generated candidate beams, fallback text, and probe metadata for
            proposal generation.

        Inputs (architectural provenance):
            Explicit mode receives beams and fallback text from a caller.
            Runtime mode receives probe prefix state and generation dependencies
            capable of producing those values.

        Outputs (downstream usage):
            Stores normalized probe beams and fallback text consumed by
            avoid-candidate proposal logic.

        Invariants/constraints:
            Callers must provide a complete explicit result or the inputs
            required to run probing. The stored beams should describe one probe
            episode for one prefix.

        """
        if beams is not None and fallback is not None:
            object.__setattr__(self, "beams", beams)
            object.__setattr__(self, "fallback", fallback)
            return
        if runtime is None or prefix_ids is None:
            raise ValueError(
                "ProbeResult requires runtime/prefix_ids "
                "or explicit beams/fallback."
            )
        resolved_beams, resolved_fallback = ProbeGenerator(
            runtime
        ).generate_beams_then_fallback(
            prefix_ids=prefix_ids,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            abs_start=abs_start,
            doc_text=doc_text,
            trajectory_debug=trajectory_debug,
            debug_replay_check=debug_replay_check,
            event_sink=event_sink,
        )
        object.__setattr__(self, "beams", resolved_beams)
        object.__setattr__(self, "fallback", resolved_fallback)


__all__ = ["ProbeGenerator", "ProbeResult"]
