"""Greedy token-by-token decode loop for streaming generation.

Purpose:
    Drive incremental generation, stream visible text, honor stop-token policy,
    and invoke core execution when Answer Engineering rules are enabled.

Architectural role:
    Central decode control loop beneath StreamSession.

Architectural direction:
    Keep greedy decode control simple while reducing entanglement among
    streaming output, rebuild handling, and intervention coordination.

Why this matters:
    This module is a central and real runtime boundary, but its current shape
    still reflects the implementation path used to combine streaming and
    intervention behavior.

What better would look like:
    Decode progression stays straightforward while streaming I/O and
    intervention/rebuild coordination have clearer ownership boundaries.

How improvement can be recognized:
    - Fewer unrelated reasons for this module to change
    - Clearer seams between decode control, stream output, and core integration
    - Less cross-cutting state coordination logic in one place

Open constraint:
    The decode-loop boundary should continue to adapt to runtime experimentation
    and operational learnings.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from answer_engineering.config.inference_defaults import (
    StreamRenderingDefaults,
)
from answer_engineering.engine.orchestration.orchestrator import (
    PlanRunner,
)
from answer_engineering.engine.runtime.runtime_types import TokenCharAlignment
from answer_engineering.engine.span_utils import validate_token_alignment
from answer_engineering.engine.telemetry.aggregation.aggregator import (
    RuntimeTelemetryAggregator,
)
from answer_engineering.engine.telemetry.events.event_sink import (
    CompositeRuntimeEventSink,
    ConsoleRuntimeEventSink,
    RecordingRuntimeEventSink,
)
from answer_engineering.engine.telemetry.snapshots.snapshots import (
    RuntimeTelemetrySnapshot,
)
from answer_engineering.inference.contracts import (
    GenerationPolicy,
    GenerationRequest,
)
from answer_engineering.inference.decode.session_orchestration import (
    ExecutionSession,
    RetokenizedAssistant,
)
from answer_engineering.inference.decode.state import (
    NestedTokenIds,
    StreamingDecodeState,
)
from answer_engineering.inference.model_types import (
    ChatGenerationRuntime,
    TextCodec,
)
from answer_engineering.inference.prompting import prompt_prefix
from answer_engineering.infra.console.reactive_visible_printer import (
    ReactiveVisiblePrinter,
)
from answer_engineering.infra.console.text_emitter import StdoutTextEmitter
from answer_engineering.infra.console.visible_layout import VisibleTextLayouter

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GreedyGenerationResult:
    """Final payload emitted by the greedy decode loop.

    Purpose:
        Package generated text, prompt/full ids, token alignment, and runtime
        telemetry from one streaming decode run.

    Architectural role:
        Decode-loop result object consumed by StreamSession.

    """

    text: str
    full_ids: torch.Tensor
    ae_telemetry: RuntimeTelemetrySnapshot | None = None

    def __iter__(self):
        """Yield result fields in tuple-unpack order.

        Purpose:
            Preserve ergonomic unpacking for callers that need generated text
            and token ids from a greedy decode result.

        Architectural role:
            Compatibility-style convenience on the decode-result value object,
            kept local to the inference boundary.

        Inputs (architectural provenance):
            Reads the generated text and full token-id sequence stored on the
            result.

        Outputs (downstream usage):
            Yields `(text, full_ids)` for notebook examples and internal callers
            that unpack decode results.

        Invariants/constraints:
            The yielded order must remain stable because tuple-unpacking call
            sites have no field names to protect them.

        """
        yield self.text
        yield self.full_ids


def _add_token_ids(target: set[int], value: NestedTokenIds) -> None:
    """Add one token id or a nested token-id collection into a set.

    Purpose:
        Normalize tokenizer stop-token metadata into the flat integer set used
        by greedy decode stop checks.

    Architectural role:
        Decode-policy helper between tokenizer-facing values and runtime control
        state.

    Inputs (architectural provenance):
        Receives a mutable target set plus an optional scalar, sequence, or
        nested collection returned by tokenizer properties or caller
        configuration.

    Outputs (downstream usage):
        Mutates the target set with every concrete token id found.

    Invariants/constraints:
        `None` values are ignored. The helper should preserve only integer token
        identities and should not impose ordering on the resulting set.

    """
    if value is None:
        return

    if isinstance(value, int):
        target.add(value)
        return

    for item in value:
        _add_token_ids(target, item)


def collect_stop_token_ids(tok: TextCodec) -> set[int]:
    """Collect EOS and other stop-token ids from the tokenizer into one set.

    Purpose:
        Build the canonical termination-token set used by greedy decoding.

    Architectural role:
        Decode-policy helper that translates tokenizer metadata and
        caller-provided stop ids into runtime control state.

    Inputs (architectural provenance):
        Receives a tokenizer-like codec, the `stop_on_eos` policy flag, and any
        additional stop ids supplied by the decode caller.

    Outputs (downstream usage):
        Returns a set of integer token ids consulted by the decode loop after
        each generated token.

    Invariants/constraints:
        EOS ids may be a scalar or a collection. Stop ids are accumulated
        without preserving order because membership checks are the only
        downstream use.

    """
    stop_ids: set[int] = set()
    _add_token_ids(stop_ids, tok.eos_token_id)
    return stop_ids


def _append_incremental_token_alignment(
    alignment: list[TokenCharAlignment],
    *,
    current_text: str,
    token_text: str,
    token_id: int | None = None,
) -> None:
    """Append one decoded piece to token-to-character alignment.

    Purpose:
        Extend the generated-text alignment as each newly decoded token piece
        becomes visible.

    Architectural role:
        Decode-loop bookkeeping helper. It keeps the generated token stream
        aligned with assistant-text character spans used later by probing and
        prefix rebuilds.

    Inputs (architectural provenance):
        Receives the current mutable alignment list and the decoded text piece
        for the next generated token.

    Outputs (downstream usage):
        Appends a `TokenCharAlignment` whose character span starts at the
        previous alignment end and whose token index matches append order.

    Invariants/constraints:
        Alignment is incremental and contiguous. The helper assumes pieces are
        appended in decode order and does not repair earlier spans.

    """
    start = len(current_text)
    end = start + len(token_text)
    alignment.append(
        TokenCharAlignment(
            token_index=len(alignment),
            char_start=start,
            char_end=end,
            piece_text=token_text,
            token_id=token_id,
        )
    )


@dataclass(frozen=True, slots=True)
class GreedyDecoder:
    """Stateful owner of one greedy decode run.

    Purpose:
        Advance generation one token at a time, stream visible text, apply core
        edits when configured, and stop when policy or stop-token conditions are
        met.

    Architectural role:
        Main control object inside the decode_loop module.

    """

    runtime: ChatGenerationRuntime
    input_ids: torch.Tensor
    request: GenerationRequest
    policy: GenerationPolicy

    def decode(self) -> GreedyGenerationResult:
        """Execute one greedy decode run until policy or stop-token termination.

        Purpose:
            Own token-by-token model stepping, optional core-edit intervention,
            streaming output, and telemetry/event collection for one generation
            request.

        Outputs:
            Returns a ``GreedyGenerationResult`` containing final text and ids,
            plus runtime telemetry when rule execution is enabled.

        Todo:
            Target:
                Keep this method as the decode-loop control owner while moving
                representation-only helper logic into decode-local utilities.

            Boundary note:
                It currently coordinates model I/O, streaming, and intervention
                hooks in one place; this is intentional, but low-level helper
                mechanics should stay factored out.

        """
        self.runtime.ensure_eval_mode()
        telemetry_sink: RecordingRuntimeEventSink | None = None
        tok = self.runtime.text_codec()
        prepared_input_ids = prompt_prefix.prepare_prefix_input_ids(
            input_ids=self.input_ids,
            tokenizer=tok,
            device=self.runtime.execution_device(),
        )

        with torch.inference_mode():
            out = self.runtime.forward(
                input_ids=prepared_input_ids, use_cache=True
            )
        retokenized = RetokenizedAssistant(tok, self.request.partial_answer)
        state = StreamingDecodeState(
            past_key_values=out.past_key_values,
            next_input=None,
            assistant_visible_text=self.request.partial_answer,
            generated_token_ids=retokenized.token_ids,
            eos_ids=collect_stop_token_ids(tok),
            device=self.runtime.execution_device(),
            prefill_next_logits=out.logits[:, -1, :],
            prefill_done=True,
            generated_token_alignment=retokenized.alignment,
        )
        stream_defaults = StreamRenderingDefaults()
        visible_printer = ReactiveVisiblePrinter(
            emitter=StdoutTextEmitter(),
            layouter=VisibleTextLayouter(wrap_width=stream_defaults.wrap_width),
            retractable_tail_chars=stream_defaults.retractable_tail_chars,
            debug_prefix=stream_defaults.debug_prefix,
        )
        if self.policy.stream_output:
            visible_printer.reset()
            visible_printer.observe_visible_text(state.current_visible_text())

        has_rules = self.policy.compiled_rules is not None
        execution_session: ExecutionSession | None = None
        if has_rules:
            assert self.policy.compiled_rules is not None
            runner = PlanRunner(
                verbose=self.policy.debug_output,
                trajectory_debug=self.policy.debug_output,
            )
            runner.runtime = self.runtime
            runner.require_model_scoring = True
            telemetry_sink = RecordingRuntimeEventSink()
            execution_session = ExecutionSession(
                plan=self.policy.compiled_rules.plan,
                runner=runner,
            )
            if self.policy.debug_output:
                runner.event_sink = CompositeRuntimeEventSink(
                    (
                        ConsoleRuntimeEventSink.with_debug_line_emitter(
                            visible_printer
                        ),
                        telemetry_sink,
                    )
                )
            else:
                runner.event_sink = telemetry_sink

        for tick in range(self.policy.max_new_tokens):
            if state.prefill_next_logits is not None:
                logits = state.prefill_next_logits
                state.prefill_next_logits = None
            else:
                assert state.next_input is not None
                with torch.inference_mode():
                    o = self.runtime.forward(
                        input_ids=state.next_input,
                        past_key_values=state.past_key_values,
                        use_cache=True,
                    )
                state.past_key_values = o.past_key_values
                logits = o.logits[:, -1, :]

            next_id = int(torch.argmax(logits, dim=-1).item())
            state.generated_token_ids.append(next_id)
            token_text = tok.decode([next_id], skip_special_tokens=True)
            old_text = state.assistant_visible_text
            new_text = old_text + token_text
            _append_incremental_token_alignment(
                state.generated_token_alignment,
                current_text=old_text,
                token_text=token_text,
                token_id=next_id,
            )
            state.assistant_visible_text = new_text
            err = validate_token_alignment(
                state.generated_token_alignment, state.assistant_visible_text
            )
            if err is not None:
                _LOG.warning(
                    "invalid_incremental_alignment_reset error=%s", err
                )
                state.generated_token_alignment = []
            state.next_input = torch.tensor(
                [[next_id]],
                dtype=torch.long,
                device=self.runtime.execution_device(),
            )

            if has_rules:
                assert execution_session is not None
                changed = execution_session.apply_step(
                    tick_index=tick,
                    state=state,
                    prompt_ids=prepared_input_ids,
                    prompt_text=self.request.question,
                )

                if changed:
                    state.applied_decisions += 1

                if state.needs_rebuild:
                    execution_session.rebuild_decode_state(
                        state=state,
                        tokenizer=tok,
                        runtime=self.runtime,
                        prepared_input_ids=prepared_input_ids,
                    )
            if self.policy.stream_output:
                visible_printer.observe_visible_text(
                    state.current_visible_text()
                )

            if self.policy.stop_on_eos and next_id in state.eos_ids:
                break

        if self.policy.stream_output:
            visible_printer.observe_visible_text(
                state.current_visible_text(),
                is_final=True,
            )
            visible_printer.terminate_visible_output_line()

        text = state.assistant_visible_text.rstrip()
        ae_telemetry = RuntimeTelemetrySnapshot.empty(
            decision_limit_reached=state.decision_limit_reached
        )
        if telemetry_sink is not None and execution_session is not None:
            aggregator = RuntimeTelemetryAggregator(
                rule_name_for=execution_session.rule_name
            )
            aggregator.observe_events(telemetry_sink.events)
            ae_telemetry = aggregator.build_snapshot(
                decision_limit_reached=state.decision_limit_reached
            )
        full_ids = (
            torch.cat(
                [
                    prepared_input_ids,
                    torch.tensor(
                        [state.generated_token_ids],
                        dtype=torch.long,
                        device=self.runtime.execution_device(),
                    ),
                ],
                dim=1,
            )
            if state.generated_token_ids
            else prepared_input_ids
        )
        return GreedyGenerationResult(
            text=text, full_ids=full_ids, ae_telemetry=ae_telemetry
        )


__all__ = [
    "collect_stop_token_ids",
    "GreedyGenerationResult",
    "GreedyDecoder",
]
