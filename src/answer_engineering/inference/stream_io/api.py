"""Per-call streaming session execution object.

Purpose:
    Build chat messages, prepare prompt ids, run the greedy decoder, and package
    the final `GenerationResult` for one generation request.

Architectural role:
    Session object created by `GenerationRuntime.generate(...)` to run one call.

Key relationships:
    - Called by `GenerationRuntime.generate(...)`.
    - Delegates token generation to `GreedyDecoder`.
    - Attaches runtime telemetry snapshots from decode results.

Failure modes / limitations:
    - This module still owns some prompt-id preparation details
      (`_build_input_ids`) that could move to prompting-focused helpers later.

"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from answer_engineering.engine.telemetry.snapshots.snapshots import (
    RuntimeTelemetrySnapshot,
)
from answer_engineering.inference.contracts import (
    GenerationPolicy,
    GenerationRequest,
    GenerationResult,
)
from answer_engineering.inference.decode.decode_loop import (
    GreedyDecoder,
)
from answer_engineering.inference.model_types import (
    ChatGenerationRuntime,
)
from answer_engineering.inference.prompting import input_ids as prompt_input_ids
from answer_engineering.inference.prompting.chat_format import (
    ChatTranscript,
)


@dataclass(frozen=True, slots=True)
class StreamSession:
    """Session object that executes one streaming generation call.

    Purpose:
        Build prompt messages, prepare input ids, run the greedy decoder, and
        package the final `GenerationResult` for the owning runtime call.

    Architectural role:
        Per-call execution coordinator at the top of the inference decode path.

    """

    runtime: ChatGenerationRuntime
    request: GenerationRequest
    policy: GenerationPolicy

    def run(self) -> GenerationResult:
        """Run one high-level streaming generation request.

        Purpose:
            Build chat messages, materialize prompt ids, invoke the greedy
            decode loop, and package the completed response into the public
            `GenerationResult`.

        Architectural role:
            Session execution method used by `GenerationRuntime.generate(...)`.

        Inputs (architectural provenance):
            Consumes a structured generation request together with the runtime,
            policy, and tokenizer stored on the session.

        Outputs (downstream usage):
            Returns the final generation result consumed by notebooks, scripts,
            and reproduction code.

        Invariants/constraints:
            Runtime, prompt construction, and decode execution must all refer to
            the same request and policy configuration.

        """
        messages = ChatTranscript(
            self.policy.system_prompt, self.request.question
        ).messages()
        input_ids = prompt_input_ids.build_input_ids(
            self.runtime,
            messages,
            partial_assistant_text=self.request.partial_answer,
        )

        started_at = perf_counter()
        generated = GreedyDecoder(
            runtime=self.runtime,
            input_ids=input_ids,
            request=self.request,
            policy=self.policy,
        ).decode()
        runtime_sec = perf_counter() - started_at
        ae_telemetry = replace(
            generated.ae_telemetry or RuntimeTelemetrySnapshot.empty(),
            runtime_sec=runtime_sec,
        )
        return GenerationResult(
            text=generated.text,
            full_ids=generated.full_ids,
            prompt_ids=input_ids,
            ae_telemetry=ae_telemetry,
            runtime_sec=runtime_sec,
        )


__all__ = ["StreamSession"]
