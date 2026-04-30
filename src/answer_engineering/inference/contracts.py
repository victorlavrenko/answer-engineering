"""Inference request, policy, and result value contracts.

Purpose:
    Define the immutable caller-facing payloads used by the public generation
    entrypoints.

Architectural role:
    Public contract module inside the inference API layer.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import torch

from answer_engineering.config.inference_defaults import GenerationDefaults
from answer_engineering.inference.model_types import Telemetry
from answer_engineering.rules.compile.compiled_rules import (
    CompiledRules,
)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """User-facing request for one generation call.

    A request contains the visible task text and, optionally, an already visible
    answer prefix that generation should continue from. Prompt templates, rules,
    model state, tokenizer state, and decoding settings live in separate public
    objects so notebooks can reuse the same request across baseline and
    rule-enabled runs.

    .. note::
        Keep this object small. It should describe caller intent, not the
        runtime machinery needed to execute that intent.

    Examples:
        ```python
        request = GenerationRequest(question=task.question)
        result = runtime.generate(
            request,
            GenerationPolicy(
                max_new_tokens=512,
                rules=subrun.compiled_rules,
            ),
        )
        print(result.text)
        ```

    Attributes:
        question: User-facing question or task text answered by the model.
        partial_answer: Optional visible answer prefix. When provided,
            generation continues from this prefix instead of starting from an
            empty answer.

    Runtime behavior:
        The runtime uses ``question`` to build prompt context and treats
        ``partial_answer`` as already-visible answer text. Validation happens at
        construction time so invalid requests fail before model loading or
        generation.

    Architectural role:
        Public request boundary between notebook/application code and
        :meth:`~answer_engineering.GenerationRuntime.generate`.

    Consumes:
        Dataset rows, notebook task objects, or direct user input that can be
        represented as one question and an optional partial answer.

    Produces:
        Caller intent consumed by
        :meth:`~answer_engineering.GenerationRuntime.generate` together with
        :class:`~answer_engineering.GenerationPolicy` and optional
        :class:`~answer_engineering.CompiledRules`.

    Invariants:
        ``question`` must be non-empty. The object must not contain tokenizer
        ids, prompt fragments, KV-cache state, compiled rules, or hidden
        planning data.

    Developer Notes:
        Preserve this class as the minimal public input contract. Runtime-only
        details should be introduced behind the generation boundary, not by
        adding implicit structure to ``question``.

    Todo:
        If richer chat inputs become public, add an explicit request type or
        field rather than overloading ``question`` with serialized hidden
        structure.

    See Also:
        :class:`~answer_engineering.GenerationPolicy`
        :class:`~answer_engineering.GenerationResult`
        :meth:`answer_engineering.GenerationRuntime.generate`

    """

    question: str
    partial_answer: str = ""

    def __post_init__(self) -> None:
        """Validate the request after dataclass construction.

        What happens internally:
            The hook enforces the basic invariants needed before a request
            reaches the runtime. A request should contain usable question text
            and should not require later code to guess whether it is valid.

        User effect:
            Invalid requests fail early, before model generation or rule
            intervention begins. This makes notebook errors easier to diagnose
            because the failing input is still close to the cell that
            constructed it.

        Developer notes:
            Keep this hook focused on request validity. Prompt assembly,
            tokenization, and runtime cache state belong in GenerationRuntime.

        """
        if not self.question:
            raise ValueError("request.question must be a non-empty string")


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    """Per-call generation settings for baseline or rule-enabled runs.

    A policy controls the public decoding budget, system prompt, stopping
    behavior, verbosity, and optional rules used for one generation call. Use
    the same policy when comparing baseline and rule-enabled subruns so
    differences are attributable to rules rather than experiment setup drift.

    .. note::
        Rule-enabled generation may perform extra probing and cache rebuild work
        even when the public token budget is unchanged. Keep policy values
        explicit in paper reproduction notebooks.

    Examples:
        ```python
        policy = GenerationPolicy(
            max_new_tokens=512,
            stop_on_eos=True,
            verbosity=1,
            rules=subrun.compiled_rules,
        )
        result = runtime.generate(
            GenerationRequest(question=task.question),
            policy,
        )
        ```

    Attributes:
        default_system_prompt: Class-level prompt used when ``system_prompt`` is
            not supplied.
        rules: Optional authored Markdown rules or a precompiled
            :class:`~answer_engineering.CompiledRules` object.
        system_prompt: System prompt used for this generation call.
        max_new_tokens: Maximum number of new answer tokens to generate.
        stop_on_eos: Whether generation stops when the model emits EOS.
        verbosity: Notebook/debug verbosity. Higher values expose more streaming
            and telemetry diagnostics.

    Methods:
        :meth:`~answer_engineering.GenerationPolicy.compiled_rules`
            Return the canonical compiled rules object for this policy, or
            ``None`` when no rules were supplied.

        :meth:`~answer_engineering.GenerationPolicy.stream_output`
            Return whether user-visible token streaming should be enabled for
            the configured verbosity.

        :meth:`~answer_engineering.GenerationPolicy.debug_output`
            Return whether debug and diagnostic output should be enabled for the
            configured verbosity.

    Runtime behavior:
        The runtime reads the policy at generation start and uses it to
        configure prompt text, stopping behavior, user-visible output, debug
        output, and the compiled rules available to orchestration.

    Architectural role:
        Public configuration boundary. It is deliberately separate from
        :class:`~answer_engineering.GenerationRequest` and runtime state.

    Consumes:
        Authored rules text, compiled rules, user-selected prompt text, and
        token budget choices from notebooks or applications.

    Produces:
        Stable execution settings consumed by
        :meth:`~answer_engineering.GenerationRuntime.generate`.

    Invariants:
        This is configuration, not execution state. Model instances, token
        caches, planning queues, probing state, scoring state, and telemetry
        capture belong behind the runtime boundary.

    Developer Notes:
        **`_compiled_rules`** Cached compiled representation of ``rules``. It
        exists so callers may pass authored text without manually invoking the
        compiler, but public code should normally treat it as an implementation
        detail.

        Add policy fields only when they are true experiment knobs. Avoid adding
        implementation conveniences that make reproduction results harder to
        compare across notebooks.

    Todo:
        Expose additional deterministic-decoding controls only after their
        behavior is stable enough for reproduction and SDK use. Keep backward
        compatibility explicitly non-guaranteed until the public surface is
        declared stable.

    See Also:
        :class:`~answer_engineering.GenerationRequest`
        :class:`~answer_engineering.GenerationResult`
        :class:`~answer_engineering.CompiledRules`
        :meth:`~answer_engineering.GenerationRuntime.generate`

    """

    default_system_prompt: ClassVar[str] = (
        "You are an experienced physician. "
        "Interpret the key findings before drawing conclusions. "
        "Synthesize them into a clinical assessment, then state "
        "the most appropriate management decision. "
        "Always give a concrete management action that can be "
        "started now when the case allows. "
        "Do not make referral the primary recommendation. "
        "Prioritize time-sensitive conditions when supported by the findings. "
        "Use only what is explicitly stated and keep reasoning concise."
    )

    rules: str | CompiledRules | None = None
    system_prompt: str = default_system_prompt
    max_new_tokens: int = GenerationDefaults().max_new_tokens
    stop_on_eos: bool = True
    verbosity: int = 0
    _compiled_rules: CompiledRules | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate generation settings after construction.

        What happens internally:
            The hook checks that policy values are usable before they reach the
            runtime. For example, token limits should be sensible and stopping
            behavior should be explicit enough that generation does not proceed
            with ambiguous settings.

        User effect:
            Policy mistakes fail close to the notebook cell that created the
            policy, rather than after a model has been loaded or several tasks
            have already run.

        Reproduction guidance:
            Treat a policy validation failure as an experiment setup error. Fix
            the policy and rerun the affected subruns so baseline and
            rule-enabled results remain comparable.

        Developer notes:
            Keep validation here limited to stable public policy invariants.
            Internal model or tokenizer compatibility checks belong in
            GenerationRuntime.

        """
        object.__setattr__(self, "_compiled_rules", _compile_rules(self.rules))

    @property
    def compiled_rules(self) -> CompiledRules | None:
        """Return the canonical compiled rules for this generation policy.

        Purpose:
            Expose the post-initialization rule representation that runtime
            stages use instead of reparsing raw markdown during generation.

        Architectural role:
            Public policy accessor at the boundary between notebook/user input
            and the compiled rule system consumed by orchestration.

        Inputs (architectural provenance):
            Reads `_compiled_rules`, which `GenerationPolicy.__post_init__`
            derives from the caller-supplied `rules` value.

        Outputs (downstream usage):
            Returns `CompiledRules` for decode/orchestration code, or `None`
            when the generation run intentionally has no answer-engineering
            rules.

        Invariants/constraints:
            Callers should treat this as the single source of truth for policy
            rules after construction; downstream runtime stages should not
            branch on raw rule text or recompile it.

        """
        return self._compiled_rules

    @property
    def stream_output(self) -> bool:
        """Return whether visible streaming output should be emitted.

        Purpose:
            Convert the public verbosity integer into the boolean stream- output
            policy used by the runtime session.

        Architectural role:
            Presentation-policy accessor on the public generation policy. It
            separates notebook-facing verbosity configuration from lower- level
            printer selection.

        Inputs (architectural provenance):
            Reads `verbosity`, supplied by the notebook, API caller, or
            reproduction configuration that builds the policy.

        Outputs (downstream usage):
            Returns a boolean consumed by generation/session code when deciding
            whether assistant-visible text should be streamed.

        Invariants/constraints:
            Verbosity level zero suppresses streaming; positive levels permit
            visible streaming without implying debug output.

        """
        return self.verbosity >= 1

    @property
    def debug_output(self) -> bool:
        """Return whether verbose runtime debug output should be emitted.

        Purpose:
            Convert the public verbosity integer into the boolean debug- output
            policy used by instrumentation and printer/debug emitters.

        Architectural role:
            Presentation-policy accessor on the public generation policy. It
            keeps debug enablement separate from ordinary streaming output.

        Inputs (architectural provenance):
            Reads `verbosity`, supplied by the notebook, API caller, or
            reproduction configuration that builds the policy.

        Outputs (downstream usage):
            Returns a boolean consumed by runtime stages that decide whether to
            emit detailed answer-engineering diagnostics.

        Invariants/constraints:
            Debug output is enabled only at verbosity level two or higher so
            ordinary streaming and debug traces remain separate policy states.

        """
        return self.verbosity >= 2


def _compile_rules(rules: str | CompiledRules | None) -> CompiledRules | None:
    if rules is None:
        return None
    if isinstance(rules, CompiledRules):
        return rules
    return CompiledRules(rules)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Result returned by one model generation call.

    A result carries the final answer text together with runtime metadata needed
    for evaluation, reporting, debugging, and paper reproduction. Generation has
    already finished when this object is returned; it is not a mutable runtime
    session.

    .. note::
        In reproduction notebooks, pass the whole result into
        :class:`~ae_paper_reproduction.RulesetEvaluationResult` so answer text,
        telemetry, token ids, and runtime timing stay aligned.

    Examples:
        ```python
        result = runtime.generate(
            request,
            policy,
        )

        evaluation = RulesetEvaluationResult(task.row, answer=result)
        print(result.text)
        ```

    Attributes:
        text: Final answer text after baseline generation or rule-enabled
            intervention.
        ae_telemetry: Optional runtime telemetry snapshot describing captured
            Answer Engineering events and decisions.
        full_ids: Optional token ids for the full prompt-plus-answer sequence.
        prompt_ids: Optional token ids for the prompt prefix used by the
            runtime.
        runtime_sec: Wall-clock runtime in seconds for this generation call.

    Runtime behavior:
        Baseline and rule-enabled runs both return the same public result shape.
        Rule-enabled runs may include richer telemetry describing triggers,
        proposals, deterministic edits, rejected candidates, and skipped
        patches.

    Architectural role:
        Public output contract of
        :meth:`~answer_engineering.GenerationRuntime.generate`.

    Consumes:
        Completed runtime session output, decoded text, token ids, telemetry,
        and timing data.

    Produces:
        Evaluation input for
        :class:`~ae_paper_reproduction.RulesetEvaluationResult` and analysis
        input for notebook reports and paper metrics.

    Invariants:
        Generated text and telemetry stay together at this boundary so
        downstream code does not need private runtime/session objects.

    Developer Notes:
        Serialization, reports, and generated paper metrics should consume these
        fields instead of ad hoc runtime payloads. Keep telemetry access typed
        where possible; avoid making external analyses depend on internal event
        dicts.

    Todo:
        Improve typed telemetry exposure so custom analyses can avoid
        payload-shape assumptions while still supporting detailed
        failure-cluster inspection.

    See Also:
        :class:`~answer_engineering.GenerationRequest`
        :class:`~answer_engineering.GenerationPolicy`
        :class:`~answer_engineering.telemetry.RuntimeTelemetrySnapshot`
        :class:`~ae_paper_reproduction.RulesetEvaluationResult`

    """

    text: str
    ae_telemetry: Telemetry
    full_ids: torch.Tensor | None
    prompt_ids: torch.Tensor | None
    runtime_sec: float


__all__ = [
    "GenerationPolicy",
    "GenerationRequest",
    "GenerationResult",
]
