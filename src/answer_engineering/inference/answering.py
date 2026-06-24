"""Model-backed inference runtime for Answer Engineering generation.

Purpose:
    Materialize tokenizer/model resources, expose forward and token-generation
    operations, and run the high-level streaming session used by the public
    generate API.

Architectural role:
    Primary runtime implementation behind GenerationRuntime.

Architectural direction:
    Keep public runtime behavior stable while reducing leakage of
    backend-specific concerns across inference helpers.

Why this matters:
    Runtime loading/materialization and generation services are intentionally
    concentrated here today, so boundary clarity is important for extension
    cost.

What better would look like:
    Public runtime contracts stay predictable while backend-specific mechanics
    remain contained in inference-owned seams.

How improvement can be recognized:
    - Stable public runtime interfaces despite backend evolution
    - Fewer backend-specific assumptions crossing helper boundaries
    - Lower cross-module edits for backend integration changes

Open constraint:
    Runtime shape should evolve with supported inference backends and
    operational requirements.

"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from huggingface_hub.utils.tqdm import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )
except ImportError:
    are_progress_bars_disabled = None
    disable_progress_bars = None
    enable_progress_bars = None

from answer_engineering.inference.contracts import (
    GenerationPolicy,
    GenerationRequest,
    GenerationResult,
)
from answer_engineering.inference.model_types import (
    CausalLMBackend,
    CausalLMForwardOutput,
    CausalLMGenerationOutput,
    ChatGenerationRuntime,
    ChatTextCodec,
    DTypeName,
    GenerationCall,
    PastKeyValues,
)
from answer_engineering.inference.stream_io.api import StreamSession


class _PadTokenCodec(ChatTextCodec, Protocol):
    """Tokenizer protocol view that permits assigning a pad token.

    Purpose:
        Narrow the tokenizer type enough for GenerationRuntime.materialize to
        backfill pad_token from eos_token when needed.

    Architectural role:
        Internal typing helper inside the runtime-loading module.

    """

    eos_token: str | None
    pad_token: str | None


class _TokenizerLoader(Protocol):
    def __call__(
        self,
        pretrained_model_name_or_path: str,
        *,
        revision: str | None = ...,
        use_fast: bool = ...,
        trust_remote_code: bool = ...,
    ) -> ChatTextCodec: ...


class _ModelLoader(Protocol):
    def __call__(
        self,
        pretrained_model_name_or_path: str,
        *,
        revision: str | None = ...,
        device_map: str | dict[str, int | str] | None = ...,
        dtype: torch.dtype | str = ...,
        low_cpu_mem_usage: bool = ...,
        trust_remote_code: bool = ...,
    ) -> CausalLMBackend: ...


@dataclass(slots=True)
class GenerationRuntime(ChatGenerationRuntime):
    """Model-backed runtime that generates answers with rule intervention.

    Create and own the model/tokenizer lifecycle for baseline and rule-enabled
    Answer Engineering experiments. After
    :meth:`~answer_engineering.GenerationRuntime.materialize` loads resources,
    call :meth:`~answer_engineering.GenerationRuntime.generate` with a request
    and policy.

    .. note::
        Model loading is expensive and should usually happen once per notebook
        session. Generation cost depends on model size, device, token budget,
        and probing caused by rules. Compare rule-enabled runs against a
        baseline on the same rows and policy.

    Examples:
        ```python
        runtime = GenerationRuntime("Qwen/Qwen2.5-7B-Instruct")
        runtime.materialize()

        request = GenerationRequest(question=task.question)
        result = runtime.generate(
            request,
            policy=GenerationPolicy(max_new_tokens=512),
            rules=subrun.compiled_rules,
        )
        print(result.text)
        ```

    Attributes:
        model_id: Hugging Face model identifier or local model path.
        revision: Optional model revision requested from the loader.
        device_map: Device placement configuration passed to model loading.
        dtype: Data type requested for model weights.
        trust_remote_code: Whether model loading may execute remote model code.
        show_hf_hub_progress_bars_on_load: Whether progress bars are shown while
            loading.
        show_hf_hub_progress_bars_on_generate: Whether progress bars are shown
            during generation.
        resolved_dtype: Final dtype after loader normalization.

    Methods:
        :meth:`~answer_engineering.GenerationRuntime.materialize`
            Load model and tokenizer resources into memory.

        :meth:`~answer_engineering.GenerationRuntime.execution_device`
            Return the resolved torch device used for tensor construction and
            inference.

        :meth:`~answer_engineering.GenerationRuntime.text_codec`
            Return the materialized tokenizer/text-codec boundary used by prompt
            encoding and decoding.

        :meth:`~answer_engineering.GenerationRuntime.ensure_eval_mode`
            Ensure the materialized model is in inference/evaluation mode.

        :meth:`~answer_engineering.GenerationRuntime.forward`
            Execute one backend forward pass for runtime, scoring, and probing
            internals.

        :meth:`~answer_engineering.GenerationRuntime.generate_tokens`
            Generate raw token ids from a prompt using the materialized backend.

        :meth:`~answer_engineering.GenerationRuntime.generate`
            Execute model-backed generation with optional rule intervention.

    Runtime behavior:
        In a baseline run, the runtime performs ordinary model-backed decoding.
        In a rule-enabled run, compiled rules may inspect candidate text, probe
        alternatives, apply deterministic edits, rebuild model context after
        edits, and emit telemetry describing generation decisions.

    Architectural role:
        Public runtime boundary. Orchestration, probing, scoring, selection,
        patching, and telemetry internals stay behind
        :meth:`~answer_engineering.GenerationRuntime.generate`.

    Consumes:
        :class:`~answer_engineering.GenerationRequest`
            User-facing request describing the question or task.

        :class:`~answer_engineering.GenerationPolicy`
            Execution policy controlling token budget, stopping rules, prompt
            text, and verbosity.

        :class:`~answer_engineering.CompiledRules` (optional)
            Rule bundle enabling deterministic interventions.

    Produces:
        :class:`~answer_engineering.GenerationResult`
            Result object containing answer text, telemetry, token ids, and
            runtime timing metadata.

    Invariants:
        Runtime resources must be materialized before execution. Rule-enabled
        generation may apply deterministic edits and rebuild model context while
        preserving the public request/policy contract.

    Developer Notes:
        Internal runtime state is intentionally encapsulated behind the public
        :meth:`~answer_engineering.GenerationRuntime.generate` interface.
        Low-level helpers such as
        :meth:`~answer_engineering.GenerationRuntime.forward` and
        :meth:`~answer_engineering.GenerationRuntime.generate_tokens` are public
        because notebook and reproduction infrastructure may need a stable
        backend seam; rename them private if they should not remain part of the
        user-facing API.

        :attr:`_model`
            Materialized model instance owned by the runtime after loading.

        :attr:`_tokenizer`
            Materialized tokenizer instance owned by the runtime after loading.

        :attr:`_device`
            Resolved execution device used by tensor construction, scoring, and
            probing.

        :attr:`_materialize_lock`
            Per-runtime synchronization primitive guarding the materialization
            boundary. Ensures that tokenizer and model resources are loaded at
            most once when multiple callers attempt to access the runtime
            concurrently.

        Telemetry snapshots are useful for paper metrics and private analysis:
        rule triggers, edits, case-type changes, and failure clusters should be
        examined without exposing session internals to notebook code.

    Todo:
        Preserve the simple runtime API while improving internal repair
        semantics, cache reuse, backend seams, and telemetry typing. Backward
        compatibility is not guaranteed while this architecture is still
        converging.

    See Also:
        :class:`~answer_engineering.GenerationRequest`
        :class:`~answer_engineering.GenerationPolicy`
        :class:`~answer_engineering.GenerationResult`
        :class:`~answer_engineering.CompiledRules`

    """

    model_id: str
    revision: str | None = None
    device_map: str | dict[str, int | str] | None = "auto"
    dtype: DTypeName = "auto"
    trust_remote_code: bool = False
    show_hf_hub_progress_bars_on_load: bool = True
    show_hf_hub_progress_bars_on_generate: bool = False
    _model: CausalLMBackend | None = field(default=None, init=False, repr=False)
    _tokenizer: ChatTextCodec | None = field(
        default=None, init=False, repr=False
    )
    _device: torch.device | None = field(default=None, init=False, repr=False)
    resolved_dtype: torch.dtype | None = field(
        default=None, init=False, repr=False
    )
    _materialize_lock: Lock = field(
        default_factory=Lock,
        init=False,
        repr=False,
    )

    def materialize(self) -> GenerationRuntime:
        """Load tokenizer, model, dtype, and execution-device state.

        Materialize Hugging Face resources once and cache the resolved runtime
        state on the runtime instance. The method returns ``self`` so notebooks
        can write explicit setup cells before running baseline and rule-enabled
        subruns.

        .. note::
            Call this once per runtime instance before generation. Reusing one
            materialized runtime avoids repeated model downloads and keeps
            baseline and rule-enabled comparisons on the same backend
            configuration.

        Examples:
            ```python
            runtime = GenerationRuntime(
                "Qwen/Qwen2.5-7B-Instruct",
                dtype="auto",
                device_map="auto",
            ).materialize()
            ```

        Returns:
            The same :class:`~answer_engineering.GenerationRuntime` instance
            after the tokenizer, model, resolved dtype, and execution device
            have been installed.

        Raises:
            RuntimeError: May be raised by downstream model/tokenizer loading
                code when resources cannot be loaded, trusted, allocated, or
                validated.

        Runtime behavior:
            Loading is lazy and idempotent. If resources already exist, the
            method leaves them in place. When the tokenizer lacks a pad token,
            the runtime backfills it from the EOS token, then loads the model
            and puts it into evaluation mode.

        Architectural role:
            Resource-materialization boundary for the public inference runtime.

        Consumes:
            Constructor configuration such as ``model_id``, ``revision``,
            ``device_map``, ``dtype``, ``trust_remote_code``, and loading
            progress-bar policy.

        Produces:
            Materialized tokenizer, model, resolved dtype, and execution device
            used by prompt encoding, forward passes, token generation, probing,
            and scoring.

        Invariants:
            After successful materialization, runtime accessors can return the
            model, tokenizer, and device. The backend should remain in
            evaluation mode for inference.

        Developer Notes:
            Keep loader-specific behavior inside this boundary. Future backend
            support should preserve the public materialize/generate lifecycle
            while moving backend-specific mechanics into narrower seams.

        Todo:
            Make backend integration cleaner without changing the
            notebook-facing setup pattern unless a deliberate public API
            revision is made.

        See Also:
            :meth:`~answer_engineering.GenerationRuntime.generate`
            :class:`~answer_engineering.GenerationRuntime`

        """
        if (
            self._model is not None
            and self._tokenizer is not None
            and self._device is not None
        ):
            return self

        with self._materialize_lock:
            if (
                self._model is not None
                and self._tokenizer is not None
                and self._device is not None
            ):
                return self

            progress_scope = (
                nullcontext()
                if self.show_hf_hub_progress_bars_on_load
                else _suspend_hf_hub_progress_bars()
            )
            with progress_scope:
                dtype = _resolve_dtype_name(self.dtype)
                load_tokenizer = cast(
                    _TokenizerLoader,
                    AutoTokenizer.from_pretrained,
                )
                tokenizer = load_tokenizer(
                    self.model_id,
                    revision=self.revision,
                    use_fast=True,
                    trust_remote_code=self.trust_remote_code,
                )
                if tokenizer.pad_token_id is None:
                    mutable_tokenizer = cast(_PadTokenCodec, tokenizer)
                    mutable_tokenizer.pad_token = mutable_tokenizer.eos_token

                load_model = cast(
                    _ModelLoader,
                    AutoModelForCausalLM.from_pretrained,
                )
                model = load_model(
                    self.model_id,
                    revision=self.revision,
                    device_map=self.device_map,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=self.trust_remote_code,
                )
                model.eval()

                self._model = model
                self._tokenizer = tokenizer
                self.resolved_dtype = dtype
                self._device = next(model.parameters()).device

        return self

    def execution_device(self) -> torch.device:
        """Return the materialized model execution device.

        Purpose:
            Expose the concrete torch device selected during runtime
            materialization.

        Architectural role:
            Cheap runtime-state accessor on the public generation runtime. It
            lets probing, scoring, and tensor-construction code allocate inputs
            on the same device as the loaded model without reaching into backend
            internals.

        Inputs (architectural provenance):
            Reads the `_device` value established by
            `GenerationRuntime.materialize` from the configured model, device
            map, and backend parameters.

        Outputs (downstream usage):
            Returns the resolved `torch.device` consumed by inference helpers
            that build tensors or compare backend placement.

        Invariants/constraints:
            The runtime must already be materialized. Access before
            materialization is a lifecycle error because no backend device has
            been resolved yet.

        """
        self.materialize()
        if self._device is None:
            raise RuntimeError("GenerationRuntime is not materialized.")
        return self._device

    def text_codec(self) -> ChatTextCodec:
        """Return the materialized tokenizer/chat text codec.

        Purpose:
            Expose the tokenizer capability object loaded for this generation
            runtime.

        Architectural role:
            Runtime-state accessor used by prompt construction, decode, probing,
            scoring, and notebook-facing code that needs the same tokenization
            semantics as the loaded model.

        Inputs (architectural provenance):
            Reads the tokenizer installed by `GenerationRuntime.materialize`
            from the configured model id, revision, and remote-code policy.

        Outputs (downstream usage):
            Returns the `ChatTextCodec` used by downstream callers for encoding,
            decoding, chat-template rendering, and offset-aware tokenizer
            operations.

        Invariants/constraints:
            The runtime must already be materialized. The returned codec must be
            the same object used by runtime generation so token ids and decoded
            text remain consistent across the pipeline.

        """
        self.materialize()
        if self._tokenizer is None:
            raise RuntimeError("GenerationRuntime is not materialized.")
        return self._tokenizer

    def ensure_eval_mode(self) -> None:
        """Ensure the materialized backend is in inference mode.

        Purpose:
            Reassert evaluation mode before generation or model-backed scoring
            uses the loaded causal language model.

        Architectural role:
            Runtime lifecycle guard between resource materialization and decode
            stages.

        Inputs (architectural provenance):
            Reads the model previously installed by `materialize`.

        Outputs (downstream usage):
            Mutates backend mode in place so later `forward` and
            `generate_tokens` calls run without training-time behavior.

        Invariants/constraints:
            Requires materialization first and raises when runtime resources are
            absent.

        """
        self.materialize()
        if self._model is None:
            raise RuntimeError("GenerationRuntime is not materialized.")
        self._model.eval()

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool | None = None,
    ) -> CausalLMForwardOutput:
        """Run one cached or uncached forward pass on the backend model.

        Purpose:
            Provide the narrow token-level inference operation needed by greedy
            decode, cache rebuild, scoring, and probing code.

        Architectural role:
            Model execution boundary implementing the `ChatGenerationRuntime`
            protocol.

        Inputs (architectural provenance):
            Receives token ids, optional attention mask, optional past key
            values, and cache policy from decode or scoring callers.

        Outputs (downstream usage):
            Returns the backend forward output consumed for logits and cache
            state.

        Invariants/constraints:
            Requires a materialized model. The method forwards tensors without
            changing prompt semantics or applying rule logic.

        """
        self.materialize()
        if self._model is None:
            raise RuntimeError("GenerationRuntime is not materialized.")
        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def generate_tokens(
        self, request: GenerationCall
    ) -> CausalLMGenerationOutput:
        """Invoke backend beam/sample generation from a structured request.

        Purpose:
            Adapt the project's typed `GenerationCall` request into the
            keyword-heavy Hugging Face `generate` API.

        Architectural role:
            Token-generation boundary used by probing and any runtime path that
            needs model-produced continuations rather than a single forward
            pass.

        Inputs (architectural provenance):
            Receives prompt tensors, attention masks, pad token id, generation
            controls, and optional custom-generate configuration from
            probing/runtime callers.

        Outputs (downstream usage):
            Returns the backend generation output consumed by candidate
            construction or continuation analysis.

        Invariants/constraints:
            Requires a materialized model. The method only adapts parameters; it
            does not score, select, patch, or interpret generated text.

        """
        self.materialize()
        if self._model is None:
            raise RuntimeError("GenerationRuntime is not materialized.")
        custom_generate = None
        trust_remote_code = False
        revision = None
        use_cache = None
        if request.custom_generate_config is not None:
            custom_generate = request.custom_generate_config.custom_generate
            trust_remote_code = request.custom_generate_config.trust_remote_code
            revision = request.custom_generate_config.revision
            use_cache = request.custom_generate_config.use_cache

        generate_fn = self._model.generate
        return generate_fn(
            input_ids=request.input_ids,
            attention_mask=request.attention_mask,
            pad_token_id=request.pad_token_id,
            max_new_tokens=request.control.max_new_tokens,
            num_beams=request.control.num_beams,
            num_beam_groups=request.control.num_beam_groups,
            diversity_penalty=request.control.diversity_penalty,
            num_return_sequences=request.control.num_return_sequences,
            do_sample=request.control.do_sample,
            temperature=request.control.temperature,
            top_p=request.control.top_p,
            top_k=request.control.top_k,
            early_stopping=request.control.early_stopping,
            return_dict_in_generate=request.control.return_dict_in_generate,
            output_scores=request.control.output_scores,
            custom_generate=custom_generate,
            trust_remote_code=trust_remote_code,
            revision=revision,
            use_cache=use_cache,
        )

    def generate(
        self, request: GenerationRequest, policy: GenerationPolicy
    ) -> GenerationResult:
        """Generate one answer from a request, policy, and optional rules.

        Run a baseline or rule-enabled generation session and return the final
        answer with telemetry and token metadata. This is the main public
        execution entrypoint used by notebooks and applications.

        .. note::
            Use the same request rows and policy values when comparing baseline
            and rule-enabled runs. Otherwise accuracy differences can reflect
            experiment setup drift instead of rule behavior.

        Examples:
            ```python
            result = runtime.generate(
                GenerationRequest(question=task.question),
                policy=GenerationPolicy(max_new_tokens=512),
                rules=subrun.compiled_rules,
            )
            evaluation = RulesetEvaluationResult(task.row, answer=result)
            ```

        Args:
            request: User-facing task input to answer.
            policy: Generation policy controlling prompt text, token budget,
                stopping behavior, and verbosity. When omitted, the default
                policy is used.

        Returns:
            GenerationResult:
                :class:`~answer_engineering.GenerationResult` is a structured
                execution outcome for this generation call. Provides the final
                answer text together with runtime telemetry, token identifiers,
                and wall-clock timing collected during the session. The result
                object is the stable public artifact consumed by evaluation
                loops, reporting pipelines, and downstream analysis.

        Raises:
            RuntimeError: Raised when generation is attempted before runtime
                resources are materialized or when backend execution fails.
            ValueError: Raised when request or policy validation rejects input
                before generation starts.

        Runtime behavior:
            Baseline generation decodes normally. Rule-enabled generation routes
            through the Answer Engineering stream session, where rules may
            trigger proposals, probes, deterministic edits, cache rebuilds, and
            telemetry events.

        Architectural role:
            Public execution boundary. The caller sees request, policy, optional
            rules, and result; orchestration internals stay behind this method.

        Consumes:
            :class:`~answer_engineering.GenerationRequest`
            :class:`~answer_engineering.GenerationPolicy`
                :class:`~answer_engineering.CompiledRules` or authored Markdown
                rules

        Produces:
            :class:`~answer_engineering.GenerationResult`

        Invariants:
            The runtime must be materialized. The public result shape should
            stay stable across baseline and rule-enabled runs even as repair
            internals evolve.

        Developer Notes:
            Keep this method small as a public adapter into orchestration.
            Detailed proposal, probing, scoring, selection, patching, and
            telemetry behavior belongs in dedicated internals and typed event
            surfaces.

        Todo:
            Continue separating orchestration control flow from telemetry
            formatting and backend mechanics. Improve branch-aware repair
            without expanding the public call signature prematurely.

        See Also:
            :meth:`~answer_engineering.GenerationRuntime.materialize`
            :class:`~answer_engineering.GenerationRequest`
            :class:`~answer_engineering.GenerationPolicy`
            :class:`~answer_engineering.GenerationResult`

        """
        self.materialize()
        self.ensure_eval_mode()
        progress_scope = (
            nullcontext()
            if self.show_hf_hub_progress_bars_on_generate
            else _suspend_hf_hub_progress_bars()
        )
        with progress_scope:
            return StreamSession(
                runtime=self, request=request, policy=policy
            ).run()


def _resolve_dtype_name(dtype: DTypeName) -> torch.dtype:
    """Resolve a configured dtype name into a concrete torch dtype.

    Purpose:
        Translate user- or policy-facing dtype strings into the exact
        `torch.dtype` object required during model materialization.

    Architectural role:
        Inference-configuration adapter inside the model-backed runtime loader.
        It keeps string parsing outside the constructor path that builds
        tokenizer, model, and execution-device state.

    Inputs (architectural provenance):
        Receives the dtype name supplied by `GenerationRuntimeSpec` or another
        runtime construction path.

    Outputs (downstream usage):
        Returns the dtype passed to Hugging Face model loading.

    Invariants/constraints:
        Only supported dtype names should resolve. Invalid names must fail
        before a partially configured runtime can be materialized.

    """
    if dtype == "auto":
        return (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


@contextmanager
def _suspend_hf_hub_progress_bars() -> Generator[None]:
    """Temporarily suspend Hugging Face Hub progress bars around a load.

    Purpose:
        Silence progress-bar side effects while a tokenizer, model, or remote
        component is being resolved.

    Architectural role:
        Infrastructure helper at the inference-loading boundary. It isolates UI
        suppression from runtime materialization so loading code can remain
        focused on model state.

    Inputs (architectural provenance):
        Reads the current Hugging Face Hub progress-bar state before entering
        the wrapped load operation.

    Outputs (downstream usage):
        Yields control to the caller with progress bars disabled, then restores
        the previous state when the block exits.

    Invariants/constraints:
        Restoration must run even when loading raises, so temporary UI policy
        does not leak into unrelated notebook or CLI code.

    """
    if (
        are_progress_bars_disabled is None
        or disable_progress_bars is None
        or enable_progress_bars is None
    ):
        yield
        return

    were_disabled = are_progress_bars_disabled()
    if not were_disabled:
        disable_progress_bars()

    try:
        yield
    finally:
        if not were_disabled:
            enable_progress_bars()


__all__ = [
    "GenerationPolicy",
    "GenerationRequest",
    "GenerationResult",
    "GenerationRuntime",
]
