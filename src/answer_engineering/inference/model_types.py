"""Typed collaboration contracts for tokenizer, model, and runtime roles.

Purpose:
    Define the protocol and value types that let inference, decode, prompting,
    probing, and scoring collaborate without depending directly on Hugging Face
    concrete classes.

Architectural role:
    Shared type boundary inside inference.

Current architecture notes:
    These protocols are behavior-shaped enough to be useful, but some remain
    close to Hugging Face capability slices rather than fully domain-specific
    roles.

"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

import torch

from answer_engineering.engine.telemetry.snapshots.snapshots import (
    RuntimeTelemetrySnapshot,
)

type DTypeName = Literal["auto", "float16", "bfloat16", "float32"]

type PastKeyValues = tuple[object, ...] | object
type TokenIds = Sequence[int] | torch.Tensor
type ChatMessage = Mapping[str, str]

type ChatTemplateResult = torch.Tensor | list[int]
type Telemetry = RuntimeTelemetrySnapshot


class OffsetEncoding(TypedDict):
    """Tokenizer return shape exposing both token ids and character offsets.

    Purpose:
        Represent the offset-mapping form used when probing or alignment-aware
        helpers need both ids and source-text spans.

    Architectural role:
        TypedDict value contract in the inference type layer.

    """

    input_ids: list[int]
    offset_mapping: list[tuple[int, int]]


class TextCodec(Protocol):
    """Tokenizer capability protocol for plain text encoding and decoding.

    Purpose:
        Define the minimal tokenizer operations needed by prompting, decode,
        probing, and scoring helpers.

    Architectural role:
        Shared protocol in the inference type boundary.

    """

    @property
    def pad_token_id(self) -> int | None:
        """Return the tokenizer pad-token id, if defined."""
        raise NotImplementedError

    @property
    def eos_token_id(self) -> int | list[int] | None:
        """Return the tokenizer EOS token id or ids."""
        raise NotImplementedError

    @property
    def bos_token_id(self) -> int | None:
        """Return the tokenizer BOS token id, if defined."""
        raise NotImplementedError

    def encode(self, text: str, *, add_special_tokens: bool = ...) -> list[int]:
        """Encode plain text into token ids.

        Purpose:
            Define the tokenizer contract used by runtime code when text must be
            turned into model input ids.

        Architectural role:
            Protocol method at the inference-backend boundary. It lets runtime
            and probing code depend on tokenizer behavior without depending on a
            concrete Hugging Face tokenizer class.

        Inputs (architectural provenance):
            Receives plain text produced by prompt assembly, document
            reconstruction, or probing-prefix construction.

        Outputs (downstream usage):
            Returns token ids consumed by model forward passes, generation,
            scoring, and cache rebuilding.

        Invariants/constraints:
            Implementations should preserve tokenizer-native ids and honor
            caller options such as special-token handling consistently with
            `decode`.

        """
        raise NotImplementedError

    def decode(self, ids: list[int], *, skip_special_tokens: bool = ...) -> str:
        """Decode token ids back into text.

        Purpose:
            Define the tokenizer contract used when generated or scored token
            ids must be projected back into visible text.

        Architectural role:
            Protocol method at the inference-backend boundary between model
            tokens and runtime document text.

        Inputs (architectural provenance):
            Receives token ids produced by generation, probing, or scoring
            utilities.

        Outputs (downstream usage):
            Returns decoded text consumed by document assembly, proposal
            generation, console output, and evaluation records.

        Invariants/constraints:
            Decoding should be consistent with `encode` for the same tokenizer
            and must respect caller options such as special-token handling.

        """
        raise NotImplementedError

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = ...,
        return_offsets_mapping: bool = ...,
    ) -> OffsetEncoding:
        """Encode text and return tokenizer fields including optional offsets.

        Purpose:
            Capture the tokenizer call shape used by alignment-aware code that
            needs token ids and, when requested, character-offset mappings.

        Architectural role:
            Protocol member in the inference type boundary between tokenizer
            backends and probing/alignment helpers.

        Inputs (architectural provenance):
            Receives source text plus tokenizer flags from prompt, probing, or
            alignment callers.

        Outputs (downstream usage):
            Returns an `OffsetEncoding` mapping consumed by code that relates
            generated token ids back to source-text spans.

        Invariants/constraints:
            Implementations must preserve backend tokenizer semantics; this
            protocol only narrows the shape consumed by the project.

        """
        raise NotImplementedError


class ChatTemplateApplier(Protocol):
    """Tokenizer protocol for chat-template rendering.

    Purpose:
        Capture the capability of exposing a chat_template string and applying
        it to structured chat messages.

    Architectural role:
        Prompting-facing protocol in the inference type layer.

    """

    @property
    def chat_template(self) -> str | None:
        """Return the tokenizer chat template string, if available."""
        raise NotImplementedError

    def apply_chat_template(
        self,
        messages: Sequence[ChatMessage],
        *,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
    ) -> ChatTemplateResult:
        """Render chat messages through the tokenizer chat template.

        Purpose:
            Capture the tokenizer capability used to convert structured chat
            messages into either rendered text or token ids.

        Architectural role:
            Prompting-facing protocol member in the inference type boundary.

        Inputs (architectural provenance):
            Receives chat messages and rendering flags from prompt construction
            code.

        Outputs (downstream usage):
            Returns rendered template text or token ids consumed by request
            encoding and generation setup.

        Invariants/constraints:
            Implementations must use the backend tokenizer's chat-template
            semantics so notebook prompts and runtime prompts stay model-
            compatible.

        """
        raise NotImplementedError


class ChatTextCodec(TextCodec, ChatTemplateApplier, Protocol):
    """Combined tokenizer protocol for plain-text and chat-template operations.

    Purpose:
        Represent a tokenizer role that can both encode or decode raw text and
        render structured chat messages through a chat template.

    Architectural role:
        Shared prompting-and-runtime protocol inside the inference type
        boundary.

    Inputs (architectural provenance):
        Implemented by tokenizer objects supplied by runtime construction.

    Outputs (downstream usage):
        Consumed by prompting helpers, stream-session setup, and chat-aware
        runtimes.

    Invariants/constraints:
        Implementations must satisfy both the `TextCodec` and
        `ChatTemplateApplier` contracts on the same object.

    """


class CausalLMForwardOutput(Protocol):
    """Protocol view of one causal-LM forward pass result.

    Purpose:
        Expose logits and optional past_key_values to decode, probing, and
        scoring code without depending on a concrete model output class.

    Architectural role:
        Shared forward-output contract inside inference.

    """

    logits: torch.Tensor
    past_key_values: PastKeyValues | None


class CausalLMBackend(Protocol):
    """Protocol for the model backend used by GenerationRuntime.

    Purpose:
        Describe the eval, forward, generate, and parameter-iteration operations
        required from a causal language model backend.

    Architectural role:
        Model capability contract in the inference type boundary.

    """

    def eval(self) -> CausalLMBackend:
        """Switch the backend into evaluation mode and return the backend."""
        raise NotImplementedError

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: PastKeyValues | None = ...,
        use_cache: bool | None = ...,
    ) -> CausalLMForwardOutput:
        """Run one forward pass on the causal language-model backend.

        Purpose:
            Define the minimal callable surface required by generation and
            scoring code.

        Architectural role:
            Protocol method at the inference boundary between repository runtime
            logic and model implementations.

        Inputs (architectural provenance):
            Receives token tensors and model-forward keyword arguments from
            decoding or scoring paths.

        Outputs (downstream usage):
            Returns backend-specific model outputs consumed by logits, cache, or
            scoring logic.

        Invariants/constraints:
            Implementations should behave like an eval-mode causal LM forward
            pass and should not hide device or dtype errors from callers.

        """
        raise NotImplementedError

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> CausalLMGenerationOutput:
        """Run backend generation with prepared tensors and generation options.

        Purpose:
            Define the model-generation operation required by greedy decode and
            probing code after inputs have already been tokenized and moved to
            the execution device.

        Architectural role:
            Protocol method at the causal-language-model backend boundary. It
            hides the concrete model class while preserving the generation
            surface runtime code needs.

        Inputs (architectural provenance):
            Receives input tensors, attention masks, cache settings, and
            generation keyword arguments prepared by inference/session code.

        Outputs (downstream usage):
            Returns backend-native generation output consumed by
            token-generation and probing adapters.

        Invariants/constraints:
            Implementations should not perform prompt assembly or rule logic.
            Inputs are expected to be model-ready before this method is called.

        """
        raise NotImplementedError

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Iterate backend parameters for runtime execution-device inference."""
        raise NotImplementedError


class CausalLMGenerationOutput(Protocol):
    """Protocol view of a backend generation result.

    Purpose:
        Expose generated sequences and optional sequence scores in the shape
        expected by probing helpers.

    Architectural role:
        Generation-output contract inside inference.

    """

    sequences: torch.Tensor
    sequences_scores: torch.Tensor | None


class CustomGenerateConfig(Protocol):
    """Protocol for runtime configuration that points generate() at a custom.

    Architectural role:
        Configuration capability contract used by grouped-beam probing helpers.

    """

    @property
    def custom_generate(self) -> str:
        """Return the custom generation code location.

        Purpose:
            Expose the configured implementation source used when probing relies
            on a custom generation routine.

        Architectural role:
            Protocol property at the inference-configuration boundary between
            runtime setup and grouped-beam probing helpers.

        Inputs (architectural provenance):
            Reads configuration supplied by the caller or model-loading layer.

        Outputs (downstream usage):
            Returns the path or repository identifier passed to custom
            generation loading code.

        Invariants/constraints:
            The value should identify code location only. Trust and revision
            policy remain separate protocol fields.

        """
        raise NotImplementedError

    @property
    def trust_remote_code(self) -> bool:
        """Return whether custom generation may trust remote code.

        Purpose:
            Expose the trust policy used when resolving custom generation
            implementations.

        Architectural role:
            Protocol property at the inference-configuration boundary. It keeps
            code location separate from the security decision to execute remote
            code.

        Inputs (architectural provenance):
            Reads configuration supplied by the caller or model-loading layer.

        Outputs (downstream usage):
            Returns the boolean passed to custom generation loading code.

        Invariants/constraints:
            This flag should be explicit because it changes the security posture
            of model-loading workflows.

        """
        raise NotImplementedError

    @property
    def revision(self) -> str | None:
        """Return the revision used when resolving the custom generation."""
        raise NotImplementedError

    @property
    def use_cache(self) -> bool | None:
        """Return whether backend generate() should use model KV cache."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class GenerationControl:
    """Immutable generation-control payload for one backend generate() call.

    Purpose:
        Carry beam-search, diversity, sampling, and score-output settings in one
        canonical value object.

    Architectural role:
        Structured generation configuration shared by runtime adapters.

    """

    max_new_tokens: int
    num_beams: int
    num_beam_groups: int | None
    diversity_penalty: float
    num_return_sequences: int
    output_scores: bool
    early_stopping: bool = True
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    return_dict_in_generate: bool = True


@dataclass(frozen=True, slots=True)
class GenerationCall:
    """Structured runtime generate() request.

    Purpose:
        Bundle input tensors, optional attention mask, and the GenerationControl
        used for one token-generation call.

    Architectural role:
        Canonical request object for TokenGenerationRuntime.generate_tokens.

    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pad_token_id: int
    control: GenerationControl
    custom_generate_config: CustomGenerateConfig | None = None


class ScoringRuntime(Protocol):
    """Protocol for runtimes that can score prepared token sequences.

    Purpose:
        Define the tokenizer, device, eval-mode, and forward capabilities needed
        by logprob-scoring helpers and rebuild logic.

    Architectural role:
        Shared scoring/runtime protocol in inference.

    """

    def text_codec(self) -> TextCodec:
        """Return the tokenizer used by this scoring-capable runtime."""
        raise NotImplementedError

    def execution_device(self) -> torch.device:
        """Return the runtime device used for tensor execution."""
        raise NotImplementedError

    def ensure_eval_mode(self) -> None:
        """Ensure the underlying model is in eval mode before scoring."""
        raise NotImplementedError

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool | None = None,
    ) -> CausalLMForwardOutput:
        """Run one scoring-capable forward pass.

        Purpose:
            Define the model forward operation used by logit inspection and
            candidate scoring paths.

        Architectural role:
            Protocol method for runtimes that expose model logits while still
            hiding the concrete backend implementation.

        Inputs (architectural provenance):
            Receives model-ready tensors, attention masks, cache inputs, and
            forward options produced by scoring or decode state code.

        Outputs (downstream usage):
            Returns backend-native outputs containing logits and optional cache
            state for scoring, greedy selection, or cache rebuilding.

        Invariants/constraints:
            The method assumes tensors already live on the correct execution
            device and should not perform tokenizer or prompt formatting work.

        """
        raise NotImplementedError


class TokenGenerationRuntime(ScoringRuntime, Protocol):
    """Protocol for runtimes that can both score prefixes and generate token.

    Purpose:
        Extend the scoring-runtime contract with a structured token-generation
        operation used by probing and decode-adjacent helpers.

    Architectural role:
        Runtime capability protocol inside the inference type layer.

    Inputs (architectural provenance):
        Implemented by concrete runtime objects built around a tokenizer,
        backend model, and device.

    Outputs (downstream usage):
        Consumed by probing, generation helpers, and other components that need
        backend `generate()` access.

    Invariants/constraints:
        Generation calls must execute against the same runtime resources exposed
        through the paired scoring contract.

    """

    def generate_tokens(
        self, request: GenerationCall
    ) -> CausalLMGenerationOutput:
        """Run token generation for one structured generation request.

        Purpose:
            Define the high-level token-generation operation used by the public
            runtime once request and policy objects have been normalized.

        Architectural role:
            Protocol method between caller-facing generation APIs and concrete
            runtime implementations.

        Inputs (architectural provenance):
            Receives a `GenerationRequest` and `GenerationPolicy` assembled by
            notebook, script, test, or application code.

        Outputs (downstream usage):
            Returns a generation result consumed by evaluation, notebooks,
            telemetry extraction, or application callers.

        Invariants/constraints:
            Implementations should respect the compiled policy and produce one
            coherent result for the supplied request.

        """
        raise NotImplementedError


class GenerationRuntimeProtocol(TokenGenerationRuntime, Protocol):
    """Combined runtime protocol used across inference helpers.

    Purpose:
        Represent a runtime that can score prepared prefixes, generate tokens,
        and expose a tokenizer/device pair.

    Architectural role:
        Broadest runtime role consumed by probing and decode-adjacent helpers.

    """


class ChatScoringRuntime(ScoringRuntime, Protocol):
    """Protocol for scoring runtimes that expose a chat-capable tokenizer.

    Purpose:
        Add access to a chat-template-aware text codec on top of the base
        scoring-runtime capabilities.

    Architectural role:
        Chat-facing runtime protocol in the inference type boundary.

    Inputs (architectural provenance):
        Implemented by concrete runtimes that support chat-formatted prompting.

    Outputs (downstream usage):
        Consumed by StreamSession and prompting helpers when preparing
        chat-style requests.

    Invariants/constraints:
        `text_codec()` must return a tokenizer that satisfies the
        `ChatTextCodec` contract.

    """

    def text_codec(self) -> ChatTextCodec:
        """Return a chat-capable tokenizer for this runtime."""
        raise NotImplementedError


class ChatGenerationRuntime(
    GenerationRuntimeProtocol, ChatScoringRuntime, Protocol
):
    """Broadest chat-capable runtime protocol used by public generation helpers.

    Purpose:
        Combine prefix scoring, token generation, and chat-tokenizer access into
        one runtime role for decode and stream-session code.

    Architectural role:
        High-level runtime protocol at the edge of the inference subsystem.

    Inputs (architectural provenance):
        Implemented by concrete generation runtimes assembled during runtime
        construction.

    Outputs (downstream usage):
        Consumed by StreamSession and decode logic that need one unified runtime
        object.

    Invariants/constraints:
        The runtime must satisfy both token-generation and chat-scoring
        contracts over one coherent execution environment.

    """


__all__ = [
    "Telemetry",
    "CausalLMBackend",
    "CausalLMForwardOutput",
    "ChatGenerationRuntime",
    "ChatMessage",
    "ChatScoringRuntime",
    "ChatTemplateApplier",
    "ChatTemplateResult",
    "ChatTextCodec",
    "DTypeName",
    "ScoringRuntime",
    "CustomGenerateConfig",
    "GenerationCall",
    "GenerationControl",
    "CausalLMGenerationOutput",
    "OffsetEncoding",
    "TokenGenerationRuntime",
    "GenerationRuntimeProtocol",
    "PastKeyValues",
    "TextCodec",
    "TokenIds",
]
