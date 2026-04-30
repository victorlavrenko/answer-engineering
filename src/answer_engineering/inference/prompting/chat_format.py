"""Chat-message construction and chat-template application helpers.

Purpose:
    Build canonical system/user/assistant message lists and apply tokenizer chat
    templates when available.

Architectural role:
    Prompt-shaping module shared by StreamSession and related inference helpers.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from answer_engineering.inference.model_types import ChatMessage


class ChatTemplateRenderer(Protocol):
    """Protocol for objects that can render chat messages through a tokenizer.

    Architectural role:
        Prompting-facing capability contract.

    """

    @property
    def chat_template(self) -> str | None:
        """Return the chat-template string, if available."""
        raise NotImplementedError

    def apply_chat_template(
        self,
        messages: Sequence[ChatMessage],
        *,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
    ) -> str:
        """Apply the chat template to structured messages, optionally."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True, init=False)
class ChatTranscript:
    """Canonical chat transcript for one generation request.

    Purpose:
        Normalize the system prompt, user prompt, and optional partial assistant
        answer into a stable chat-message sequence.

    Architectural role:
        Prompt-shaping value object used by StreamSession and related helpers.

    """

    system_prompt: str
    user_prompt: str
    partial_answer: str

    def __init__(
        self, system_prompt: str, user_prompt: str, partial_answer: str = ""
    ) -> None:
        """Normalize prompt fields and store an optional partial answer.

        Purpose:
            Build the canonical chat transcript representation used before
            tokenizer template application.

        Architectural role:
            Prompting boundary between caller-provided messages and
            model-specific chat formatting.

        Inputs (architectural provenance):
            Receives system, user, and assistant-prefix text from runtime
            requests, notebooks, or tests.

        Outputs (downstream usage):
            Stores normalized message fields consumed by chat-template rendering
            and prompt-prefix construction.

        Invariants/constraints:
            Construction should preserve caller intent while avoiding later
            ambiguity about whether an assistant prefix is present.

        """
        object.__setattr__(self, "system_prompt", system_prompt.strip())
        object.__setattr__(self, "user_prompt", user_prompt.strip())
        object.__setattr__(self, "partial_answer", partial_answer)

    def messages(self) -> list[ChatMessage]:
        """Return the canonical chat-message list for this transcript.

        Purpose:
            Materialize the normalized system, user, and optional partial
            assistant messages in the order expected by chat-template rendering.

        Architectural role:
            Primary message-construction method on the prompting transcript
            object.

        Inputs (architectural provenance):
            Reads the normalized transcript fields stored on `ChatTranscript`.

        Outputs (downstream usage):
            Returns structured chat messages consumed by chat-template
            application and stream-session setup.

        Invariants/constraints:
            Message order must remain system first, user second, and partial
            assistant message last when present.

        """
        messages: list[ChatMessage] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]
        if self.partial_answer:
            messages.append(
                {"role": "assistant", "content": self.partial_answer}
            )
        return messages

    @staticmethod
    def apply_template_from_messages(
        *,
        tokenizer: ChatTemplateRenderer,
        messages: Sequence[ChatMessage],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply a tokenizer chat template to the supplied messages.

        Purpose:
            Render structured chat messages into the exact prompt text expected
            by a chat-capable tokenizer.

        Architectural role:
            Model-formatting boundary between repository transcript objects and
            tokenizer-specific chat template behavior.

        Inputs (architectural provenance):
            Receives normalized messages, a chat-template renderer, and the
            generation prompt flag chosen by the caller.

        Outputs (downstream usage):
            Returns prompt text consumed by tokenization, prefix expansion, and
            runtime generation.

        Invariants/constraints:
            The method should delegate template semantics to the renderer and
            should not silently invent model-specific formatting rules.

        """
        if tokenizer.chat_template is None:
            raise ValueError(
                "Tokenizer chat_template is missing; load an "
                "instruct/chat tokenizer with template."
            )
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        )


__all__ = ["ChatTranscript"]
