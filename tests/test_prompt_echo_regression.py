from __future__ import annotations

from collections.abc import Mapping, Sequence

from answer_engineering.inference.prompting.chat_format import (
    ChatTranscript,
)


class _TemplateTokenizer:
    chat_template = "test-template"
    name_or_path = "unit-test-tokenizer"

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        del tokenize
        rendered = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        if add_generation_prompt:
            return rendered + "\nASSISTANT:"
        return rendered


def _fake_generate(prompt: str, question_line: str) -> str:
    if not prompt.endswith("ASSISTANT:"):
        return f"{question_line}\n{question_line}"
    return "ANSWER: steroid"


def test_chat_prompt_has_assistant_generation_marker_to_prevent_echo() -> None:
    question = "What is the best management at this time?"
    messages = ChatTranscript(
        system_prompt="You are helpful", user_prompt=question
    ).messages()
    tokenizer = _TemplateTokenizer()

    prompt = ChatTranscript.apply_template_from_messages(
        tokenizer=tokenizer,
        messages=messages,
    )

    assert prompt.endswith("ASSISTANT:")
    answer = _fake_generate(prompt, question)
    assert answer.startswith("ANSWER:")
    assert answer.count(question) <= 1
