from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from _pytest.capture import CaptureFixture

from answer_engineering.inference.contracts import (
    GenerationPolicy,
    GenerationRequest,
)
from answer_engineering.inference.model_types import (
    ChatGenerationRuntime,
    OffsetEncoding,
)
from answer_engineering.inference.stream_io.api import StreamSession


class _ChatOrdinalCodec:
    bos_token_id: int | None = None
    pad_token_id: int | None = 0
    eos_token_id: int | None = 0
    chat_template: str | None = "stub-template"

    def encode(
        self, text: str, *, add_special_tokens: bool = False
    ) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(
        self, ids: list[int], *, skip_special_tokens: bool = True
    ) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in ids)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> OffsetEncoding:
        del add_special_tokens, return_offsets_mapping
        ids = [ord(ch) for ch in text]
        offsets = [(i, i + 1) for i, _ in enumerate(text)]
        return {"input_ids": ids, "offset_mapping": offsets}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str | list[int]:
        prompt = "\n".join(msg["content"] for msg in messages)
        if add_generation_prompt:
            prompt += "\nassistant:"
        if tokenize:
            ids = self.encode(prompt, add_special_tokens=False)
            return ids or [1]
        return prompt


class _ModelOutput:
    logits: torch.Tensor
    past_key_values: object | None

    def __init__(
        self, logits: torch.Tensor, past_key_values: object | None = None
    ):
        self.logits = logits
        self.past_key_values = past_key_values


@dataclass
class _ChatRuntime:
    tokenizer: _ChatOrdinalCodec

    def execution_device(self) -> torch.device:
        return torch.device("cpu")

    def ensure_eval_mode(self) -> None:
        return None

    def text_codec(self) -> _ChatOrdinalCodec:
        return self.tokenizer

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = False,
    ) -> _ModelOutput:
        del attention_mask, past_key_values, use_cache
        batch = int(input_ids.shape[0])
        n = int(input_ids.shape[1])
        vocab = 256
        logits = torch.full((batch, n, vocab), -100.0)
        logits[:, :, ord("x")] = 5.0
        logits[:, :, ord("z")] = 1.0
        return _ModelOutput(logits)


def test_stream_session_verbosity_one_has_stream_without_debug(
    capsys: CaptureFixture[str],
) -> None:
    runtime = _ChatRuntime(tokenizer=_ChatOrdinalCodec())
    request = GenerationRequest(question="Q")
    policy = GenerationPolicy(verbosity=1, stop_on_eos=False, max_new_tokens=2)

    _ = StreamSession(
        runtime=cast(ChatGenerationRuntime, runtime),
        request=request,
        policy=policy,
    ).run()

    out = capsys.readouterr().out
    assert out == "xx\n"
    assert "[AE]" not in out


def test_stream_session_verbosity_two_flushes_before_debug(
    capsys: CaptureFixture[str],
) -> None:
    runtime = _ChatRuntime(tokenizer=_ChatOrdinalCodec())
    request = GenerationRequest(question="Q", partial_answer="seed")
    policy = GenerationPolicy(
        verbosity=2,
        stop_on_eos=False,
        max_new_tokens=1,
        rules="## Replace: x\n\nWith:\n\n- z",
    )

    _ = StreamSession(
        runtime=cast(ChatGenerationRuntime, runtime),
        request=request,
        policy=policy,
    ).run()

    out = capsys.readouterr().out
    assert "[AE]" in out
