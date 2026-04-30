from __future__ import annotations

import torch

from answer_engineering.inference.model_types import (
    GenerationCall,
    OffsetEncoding,
)


class OrdinalTextCodecStub:
    bos_token_id: int | None = None
    pad_token_id: int | None = 0
    eos_token_id: int | None = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
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


class StubModelOutput:
    logits: torch.Tensor
    past_key_values: object | None

    def __init__(
        self, logits: torch.Tensor, past_key_values: object | None = None
    ) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class StubCausalLMBackend:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def eval(self) -> None:
        return None

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = False,
    ) -> StubModelOutput:
        del attention_mask, past_key_values, use_cache
        batch: int = int(input_ids.shape[0])
        n: int = int(input_ids.shape[1])
        vocab = 256
        logits = torch.full((batch, n, vocab), -100.0)
        logits[:, :, ord("x")] = 4.0
        logits[:, :, ord("y")] = 1.0
        logits[:, :, ord("z")] = 0.5
        return StubModelOutput(logits)


class StubGenerationOutput:
    def __init__(self, sequences: torch.Tensor) -> None:
        self.sequences = sequences
        self.sequences_scores: torch.Tensor | None = None


class GenerationRuntimeStub:
    def __init__(
        self, model: StubCausalLMBackend, tokenizer: OrdinalTextCodecStub
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = model.device

    @classmethod
    def loaded_runtime(cls) -> GenerationRuntimeStub:
        return cls(
            model=StubCausalLMBackend(), tokenizer=OrdinalTextCodecStub()
        )

    def text_codec(self) -> OrdinalTextCodecStub:
        return self._tokenizer

    def execution_device(self) -> torch.device:
        return self._device

    def ensure_eval_mode(self) -> None:
        self._model.eval()

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = False,
    ) -> StubModelOutput:
        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def generate_tokens(self, request: GenerationCall) -> StubGenerationOutput:
        return StubGenerationOutput(sequences=request.input_ids)
