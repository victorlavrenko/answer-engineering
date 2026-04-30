from __future__ import annotations

from answer_engineering.inference.decode.decode_loop import (
    collect_stop_token_ids,
)
from answer_engineering.inference.model_types import OffsetEncoding


class _Tokenizer:
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    bos_token_id: int | None = None

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return list()

    @staticmethod
    def decode(ids: list[int], *, skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return ""

    @staticmethod
    def __call__(
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> OffsetEncoding:
        del add_special_tokens, return_offsets_mapping
        ids = [ord(ch) for ch in text]
        offsets = [(idx, idx + 1) for idx, _ in enumerate(text)]
        return {"input_ids": ids, "offset_mapping": offsets}


def test_collect_stop_token_ids_includes_tokenizer_eos() -> None:
    ids = collect_stop_token_ids(_Tokenizer())
    assert ids == {2}
