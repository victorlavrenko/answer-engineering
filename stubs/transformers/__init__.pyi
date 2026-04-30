from pathlib import Path
from typing import Protocol

class _TokenizerLike(Protocol):
    pad_token_id: int | None
    eos_token: str | None
    pad_token: str | None

class _ModelLike(Protocol):
    def eval(self) -> None: ...

class AutoTokenizer:
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: str,
        *,
        revision: str | None = ...,
        use_fast: bool = ...,
        trust_remote_code: bool = ...,
    ) -> _TokenizerLike: ...

class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: str | Path,
        *,
        revision: str | None = ...,
        device_map: str | dict[str, int | str] | None = ...,
        dtype: object = ...,
        low_cpu_mem_usage: bool = ...,
        trust_remote_code: bool = ...,
    ) -> _ModelLike: ...
