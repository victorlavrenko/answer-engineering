from collections.abc import Iterable, Iterator, Mapping, Sequence

class Value:
    dtype: str

class Features(Mapping[str, Value]): ...

class Dataset(Iterable[Mapping[str, object]]):
    features: Features

    @classmethod
    def from_list(cls, rows: list[dict[str, object]]) -> Dataset: ...
    def __iter__(self) -> Iterator[Mapping[str, object]]: ...

def load_dataset(
    path: str,
    name: str | None = ...,
    data_dir: str | None = ...,
    data_files: str
    | Sequence[str]
    | Mapping[str, str | Sequence[str]]
    | None = ...,
    *,
    split: str,
    cache_dir: str | None = ...,
    revision: str | None = ...,
    streaming: bool = ...,
    **config_kwargs: object,
) -> Dataset: ...
