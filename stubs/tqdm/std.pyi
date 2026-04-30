from collections.abc import Iterable, Iterator, Mapping

class tqdm[T](Iterator[T]):
    def __init__(
        self,
        iterable: Iterable[T],
        *,
        desc: str | None = ...,
        unit: str | None = ...,
        dynamic_ncols: bool = ...,
    ) -> None: ...
    def __iter__(self) -> Iterator[T]: ...
    def __next__(self) -> T: ...
    def set_postfix(
        self,
        ordered_dict: Mapping[str, object] | None = None,
        refresh: bool | None = True,
        **kwargs: object,
    ) -> None: ...
    def update(self, n: int = 1) -> None: ...
    def close(self) -> None: ...
