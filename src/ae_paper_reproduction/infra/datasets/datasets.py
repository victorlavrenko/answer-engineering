"""Reproduction dataset adapters, normalization, caching, and row selection.

Purpose:
    Provide the dataset-boundary abstractions and concrete adapters used by
    reproduction planning and evaluation workflows.

Architectural role:
    Reproduction dataset adapter boundary between external data sources and
    canonical `DatasetRow` execution inputs.

Architectural direction:
    Preserve one truthful dataset boundary while reducing mixing of
    normalization policy, materialization policy, and backend-specific
    adaptation concerns.

Why this matters:
    This module currently mixes canonical row normalization, caching, selection,
    and concrete Hugging Face–backed adaptation in one large seam.

What better would look like:
    New dataset sources can be added with less cross-module editing while row
    identity and metadata semantics stay easy to explain.

How improvement can be recognized:
    - Clearer ownership boundaries between normalization, caching, and backend
      adapters
    - Lower extension cost when introducing new dataset sources
    - Fewer backend-specific assumptions in canonical dataset contracts

Open constraint:
    The boundary must remain flexible while evaluation workflows continue to
    evolve.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol, cast

import datasets as hf_datasets
from ae_paper_reproduction.core.evaluation.result_types import (
    DatasetRow,
    ExternalRow,
)
from datasets import Value


def _empty_row_index() -> dict[str, DatasetRow]:
    """Create a fresh question-id index for cached dataset rows.

    Purpose:
        Supply the default dictionary used to cache dataset rows by stable
        question identifier.

    Architectural role:
        Small construction helper inside the cached-dataset adapter layer.

    Inputs (architectural provenance):
        Called automatically by dataclass field initialization when a cached
        dataset is constructed.

    Outputs (downstream usage):
        Returns the mutable row index consumed by lookup and caching methods on
        `CachedDataset`.

    Invariants/constraints:
        Each call must return a new empty dictionary so dataset instances do not
        share mutable state.

    """
    return {}


class Dataset(Protocol):
    """Protocol for materialized benchmark datasets.

    A dataset exposes benchmark rows through a small, stable interface used by
    notebook planning and reproduction sessions. Concrete adapters can load rows
    from Hugging Face or another source as long as they preserve the public row
    contract.

    .. note::
        Call :meth:`~ae_paper_reproduction.Dataset.materialize` before selecting
        rows for subruns. Materialization makes row access deterministic for the
        rest of the notebook session.

    Examples:
        ```python
        dataset = CachedHFDataset(DATASET_ID, SPLIT)
        dataset.materialize()

        rows = dataset.rows(n=10, case_type="baseline")
        first = dataset.row(rows[0].id)
        ```

    Methods:
        :meth:`~ae_paper_reproduction.Dataset.materialize`
            Load or prepare the dataset and return the dataset object.

        :meth:`~ae_paper_reproduction.Dataset.iter_rows`
            Iterate over materialized rows.

        :meth:`~ae_paper_reproduction.Dataset.row`
            Return one row by question id.

        :meth:`~ae_paper_reproduction.Dataset.metadata`
            Return dataset-level metadata.

        :meth:`~ae_paper_reproduction.Dataset.rows`
            Return selected rows by count, id, and optional case type.

    Runtime behavior:
        The planning layer uses this protocol to select row subsets before any
        model generation occurs. Dataset access should not compile rules or call
        the runtime.

    Architectural role:
        Data-source boundary for paper reproduction. It lets notebook planning
        work against a protocol rather than a specific Hugging Face
        implementation.

    Consumes:
        External dataset storage, cache state, or already materialized row
        objects, depending on the concrete implementation.

    Produces:
        Dataset rows consumed by :class:`~ae_paper_reproduction.SubrunTask` and
        :class:`~ae_paper_reproduction.RulesetEvaluationResult`.

    Invariants:
        Row ids should be stable within one materialized dataset. Filtering by
        case type or question id should not mutate dataset contents.

    Developer Notes:
        Keep generation concerns outside this protocol. Dataset adapters should
        own loading, validation, row normalization, and cache semantics only.

    Todo:
        Add richer sampling or split metadata only when notebooks need it as an
        explicit public feature. Avoid hiding sampling policy inside adapters.

    See Also:
        :class:`~ae_paper_reproduction.CachedHFDataset`
        :class:`~ae_paper_reproduction.SubrunTask`
        :class:`~ae_paper_reproduction.NotebookSubruns`

    """

    def materialize(self) -> Dataset:
        """Materialize the dataset adapter and return the ready adapter.

        Purpose:
            Define one required dataset operation for reproduction workflows
            that need stable access to evaluation data.

        Architectural role:
            Protocol member on the reproduction dataset boundary.
            Implementations hide Hugging Face or cache-specific mechanics behind
            canonical row access.

        Inputs (architectural provenance):
            Receives caller-provided selection or identity arguments when
            applicable and reads dataset state owned by the concrete adapter.

        Outputs (downstream usage):
            Returns the materialized adapter, normally `self`, so planning and
            execution code can chain setup before row access.

        Invariants/constraints:
            Implementations must expose rows using the same canonical identity
            and normalization rules across all `Dataset` protocol methods.

        """
        raise NotImplementedError

    def iter_rows(self) -> Iterator[DatasetRow]:
        """Iterate canonical rows exposed by this dataset adapter.

        Purpose:
            Define one required dataset operation for reproduction workflows
            that need stable access to evaluation data.

        Architectural role:
            Protocol member on the reproduction dataset boundary.
            Implementations hide Hugging Face or cache-specific mechanics behind
            canonical row access.

        Inputs (architectural provenance):
            Receives caller-provided selection or identity arguments when
            applicable and reads dataset state owned by the concrete adapter.

        Outputs (downstream usage):
            Yields `DatasetRow` values consumed by task selection, evaluation
            loops, and reporting code.

        Invariants/constraints:
            Implementations must expose rows using the same canonical identity
            and normalization rules across all `Dataset` protocol methods.

        """
        raise NotImplementedError

    def row(self, question_id: str) -> DatasetRow:
        """Return one canonical row by question identifier.

        Purpose:
            Define one required dataset operation for reproduction workflows
            that need stable access to evaluation data.

        Architectural role:
            Protocol member on the reproduction dataset boundary.
            Implementations hide Hugging Face or cache-specific mechanics behind
            canonical row access.

        Inputs (architectural provenance):
            Receives caller-provided selection or identity arguments when
            applicable and reads dataset state owned by the concrete adapter.

        Outputs (downstream usage):
            Returns the canonical `DatasetRow` for callers that need direct
            indexed access to a known case.

        Invariants/constraints:
            Implementations must expose rows using the same canonical identity
            and normalization rules across all `Dataset` protocol methods.

        """
        raise NotImplementedError

    def metadata(self) -> Mapping[str, str]:
        """Return lightweight metadata describing the dataset source.

        Purpose:
            Define one required dataset operation for reproduction workflows
            that need stable access to evaluation data.

        Architectural role:
            Protocol member on the reproduction dataset boundary.
            Implementations hide Hugging Face or cache-specific mechanics behind
            canonical row access.

        Inputs (architectural provenance):
            Receives caller-provided selection or identity arguments when
            applicable and reads dataset state owned by the concrete adapter.

        Outputs (downstream usage):
            Returns string metadata consumed by planning, telemetry, and reports
            without forcing callers to inspect adapter internals.

        Invariants/constraints:
            Implementations must expose rows using the same canonical identity
            and normalization rules across all `Dataset` protocol methods.

        """
        raise NotImplementedError

    def rows(
        self,
        *,
        n: int | None = None,
        question_id: str | None = None,
        case_type: str | None = None,
    ) -> tuple[DatasetRow, ...]:
        """Return a filtered tuple of canonical dataset rows.

        Purpose:
            Provide the row-selection operation used by planning and
            reproduction execution without exposing adapter-specific storage
            details.

        Architectural role:
            Public dataset protocol member at the reproduction data boundary.

        Inputs (architectural provenance):
            `n`, `question_id`, and `case_type` come from notebook, script, or
            subrun-selection code.

        Outputs (downstream usage):
            Returns canonical `DatasetRow` values consumed by subrun task
            selection and evaluation setup.

        Invariants/constraints:
            Implementations should apply filters consistently with `iter_rows`
            and `row`; row identity and ordering must remain stable for a
            materialized dataset.

        """
        raise NotImplementedError


@dataclass(slots=True)
class CachedDataset(Dataset, ABC):
    """Abstract cached dataset adapter over canonical `DatasetRow` values.

    Purpose:
        Provide shared caching, lookup, and row-selection behavior for datasets
        whose external rows can be normalized into canonical reproduction rows.

    Architectural role:
        Base adapter in the reproduction data-access layer.

    Inputs (architectural provenance):
        Constructed by reproduction code with dataset identity and field-name
        configuration.

    Outputs (downstream usage):
        Exposes canonical dataset rows consumed by planning and execution
        layers.

    Invariants/constraints:
        The cache, row index, and field mappings must all refer to the same
        external dataset schema.

    Todo:
        Target:
            This boundary logically belongs closer to reproduction data access
            than to a generic infra bucket.

        Boundary note:
            The code is useful as infrastructure-style adapter code, but its
            semantics are reproduction-specific.

    """

    dataset_id: str
    split: str
    question_field: str = "question"
    gold_field: str = "gold"
    id_field: str = "id"
    case_type_field: str = "case_type"
    _rows_cache: tuple[DatasetRow, ...] | None = field(
        default=None, init=False, repr=False
    )
    _rows_by_id: dict[str, DatasetRow] = field(
        default_factory=_empty_row_index,
        init=False,
        repr=False,
    )

    def metadata(self) -> Mapping[str, str]:
        """Return lightweight metadata describing this dataset source.

        Purpose:
            Expose stable identifying information about the configured dataset
            without forcing row materialization.

        Architectural role:
            Metadata accessor on the cached dataset adapter.

        Inputs (architectural provenance):
            Reads dataset identity fields stored on the adapter instance.

        Outputs (downstream usage):
            Returns a small mapping consumed by reporting, labeling, and
            debugging code.

        Invariants/constraints:
            Metadata values must describe the same dataset source that row
            iteration will later expose.

        """
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
        }

    def materialize(self) -> CachedDataset:
        """Load and cache all rows for repeated reuse.

        Purpose:
            Populate the in-memory row cache on first call and return the same
            dataset object for fluent use.

        Architectural role:
            Materialization operation on the cached dataset adapter.

        Inputs (architectural provenance):
            Invoked by reproduction planning or execution code when repeated row
            access is expected.

        Outputs (downstream usage):
            Returns `self` with `_rows_cache` populated for subsequent fast
            iteration and lookup.

        Invariants/constraints:
            Materialization must preserve row order and keep the row-id index
            synchronized with the cache.

        """
        if self._rows_cache is None:
            self._rows_cache = tuple(self._iter_uncached_rows())
        return self

    def iter_rows(self) -> Iterator[DatasetRow]:
        """Iterate canonical dataset rows.

        Purpose:
            Yield dataset rows from the in-memory cache when available,
            otherwise stream them through uncached normalization.

        Architectural role:
            Primary row-iteration method on the cached dataset adapter.

        Inputs (architectural provenance):
            Invoked by reproduction code that needs to traverse dataset rows in
            source order.

        Outputs (downstream usage):
            Returns an iterator of canonical `DatasetRow` values.

        Invariants/constraints:
            Row iteration must preserve the underlying dataset order exposed by
            the external source.

        """
        if self._rows_cache is not None:
            return iter(self._rows_cache)
        return self._iter_uncached_rows()

    def row(self, question_id: str) -> DatasetRow:
        """Return one canonical dataset row by question identifier.

        Purpose:
            Resolve a stable question id to its normalized dataset row, using
            the cache or streaming lookup as needed.

        Architectural role:
            Direct row-lookup method on the cached dataset adapter.

        Inputs (architectural provenance):
            Consumes a question id requested by planning, debugging, or targeted
            evaluation flows.

        Outputs (downstream usage):
            Returns the matching canonical dataset row.

        Invariants/constraints:
            If a row is returned, its `id` must equal the normalized requested
            question id.

        """
        normalized_question_id = str(question_id)

        cached_row = self._rows_by_id.get(normalized_question_id)
        if cached_row is not None:
            return cached_row

        if self._rows_cache is None:
            for row in self._iter_uncached_rows():
                if row.id == normalized_question_id:
                    return row

        msg = f"QUESTION_ID={question_id!r} not found in dataset"
        raise ValueError(msg)

    def rows(
        self,
        *,
        n: int | None = None,
        question_id: str | None = None,
        case_type: str | None = None,
    ) -> tuple[DatasetRow, ...]:
        """Select a tuple of rows using simple dataset-side filters.

        Purpose:
            Return either one row by question id or an ordered prefix of rows
            optionally filtered by case type and limited by count.

        Architectural role:
            Convenience row-selection method on the cached dataset adapter.

        Inputs (architectural provenance):
            Consumes simple selection criteria from reproduction planning or
            execution code.

        Outputs (downstream usage):
            Returns an immutable tuple of selected `DatasetRow` values.

        Invariants/constraints:
            Selection must preserve dataset iteration order after filtering.

        """
        if question_id is not None:
            selected_row = self.row(question_id)
            if case_type is not None and selected_row.case_type != case_type:
                msg = f"QUESTION_ID={question_id!r} not found in dataset"
                raise ValueError(msg)
            return (selected_row,)

        selected_rows: list[DatasetRow] = []
        for row in self.iter_rows():
            if case_type is not None and row.case_type != case_type:
                continue
            selected_rows.append(row)
            if n is not None and len(selected_rows) >= n:
                return tuple(selected_rows)

        return tuple(selected_rows)

    def _iter_uncached_rows(self) -> Iterator[DatasetRow]:
        """Normalize and yield external rows without relying on the full cache.

        Purpose:
            Convert raw external-row mappings into canonical `DatasetRow` values
            and remember each row in the id index as it is seen.

        Architectural role:
            Shared uncached normalization pipeline inside the cached dataset
            adapter.

        Inputs (architectural provenance):
            Consumes raw external rows yielded by the concrete
            `_iter_external_rows()` implementation.

        Outputs (downstream usage):
            Yields canonical rows consumed by iteration, materialization, and
            lookup helpers.

        Invariants/constraints:
            Every yielded row must be remembered in `_rows_by_id` under its
            canonical id.

        """
        for raw_row in self._iter_external_rows():
            row = DatasetRow.from_external_row(
                raw_row,
                id_field=self.id_field,
                case_type_field=self.case_type_field,
                question_field=self.question_field,
                gold_field=self.gold_field,
            )
            self._remember_row(row)
            yield row

    def _remember_row(self, row: DatasetRow) -> None:
        """Remember a normalized row in the question-id index.

        Purpose:
            Update the internal row lookup table so later direct `row()` calls
            can resolve ids without re-reading earlier rows.

        Architectural role:
            Cache-index maintenance helper inside the cached dataset adapter.

        Inputs (architectural provenance):
            Consumes one canonical `DatasetRow` produced during normalization or
            materialization.

        Outputs (downstream usage):
            Mutates the internal id index used by row lookup.

        Invariants/constraints:
            The stored mapping key must equal `row.id`.

        """
        self._rows_by_id[row.id] = row

    @abstractmethod
    def _iter_external_rows(self) -> Iterator[ExternalRow]:
        """Iterate raw rows from the loaded Hugging Face dataset.

        Purpose:
            Read the validated dataset object and yield each row as a normalized
            mapping suitable for later conversion into `DatasetRow`.

        Architectural role:
            Hugging Face–specific raw-row iterator inside the cached dataset
            adapter.

        Inputs (architectural provenance):
            Consumes the dataset object returned by
            `_load_and_validate_dataset()`.

        Outputs (downstream usage):
            Yields raw row mappings consumed by the inherited
            uncached-normalization path.

        Invariants/constraints:
            Every yielded value must be a mapping with string keys.

        """
        raise NotImplementedError


@dataclass(slots=True)
class CachedHFDataset(CachedDataset):
    """Hugging Face dataset adapter with row caching.

    Load a benchmark split from Hugging Face, validate the expected schema, and
    cache normalized rows for deterministic notebook access. This is the
    concrete dataset implementation used by the reproduction notebook.

    .. note::
        Materialize the dataset once before constructing
        :class:`~ae_paper_reproduction.NotebookSubruns`. Reusing the cached rows
        keeps subrun selection and evaluation aligned.

    Examples:
        ```python
        dataset = CachedHFDataset(DATASET_ID, SPLIT)
        dataset.materialize()

        subruns = NotebookSubruns(
            NOTEBOOK_NAME,
            dataset=dataset,
            model=runtime,
        )
        ```

    Attributes:
        dataset_id: Hugging Face dataset identifier.
        split: Dataset split name to load.

    Runtime behavior:
        The adapter loads external rows lazily, validates them, normalizes them
        into internal dataset rows, and remembers rows by id for later ``row``
        and ``rows`` calls.

    Architectural role:
        Concrete data-source implementation behind the public
        :class:`~ae_paper_reproduction.Dataset` protocol.

    Consumes:
        Hugging Face dataset id, split name, and row values with the expected
        benchmark schema.

    Produces:
        Cached dataset rows consumed by subrun planning and evaluation
        reporting.

    Invariants:
        Materialized rows should have stable ids, questions, gold answers, and
        case types. Schema validation should fail early when the external
        dataset shape is incompatible with reproduction code.

    Developer Notes:
        Keep Hugging Face-specific validation here rather than leaking it into
        notebook planning. If additional data sources are added, implement the
        dataset protocol instead of branching in notebook code.

    Todo:
        Consider explicit cache invalidation and dataset fingerprint reporting
        for paper reproducibility. Avoid promising backward compatibility before
        the data artifact format is finalized.

    See Also:
        :class:`~ae_paper_reproduction.Dataset`
        :class:`~ae_paper_reproduction.NotebookSubruns`
        :class:`~ae_paper_reproduction.SubrunTask`

    """

    revision: str | None = None
    _dataset_cache: hf_datasets.Dataset | None = field(
        default=None, init=False, repr=False
    )

    def _iter_external_rows(self) -> Iterator[ExternalRow]:
        """Iterate raw rows from the loaded Hugging Face dataset.

        Purpose:
            Read the validated dataset object and yield each row as a normalized
            mapping suitable for later conversion into `DatasetRow`.

        Architectural role:
            Hugging Face–specific raw-row iterator inside the cached dataset
            adapter.

        Inputs (architectural provenance):
            Consumes the dataset object returned by
            `_load_and_validate_dataset()`.

        Outputs (downstream usage):
            Yields raw row mappings consumed by the inherited
            uncached-normalization path.

        Invariants/constraints:
            Every yielded value must be a mapping with string keys.

        """
        dataset = self._load_and_validate_dataset()
        raw_rows = cast(Iterable[object], dataset)

        for raw_row in raw_rows:
            yield self._require_mapping_row(raw_row)

    def _load_and_validate_dataset(self) -> hf_datasets.Dataset:
        """Load the configured Hugging Face dataset split and cache it after.

        Purpose:
            Materialize the external dataset object exactly once, verify
            required fields, and reuse the loaded object across later row
            iterations.

        Architectural role:
            Dataset-loading boundary method for the Hugging Face adapter.

        Inputs (architectural provenance):
            Uses dataset id, split, and optional revision stored on the adapter
            instance.

        Outputs (downstream usage):
            Returns a validated `hf_datasets.Dataset` consumed by raw-row
            iteration.

        Invariants/constraints:
            The cached dataset object must satisfy `_require_expected_schema()`
            before it is stored.

        """
        if self._dataset_cache is None:
            load_dataset = cast(
                Callable[..., hf_datasets.Dataset],
                hf_datasets.load_dataset,
            )
            dataset = load_dataset(
                self.dataset_id,
                split=self.split,
                revision=self.revision,
                streaming=False,
            )
            self._require_expected_schema(dataset)
            self._dataset_cache = dataset
        return self._dataset_cache

    def _require_expected_schema(self, dataset: hf_datasets.Dataset) -> None:
        """Validate that the loaded dataset exposes the required string fields.

        Purpose:
            Enforce the schema assumptions needed to convert external rows into
            canonical reproduction rows.

        Architectural role:
            Schema-validation helper inside the Hugging Face dataset adapter.

        Inputs (architectural provenance):
            Consumes the loaded Hugging Face dataset object before row iteration
            begins.

        Outputs (downstream usage):
            Raises on schema mismatch; otherwise leaves the dataset approved for
            later use.

        Invariants/constraints:
            Each configured id, case-type, question, and gold field must exist
            and be represented as a string-like feature.

        """
        for field_name in (
            self.id_field,
            self.case_type_field,
            self.question_field,
            self.gold_field,
        ):
            if field_name not in dataset.features:
                msg = f"Dataset schema missing required field: {field_name}"
                raise KeyError(msg)

            feature = cast(hf_datasets.Features, dataset.features[field_name])
            if not isinstance(feature, Value) or feature.dtype not in {
                "string",
                "large_string",
            }:
                msg = (
                    f"Dataset schema field {field_name!r} must be "
                    f"a string feature, got {feature!r}"
                )
                raise TypeError(msg)

    def _require_mapping_row(self, value: object) -> ExternalRow:
        """Normalize one external row to a string-keyed mapping.

        Purpose:
            Enforce that a row yielded by the Hugging Face dataset is
            mapping-like and that every key is a string before later canonical
            conversion.

        Architectural role:
            Representation-boundary row validator inside the Hugging Face
            dataset adapter.

        Inputs (architectural provenance):
            Consumes one raw row object yielded by the external dataset library.

        Outputs (downstream usage):
            Returns a normalized mapping consumed by `_iter_external_rows()` and
            later `DatasetRow.from_external_row()`.

        Invariants/constraints:
            Returned mappings must contain only string keys.

        """
        if not isinstance(value, Mapping):
            msg = f"Expected mapping row, got {type(value).__name__}"
            raise TypeError(msg)

        raw_mapping = cast(Mapping[object, object], value)
        normalized: dict[str, object] = {}

        for key, item in raw_mapping.items():
            if not isinstance(key, str):
                msg = f"Expected string key, got {type(key).__name__}"
                raise TypeError(msg)
            normalized[key] = item

        return normalized


__all__ = [
    "CachedDataset",
    "CachedHFDataset",
    "Dataset",
]
