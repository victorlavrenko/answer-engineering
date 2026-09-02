"""Pinned dataset loading and strict multiple-choice normalization."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, TypeGuard

from .answer_utils import (
    ChoiceNormalizationError,
    canonical_gold_labels,
    canonicalize_choices,
    normalize_text,
)
from .exclusions import QuestionExclusions
from .io_utils import sha256_file
from .json_types import JsonObject, json_object, object_sequence, object_value
from .question_text import strip_exact_embedded_choice_suffix
from .types import Question

DEFAULT_DATASETS = ("medqa", "medxpertqa", "afrimedqa")
DEFAULT_SEED = 20260722
SELECTION_ALGORITHM = "normalize-deduplicate-exclude-shuffle-v4"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One immutable upstream dataset source."""

    name: str
    repository: str
    config: str | None
    split: str
    revision: str
    source_split: str
    embedded_choice_block: bool = False
    gold_index_bases: tuple[tuple[str, int], ...] = (
        ("answer_idx", 0),
        ("answer_index", 0),
        ("answer", 1),
        ("label", 1),
        ("gold", 1),
        ("correct", 1),
    )

    def to_dict(self) -> JsonObject:
        """Return an explicitly typed JSON representation."""
        return json_object(
            {
                "name": self.name,
                "repository": self.repository,
                "config": self.config,
                "split": self.split,
                "revision": self.revision,
                "source_split": self.source_split,
                "embedded_choice_block": self.embedded_choice_block,
                "gold_index_bases": [list(item) for item in self.gold_index_bases],
            },
            path=f"dataset spec {self.name}",
        )


DATASET_SPECS: dict[str, DatasetSpec] = {
    "medqa": DatasetSpec(
        "medqa",
        "GBaker/MedQA-USMLE-4-options",
        None,
        "test",
        "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",
        "test",
    ),
    "medxpertqa": DatasetSpec(
        "medxpertqa",
        "TsinghuaC3I/MedXpertQA",
        "Text",
        "test",
        "7e7c465a68eb2b866926bfa59c8c9d17a8daba65",
        "test",
        True,
    ),
    "afrimedqa": DatasetSpec(
        "afrimedqa",
        "afrimedqa/afrimedqa_v2",
        None,
        "train",
        "3b4382fa0bb51bfc026f5813021ab0ec7be9de8f",
        "internal test partition",
    ),
}

STEM_FIELDS = ("question_clean", "question", "question_text", "stem", "prompt", "query")
CHOICE_FIELDS = ("options", "choices", "answer_options", "answer_choices", "mcq_options")
GOLD_FIELDS = (
    "answer_idx",
    "answer_index",
    "correct_option",
    "correct_answer",
    "answer",
    "label",
    "gold",
    "correct",
)
ID_FIELDS = ("id", "sample_id", "question_id", "qid", "uid", "index")

DEFAULT_GOLD_INDEX_BASES: dict[str, int] = {
    "answer_idx": 0,
    "answer_index": 0,
    "answer": 1,
    "label": 1,
    "gold": 1,
    "correct": 1,
}
MEDIA_FIELDS = (
    "image",
    "images",
    "image_path",
    "image_paths",
    "figure",
    "figures",
    "table",
    "tables",
    "audio",
    "video",
)
MEDIA_REFERENCE_RE = re.compile(
    r"\b(?:shown|depicted|illustrated|pictured)\s+(?:above|below|here)|"
    r"\b(?:the|this|following)\s+(?:image|figure|radiograph|x[- ]?ray|ct|mri|"
    r"ultrasound|photograph|ecg|ekg|tracing|graph|chart|table|diagram)|"
    r"\bin\s+(?:the|this)\s+(?:image|figure|table|radiograph|scan|tracing)\b",
    re.IGNORECASE,
)


class DatasetNormalizationError(ValueError):
    """Raised when a row cannot be normalized without guessing."""


class DatasetLoader(Protocol):
    """Typed subset of ``datasets.load_dataset`` used by this package."""

    def __call__(
        self,
        repository: str,
        config: str | None,
        *,
        split: str,
        revision: str,
    ) -> Iterable[object]: ...


def _is_dataset_loader(value: object) -> TypeGuard[DatasetLoader]:
    """Validate the dynamically imported dataset loader callable."""
    return callable(value)


def source_descriptor(dataset_names: Sequence[str], input_jsonl: Path | None) -> JsonObject:
    """Return the complete source identity used in the experiment manifest."""
    if input_jsonl is not None:
        resolved = input_jsonl.resolve()
        return {
            "kind": "jsonl",
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        }
    unknown = [name for name in dataset_names if name not in DATASET_SPECS]
    if unknown:
        raise ValueError("unknown datasets: " + ", ".join(unknown))
    return {
        "kind": "huggingface",
        "datasets": [DATASET_SPECS[name].to_dict() for name in dataset_names],
    }


def _first_text(row: Mapping[str, object], fields: Sequence[str]) -> tuple[str, str]:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split()), field
    raise DatasetNormalizationError("missing non-empty question stem")


def _candidate_choices(row: Mapping[str, object]) -> list[tuple[str, dict[str, str]]]:
    candidates: list[tuple[str, dict[str, str]]] = []
    errors: list[str] = []
    for field in CHOICE_FIELDS:
        if field not in row or row[field] in (None, "", [], {}):
            continue
        try:
            choices = canonicalize_choices(row[field])
        except (ChoiceNormalizationError, TypeError, ValueError) as exc:
            errors.append(f"{field}: {exc}")
            continue
        candidates.append((field, choices))
    # Some sources expose A/B/C/D as top-level columns.
    top_level = {
        label: row[label]
        for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if row.get(label) not in (None, "")
    }
    if top_level:
        try:
            candidates.append(("top-level labels", canonicalize_choices(top_level)))
        except (ChoiceNormalizationError, TypeError, ValueError) as exc:
            errors.append(f"top-level labels: {exc}")
    if not candidates:
        detail = "; ".join(errors) if errors else "no choice field"
        raise DatasetNormalizationError(f"no usable choices ({detail})")
    canonical = candidates[0][1]
    conflicts = [field for field, choices in candidates[1:] if choices != canonical]
    if conflicts:
        raise DatasetNormalizationError(
            "conflicting choice encodings: " + candidates[0][0] + " vs " + ", ".join(conflicts)
        )
    return candidates


def _gold(
    row: Mapping[str, object],
    choices: Mapping[str, str],
    index_bases: Mapping[str, int],
) -> tuple[str, str, str]:
    parsed: list[tuple[str, set[str], object]] = []
    malformed: list[str] = []
    for field in GOLD_FIELDS:
        if field not in row or row[field] in (None, "", [], {}):
            continue
        try:
            labels = canonical_gold_labels(row[field], choices, index_base=index_bases.get(field))
        except (TypeError, ValueError) as exc:
            malformed.append(f"{field}: {exc}")
            continue
        if labels:
            parsed.append((field, labels, row[field]))
        else:
            malformed.append(f"{field}: unrecognized value {row[field]!r}")
    if not parsed:
        raise DatasetNormalizationError("no usable gold field; " + "; ".join(malformed))
    union: set[str] = {label for _, labels, _ in parsed for label in labels}
    if len(union) != 1 or any(labels != union for _, labels, _ in parsed):
        detail = ", ".join(f"{field}={sorted(labels)}" for field, labels, _ in parsed)
        raise DatasetNormalizationError(f"conflicting gold fields: {detail}")
    label = next(iter(union))
    sources = "+".join(field for field, _, _ in parsed)
    raw = json.dumps(
        {
            "parsed": {field: value for field, _, value in parsed},
            "malformed": malformed,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return label, sources, raw


def _has_media_dependency(row: Mapping[str, object], stem: str) -> bool:
    for field in MEDIA_FIELDS:
        value = row.get(field)
        if value not in (None, "", [], {}, False):
            return True
    return bool(MEDIA_REFERENCE_RE.search(stem))


def _identifier(row: Mapping[str, object], dataset: str, index: int) -> str:
    for field in ID_FIELDS:
        value = row.get(field)
        if value not in (None, ""):
            return f"{dataset}:{value}"
    return f"{dataset}:row-{index}"


def _clean_stem(
    stem: str,
    choices: Mapping[str, str],
    *,
    require_embedded_choice_block: bool,
) -> str:
    cleaned, removed = strip_exact_embedded_choice_suffix(stem, choices)
    if require_embedded_choice_block and not removed:
        raise DatasetNormalizationError(
            "expected terminal Answer Choices block to match structured options"
        )
    return cleaned


def normalize_row(
    row: Mapping[str, object],
    *,
    dataset: str,
    source_split: str,
    index: int,
    gold_index_bases: Mapping[str, int] | None = None,
    embedded_choice_block: bool = False,
) -> Question:
    """Normalize one row, rejecting ambiguity instead of guessing."""
    stem, _stem_field = _first_text(row, STEM_FIELDS)
    if _has_media_dependency(row, stem):
        raise DatasetNormalizationError("question depends on omitted media or table content")
    choice_candidates = _candidate_choices(row)
    choices = choice_candidates[0][1]
    stem = _clean_stem(
        stem,
        choices,
        require_embedded_choice_block=embedded_choice_block,
    )
    gold, source, raw = _gold(
        row, choices, DEFAULT_GOLD_INDEX_BASES if gold_index_bases is None else gold_index_bases
    )
    return Question(
        id=_identifier(row, dataset, index),
        dataset=dataset,
        stem=stem,
        choices=choices,
        gold=gold,
        gold_text=choices[gold],
        gold_source=source,
        gold_raw=raw,
        source_split=source_split,
    )


def _partition_filter(dataset: str, row: Mapping[str, object]) -> bool:
    if dataset != "afrimedqa":
        return True
    # The paper source uses the phase-2 internal test partition when present.
    for field in ("split", "partition", "subset"):
        value = row.get(field)
        if value is not None:
            normalized = normalize_text(str(value))
            return normalized in {"test", "internal test", "internaltest"}
    return True


def _normalize_rows(
    rows: Iterable[object],
    *,
    dataset: str,
    source_split: str,
    gold_index_bases: Mapping[str, int] | None = None,
    embedded_choice_block: bool = False,
) -> tuple[list[Question], JsonObject]:
    accepted: list[Question] = []
    rejected: Counter[str] = Counter()
    identifiers: set[str] = set()
    for index, raw_row in enumerate(rows):
        row = object_value(raw_row, name=f"{dataset}[{index}]")
        if not _partition_filter(dataset, row):
            rejected["outside requested partition"] += 1
            continue
        try:
            question = normalize_row(
                row,
                dataset=dataset,
                source_split=source_split,
                index=index,
                gold_index_bases=gold_index_bases,
                embedded_choice_block=embedded_choice_block,
            )
        except DatasetNormalizationError as exc:
            rejected[str(exc)] += 1
            continue
        if question.id in identifiers:
            rejected["duplicate source identifier"] += 1
            continue
        identifiers.add(question.id)
        accepted.append(question)
    return accepted, {
        "dataset": dataset,
        "normalized": len(accepted),
        "rejected": sum(rejected.values()),
        "rejection_reasons": dict(rejected.most_common()),
    }


def load_hf_dataset(name: str) -> tuple[list[Question], JsonObject]:
    """Load one exact, pinned upstream dataset without scientific fallbacks."""
    try:
        spec = DATASET_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset: {name}") from exc
    try:
        loader_value: object = getattr(import_module("datasets"), "load_dataset", None)
        if not _is_dataset_loader(loader_value):
            raise RuntimeError("datasets.load_dataset is unavailable")
        loader = loader_value
        dataset = loader(
            spec.repository,
            spec.config,
            split=spec.split,
            revision=spec.revision,
        )
    except Exception as exc:  # external loader errors must retain exact source context
        raise RuntimeError(
            f"failed to load pinned dataset {spec.repository}@{spec.revision} "
            f"config={spec.config!r} split={spec.split!r}: {exc}"
        ) from exc
    questions, audit = _normalize_rows(
        dataset,
        dataset=name,
        source_split=spec.source_split,
        gold_index_bases=dict(spec.gold_index_bases),
        embedded_choice_block=spec.embedded_choice_block,
    )
    audit["source"] = spec.to_dict()
    return questions, audit


def _deduplicate_and_sample(
    candidates: Sequence[Question],
    *,
    count: int,
    seed: int,
    dataset: str,
    exclusions: QuestionExclusions | None = None,
    reserved_stems: frozenset[str] = frozenset(),
) -> tuple[list[Question], int, int, int]:
    if count < 1:
        raise ValueError("questions_per_dataset must be positive")
    by_stem: dict[str, Question] = {}
    duplicate_stems = 0
    for question in candidates:
        key = normalize_text(question.stem)
        if key in by_stem:
            duplicate_stems += 1
            continue
        by_stem[key] = question
    unique = list(by_stem.values())
    after_previous = (
        [question for question in unique if not exclusions.contains(question)]
        if exclusions is not None
        else unique
    )
    excluded_previous = len(unique) - len(after_previous)
    eligible = [
        question
        for question in after_previous
        if normalize_text(question.stem) not in reserved_stems
    ]
    excluded_cross_dataset = len(after_previous) - len(eligible)
    if len(eligible) < count:
        raise RuntimeError(
            f"dataset {dataset} has only {len(eligible)} eligible unused unique questions "
            f"after excluding {excluded_previous} previously used and "
            f"{excluded_cross_dataset} cross-dataset duplicates; requested {count}"
        )
    rng = random.Random(f"{seed}:{dataset}")
    rng.shuffle(eligible)
    return eligible[:count], duplicate_stems, excluded_previous, excluded_cross_dataset


def load_local_jsonl(path: Path) -> tuple[list[Question], JsonObject]:
    """Read and strictly validate normalized or raw local rows."""
    rows: list[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value: object = json.loads(line)
            try:
                row = object_value(value, name=f"line {line_number}")
            except ValueError as exc:
                raise ValueError(f"line {line_number} is not a JSON object") from exc
            rows.append(row)
    grouped: dict[str, list[Question]] = {}
    audits: list[JsonObject] = []
    for dataset in sorted({str(row.get("dataset", "local")) for row in rows}):
        subset = [row for row in rows if str(row.get("dataset", "local")) == dataset]
        normalized: list[Question] = []
        rejected: Counter[str] = Counter()
        for index, row in enumerate(subset):
            try:
                if all(
                    field in row
                    for field in (
                        "id",
                        "dataset",
                        "stem",
                        "choices",
                        "gold",
                        "gold_text",
                        "gold_source",
                        "gold_raw",
                        "source_split",
                    )
                ):
                    question = Question.from_dict(row)
                else:
                    question = normalize_row(
                        row,
                        dataset=dataset,
                        source_split=str(row.get("source_split", "local")),
                        index=index,
                        gold_index_bases=DEFAULT_GOLD_INDEX_BASES,
                    )
            except (DatasetNormalizationError, ValueError, TypeError) as exc:
                rejected[str(exc)] += 1
                continue
            normalized.append(question)
        grouped[dataset] = normalized
        audits.append(
            {
                "dataset": dataset,
                "normalized": len(normalized),
                "rejected": sum(rejected.values()),
                "rejection_reasons": dict(rejected.most_common()),
            }
        )
    audit = json_object(
        {
            "kind": "jsonl",
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "datasets": audits,
        },
        path="local dataset audit",
    )
    return [question for values in grouped.values() for question in values], audit


def select_questions(
    *,
    dataset_names: Sequence[str],
    questions_per_dataset: int,
    seed: int,
    input_jsonl: Path | None = None,
    exclusions: QuestionExclusions | None = None,
    exclusion_identity: JsonObject | None = None,
    exclusion_audit: Sequence[JsonObject] = (),
) -> tuple[list[Question], JsonObject]:
    """Load, filter, deduplicate, and deterministically sample exact dataset counts."""
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("dataset names must not be repeated")
    loading_audits: list[JsonObject]
    if input_jsonl is not None:
        all_questions, source_audit = load_local_jsonl(input_jsonl)
        grouped: dict[str, list[Question]] = {
            name: [question for question in all_questions if question.dataset == name]
            for name in dataset_names
        }
        missing = [name for name, values in grouped.items() if not values]
        if missing:
            raise RuntimeError("local input contains no questions for: " + ", ".join(missing))
        source = source_descriptor(dataset_names, input_jsonl)
        loading_audits = []
        loading_value = source_audit.get("datasets")
        if loading_value is not None:
            loading_sequence = object_sequence(loading_value)
            if loading_sequence is None:
                raise RuntimeError("local dataset audit has a non-array datasets field")
            loading_audits.extend(
                json_object(item, path=f"local dataset audit.datasets[{index}]")
                for index, item in enumerate(loading_sequence)
            )
    else:
        grouped = {}
        loading_audits = []
        for name in dataset_names:
            questions, audit = load_hf_dataset(name)
            grouped[name] = questions
            loading_audits.append(audit)
        source = source_descriptor(dataset_names, None)

    selected: list[Question] = []
    selection_audits: list[JsonObject] = []
    seen_ids: set[str] = set()
    selected_stems: set[str] = set()
    for name in dataset_names:
        (
            sample,
            duplicate_stems,
            excluded_previous,
            excluded_cross_dataset,
        ) = _deduplicate_and_sample(
            grouped[name],
            count=questions_per_dataset,
            seed=seed,
            dataset=name,
            exclusions=exclusions,
            reserved_stems=frozenset(selected_stems),
        )
        for question in sample:
            if question.id in seen_ids:
                raise RuntimeError(
                    f"duplicate normalized question id across datasets: {question.id}"
                )
            seen_ids.add(question.id)
            selected_stems.add(normalize_text(question.stem))
        selected.extend(sample)
        unique_stems = len(grouped[name]) - duplicate_stems
        selection_audits.append(
            {
                "dataset": name,
                "normalized_candidates": len(grouped[name]),
                "unique_stems": unique_stems,
                "previously_used_removed": excluded_previous,
                "cross_dataset_duplicates_removed": excluded_cross_dataset,
                "eligible_unused": unique_stems - excluded_previous - excluded_cross_dataset,
                "selected": len(sample),
                "selected_ids": [question.id for question in sample],
            }
        )
    audit = json_object(
        {
            "source": source,
            "selection": {
                "datasets": list(dataset_names),
                "questions_per_dataset": questions_per_dataset,
                "seed": seed,
                "algorithm": SELECTION_ALGORITHM,
                "question_exclusions": exclusion_identity or {"sources": []},
            },
            "loading": loading_audits,
            "exclusion_sources": list(exclusion_audit),
            "datasets": selection_audits,
            "total_selected": len(selected),
        },
        path="dataset selection audit",
    )
    return selected, audit
