"""Cross-run question exclusion for independent confirmatory replications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .answer_utils import ChoiceNormalizationError, canonicalize_choices, normalize_text
from .io_utils import read_jsonl, sha256_file
from .json_types import JsonObject, json_object
from .question_text import strip_exact_embedded_choice_suffix
from .types import Question

_QUESTION_PATHS = (
    Path("data/questions.jsonl"),
    Path("questions.jsonl"),
    Path(".instruction-duplication/data/questions.jsonl"),
    Path(".instruction-duplication/questions.jsonl"),
)


@dataclass(frozen=True, slots=True)
class QuestionExclusions:
    """Identifiers and normalized stems already used in previous runs."""

    ids: frozenset[str]
    stems: frozenset[str]

    def contains(self, question: Question) -> bool:
        """Return whether a candidate was already used by source ID or exact stem."""
        return question.id in self.ids or normalize_text(question.stem) in self.stems


def _questions_file(workspace: Path) -> Path:
    if workspace.is_file():
        if workspace.name != "questions.jsonl":
            raise ValueError(f"question-exclusion file must be named questions.jsonl: {workspace}")
        return workspace
    for relative in _QUESTION_PATHS:
        candidate = workspace / relative
        if candidate.is_file():
            return candidate
    checked = ", ".join(str(workspace / relative) for relative in _QUESTION_PATHS)
    raise RuntimeError(f"could not find questions.jsonl in exclusion workspace; checked {checked}")


def _row_identity(row: Mapping[str, object], *, source: Path, index: int) -> tuple[str, str, str]:
    identifier = str(row.get("id", "")).strip()
    dataset = str(row.get("dataset", "")).strip()
    stem = str(row.get("stem", "")).strip()
    if not identifier or not dataset or not stem:
        raise ValueError(
            f"{source}:{index}: exclusion rows require non-empty id, dataset, and stem"
        )
    return identifier, dataset, stem


def _comparison_stem(row: Mapping[str, object], stem: str) -> str:
    """Normalize a historical ledger stem for cross-version content exclusion."""
    raw_choices = row.get("choices")
    if raw_choices in (None, "", [], {}):
        return stem
    try:
        choices = canonicalize_choices(raw_choices)
    except (ChoiceNormalizationError, TypeError, ValueError):
        return stem
    cleaned, _removed = strip_exact_embedded_choice_suffix(stem, choices)
    return cleaned


def load_question_exclusions(
    workspaces: tuple[Path, ...],
) -> tuple[QuestionExclusions, JsonObject, list[JsonObject]]:
    """Load prior selected questions without requiring their software version.

    Only the immutable question ledger is consumed.  Old manifests, databases,
    judge versions, and package layouts are intentionally irrelevant to exclusion.
    """
    ids: set[str] = set()
    stems: set[str] = set()
    identities: list[JsonObject] = []
    audits: list[JsonObject] = []
    seen_hashes: set[str] = set()
    for raw_workspace in workspaces:
        workspace = raw_workspace.resolve()
        questions_file = _questions_file(workspace)
        digest = sha256_file(questions_file)
        if digest in seen_hashes:
            raise ValueError(f"duplicate exclusion question ledger: {questions_file}")
        seen_hashes.add(digest)
        rows = read_jsonl(questions_file)
        before_ids = len(ids)
        before_stems = len(stems)
        datasets: set[str] = set()
        for index, row in enumerate(rows, 1):
            identifier, dataset, stem = _row_identity(row, source=questions_file, index=index)
            ids.add(identifier)
            stems.add(normalize_text(_comparison_stem(row, stem)))
            datasets.add(dataset)
        identity: JsonObject = {
            "questions_sha256": digest,
            "question_rows": len(rows),
        }
        identities.append(identity)
        audits.append(
            json_object(
                {
                    **identity,
                    "workspace": str(workspace),
                    "questions_file": str(questions_file),
                    "datasets": sorted(datasets),
                    "new_ids": len(ids) - before_ids,
                    "new_stems": len(stems) - before_stems,
                },
                path=f"question exclusion audit for {workspace}",
            )
        )
    identities.sort(key=lambda item: str(item["questions_sha256"]))
    return (
        QuestionExclusions(frozenset(ids), frozenset(stems)),
        json_object(
            {
                "sources": identities,
                "unique_ids": len(ids),
                "unique_stems": len(stems),
            },
            path="question exclusion identity",
        ),
        audits,
    )
