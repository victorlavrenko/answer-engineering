from __future__ import annotations

from pathlib import Path

import pytest

from instruction_duplication.lexical import build_reference, compile_reference
from instruction_duplication.types import Question


@pytest.fixture
def question() -> Question:
    return Question(
        id="q1",
        dataset="test",
        stem="A patient has sudden left hearing loss for 2 hours and no ear pain.",
        choices={
            "A": "Sudden sensorineural hearing loss",
            "B": "Cerumen impaction",
            "C": "Otitis externa",
            "D": "Otosclerosis",
        },
        gold="A",
        gold_text="Sudden sensorineural hearing loss",
        gold_source="fixture",
        gold_raw="A",
        source_split="test",
    )


@pytest.fixture
def lexical_reference(question: Question):
    return compile_reference(build_reference([question]))


@pytest.fixture
def local_jsonl(tmp_path: Path, question: Question) -> Path:
    path = tmp_path / "questions.jsonl"
    import json

    rows = [
        question.to_dict(),
        {
            **question.to_dict(),
            "id": "q2",
            "dataset": "other",
            "stem": "A second unique stem with right-sided pain.",
            "gold_raw": "A",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path
