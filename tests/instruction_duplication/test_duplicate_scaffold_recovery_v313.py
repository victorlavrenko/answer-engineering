from __future__ import annotations

from instruction_duplication.protocol import SECTION_GUIDANCE
from instruction_duplication.trajectory import recover_protocol

CHOICES = {
    "A": "Conservative management",
    "B": "Controlled delivery",
}


def heading(index: int, title: str, role: str) -> str:
    return f"## {index}. {title} — {SECTION_GUIDANCE[role]}"


def test_empty_template_followed_by_filled_suffix_recovers_filled_suffix() -> None:
    raw = f"""## 1. Facts — {SECTION_GUIDANCE["facts"]}

The patient has several concrete findings that matter.

## 2. Implications — {SECTION_GUIDANCE["implications"]}

Those findings support a controlled intervention rather than conservative management.

{heading(3, "Provisional answer", "provisional_answer")}

{heading(4, "Best alternative", "second_best")}

{heading(5, "Decisive distinction", "decisive_fact")}

{heading(6, "What would change the answer", "answer_changing_change")}

{heading(7, "Reconsideration", "rereasoning")}

{heading(8, "Final answer", "final_answer")}

A short bridge sentence explains that the model will now give the actual decisions.

## Provisional answer
B. Controlled delivery is provisionally selected because the findings favor intervention.

## 4. Best alternative
A. Conservative management is the best alternative but is less suitable here.

## 5. Decisive distinction
The decisive distinction is the current high-risk finding.

## 6. What would change the answer
If that high-risk finding were absent, conservative management would become preferable.

## 7. Reconsideration
I retain B because the original high-risk finding is still present.

## 8. Final answer
B. Controlled delivery
"""
    recovered = recover_protocol(raw, CHOICES)

    assert recovered.semantic_provisional_option == "B"
    assert recovered.semantic_second_best_option == "A"
    assert recovered.semantic_final_option == "B"

    assert "provisionally selected" in recovered.sections["provisional_answer"]
    assert "best alternative but is less suitable" in recovered.sections["second_best"]
    assert "high-risk finding were absent" in recovered.sections["answer_changing_change"]
    assert "retain B" in recovered.sections["rereasoning"]

    assert "A short bridge sentence" not in recovered.sections["implications"]
    assert "Provisional answer" not in recovered.sections["implications"]

    for role in (
        "provisional_answer",
        "second_best",
        "decisive_fact",
        "answer_changing_change",
        "rereasoning",
        "final_answer",
    ):
        assert recovered.semantic_start_counts[role] == 2


def test_filled_first_scaffold_is_not_displaced_by_later_empty_template() -> None:
    raw = f"""## 1. Facts
A concrete finding is present.

## 2. Implications
That finding supports option B.

## 3. Provisional answer
B. Controlled delivery is the best answer because the finding supports it.

## 4. Best alternative
A. Conservative management is the best alternative but is less suitable.

## 5. Decisive distinction
The decisive distinction is the high-risk finding.

## 6. What would change the answer
If the high-risk finding disappeared, A would become preferable.

## 7. Reconsideration
I retain B because the original finding remains present.

## 8. Final answer
B. Controlled delivery

{heading(3, "Provisional answer", "provisional_answer")}

{heading(4, "Best alternative", "second_best")}

{heading(5, "Decisive distinction", "decisive_fact")}

{heading(6, "What would change the answer", "answer_changing_change")}

{heading(7, "Reconsideration", "rereasoning")}

{heading(8, "Final answer", "final_answer")}
"""
    recovered = recover_protocol(raw, CHOICES)

    assert recovered.semantic_provisional_option == "B"
    assert recovered.semantic_second_best_option == "A"
    assert recovered.semantic_final_option == "B"
    assert "high-risk finding disappeared" in recovered.sections["answer_changing_change"]


def test_single_scaffold_recovery_is_unchanged() -> None:
    raw = """## 1. Facts
A concrete finding is present.

## 2. Implications
That finding supports B.

## 3. Provisional answer
B. Controlled delivery is the best answer.

## 4. Best alternative
A. Conservative management is the best alternative.

## 5. Decisive distinction
The high-risk finding distinguishes B from A.

## 6. What would change the answer
Removing the high-risk finding would make A preferable.

## 7. Reconsideration
I retain B because the finding remains.

## 8. Final answer
B. Controlled delivery
"""
    recovered = recover_protocol(raw, CHOICES)
    assert recovered.semantic_provisional_option == "B"
    assert recovered.semantic_second_best_option == "A"
    assert recovered.semantic_final_option == "B"
    assert "Removing the high-risk finding" in recovered.sections["answer_changing_change"]
