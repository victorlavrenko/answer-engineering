from __future__ import annotations

import pytest

from instruction_duplication.trajectory import recover_protocol

CHOICES = {"A": "Alpha answer", "B": "Beta answer"}


def response_with_step6_heading(heading: str) -> str:
    return f"""1. Facts
A concrete fact is present.

2. Implications
The fact supports Alpha.

3. Provisional answer
A. Alpha answer is provisionally selected because it fits.

4. Best alternative
B. Beta answer is the best alternative because it is less suitable.

5. Decisive distinction
The decisive distinction favors Alpha.

{heading}
A meaningful hypothetical change is discussed here with enough generated content to be non-trivial.

7. Reconsideration
I retain A because the original facts still favor Alpha.

8. Final answer
A. Alpha answer
"""


@pytest.mark.parametrize(
    "heading",
    (
        # Observed misses in the 2026-08-12 generations.
        "### **6. What Would Change the Answer to A (Head CT)?**",
        "### **6. What Would Change the Answer to J?**",
        "### **6. What would change the answer to I (TIA)?**",
        "6. What would change the question",
        # Canonical and natural synonyms.
        "6. What would change the answer",
        "6. Answer change",
        "6. Answer-changing scenario",
        "6. Counterfactual",
        "6. Counterfactual scenario",
        "6. What change would alter the answer?",
        "6. What change would switch the answer to option B?",
        "6. How could the answer change?",
        "6. What would make the best alternative win?",
        "6. What would make option B correct?",
        "6. Change needed to alter the answer",
        "6. Change required for the best alternative to become best",
        "6. Change that would make the alternative win",
        "6. Conditions for the best alternative to win",
        "6. When would option B be correct?",
        "6. How to make the alternative correct",
        # Wider wording is allowed only because explicit Step 6 is present.
        "6. What would change?",
        "6. What needs to change?",
        "6. What would change the answer to Head CT",
        "Step 6: What would make Head CT the preferred answer",
        # Strong unnumbered semantic headings remain acceptable.
        "### What would make the best alternative win?",
        "### Counterfactual scenario",
    ),
)
def test_step6_natural_heading_variants_are_recovered(heading: str) -> None:
    recovered = recover_protocol(response_with_step6_heading(heading), CHOICES)
    assert recovered.semantic_start_counts["answer_changing_change"] == 1
    assert "meaningful hypothetical change" in recovered.sections["answer_changing_change"]


@pytest.mark.parametrize(
    "heading",
    (
        "6. Provisional answer",
        "6. Alternative diagnosis",
        "6. What would change the treatment?",
        "5. What would change?",
        "What would change?",
        "What would change the treatment?",
        "The answer would change if the patient developed fever.",
    ),
)
def test_step6_recognizer_does_not_accept_unrelated_or_body_text(heading: str) -> None:
    recovered = recover_protocol(response_with_step6_heading(heading), CHOICES)
    assert recovered.semantic_start_counts["answer_changing_change"] == 0
    assert recovered.sections["answer_changing_change"] == ""


def test_numbered_fallback_does_not_double_count_canonical_heading() -> None:
    recovered = recover_protocol(
        response_with_step6_heading("6. What would change the answer"),
        CHOICES,
    )
    assert recovered.semantic_start_counts["answer_changing_change"] == 1
