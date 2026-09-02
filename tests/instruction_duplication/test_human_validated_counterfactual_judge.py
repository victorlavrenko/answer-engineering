"""Regression tests derived from the blinded human-validation failure modes."""

from __future__ import annotations

from instruction_duplication.judge import _case_specific_counterfactual
from instruction_duplication.types import Question


def q(stem: str, choices: dict[str, str]) -> Question:
    # Production Question objects require labels to be contiguous from A.  The
    # human-audit regression examples intentionally retain their original option
    # letters (for example F/H), so fill gaps with inert distractors rather than
    # renumbering the audited labels.
    highest = max(ord(label) for label in choices)
    expanded = {
        chr(code): choices.get(chr(code), f"Unused distractor {chr(code)}")
        for code in range(ord("A"), highest + 1)
    }
    gold = next(iter(choices))
    return Question(
        id="human-validation-regression",
        dataset="test",
        stem=stem,
        choices=expanded,
        gold=gold,
        gold_text=expanded[gold],
        gold_source="fixture",
        gold_raw=gold,
        source_split="test",
    )


def test_natural_diagnostic_shift_counts() -> None:
    question = q(
        "Colonoscopy shows continuous inflammation confined to the colon.",
        {"A": "Ulcerative colitis", "D": "Crohn's disease"},
    )
    text = (
        "If colonoscopy instead showed skip lesions or small-bowel involvement, "
        "this would strongly suggest Crohn's disease over ulcerative colitis."
    )
    assert _case_specific_counterfactual(question, text, "D") is True


def test_two_co_winners_in_same_hypothetical_do_not_count() -> None:
    question = q(
        "A pitcher feels a pop during acceleration.",
        {"F": "Release phase", "H": "Deceleration phase"},
    )
    text = (
        "If the popping sensation occurred immediately after the ball was released, "
        "then F. Release phase or H. Deceleration phase would become the best answer. "
        "The altered timing changes the likely phase of injury."
    )
    assert _case_specific_counterfactual(question, text, "H") is False


def test_change_answer_to_favor_declared_alternative_counts() -> None:
    question = q(
        "Sudden headache and neck pain with a normal initial CT and no focal deficits.",
        {"F": "CT angiogram of the head and neck", "G": "Observation"},
    )
    text = (
        "What would change the answer to favor the best alternative (F. CT angiogram of the "
        "head and neck) would be new focal neurological deficits or a family history of aneurysm. "
        "If the patient developed focal signs, urgent vascular evaluation would be warranted."
    )
    assert _case_specific_counterfactual(question, text, "F") is True


def test_rewriting_what_question_asks_does_not_count() -> None:
    question = q(
        "A patient has clinical diphtheria. Which statement is true?",
        {"B": "It is caused by gram positive bacilli", "D": "Palatal paralysis is recognized"},
    )
    text = (
        "If the question were changed to: 'Which of the following microbiological statements "
        "about diphtheria is true?', then B would become the best answer because the organism "
        "is a gram-positive bacillus."
    )
    assert _case_specific_counterfactual(question, text, "B") is False


def test_rewriting_an_answer_choice_does_not_count() -> None:
    question = q(
        "A patient has clinical diphtheria. Which statement is true?",
        {"B": "It is caused by gram positive bacilli", "D": "Palatal paralysis is recognized"},
    )
    text = (
        "If option D had said that palatal paralysis occurs in every case, D would be false and "
        "B would become the best answer because B would then be the only accurate option."
    )
    assert _case_specific_counterfactual(question, text, "B") is False


def test_question_wording_can_introduce_a_real_case_fact() -> None:
    question = q(
        "An immediate hemolytic transfusion reaction occurs minutes after transfusion.",
        {"A": "ABO incompatibility", "D": "Bacterial contamination"},
    )
    text = (
        "If the question specified a delayed haemolytic reaction instead of an immediate one, "
        "D. Bacterial contamination could become the best answer because delayed reactions can "
        "arise from bacterial growth in stored blood."
    )
    assert _case_specific_counterfactual(question, text, "D") is True


def test_changing_a_clinical_finding_counts_even_if_phrased_as_question_specified() -> None:
    question = q(
        "The patient has necrolytic migratory erythema.",
        {"A": "Glucagonoma", "D": "Celiac disease"},
    )
    text = (
        "If the question specified dermatitis herpetiformis instead of necrolytic migratory "
        "erythema, the best answer would shift to D. Celiac disease because dermatitis "
        "herpetiformis is its classic skin manifestation."
    )
    assert _case_specific_counterfactual(question, text, "D") is True


def test_wrong_winning_option_does_not_satisfy_declared_alternative() -> None:
    question = q(
        "An older patient has giant-cell-arteritis features.",
        {
            "B": "Atherosclerotic peripheral artery disease",
            "J": "Takayasu arteritis",
        },
    )
    text = (
        "If the patient were a 42-year-old woman with pulse deficits, the diagnosis would shift "
        "toward J. Takayasu arteritis and Takayasu would become the more likely diagnosis."
    )
    assert _case_specific_counterfactual(question, text, "B") is False


def test_best_alternative_anaphor_counts_with_concrete_case_change() -> None:
    question = q(
        "Several relatives are present, but the usual caregiver has the most recent information.",
        {"A": "His mother", "C": "His father", "E": "His friend"},
    )
    text = (
        "The smallest change that would make the best alternative (father) the best answer is "
        "specifying that the father has been the patient's primary caregiver for the past year "
        "and that the mother has not been involved in his recent care."
    )
    assert _case_specific_counterfactual(question, text, "C") is True


def test_prior_attempt_can_be_named_as_strongest_remaining_risk_factor() -> None:
    question = q(
        "A suicidal patient owns an accessible firearm and has a previous suicide attempt.",
        {"A": "Access to a firearm", "F": "Previous attempt"},
    )
    text = (
        "If the patient had no access to a firearm because it had been sold or secured elsewhere, "
        "then the previous attempt (F) would become the most significant remaining risk factor. "
        "Removing access to the lethal means changes which factor is strongest."
    )
    assert _case_specific_counterfactual(question, text, "F") is True


def test_more_relevant_without_winning_does_not_count() -> None:
    question = q(
        "An immediate transfusion reaction follows incompatible blood.",
        {"A": "ABO incompatibility", "D": "Bacterial contamination"},
    )
    text = (
        "If the reaction were delayed by several days, bacterial contamination might become more "
        "relevant. However, this does not establish it as the best answer."
    )
    assert _case_specific_counterfactual(question, text, "D") is False


def test_generic_counterfactual_boilerplate_still_fails() -> None:
    question = q("Sudden unilateral hearing loss without ear pain.", {"A": "SNHL", "B": "Cerumen"})
    text = (
        "If a different finding were present, the other option would become the best answer "
        "because it would fit better."
    )
    assert _case_specific_counterfactual(question, text, "B") is False
