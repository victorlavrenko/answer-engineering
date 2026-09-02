from __future__ import annotations

import pytest

from instruction_duplication.judge import _commitments, extract_protocol_final_answer, judge
from instruction_duplication.lexical import build_reference, compile_reference
from instruction_duplication.protocol import SECTION_GUIDANCE
from instruction_duplication.provider import fake_response
from instruction_duplication.records import GenerationCell
from instruction_duplication.trajectory import recover_protocol
from instruction_duplication.types import Question


def valid_response(question):
    cell = GenerationCell(
        cell_id="a" * 64,
        question_id=question.id,
        model_id="test-model",
        condition_id="system",
        copies=1,
        dataset=question.dataset,
        stem=question.stem,
        choices=question.choices,
        gold=question.gold,
    )
    return fake_response(cell)["choices"][0]["message"]["content"]


def test_valid_protocol_gets_complete_semantic_roles(question, lexical_reference):
    result = judge(question, "completed", valid_response(question), "system", lexical_reference)
    assert result["trajectory_complete"] is True
    assert result["identified_role_count"] == 8
    assert result["unique_role_count"] == 8
    assert result["substantive_role_count"] == 8
    assert result["role_completeness_score"] == 1.0
    assert result["all_roles_substantive"] == 1.0
    assert result["roles_in_requested_order"] == 1.0
    assert result["complete_role_scaffold"] == 1.0
    assert result["preanswer_discussion_complete"] == 1.0
    assert result["accuracy"] == 1


@pytest.mark.parametrize("fence", ("```", "```text", "```markdown", "```xml"))
def test_whole_response_code_fence_is_presentation_only(question, lexical_reference, fence):
    raw = fence + "\n" + valid_response(question) + "\n```"
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["substantive_role_count"] == 8
    assert result["complete_role_scaffold"] == 1.0
    assert result["accuracy_parseable"] is True


def test_markdown_alias_headings_receive_full_semantic_credit(question, lexical_reference):
    raw = valid_response(question)
    replacements = {
        "1. Facts": "## Step 1: Case Facts",
        "2. Implications": "## Step 2: Interpretation",
        "3. Provisional answer": "## Step 3: Initial Answer",
        "4. Best alternative": "## Step 4: Second-Best Answer",
        "5. Decisive distinction": "## Step 5: Key Distinction",
        "6. What would change the answer": "## Step 6: Counterfactual Change",
        "7. Reconsideration": "## Step 7: Re-evaluation",
        "8. Final answer": "## Step 8: Final Answer",
    }
    for source, target in replacements.items():
        raw = raw.replace(source, target)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["identified_role_count"] == 8
    assert result["substantive_role_count"] == 8
    assert result["complete_role_scaffold"] == 1.0
    assert result["accuracy"] == 1


def test_same_line_plain_headings_are_segmented(question, lexical_reference):
    raw = valid_response(question)
    for heading in (
        "Facts",
        "Implications",
        "Provisional answer",
        "Best alternative",
        "Decisive distinction",
        "What would change the answer",
        "Reconsideration",
        "Final answer",
    ):
        raw = raw.replace(f". {heading}\n", f". {heading}: ", 1)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["substantive_role_count"] == 8
    assert result["complete_role_scaffold"] == 1.0


def test_voluntary_xml_is_accepted_but_not_required(question, lexical_reference):
    raw = valid_response(question)
    replacements = {
        "1. Facts": "<facts>",
        "2. Implications": "<implications>",
        "3. Provisional answer": '<provisional_answer option="A">',
        "4. Best alternative": '<second_best option="B">',
        "5. Decisive distinction": "<decisive_fact>",
        "6. What would change the answer": "<answer_changing_change>",
        "7. Reconsideration": '<rereasoning decision="retain">',
        "8. Final answer": '<final_answer option="A">',
    }
    for source, target in replacements.items():
        raw = raw.replace(source, target)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["substantive_role_count"] == 8
    assert result["complete_role_scaffold"] == 1.0
    assert result["section_schema_valid_count"] == 8
    assert result["accuracy"] == 1


def test_reversed_role_tag_is_still_an_identifiable_boundary(question, lexical_reference):
    raw = valid_response(question).replace("2. Implications", "</implications>", 1)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["identified_role_count"] == 8
    assert result["substantive_role_count"] == 8
    assert result["complete_role_scaffold"] == 1.0


def test_missing_role_heading_loses_presence_and_content_credit(question, lexical_reference):
    raw = valid_response(question).replace("3. Provisional answer\n", "", 1)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["identified_role_count"] == 7
    assert result["substantive_role_count"] < 8
    assert result["section_present"]["provisional_answer"] is False
    assert result["section_substantive"]["provisional_answer"] is False
    assert result["complete_role_scaffold"] == 0.0
    assert result["trajectory_complete"] is False


def test_duplicate_role_heading_is_reported_without_erasing_other_roles(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        "5. Decisive distinction\n",
        "5. Decisive distinction\n5. Decisive distinction: Additional distinction.\n",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["identified_role_count"] == 8
    assert result["unique_role_count"] == 7
    assert result["complete_role_scaffold"] == 0.0
    assert any(
        "duplicate reasoning role: decisive_fact" in error for error in result["trajectory_errors"]
    )


def test_exact_protocol_guidance_echo_is_removed_without_erasing_content(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        question.stem,
        SECTION_GUIDANCE["facts"] + " " + question.stem,
        1,
    )
    recovered = recover_protocol(raw, question.choices)
    assert recovered.sections["facts"] == question.stem
    assert (
        judge(question, "completed", raw, "system", lexical_reference)["preanswer_tfidf_recall"] > 0
    )


def test_reconsideration_must_say_retain_or_revise(question, lexical_reference):
    raw = valid_response(question).replace(
        "Retain option A after checking the stem again",
        "Review option A after checking the stem again",
    )
    recovered = recover_protocol(raw, question.choices)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert recovered.semantic_decision is None
    assert any("reconsideration" in error for error in recovered.errors)
    assert result["role_reconsideration_complete"] == 0.0
    assert result["trajectory_complete"] is False


def test_early_commitment_is_detected_and_kept_separate_from_role_count(
    question, lexical_reference
):
    raw = valid_response(question).replace("1. Facts\n", "1. Facts\nThe answer is option A. ", 1)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_itt"] == 1.0
    assert result["substantive_role_count"] == 8
    assert result["minimum_repair_scaffold"] == 0.0
    assert result["early_commitments"]


def test_generation_failure_has_itt_values(question, lexical_reference):
    result = judge(question, "truncated", None, "system", lexical_reference)
    assert result["generation_usable"] is False
    assert result["minimum_repair_scaffold"] == 0.0
    assert result["preprovisional_commitment_itt"] == 1.0
    assert result["accuracy"] == 0


def test_baseline_keeps_repair_metrics_undefined(question, lexical_reference):
    result = judge(question, "completed", "Final answer: A", "zero", lexical_reference)
    assert result["accuracy"] == 1
    assert result["minimum_repair_scaffold"] is None


def test_topic_only_stem_keeps_accuracy_but_excludes_repair_endpoints() -> None:
    question = Question(
        id="topic",
        dataset="test",
        stem="Concerning dopamine stress echocardiography, which is true?",
        choices={"A": "One", "B": "Two"},
        gold="A",
        gold_text="One",
        gold_source="fixture",
        gold_raw="A",
        source_split="test",
    )
    reference = compile_reference(build_reference([question]))
    result = judge(
        question,
        "completed",
        valid_response(question),
        "system",
        reference,
    )
    assert result["repair_endpoint_eligible"] is False
    assert result["minimum_repair_scaffold"] is None
    assert result["preanswer_tfidf_recall"] is None
    assert result["accuracy"] == 1


def test_baseline_explicit_answer_cue_accepts_lowercase_label(question, lexical_reference):
    result = judge(
        question,
        "completed",
        "**Final Answer: a. Sudden sensorineural hearing loss**",
        "zero",
        lexical_reference,
    )
    assert result["accuracy_parseable"] is True
    assert result["final_option"] == "A"
    assert result["accuracy"] == 1


def test_final_answer_text_variation_does_not_change_unambiguous_option(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        f"Option {question.gold}: {question.gold_text}",
        f"Option {question.gold}: {question.gold_text}!!!",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["trajectory_complete"] is True
    assert result["minimum_repair_scaffold"] == 1.0
    assert result["accuracy"] == 1


def test_conflicting_final_body_invalidates_accuracy(question, lexical_reference):
    other = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"8. Final answer\nOption {question.gold}: {question.gold_text}",
        f'<final_answer option="{question.gold}">\nOption {other}: {question.choices[other]}',
    )
    answer = extract_protocol_final_answer(raw, question.choices)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert answer.status == "conflicting"
    assert result["accuracy_parseable"] is False
    assert result["accuracy"] == 0
    assert result["section_substantive"]["final_answer"] is False


def test_lowercase_article_is_not_early_option_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n", "1. Facts\nThe answer is a treatment category under review. ", 1
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_itt"] == 0.0


def test_selection_verb_before_lowercase_article_is_not_option_commitment(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        "1. Facts\n", "1. Facts\nThe findings favor a benign diagnosis. ", 1
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 0.0


def test_conditional_answer_is_not_current_commitment(question, lexical_reference):
    other = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        "1. Facts\n",
        f"1. Facts\nIf the key fact were absent, the correct answer would be {other}. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 0.0


def test_explicit_most_likely_option_is_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n", "1. Facts\nOption A is the most likely candidate for being false. ", 1
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0


def test_separate_role_and_coverage_components_are_bounded(question, lexical_reference):
    result = judge(question, "completed", valid_response(question), "system", lexical_reference)
    for key in (
        "role_completeness_score",
        "all_roles_substantive",
        "roles_in_requested_order",
        "complete_role_scaffold",
        "protocol_format_score",
        "structure_score",
        "section_depth_score",
        "substantive_role_score",
        "contrastive_role_score",
        "visible_substantive_role_score",
        "visible_contrastive_role_score",
        "facts_tfidf_recall",
        "implications_tfidf_recall",
        "preanswer_tfidf_recall",
        "facts_implications_shared_tfidf_recall",
        "section_present_fraction",
        "section_unique_fraction",
        "section_substantive_fraction",
        "section_schema_valid_fraction",
    ):
        value = result[key]
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0
    token_count = result["preanswer_tfidf_token_count"]
    assert isinstance(token_count, float)
    assert token_count > 0
    for key in (
        "section_present_count",
        "section_unique_count",
        "section_substantive_count",
        "section_schema_valid_count",
        "identified_role_count",
        "unique_role_count",
        "substantive_role_count",
    ):
        count = result[key]
        assert isinstance(count, int)
        assert 0 <= count <= 8


def test_option_label_supplies_identity_for_provisional_reasoning(question, lexical_reference):
    raw = valid_response(question).replace(
        f"Option {question.gold}, {question.gold_text}, is provisional because it best fits the facts.",
        f"Option {question.gold} is favored because the findings fit it best.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["provisional_answer"] is True


def test_explicit_choice_list_in_facts_is_excluded_and_fails_facts_role(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        question.stem,
        question.stem
        + " Option A describes sudden sensorineural hearing loss."
        + " Option B describes cerumen impaction.",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["facts_choice_list_leakage"] is True
    assert result["section_substantive"]["facts"] is False
    assert result["minimum_repair_scaffold"] == 0.0


def test_single_option_reference_in_facts_is_not_misclassified_as_choice_list(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        question.stem,
        question.stem + " Option A is mentioned only as a single reference.",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["facts_choice_list_leakage"] is False


def test_bare_answer_name_is_not_a_substantive_provisional_rationale(question, lexical_reference):
    raw = valid_response(question).replace(
        f"Option {question.gold}, {question.gold_text}, is provisional because it best fits the facts.",
        f"Option {question.gold}: {question.gold_text}",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["provisional_answer"] is False
    assert result["minimum_repair_scaffold"] == 0.0


def test_bare_answer_name_is_not_a_substantive_second_best_rationale(question, lexical_reference):
    other = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"Option {other}, {question.choices[other]}, is the best alternative but loses on the decisive finding.",
        f"Option {other}: {question.choices[other]}",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["second_best"] is False
    assert result["contrastive_scaffold_complete"] == 0.0


def test_cue_free_explanation_can_still_be_a_substantive_rationale(question, lexical_reference):
    raw = valid_response(question).replace(
        f"Option {question.gold}, {question.gold_text}, is provisional because it best fits the facts.",
        f"Option {question.gold}: {question.gold_text}. Progressive cochlear injury produces persistent sensorineural "
        "hearing loss requiring urgent treatment.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["provisional_answer"] is True


def test_protocol_echo_is_not_a_substantive_rationale(question, lexical_reference):
    other = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"Option {other}, {question.choices[other]}, is the best alternative but loses on the decisive finding.",
        SECTION_GUIDANCE["second_best"],
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["second_best"] is False


def test_generic_rereasoning_without_case_fact_is_not_substantive(question, lexical_reference):
    raw = valid_response(question).replace(
        f"Retain option {question.gold} after checking the stem again; the decisive evidence remains: {question.stem}",
        "After reviewing the answer, the decision is to retain the provisional answer.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["rereasoning"] is False


def test_prompt_echo_is_not_a_case_specific_answer_changing_change(question, lexical_reference):
    other = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"If the decisive finding changed to support {question.choices[other]}, option {other} would become best.",
        SECTION_GUIDANCE["answer_changing_change"].replace("Y", other),
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["answer_changing_change"] is False


def test_counterfactual_may_introduce_new_case_specific_facts(question, lexical_reference):
    other = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"If the decisive finding changed to support {question.choices[other]}, option {other} would become best.",
        "If the patient instead developed a persistent conductive air-bone gap, "
        "cerumen impaction would become the better explanation.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["answer_changing_change"] is True


def test_generic_counterfactual_boilerplate_is_not_case_specific(question, lexical_reference):
    other = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"If the decisive finding changed to support {question.choices[other]}, option {other} would become best.",
        "If a different finding were present, the other option would become the best answer because it would fit better.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["answer_changing_change"] is False


def test_merely_making_second_best_more_relevant_is_not_answer_changing(
    question, lexical_reference
):
    other = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"If the decisive finding changed to support {question.choices[other]}, option {other} would become best.",
        f"If a conductive pattern appeared, option {other} might become more relevant, "
        "but it would still not override the current answer.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["answer_changing_change"] is False


def test_reversing_question_polarity_is_not_a_case_fact_change(question, lexical_reference):
    other = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"If the decisive finding changed to support {question.choices[other]}, option {other} would become best.",
        f"If the stem asked for the least likely diagnosis, option {other} would become best. "
        "That would shift the focus from most likely to least likely.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["answer_changing_change"] is False


def test_rewriting_the_question_is_not_an_answer_changing_case_fact(question, lexical_reference):
    other = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"If the decisive finding changed to support {question.choices[other]}, option {other} would become best.",
        f"If the stem were rephrased to ask for option {other}, option {other} would become best.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["answer_changing_change"] is False


def test_counterfactual_must_target_the_declared_second_best(question, lexical_reference):
    labels = list(question.choices)
    second = next(label for label in labels if label != question.gold)
    third = next(label for label in labels if label not in {question.gold, second})
    raw = valid_response(question).replace(
        f"If the decisive finding changed to support {question.choices[second]}, option {second} would become best.",
        f"If a new conductive pattern appeared, option {third} would become best.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["answer_changing_change"] is False


def test_direct_option_commitment_without_punctuation_is_detected(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n", "1. Facts\nOption A seems most likely. ", 1
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0


def test_choice_enumeration_is_not_early_answer_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        "1. Facts\nLet's analyze each choice:\nA: one possibility.\nB: another possibility. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 0.0


def test_same_line_bare_answer_label_is_still_a_commitment(question, lexical_reference):
    raw = valid_response(question).replace("1. Facts\n", "1. Facts\nAnswer: A. ", 1)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0


def test_paraphrased_case_grounded_implication_is_substantive(question, lexical_reference):
    raw = valid_response(question).replace(
        f"2. Implications\n{question.stem} supports option {question.gold} and argues against alternatives.",
        "2. Implications\nThe presentation supports a cochlear process and argues against a mechanical "
        "external-ear explanation, so the contrast among the choices is clinically meaningful.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["implications"] is True


def test_multiple_unpunctuated_option_references_count_as_choice_list_leakage(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        question.stem,
        question.stem + " Option A describes one diagnosis while Option B describes another.",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["facts_choice_list_leakage"] is True


def test_lexical_grounding_normalizes_possessives_hyphens_and_inflections():
    from instruction_duplication.lexical import score_anchor_recall

    assert score_anchor_recall("melanoma", "melanoma's thickness") > 0
    assert score_anchor_recall("six-hour symptom", "six hour symptom") > 0
    assert score_anchor_recall("behavioral change", "behavioral changes") > 0


def test_lexical_grounding_accepts_reviewed_acronym():
    from instruction_duplication.lexical import score_anchor_recall

    assert (
        score_anchor_recall(
            "Group B streptococcus colonization in pregnancy",
            "GBS colonization is clinically important.",
        )
        > 0
    )


def test_lexical_grounding_accepts_abbreviation_explicitly_paired_in_stem():
    from instruction_duplication.lexical import score_anchor_recall

    assert score_anchor_recall("computed tomography (CT)", "CT") == 1.0


@pytest.mark.parametrize(
    ("stem", "candidate"),
    (
        ("chronic tinnitus", "CT"),
        ("complain of", "CO"),
        ("initial vital", "IV"),
        ("she takes", "ST"),
        ("morphine IV", "MI"),
    ),
)
def test_lexical_grounding_does_not_invent_abbreviations_from_initials(stem, candidate):
    from instruction_duplication.lexical import score_anchor_recall

    assert score_anchor_recall(stem, candidate) == 0.0


def test_reviewed_abbreviation_requires_its_reviewed_expansion_in_stem():
    from instruction_duplication.lexical import score_anchor_recall

    assert score_anchor_recall("myocardial infarction", "MI") == 1.0
    assert score_anchor_recall("morphine infusion", "MI") == 0.0


def test_decisive_fact_can_be_choice_grounded_with_new_explanatory_content(
    question, lexical_reference
):
    other = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"5. Decisive distinction\n{question.stem}",
        "5. Decisive distinction\n"
        f"{question.choices[question.gold]} fits the cochlear pattern, whereas "
        f"{question.choices[other]} would require a persistent conductive mechanism.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["decisive_fact"] is True


def test_long_choice_name_alone_is_not_a_decisive_fact(question, lexical_reference):
    raw = valid_response(question).replace(
        f"5. Decisive distinction\n{question.stem}",
        f"5. Decisive distinction\n{question.choices[question.gold]}",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["decisive_fact"] is False


def test_rereasoning_can_reconnect_to_prior_case_specific_trajectory(question, lexical_reference):
    raw = valid_response(question).replace(
        f"Retain option {question.gold} after checking the stem again; the decisive evidence remains: {question.stem}",
        "Retain the provisional answer after rechecking the progressive cochlear injury "
        "and persistent sensorineural pattern described above.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_substantive"]["rereasoning"] is True


def test_hypothetical_relative_clause_is_not_early_answer_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        "1. Facts\nA new laboratory finding which would favor A is absent in the actual stem. ",
        1,
    )
    result = judge(question, "completed", raw, "before", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 0.0
    assert result["early_commitments"] == []


def test_first_person_modal_selection_is_still_a_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        "1. Facts\nThe findings are mixed, but I would choose A based on the current stem. ",
        1,
    )
    result = judge(question, "completed", raw, "before", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0
    assert any(event["option"] == "A" for event in result["early_commitments"])


def test_unrecognized_provisional_heading_does_not_create_false_early_commitment(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        "3. Provisional answer",
        "3. Selection",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["trajectory_complete"] is False
    assert result["preprovisional_commitment_observed"] == 0.0


def test_choice_text_followed_by_most_likely_diagnosis_is_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        f"1. Facts\n{question.choices[question.gold]} is the most likely diagnosis. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0
    assert any(event["option"] == question.gold for event in result["early_commitments"])


def test_choice_text_as_likely_answer_is_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        f"1. Facts\nThe findings point toward {question.choices[question.gold]} as a likely answer. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0


def test_correct_answer_with_leading_article_choice_text_is_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        f"1. Facts\nThe correct answer must be the {question.choices[question.gold]}. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0


def test_option_most_suspect_is_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        "1. Facts\nOption A is the most suspect among the listed statements. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0


def test_preanswer_discussion_is_separate_from_later_protocol_failure(question, lexical_reference):
    raw = valid_response(question).replace("5. Decisive distinction\n", "", 1)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["trajectory_complete"] is False
    assert result["protocol_scaffold_complete"] == 0.0
    assert result["preanswer_discussion_complete"] == 1.0


def test_early_commitment_blocks_preanswer_discussion(question, lexical_reference):
    raw = valid_response(question).replace("1. Facts\n", "1. Facts\nAnswer: A. ", 1)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preanswer_discussion_complete"] == 0.0


def test_instructed_generation_failure_scores_all_question_compliance_zero(
    question, lexical_reference
):
    result = judge(question, "failed", None, "system", lexical_reference)
    assert result["protocol_scaffold_complete"] == 0.0
    assert result["preanswer_discussion_complete"] == 0.0


def test_option_statement_can_be_called_correct_without_answer_commitment(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        "1. Facts\n",
        "1. Facts\nOption B is correct in describing one secondary property, while the options still require comparison. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 0.0


def test_conclusive_option_correct_is_answer_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        "1. Facts\nThe alternatives fail the stem. Therefore, option A is correct. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0


def test_option_supported_as_correct_answer_is_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        "1. Facts\nThis distinction strongly supports option A as the correct answer. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0
    assert any(event["option"] == "A" for event in result["early_commitments"])


def test_unmapped_text_as_likely_answer_is_still_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        "1. Facts\nThis immediately points toward Mycoplasma pneumoniae as a likely answer. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0
    assert any(event["option"] is None for event in result["early_commitments"])


def test_bare_label_only_option_is_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        "1. Facts\nC is the only option that directly satisfies the question. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0
    assert any(event["option"] == "C" for event in result["early_commitments"])


def test_conditional_likely_answer_is_not_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        "1. Facts\nIf fever were present, option B would become a likely answer. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 0.0


def test_generic_correct_answer_criterion_is_not_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        "1. Facts\nThe correct answer must be a structure supplied by the mandibular division. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 0.0


def test_most_likely_target_text_is_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        f"1. Facts\nThe most likely clinical finding in this patient is {question.choices[question.gold]}. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0


def test_choice_made_most_likely_finding_is_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        f"1. Facts\nThe pattern favors the syndrome, making {question.choices[question.gold]} the most likely associated finding. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0


def test_only_option_label_prefix_is_commitment(question, lexical_reference):
    raw = valid_response(question).replace(
        "1. Facts\n",
        "1. Facts\nOnly option A matches the stated requirement. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["preprovisional_commitment_observed"] == 1.0


def test_unique_choice_dominance_before_provisional_is_commitment() -> None:
    choices = {"A": "first", "B": "second", "C": "third"}
    text = (
        "Only one option correctly describes the mechanism. "
        "Option B appears consistent with the stated physiology."
    )
    events = _commitments(text, choices)
    assert any(event["option"] == "B" for event in events)


def test_generic_best_choice_rule_without_identity_is_not_commitment() -> None:
    choices = {"A": "100 units", "B": "250 units", "C": "500 units"}
    text = "The option that matches the standard dose is the best choice."
    assert _commitments(text, choices) == []


def test_lexical_measurement_ignores_leading_source_item_number():
    from instruction_duplication.lexical import (
        build_reference,
        compile_reference,
        score_anchor_recall,
        score_preanswer,
    )
    from instruction_duplication.types import Question

    numbered = Question(
        id="q-numbered",
        dataset="fixture",
        stem="121 . 14-hour-old neonate is jaundiced.",
        choices={"A": "x", "B": "y"},
        gold="A",
        gold_text="x",
        gold_source="fixture",
        gold_raw="A",
        source_split="test",
    )
    plain = Question(
        id="q-plain",
        dataset="fixture",
        stem="14-hour-old neonate is jaundiced.",
        choices={"A": "x", "B": "y"},
        gold="A",
        gold_text="x",
        gold_source="fixture",
        gold_raw="A",
        source_split="test",
    )
    numbered_reference = compile_reference(build_reference([numbered]))
    plain_reference = compile_reference(build_reference([plain]))
    candidate = "The 14-hour-old neonate is jaundiced. This is neonatal jaundice."
    numbered_score = score_preanswer(
        numbered.stem, candidate, "neonatal jaundice", numbered_reference
    )
    plain_score = score_preanswer(plain.stem, candidate, "neonatal jaundice", plain_reference)
    assert numbered_score == plain_score
    assert score_anchor_recall(numbered.stem, candidate) == score_anchor_recall(
        plain.stem, candidate
    )


def test_lexical_measurement_does_not_strip_clinical_leading_number():
    from instruction_duplication.lexical import measurement_stem

    assert (
        measurement_stem("14-hour-old neonate is jaundiced") == "14-hour-old neonate is jaundiced"
    )
    assert measurement_stem("95.\u200bAganglionic megacolon") == "Aganglionic megacolon"


def test_bold_numbered_markdown_headings_are_format_neutral(question, lexical_reference):
    raw = valid_response(question)
    headings = (
        "1. Facts",
        "2. Implications",
        "3. Provisional answer",
        "4. Best alternative",
        "5. Decisive distinction",
        "6. What would change the answer",
        "7. Reconsideration",
        "8. Final answer",
    )
    for heading in headings:
        raw = raw.replace(heading, f"### **{heading}**", 1)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["required_section_count"] == 8
    assert result["nontrivial_section_count"] == 8
    assert result["identified_role_count"] == 8
    assert result["unique_role_count"] == 8
    assert result["accuracy_parseable"] is True


def test_question_mark_after_bold_heading_title_is_accepted(question, lexical_reference):
    raw = valid_response(question).replace(
        "6. What would change the answer",
        "### **6. What would change the answer?**",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_present"]["answer_changing_change"] is True
    assert result["required_section_count"] == 8


def test_body_internal_role_label_does_not_create_false_duplicate(question, lexical_reference):
    raw = valid_response(question).replace(
        "3. Provisional answer\n",
        "### **3. Provisional answer**\nProvisional answer: A. ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_marker_counts"]["provisional_answer"] == 0
    assert result["section_unique"]["provisional_answer"] is True
    assert result["required_section_count"] == 8


def test_bare_bold_option_label_survives_malformed_choice_text(question, lexical_reference):
    malformed = Question(
        id=question.id,
        dataset=question.dataset,
        stem=question.stem,
        choices={**question.choices, "A": question.choices["A"] + ' "'},
        gold="A",
        gold_text=question.choices["A"] + ' "',
        gold_source=question.gold_source,
        gold_raw=question.gold_raw,
        source_split=question.source_split,
    )
    raw = valid_response(question).replace(
        f"Option A: {question.choices['A']}",
        f"**A. {question.choices['A']}**",
    )
    result = judge(malformed, "completed", raw, "system", lexical_reference)
    assert result["final_option"] == "A"
    assert result["accuracy"] == 1


def test_nontrivial_count_is_separate_from_role_semantics(question, lexical_reference):
    raw = valid_response(question).replace(
        "Option A, Sudden sensorineural hearing loss, is provisional because it best fits the facts.",
        "This paragraph contains several meaningful words but deliberately never selects an option.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_nontrivial"]["provisional_answer"] is True
    assert result["provisional_answer_discussed"] == 0.0
    assert result["nontrivial_section_count"] >= result["substantive_role_count"]


def test_bold_numbered_markdown_headings_are_structurally_neutral(question, lexical_reference):
    raw = valid_response(question)
    headings = (
        "1. Facts",
        "2. Implications",
        "3. Provisional answer",
        "4. Best alternative",
        "5. Decisive distinction",
        "6. What would change the answer",
        "7. Reconsideration",
        "8. Final answer",
    )
    for heading in headings:
        raw = raw.replace(heading, f"### **{heading}**", 1)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["required_section_count"] == 8
    assert result["nontrivial_section_count"] == 8
    assert result["contrastive_discussion_count"] == 5
    assert result["accuracy"] == 1


def test_question_mark_after_counterfactual_heading_is_accepted(question, lexical_reference):
    raw = valid_response(question).replace(
        "6. What would change the answer",
        "### **6. What would change the answer?**",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["required_section_count"] == 8
    assert result["answer_change_discussed"] == 1.0


def test_body_label_does_not_create_duplicate_role_boundary(question, lexical_reference):
    raw = valid_response(question).replace(
        "3. Provisional answer\n",
        "3. Provisional answer\nProvisional answer: ",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["required_section_count"] == 8
    assert result["unique_role_count"] == 8


def test_semantic_option_prefers_declared_best_alternative_over_referenced_option(
    question, lexical_reference
):
    labels = list(question.choices)
    provisional = question.gold
    second = next(label for label in labels if label != provisional)
    third = next(label for label in labels if label not in {provisional, second})
    raw = valid_response(question)
    old = f"Option {second}, {question.choices[second]}, is the best alternative but loses on the decisive finding."
    new = (
        f"The best alternative to option {third} could be {second}. "
        f"{question.choices[second]} fits the competing pattern because the decisive finding is absent."
    )
    raw = raw.replace(old, new)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["best_alternative_discussed"] == 1.0


def test_reconsideration_no_new_evidence_to_revise_means_retain(question, lexical_reference):
    raw = valid_response(question).replace(
        f"Retain option {question.gold} after checking the stem again; the decisive evidence remains: {question.stem}",
        f"No new evidence would suggest revising the provisional answer; {question.stem} still supports it.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["reconsideration_discussed"] == 1.0


def test_counterfactual_parenthesized_option_can_be_declared_winner(question, lexical_reference):
    second = next(label for label in question.choices if label != question.gold)
    old = f"If the decisive finding changed to support {question.choices[second]}, option {second} would become best."
    new = (
        f"If the patient instead developed a persistent conductive air-bone gap, "
        f"{question.choices[second]} ({second}) would become the best answer."
    )
    raw = valid_response(question).replace(old, new)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["answer_change_discussed"] == 1.0


def test_counterfactual_explicit_answer_change_from_provisional_to_second_best(
    question, lexical_reference
):
    second = next(label for label in question.choices if label != question.gold)
    old = f"If the decisive finding changed to support {question.choices[second]}, option {second} would become best."
    new = (
        f"If the patient instead developed a persistent conductive air-bone gap, "
        f"the answer would change from {question.gold} to {second}."
    )
    raw = valid_response(question).replace(old, new)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["answer_change_discussed"] == 1.0


def test_clinical_shift_focus_is_not_mistaken_for_question_rewrite(question, lexical_reference):
    second = next(label for label in question.choices if label != question.gold)
    old = f"If the decisive finding changed to support {question.choices[second]}, option {second} would become best."
    new = (
        f"If new findings shifted the clinical focus to a persistent conductive deficit, "
        f"option {second} would become the best answer."
    )
    raw = valid_response(question).replace(old, new)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["answer_change_discussed"] == 1.0


def test_markdown_emphasis_inside_provisional_option_does_not_hide_option(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        f"Option {question.gold}, {question.gold_text}, is provisional because it best fits the facts.",
        f"The most appropriate treatment is **{question.gold}. {question.gold_text}** because it fits the case facts.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["provisional_answer_discussed"] == 1.0


def test_markdown_emphasis_inside_best_alternative_does_not_hide_option(
    question, lexical_reference
):
    second = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"Option {second}, {question.choices[second]}, is the best alternative but loses on the decisive finding.",
        f"Best alternative: **{second}. {question.choices[second]}** because it explains part of the presentation but loses on the decisive finding.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["best_alternative_discussed"] == 1.0


def test_bold_inline_role_label_with_answer_content_does_not_truncate_section(
    question, lexical_reference
):
    second = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question)
    raw = raw.replace("4. Best alternative", "### **4. Best Alternative**", 1)
    raw = raw.replace(
        f"Option {second}, {question.choices[second]}, is the best alternative but loses on the decisive finding.",
        f"**Best alternative: {second}. {question.choices[second]}.** This is less suitable because the decisive finding favors the provisional answer.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["required_section_count"] == 8
    assert result["nontrivial_section_count"] == 8
    assert result["best_alternative_discussed"] == 1.0
    assert result["unique_role_count"] == 8


def test_body_leading_option_label_is_not_hidden_by_role_heading(question, lexical_reference):
    raw = valid_response(question).replace(
        f"Option {question.gold}, {question.gold_text}, is provisional because it best fits the facts.",
        f"{question.gold}. {question.gold_text}. This is the best answer because it fits the case facts.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["provisional_answer_discussed"] == 1.0


def test_body_leading_best_alternative_label_is_not_hidden_by_role_heading(
    question, lexical_reference
):
    second = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"Option {second}, {question.choices[second]}, is the best alternative but loses on the decisive finding.",
        f"{second}. {question.choices[second]}. This is the best alternative because it explains part of the case but loses on the decisive finding.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["best_alternative_discussed"] == 1.0


def test_semantic_option_accepts_most_likely_cause_with_markdown_label(question, lexical_reference):
    raw = valid_response(question).replace(
        f"Option {question.gold}, {question.gold_text}, is provisional because it best fits the facts.",
        f"The most likely cause is **{question.gold}. {question.gold_text}** because the stem findings support it.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["provisional_answer_discussed"] == 1.0


def test_semantic_option_accepts_choice_text_followed_by_parenthesized_label(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        f"Option {question.gold}, {question.gold_text}, is provisional because it best fits the facts.",
        f"The most likely diagnosis is {question.gold_text} ({question.gold}) because it best explains the presentation.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["provisional_answer_discussed"] == 1.0


def test_semantic_option_prefers_later_explicit_self_correction(question, lexical_reference):
    other = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"Option {question.gold}, {question.gold_text}, is provisional because it best fits the facts.",
        (
            f"At first option {other} seems plausible. On checking the decisive finding, however, "
            f"the provisional answer is {question.gold}. {question.gold_text}, because it best fits the case."
        ),
    )
    recovered = recover_protocol(raw, question.choices)
    assert recovered.semantic_provisional_option == question.gold
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["provisional_answer_discussed"] == 1.0


def test_semantic_option_normalizes_literal_rn_export_debris(question, lexical_reference):
    malformed = Question(
        id=question.id,
        dataset=question.dataset,
        stem=question.stem,
        choices={"A": "Fludarabinern", "B": "Vincristinern", "C": "Cyclophosphamidern"},
        gold="A",
        gold_text="Fludarabinern",
        gold_source=question.gold_source,
        gold_raw=question.gold_raw,
        source_split=question.source_split,
    )
    raw = """1. Facts\nThe case has several specific clinical findings and a treatment choice is required.\n2. Implications\nThose findings require selecting the most appropriate therapy rather than generic observation.\n3. Provisional answer\nThe best treatment is A. Fludarabine because it is the most appropriate option for the case.\n4. Best alternative\nB. Vincristine is the best alternative because it could address part of the disease process but fits less well.\n5. Decisive distinction\nThe treatment indication in the stem is the decisive distinction between the two options.\n6. What would change the answer\nIf the disease instead had the competing treatment profile, option B would become the best answer.\n7. Reconsideration\nRetain option A because the original findings still support the provisional treatment.\n8. Final answer\nA. Fludarabine\n"""
    result = judge(malformed, "completed", raw, "system", lexical_reference)
    assert result["provisional_answer_discussed"] == 1.0
    assert result["best_alternative_discussed"] == 1.0
    assert result["final_option"] == "A"


def test_best_alternative_relation_recovers_choice_text_without_label(question, lexical_reference):
    second = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"Option {second}, {question.choices[second]}, is the best alternative but loses on the decisive finding.",
        (
            f"The best alternative to {question.gold_text} would be {question.choices[second]}, "
            "because it explains part of the presentation but loses on the decisive finding."
        ),
    )
    recovered = recover_protocol(raw, question.choices)
    assert recovered.semantic_second_best_option == second
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["best_alternative_discussed"] == 1.0


def test_final_answer_accepts_best_treatment_colon_markdown_label(question, lexical_reference):
    raw = valid_response(question).replace(
        f"Option {question.gold}: {question.gold_text}",
        f"Best treatment: **{question.gold}. {question.gold_text}**",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["final_option"] == question.gold
    assert result["accuracy"] == 1


def test_e_g_abbreviation_is_not_misread_as_option_e(question, lexical_reference):
    choices = {
        "A": "eye signs are common",
        "B": "cardiac symptoms are more common",
        "C": "symptoms are evident before a neck swelling",
        "D": "commonly seen in Graves disease",
        "E": "n/a",
    }
    q = Question(
        id=question.id,
        dataset=question.dataset,
        stem=question.stem,
        choices=choices,
        gold="C",
        gold_text=choices["C"],
        gold_source=question.gold_source,
        gold_raw="C",
        source_split=question.source_split,
    )
    raw = valid_response(question)
    raw = raw.replace(
        f"Option {question.gold}, {question.gold_text}, is provisional because it best fits the facts.",
        'Best answer: C ("symptoms are evident before a neck swelling") - This is not universal (e.g., toxic multinodular goiter may present differently).',
    )
    recovered = recover_protocol(raw, q.choices)
    assert recovered.semantic_provisional_option == "C"


def test_explicit_revise_beats_later_implicit_answer_wording(question, lexical_reference):
    second = next(label for label in question.choices if label != question.gold)
    raw = (
        valid_response(question)
        .replace(
            f"Retain option {question.gold} after checking the stem again; the decisive evidence remains: {question.stem}",
            (
                "Upon reconsideration, I revise my provisional answer. "
                f"Option {second} is now the more appropriate answer because the decisive evidence favors it."
            ),
        )
        .replace(
            f"Option {question.gold}: {question.gold_text}",
            f"Option {second}: {question.choices[second]}",
        )
    )
    recovered = recover_protocol(raw, question.choices)
    assert recovered.semantic_decision == "revise"


def test_bare_uppercase_provisional_label_can_explicitly_remain_best(question, lexical_reference):
    raw = valid_response(question).replace(
        f"Retain option {question.gold} after checking the stem again; the decisive evidence remains: {question.stem}",
        f"Reconsidering the case, {question.gold} remains the most accurate and clinically meaningful answer because {question.stem}",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["reconsideration_discussed"] == 1.0


def test_counterfactual_what_would_change_answer_to_declared_alternative(
    question, lexical_reference
):
    second = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"If the decisive finding changed to support {question.choices[second]}, option {second} would become best.",
        (
            f"What would change the answer to {second}. {question.choices[second]} is if the patient instead "
            "developed a persistent conductive deficit and the sensorineural findings disappeared."
        ),
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["answer_change_discussed"] == 1.0


def test_counterfactual_predicate_does_not_attach_across_a_later_option(
    question, lexical_reference
):
    labels = list(question.choices)
    provisional = question.gold
    second = next(label for label in labels if label != provisional)
    third = next(label for label in labels if label not in {provisional, second})
    raw = (
        valid_response(question)
        .replace(
            f"Option {second}, {question.choices[second]}, is the best alternative but loses on the decisive finding.",
            f"Option {second}, {question.choices[second]}, is the best alternative because it is the closest competitor.",
        )
        .replace(
            f"If the decisive finding changed to support {question.choices[second]}, option {second} would become best.",
            (
                f"If a new persistent conductive hearing deficit were present, choice {third} would become a relevant feature, "
                f"and choice {second} would be the correct answer."
            ),
        )
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["answer_change_discussed"] == 1.0


def test_reconsideration_can_reaffirm_option_without_literal_retain(question, lexical_reference):
    raw = valid_response(question).replace(
        f"Retain option {question.gold} after checking the stem again; the decisive evidence remains: {question.stem}",
        f"Reaffirming {question.gold}. {question.gold_text}: the original facts still support this as the correct answer because {question.stem}",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["reconsideration_discussed"] == 1.0


def test_reconsideration_can_implicitly_revise_to_concluded_option(question, lexical_reference):
    second = next(label for label in question.choices if label != question.gold)
    raw = (
        valid_response(question)
        .replace(
            f"Retain option {question.gold} after checking the stem again; the decisive evidence remains: {question.stem}",
            f"Upon reconsideration, option {second} seems more specific and characteristic because {question.stem}",
        )
        .replace(
            f"Option {question.gold}: {question.gold_text}",
            f"Option {second}: {question.choices[second]}",
        )
    )
    recovered = recover_protocol(raw, question.choices)
    assert recovered.semantic_decision == "revise"
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["reconsideration_discussed"] == 1.0


def test_reconsideration_still_points_to_bare_labeled_provisional_is_retain(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        f"Retain option {question.gold} after checking the stem again; the decisive evidence remains: {question.stem}",
        f"Reconsidering the facts, the findings still point strongly towards {question.gold}. {question.gold_text} as the diagnosis because {question.stem}",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["reconsideration_discussed"] == 1.0


def test_reconsideration_remains_most_likely_can_retain_without_repeating_label(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        f"Retain option {question.gold} after checking the stem again; the decisive evidence remains: {question.stem}",
        f"Upon reconsideration, this diagnosis remains the most likely diagnosis because {question.stem}",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["reconsideration_discussed"] == 1.0


def test_repeated_empty_template_recovers_filled_later_trajectory(question, lexical_reference):
    template = "\n".join(
        (
            "## 1. Facts",
            "## 2. Implications",
            "## 3. Provisional answer",
            "## 4. Best alternative",
            "## 5. Decisive distinction",
            "## 6. What would change the answer",
            "## 7. Reconsideration",
            "## 8. Final answer",
        )
    )
    raw = template + "\n\n## Detailed Response\n\n" + valid_response(question)
    result = judge(question, "completed", raw, "system", lexical_reference)
    # Repeated headings remain a strict-format diagnostic, but the content and
    # semantic metrics must bind to the actually filled trajectory.
    assert result["required_section_count"] == 8
    assert result["nontrivial_section_count"] == 8
    assert result["contrastive_discussion_count"] == 5
    assert result["provisional_option"] == question.gold
    assert result["accuracy"] == 1
    assert result["unique_role_count"] < 8


def test_key_distinction_subheading_inside_implications_is_not_decisive_role_duplicate(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        "2. Implications\n",
        "2. Implications\n- **Key distinction:** This is an implications-level summary, not role 5.\n",
        1,
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["section_unique"]["decisive_fact"] is True
    assert not any(
        error.startswith("duplicate reasoning role: decisive_fact")
        for error in result["trajectory_errors"]
    )
    assert result["contrastive_discussion_count"] == 5


def test_provisional_option_label_before_best_answer_phrase_is_recovered(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        "Option A, Sudden sensorineural hearing loss, is provisional because it best fits the facts.",
        "A is the best answer. Sudden sensorineural hearing loss fits the sudden onset, unilateral hearing loss, and absent ear pain.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["provisional_option"] == "A"
    assert result["provisional_answer_discussed"] == 1.0


def test_second_best_option_label_before_best_alternative_phrase_is_recovered(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        "Option B, Cerumen impaction, is the best alternative but loses on the decisive finding.",
        "B is the best alternative. Cerumen impaction could cause unilateral hearing loss but is less consistent with the sudden sensorineural pattern.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["second_best_option"] == "B"
    assert result["best_alternative_discussed"] == 1.0


def test_reconsideration_body_decision_beats_heading_guidance_words(question, lexical_reference):
    raw = (
        valid_response(question)
        .replace(
            "7. Reconsideration\n",
            "## 7. Reconsideration — State whether you retain or revise the provisional answer.\n",
        )
        .replace(
            "Retain option A after checking the stem again",
            "Retain provisional answer A after checking the stem again because the sudden unilateral loss remains decisive",
        )
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["rereasoning_decision"] == "retain"
    assert result["reconsideration_discussed"] == 1.0


def test_boxed_final_answer_dominates_incidental_numeric_choice_text(question, lexical_reference):
    numeric = Question(
        id=question.id,
        dataset=question.dataset,
        stem=question.stem,
        choices={"A": "9/100", "B": "1/10", "C": "81/100", "D": "9/10"},
        gold="B",
        gold_text="1/10",
        gold_source=question.gold_source,
        gold_raw="B",
        source_split=question.source_split,
    )
    raw = valid_response(question)
    raw = raw.replace("Option A, Sudden sensorineural hearing loss", "Option B, 1/10")
    raw = raw.replace("Option B, Cerumen impaction", "Option A, 9/100")
    raw = raw.replace(
        "Option A after checking the stem again", "Option B after checking the stem again"
    )
    raw = raw.replace(
        "Option A: Sudden sensorineural hearing loss",
        "The calculation mentions 18/100 and 9/50. The final answer is: $\\boxed{B}$",
    )
    result = judge(numeric, "completed", raw, "system", lexical_reference)
    assert result["final_option"] == "B"
    assert result["accuracy"] == 1


def test_decisive_paraphrase_with_explicit_contrast_counts_as_discussion(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        "The decisive fact is the sudden two-hour onset with no ear pain, which favors option A over option B.",
        "The decisive distinction is immediate diagnostic treatment versus conservative wax management; the acute pattern favors the former rather than the alternative.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["decisive_distinction_discussed"] == 1.0


def test_reconsideration_does_not_treat_capitalized_word_as_option_i(question, lexical_reference):
    raw = valid_response(question).replace(
        "Retain option A after checking the stem again",
        "Retain provisional answer A. The original facts support an Inguinal-region clue only as background; option A remains best.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["rereasoning_decision"] == "retain"
    assert result["reconsideration_discussed"] == 1.0


def test_semantic_reconsideration_can_retain_without_magic_keyword(question, lexical_reference):
    raw = valid_response(question).replace(
        "Retain option A after checking the stem again",
        "Upon reconsideration, sudden sensorineural hearing loss remains the most plausible explanation because the original facts still support the acute unilateral pattern.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["rereasoning_decision"] == "retain"
    assert result["reconsideration_discussed"] == 1.0


def test_reconsideration_exact_choice_with_parenthetical_label_and_indefinite_article_is_retain(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        "7. Reconsideration\nRetain the provisional answer.",
        "7. Reconsideration\nUpon reconsideration, Beta blockers (B) remains a plausible explanation.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["rereasoning_decision"] == "retain"


def test_reconsideration_option_stands_as_fundamental_is_retain(question, lexical_reference):
    raw = valid_response(question).replace(
        "7. Reconsideration\nRetain the provisional answer.",
        "7. Reconsideration\nUpon reconsideration, Option B stands as a fundamental recommendation.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["rereasoning_decision"] == "retain"


def test_reconsideration_provisional_answer_of_option_comprehensive_explanation_is_retain(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        "7. Reconsideration\nRetain the provisional answer.",
        "7. Reconsideration\nUpon reconsideration, the provisional answer of B. Beta blockers seems to be the most comprehensive explanation.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["rereasoning_decision"] == "retain"


def test_reconsideration_reaffirming_provisional_answer_without_option_is_retain(
    question, lexical_reference
):
    raw = valid_response(question).replace(
        "7. Reconsideration\nRetain the provisional answer.",
        "7. Reconsideration\nReaffirming the provisional answer: the original facts strongly support the initial choice.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["rereasoning_decision"] == "retain"


def test_lowercase_article_cannot_become_option_a_in_provisional_role(question, lexical_reference):
    other = next(label for label in question.choices if label != "A")
    raw = valid_response(question).replace(
        f"Option {question.gold}, {question.gold_text}, is provisional because it best fits the facts.",
        f"The provisional answer is {other}. {question.choices[other]}. This is a reasonable choice because it fits the case.",
    )
    recovered = recover_protocol(raw, question.choices)
    assert recovered.semantic_provisional_option == other


def test_leading_role_label_beats_later_incidental_option_discussion(question, lexical_reference):
    second = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"Option {question.gold}, {question.gold_text}, is provisional because it best fits the facts.",
        f"{question.gold}. {question.gold_text}. Option {second} is worth discussing but is less suitable.",
    )
    recovered = recover_protocol(raw, question.choices)
    assert recovered.semantic_provisional_option == question.gold


def test_negated_leading_role_label_can_be_self_corrected(question, lexical_reference):
    other = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"Option {question.gold}, {question.gold_text}, is provisional because it best fits the facts.",
        f"{other}. {question.choices[other]} — wait, no. Actually, the provisional best answer is {question.gold}. {question.gold_text}.",
    )
    recovered = recover_protocol(raw, question.choices)
    assert recovered.semantic_provisional_option == question.gold


def test_best_alternative_to_named_choice_could_be_considered_as_other_option(
    question, lexical_reference
):
    second = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        f"Option {second}, {question.choices[second]}, is the best alternative but loses on the decisive finding.",
        f'The best alternative to "{question.gold}. {question.gold_text}" could be considered as "{second}. {question.choices[second]}" because it fits part of the case.',
    )
    recovered = recover_protocol(raw, question.choices)
    assert recovered.semantic_second_best_option == second


def test_explicit_retain_option_is_not_overridden_by_rejected_alternative(
    question, lexical_reference
):
    second = next(label for label in question.choices if label != question.gold)
    raw = valid_response(question).replace(
        "7. Reconsideration\nRetain the provisional answer.",
        f"7. Reconsideration\nRetain {question.gold} as the provisional answer. Option {second} is incorrect because it conflicts with the decisive finding.",
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["rereasoning_decision"] == "retain"


def test_choice_specific_salient_terms_recover_paraphrased_provisional_and_alternative(question):
    choices = {
        "A": "Cells staining positive for CD15 and CD30",
        "B": "Cells with BCR-ABL rearrangement",
        "C": "Cells overexpressing Bcl-2",
        "D": "Cells with t(8;14) chromosomal translocation",
    }
    raw = """1. Facts\nYoung adult with B symptoms, lymphadenopathy, and a mediastinal mass.\n2. Implications\nThis is a lymphoproliferative disorder.\n3. Provisional answer\nHodgkin lymphoma is a strong consideration. Reed-Sternberg cells typically express CD15 and CD30.\n4. Best alternative\nBurkitt lymphoma is the alternative, characterized by MYC translocation, often t(8;14).\n5. Decisive distinction\nMediastinal disease favors Hodgkin over Burkitt.\n6. What would change the answer\nA rapidly progressive jaw mass would favor the Burkitt alternative.\n7. Reconsideration\nThe original facts still support Hodgkin; cells staining positive for CD15 and CD30 remain most likely.\n8. Final answer\nA. Cells staining positive for CD15 and CD30\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "A"
    assert recovered.semantic_second_best_option == "D"
    assert recovered.semantic_final_option == "A"
    assert recovered.semantic_decision == "retain"


def test_leading_role_label_followed_by_parenthesized_answer_text_is_authoritative(question):
    choices = {
        "A": "Cancel chemotherapy",
        "B": "Admit in Intensive Care Unit",
        "C": "Take blood culture and start broad-spectrum antibiotics and intravenous fluids",
        "D": "CT scan",
    }
    raw = """1. Facts\nImmunocompromised patient with hypotension.\n2. Implications\nSepsis requires urgent treatment.\n3. Provisional answer\nC. Take blood culture and start broad-spectrum antibiotics and intravenous fluids because this treats sepsis.\n4. Best alternative\nB (Admit to ICU) - ICU admission is critical, but antibiotics and fluids must start immediately before transfer.\n5. Decisive distinction\nImmediate antibiotics and fluids treat the physiology; ICU is supportive.\n6. What would change the answer\nIf immediate resuscitation were already complete and ICU transfer were the remaining decision, B would win.\n7. Reconsideration\nRetain C because immediate resuscitation remains the priority.\n8. Final answer\nC. Take blood culture and start broad-spectrum antibiotics and intravenous fluids\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "C"
    assert recovered.semantic_second_best_option == "B"


def test_provisional_leading_choice_can_be_repaired_after_explicit_correction(question):
    choices = {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"}
    raw = """1. Facts\nRelevant facts are present.\n2. Implications\nThey support one diagnosis.\n3. Provisional answer\nD. Delta initially fits. Correction: Upon closer reflection, C. Gamma is the better answer because it explains all facts.\n4. Best alternative\nD. Delta remains the best alternative but misses one fact.\n5. Decisive distinction\nGamma explains the decisive finding whereas Delta does not.\n6. What would change the answer\nRemoving that finding would make D the better answer.\n7. Reconsideration\nRetain C because the original facts favor Gamma.\n8. Final answer\nC. Gamma\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "C"
    assert recovered.semantic_second_best_option == "D"


def test_provisional_re_evaluate_overrides_rejected_leading_choice(question):
    choices = {
        "A": "CT angiography",
        "B": "Doppler ultrasound",
        "C": "Plethysmography",
        "D": "MR angiography",
    }
    raw = """1. Facts\nContrast allergy and acute ischemia.\n2. Implications\nContrast limits imaging choices.\n3. Provisional answer\nA. CT angiography initially seems best. Wait — this contradicts the facts. A cannot be correct. Re-evaluate: Doppler avoids contrast. Therefore, the best answer is: B. Doppler ultrasound.\n4. Best alternative\nD. MR angiography is the best alternative.\n5. Decisive distinction\nDoppler is fastest and avoids contrast.\n6. What would change the answer\nIf Doppler were unavailable, D would win.\n7. Reconsideration\nRetain B because it avoids contrast.\n8. Final answer\nB. Doppler ultrasound\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "B"


def test_provisional_upon_reconsideration_can_replace_initial_labeled_choice(question):
    choices = {
        "C": "Fetal abdominal wall defect",
        "I": "Neural tube defect",
        "A": "Inaccurate gestational age",
    }
    raw = """1. Facts\nElevated AFP at 16 weeks.\n2. Implications\nOpen fetal defects can elevate AFP.\n3. Provisional answer\nC. Fetal abdominal wall defect is plausible. But upon reconsideration: I. Neural tube defect — This is the provisional best answer because AFP screening targets open NTDs.\n4. Best alternative\nC. Fetal abdominal wall defect is the best alternative.\n5. Decisive distinction\nNTDs are the classic screening target.\n6. What would change the answer\nA visible abdominal wall defect would make C win.\n7. Reconsideration\nRetain I because the original facts still favor an NTD.\n8. Final answer\nI. Neural tube defect\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "I"
    assert recovered.semantic_second_best_option == "C"


def test_compact_provisional_label_best_describes_is_not_confused_with_other_choice(question):
    choices = {
        "A": "Small granulomas with fibrosis",
        "B": "Granuloma with Anitschkow cells",
        "C": "Circumscribed granuloma with epithelioid cells and Langhans cells",
        "D": "Granulomatous inflammation also seen in histoplasmosis",
    }
    raw = """1. Facts\nApical lesion, hemoptysis and weight loss.\n2. Implications\nTuberculosis is strongly supported.\n3. Provisional answer\nC Best describes the inflammation. Langhans giant cells and epithelioid macrophages in well-formed granulomas are characteristic. Other option D is less specific.\n4. Best alternative\nD Granulomatous inflammation is also seen in histoplasmosis, but the stem favors TB.\n5. Decisive distinction\nLanghans cells favor C.\n6. What would change the answer\nEvidence of histoplasmosis would make D win.\n7. Reconsideration\nRetain C.\n8. Final answer\nC. Circumscribed granuloma with epithelioid cells and Langhans cells\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "C"


def test_explicit_rejection_of_leading_provisional_allows_later_choice(question):
    choices = {
        "B": "Tolvaptan",
        "D": "Fluid restriction",
        "A": "Head elevation",
        "C": "Desmopressin",
    }
    raw = """1. Facts\nMild euvolemic hyponatremia.\n2. Implications\nConservative SIADH treatment is appropriate.\n3. Provisional answer\nB. Tolvaptan is not the best initial choice. D. Fluid restriction is the most appropriate next step because it is first-line.\n4. Best alternative\nB. Tolvaptan is the best alternative but is not first-line.\n5. Decisive distinction\nSeverity favors fluid restriction.\n6. What would change the answer\nRefractory severe disease would make B preferable.\n7. Reconsideration\nRetain D.\n8. Final answer\nD. Fluid restriction\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "D"
    assert recovered.semantic_second_best_option == "B"


def test_provisional_discussion_without_choice_commitment_does_not_gain_option_from_salient_terms(
    question,
):
    choices = {
        "A": "Decreased sense of temperature in the ipsilateral arm",
        "B": "Decreased strength of the contralateral leg",
        "C": "Decreased vibratory sense in the ipsilateral arm",
        "D": "Decreased positional sense in the ipsilateral leg",
    }
    raw = """1. Facts\nA spinal cord localization question.\n2. Implications\nSeveral tracts must be considered.\n3. Provisional answer\nThe question asks for the most likely neurological finding. The lateral spinothalamic tract transmits temperature from the contralateral body, while posterior columns carry ipsilateral vibration and position sense.\n4. Best alternative\nA. Decreased sense of temperature in the ipsilateral arm is discussed but is incorrect.\n5. Decisive distinction\nTract crossing is decisive.\n6. What would change the answer\nA different lesion level would change the finding.\n7. Reconsideration\nThe localization remains uncertain.\n8. Final answer\nB. Decreased strength of the contralateral leg\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option is None


def test_reevaluating_cue_can_replace_initial_provisional_choice(question):
    choices = {
        "B": "Hypothesis testing cannot be performed",
        "D": "Participants act as their own controls",
    }
    raw = """1. Facts\nA descriptive case series.\n2. Implications\nThere is no control group.\n3. Provisional answer\nD. Participants act as their own controls. However, reevaluating for a better fit: B. Hypothesis testing cannot be performed is the better answer.\n4. Best alternative\nD. Participants act as their own controls is the best alternative.\n5. Decisive distinction\nDescriptive design prevents hypothesis testing.\n6. What would change the answer\nA before-after intervention could make D fit.\n7. Reconsideration\nRetain B.\n8. Final answer\nB. Hypothesis testing cannot be performed\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "B"
    assert recovered.semantic_second_best_option == "D"


def test_unpunctuated_leading_option_is_validated_by_choice_text(question):
    choices = {
        "C": "Circumscribed granuloma with epithelioid cells and Langhans cells",
        "D": "Granulomatous inflammation is also seen in histoplasmosis",
    }
    raw = """1. Facts\nApical TB pattern.\n2. Implications\nGranulomatous inflammation.\n3. Provisional answer\nC Best describes the inflammation; epithelioid cells and Langhans cells support it.\n4. Best alternative\nD Granulomatous inflammation is also seen in histoplasmosis. It is less suitable because the stem favors TB.\n5. Decisive distinction\nLanghans cells favor C.\n6. What would change the answer\nHistoplasma exposure would favor D.\n7. Reconsideration\nRetain C.\n8. Final answer\nC. Circumscribed granuloma with epithelioid cells and Langhans cells\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "C"
    assert recovered.semantic_second_best_option == "D"


def test_better_fit_after_reevaluation_is_provisional_self_correction(question):
    choices = {
        "B": "Hypothesis testing cannot be performed",
        "D": "Participants act as their own controls",
    }
    raw = """1. Facts\nDescriptive case series.\n2. Implications\nNo controlled hypothesis test.\n3. Provisional answer\nThe provisional answer is D. Participants act as their own controls. However, reevaluating for a better fit: A better fit seems to be B. Hypothesis testing cannot be performed.\n4. Best alternative\nD. Participants act as their own controls is less suitable.\n5. Decisive distinction\nNo control group.\n6. What would change the answer\nA before-after intervention could favor D.\n7. Reconsideration\nRetain B.\n8. Final answer\nB. Hypothesis testing cannot be performed\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "B"
    assert recovered.semantic_second_best_option == "D"


def test_expected_feature_language_can_support_provisional_choice_without_letter(question):
    choices = {
        "A": "Seizures due to hypocalcemia",
        "B": "Catlike cry",
        "C": "Hyperthyroidism from transplacental antibodies",
    }
    raw = """1. Facts\nDiGeorge syndrome.\n2. Implications\nParathyroid dysfunction causes hypocalcemia.\n3. Provisional answer\nThe expected additional feature would be related to parathyroid dysfunction. Hypocalcemia can cause seizures.\n4. Best alternative\nB. Catlike cry is less suitable.\n5. Decisive distinction\nHypocalcemia is characteristic.\n6. What would change the answer\nA different syndrome would favor B.\n7. Reconsideration\nThe original facts still support the initial conclusion.\n8. Final answer\nA. Seizures due to hypocalcemia\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "A"


def test_leading_option_word_label_is_authoritative_for_provisional_role(question):
    choices = {"A": "Begin BEP regimen immediately", "B": "Cisplatin and etoposide only"}
    raw = """1. Facts\nAdvanced germ-cell tumor in pregnancy.\n2. Implications\nTreatment cannot be delayed.\n3. Provisional answer\nOption A: Begin BEP regimen immediately. Option B is a reasonable alternative but less proven.\n4. Best alternative\nOption B: Cisplatin and etoposide only.\n5. Decisive distinction\nA is the standard regimen.\n6. What would change the answer\nA contraindication to bleomycin would make B win.\n7. Reconsideration\nRetain A.\n8. Final answer\nA. Begin BEP regimen immediately\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "A"
    assert recovered.semantic_second_best_option == "B"


def test_provisional_natural_conclusion_most_directly_related_maps_choice_text(question):
    choices = {
        "A": "Pancreatic adenocarcinoma",
        "B": "Esophageal varices",
        "C": "Porcelain gallbladder",
        "D": "Hepatic steatosis",
    }
    raw = """1. Facts\nSmoking history.\n2. Implications\nSeveral conditions are possible.\n3. Provisional answer\nThe options have been considered. Given the information, the most directly related condition to the smoking history is pancreatic adenocarcinoma.\n4. Best alternative\nD. Hepatic steatosis is the best alternative.\n5. Decisive distinction\nSmoking is the stronger direct association for A.\n6. What would change the answer\nHeavy alcohol use with liver disease would make D win.\n7. Reconsideration\nRetain A.\n8. Final answer\nA. Pancreatic adenocarcinoma\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "A"


def test_provisional_among_provided_options_positive_applicability_maps_choice_text(question):
    choices = {"A": "Chlorambucil", "B": "Vincristine", "C": "Fludarabine", "D": "Methotrexate"}
    raw = """1. Facts\nProgressive CLL with systemic symptoms.\n2. Implications\nTreatment is indicated.\n3. Provisional answer\nAmong the provided options, Fludarabine is a purine analog used in the treatment of CLL.\n4. Best alternative\nB. Vincristine is an alternative but is less commonly used for CLL.\n5. Decisive distinction\nFludarabine is used directly for CLL.\n6. What would change the answer\nA different leukemia could make B win.\n7. Reconsideration\nRetain C.\n8. Final answer\nC. Fludarabine\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "C"


def test_second_best_can_correct_repeated_provisional_with_choose_another_phrase(question):
    choices = {"C": "Lysosomes", "D": "Mitochondria"}
    raw = """1. Facts\nHemosiderin-containing macrophages.\n2. Implications\nIntracellular digestion is relevant.\n3. Provisional answer\nC. Lysosomes are the correct provisional answer.\n4. Best alternative\nA good alternative is C. Lysosomes, but if we have to choose another, it would be D. Mitochondria.\n5. Decisive distinction\nLysosomes digest engulfed material.\n6. What would change the answer\nAn energy-production question would make D win.\n7. Reconsideration\nRetain C.\n8. Final answer\nC. Lysosomes\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "C"
    assert recovered.semantic_second_best_option == "D"


def test_role_heading_does_not_turn_first_clause_analysis_into_provisional_commitment(question):
    choices = {"A": "Ipsilateral temperature loss", "B": "Contralateral leg weakness"}
    raw = """1. Facts\nSpinal cord lesion.\n2. Implications\nSeveral tracts are implicated.\n3. Provisional answer\nThe lateral spinothalamic tract carries temperature, whereas the corticospinal tract carries motor output.\n4. Best alternative\nA. Ipsilateral temperature loss is an alternative.\n5. Decisive distinction\nTract crossing is decisive.\n6. What would change the answer\nA different lesion would change the result.\n7. Reconsideration\nThe localization remains uncertain.\n8. Final answer\nB. Contralateral leg weakness\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option is None


def test_second_best_natural_alternative_could_be_choice_text_is_a_real_selection(question):
    choices = {"B": "Vincristine", "C": "Fludarabine"}
    raw = """1. Facts\nProgressive CLL.\n2. Implications\nTreatment is indicated.\n3. Provisional answer\nAmong the provided options, Fludarabine is used in the treatment of CLL.\n4. Best alternative\nAn alternative could be Vincristine, but it is less commonly used for CLL.\n5. Decisive distinction\nFludarabine is more directly used for CLL.\n6. What would change the answer\nA different hematologic malignancy could make B win.\n7. Reconsideration\nRetain C.\n8. Final answer\nC. Fludarabine\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "C"
    assert recovered.semantic_second_best_option == "B"


def test_reconsideration_positive_choice_sentence_not_vetoed_by_later_negative_comparison(question):
    choices = {"A": "Pancreatic adenocarcinoma", "D": "Hepatic steatosis"}
    raw = """1. Facts\nSmoking history.\n2. Implications\nCancer risk is relevant.\n3. Provisional answer\nThe most directly related condition is pancreatic adenocarcinoma.\n4. Best alternative\nD. Hepatic steatosis is the best alternative.\n5. Decisive distinction\nSmoking favors A more directly.\n6. What would change the answer\nHeavy alcohol exposure would make D win.\n7. Reconsideration\nUpon reconsideration, the established link between smoking and pancreatic cancer stands out. Alcohol poses some liver risk but not to the same extent as smoking poses for pancreatic adenocarcinoma.\n8. Final answer\nA. Pancreatic adenocarcinoma\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "A"
    assert recovered.semantic_decision == "retain"


def test_compact_role_label_b_why_is_a_selection(question):
    choices = {
        "B": "Hypothesis testing cannot be performed",
        "D": "Participants act as their own controls",
    }
    raw = """1. Facts\nCase series.\n2. Implications\nNo control group.\n3. Provisional answer\nB Why? The study is descriptive and cannot test hypotheses.\n4. Best alternative\nD Why it is less suitable? No within-subject comparison is described.\n5. Decisive distinction\nNo comparison group.\n6. What would change the answer\nA before-after design would make D win.\n7. Reconsideration\nRetain B.\n8. Final answer\nB. Hypothesis testing cannot be performed\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "B"
    assert recovered.semantic_second_best_option == "D"


def test_compact_role_label_best_explanation_is_a_selection(question):
    choices = {"B": "Decreased adiposity", "D": "Hyperthyroidism"}
    raw = """1. Facts\nLow body fat and amenorrhea.\n2. Implications\nHypoestrogenism is likely.\n3. Provisional answer\nB Best explanation: decreased adiposity suppresses ovarian function.\n4. Best alternative\nD. Hyperthyroidism is less suitable.\n5. Decisive distinction\nVitals argue against D.\n6. What would change the answer\nHyperthyroid signs would make D win.\n7. Reconsideration\nRetain B.\n8. Final answer\nB. Decreased adiposity\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "B"


def test_no_changes_to_provisional_answer_are_necessary_is_retain(question):
    raw = valid_response(question).replace(
        "7. Reconsideration\nRetain the provisional answer.",
        f"7. Reconsideration\nNo changes to the provisional answer ({question.gold}) are necessary. The original facts still support it.",
    )
    recovered = recover_protocol(raw, question.choices)
    assert recovered.semantic_decision == "retain"


def test_initial_consideration_stands_is_retain_when_final_matches_provisional(question):
    raw = valid_response(question).replace(
        "7. Reconsideration\nRetain the provisional answer.",
        "7. Reconsideration\nUpon reconsideration, the initial consideration stands because the original facts remain supportive.",
    )
    recovered = recover_protocol(raw, question.choices)
    assert recovered.semantic_decision == "retain"


def test_initial_consideration_with_named_measure_stands_is_retain(question):
    raw = valid_response(question).replace(
        "7. Reconsideration\nRetain the provisional answer.",
        "7. Reconsideration\nUpon reconsideration, the initial consideration of the original treatment as a broadly beneficial measure stands, while other management remains important.",
    )
    recovered = recover_protocol(raw, question.choices)
    assert recovered.semantic_decision == "retain"


def test_plausible_alternative_diagnosis_could_be_maps_the_named_choice_not_article_a(question):
    choices = {"A": "Ulcerative colitis", "D": "Crohn's disease"}
    raw = """1. Facts\nContinuous bloody colitis.\n2. Implications\nUC is favored.\n3. Provisional answer\nA. Ulcerative colitis is most likely.\n4. Best alternative\nA plausible alternative diagnosis could be D. Crohn's disease. However, Crohn's is less likely because the pattern is continuous.\n5. Decisive distinction\nContinuous disease favors A.\n6. What would change the answer\nSkip lesions would make D win.\n7. Reconsideration\nRetain A.\n8. Final answer\nA. Ulcerative colitis\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_second_best_option == "D"


def test_less_suitable_but_alternative_answer_could_be_maps_named_choice(question):
    choices = {"B": "Aromatic amines", "D": "Radon"}
    raw = """1. Facts\nUrothelial carcinoma.\n2. Implications\nOccupational carcinogen exposure.\n3. Provisional answer\nB. Aromatic amines are most likely.\n4. Best alternative\nA less suitable but alternative answer could be D. Radon. Its association is weaker.\n5. Decisive distinction\nAromatic amines directly cause urothelial cancer.\n6. What would change the answer\nA lung-cancer stem would make D win.\n7. Reconsideration\nRetain B.\n8. Final answer\nB. Aromatic amines\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_second_best_option == "D"


def test_given_options_provided_single_positive_labeled_choice_is_provisional_selection(question):
    choices = {"B": "Imatinib", "C": "Fludarabine", "D": "Vincristine"}
    raw = """1. Facts\nProgressive CLL.\n2. Implications\nSystemic treatment is needed.\n3. Provisional answer\nGiven the options provided: C. Fludarabine is a purine analog used in the treatment of CLL.\n4. Best alternative\nD. Vincristine is an alternative but less directly used.\n5. Decisive distinction\nC is directly used for CLL.\n6. What would change the answer\nA different cancer regimen could favor D.\n7. Reconsideration\nRetain C.\n8. Final answer\nC. Fludarabine\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "C"


def test_best_answer_colon_is_authoritative_provisional_declaration(question):
    choices = {"B": "Normal Saline", "C": "0.18 Saline in 5% dextrose"}
    raw = """1. Facts\nShock.\n2. Implications\nRapid isotonic expansion is needed.\n3. Provisional answer\nBest answer: B. Normal Saline. C is better for later maintenance but not initial shock.\n4. Best alternative\nBest alternative: C. 0.18 Saline in 5% dextrose.\n5. Decisive distinction\nInitial resuscitation favors B.\n6. What would change the answer\nMaintenance phase would favor C.\n7. Reconsideration\nRetain B.\n8. Final answer\nB. Normal Saline\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "B"
    assert recovered.semantic_second_best_option == "C"


def test_alternative_answer_colon_is_authoritative_second_best_declaration(question):
    choices = {"B": "Spina bifida", "C": "Maternal NSAIDs"}
    raw = """1. Facts\nOligohydramnios question.\n2. Implications\nOne option is the exception.\n3. Provisional answer\nBest answer: B. Spina bifida.\n4. Best alternative\nAlternative answer: C. Maternal NSAIDs. This is less suitable than B.\n5. Decisive distinction\nB fits the test framing better.\n6. What would change the answer\nDrug exposure emphasis would make C win.\n7. Reconsideration\nRetain B.\n8. Final answer\nB. Spina bifida\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_second_best_option == "C"


def test_labeled_option_that_seems_universally_applicable_can_conclude_provisional(question):
    choices = {"B": "Gram positive bacilli", "D": "Palatal paralysis"}
    raw = """1. Facts\nDiphtheria statements.\n2. Implications\nSeveral statements are true.\n3. Provisional answer\nOption B is factually correct. Option D is also correct. Given this, B seems universally applicable and directly answers the basic cause question.\n4. Best alternative\nThe best alternative to B would be D.\n5. Decisive distinction\nB is more fundamental.\n6. What would change the answer\nA complication-focused stem would favor D.\n7. Reconsideration\nRetain B.\n8. Final answer\nB. Gram positive bacilli\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "B"
    assert recovered.semantic_second_best_option == "D"


def test_between_these_later_better_choice_revises_initial_provisional_label(question):
    choices = {"E": "< 7", "F": "< 9"}
    raw = """1. Facts\nCAD with transfusion question.\n2. Implications\nA higher threshold may be appropriate.\n3. Provisional answer\nE. < 7 is the general restrictive threshold. F. < 9 fits cardiovascular disease. Between these, F is the better choice for a patient with CAD.\n4. Best alternative\nE. < 7 is the best alternative.\n5. Decisive distinction\nCAD favors F.\n6. What would change the answer\nNo CAD would make E win.\n7. Reconsideration\nRetain F.\n8. Final answer\nF. < 9\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_provisional_option == "F"
    assert recovered.semantic_second_best_option == "E"


def test_second_best_different_answer_that_could_be_considered_is_choice(question):
    choices = {"E": "Nephropathy", "J": "Retinopathy"}
    raw = """1. Facts\nPregnancy with diabetic complications.\n2. Implications\nRetinopathy can worsen.\n3. Provisional answer\nJ. Retinopathy is the best answer.\n4. Best alternative\nA different answer that could be considered is E. Nephropathy. It is less likely to progress as rapidly.\n5. Decisive distinction\nPregnancy rapidly worsens retinopathy.\n6. What would change the answer\nMore severe renal disease would make E win.\n7. Reconsideration\nThe pregnancy-specific risk makes J. Retinopathy a compelling choice.\n8. Final answer\nJ. Retinopathy\n"""
    recovered = recover_protocol(raw, choices)
    assert recovered.semantic_second_best_option == "E"
    assert recovered.semantic_decision == "retain"


def test_counterfactual_might_make_declared_alternative_more_suitable(question, lexical_reference):
    choices = {"A": "Sudden sensorineural hearing loss", "B": "Alternative regimen"}
    raw = """1. Facts
Acute unilateral hearing loss.
2. Implications
The current facts favor A.
3. Provisional answer
A. Sudden sensorineural hearing loss is preferred because it fits the acute pattern.
4. Best alternative
B. Alternative regimen is a different alternative but less suitable now.
5. Decisive distinction
The current acute pattern favors A.
6. What would change the answer
If the acute pattern were absent, option B might be more suitable because the original discriminator would disappear.
7. Reconsideration
Option A makes it highly suitable for the current acute pattern.
8. Final answer
A. Sudden sensorineural hearing loss
"""
    custom = type(question)(
        id=question.id,
        dataset=question.dataset,
        stem=question.stem,
        choices=choices,
        gold="A",
        gold_text=choices["A"],
        gold_source=question.gold_source,
        gold_raw="A",
        source_split=question.source_split,
    )
    result = judge(custom, "completed", raw, "system", lexical_reference)
    assert result["answer_change_discussed"] == 1.0
    assert result["reconsideration_discussed"] == 1.0
    assert result["rereasoning_decision"] == "retain"


def test_early_commitment_rejects_hedged_differential_diagnosis(question):
    text = (
        "Diagnosis: Sudden sensorineural hearing loss is possible given the timing, "
        "but the evidence is not yet decisive."
    )
    assert _commitments(text, question.choices) == []


def test_early_commitment_rejects_question_restatement(question):
    text = (
        "Key question: What is the most appropriate next step? We should compare the options first."
    )
    assert _commitments(text, question.choices) == []


def test_early_commitment_keeps_firm_choice_text_selection(question):
    events = _commitments(
        "Sudden sensorineural hearing loss is the most likely diagnosis in this case.",
        question.choices,
    )
    assert any(event["option"] == "A" for event in events)


def test_second_best_leading_option_survives_less_suitable_rationale(question):
    from instruction_duplication.trajectory import _semantic_option

    text = (
        "Option B: Cerumen impaction. This is incorrect for the sudden sensorineural pattern, "
        "but it is the least-wrong alternative because it can also cause hearing loss."
    )
    assert _semantic_option(text, question.choices, role="second_best") == "B"


def test_natural_strong_alternative_with_parenthesized_label_is_recovered(question):
    from instruction_duplication.trajectory import _semantic_option

    text = (
        "A strong alternative to Sudden sensorineural hearing loss could be "
        "Cerumen impaction (B), which can cause hearing loss but not this abrupt pattern."
    )
    assert _semantic_option(text, question.choices, role="second_best") == "B"


def test_parenthesized_strong_candidate_is_recovered_as_provisional(question):
    from instruction_duplication.trajectory import _semantic_option

    text = (
        "Among the options, Cerumen impaction (B) is a strong candidate because it can cause "
        "hearing loss, although the timing is unusual."
    )
    assert _semantic_option(text, question.choices, role="provisional_answer") == "B"
