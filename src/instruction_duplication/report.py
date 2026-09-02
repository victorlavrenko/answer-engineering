"""Readable text rendering for the versioned statistical analysis."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence

from .json_types import is_object_sequence, is_string_mapping, number_value, object_value

LABELS = {
    "required_section_count": "Required sections present (of 8)",
    "nontrivial_section_count": "Required sections with non-trivial content (of 8)",
    "contrastive_discussion_count": "Provisional/contrastive stages substantively discussed (of 5)",
    "contrastive_discussion_score": "Provisional/contrastive discussion score",
    "validated_contrastive_discussion_count": "Validated provisional/contrastive stages completed (of 5)",
    "validated_contrastive_discussion_score": "Validated provisional/contrastive discussion score",
    "provisional_answer_discussed": "Provisional answer actually selected and explained",
    "best_alternative_discussed": "Best alternative actually selected and explained",
    "decisive_distinction_discussed": "Decisive distinction actually discussed",
    "answer_change_discussed": "Answer-changing counterfactual actually discussed",
    "reconsideration_discussed": "Reconsideration actually performed",
    "identified_role_count": "Identifiable requested roles (of 8)",
    "unique_role_count": "Uniquely identifiable requested roles (of 8)",
    "substantive_role_count": "Substantively completed reasoning roles (of 8; exploratory semantic Step 6)",
    "validated_role_count": "Roles meeting validated completion criteria (of 8)",
    "validated_role_completeness_score": "Validated role-completion fraction",
    "role_completeness_score": "Substantive role-completion fraction",
    "all_roles_substantive": "All 8 roles substantively completed (exploratory semantic Step 6)",
    "all_roles_validated_complete": "All 8 roles meet validated completion criteria",
    "roles_in_requested_order": "All 8 roles present once in requested order",
    "complete_role_scaffold": "Complete ordered substantive role scaffold (exploratory semantic Step 6)",
    "validated_complete_role_scaffold": "Complete ordered validated role scaffold",
    "role_facts_complete": "Facts role substantive",
    "role_implications_complete": "Implications role substantive",
    "role_provisional_answer_complete": "Provisional-answer role substantive",
    "role_best_alternative_complete": "Best-alternative role substantive",
    "role_decisive_distinction_complete": "Decisive-distinction role substantive",
    "role_answer_changing_change_complete": "What-would-change semantic completion (exploratory)",
    "role_answer_changing_change_nontrivial": "What-would-change section has non-trivial content",
    "role_reconsideration_complete": "Reconsideration role substantive",
    "role_final_answer_complete": "Final-answer role substantive",
    "protocol_scaffold_complete": "Full format-neutral reasoning scaffold",
    "preanswer_discussion_complete": "Pre-answer discussion scaffold",
    "visible_reasoning_scaffold_complete": "Complete format-neutral reasoning scaffold",
    "visible_preanswer_discussion_complete": ("Visible pre-answer discussion (format-tolerant)"),
    "minimum_repair_scaffold": "Repair-eligible full protocol scaffold",
    "contrastive_scaffold_complete": "Contrastive/rereasoning scaffold",
    "correct_with_minimum_repair_scaffold": "Correct answer with minimum repair scaffold",
    "protocol_format_score": "Voluntary XML-serialization diagnostic",
    "structure_score": "Clearly segmentable role structure",
    "section_depth_score": "Nontrivial section-depth score",
    "substantive_role_score": "Section-substance score",
    "contrastive_role_score": "Contrastive-role score",
    "visible_substantive_role_score": "Section-substance score (all instructed questions)",
    "visible_contrastive_role_score": "Contrastive-role score (all instructed questions)",
    "facts_tfidf_recall": "TF-IDF stem recall in facts",
    "implications_tfidf_recall": "TF-IDF stem recall in implications",
    "preanswer_tfidf_recall": "Facts + implications TF-IDF stem recall",
    "preprovisional_tfidf_recall": "Pre-provisional TF-IDF stem recall",
    "preanswer_high_idf_tfidf_recall": "Pre-answer high-IDF stem recall",
    "facts_implications_shared_tfidf_recall": "TF-IDF anchors shared by facts/implications",
    "facts_implications_shared_high_idf_recall": ("High-IDF anchors shared by facts/implications"),
    "preanswer_tfidf_density_per_100_tokens": "Pre-answer TF-IDF mass per 100 content tokens",
    "preanswer_tfidf_token_count": "Pre-answer content-token count",
    "section_present_count": "Identifiable requested roles (of 8)",
    "section_unique_count": "Uniquely identifiable requested roles (of 8)",
    "section_nontrivial_count": "Required sections with non-trivial content (of 8)",
    "section_substantive_count": "Substantive section roles (of 8)",
    "section_schema_valid_count": "Voluntary XML markers matching the legacy schema (of 8)",
    "trajectory_complete": "Format-neutral trajectory structure complete",
    "semantic_trajectory_complete": "Format-neutral trajectory structure complete",
    "facts_choice_list_leakage": "Explicit choice-list leakage in facts",
    "accuracy_parseable": "Answer parse rate",
    "generation_usable": "Successful generation rate",
    "facts_anchor_recall": "Polarity-aware facts anchor recall",
    "shared_anchor_recall": "Polarity-aware shared anchor recall",
    "atomic_fact_coverage": "Atomic facts represented",
    "atomic_implication_trace_coverage": "Atomic facts traced into implications",
    "hard_qualifier_fact_recall": "Hard qualifiers preserved in facts",
    "hard_qualifier_trace_recall": "Hard qualifiers preserved in implications",
    "preprovisional_commitment_observed": "Observed pre-provisional answer commitment",
    "preprovisional_commitment_itt": "Pre-provisional commitment or generation failure (ITT adverse)",
    "implications_anchor_recall": "Polarity-aware implications anchor recall",
    "preanswer_anchor_recall": "Polarity-aware combined pre-answer anchor recall",
    "accuracy": "Accuracy (ITT)",
}

HEADLINE_MODEL_OUTCOMES = (
    "required_section_count",
    "nontrivial_section_count",
    "preprovisional_tfidf_recall",
    "validated_contrastive_discussion_score",
    "validated_role_count",
    "all_roles_validated_complete",
    "validated_complete_role_scaffold",
    "role_facts_complete",
    "role_implications_complete",
    "role_provisional_answer_complete",
    "role_best_alternative_complete",
    "role_decisive_distinction_complete",
    "role_answer_changing_change_nontrivial",
    "role_reconsideration_complete",
    "role_final_answer_complete",
    "preanswer_tfidf_recall",
    "preprovisional_commitment_itt",
    "accuracy",
)
LOWER_IS_BETTER_OUTCOMES = {"preprovisional_commitment_itt"}


def _value(value: object, digits: int = 4) -> str:
    return "NA" if value is None else f"{number_value(value, name='report value'):.{digits}f}"


def _p(value: object) -> str:
    if value is None:
        return "NA"
    number = number_value(value, name="p-value")
    return "<0.0001" if number < 0.0001 else f"{number:.4f}"


def _object(value: object, name: str) -> Mapping[str, object]:
    return object_value(value, name=name)


def _object_list(value: object, name: str) -> list[Mapping[str, object]]:
    if not is_object_sequence(value):
        raise ValueError(f"{name} must be a list")
    return [_object(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _contrast_lines(item: Mapping[str, object], *, adjusted: bool = True) -> list[str]:
    outcome = str(item["outcome"])
    label = LABELS.get(outcome, outcome)
    confidence = int(number_value(item.get("confidence_level", 0.95), name="confidence") * 100)
    p_label = "raw p" if adjusted else "p"
    line = (
        f"  {label}: treatment={_value(item.get('treatment_mean'))}; "
        f"control={_value(item.get('control_mean'))}; effect={_value(item.get('effect'))}; "
        f"{confidence}% CI [{_value(item.get('ci_low'))}, {_value(item.get('ci_high'))}]; "
        f"{p_label}={_p(item.get('p_value'))}"
    )
    if adjusted:
        line += f"; Holm p={_p(item.get('holm_p_value'))}"
    line += (
        f"; question clusters={item.get('question_clusters', 0)}; "
        f"incomplete model-question units={item.get('incomplete_model_question_units', 0)}"
    )
    return [line]


def _section(
    title: str,
    items: Sequence[Mapping[str, object]],
    *,
    adjusted: bool = True,
) -> list[str]:
    lines = [title]
    if not items:
        lines.append("  No estimable contrasts.")
    for item in items:
        lines.extend(_contrast_lines(item, adjusted=adjusted))
    lines.append("")
    return lines


def _effect_direction(outcome: str) -> str:
    return "lower_is_better" if outcome in LOWER_IS_BETTER_OUTCOMES else "higher_is_better"


def _benefit_adjusted_effect(item: Mapping[str, object]) -> float | None:
    value = item.get("effect")
    if value is None:
        return None
    effect = number_value(value, name="model effect")
    return -effect if str(item["outcome"]) in LOWER_IS_BETTER_OUTCOMES else effect


def _rank_model_effects(
    items: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    def sort_key(item: Mapping[str, object]) -> tuple[float, str]:
        adjusted = _benefit_adjusted_effect(item)
        if adjusted is None:
            raise ValueError("ranked model effect cannot be missing")
        return -adjusted, str(item.get("model_id", ""))

    return sorted(
        (item for item in items if _benefit_adjusted_effect(item) is not None),
        key=sort_key,
    )


def _csv_number(value: object) -> str:
    if value is None:
        return ""
    return f"{number_value(value, name='CSV number'):.12g}"


def _all_model_effects(analysis: Mapping[str, object]) -> list[Mapping[str, object]]:
    return _object_list(
        analysis["model_specific_repetition_all_metrics"],
        "analysis.model_specific_repetition_all_metrics",
    )


def render_model_effects_csv(analysis: Mapping[str, object]) -> str:
    """Render a tidy complete model-by-metric repetition-effect table."""
    items = _all_model_effects(analysis)
    by_outcome: dict[str, list[Mapping[str, object]]] = {}
    for item in items:
        by_outcome.setdefault(str(item["outcome"]), []).append(item)
    ranks = {
        (outcome, str(item["model_id"])): rank
        for outcome, outcome_items in by_outcome.items()
        for rank, item in enumerate(_rank_model_effects(outcome_items), start=1)
    }
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "outcome",
            "label",
            "direction",
            "model_id",
            "control_mean",
            "treatment_mean",
            "effect",
            "benefit_adjusted_gain",
            "benefit_rank",
            "ci_low",
            "ci_high",
            "p_value",
            "holm_p_value",
            "question_clusters",
            "incomplete_model_question_units",
        )
    )
    for item in items:
        outcome = str(item["outcome"])
        adjusted = _benefit_adjusted_effect(item)
        writer.writerow(
            (
                outcome,
                LABELS.get(outcome, outcome),
                _effect_direction(outcome),
                item["model_id"],
                _csv_number(item.get("control_mean")),
                _csv_number(item.get("treatment_mean")),
                _csv_number(item.get("effect")),
                _csv_number(adjusted),
                ranks.get((outcome, str(item["model_id"])), ""),
                _csv_number(item.get("ci_low")),
                _csv_number(item.get("ci_high")),
                _csv_number(item.get("p_value")),
                _csv_number(item.get("holm_p_value")),
                item.get("question_clusters", 0),
                item.get("incomplete_model_question_units", 0),
            )
        )
    return output.getvalue()


def render_model_effect_summary_csv(analysis: Mapping[str, object]) -> str:
    """Render one row per metric with model effects and automatic best/worst columns."""
    items = _all_model_effects(analysis)
    outcomes = list(dict.fromkeys(str(item["outcome"]) for item in items))
    models = sorted({str(item["model_id"]) for item in items})
    lookup = {(str(item["outcome"]), str(item["model_id"])): item for item in items}
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "outcome",
            "label",
            "direction",
            *models,
            "best_model",
            "best_effect",
            "worst_model",
            "worst_effect",
        )
    )
    for outcome in outcomes:
        ranked = _rank_model_effects([lookup[(outcome, model)] for model in models])
        best = ranked[0] if ranked else None
        worst = ranked[-1] if ranked else None
        writer.writerow(
            (
                outcome,
                LABELS.get(outcome, outcome),
                _effect_direction(outcome),
                *(_csv_number(lookup[(outcome, model)].get("effect")) for model in models),
                "" if best is None else best["model_id"],
                "" if best is None else _csv_number(best.get("effect")),
                "" if worst is None else worst["model_id"],
                "" if worst is None else _csv_number(worst.get("effect")),
            )
        )
    return output.getvalue()


def _headline_model_rank_lines(items: Sequence[Mapping[str, object]]) -> list[str]:
    by_outcome: dict[str, list[Mapping[str, object]]] = {}
    for item in items:
        by_outcome.setdefault(str(item["outcome"]), []).append(item)
    lines = [
        "HEADLINE MODEL EFFECT RANKINGS",
        "  Effects are two-copy minus one-copy; lower is better only for the adverse "
        "premature-commitment endpoint.",
    ]
    for outcome in HEADLINE_MODEL_OUTCOMES:
        ranked = _rank_model_effects(by_outcome.get(outcome, []))
        if not ranked:
            continue
        best = ranked[0]
        worst = ranked[-1]
        lines.append(
            f"  {LABELS[outcome]}: best={best['model_id']} ({_value(best.get('effect'))}); "
            f"worst={worst['model_id']} ({_value(worst.get('effect'))})"
        )
    lines.extend(
        [
            "  Complete per-model statistics: model-effects.csv; one-row-per-metric matrix: "
            "model-effect-summary.csv.",
            "",
        ]
    )
    return lines


def render_report(analysis: Mapping[str, object]) -> str:
    """Render means, effects, denominators, and correction families."""
    sample = _object(analysis["sample"], "analysis.sample")
    status_counts = _object(sample["status_counts"], "analysis.sample.status_counts")
    estimand = _object(analysis["estimand"], "analysis.estimand")
    primary = _object_list(
        analysis["primary_repetition_contrasts"], "analysis.primary_repetition_contrasts"
    )
    repetition_length = _object_list(
        analysis["repetition_length_robustness"],
        "analysis.repetition_length_robustness",
    )
    role_diagnostics = _object_list(
        analysis["role_diagnostic_contrasts"], "analysis.role_diagnostic_contrasts"
    )
    trailing = _object_list(analysis["trailing_copy_contrasts"], "analysis.trailing_copy_contrasts")
    trailing_length = _object_list(
        analysis["trailing_copy_length_robustness"],
        "analysis.trailing_copy_length_robustness",
    )
    one_copy_after = _object_list(
        analysis["one_copy_after_query_contrasts"], "analysis.one_copy_after_query_contrasts"
    )
    one_copy_after_length = _object_list(
        analysis["one_copy_after_query_length_robustness"],
        "analysis.one_copy_after_query_length_robustness",
    )
    one_copy_after_vs_before = _object_list(
        analysis["one_copy_after_vs_before_contrasts"],
        "analysis.one_copy_after_vs_before_contrasts",
    )
    one_copy_after_vs_before_length = _object_list(
        analysis["one_copy_after_vs_before_length_robustness"],
        "analysis.one_copy_after_vs_before_length_robustness",
    )
    one_copy_after_vs_system = _object_list(
        analysis["one_copy_after_vs_system_contrasts"],
        "analysis.one_copy_after_vs_system_contrasts",
    )
    one_copy_after_vs_system_length = _object_list(
        analysis["one_copy_after_vs_system_length_robustness"],
        "analysis.one_copy_after_vs_system_length_robustness",
    )
    exploratory = _object_list(analysis["exploratory_contrasts"], "analysis.exploratory_contrasts")
    sensitivity = _object_list(
        analysis["per_protocol_sensitivity"], "analysis.per_protocol_sensitivity"
    )
    model_repetition = _object_list(
        analysis["model_specific_repetition_contrasts"],
        "analysis.model_specific_repetition_contrasts",
    )
    model_robustness = _object_list(
        analysis["model_specific_repetition_robustness"],
        "analysis.model_specific_repetition_robustness",
    )
    model_all_metrics = _all_model_effects(analysis)
    model_trailing = _object_list(
        analysis["model_specific_trailing_contrasts"],
        "analysis.model_specific_trailing_contrasts",
    )
    model_one_copy_after = _object_list(
        analysis["model_specific_one_copy_after_contrasts"],
        "analysis.model_specific_one_copy_after_contrasts",
    )
    dataset_repetition = _object_list(
        analysis["dataset_specific_repetition_contrasts"],
        "analysis.dataset_specific_repetition_contrasts",
    )
    dataset_trailing = _object_list(
        analysis["dataset_specific_trailing_contrasts"],
        "analysis.dataset_specific_trailing_contrasts",
    )
    dataset_one_copy_after = _object_list(
        analysis["dataset_specific_one_copy_after_contrasts"],
        "analysis.dataset_specific_one_copy_after_contrasts",
    )
    accuracy = _object(analysis["accuracy_contrast"], "analysis.accuracy_contrast")
    confirmatory = _object(
        analysis["confirmatory_primary_contrast"],
        "analysis.confirmatory_primary_contrast",
    )
    key_secondary = _object_list(
        analysis["key_secondary_contrasts"], "analysis.key_secondary_contrasts"
    )
    lexical = _object(analysis["lexical_measurement"], "analysis.lexical_measurement")
    source = _object(lexical["source"], "analysis.lexical_measurement.source")

    lines = [
        "INSTRUCTION DUPLICATION EXPERIMENT — VERSIONED ANALYSIS",
        "",
        "SAMPLE",
        f"  Models: {sample['models']}",
        f"  Datasets: {sample['datasets']}",
        f"  Questions: {sample['questions']}",
        f"  Planned cells: {sample['cells']}",
        f"  Usable generations: {sample['usable_cells']}",
        "  Status counts: " + ", ".join(f"{key}={value}" for key, value in status_counts.items()),
        "",
        "ESTIMAND",
        f"  {estimand['primary']}",
        f"  {estimand['failure_policy']}",
        "",
        "LEXICAL MEASUREMENT",
        f"  Version: {lexical['lexical_version']}",
        f"  Reference: {lexical['reference_scope']} (documents={lexical['document_count']})",
        f"  IDF: {lexical['idf_formula']}",
        f"  Source version: {source.get('version', source.get('kind'))}",
        f"  Source commit: {source.get('commit', 'NA')}",
        f"  Source SHA-256: {source.get('sha256', 'NA')}",
        f"  Reference SHA-256: {lexical['reference_hash']}",
        "",
    ]
    lines += [
        "DECLARED CONFIRMATORY PRIMARY ENDPOINT",
        *_contrast_lines(confirmatory, adjusted=False),
        "",
    ]
    lines += _section(
        "KEY SECONDARY ENDPOINT FAMILY (Holm-adjusted across five endpoints)",
        key_secondary,
        adjusted=True,
    )
    lines += _section(
        "PRIMARY MEASUREMENT LAYERS (unadjusted; interpreted separately)",
        primary,
        adjusted=False,
    )
    lines += _section(
        "TWO-COPY VS ONE-COPY LENGTH/DENSITY ROBUSTNESS (unadjusted)",
        repetition_length,
        adjusted=False,
    )
    lines += _section(
        "ROLE-COMPLETENESS DIAGNOSTICS (unadjusted; interpreted separately)",
        role_diagnostics,
        adjusted=False,
    )
    lines += _section(
        "TRAILING-COPY CONTRASTS (placement analysis; unadjusted)",
        trailing,
        adjusted=False,
    )
    lines += _section(
        "TRAILING-COPY LENGTH ROBUSTNESS (unadjusted)",
        trailing_length,
        adjusted=False,
    )
    lines += _section(
        "ONE-COPY AFTER-QUERY PLACEMENT (after vs mean(system, before); unadjusted)",
        one_copy_after,
        adjusted=False,
    )
    lines += _section(
        "ONE-COPY AFTER-QUERY LENGTH ROBUSTNESS (unadjusted)",
        one_copy_after_length,
        adjusted=False,
    )
    lines += _section(
        "ONE-COPY PLACEMENT DIAGNOSTIC (after vs before; unadjusted)",
        one_copy_after_vs_before,
        adjusted=False,
    )
    lines += _section(
        "ONE-COPY AFTER VS BEFORE LENGTH ROBUSTNESS (unadjusted)",
        one_copy_after_vs_before_length,
        adjusted=False,
    )
    lines += _section(
        "ONE-COPY PLACEMENT DIAGNOSTIC (after vs system; unadjusted)",
        one_copy_after_vs_system,
        adjusted=False,
    )
    lines += _section(
        "ONE-COPY AFTER VS SYSTEM LENGTH ROBUSTNESS (unadjusted)",
        one_copy_after_vs_system_length,
        adjusted=False,
    )
    lines += _section(
        "EXPLORATORY DIAGNOSTICS (unadjusted; not confirmatory)",
        exploratory,
        adjusted=False,
    )
    lines += ["ACCURACY", *_contrast_lines(accuracy, adjusted=False), ""]
    lines += _headline_model_rank_lines(model_all_metrics)
    lines += _section(
        "PER-PROTOCOL SENSITIVITY (successful generations only; unadjusted)",
        sensitivity,
        adjusted=False,
    )

    for title, items in (
        ("MODEL-SPECIFIC REPETITION HETEROGENEITY", model_repetition),
        ("MODEL-SPECIFIC TRAILING-COPY HETEROGENEITY", model_trailing),
        ("MODEL-SPECIFIC ONE-COPY AFTER-QUERY HETEROGENEITY", model_one_copy_after),
    ):
        lines.append(title + " (Holm-adjusted across models within endpoint)")
        for item in items:
            lines.append(f"  [{item['model_id']}] " + _contrast_lines(item)[0].strip())
        lines.append("")
    lines.append(
        "MODEL-SPECIFIC REPETITION ROBUSTNESS (Holm-adjusted across models within endpoint)"
    )
    for item in model_robustness:
        lines.append(f"  [{item['model_id']}] " + _contrast_lines(item)[0].strip())
    lines.append("")
    for title, items in (
        ("DATASET-SPECIFIC REPETITION HETEROGENEITY", dataset_repetition),
        ("DATASET-SPECIFIC TRAILING-COPY HETEROGENEITY", dataset_trailing),
        ("DATASET-SPECIFIC ONE-COPY AFTER-QUERY HETEROGENEITY", dataset_one_copy_after),
    ):
        lines.append(title + " (unadjusted; descriptive)")
        for item in items:
            lines.append(
                f"  [{item['dataset']}] " + _contrast_lines(item, adjusted=False)[0].strip()
            )
        lines.append("")

    complexity_value = analysis.get("question_complexity_subgroup_contrasts")
    if is_string_mapping(complexity_value):
        complexities = _object(complexity_value, "analysis.question_complexity")
        lines.append("QUESTION-COMPLEXITY SUBGROUPS (unadjusted; exploratory)")
        for complexity, raw_items in complexities.items():
            items = _object_list(raw_items, f"analysis.question_complexity.{complexity}")
            for item in items:
                lines.append(
                    f"  [{complexity}] " + _contrast_lines(item, adjusted=False)[0].strip()
                )
        lines.append("")

    factorial = _object_list(
        analysis.get("factorial_decomposition", []),
        "analysis.factorial_decomposition",
    )
    lines.append("FULL 2x2x2 FACTORIAL DECOMPOSITION (unadjusted; exploratory)")
    for item in factorial:
        confidence = int(number_value(item.get("confidence_level", 0.95), name="confidence") * 100)
        lines.append(
            f"  [{item['outcome']}; {item['factorial_term']}] "
            f"effect={_value(item.get('effect'))}; {confidence}% CI "
            f"[{_value(item.get('ci_low'))}, {_value(item.get('ci_high'))}]; "
            f"p={_p(item.get('p_value'))}; clusters={item.get('question_clusters', 0)}"
        )
    lines.append("")

    quality_value = analysis.get("question_quality")
    if is_string_mapping(quality_value):
        quality = _object(quality_value, "analysis.question_quality")
        lines.extend(
            [
                "QUESTION QUALITY AND APPLICABILITY",
                f"  Repair-endpoint eligible: {quality.get('repair_endpoint_eligible', 0)}",
                f"  Repair-endpoint inapplicable: {quality.get('repair_endpoint_inapplicable', 0)}",
                f"  Targeted-review flags: {quality.get('targeted_review_flags', 0)}",
                "  Automatic fact inventories are secondary measurements until human validation.",
                "",
            ]
        )
    human_value = analysis.get("human_validation")
    if is_string_mapping(human_value):
        human = _object(human_value, "analysis.human_validation")
        lines.extend(
            [
                "BLINDED HUMAN VALIDATION EXPORT",
                f"  Candidate matched pairs: {human.get('candidate_pairs', 0)}",
                f"  Exported matched pairs: {human.get('exported_pairs', 0)}",
                "  Condition key is stored separately and should remain hidden until ratings freeze.",
                "",
            ]
        )

    condition_means = _object(analysis["condition_means"], "analysis.condition_means")
    lines.append("CONDITION MEANS (mean; n)")
    for condition, raw_metrics in condition_means.items():
        metrics = _object(raw_metrics, f"analysis.condition_means.{condition}")
        rendered_parts: list[str] = []
        for outcome, raw_values in metrics.items():
            values = _object(raw_values, f"analysis.condition_means.{condition}.{outcome}")
            rendered_parts.append(
                f"{LABELS.get(outcome, outcome)}={_value(values['mean'])};n={values['n']}"
            )
        lines.append(f"  {condition}: {', '.join(rendered_parts)}")
    lines.append("")
    lines.append(
        "Paper-facing validated role completion uses seven human-validated semantic role "
        "judges plus a deliberately simple Step-6 observable: the identified What-would-change "
        "section contains non-trivial generated content. The stricter semantic Step-6 "
        "counterfactual judge is retained only as an exploratory diagnostic and does not enter "
        "the validated aggregates. Presence, uniqueness, order, lexical exposure, premature "
        "commitment, generation success, and answer accuracy remain separate measurements. "
        "TF-IDF and automatic fact/qualifier checks measure visible lexical exposure, not medical truth."
    )
    return "\n".join(lines).rstrip() + "\n"


def render_paper_report(analysis: Mapping[str, object]) -> str:
    """Render only the results and qualifications normally needed in a paper."""
    sample = _object(analysis["sample"], "analysis.sample")
    status_counts = _object(sample["status_counts"], "analysis.sample.status_counts")
    primary = _object(
        analysis["confirmatory_primary_contrast"],
        "analysis.confirmatory_primary_contrast",
    )
    secondary = _object_list(
        analysis["key_secondary_contrasts"], "analysis.key_secondary_contrasts"
    )
    quality_value = analysis.get("question_quality", {})
    quality: Mapping[str, object] = (
        _object(quality_value, "analysis.question_quality")
        if is_string_mapping(quality_value)
        else {}
    )
    lexical = _object(analysis["lexical_measurement"], "analysis.lexical_measurement")
    repetition_length = _object_list(
        analysis["repetition_length_robustness"],
        "analysis.repetition_length_robustness",
    )
    role_diagnostics = _object_list(
        analysis["role_diagnostic_contrasts"],
        "analysis.role_diagnostic_contrasts",
    )
    model_repetition = _all_model_effects(analysis)
    lines = [
        "INSTRUCTION DUPLICATION EXPERIMENT — PAPER REPORT",
        "",
        "SAMPLE",
        (
            f"  {sample['questions']} questions, {sample['models']} models, "
            f"{sample['datasets']} datasets, {sample['cells']} planned cells."
        ),
        (
            f"  Usable generations: {sample['usable_cells']}/{sample['cells']}; "
            + ", ".join(f"{key}={value}" for key, value in status_counts.items())
        ),
        (
            "  Repair-endpoint eligible questions: "
            f"{quality.get('repair_endpoint_eligible', 'NA')}/{quality.get('questions', 'NA')}."
        ),
        "",
        "CONFIRMATORY PRIMARY: TWO COPIES VS ONE COPY",
        *_contrast_lines(primary, adjusted=False),
        "",
        "KEY SECONDARY ENDPOINTS (Holm-adjusted across five endpoints)",
    ]
    for item in secondary:
        lines.extend(_contrast_lines(item, adjusted=True))
    lines.extend(["", "ROBUSTNESS: TWO COPIES VS ONE COPY (unadjusted)"])
    for item in repetition_length:
        lines.extend(_contrast_lines(item, adjusted=False))
    focused_role_diagnostics = [
        item
        for item in role_diagnostics
        if item.get("outcome")
        in {
            "identified_role_count",
            "unique_role_count",
            "validated_role_count",
            "all_roles_validated_complete",
            "roles_in_requested_order",
            "validated_complete_role_scaffold",
            "role_facts_complete",
            "role_implications_complete",
            "role_provisional_answer_complete",
            "role_best_alternative_complete",
            "role_decisive_distinction_complete",
            "role_answer_changing_change_nontrivial",
            "role_reconsideration_complete",
            "role_final_answer_complete",
        }
    ]
    lines.extend(["", "ROLE-COMPLETENESS COMPONENTS (unadjusted)"])
    for item in focused_role_diagnostics:
        lines.extend(_contrast_lines(item, adjusted=False))

    model_focus_order = {
        outcome: index
        for index, outcome in enumerate(
            (
                "validated_role_count",
                "all_roles_validated_complete",
                "validated_complete_role_scaffold",
                "role_facts_complete",
                "role_implications_complete",
                "role_provisional_answer_complete",
                "role_best_alternative_complete",
                "role_decisive_distinction_complete",
                "role_answer_changing_change_nontrivial",
                "role_reconsideration_complete",
                "role_final_answer_complete",
                "preanswer_tfidf_recall",
                "preprovisional_commitment_itt",
                "accuracy",
            )
        )
    }
    model_focus = [item for item in model_repetition if item.get("outcome") in model_focus_order]
    model_focus.sort(
        key=lambda item: (
            model_focus_order[str(item["outcome"])],
            str(item.get("model_id", "")),
        )
    )
    lines.extend(
        [
            "",
            "MODEL-SPECIFIC ROBUSTNESS (Holm-adjusted across models within endpoint)",
        ]
    )
    lines.extend(
        f"  [{item['model_id']}] " + _contrast_lines(item)[0].strip() for item in model_focus
    )
    lines.extend(
        [
            "",
            "METHOD NOTE",
            (
                "  Lexical exposure uses frozen global PubMed document frequencies "
                f"({lexical['document_count']} abstracts), not frequencies from this sample."
            ),
            (
                "  Paper-facing validated role metrics use seven semantic role judges plus "
                "non-trivial Step-6 content. The stricter semantic Step-6 counterfactual judge "
                "is retained as an exploratory diagnostic because it did not generalize reliably "
                "under blinded human validation. No medical-correctness judgment is implied."
            ),
            (
                "  Role recognition accepts plain, numbered, Markdown/emphasized, or voluntarily "
                "emitted XML headings. Section presence is scored separately from content depth "
                "and from semantic role completion."
            ),
            (
                "  TF-IDF, automatic fact coverage, and qualifier checks measure visible "
                "lexical/structural exposure, not medical truth; use the blinded audit export "
                "for human validation. No cross-component AE score is constructed."
            ),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
