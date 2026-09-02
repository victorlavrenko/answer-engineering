from __future__ import annotations

import pytest

from instruction_duplication.json_types import is_object_sequence, object_value
from instruction_duplication.records import AnalysisRow
from instruction_duplication.stats import (
    DOUBLE,
    MODEL_COMPARISON_OUTCOMES,
    SINGLE,
    TRAILING_CONTROL,
    TRAILING_TREATMENT,
    analysis_work_units,
    bootstrap_ci,
    build_analysis,
    contrast,
    holm_adjust,
    sign_flip_p_value,
)


def rows_with_failure():
    rows = []
    conditions = SINGLE + DOUBLE
    for qid in ("q1", "q2"):
        for condition in conditions:
            value = 1.0
            usable = True
            if qid == "q2" and condition == "system_before":
                value = 0.0
                usable = False
            rows.append(
                AnalysisRow(
                    question_id=qid,
                    model_id="m1",
                    dataset="d1",
                    condition_id=condition,
                    status="completed" if usable else "failed",
                    judgment={
                        "identified_role_count": 8.0 * value,
                        "unique_role_count": 8.0 * value,
                        "substantive_role_count": 8.0 * value,
                        "validated_role_count": 8.0 * value,
                        "validated_role_completeness_score": value,
                        "role_completeness_score": value,
                        "all_roles_substantive": value,
                        "all_roles_validated_complete": value,
                        "validated_complete_role_scaffold": value,
                        "roles_in_requested_order": value,
                        "complete_role_scaffold": value,
                        "role_facts_complete": value,
                        "role_implications_complete": value,
                        "role_provisional_answer_complete": value,
                        "role_best_alternative_complete": value,
                        "role_decisive_distinction_complete": value,
                        "role_answer_changing_change_complete": value,
                        "role_answer_changing_change_nontrivial": value,
                        "role_reconsideration_complete": value,
                        "role_final_answer_complete": value,
                        "protocol_scaffold_complete": value,
                        "preanswer_discussion_complete": value,
                        "visible_reasoning_scaffold_complete": value,
                        "visible_preanswer_discussion_complete": value,
                        "minimum_repair_scaffold": value,
                        "contrastive_scaffold_complete": value,
                        "protocol_format_score": value,
                        "section_present_count": 8.0 * value,
                        "section_unique_count": 8.0 * value,
                        "section_substantive_count": 8.0 * value,
                        "section_schema_valid_count": 8.0 * value,
                        "structure_score": value,
                        "section_depth_score": value,
                        "substantive_role_score": value,
                        "contrastive_role_score": value,
                        "validated_contrastive_discussion_count": 5.0 * value,
                        "validated_contrastive_discussion_score": value,
                        "visible_substantive_role_score": value,
                        "visible_contrastive_role_score": value,
                        "facts_tfidf_recall": value,
                        "implications_tfidf_recall": value,
                        "preanswer_tfidf_recall": value,
                        "preanswer_high_idf_tfidf_recall": value,
                        "preanswer_tfidf_density_per_100_tokens": 2.0 * value,
                        "preanswer_tfidf_token_count": 10.0 * value,
                        "facts_implications_shared_tfidf_recall": value,
                        "facts_implications_shared_high_idf_recall": value,
                        "facts_anchor_recall": value,
                        "shared_anchor_recall": value,
                        "preprovisional_commitment_observed": 0.0 if usable else None,
                        "preprovisional_commitment_itt": 0.0 if usable else 1.0,
                        "implications_anchor_recall": value,
                        "preanswer_anchor_recall": value,
                        "trajectory_complete": value,
                        "accuracy": value,
                        "generation_usable": usable,
                    },
                )
            )
    return rows


def test_itt_keeps_question_with_failed_treatment_cell():
    result = contrast(
        rows_with_failure(),
        "minimum_repair_scaffold",
        DOUBLE,
        SINGLE,
        permutations=100,
        bootstraps=100,
        confidence_level=0.95,
        seed=1,
    )
    assert result["question_clusters"] == 2
    assert result["treatment_mean"] == pytest.approx(5 / 6)
    assert result["control_mean"] == 1


def test_per_protocol_is_separate_and_shows_incompleteness():
    result = contrast(
        rows_with_failure(),
        "minimum_repair_scaffold",
        DOUBLE,
        SINGLE,
        permutations=100,
        bootstraps=100,
        confidence_level=0.95,
        seed=1,
        usable_only=True,
    )
    assert result["question_clusters"] == 1
    assert result["incomplete_model_question_units"] == 1


def test_pooled_endpoints_are_unadjusted_and_denominators_present():
    analysis = build_analysis(rows_with_failure(), permutations=100, bootstraps=100)
    for key in (
        "primary_repetition_contrasts",
        "role_diagnostic_contrasts",
        "trailing_copy_contrasts",
        "one_copy_after_query_contrasts",
        "one_copy_after_vs_before_contrasts",
        "one_copy_after_vs_system_contrasts",
        "trailing_copy_length_robustness",
        "one_copy_after_query_length_robustness",
        "one_copy_after_vs_before_length_robustness",
        "one_copy_after_vs_system_length_robustness",
        "exploratory_contrasts",
        "per_protocol_sensitivity",
        "dataset_specific_repetition_contrasts",
        "dataset_specific_trailing_contrasts",
        "dataset_specific_one_copy_after_contrasts",
    ):
        contrasts = analysis[key]
        assert is_object_sequence(contrasts) and contrasts
        assert all(
            "holm_p_value" not in object_value(item, name=f"{key} contrast") for item in contrasts
        )

    accuracy_contrast = object_value(analysis["accuracy_contrast"], name="accuracy contrast")
    assert "holm_p_value" not in accuracy_contrast

    primary = analysis["primary_repetition_contrasts"]
    assert is_object_sequence(primary)
    assert len(primary) == 7
    condition_means = object_value(analysis["condition_means"], name="condition_means")
    system = object_value(condition_means["system"], name="condition_means.system")
    accuracy = object_value(system["accuracy"], name="condition_means.system.accuracy")
    assert accuracy["n"] == 2


def test_model_specific_holm_is_scoped_within_each_endpoint():
    rows = rows_with_failure()
    rows.extend(
        AnalysisRow(
            question_id=row.question_id,
            model_id="m2",
            dataset=row.dataset,
            condition_id=row.condition_id,
            status=row.status,
            judgment=row.judgment,
        )
        for row in rows_with_failure()
    )
    analysis = build_analysis(rows, permutations=100, bootstraps=100)
    for key in (
        "model_specific_repetition_contrasts",
        "model_specific_trailing_contrasts",
        "model_specific_one_copy_after_contrasts",
        "model_specific_repetition_robustness",
        "model_specific_repetition_all_metrics",
    ):
        model_specific = analysis[key]
        assert is_object_sequence(model_specific)
        items = [object_value(item, name=f"{key} contrast") for item in model_specific]
        outcomes = {str(item["outcome"]) for item in items}
        for outcome in outcomes:
            family = [item for item in items if item["outcome"] == outcome]
            assert len(family) == 2
            expected = holm_adjust(
                [float(item["p_value"]) if item["p_value"] is not None else None for item in family]
            )
            actual = [
                float(item["holm_p_value"]) if item["holm_p_value"] is not None else None
                for item in family
            ]
            assert actual == expected

    all_metrics = analysis["model_specific_repetition_all_metrics"]
    assert is_object_sequence(all_metrics)
    all_items = [object_value(item, name="all model metric") for item in all_metrics]
    assert len(all_items) == 2 * len(MODEL_COMPARISON_OUTCOMES)
    assert {str(item["outcome"]) for item in all_items} == set(MODEL_COMPARISON_OUTCOMES)


def test_holm_known_values():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_resampling_parameters_are_validated():
    with pytest.raises(ValueError):
        sign_flip_p_value([1], 0, 1)
    with pytest.raises(ValueError):
        bootstrap_ci([1], 0, 1)
    with pytest.raises(ValueError):
        bootstrap_ci([1], 10, 1, 1.5)


def test_analysis_progress_reaches_exact_work_unit_total_without_changing_results():
    rows = rows_with_failure()
    baseline = build_analysis(rows, permutations=20, bootstraps=20)
    events = []
    instrumented = build_analysis(
        rows,
        permutations=20,
        bootstraps=20,
        progress_sink=events.append,
    )
    expected = analysis_work_units(rows)
    assert instrumented == baseline
    assert events
    assert all(event.total == expected for event in events)
    assert events[-1].completed == expected
    assert events[-1].phase == "dataset-specific one-copy placement heterogeneity"


def test_one_copy_after_query_contrast_uses_only_single_copy_conditions():
    rows = rows_with_failure()
    analysis = build_analysis(rows, permutations=20, bootstraps=20)
    contrasts = analysis["one_copy_after_query_contrasts"]
    assert is_object_sequence(contrasts)
    items = [object_value(item, name="one-copy after-query contrast") for item in contrasts]
    assert {str(item["outcome"]) for item in items} == {
        "required_section_count",
        "nontrivial_section_count",
        "preprovisional_tfidf_recall",
        "validated_contrastive_discussion_score",
        "validated_role_count",
        "all_roles_validated_complete",
        "preprovisional_commitment_itt",
        "trajectory_complete",
        "accuracy",
    }
    for item in items:
        assert item["treatment_conditions"] == ["after"]
        assert item["control_conditions"] == ["system", "before"]
        assert "holm_p_value" not in item


def test_placement_length_robustness_reports_token_count_and_density():
    analysis = build_analysis(rows_with_failure(), permutations=20, bootstraps=20)
    for key in (
        "repetition_length_robustness",
        "trailing_copy_length_robustness",
        "one_copy_after_query_length_robustness",
        "one_copy_after_vs_before_length_robustness",
        "one_copy_after_vs_system_length_robustness",
    ):
        contrasts = analysis[key]
        assert is_object_sequence(contrasts)
        assert {str(object_value(item, name=key)["outcome"]) for item in contrasts} == {
            "preanswer_tfidf_token_count",
            "preanswer_tfidf_density_per_100_tokens",
        }


def test_one_copy_pairwise_diagnostics_keep_copy_count_fixed():
    analysis = build_analysis(rows_with_failure(), permutations=20, bootstraps=20)
    for key, control in (
        ("one_copy_after_vs_before_contrasts", "before"),
        ("one_copy_after_vs_system_contrasts", "system"),
    ):
        contrasts = analysis[key]
        assert is_object_sequence(contrasts)
        for raw_item in contrasts:
            item = object_value(raw_item, name=key)
            assert item["treatment_conditions"] == ["after"]
            assert item["control_conditions"] == [control]


def test_one_copy_after_query_effect_averages_system_and_before_controls():
    rows = []
    values = {"system": 0.0, "before": 1.0, "after": 1.0}
    for condition, value in values.items():
        rows.append(
            AnalysisRow(
                question_id="q1",
                model_id="m1",
                dataset="d1",
                condition_id=condition,
                status="completed",
                judgment={"preanswer_tfidf_recall": value},
            )
        )
    result = contrast(
        rows,
        "preanswer_tfidf_recall",
        ("after",),
        ("system", "before"),
        permutations=20,
        bootstraps=20,
        confidence_level=0.95,
        seed=1,
    )
    assert result["treatment_mean"] == 1.0
    assert result["control_mean"] == 0.5
    assert result["effect"] == 0.5


def test_heterogeneity_estimands_use_declared_condition_sets():
    analysis = build_analysis(rows_with_failure(), permutations=20, bootstraps=20)
    expected = {
        "model_specific_repetition_contrasts": (list(DOUBLE), list(SINGLE)),
        "model_specific_trailing_contrasts": (
            list(TRAILING_TREATMENT),
            list(TRAILING_CONTROL),
        ),
        "model_specific_one_copy_after_contrasts": (
            ["after"],
            ["system", "before"],
        ),
        "dataset_specific_repetition_contrasts": (list(DOUBLE), list(SINGLE)),
        "dataset_specific_trailing_contrasts": (
            list(TRAILING_TREATMENT),
            list(TRAILING_CONTROL),
        ),
        "dataset_specific_one_copy_after_contrasts": (
            ["after"],
            ["system", "before"],
        ),
    }
    for key, (treatment, control) in expected.items():
        contrasts = analysis[key]
        assert is_object_sequence(contrasts) and contrasts
        for raw_item in contrasts:
            item = object_value(raw_item, name=key)
            assert item["treatment_conditions"] == treatment
            assert item["control_conditions"] == control


def test_analysis_metadata_records_every_resampling_seed_family():
    analysis = build_analysis(rows_with_failure(), permutations=20, bootstraps=20)
    parameters = object_value(analysis["parameters"], name="analysis.parameters")
    seeds = object_value(parameters["seeds"], name="analysis.parameters.seeds")
    assert seeds == {
        "primary_base": 13_000,
        "repetition_length_base": 14_000,
        "trailing_base": 16_000,
        "trailing_length_base": 17_000,
        "role_diagnostics_base": 18_000,
        "exploratory_base": 20_000,
        "accuracy": 21_100,
        "sensitivity_base": 23_000,
        "model_repetition_base": 30_000,
        "model_repetition_robustness_base": 40_000,
        "model_repetition_additional_metrics_base": 45_000,
        "dataset_repetition_base": 50_000,
        "one_copy_after_base": 60_000,
        "one_copy_after_length_base": 60_500,
        "one_copy_after_vs_before_base": 61_000,
        "one_copy_after_vs_before_length_base": 61_500,
        "one_copy_after_vs_system_base": 62_000,
        "one_copy_after_vs_system_length_base": 62_500,
        "model_trailing_base": 70_000,
        "model_one_copy_after_base": 80_000,
        "dataset_trailing_base": 90_000,
        "dataset_one_copy_after_base": 100_000,
        "question_complexity_base": 110_000,
        "factorial_base": 120_000,
    }
