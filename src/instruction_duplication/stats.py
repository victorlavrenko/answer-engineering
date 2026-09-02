"""Versioned paired ITT analysis with endpoint-scoped multiplicity control.

Paper mapping: implements the paired one-copy/two-copy contrasts, question-cluster
bootstrap intervals, paired sign-flip tests, Holm correction, and factorial summaries
described in ``Experiment -> Inference and Audit Design`` and ``Results``."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import __version__
from .json_types import JsonObject, json_object
from .manifest import ANALYSIS_VERSION
from .protocol import CONDITION_BY_ID, CONDITIONS
from .records import AnalysisRow

SINGLE = ("system", "before", "after")
DOUBLE = ("system_before", "system_after", "before_after")
TRAILING_TREATMENT = ("system_after", "before_after")
TRAILING_CONTROL = ("system", "before")
ONE_COPY_AFTER = ("after",)
ONE_COPY_CONVENTIONAL = ("system", "before")
ONE_COPY_BEFORE = ("before",)
ONE_COPY_SYSTEM = ("system",)

# Judge v2 keeps the four user-facing constructs separate: requested-section
# presence, non-trivial section content, pre-provisional PubMed-IDF stem exposure,
# and substantive provisional/contrastive discussion. Accuracy, ordering, and the
# older semantic-role diagnostics remain separate; no cross-component "AE score"
# is constructed.
PRIMARY_OUTCOMES = (
    "required_section_count",
    "nontrivial_section_count",
    "preprovisional_tfidf_recall",
    "validated_contrastive_discussion_score",
    "validated_role_count",
    "all_roles_validated_complete",
    "preprovisional_commitment_itt",
)
# Judge v2 makes non-trivial section completion the future-run primary compliance endpoint.
# The 2026-08-12 run was used to develop this judge and must therefore be treated as
# judge-development/exploratory evidence rather than a fresh confirmatory test.
CONFIRMATORY_PRIMARY_OUTCOME = "nontrivial_section_count"
KEY_SECONDARY_OUTCOMES = (
    "required_section_count",
    "preprovisional_tfidf_recall",
    "validated_contrastive_discussion_score",
    "validated_role_count",
    "preprovisional_commitment_itt",
    "accuracy",
)
ROLE_DIAGNOSTIC_OUTCOMES = (
    "required_section_count",
    "nontrivial_section_count",
    "contrastive_discussion_count",
    "contrastive_discussion_score",
    "validated_contrastive_discussion_count",
    "validated_contrastive_discussion_score",
    "provisional_answer_discussed",
    "best_alternative_discussed",
    "decisive_distinction_discussed",
    "answer_change_discussed",
    "reconsideration_discussed",
    "identified_role_count",
    "unique_role_count",
    "substantive_role_count",
    "role_completeness_score",
    "all_roles_substantive",
    "validated_role_count",
    "validated_role_completeness_score",
    "all_roles_validated_complete",
    "roles_in_requested_order",
    "complete_role_scaffold",
    "validated_complete_role_scaffold",
    "role_facts_complete",
    "role_implications_complete",
    "role_provisional_answer_complete",
    "role_best_alternative_complete",
    "role_decisive_distinction_complete",
    "role_answer_changing_change_complete",
    "role_answer_changing_change_nontrivial",
    "role_reconsideration_complete",
    "role_final_answer_complete",
    "protocol_scaffold_complete",
    "preanswer_discussion_complete",
    "contrastive_scaffold_complete",
    "visible_preanswer_discussion_complete",
    "structure_score",
    "section_depth_score",
    "substantive_role_score",
    "contrastive_role_score",
    "trajectory_complete",
)
TRAILING_OUTCOMES = (
    *PRIMARY_OUTCOMES,
    "trajectory_complete",
)
ONE_COPY_PLACEMENT_OUTCOMES = (
    *PRIMARY_OUTCOMES,
    "trajectory_complete",
    "accuracy",
)
PLACEMENT_LENGTH_OUTCOMES = (
    "preanswer_tfidf_token_count",
    "preanswer_tfidf_density_per_100_tokens",
)
HETEROGENEITY_OUTCOMES = PRIMARY_OUTCOMES
MODEL_ROBUSTNESS_OUTCOMES = (
    "preanswer_tfidf_token_count",
    "preanswer_tfidf_density_per_100_tokens",
)
MEANS_ONLY_OUTCOMES = ("preprovisional_commitment_observed",)
EXPLORATORY_OUTCOMES = (
    "facts_tfidf_recall",
    "implications_tfidf_recall",
    "preanswer_high_idf_tfidf_recall",
    "facts_implications_shared_high_idf_recall",
    "preanswer_tfidf_density_per_100_tokens",
    "facts_anchor_recall",
    "implications_anchor_recall",
    "preanswer_anchor_recall",
    "shared_anchor_recall",
    "atomic_fact_coverage",
    "atomic_implication_trace_coverage",
    "hard_qualifier_fact_recall",
    "hard_qualifier_trace_recall",
)
# Complete model-level repetition table with every component kept separate.
MODEL_COMPARISON_OUTCOMES = tuple(
    dict.fromkeys(
        (
            *HETEROGENEITY_OUTCOMES,
            *MODEL_ROBUSTNESS_OUTCOMES,
            *ROLE_DIAGNOSTIC_OUTCOMES,
            *EXPLORATORY_OUTCOMES,
            "accuracy",
        )
    )
)
ADDITIONAL_MODEL_COMPARISON_OUTCOMES = tuple(
    outcome
    for outcome in MODEL_COMPARISON_OUTCOMES
    if outcome not in {*HETEROGENEITY_OUTCOMES, *MODEL_ROBUSTNESS_OUTCOMES}
)
COMPLEXITY_GROUPS = ("one_fact", "two_facts", "three_or_more_facts")
COMPLEXITY_OUTCOMES = (
    "nontrivial_section_count",
    "preprovisional_tfidf_recall",
    "validated_contrastive_discussion_score",
    "atomic_fact_coverage",
)
FACTORIAL_OUTCOMES = (
    "required_section_count",
    "nontrivial_section_count",
    "preprovisional_tfidf_recall",
    "validated_contrastive_discussion_score",
    "accuracy",
)
FACTORIAL_TERMS = (
    ("system",),
    ("before",),
    ("after",),
    ("system", "before"),
    ("system", "after"),
    ("before", "after"),
    ("system", "before", "after"),
)
ZERO_AS_ABSENT_PROTOCOL = {
    "required_section_count",
    "nontrivial_section_count",
    "contrastive_discussion_count",
    "contrastive_discussion_score",
    "validated_contrastive_discussion_count",
    "validated_contrastive_discussion_score",
    "provisional_answer_discussed",
    "best_alternative_discussed",
    "decisive_distinction_discussed",
    "answer_change_discussed",
    "reconsideration_discussed",
    "identified_role_count",
    "unique_role_count",
    "substantive_role_count",
    "role_completeness_score",
    "all_roles_substantive",
    "validated_role_count",
    "validated_role_completeness_score",
    "all_roles_validated_complete",
    "roles_in_requested_order",
    "complete_role_scaffold",
    "validated_complete_role_scaffold",
    "role_facts_complete",
    "role_implications_complete",
    "role_provisional_answer_complete",
    "role_best_alternative_complete",
    "role_decisive_distinction_complete",
    "role_answer_changing_change_complete",
    "role_answer_changing_change_nontrivial",
    "role_reconsideration_complete",
    "role_final_answer_complete",
    "protocol_scaffold_complete",
    "preanswer_discussion_complete",
    "visible_reasoning_scaffold_complete",
    "visible_preanswer_discussion_complete",
    "minimum_repair_scaffold",
    "contrastive_scaffold_complete",
    "correct_with_minimum_repair_scaffold",
    "preanswer_tfidf_recall",
    "preprovisional_tfidf_recall",
    "facts_implications_shared_tfidf_recall",
}


@dataclass(frozen=True, slots=True)
class AnalysisProgress:
    """One deterministic analysis-work snapshot for CLI progress reporting."""

    completed: int
    total: int
    phase: str
    detail: str


type AnalysisProgressSink = Callable[[AnalysisProgress], None]


@dataclass(slots=True)
class _AnalysisProgressTracker:
    """Track completed contrasts without affecting analysis order or RNG seeds."""

    total: int
    sink: AnalysisProgressSink | None
    completed: int = 0

    def begin(self, phase: str, detail: str) -> None:
        self._emit(phase, detail)

    def finish(self, phase: str, detail: str) -> None:
        self.completed += 1
        self._emit(phase, detail)

    def _emit(self, phase: str, detail: str) -> None:
        if self.sink is None:
            return
        self.sink(
            AnalysisProgress(
                completed=self.completed,
                total=self.total,
                phase=phase,
                detail=detail,
            )
        )


def analysis_work_units(rows: Sequence[AnalysisRow]) -> int:
    """Return the number of statistical contrasts computed by ``build_analysis``."""
    model_count = len({row.model_id for row in rows})
    dataset_count = len({row.dataset for row in rows})
    return (
        len(PRIMARY_OUTCOMES)
        + len(PLACEMENT_LENGTH_OUTCOMES)
        + len(TRAILING_OUTCOMES)
        + len(PLACEMENT_LENGTH_OUTCOMES)
        + 3 * len(ONE_COPY_PLACEMENT_OUTCOMES)
        + 3 * len(PLACEMENT_LENGTH_OUTCOMES)
        + len(ROLE_DIAGNOSTIC_OUTCOMES)
        + len(EXPLORATORY_OUTCOMES)
        + 1
        + len(PRIMARY_OUTCOMES)
        + 3 * model_count * len(HETEROGENEITY_OUTCOMES)
        + model_count * len(MODEL_ROBUSTNESS_OUTCOMES)
        + model_count * len(ADDITIONAL_MODEL_COMPARISON_OUTCOMES)
        + 3 * dataset_count * len(HETEROGENEITY_OUTCOMES)
        + len(COMPLEXITY_GROUPS) * len(COMPLEXITY_OUTCOMES)
        + len(FACTORIAL_OUTCOMES) * len(FACTORIAL_TERMS)
    )


def _numeric(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 0:
        raise ValueError("quantile requires values")
    position = (len(sorted_values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def sign_flip_p_value(values: Sequence[float], iterations: int, seed: int) -> float | None:
    """Return a two-sided paired sign-flip p-value, exact for small samples.

    Monte Carlo draws are vectorized in bounded chunks so the same scientific
    procedure remains practical when the final experiment contains hundreds of
    question clusters.
    """
    if iterations < 1:
        raise ValueError("permutations must be positive")
    if not values:
        return None
    observed = abs(sum(values) / len(values))
    if observed < 1e-15:
        return 1.0
    if len(values) <= 16:
        draws = itertools.product((-1.0, 1.0), repeat=len(values))
        extreme = 0
        total = 0
        for signs in draws:
            total += 1
            signed_sum = sum(sign * value for sign, value in zip(signs, values, strict=True))
            statistic = abs(signed_sum / len(values))
            if statistic >= observed - 1e-15:
                extreme += 1
        return extreme / total

    vector = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = iterations
    chunk_size = 4096
    while remaining:
        batch = min(chunk_size, remaining)
        sign_draws = rng.integers(0, 2, size=(batch, vector.size), dtype=np.int8)
        sign_draws = sign_draws * 2 - 1
        statistics = np.abs(sign_draws @ vector / vector.size)
        extreme += int(np.count_nonzero(statistics >= observed - 1e-15))
        remaining -= batch
    return (extreme + 1) / (iterations + 1)


def bootstrap_ci(
    values: Sequence[float],
    iterations: int,
    seed: int,
    confidence_level: float = 0.95,
) -> tuple[float | None, float | None]:
    """Return a linear-interpolated percentile bootstrap interval by question."""
    if iterations < 1:
        raise ValueError("bootstraps must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    if not values:
        return None, None

    vector = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    offset = 0
    chunk_size = 4096
    while offset < iterations:
        batch = min(chunk_size, iterations - offset)
        indices = rng.integers(0, vector.size, size=(batch, vector.size))
        draws[offset : offset + batch] = vector[indices].mean(axis=1)
        offset += batch
    draws.sort()
    alpha = (1 - confidence_level) / 2
    return (
        float(_quantile(draws.tolist(), alpha)),
        float(_quantile(draws.tolist(), 1 - alpha)),
    )


def _judgment_value(judgment: Mapping[str, object], key: str) -> object:
    return judgment.get(key)


def _unit_values(
    rows: Iterable[AnalysisRow],
    outcome: str,
    *,
    model_id: str | None = None,
    dataset: str | None = None,
    usable_only: bool = False,
) -> dict[tuple[str, str], dict[str, float]]:
    units: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if model_id is not None and row.model_id != model_id:
            continue
        if dataset is not None and row.dataset != dataset:
            continue
        judgment = row.judgment or {}
        if usable_only and not bool(_judgment_value(judgment, "generation_usable")):
            continue
        value = _numeric(_judgment_value(judgment, outcome))
        if value is None:
            continue
        units[(row.question_id, row.model_id)][row.condition_id] = value
    return units


def _paired_data(
    rows: Iterable[AnalysisRow],
    outcome: str,
    treatment: Sequence[str],
    control: Sequence[str],
    *,
    model_id: str | None = None,
    dataset: str | None = None,
    usable_only: bool = False,
) -> tuple[list[float], list[float], list[float], int]:
    units = _unit_values(
        rows,
        outcome,
        model_id=model_id,
        dataset=dataset,
        usable_only=usable_only,
    )
    by_question: dict[str, list[tuple[float, float]]] = defaultdict(list)
    required = tuple(treatment) + tuple(control)
    incomplete = 0
    for (question_id, _model), values in units.items():
        if not all(condition in values for condition in required):
            incomplete += 1
            continue
        treated = sum(values[condition] for condition in treatment) / len(treatment)
        baseline = sum(values[condition] for condition in control) / len(control)
        by_question[question_id].append((treated, baseline))
    treatment_means: list[float] = []
    control_means: list[float] = []
    differences: list[float] = []
    for pairs in by_question.values():
        treated = sum(value[0] for value in pairs) / len(pairs)
        baseline = sum(value[1] for value in pairs) / len(pairs)
        treatment_means.append(treated)
        control_means.append(baseline)
        differences.append(treated - baseline)
    return treatment_means, control_means, differences, incomplete


def contrast(
    rows: Iterable[AnalysisRow],
    outcome: str,
    treatment: Sequence[str],
    control: Sequence[str],
    *,
    permutations: int,
    bootstraps: int,
    confidence_level: float,
    seed: int,
    model_id: str | None = None,
    dataset: str | None = None,
    usable_only: bool = False,
) -> JsonObject:
    """Compute one question-clustered paired contrast with visible denominators."""
    treated, baseline, differences, incomplete = _paired_data(
        rows,
        outcome,
        treatment,
        control,
        model_id=model_id,
        dataset=dataset,
        usable_only=usable_only,
    )
    low, high = bootstrap_ci(differences, bootstraps, seed + 1, confidence_level)
    result: JsonObject = {
        "outcome": outcome,
        "estimand": "question-clustered mean paired difference",
        "population": "per-protocol sensitivity" if usable_only else "intention-to-treat",
        "treatment_conditions": list(treatment),
        "control_conditions": list(control),
        "treatment_mean": _mean(treated),
        "control_mean": _mean(baseline),
        "effect": _mean(differences),
        "ci_low": low,
        "ci_high": high,
        "confidence_level": confidence_level,
        "p_value": sign_flip_p_value(differences, permutations, seed + 2),
        "question_clusters": len(differences),
        "incomplete_model_question_units": incomplete,
    }
    if model_id is not None:
        result["model_id"] = model_id
    if dataset is not None:
        result["dataset"] = dataset
    return result


def _factorial_value(row: AnalysisRow, outcome: str) -> float | None:
    judgment = row.judgment or {}
    value = _numeric(_judgment_value(judgment, outcome))
    if value is None and row.condition_id == "zero" and outcome in ZERO_AS_ABSENT_PROTOCOL:
        return 0.0
    return value


def factorial_contrast(
    rows: Iterable[AnalysisRow],
    outcome: str,
    term: Sequence[str],
    *,
    permutations: int,
    bootstraps: int,
    confidence_level: float,
    seed: int,
) -> JsonObject:
    """Estimate one balanced 2^3 factorial effect, clustered equally by question."""
    valid_factors = {"system", "before", "after"}
    if not term or set(term) - valid_factors:
        raise ValueError(f"invalid factorial term: {term!r}")
    units: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = _factorial_value(row, outcome)
        if value is not None:
            units[(row.question_id, row.model_id)][row.condition_id] = value
    required = tuple(condition.id for condition in CONDITIONS)
    incomplete = 0
    by_question: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (question_id, _model_id), values in units.items():
        if not all(condition in values for condition in required):
            incomplete += 1
            continue
        positive: list[float] = []
        negative: list[float] = []
        for condition_id, value in values.items():
            condition = CONDITION_BY_ID[condition_id]
            flags = {
                "system": condition.system,
                "before": condition.before,
                "after": condition.after,
            }
            sign = math.prod(1 if flags[factor] else -1 for factor in term)
            (positive if sign > 0 else negative).append(value)
        by_question[question_id].append((_mean(positive) or 0.0, _mean(negative) or 0.0))
    positive_means: list[float] = []
    negative_means: list[float] = []
    differences: list[float] = []
    for pairs in by_question.values():
        positive_mean = sum(item[0] for item in pairs) / len(pairs)
        negative_mean = sum(item[1] for item in pairs) / len(pairs)
        positive_means.append(positive_mean)
        negative_means.append(negative_mean)
        differences.append(positive_mean - negative_mean)
    low, high = bootstrap_ci(differences, bootstraps, seed + 1, confidence_level)
    return {
        "outcome": outcome,
        "factorial_term": " x ".join(term),
        "estimand": "balanced factorial high-minus-low effect",
        "zero_condition_policy": (
            "absent requested protocol scores zero"
            if outcome in ZERO_AS_ABSENT_PROTOCOL
            else "observed outcome"
        ),
        "positive_mean": _mean(positive_means),
        "negative_mean": _mean(negative_means),
        "effect": _mean(differences),
        "ci_low": low,
        "ci_high": high,
        "confidence_level": confidence_level,
        "p_value": sign_flip_p_value(differences, permutations, seed + 2),
        "question_clusters": len(differences),
        "incomplete_model_question_units": incomplete,
    }


def _complexity_rows(rows: Sequence[AnalysisRow], complexity: str) -> list[AnalysisRow]:
    return [row for row in rows if str((row.judgment or {}).get("fact_complexity")) == complexity]


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Apply Holm step-down correction to one explicitly declared family."""
    adjusted: list[float | None] = [None] * len(p_values)
    finite = sorted(
        ((value, index) for index, value in enumerate(p_values) if value is not None),
        key=lambda item: item[0],
    )
    running = 0.0
    count = len(finite)
    for rank, (value, index) in enumerate(finite):
        running = max(running, (count - rank) * value)
        adjusted[index] = min(1.0, running)
    return adjusted


def _adjust_family(results: Sequence[JsonObject], field: str = "holm_p_value") -> None:
    adjusted = holm_adjust([_numeric(item.get("p_value")) for item in results])
    for item, value in zip(results, adjusted, strict=True):
        item[field] = value


def _condition_means(rows: Sequence[AnalysisRow]) -> JsonObject:
    outcomes = (
        PRIMARY_OUTCOMES
        + ROLE_DIAGNOSTIC_OUTCOMES
        + EXPLORATORY_OUTCOMES
        + MEANS_ONLY_OUTCOMES
        + (
            "trajectory_complete",
            "preanswer_tfidf_token_count",
            "facts_choice_list_leakage",
            "accuracy",
            "accuracy_parseable",
            "generation_usable",
        )
    )
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        judgment = row.judgment or {}
        for outcome in outcomes:
            value = _numeric(_judgment_value(judgment, outcome))
            if value is not None:
                grouped[row.condition_id][outcome].append(value)
    return {
        condition: {
            outcome: {"mean": _mean(values), "n": len(values)}
            for outcome, values in metrics.items()
        }
        for condition, metrics in grouped.items()
    }


def _outcome_contrasts(
    rows: Sequence[AnalysisRow],
    outcomes: Sequence[str],
    treatment: Sequence[str],
    control: Sequence[str],
    *,
    permutations: int,
    bootstraps: int,
    confidence_level: float,
    seed_base: int,
    phase: str,
    progress: _AnalysisProgressTracker,
    usable_only: bool = False,
    model_id: str | None = None,
    dataset: str | None = None,
) -> list[JsonObject]:
    results: list[JsonObject] = []
    for index, outcome in enumerate(outcomes):
        progress.begin(phase, outcome)
        results.append(
            contrast(
                rows,
                outcome,
                treatment,
                control,
                permutations=permutations,
                bootstraps=bootstraps,
                confidence_level=confidence_level,
                seed=seed_base + index * 100,
                usable_only=usable_only,
                model_id=model_id,
                dataset=dataset,
            )
        )
        progress.finish(phase, outcome)
    return results


def _heterogeneity_contrasts(
    rows: Sequence[AnalysisRow],
    groups: Sequence[str],
    *,
    group_field: Literal["model_id", "dataset"],
    outcomes: Sequence[str],
    treatment: Sequence[str],
    control: Sequence[str],
    permutations: int,
    bootstraps: int,
    confidence_level: float,
    seed_base: int,
    phase: str,
    progress: _AnalysisProgressTracker,
) -> list[JsonObject]:
    results: list[JsonObject] = []
    for group_index, group in enumerate(groups):
        for outcome_index, outcome in enumerate(outcomes):
            detail = f"{group}: {outcome}"
            progress.begin(phase, detail)
            results.append(
                contrast(
                    rows,
                    outcome,
                    treatment,
                    control,
                    permutations=permutations,
                    bootstraps=bootstraps,
                    confidence_level=confidence_level,
                    seed=seed_base + group_index * 1_000 + outcome_index * 100,
                    model_id=group if group_field == "model_id" else None,
                    dataset=group if group_field == "dataset" else None,
                )
            )
            progress.finish(phase, detail)
    return results


def _adjust_within_outcome(results: Sequence[JsonObject]) -> None:
    """Apply Holm separately across groups for each reported outcome."""
    by_outcome: dict[str, list[JsonObject]] = defaultdict(list)
    for result in results:
        by_outcome[str(result["outcome"])].append(result)
    for family in by_outcome.values():
        _adjust_family(family)


def _sample_summary(
    rows: Sequence[AnalysisRow],
    models: Sequence[str],
    datasets: Sequence[str],
) -> JsonObject:
    status_counts: dict[str, int] = defaultdict(int)
    usable = 0
    for row in rows:
        status_counts[row.status] += 1
        judgment = row.judgment or {}
        usable += int(bool(_judgment_value(judgment, "generation_usable")))
    return {
        "models": len(models),
        "datasets": len(datasets),
        "questions": len({row.question_id for row in rows}),
        "cells": len(rows),
        "usable_cells": usable,
        "status_counts": dict(sorted(status_counts.items())),
    }


def _analysis_metadata(
    permutations: int,
    bootstraps: int,
    confidence_level: float,
) -> JsonObject:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "package_version": __version__,
        "status": "versioned deterministic reanalysis",
        "parameters": {
            "permutations": permutations,
            "bootstraps": bootstraps,
            "confidence_level": confidence_level,
            "seeds": {
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
            },
        },
        "estimand": {
            "primary": (
                "mean of three two-copy conditions minus mean of three one-copy "
                "conditions, paired within model-question then equally clustered by question"
            ),
            "failure_policy": (
                "ITT: instructed generation failures score 0 on beneficial role/coverage outcomes "
                "and 1 on adverse early commitment; accuracy failures score 0"
            ),
            "sensitivity": "complete successful generations only, reported separately",
        },
        "multiplicity_families": {
            "confirmatory_primary": (
                "nontrivial_section_count is the frozen primary compliance endpoint for future "
                "runs; the 2026-08-12 run was used to develop judge v2 and is exploratory"
            ),
            "key_secondary": (
                "Holm across required-section presence, PubMed-IDF pre-provisional exposure, "
                "contrastive-discussion completion, substantive semantic-role completion, "
                "adverse pre-provisional commitment, and accuracy"
            ),
            "measurement_layers": (
                "reported transparently; only the declared primary and key-secondary family "
                "support confirmatory wording"
            ),
            "role_diagnostics": (
                "none; role presence, uniqueness, order, and each role's substantive "
                "completion are transparent secondary diagnostics"
            ),
            "trailing": "none; trailing-copy placement contrasts are reported separately",
            "placement_length": (
                "none; token count and TF-IDF mass per 100 tokens are robustness diagnostics"
            ),
            "one_copy_placement": (
                "none; one-copy after-query placement and pairwise placement diagnostics "
                "are reported separately"
            ),
            "per_protocol": "none; sensitivity endpoints mirror the pooled measurement endpoints",
            "model_specific": (
                "Holm across models separately within each endpoint and estimand; "
                "repetition, trailing-copy addition, and one-copy placement are distinct families"
            ),
            "model_robustness": (
                "Holm across models separately within each repetition-robustness endpoint"
            ),
            "model_all_metrics": (
                "Holm across models separately within every pooled two-copy-vs-one-copy "
                "endpoint; complete effects are exported as tidy and matrix CSV files"
            ),
            "dataset_specific": (
                "none; dataset heterogeneity is descriptive and reported separately by estimand"
            ),
            "question_complexity": "none; explicitly exploratory subgroups",
            "factorial": "none; exploratory main effects and interactions",
            "exploratory": "none; exploratory diagnostics are explicitly unadjusted",
            "accuracy": "included in the Holm-adjusted key-secondary family",
        },
    }


def build_analysis(
    rows: Sequence[AnalysisRow],
    *,
    permutations: int = 50_000,
    bootstraps: int = 10_000,
    confidence_level: float = 0.95,
    progress_sink: AnalysisProgressSink | None = None,
) -> JsonObject:
    """Build ITT contrasts, sensitivity analyses, and heterogeneity tables."""
    if permutations < 1 or bootstraps < 1:
        raise ValueError("permutations and bootstraps must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    progress = _AnalysisProgressTracker(
        total=analysis_work_units(rows),
        sink=progress_sink,
    )
    primary = _outcome_contrasts(
        rows,
        PRIMARY_OUTCOMES,
        DOUBLE,
        SINGLE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=13_000,
        phase="primary repetition contrasts",
        progress=progress,
    )
    repetition_length = _outcome_contrasts(
        rows,
        PLACEMENT_LENGTH_OUTCOMES,
        DOUBLE,
        SINGLE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=14_000,
        phase="repetition length robustness",
        progress=progress,
    )
    trailing = _outcome_contrasts(
        rows,
        TRAILING_OUTCOMES,
        TRAILING_TREATMENT,
        TRAILING_CONTROL,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=16_000,
        phase="trailing-copy contrasts",
        progress=progress,
    )
    trailing_length = _outcome_contrasts(
        rows,
        PLACEMENT_LENGTH_OUTCOMES,
        TRAILING_TREATMENT,
        TRAILING_CONTROL,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=17_000,
        phase="trailing-copy length robustness",
        progress=progress,
    )
    one_copy_after = _outcome_contrasts(
        rows,
        ONE_COPY_PLACEMENT_OUTCOMES,
        ONE_COPY_AFTER,
        ONE_COPY_CONVENTIONAL,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=60_000,
        phase="one-copy after-query placement",
        progress=progress,
    )
    one_copy_after_length = _outcome_contrasts(
        rows,
        PLACEMENT_LENGTH_OUTCOMES,
        ONE_COPY_AFTER,
        ONE_COPY_CONVENTIONAL,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=60_500,
        phase="one-copy after-query length robustness",
        progress=progress,
    )
    one_copy_after_vs_before = _outcome_contrasts(
        rows,
        ONE_COPY_PLACEMENT_OUTCOMES,
        ONE_COPY_AFTER,
        ONE_COPY_BEFORE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=61_000,
        phase="one-copy after vs before",
        progress=progress,
    )
    one_copy_after_vs_before_length = _outcome_contrasts(
        rows,
        PLACEMENT_LENGTH_OUTCOMES,
        ONE_COPY_AFTER,
        ONE_COPY_BEFORE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=61_500,
        phase="one-copy after vs before length robustness",
        progress=progress,
    )
    one_copy_after_vs_system = _outcome_contrasts(
        rows,
        ONE_COPY_PLACEMENT_OUTCOMES,
        ONE_COPY_AFTER,
        ONE_COPY_SYSTEM,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=62_000,
        phase="one-copy after vs system",
        progress=progress,
    )
    one_copy_after_vs_system_length = _outcome_contrasts(
        rows,
        PLACEMENT_LENGTH_OUTCOMES,
        ONE_COPY_AFTER,
        ONE_COPY_SYSTEM,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=62_500,
        phase="one-copy after vs system length robustness",
        progress=progress,
    )
    role_diagnostics = _outcome_contrasts(
        rows,
        ROLE_DIAGNOSTIC_OUTCOMES,
        DOUBLE,
        SINGLE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=18_000,
        phase="role-completeness diagnostics",
        progress=progress,
    )
    exploratory = _outcome_contrasts(
        rows,
        EXPLORATORY_OUTCOMES,
        DOUBLE,
        SINGLE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=20_000,
        phase="exploratory diagnostics",
        progress=progress,
    )
    progress.begin("accuracy", "accuracy")
    accuracy = contrast(
        rows,
        "accuracy",
        DOUBLE,
        SINGLE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed=21_100,
    )
    progress.finish("accuracy", "accuracy")
    confirmatory_primary = next(
        item for item in primary if item["outcome"] == CONFIRMATORY_PRIMARY_OUTCOME
    )
    secondary_by_outcome = {str(item["outcome"]): item for item in primary}
    secondary_by_outcome["accuracy"] = accuracy
    key_secondary = [
        json_object(secondary_by_outcome[outcome], path=f"key secondary {outcome}")
        for outcome in KEY_SECONDARY_OUTCOMES
    ]
    _adjust_family(key_secondary)
    sensitivity = _outcome_contrasts(
        rows,
        PRIMARY_OUTCOMES,
        DOUBLE,
        SINGLE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=23_000,
        phase="per-protocol sensitivity",
        progress=progress,
        usable_only=True,
    )
    complexity_contrasts: dict[str, list[JsonObject]] = {}
    for group_index, complexity in enumerate(COMPLEXITY_GROUPS):
        subgroup = _complexity_rows(rows, complexity)
        complexity_contrasts[complexity] = _outcome_contrasts(
            subgroup,
            COMPLEXITY_OUTCOMES,
            DOUBLE,
            SINGLE,
            permutations=permutations,
            bootstraps=bootstraps,
            confidence_level=confidence_level,
            seed_base=110_000 + group_index * 1_000,
            phase="question-complexity subgroups",
            progress=progress,
        )
    factorial: list[JsonObject] = []
    for outcome_index, outcome in enumerate(FACTORIAL_OUTCOMES):
        for term_index, term in enumerate(FACTORIAL_TERMS):
            detail = f"{outcome}: {' x '.join(term)}"
            progress.begin("factorial decomposition", detail)
            factorial.append(
                factorial_contrast(
                    rows,
                    outcome,
                    term,
                    permutations=permutations,
                    bootstraps=bootstraps,
                    confidence_level=confidence_level,
                    seed=120_000 + outcome_index * 10_000 + term_index * 100,
                )
            )
            progress.finish("factorial decomposition", detail)
    models = sorted({row.model_id for row in rows})
    datasets = sorted({row.dataset for row in rows})
    model_specific_repetition = _heterogeneity_contrasts(
        rows,
        models,
        group_field="model_id",
        outcomes=HETEROGENEITY_OUTCOMES,
        treatment=DOUBLE,
        control=SINGLE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=30_000,
        phase="model-specific repetition heterogeneity",
        progress=progress,
    )
    _adjust_within_outcome(model_specific_repetition)
    model_robustness = _heterogeneity_contrasts(
        rows,
        models,
        group_field="model_id",
        outcomes=MODEL_ROBUSTNESS_OUTCOMES,
        treatment=DOUBLE,
        control=SINGLE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=40_000,
        phase="model-specific repetition robustness",
        progress=progress,
    )
    _adjust_within_outcome(model_robustness)
    model_additional_metrics = _heterogeneity_contrasts(
        rows,
        models,
        group_field="model_id",
        outcomes=ADDITIONAL_MODEL_COMPARISON_OUTCOMES,
        treatment=DOUBLE,
        control=SINGLE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=45_000,
        phase="model-specific additional repetition metrics",
        progress=progress,
    )
    _adjust_within_outcome(model_additional_metrics)
    model_metric_lookup = {
        (str(item["outcome"]), str(item["model_id"])): item
        for item in (
            *model_specific_repetition,
            *model_robustness,
            *model_additional_metrics,
        )
    }
    model_all_metrics = [
        model_metric_lookup[(outcome, model)]
        for outcome in MODEL_COMPARISON_OUTCOMES
        for model in models
    ]
    model_specific_trailing = _heterogeneity_contrasts(
        rows,
        models,
        group_field="model_id",
        outcomes=HETEROGENEITY_OUTCOMES,
        treatment=TRAILING_TREATMENT,
        control=TRAILING_CONTROL,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=70_000,
        phase="model-specific trailing-copy heterogeneity",
        progress=progress,
    )
    _adjust_within_outcome(model_specific_trailing)
    model_specific_one_copy_after = _heterogeneity_contrasts(
        rows,
        models,
        group_field="model_id",
        outcomes=HETEROGENEITY_OUTCOMES,
        treatment=ONE_COPY_AFTER,
        control=ONE_COPY_CONVENTIONAL,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=80_000,
        phase="model-specific one-copy placement heterogeneity",
        progress=progress,
    )
    _adjust_within_outcome(model_specific_one_copy_after)

    dataset_specific_repetition = _heterogeneity_contrasts(
        rows,
        datasets,
        group_field="dataset",
        outcomes=HETEROGENEITY_OUTCOMES,
        treatment=DOUBLE,
        control=SINGLE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=50_000,
        phase="dataset-specific repetition heterogeneity",
        progress=progress,
    )
    dataset_specific_trailing = _heterogeneity_contrasts(
        rows,
        datasets,
        group_field="dataset",
        outcomes=HETEROGENEITY_OUTCOMES,
        treatment=TRAILING_TREATMENT,
        control=TRAILING_CONTROL,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=90_000,
        phase="dataset-specific trailing-copy heterogeneity",
        progress=progress,
    )
    dataset_specific_one_copy_after = _heterogeneity_contrasts(
        rows,
        datasets,
        group_field="dataset",
        outcomes=HETEROGENEITY_OUTCOMES,
        treatment=ONE_COPY_AFTER,
        control=ONE_COPY_CONVENTIONAL,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=100_000,
        phase="dataset-specific one-copy placement heterogeneity",
        progress=progress,
    )
    return json_object(
        {
            **_analysis_metadata(permutations, bootstraps, confidence_level),
            "confirmatory_primary_contrast": confirmatory_primary,
            "key_secondary_contrasts": key_secondary,
            "primary_repetition_contrasts": primary,
            "repetition_length_robustness": repetition_length,
            "accuracy_contrast": accuracy,
            "trailing_copy_contrasts": trailing,
            "trailing_copy_length_robustness": trailing_length,
            "one_copy_after_query_contrasts": one_copy_after,
            "one_copy_after_query_length_robustness": one_copy_after_length,
            "one_copy_after_vs_before_contrasts": one_copy_after_vs_before,
            "one_copy_after_vs_before_length_robustness": one_copy_after_vs_before_length,
            "one_copy_after_vs_system_contrasts": one_copy_after_vs_system,
            "one_copy_after_vs_system_length_robustness": one_copy_after_vs_system_length,
            "role_diagnostic_contrasts": role_diagnostics,
            "exploratory_contrasts": exploratory,
            "per_protocol_sensitivity": sensitivity,
            "question_complexity_subgroup_contrasts": complexity_contrasts,
            "factorial_decomposition": factorial,
            "model_specific_repetition_contrasts": model_specific_repetition,
            "model_specific_repetition_robustness": model_robustness,
            "model_specific_repetition_all_metrics": model_all_metrics,
            "model_specific_trailing_contrasts": model_specific_trailing,
            "model_specific_one_copy_after_contrasts": model_specific_one_copy_after,
            "dataset_specific_repetition_contrasts": dataset_specific_repetition,
            "dataset_specific_trailing_contrasts": dataset_specific_trailing,
            "dataset_specific_one_copy_after_contrasts": dataset_specific_one_copy_after,
            "condition_means": _condition_means(rows),
            "sample": _sample_summary(rows, models, datasets),
        },
        path="analysis",
    )
