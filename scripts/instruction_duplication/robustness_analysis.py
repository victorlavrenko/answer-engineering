#!/usr/bin/env python3
"""Post-hoc robustness analyses on the frozen paper judgments.

Paper mapping: implements the checks reported in ``Results`` and ``Limitations and
Reproducibility``: full 2x2x2 factorial reporting, additive residuals, length-adjusted
TF-IDF, split generation status, raw accuracy counts, lexical-eligibility provenance,
and human-audit sign diagnostics. It reads stored generations/judgments only and does
not regenerate or rejudge cells."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from instruction_duplication.records import AnalysisRow
from instruction_duplication.stats import (
    FACTORIAL_TERMS,
    ZERO_AS_ABSENT_PROTOCOL,
    bootstrap_ci,
    factorial_contrast,
    sign_flip_p_value,
)

VERSION = "aaai27-robustness-v2"
SINGLE = ("system", "before", "after")
DOUBLE = ("system_before", "system_after", "before_after")
CONDITIONS = (
    "zero",
    "system",
    "before",
    "after",
    "system_before",
    "system_after",
    "before_after",
    "system_before_after",
)
FLAGS = {
    "zero": (0, 0, 0),
    "system": (1, 0, 0),
    "before": (0, 1, 0),
    "after": (0, 0, 1),
    "system_before": (1, 1, 0),
    "system_after": (1, 0, 1),
    "before_after": (0, 1, 1),
    "system_before_after": (1, 1, 1),
}
DISPLAY_OUTCOMES = (
    "nontrivial_section_count",
    "preprovisional_tfidf_recall",
    "validated_contrastive_discussion_score",
    "validated_role_count",
    "all_roles_validated_complete",
    "accuracy",
)
FACTORIAL_OUTCOMES = (
    "nontrivial_section_count",
    "preprovisional_tfidf_recall",
    "validated_contrastive_discussion_score",
    "validated_role_count",
    "all_roles_validated_complete",
    "accuracy",
)
ADD_TARGETS = {
    "system_before": ("system", "before"),
    "system_after": ("system", "after"),
    "before_after": ("before", "after"),
    "system_before_after": ("system", "before", "after"),
}
MAIN_FAMILY = (
    "nontrivial_section_count",
    "preprovisional_tfidf_recall",
    "validated_contrastive_discussion_score",
    "validated_role_count",
    "premature_commitment_completed",
    "accuracy",
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def outcome_value(row: dict[str, object], outcome: str) -> float | None:
    j = row.get("judgment") or {}
    if outcome == "premature_commitment_completed":
        return numeric(j.get("preprovisional_commitment_observed"))
    value = numeric(j.get(outcome))
    if value is None and row["condition_id"] == "zero" and outcome in ZERO_AS_ABSENT_PROTOCOL:
        if outcome == "preprovisional_tfidf_recall" and not bool(j.get("repair_endpoint_eligible")):
            return None
        return 0.0
    return value


def mean(xs: Iterable[float]) -> float | None:
    vals = list(xs)
    return sum(vals) / len(vals) if vals else None


def paired_contrast(
    rows: list[dict[str, object]],
    outcome: str,
    treatment: tuple[str, ...],
    control: tuple[str, ...],
    *,
    require_completed: bool = False,
    seed: int = 8000,
) -> dict[str, object]:
    units: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if require_completed and row["status"] != "completed":
            continue
        value = outcome_value(row, outcome)
        if value is not None:
            units[(row["question_id"], row["model_id"])][row["condition_id"]] = value
    by_question: dict[str, list[tuple[float, float]]] = defaultdict(list)
    incomplete = 0
    for (qid, _mid), d in units.items():
        if not all(c in d for c in (*treatment, *control)):
            incomplete += 1
            continue
        by_question[qid].append(
            (mean(d[c] for c in treatment) or 0.0, mean(d[c] for c in control) or 0.0)
        )
    tmeans, cmeans, diffs = [], [], []
    for pairs in by_question.values():
        t = mean(x[0] for x in pairs) or 0.0
        c = mean(x[1] for x in pairs) or 0.0
        tmeans.append(t)
        cmeans.append(c)
        diffs.append(t - c)
    lo, hi = bootstrap_ci(diffs, 10000, seed + 1, 0.95)
    return {
        "outcome": outcome,
        "treatment_conditions": list(treatment),
        "control_conditions": list(control),
        "treatment_mean": mean(tmeans),
        "control_mean": mean(cmeans),
        "effect": mean(diffs),
        "ci_low": lo,
        "ci_high": hi,
        "p_value": sign_flip_p_value(diffs, 50000, seed + 2),
        "question_clusters": len(diffs),
        "incomplete_model_question_units": incomplete,
        "population": "completed generations" if require_completed else "ITT / defined outcome",
    }


def holm(results: list[dict[str, object]]) -> None:
    ordered = sorted(enumerate(results), key=lambda x: float(x[1]["p_value"]))
    m = len(ordered)
    running = 0.0
    for rank, (idx, result) in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * float(result["p_value"]))
        running = max(running, adjusted)
        results[idx]["holm_p_value"] = running


def condition_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for cond in CONDITIONS:
        rr = [r for r in rows if r["condition_id"] == cond]
        record: dict[str, object] = {
            "condition": cond,
            "system": FLAGS[cond][0],
            "before": FLAGS[cond][1],
            "after": FLAGS[cond][2],
            "copies": sum(FLAGS[cond]),
            "n_scheduled": len(rr),
            "completed": sum(r["status"] == "completed" for r in rr),
            "truncated": sum(r["status"] == "truncated" for r in rr),
            "failed": sum(r["status"] == "failed" for r in rr),
            "mean_output_tokens": mean(float(r.get("output_tokens") or 0) for r in rr),
        }
        for outcome in DISPLAY_OUTCOMES:
            vals = [v for r in rr if (v := outcome_value(r, outcome)) is not None]
            record[outcome] = mean(vals)
            record[outcome + "_n"] = len(vals)
        out.append(record)
    return out


def as_analysis_rows(rows: list[dict[str, object]]) -> list[AnalysisRow]:
    return [
        AnalysisRow(
            r["question_id"],
            r["model_id"],
            r["condition_id"],
            r["status"],
            r["dataset"],
            r.get("judgment"),
        )
        for r in rows
    ]


def additive_residuals(
    rows: list[dict[str, object]], outcome: str, *, seed: int
) -> list[dict[str, object]]:
    units: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = outcome_value(row, outcome)
        if value is not None:
            units[(row["question_id"], row["model_id"])][row["condition_id"]] = value
    results = []
    for offset, (target, singletons) in enumerate(ADD_TARGETS.items()):
        byq: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for (qid, _mid), d in units.items():
            needed = ("zero", target, *singletons)
            if not all(c in d for c in needed):
                continue
            if len(singletons) == 2:
                pred = d[singletons[0]] + d[singletons[1]] - d["zero"]
            else:
                pred = sum(d[c] for c in singletons) - 2 * d["zero"]
            byq[qid].append((d[target], pred, d[target] - pred))
        obs = []
        pred = []
        dif = []
        for vals in byq.values():
            obs.append(mean(v[0] for v in vals) or 0.0)
            pred.append(mean(v[1] for v in vals) or 0.0)
            dif.append(mean(v[2] for v in vals) or 0.0)
        lo, hi = bootstrap_ci(dif, 10000, seed + 10 * offset + 1, 0.95)
        results.append(
            {
                "outcome": outcome,
                "target_condition": target,
                "additive_basis": ["zero", *singletons],
                "observed_mean": mean(obs),
                "predicted_additive_mean": mean(pred),
                "residual_observed_minus_predicted": mean(dif),
                "ci_low": lo,
                "ci_high": hi,
                "p_value": sign_flip_p_value(dif, 50000, seed + 10 * offset + 2),
                "question_clusters": len(dif),
                "note": "Protocol-dependent zero-copy values are coded 0 (target protocol absent); bounded outcomes therefore exhibit expected saturation/sub-additivity.",
            }
        )
    return results


def fixed_effect_length_adjustment(rows: list[dict[str, object]]) -> dict[str, object]:
    # Only eligible one-/two-copy cells with both lexical recall and pre-answer token count.
    obs = []
    for r in rows:
        if r["condition_id"] not in (*SINGLE, *DOUBLE):
            continue
        j = r.get("judgment") or {}
        y = numeric(j.get("preprovisional_tfidf_recall"))
        L = numeric(j.get("preanswer_tfidf_token_count"))
        if y is None or L is None:
            continue
        obs.append(
            (r["question_id"], r["model_id"], 1.0 if r["condition_id"] in DOUBLE else 0.0, y, L)
        )
    # Demean within question-model = fixed effects without huge dummy matrix.
    grouped: dict[tuple[str, str], list[tuple[str, str, float, float, float]]] = defaultdict(list)
    for row in obs:
        grouped[(row[0], row[1])].append(row)
    yd = []
    X = []
    clusters = []
    for vals in grouped.values():
        mt = mean(v[2] for v in vals) or 0.0
        my = mean(v[3] for v in vals) or 0.0
        ml = mean(v[4] for v in vals) or 0.0
        for q, _m, t, y, L in vals:
            yd.append(y - my)
            X.append([t - mt, L - ml])
            clusters.append(q)
    y = np.asarray(yd, float)
    X = np.asarray(X, float)
    xtxi = np.linalg.inv(X.T @ X)
    beta = xtxi @ (X.T @ y)
    resid = y - X @ beta
    # CR1 question-cluster robust covariance.
    cluster_ids = {q: i for i, q in enumerate(sorted(set(clusters)))}
    score = np.zeros((len(cluster_ids), X.shape[1]))
    for i, q in enumerate(clusters):
        score[cluster_ids[q]] += X[i] * resid[i]
    meat = score.T @ score
    n = len(y)
    k = X.shape[1]
    G = len(cluster_ids)
    dfc = (G / (G - 1)) * ((n - 1) / (n - k))
    cov = dfc * xtxi @ meat @ xtxi
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p = [math.erfc(abs(v) / math.sqrt(2.0)) for v in z]
    ci = np.column_stack((beta - 1.959963984540054 * se, beta + 1.959963984540054 * se))
    # Unadjusted group means for context.
    one = [x[3] for x in obs if x[2] == 0]
    two = [x[3] for x in obs if x[2] == 1]
    L1 = [x[4] for x in obs if x[2] == 0]
    L2 = [x[4] for x in obs if x[2] == 1]
    return {
        "model": "within-question-model OLS: recall ~ two_copy + preanswer_content_tokens; question-cluster robust SE",
        "n_cells": len(obs),
        "question_model_fixed_effects": len(grouped),
        "question_clusters": G,
        "unadjusted_one_copy_recall": mean(one),
        "unadjusted_two_copy_recall": mean(two),
        "unadjusted_one_copy_tokens": mean(L1),
        "unadjusted_two_copy_tokens": mean(L2),
        "two_copy_coefficient": float(beta[0]),
        "two_copy_se": float(se[0]),
        "two_copy_ci_low": float(ci[0, 0]),
        "two_copy_ci_high": float(ci[0, 1]),
        "two_copy_p_value": float(p[0]),
        "tokens_coefficient": float(beta[1]),
        "tokens_se": float(se[1]),
        "tokens_p_value": float(p[1]),
        "interpretation": "The +1.38 pp unadjusted lexical effect attenuates to about +0.36 pp and narrowly misses p=.05 under this linear fixed-effect adjustment; therefore an effect independent of length is not robustly established.",
    }


def matched_pair_length_adjustment(
    rows: list[dict[str, object]], pairs: tuple[tuple[str, str], ...], *, label: str
) -> dict[str, object]:
    """Length-adjusted lexical effect with a fixed effect for each question-model-pair."""
    pair_map = {}
    for i, (control, treatment) in enumerate(pairs):
        pair_map[control] = (i, 0.0)
        pair_map[treatment] = (i, 1.0)
    obs = []
    for r in rows:
        if r["condition_id"] not in pair_map:
            continue
        j = r.get("judgment") or {}
        y = numeric(j.get("preprovisional_tfidf_recall"))
        L = numeric(j.get("preanswer_tfidf_token_count"))
        if y is None or L is None:
            continue
        pair_id, t = pair_map[r["condition_id"]]
        unit = (r["question_id"], r["model_id"], pair_id)
        obs.append((r["question_id"], unit, t, y, L))
    grouped: dict[
        tuple[str, str, int], list[tuple[str, tuple[str, str, int], float, float, float]]
    ] = defaultdict(list)
    for row in obs:
        grouped[row[1]].append(row)
    yd = []
    X = []
    clusters = []
    for vals in grouped.values():
        mt = mean(v[2] for v in vals) or 0.0
        my = mean(v[3] for v in vals) or 0.0
        ml = mean(v[4] for v in vals) or 0.0
        for q, _u, t, y, L in vals:
            yd.append(y - my)
            X.append([t - mt, L - ml])
            clusters.append(q)
    y = np.asarray(yd, float)
    X = np.asarray(X, float)
    xtxi = np.linalg.inv(X.T @ X)
    beta = xtxi @ (X.T @ y)
    resid = y - X @ beta
    cluster_ids = {q: i for i, q in enumerate(sorted(set(clusters)))}
    score = np.zeros((len(cluster_ids), X.shape[1]))
    for i, q in enumerate(clusters):
        score[cluster_ids[q]] += X[i] * resid[i]
    n = len(y)
    k = X.shape[1]
    G = len(cluster_ids)
    cov = (G / (G - 1)) * ((n - 1) / (n - k)) * xtxi @ (score.T @ score) @ xtxi
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p = [math.erfc(abs(v) / math.sqrt(2.0)) for v in z]
    return {
        "label": label,
        "pairs": [list(x) for x in pairs],
        "n_cells": len(obs),
        "pair_fixed_effects": len(grouped),
        "question_clusters": G,
        "control_mean_recall": mean(x[3] for x in obs if x[2] == 0),
        "treatment_mean_recall": mean(x[3] for x in obs if x[2] == 1),
        "control_mean_tokens": mean(x[4] for x in obs if x[2] == 0),
        "treatment_mean_tokens": mean(x[4] for x in obs if x[2] == 1),
        "treatment_coefficient": float(beta[0]),
        "treatment_se": float(se[0]),
        "treatment_ci_low": float(beta[0] - 1.959963984540054 * se[0]),
        "treatment_ci_high": float(beta[0] + 1.959963984540054 * se[0]),
        "treatment_p_value": float(p[0]),
        "tokens_coefficient": float(beta[1]),
        "tokens_p_value": float(p[1]),
    }


def status_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    by_copy = {}
    for copies in range(4):
        rr = [r for r in rows if sum(FLAGS[r["condition_id"]]) == copies]
        by_copy[str(copies)] = {
            "n": len(rr),
            "completed": sum(r["status"] == "completed" for r in rr),
            "truncated": sum(r["status"] == "truncated" for r in rr),
            "failed": sum(r["status"] == "failed" for r in rr),
        }
    trunc = paired_status_contrast(rows, "truncated", seed=16001)
    fail = paired_status_contrast(rows, "failed", seed=16011)
    not_completed = paired_status_contrast(rows, "not_completed", seed=16021)
    return {
        "by_copy_count": by_copy,
        "two_vs_one_truncation": trunc,
        "two_vs_one_hard_failure": fail,
        "two_vs_one_not_completed": not_completed,
    }


def paired_status_contrast(
    rows: list[dict[str, object]], kind: str, *, seed: int
) -> dict[str, object]:
    units: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["condition_id"] not in (*SINGLE, *DOUBLE):
            continue
        if kind == "truncated":
            v = float(r["status"] == "truncated")
        elif kind == "failed":
            v = float(r["status"] == "failed")
        else:
            v = float(r["status"] != "completed")
        units[(r["question_id"], r["model_id"])][r["condition_id"]] = v
    byq: dict[str, list[float]] = defaultdict(list)
    for (q, _m), d in units.items():
        if all(c in d for c in (*SINGLE, *DOUBLE)):
            byq[q].append((mean(d[c] for c in DOUBLE) or 0) - (mean(d[c] for c in SINGLE) or 0))
    dif = [mean(v) or 0 for v in byq.values()]
    lo, hi = bootstrap_ci(dif, 10000, seed + 1, 0.95)
    return {
        "outcome": kind,
        "effect": mean(dif),
        "ci_low": lo,
        "ci_high": hi,
        "p_value": sign_flip_p_value(dif, 50000, seed + 2),
        "question_clusters": len(dif),
    }


def accuracy_counts(rows: list[dict[str, object]]) -> dict[str, object]:
    result = {}
    for label, conds in (("one_copy", SINGLE), ("two_copy", DOUBLE)):
        rr = [r for r in rows if r["condition_id"] in conds]
        num = sum(int((r.get("judgment") or {}).get("accuracy") or 0) for r in rr)
        result[label] = {
            "correct": num,
            "scheduled": len(rr),
            "accuracy": num / len(rr),
            "by_condition": {},
        }
        for cond in conds:
            cc = [r for r in rr if r["condition_id"] == cond]
            n = sum(int((r.get("judgment") or {}).get("accuracy") or 0) for r in cc)
            result[label]["by_condition"][cond] = {
                "correct": n,
                "scheduled": len(cc),
                "accuracy": n / len(cc),
            }
    return result


def eligibility(qc_rows: list[dict[str, object]]) -> dict[str, object]:
    eligible = [r for r in qc_rows if r.get("repair_endpoint_eligible")]
    excluded = [r for r in qc_rows if not r.get("repair_endpoint_eligible")]
    return {
        "questions_total": len(qc_rows),
        "eligible": len(eligible),
        "excluded": len(excluded),
        "exclusion_flag_counts": {
            "|".join(flags): n
            for flags, n in Counter(tuple(r.get("review_flags") or []) for r in excluded).items()
        },
        "policy": "Question-level, response-independent automatic fact inventory. A question is eligible iff the stem contains at least one standalone declarative/scenario fact outside the interrogative tail; choice-dependent/topic-only stems have no repair fact inventory. The frozen eligibility decision is reused for every model and all eight conditions.",
        "treatment_independent": True,
    }


def audit_summary(human_dir: Path) -> dict[str, object]:
    design = json.loads((human_dir / "aaai27-human-validation-design.json").read_text())
    report = json.loads((human_dir / "aaai27-human-validation-report.json").read_text())
    key = read_jsonl(human_dir / "aaai27-human-validation-key.jsonl")
    ratings_doc = json.loads(
        next(human_dir.glob("aaai27-human-validation-ratings-*.json")).read_text()
    )
    ratings = {r["task_id"]: r["rating"] for r in ratings_doc["ratings"]}
    primary = [k for k in key if k["machine_stratum"] == "improvement"]
    discordant = [k for k in primary if ratings[k["task_id"]] != "same"]
    wins = sum(ratings[k["task_id"]] == k["mechanical_preferred_response"] for k in discordant)
    losses = sum(
        ratings[k["task_id"]] in {"A", "B"}
        and ratings[k["task_id"]] != k["mechanical_preferred_response"]
        for k in discordant
    )
    n = wins + losses
    p = (
        2.0 ** (-n)
        if n and losses == 0
        else sum(math.comb(n, i) * 0.5**n for i in range(wins, n + 1))
        if n
        else 1.0
    )
    # One-sided exact lower bound for k=n: alpha^(1/n).
    lower = 0.05 ** (1 / n) if n and wins == n else None
    primary_side = Counter(k["treatment_response"] for k in primary)
    confirm_side = Counter(
        k["treatment_response"]
        for k in primary
        if ratings[k["task_id"]] == k["mechanical_preferred_response"]
    )
    tie = [k for k in key if k["machine_stratum"] == "tie"]
    return {
        "pairing": "Within the same question and model, each one-copy condition is paired with a two-copy condition made by adding exactly one instruction location: system->system_before/system_after; before->system_before/before_after; after->system_after/before_after.",
        "primary_selection": design["selection"],
        "primary_ratings": report["primary"],
        "conditional_sign_test": {
            "wins": wins,
            "losses": losses,
            "ties": report["primary"]["same"],
            "one_sided_exact_p_value": p,
            "one_sided_95_lower_bound_win_probability_among_discordant": lower,
        },
        "orientation_check": {
            "primary_treatment_side_counts": dict(primary_side),
            "confirmed_treatment_side_counts": dict(confirm_side),
            "tie_sentinel_ratings": [ratings[k["task_id"]] for k in tie],
        },
        "bias_note": "All three tie sentinels were rated B, so a side/style preference cannot be excluded. However, the 30 primary treatment orientations were exactly balanced (15 A/15 B), and the 10 confirmations split 5 A/5 B; the primary directional result is therefore not explained by a simple global B-side preference.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--human-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cells = read_jsonl(args.workspace / "results" / "cells-and-judgments.jsonl")
    qc = read_jsonl(args.workspace / "data" / "question-qc.jsonl")
    analysis_rows = as_analysis_rows(cells)

    main_results = []
    seed_by_outcome = {
        "nontrivial_section_count": 13_100,
        "preprovisional_tfidf_recall": 13_200,
        "validated_contrastive_discussion_score": 13_300,
        "validated_role_count": 13_400,
        "premature_commitment_completed": 23_600,
        "accuracy": 21_100,
    }
    for outcome in MAIN_FAMILY:
        if outcome == "premature_commitment_completed":
            r = paired_contrast(
                cells,
                outcome,
                DOUBLE,
                SINGLE,
                require_completed=True,
                seed=seed_by_outcome[outcome],
            )
        else:
            r = paired_contrast(cells, outcome, DOUBLE, SINGLE, seed=seed_by_outcome[outcome])
        main_results.append(r)
    holm(main_results)

    factorial = []
    for oi, outcome in enumerate(FACTORIAL_OUTCOMES):
        for ti, term in enumerate(FACTORIAL_TERMS):
            factorial.append(
                factorial_contrast(
                    analysis_rows,
                    outcome,
                    term,
                    permutations=50000,
                    bootstraps=10000,
                    confidence_level=0.95,
                    seed=30000 + oi * 100 + ti * 10,
                )
            )

    additive = []
    for i, outcome in enumerate(FACTORIAL_OUTCOMES):
        additive.extend(additive_residuals(cells, outcome, seed=40000 + i * 100))

    result = {
        "version": VERSION,
        "source_judge_version": (cells[0].get("judgment") or {}).get("judge_version"),
        "no_regeneration": True,
        "no_rejudging": True,
        "condition_table": condition_table(cells),
        "main_two_vs_one_exploratory_family": main_results,
        "one_copy_after_vs_conventional": [
            paired_contrast(cells, outcome, ("after",), ("system", "before"), seed=25000 + i * 20)
            for i, outcome in enumerate(
                (
                    "nontrivial_section_count",
                    "preprovisional_tfidf_recall",
                    "validated_contrastive_discussion_score",
                    "validated_role_count",
                    "all_roles_validated_complete",
                    "accuracy",
                )
            )
        ],
        "trailing_copy_raw_contrasts": [
            paired_contrast(
                cells,
                outcome,
                ("system_after", "before_after"),
                ("system", "before"),
                seed=26000 + i * 20,
            )
            for i, outcome in enumerate(
                (
                    "nontrivial_section_count",
                    "preprovisional_tfidf_recall",
                    "validated_contrastive_discussion_score",
                    "validated_role_count",
                    "all_roles_validated_complete",
                    "accuracy",
                )
            )
        ],
        "factorial_terms": factorial,
        "additive_prediction_residuals": additive,
        "length_adjusted_tfidf": fixed_effect_length_adjustment(cells),
        "trailing_copy_length_adjusted_tfidf": matched_pair_length_adjustment(
            cells,
            (("system", "system_after"), ("before", "before_after")),
            label="add after-query duplicate to system-only or before-only prompt",
        ),
        "generation_status": status_summary(cells),
        "accuracy_counts": accuracy_counts(cells),
        "tfidf_eligibility": eligibility(qc),
        "human_audit": audit_summary(args.human_dir),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
