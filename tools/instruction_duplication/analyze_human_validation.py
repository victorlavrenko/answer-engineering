#!/usr/bin/env python3
"""Decode frozen human-validation ratings after annotation is complete."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def sign(value: float, eps: float = 1e-12) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def human_ab_direction(task_kind: str, rating: str) -> int | None:
    if rating == "cannot_tell" or not rating:
        return None
    return {"A": 1, "B": -1, "same": 0}.get(rating)


def treatment_direction(ab_direction: int, treatment_response: str) -> int:
    return ab_direction if treatment_response == "A" else -ab_direction


def machine_ab_direction(row: dict[str, object]) -> int:
    a = float(row["mechanical_a"])
    b = float(row["mechanical_b"])
    return sign(a - b)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def clustered_ci(rows: list[dict[str, object]], *, seed: int, draws: int) -> tuple[float, float]:
    by_question: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_question[str(row["question_id"])].append(float(row["correction"]))
    questions = sorted(by_question)
    if not questions:
        return math.nan, math.nan
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(draws):
        sampled = [rng.choice(questions) for _ in questions]
        values = [value for q in sampled for value in by_question[q]]
        boot.append(sum(values) / len(values))
    return percentile(boot, 0.025), percentile(boot, 0.975)


def summarize(
    key_rows: list[dict[str, object]], ratings_doc: dict[str, object], *, draws: int
) -> dict[str, object]:
    ratings = {str(row["task_id"]): row.get("rating") for row in ratings_doc.get("ratings", [])}
    decoded: list[dict[str, object]] = []
    cannot = 0
    missing = 0
    for key in key_rows:
        task_id = str(key["task_id"])
        rating = ratings.get(task_id)
        if rating is None:
            missing += 1
            continue
        h_ab = human_ab_direction(str(key["task_kind"]), str(rating))
        if h_ab is None:
            cannot += 1
            continue
        m_ab = machine_ab_direction(key)
        h_t = treatment_direction(h_ab, str(key["treatment_response"]))
        m_t = treatment_direction(m_ab, str(key["treatment_response"]))
        decoded.append(
            {
                "task_id": task_id,
                "task_kind": key["task_kind"],
                "role": key.get("role"),
                "question_id": key["question_id"],
                "human_ab": h_ab,
                "machine_ab": m_ab,
                "human_treatment_direction": h_t,
                "machine_treatment_direction": m_t,
                "correction": h_t - m_t,
            }
        )

    def one_group(group: list[dict[str, object]], seed: int) -> dict[str, object]:
        if not group:
            return {"evaluable": 0}
        corrections = [float(row["correction"]) for row in group]
        ci_low, ci_high = clustered_ci(group, seed=seed, draws=draws)
        agreement = sum(row["human_ab"] == row["machine_ab"] for row in group) / len(group)
        non_tied = [row for row in group if row["human_ab"] != 0 or row["machine_ab"] != 0]
        non_tied_agreement = (
            sum(row["human_ab"] == row["machine_ab"] for row in non_tied) / len(non_tied)
            if non_tied
            else None
        )
        return {
            "evaluable": len(group),
            "raw_direction_agreement": agreement,
            "non_tied_direction_agreement": non_tied_agreement,
            "mean_signed_correction": sum(corrections) / len(corrections),
            "correction_ci_95_question_clustered": [ci_low, ci_high],
        }

    lexical = [row for row in decoded if row["task_kind"] == "lexical_coverage"]
    roles = [row for row in decoded if row["task_kind"] == "role"]
    role_groups: dict[str, object] = {}
    for role in sorted({str(row["role"]) for row in roles}):
        role_groups[role] = one_group(
            [row for row in roles if row["role"] == role], 1000 + len(role_groups)
        )

    total = len(key_rows)
    return {
        "human_audit_version": ratings_doc.get("human_audit_version"),
        "reviewer_id": ratings_doc.get("reviewer_id"),
        "tasks_in_key": total,
        "rated": total - missing,
        "missing": missing,
        "cannot_tell": cannot,
        "cannot_tell_rate_of_rated": cannot / (total - missing) if total != missing else None,
        "overall": one_group(decoded, 1729),
        "lexical_coverage": one_group(lexical, 1730),
        "protocol_roles": one_group(roles, 1731),
        "by_role": role_groups,
    }


def render_text(summary: dict[str, object]) -> str:
    def pct(value: object) -> str:
        return "n/a" if value is None else f"{100 * float(value):.1f}%"

    def group(name: str, obj: dict[str, object]) -> list[str]:
        if not obj.get("evaluable"):
            return [f"{name}: no evaluable ratings"]
        ci = obj["correction_ci_95_question_clustered"]
        return [
            f"{name}: n={obj['evaluable']}",
            f"  raw human\N{EN DASH}mechanical directional agreement: {pct(obj['raw_direction_agreement'])}",
            f"  non-tied directional agreement: {pct(obj.get('non_tied_direction_agreement'))}",
            f"  mean signed correction (human \N{MINUS SIGN} mechanical treatment direction): {obj['mean_signed_correction']:+.3f}",
            f"  question-clustered 95% bootstrap CI: [{ci[0]:+.3f}, {ci[1]:+.3f}]",
        ]

    lines = [
        "Instruction Duplication — blinded human validation",
        f"Rated: {summary['rated']}/{summary['tasks_in_key']}; missing={summary['missing']}; cannot tell={summary['cannot_tell']}",
        "",
    ]
    lines.extend(group("Overall", summary["overall"]))
    lines.append("")
    lines.extend(group("Lexical coverage", summary["lexical_coverage"]))
    lines.append("")
    lines.extend(group("Protocol roles", summary["protocol_roles"]))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstraps", type=int, default=10000)
    args = parser.parse_args()
    key_rows = read_jsonl(args.key)
    ratings_doc = json.loads(args.ratings.read_text(encoding="utf-8"))
    summary = summarize(key_rows, ratings_doc, draws=args.bootstraps)
    text = render_text(summary)
    print(text, end="")
    if args.output:
        args.output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        args.output.with_suffix(".txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
