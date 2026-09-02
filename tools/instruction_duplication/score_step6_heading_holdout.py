#!/usr/bin/env python3
# Compare frozen Step-6 human ratings with the hidden mechanical key.

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_key(path: Path) -> dict[str, dict[str, object]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[str(row["task_id"])] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    ratings_doc = json.loads(args.ratings.read_text(encoding="utf-8"))
    key = read_key(args.key)
    ratings = {str(r["task_id"]): str(r["rating"]) for r in ratings_doc["ratings"]}
    missing = sorted(set(key) - set(ratings))
    extra = sorted(set(ratings) - set(key))
    if missing or extra:
        raise SystemExit(f"ratings/key mismatch: missing={len(missing)}, extra={len(extra)}")
    confusion = {
        "human_pass_machine_pass": 0,
        "human_pass_machine_fail": 0,
        "human_fail_machine_pass": 0,
        "human_fail_machine_fail": 0,
    }
    comparable = []
    disagreements = []
    cannot_tell = 0
    for task_id, human in ratings.items():
        machine = bool(key[task_id]["machine_pass"])
        if human == "cannot_tell":
            cannot_tell += 1
            continue
        if human not in {"pass", "fail"}:
            raise SystemExit(f"invalid rating for {task_id}: {human!r}")
        hp = human == "pass"
        comparable.append(hp == machine)
        confusion[f"human_{'pass' if hp else 'fail'}_machine_{'pass' if machine else 'fail'}"] += 1
        if hp != machine:
            disagreements.append(
                {
                    "task_id": task_id,
                    "human": human,
                    "machine": "pass" if machine else "fail",
                    **{k: v for k, v in key[task_id].items() if k != "machine_pass"},
                }
            )
    n = len(comparable)
    agreement = sum(comparable) / n if n else float("nan")
    human_pass = confusion["human_pass_machine_pass"] + confusion["human_pass_machine_fail"]
    human_fail = confusion["human_fail_machine_pass"] + confusion["human_fail_machine_fail"]
    sensitivity = confusion["human_pass_machine_pass"] / human_pass if human_pass else float("nan")
    specificity = confusion["human_fail_machine_fail"] / human_fail if human_fail else float("nan")
    lines = [
        "STEP-6 HEADING/NON-TRIVIAL-CONTENT HOLDOUT",
        f"Comparable ratings: {n}; cannot tell: {cannot_tell}",
        f"Exact agreement: {sum(comparable)}/{n} = {agreement:.1%}" if n else "Exact agreement: NA",
        f"Human-pass sensitivity: {sensitivity:.1%}"
        if human_pass
        else "Human-pass sensitivity: NA",
        f"Human-fail specificity: {specificity:.1%}"
        if human_fail
        else "Human-fail specificity: NA",
        "",
        "Confusion matrix:",
        f"  human PASS / machine PASS: {confusion['human_pass_machine_pass']}",
        f"  human PASS / machine FAIL: {confusion['human_pass_machine_fail']}",
        f"  human FAIL / machine PASS: {confusion['human_fail_machine_pass']}",
        f"  human FAIL / machine FAIL: {confusion['human_fail_machine_fail']}",
        "",
        f"Disagreements: {len(disagreements)}",
    ]
    for row in disagreements:
        lines += [
            f"  {row['task_id']}: human={row['human']}, machine={row['machine']}",
            f"    model={row.get('model_id')} dataset={row.get('dataset')} condition={row.get('condition_id')} question={row.get('question_id')}",
            f"    heading hints={row.get('step6_hint_lines', [])}",
        ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    out = args.out or args.ratings.with_name("step6-heading-comparison.txt")
    out.write_text(text, encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
