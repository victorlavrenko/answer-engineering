#!/usr/bin/env python3
# Export a fresh blinded human holdout for the simple Step-6 completion metric.
# Run on a NEW analyzed workspace. Keep the generated key unopened until ratings freeze.

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

EXPORT_VERSION = "step6-heading-holdout-v1"

STEP6_HINT_RE = re.compile(
    r"(?im)^[ \\t]*(?:\\#{1,6}[ \\t]*)?(?:[-+*][ \\t]+)?(?:\\*\\*|__)?"
    r"(?:step[ \\t]*)?6[.:)\\-][ \\t]*[^\\r\\n]{0,180}"
    r"(?:change|counterfactual|answer|question|alternative)"
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


def question_ids_from_workspace(workspace: Path) -> set[str]:
    path = workspace / "results" / "cells-and-judgments.jsonl"
    if not path.is_file():
        return set()
    return {str(row["question_id"]) for row in read_jsonl(path) if "question_id" in row}


def stable_rank(seed: str, cell_id: str) -> str:
    return hashlib.sha256(f"{seed}|{cell_id}".encode()).hexdigest()


def hint_lines(content: str) -> list[str]:
    return [line.strip()[:240] for line in content.splitlines() if STEP6_HINT_RE.search(line)][:4]


def choose(
    candidates: list[dict[str, object]],
    count: int,
    *,
    seed: str,
    used_questions: set[str],
    prefer_hints: bool,
) -> list[dict[str, object]]:
    remaining = [row for row in candidates if str(row["question_id"]) not in used_questions]
    selected: list[dict[str, object]] = []
    model_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()

    while remaining and len(selected) < count:
        eligible = [row for row in remaining if str(row["question_id"]) not in used_questions]
        if not eligible:
            break

        def key(row: dict[str, object]) -> tuple[int, int, int, int, str]:
            hinted = bool(hint_lines(str(row.get("content") or "")))
            return (
                model_counts[str(row["model_id"])],
                dataset_counts[str(row["dataset"])],
                condition_counts[str(row["condition_id"])],
                0 if (prefer_hints and hinted) else 1,
                stable_rank(seed, str(row["cell_id"])),
            )

        picked = min(eligible, key=key)
        selected.append(picked)
        used_questions.add(str(picked["question_id"]))
        model_counts[str(picked["model_id"])] += 1
        dataset_counts[str(picked["dataset"])] += 1
        condition_counts[str(picked["condition_id"])] += 1
        remaining = [
            row for row in remaining if str(row["question_id"]) != str(picked["question_id"])
        ]
    return selected


def build_html(audit_id: str, tasks: list[dict[str, str]]) -> str:
    payload = json.dumps(tasks, ensure_ascii=False).replace("</", "<\\/")
    audit_literal = json.dumps(audit_id)
    version_literal = json.dumps(EXPORT_VERSION)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blinded Step-6 heading validation</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f5f5f5;color:#171717}}main{{max-width:1100px;margin:auto;padding:24px}}
.card{{background:white;border:1px solid #ddd;border-radius:10px;padding:20px;margin:18px 0}}pre{{white-space:pre-wrap;word-break:break-word;max-height:620px;overflow:auto;background:#fafafa;border:1px solid #e5e5e5;padding:16px}}
label{{margin-right:18px;font-weight:600}}button{{padding:9px 14px;margin-right:8px}}.small{{color:#555;font-size:.92rem}}.rule{{background:#fffbe8;border-left:4px solid #c09a00;padding:12px 16px}}.sticky{{position:sticky;top:0;background:#f5f5f5;padding:10px 0;z-index:2}}
</style></head><body><main>
<h1>Blinded Step-6 completion validation</h1>
<div class="rule"><b>Judge only the simple observable construct.</b> PASS if the response visibly contains a distinct Step-6 / “what would change the answer” section (a natural synonymous heading is fine) and that section has non-trivial generated content. FAIL if the section is absent, the heading is present but empty, it only repeats the instruction/template, or it contains only tiny filler. Do <b>not</b> judge medical correctness and do <b>not</b> decide whether the hypothetical really would change the answer.</div>
<p class="small">The sample is intentionally enriched for difficult machine pass/fail cases and is not a prevalence estimate. Machine labels, model identity, dataset, condition, and treatment identity are hidden. Audit ID: <code>{html.escape(audit_id)}</code>.</p>
<div class="sticky"><span id="progress"></span> <button id="download">Download frozen ratings JSON</button><button id="clear">Clear saved ratings</button></div><div id="tasks"></div>
<script>
const AUDIT_ID={audit_literal};const VERSION={version_literal};const TASKS={payload};const STORAGE="step6-heading-ratings:"+AUDIT_ID;let ratings=JSON.parse(localStorage.getItem(STORAGE)||"{{}}");
function save(){{localStorage.setItem(STORAGE,JSON.stringify(ratings));document.getElementById("progress").textContent=Object.keys(ratings).length+" / "+TASKS.length+" rated";}}
function render(){{const root=document.getElementById("tasks");TASKS.forEach((task,idx)=>{{const card=document.createElement("section");card.className="card";const h=document.createElement("h2");h.textContent=`Task ${{idx+1}} of ${{TASKS.length}}`;card.appendChild(h);const p=document.createElement("pre");p.textContent=task.response;card.appendChild(p);const controls=document.createElement("div");[["pass","PASS"],["fail","FAIL"],["cannot_tell","Cannot tell"]].forEach(([value,labelText])=>{{const label=document.createElement("label");const input=document.createElement("input");input.type="radio";input.name=task.task_id;input.value=value;input.checked=ratings[task.task_id]===value;input.addEventListener("change",()=>{{ratings[task.task_id]=value;save();}});label.appendChild(input);label.append(" "+labelText);controls.appendChild(label);}});card.appendChild(controls);root.appendChild(card);}});save();}}
document.getElementById("download").addEventListener("click",()=>{{const missing=TASKS.filter(t=>!ratings[t.task_id]);if(missing.length){{alert(missing.length+" tasks are still unrated.");return;}}const data={{audit_id:AUDIT_ID,export_version:VERSION,ratings:TASKS.map(t=>({{task_id:t.task_id,rating:ratings[t.task_id]}}))}};const blob=new Blob([JSON.stringify(data,null,2)+"\\n"],{{type:"application/json"}});const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download="step6-heading-human-ratings.json";a.click();URL.revokeObjectURL(url);}});
document.getElementById("clear").addEventListener("click",()=>{{if(confirm("Delete locally saved ratings for this audit?")){{ratings={{}};localStorage.removeItem(STORAGE);location.reload();}}}});render();
</script></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--exclude-workspace",
        type=Path,
        action="append",
        default=[],
        help="Exclude every question ID appearing in this older workspace; repeatable.",
    )
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", default="step6-heading-holdout-2026-08-18")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.n < 4:
        raise SystemExit("--n must be at least 4")

    source = args.workspace / "results" / "cells-and-judgments.jsonl"
    if not source.is_file():
        raise SystemExit(
            f"missing {source}; run deterministic judge/analyze on this workspace first"
        )

    excluded: set[str] = set()
    for workspace in args.exclude_workspace:
        excluded |= question_ids_from_workspace(workspace)

    rows: list[dict[str, object]] = []
    for row in read_jsonl(source):
        if (
            str(row.get("question_id")) in excluded
            or row.get("status") != "completed"
            or int(row.get("copies", 0)) <= 0
        ):
            continue
        content = str(row.get("content") or "").strip()
        judgment = row.get("judgment")
        if not content or not isinstance(judgment, dict):
            continue
        value = judgment.get("role_answer_changing_change_nontrivial")
        if value not in (0, 0.0, 1, 1.0):
            continue
        copy = dict(row)
        copy["_machine_pass"] = int(float(value))
        rows.append(copy)

    failures = [row for row in rows if row["_machine_pass"] == 0]
    passes = [row for row in rows if row["_machine_pass"] == 1]
    if not failures:
        raise SystemExit(
            "No mechanical Step-6 failures remain after exclusions. Use a larger/new workspace; the holdout intentionally needs failure cases."
        )
    if not passes:
        raise SystemExit("No mechanical Step-6 passes are available.")

    used_questions: set[str] = set()
    desired_fail = min(args.n // 2, len({str(r["question_id"]) for r in failures}))
    selected_fail = choose(
        failures,
        desired_fail,
        seed=args.seed + "|fail",
        used_questions=used_questions,
        prefer_hints=True,
    )
    desired_pass = min(
        args.n - len(selected_fail),
        len({str(r["question_id"]) for r in passes if str(r["question_id"]) not in used_questions}),
    )
    selected_pass = choose(
        passes,
        desired_pass,
        seed=args.seed + "|pass",
        used_questions=used_questions,
        prefer_hints=False,
    )
    selected = selected_fail + selected_pass
    selected.sort(key=lambda row: stable_rank(args.seed + "|order", str(row["cell_id"])))

    task_rows = []
    key_rows = []
    for row in selected:
        task_id = hashlib.sha256(
            f"{EXPORT_VERSION}|{args.seed}|{row['cell_id']}".encode()
        ).hexdigest()[:24]
        task_rows.append({"task_id": task_id, "response": str(row["content"])})
        key_rows.append(
            {
                "task_id": task_id,
                "machine_pass": int(row["_machine_pass"]),
                "cell_id": row["cell_id"],
                "question_id": row["question_id"],
                "model_id": row["model_id"],
                "dataset": row["dataset"],
                "condition_id": row["condition_id"],
                "copies": row["copies"],
                "step6_hint_lines": hint_lines(str(row["content"])),
            }
        )

    audit_basis = {
        "version": EXPORT_VERSION,
        "seed": args.seed,
        "task_ids": [r["task_id"] for r in task_rows],
    }
    audit_id = hashlib.sha256(json.dumps(audit_basis, sort_keys=True).encode()).hexdigest()[:20]
    out_dir = args.out_dir or args.workspace / "results" / "step6-heading-holdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "step6-heading-holdout.html"
    key_path = out_dir / "step6-heading-holdout-key.jsonl"
    schema_path = out_dir / "step6-heading-holdout-schema.json"
    html_path.write_text(build_html(audit_id, task_rows), encoding="utf-8")
    with key_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in key_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    schema = {
        "audit_id": audit_id,
        "export_version": EXPORT_VERSION,
        "requested_n": args.n,
        "exported_n": len(selected),
        "mechanical_failures": len(selected_fail),
        "mechanical_passes": len(selected_pass),
        "unique_questions": len({str(r["question_id"]) for r in selected}),
        "excluded_question_ids": len(excluded),
        "selection_note": "case-enriched construct validation: mechanical failures are oversampled; likely heading near-misses are preferred; at most one response per question",
        "human_rule": "PASS iff a distinct Step-6-equivalent section is visible and has non-trivial generated content; do not judge medical/counterfactual correctness",
    }
    schema_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {html_path}")
    print(f"wrote hidden key {key_path}")
    print(f"audit_id={audit_id}")
    print(
        f"tasks={len(selected)} (mechanical failures={len(selected_fail)}, passes={len(selected_pass)}), unique questions={schema['unique_questions']}"
    )
    print("Rate the HTML first. Do not inspect the key until ratings are frozen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
