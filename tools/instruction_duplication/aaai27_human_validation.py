#!/usr/bin/env python3
"""AAAI-27-oriented blinded human validation for the paper-facing completion effect.

Paper mapping: implements the sampling, blinding, freezing, and exact-test workflow
described in ``Experiment -> Inference and Audit Design`` and interpreted in
``Evaluation Validity and Downstream Control``.

Subcommands:
    export   create the blinded audit and a pre-rating commitment;
    freeze   validate and cryptographically freeze completed human ratings;
    score    decode only after freezing and run the pre-specified exact test;
    package  build an anonymous ZIP of validation artifacts for supplementary upload.

The primary audit size is calculated from the frozen exact-binomial design rather than
chosen after viewing ratings."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import zipfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from instruction_duplication import audit, judge
from instruction_duplication.io_utils import read_jsonl

VERSION = "aaai27-effect-validation-v1"
DEFAULT_SEED = "instruction-duplication-aaai27-effect-validation-2026-08-19-v1"
ALL8_FIELD = "all_roles_validated_complete"
STRATA = ("improvement", "degradation", "tie")


# ---------- exact binomial design / inference ----------


def binomial_upper_tail(n: int, k: int, p: float) -> float:
    """P[X >= k] for X~Binomial(n,p), computed without SciPy."""
    return sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k, n + 1))


def exact_design(
    *,
    p0: float,
    p1: float,
    alpha: float,
    power: float,
    max_n: int = 10000,
) -> dict[str, float | int]:
    if not (0.0 < p0 < p1 < 1.0):
        raise ValueError("require 0 < p0 < p1 < 1")
    if not (0.0 < alpha < 1.0 and 0.0 < power < 1.0):
        raise ValueError("alpha and power must lie in (0,1)")
    for n in range(1, max_n + 1):
        critical = next(
            (k for k in range(n + 1) if binomial_upper_tail(n, k, p0) <= alpha),
            None,
        )
        if critical is None:
            continue
        achieved_power = binomial_upper_tail(n, critical, p1)
        if achieved_power >= power:
            return {
                "n": n,
                "critical_confirmations": critical,
                "null_tail_at_critical": binomial_upper_tail(n, critical, p0),
                "achieved_power_at_p1": achieved_power,
                "p0": p0,
                "p1": p1,
                "alpha": alpha,
                "target_power": power,
            }
    raise RuntimeError("no exact-binomial design found under max_n")


def clopper_pearson_lower_one_sided(
    successes: int,
    n: int,
    alpha: float,
) -> float:
    """One-sided exact (1-alpha) lower bound for a binomial proportion."""
    if successes <= 0:
        return 0.0
    if successes >= n:
        # Solve p^n = alpha.
        return alpha ** (1.0 / n)

    # Lower p such that P_p[X >= successes] = alpha.
    lo, hi = 0.0, successes / n
    for _ in range(100):
        mid = (lo + hi) / 2.0
        tail = binomial_upper_tail(n, successes, mid)
        if tail < alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------- stable selection / exclusions ----------


def stable_hash(seed: str, *parts: object) -> str:
    raw = seed + "\0" + "\0".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl_objects(path: Path) -> list[dict[str, object]]:
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


def numeric_judgment(row: Mapping[str, object], field: str) -> float | None:
    judgment = row.get("judgment")
    if not isinstance(judgment, Mapping):
        return None
    value = judgment.get(field)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def role_differences(candidate: Mapping[str, object]) -> list[dict[str, object]]:
    control = candidate["control"]
    treatment = candidate["treatment"]
    assert isinstance(control, Mapping) and isinstance(treatment, Mapping)
    out: list[dict[str, object]] = []
    for role, field, title, context in audit.ROLE_SPECS:
        c = numeric_judgment(control, field)
        t = numeric_judgment(treatment, field)
        if c is None or t is None:
            continue
        if abs(t - c) > 1e-12:
            out.append(
                {
                    "role": role,
                    "field": field,
                    "title": title,
                    "context": context,
                    "control_score": c,
                    "treatment_score": t,
                }
            )
    return out


def all8_stratum(candidate: Mapping[str, object]) -> str | None:
    control = candidate["control"]
    treatment = candidate["treatment"]
    if not isinstance(control, Mapping) or not isinstance(treatment, Mapping):
        return None
    if not str(control.get("content") or "").strip():
        return None
    if not str(treatment.get("content") or "").strip():
        return None
    c = numeric_judgment(control, ALL8_FIELD)
    t = numeric_judgment(treatment, ALL8_FIELD)
    if c is None or t is None:
        return None
    if t > c:
        return "improvement"
    if t < c:
        return "degradation"
    return "tie"


def load_spoiled_questions(path: Path | None) -> tuple[set[str], list[dict[str, object]]]:
    if path is None:
        return set(), []
    rows = read_jsonl_objects(path)
    questions = {str(row["question_id"]) for row in rows}
    return questions, rows


def choose_hash_sample(
    pool: list[dict[str, object]],
    *,
    n: int,
    seed: str,
    label: str,
) -> list[dict[str, object]]:
    if len(pool) < n:
        raise RuntimeError(f"need {n} {label} cases but only {len(pool)} are eligible")
    ordered = sorted(
        pool,
        key=lambda candidate: stable_hash(
            seed,
            "sample",
            label,
            candidate["pair_id"],
        ),
    )
    return ordered[:n]


def allocate_primary_by_pattern(
    pool: list[dict[str, object]],
    *,
    n: int,
    seed: str,
) -> list[dict[str, object]]:
    """Proportionally sample all8-improvement transition patterns.

    A pattern is the tuple of role names whose validated binary status changes.
    This keeps the 30-item audit representative of both single-role and multi-role
    all-eight completion transitions rather than silently auditing only the easiest
    cases.
    """
    by_pattern: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for candidate in pool:
        pattern = tuple(str(item["role"]) for item in role_differences(candidate))
        if not pattern:
            raise RuntimeError("all8 improvement without an atomic role difference")
        by_pattern.setdefault(pattern, []).append(candidate)

    counts = {pattern: len(rows) for pattern, rows in by_pattern.items()}
    total = sum(counts.values())
    raw = {pattern: n * count / total for pattern, count in counts.items()}
    allocation = {pattern: int(value) for pattern, value in raw.items()}
    left = n - sum(allocation.values())

    order = sorted(
        counts,
        key=lambda pattern: (
            -(raw[pattern] - allocation[pattern]),
            -counts[pattern],
            stable_hash(seed, "pattern", *pattern),
        ),
    )
    for pattern in order[:left]:
        allocation[pattern] += 1

    # If a very small pattern got over-allocated due to a future design change,
    # move excess to the largest pools with spare capacity.
    excess = 0
    for pattern in allocation:
        if allocation[pattern] > counts[pattern]:
            excess += allocation[pattern] - counts[pattern]
            allocation[pattern] = counts[pattern]
    while excess:
        spare = [pattern for pattern in counts if allocation[pattern] < counts[pattern]]
        if not spare:
            raise RuntimeError("not enough improvement cases for primary audit")
        spare.sort(
            key=lambda pattern: (
                -(counts[pattern] - allocation[pattern]),
                stable_hash(seed, "spare", *pattern),
            )
        )
        allocation[spare[0]] += 1
        excess -= 1

    selected: list[dict[str, object]] = []
    for pattern in sorted(by_pattern, key=lambda p: stable_hash(seed, "pattern-order", *p)):
        need = allocation[pattern]
        if need:
            selected.extend(
                choose_hash_sample(
                    by_pattern[pattern],
                    n=need,
                    seed=seed,
                    label="primary:" + "+".join(pattern),
                )
            )
    if len(selected) != n:
        raise RuntimeError(f"primary allocation produced {len(selected)} cases, expected {n}")
    return selected


def choose_tie_role(candidate: Mapping[str, object], seed: str) -> dict[str, object]:
    """Pick one mechanically tied role for a tie sentinel."""
    control = candidate["control"]
    treatment = candidate["treatment"]
    assert isinstance(control, Mapping) and isinstance(treatment, Mapping)
    tied: list[dict[str, object]] = []
    for role, field, title, context in audit.ROLE_SPECS:
        c = numeric_judgment(control, field)
        t = numeric_judgment(treatment, field)
        if c is None or t is None or abs(c - t) > 1e-12:
            continue
        tied.append(
            {
                "role": role,
                "field": field,
                "title": title,
                "context": context,
                "control_score": c,
                "treatment_score": t,
            }
        )
    if not tied:
        raise RuntimeError("tie sentinel has no tied role")
    tied.sort(
        key=lambda item: stable_hash(
            seed,
            "tie-role",
            candidate["pair_id"],
            item["role"],
        )
    )
    return tied[0]


# ---------- human-facing source-term highlighting ----------

HIGHLIGHTED_CASE_GROUNDED_ROLES = {
    "facts",
    "implications",
    "decisive_fact",
    "rereasoning",
}


def _role_section_text(row: Mapping[str, object], role: str) -> str:
    """Return the single section used for symmetric source-term highlighting."""
    sections = audit._sections(row)
    if role == "facts":
        raw = sections.get("facts", "")
        # Mirror the frozen Facts judge: an explicit multi-option answer-choice
        # list does not supply Facts evidence.
        cleaned, _choice_list_leakage = judge._facts_without_choice_list(
            raw,
            choice_map(row),
        )
        return cleaned
    if role == "implications":
        return sections.get("implications", "")
    if role == "decisive_fact":
        return sections.get("decisive_fact", "")
    if role == "rereasoning":
        return sections.get("rereasoning", "")
    return ""


def _source_coverage_counts(stem: str, text: str) -> Counter[str]:
    """Return conservative stem-term counts recovered by one displayed section.

    Only terms from the case stem can receive credit. Answer choices are never
    reference text. Generic filler that does not recover a stem term therefore
    cannot create a highlight.
    """
    return audit._preanswer_candidate_counts(stem, text, "")


def source_coverage_html(
    *,
    stem: str,
    first: Mapping[str, object],
    second: Mapping[str, object],
    role: str,
    reference: object,
) -> tuple[str | None, str | None]:
    """Render the old human-audit PubMed-IDF highlighting symmetrically.

    Yellow marks stem terms recovered only on that side. Darker means higher
    PubMed IDF. A faint underline marks terms recovered on both sides.
    """
    if role not in HIGHLIGHTED_CASE_GROUNDED_ROLES:
        return None, None

    first_counts = _source_coverage_counts(stem, _role_section_text(first, role))
    second_counts = _source_coverage_counts(stem, _role_section_text(second, role))
    only_first = audit._asymmetric_coverage_counts(first_counts, second_counts)
    only_second = audit._asymmetric_coverage_counts(second_counts, first_counts)
    common = audit._common_coverage_counts(first_counts, second_counts)
    return (
        audit._highlight_stem_coverage(stem, only_first, common, reference),
        audit._highlight_stem_coverage(stem, only_second, common, reference),
    )


# ---------- blinded task construction ----------


def choice_map(row: Mapping[str, object]) -> dict[str, str]:
    value = row.get("choices")
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(text) for key, text in value.items()}


def build_task(
    candidate: Mapping[str, object],
    *,
    task_id: str,
    stratum: str,
    treatment_side: str,
    seed: str,
    lexical_reference: object,
) -> tuple[dict[str, object], dict[str, object]]:
    control = candidate["control"]
    treatment = candidate["treatment"]
    assert isinstance(control, Mapping) and isinstance(treatment, Mapping)
    if treatment_side == "A":
        first, second = treatment, control
    else:
        first, second = control, treatment

    changed = role_differences(candidate)
    if stratum == "tie":
        criteria = [choose_tie_role(candidate, seed)]
    else:
        criteria = changed
        if not criteria:
            raise RuntimeError("non-tie all8 transition without changed role")

    criterion_rows: list[dict[str, object]] = []
    for item in criteria:
        role = str(item["role"])
        context = item["context"]
        excerpt_a = audit._excerpt(first, context)
        excerpt_b = audit._excerpt(second, context)
        coverage_a_html, coverage_b_html = source_coverage_html(
            stem=str(first["stem"]),
            first=first,
            second=second,
            role=role,
            reference=lexical_reference,
        )
        criterion_rows.append(
            {
                "role": role,
                "title": str(item["title"]),
                "rule": audit.ROLE_AUDIT_GUIDANCE[role]["rule"],
                "good": audit.ROLE_AUDIT_GUIDANCE[role]["good"],
                "bad": audit.ROLE_AUDIT_GUIDANCE[role]["bad"],
                "response_a": excerpt_a,
                "response_b": excerpt_b,
                "source_coverage_a_html": coverage_a_html,
                "source_coverage_b_html": coverage_b_html,
                "source_coverage_note": (
                    "Visual aid only. Highlighting is computed from the case stem, not "
                    "from answer choices. Yellow terms are recovered only on this side; "
                    "darker means rarer by the frozen PubMed-IDF reference. Faintly "
                    "underlined terms are recovered on both sides. Generic filler gets "
                    "no highlight. For Facts, any explicit multi-option answer-choice "
                    "list is removed before coverage is computed. Do not mechanically "
                    "count highlights: a valid paraphrase can still count."
                    if coverage_a_html is not None
                    else None
                ),
            }
        )

    blind = {
        "human_audit_version": VERSION,
        "task_id": task_id,
        "question": str(first["stem"]),
        "choices": choice_map(first),
        "criteria": criterion_rows,
    }

    machine_ab = 0
    if stratum == "improvement":
        machine_ab = 1 if treatment_side == "A" else -1
    elif stratum == "degradation":
        machine_ab = -1 if treatment_side == "A" else 1

    key = {
        "task_id": task_id,
        "machine_stratum": stratum,
        "pair_id": str(candidate["pair_id"]),
        "question_id": str(candidate["question_id"]),
        "model_id": str(candidate["model_id"]),
        "dataset": str(candidate["dataset"]),
        "control_condition": str(candidate["control_condition"]),
        "treatment_condition": str(candidate["treatment_condition"]),
        "treatment_response": treatment_side,
        "mechanical_preferred_response": (
            "A" if machine_ab > 0 else "B" if machine_ab < 0 else "same"
        ),
        "changed_roles": [str(item["role"]) for item in criteria],
        "all8_control": numeric_judgment(control, ALL8_FIELD),
        "all8_treatment": numeric_judgment(treatment, ALL8_FIELD),
    }
    return blind, key


def assign_treatment_sides(
    items: list[tuple[str, dict[str, object]]],
    *,
    seed: str,
) -> dict[str, str]:
    """Balance A/B within each hidden stratum as closely as mathematically possible."""
    result: dict[str, str] = {}
    for label in STRATA:
        group = [item for item in items if item[0] == label]
        group.sort(
            key=lambda item: stable_hash(
                seed,
                "orientation",
                label,
                item[1]["pair_id"],
            )
        )
        split = len(group) // 2
        for index, (_label, candidate) in enumerate(group):
            task_id = stable_hash(
                VERSION,
                seed,
                label,
                candidate["pair_id"],
            )[:20]
            result[task_id] = "A" if index < split else "B"
    return result


def render_html(tasks: list[dict[str, object]], audit_id: str) -> str:
    payload = json.dumps(tasks, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blinded human validation</title>
<style>
:root{{--bg:#f4f5f7;--card:#fff;--ink:#17202e;--muted:#66717f;--line:#d7dce3;--accent:#1769df}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,Helvetica,sans-serif}}
main{{max-width:1260px;margin:auto;padding:18px}}.top{{display:flex;justify-content:space-between;align-items:center;gap:18px}}
h1{{font-size:23px;margin:0}}.sub{{font-size:13px;color:var(--muted);margin-top:4px}}input{{padding:9px;border:1px solid var(--line);border-radius:7px}}
button{{border:1px solid #adb5c0;background:#fff;border-radius:9px;padding:14px 13px;font-size:16px;font-weight:700;cursor:pointer}}
button:hover{{border-color:var(--accent)}}.controls{{display:flex;gap:8px;align-items:center}}.progress{{font-weight:700;margin:16px 0 6px}}
.bar{{height:6px;background:#dde2e8;border-radius:4px;overflow:hidden;margin-bottom:14px}}.bar>div{{height:100%;width:0;background:var(--accent)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:15px;margin-bottom:13px}}
.instruction{{font-size:17px;line-height:1.45}}.criterion-title{{font-size:19px;font-weight:700;margin-bottom:6px}}
.rule{{font-size:16px;line-height:1.4}}.examples{{font-size:12.5px;color:var(--muted);margin-top:7px;line-height:1.35}}
.coverage-note{{font-size:12.5px;color:var(--muted);line-height:1.4;margin:10px 0 6px}}
.coverage-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px}}
.coverage-cell{{border:1px solid #e2e6eb;border-radius:8px;padding:10px;background:#fafbfc}}
.coverage-cell h4{{font-size:13px;margin:0 0 6px}}
.stemcopy{{font-size:13.5px;line-height:1.45}}
mark.lex{{background:rgba(255,214,64,var(--mark-alpha));padding:0;border-radius:2px}}
.lex-common{{text-decoration:underline;text-decoration-color:#aeb7c2;text-decoration-thickness:2px;text-underline-offset:2px}}
.responses{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}}.response h3{{font-size:15px;margin:0 0 7px}}
pre{{white-space:pre-wrap;word-break:break-word;font-family:Arial,Helvetica,sans-serif;font-size:14.5px;line-height:1.43;margin:0;max-height:340px;overflow:auto}}
details summary{{cursor:pointer;font-weight:700}}.case{{white-space:pre-wrap;line-height:1.4;margin-top:8px}}.choice{{font-size:13px;margin-top:3px}}
.buttons{{position:sticky;bottom:0;display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:9px;padding:13px 0;background:linear-gradient(transparent,var(--bg) 26%)}}
.same{{background:#f7f8fa}}.hidden{{display:none}}.done{{text-align:center;padding:45px 20px}}
@media(max-width:760px){{.top{{flex-direction:column;align-items:flex-start}}.responses{{grid-template-columns:1fr}}.buttons{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<div class="top"><div><h1>Blinded human validation</h1>
<div class="sub">One matched pair per screen. Treatment, model, automatic judgment, and gold answer are hidden.</div></div>
<div class="controls"><input id="rater" placeholder="Rater code, e.g. R1"><button id="downloadTop">Download ratings</button></div></div>
<div id="progress" class="progress"></div><div class="bar"><div id="bar"></div></div>
<section id="task">
<div class="card instruction"><b>Decision rule.</b> For every criterion shown below, decide whether A or B satisfies the criterion. Choose a side only if that side satisfies <b>all</b> shown criteria while the other side fails at least one. If both satisfy all shown criteria, or neither does, choose SAME. Use CAN'T TELL only if the text is genuinely insufficient.</div>
<div id="criteria"></div>
<details class="card"><summary>Case context — open only if needed</summary><div id="question" class="case"></div><div id="choices"></div></details>
<div class="buttons"><button id="aBtn">LEFT / A [1]</button><button id="sameBtn" class="same">SAME [2]</button><button id="bBtn">RIGHT / B [3]</button><button id="cantBtn">CAN'T TELL [4]</button></div>
</section>
<section id="done" class="card done hidden"><h2>All comparisons are rated.</h2><p>Download the ratings and freeze them before opening the hidden key.</p><button id="downloadDone">Download completed ratings</button></section>
<script>
const TASKS={payload};
const AUDIT_ID={json.dumps(audit_id)};
const VERSION={json.dumps(VERSION)};
const STORE='aaai27-hv-'+AUDIT_ID;
let state={{ratings:{{}},rater:'',index:0}};
try{{const saved=JSON.parse(localStorage.getItem(STORE)||'null');if(saved)state=saved}}catch(e){{}}
const $=id=>document.getElementById(id);
$('rater').value=state.rater||'';
$('rater').oninput=e=>{{state.rater=e.target.value;save();}};
function save(){{localStorage.setItem(STORE,JSON.stringify(state));}}
function nextUnrated(start){{for(let i=start;i<TASKS.length;i++)if(!state.ratings[TASKS[i].task_id])return i;for(let i=0;i<start;i++)if(!state.ratings[TASKS[i].task_id])return i;return TASKS.length;}}
function criterionCard(c){{
 const box=document.createElement('section');box.className='card';
 const title=document.createElement('div');title.className='criterion-title';title.textContent=c.title;
 const rule=document.createElement('div');rule.className='rule';rule.textContent=c.rule;
 const ex=document.createElement('div');ex.className='examples';ex.textContent='Counts: '+c.good+'   ·   Does not count: '+c.bad;
 if(c.source_coverage_a_html!==null && c.source_coverage_a_html!==undefined){{
  const note=document.createElement('div');note.className='coverage-note';note.textContent=c.source_coverage_note;
  const coverage=document.createElement('div');coverage.className='coverage-grid';
  for(const [label,sourceHtml] of [['A',c.source_coverage_a_html],['B',c.source_coverage_b_html]]){{
   const cell=document.createElement('div');cell.className='coverage-cell';
   const h=document.createElement('h4');h.textContent=label==='A'?'Case terms preserved by LEFT / A':'Case terms preserved by RIGHT / B';
   const body=document.createElement('div');body.className='stemcopy';body.innerHTML=sourceHtml;
   cell.append(h,body);coverage.appendChild(cell);
  }}
  box.append(title,rule,ex,note,coverage);
 }}else{{
  box.append(title,rule,ex);
 }}
 const responses=document.createElement('div');responses.className='responses';
 for(const [label,text] of [['A',c.response_a],['B',c.response_b]]){{
  const cell=document.createElement('div');cell.className='response';
  const h=document.createElement('h3');h.textContent=label==='A'?'LEFT — A':'RIGHT — B';
  const pre=document.createElement('pre');pre.textContent=text;
  cell.append(h,pre);responses.appendChild(cell);
 }}
 box.append(responses);return box;
}}
function render(){{
 const done=Object.keys(state.ratings).length;
 $('bar').style.width=`${{100*done/TASKS.length}}%`;
 $('progress').textContent=done>=TASKS.length?`${{TASKS.length}} / ${{TASKS.length}} · complete`:`${{Math.min(done+1,TASKS.length)}} / ${{TASKS.length}} · ${{done}} completed`;
 if(done>=TASKS.length){{$('task').classList.add('hidden');$('done').classList.remove('hidden');return}}
 $('task').classList.remove('hidden');$('done').classList.add('hidden');
 if(state.index>=TASKS.length)state.index=nextUnrated(0);
 const t=TASKS[state.index];const root=$('criteria');root.textContent='';
 t.criteria.forEach(c=>root.appendChild(criterionCard(c)));
 $('question').textContent=t.question;const choices=$('choices');choices.textContent='';
 Object.entries(t.choices||{{}}).forEach(([k,v])=>{{const d=document.createElement('div');d.className='choice';d.textContent=`${{k}}. ${{v}}`;choices.appendChild(d)}})
}}
function rate(value){{const t=TASKS[state.index];if(!t)return;state.ratings[t.task_id]=value;state.index=nextUnrated(state.index+1);save();render()}}
function download(){{
 const rows=TASKS.map(t=>({{task_id:t.task_id,rating:state.ratings[t.task_id]||null}}));
 const out={{human_audit_version:VERSION,audit_id:AUDIT_ID,reviewer_id:state.rater||null,exported_at:new Date().toISOString(),complete:rows.every(r=>r.rating),ratings:rows}};
 const blob=new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='aaai27-human-validation-ratings-'+AUDIT_ID+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)
}}
$('aBtn').onclick=()=>rate('A');$('sameBtn').onclick=()=>rate('same');$('bBtn').onclick=()=>rate('B');$('cantBtn').onclick=()=>rate('cannot_tell');
$('downloadTop').onclick=download;$('downloadDone').onclick=download;
document.addEventListener('keydown',e=>{{if(e.target.tagName==='INPUT')return;if(e.key==='1')rate('A');else if(e.key==='2')rate('same');else if(e.key==='3')rate('B');else if(e.key==='4')rate('cannot_tell')}});
state.index=nextUnrated(0);save();render();
</script></main></body></html>"""


# ---------- export ----------


def export_command(args: argparse.Namespace) -> int:
    design = exact_design(
        p0=args.p0,
        p1=args.p1,
        alpha=args.alpha,
        power=args.power,
    )
    primary_n = int(design["n"])
    sentinel_n = max(1, math.ceil(primary_n * args.sentinel_fraction))

    cells_path = args.workspace / "results" / "cells-and-judgments.jsonl"
    old_key_path = args.workspace / "results" / "blinded-matched-pair-key.jsonl"
    if not cells_path.is_file():
        raise SystemExit(f"missing {cells_path}")
    if not old_key_path.is_file():
        raise SystemExit(
            f"missing {old_key_path}; final audit conservatively excludes old-audit questions"
        )

    rows = read_jsonl(cells_path)
    candidates = audit._base_candidates(rows)
    # Reuse the exact frozen PubMed-IDF lexical reference from the workspace.
    lexical_reference = audit._load_lexical_reference(
        args.workspace / "results" / "aaai27-human-validation-audit.jsonl"
    )
    old_audit_questions = {str(row["question_id"]) for row in read_jsonl_objects(old_key_path)}
    spoiled_questions, spoiled_rows = load_spoiled_questions(args.spoiled)
    excluded_questions = old_audit_questions | spoiled_questions

    eligible: list[dict[str, object]] = []
    for candidate in candidates:
        if str(candidate["question_id"]) in excluded_questions:
            continue
        if all8_stratum(candidate) is None:
            continue
        eligible.append(candidate)

    improvements = [c for c in eligible if all8_stratum(c) == "improvement"]
    degradations = [c for c in eligible if all8_stratum(c) == "degradation"]

    # Clean tie sentinels: both sides mechanically pass the all-eight endpoint, so
    # every atomic role is also a pass on both sides.  Human should therefore choose
    # SAME on the randomly selected tied role if the judge is behaving sensibly.
    ties = []
    for candidate in eligible:
        if all8_stratum(candidate) != "tie":
            continue
        control = candidate["control"]
        treatment = candidate["treatment"]
        assert isinstance(control, Mapping) and isinstance(treatment, Mapping)
        if (
            numeric_judgment(control, ALL8_FIELD) == 1.0
            and numeric_judgment(treatment, ALL8_FIELD) == 1.0
        ):
            ties.append(candidate)

    primary = allocate_primary_by_pattern(
        improvements,
        n=primary_n,
        seed=args.seed,
    )
    negative = choose_hash_sample(
        degradations,
        n=sentinel_n,
        seed=args.seed,
        label="degradation-sentinel",
    )
    tie_sentinels = choose_hash_sample(
        ties,
        n=sentinel_n,
        seed=args.seed,
        label="tie-sentinel",
    )

    selected_items: list[tuple[str, dict[str, object]]] = (
        [("improvement", c) for c in primary]
        + [("degradation", c) for c in negative]
        + [("tie", c) for c in tie_sentinels]
    )

    # Do not reuse a matched pair.  Collision is unlikely but enforce it.
    pair_ids = [str(candidate["pair_id"]) for _label, candidate in selected_items]
    if len(set(pair_ids)) != len(pair_ids):
        raise RuntimeError("same matched pair selected more than once")

    sides = assign_treatment_sides(selected_items, seed=args.seed)

    blind_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    for label, candidate in selected_items:
        task_id = stable_hash(
            VERSION,
            args.seed,
            label,
            candidate["pair_id"],
        )[:20]
        blind, key = build_task(
            candidate,
            task_id=task_id,
            stratum=label,
            treatment_side=sides[task_id],
            seed=args.seed,
            lexical_reference=lexical_reference,
        )
        blind_rows.append(blind)
        key_rows.append(key)

    # Shuffle display order independently from stratum and orientation.
    order = sorted(
        range(len(blind_rows)),
        key=lambda i: stable_hash(args.seed, "display", blind_rows[i]["task_id"]),
    )
    blind_rows = [blind_rows[i] for i in order]
    key_by_id = {row["task_id"]: row for row in key_rows}
    key_rows = [key_by_id[row["task_id"]] for row in blind_rows]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.key_out.parent.mkdir(parents=True, exist_ok=True)

    audit_path = args.out_dir / "aaai27-human-validation-audit.jsonl"
    schema_path = args.out_dir / "aaai27-human-validation-design.json"
    exclusions_path = args.out_dir / "aaai27-human-validation-exclusions.json"
    html_path = args.out_dir / "aaai27-human-validation.html"
    methods_path = args.out_dir / "aaai27-human-validation-method.tex"

    with audit_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in blind_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with args.key_out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in key_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    audit_id = hashlib.sha256(audit_path.read_bytes()).hexdigest()[:24]

    primary_patterns = Counter(
        tuple(row["changed_roles"]) for row in key_rows if row["machine_stratum"] == "improvement"
    )
    population_patterns = Counter(
        tuple(str(item["role"]) for item in role_differences(candidate))
        for candidate in improvements
    )

    schema = {
        "human_audit_version": VERSION,
        "audit_id": audit_id,
        "primary_hypothesis": {
            "estimand": (
                "probability that a blinded human confirms the mechanically preferred "
                "direction on automatic all-eight treatment improvements"
            ),
            "null": f"p <= {args.p0}",
            "design_alternative": f"p = {args.p1}",
            "test": "exact one-sided binomial upper-tail test",
            "alpha": args.alpha,
            "target_power": args.power,
            "sample_size_calculation": design,
            "cannot_tell_policy": (
                "counts as non-confirmation in the primary test; denominator remains fixed"
            ),
            "success_definition": (
                "human selects the same A/B side as the mechanical direction on a primary "
                "all-eight improvement task"
            ),
        },
        "sentinels": {
            "fraction_of_primary_each": args.sentinel_fraction,
            "degradation_tasks": sentinel_n,
            "tie_tasks": sentinel_n,
            "inferential_status": (
                "descriptive negative/demand-characteristic controls; not separately powered"
            ),
        },
        "human_interface": {
            "source_term_highlighting": True,
            "highlighted_roles": sorted(HIGHLIGHTED_CASE_GROUNDED_ROLES),
            "reference": "workspace frozen PubMed-IDF lexical-reference.json",
            "highlighting_is_symmetric_visual_aid_only": True,
            "answer_choices_are_not_reference_text": True,
            "facts_explicit_choice_list_removed_before_highlighting": True,
            "generic_non_stem_filler_cannot_be_highlighted": True,
            "rare_terms_are_darker": True,
            "unhighlighted_paraphrases_may_still_count": True,
        },
        "blinding": {
            "treatment_identity_hidden": True,
            "mechanical_preference_hidden": True,
            "machine_stratum_hidden": True,
            "model_hidden": True,
            "condition_hidden": True,
            "gold_answer_hidden": True,
            "A_B_orientation_deterministically_randomized": True,
        },
        "selection": {
            "primary_population": (
                "all eligible one-copy->two-copy matched pairs with "
                "all_roles_validated_complete 0->1"
            ),
            "primary_sampling": (
                "stable-hash deterministic random sampling proportionally across atomic "
                "role-change patterns"
            ),
            "sentinel_sampling": "stable-hash deterministic random sampling within stratum",
            "selection_uses_score_magnitude": False,
            "selection_uses_response_quality": False,
            "selection_uses_gold_answer": False,
            "seed": args.seed,
        },
        "population_after_exclusions": {
            "eligible_matched_pairs": len(eligible),
            "all8_improvements": len(improvements),
            "all8_degradations": len(degradations),
            "all8_ties": sum(1 for c in eligible if all8_stratum(c) == "tie"),
            "clean_pass_pass_ties_for_sentinels": len(ties),
            "improvement_change_patterns": {
                "+".join(pattern): count
                for pattern, count in sorted(
                    population_patterns.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            },
        },
        "sample": {
            "total_tasks": len(key_rows),
            "primary_improvements": primary_n,
            "degradation_sentinels": sentinel_n,
            "tie_sentinels": sentinel_n,
            "datasets": dict(Counter(row["dataset"] for row in key_rows)),
            "models": dict(Counter(row["model_id"] for row in key_rows)),
            "treatment_response": dict(Counter(row["treatment_response"] for row in key_rows)),
            "primary_change_patterns": {
                "+".join(pattern): count
                for pattern, count in sorted(
                    primary_patterns.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            },
            "unique_questions": len({row["question_id"] for row in key_rows}),
            "unique_question_model_units": len(
                {(row["question_id"], row["model_id"]) for row in key_rows}
            ),
            "unique_matched_pairs": len({row["pair_id"] for row in key_rows}),
        },
        "exclusions": {
            "old_human_audit_whole_questions": len(old_audit_questions),
            "explicit_spoiled_whole_questions": len(spoiled_questions),
            "total_excluded_whole_questions": len(excluded_questions),
        },
        "scope": (
            "This is measurement validation of machine-detected changes supporting the "
            "paper-facing completion endpoint. It does not replace the paired statistical "
            "test on the full 300-question experiment."
        ),
        "design_choice_note": (
            "AAAI-27 does not specify p0, p1, alpha, power, or a minimum human-validation "
            "sample size. These values are pre-specified study-design choices and are "
            "recorded here for reproducibility."
        ),
    }
    schema_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    exclusions_path.write_text(
        json.dumps(
            {
                "old_human_audit_question_ids": sorted(old_audit_questions),
                "explicit_spoiled_ledger": spoiled_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    html_path.write_text(render_html(blind_rows, audit_id), encoding="utf-8")

    method = rf"""\paragraph{{Blinded human validation.}}
We pre-specified a blinded human audit of machine-detected changes in the
paper-facing eight-role completion endpoint.  The primary estimand was the
probability that a human comparison confirmed the automatic direction among matched
one-copy$\rightarrow$two-copy cases scored as completion improvements.  The sample
size was the minimum required by an exact one-sided binomial design testing
$H_0:p\leq {args.p0:.2f}$ against design alternative $p={args.p1:.2f}$ at
$\alpha={args.alpha:.2f}$ with at least {100 * args.power:.0f}\% power; this yielded
$n={primary_n}$ and a pre-specified rejection threshold of
{int(design["critical_confirmations"])}/{primary_n} confirmations.  Automatic
improvement cases were deterministically sampled by a frozen seed, proportionally
across role-change patterns.  We additionally included {sentinel_n} degradation
and {sentinel_n} tie sentinels as unpowered blinded controls.  Treatment identity,
automatic preference, model, placement condition, and gold answer were hidden;
A/B orientation was randomized independently of content.  ``Cannot tell'' was
pre-specified as non-confirmation in the primary test, keeping the denominator
fixed.  Ratings were cryptographically frozen before the decoding key was opened.
"""
    methods_path.write_text(method, encoding="utf-8")

    commitment = {
        "human_audit_version": VERSION,
        "audit_id": audit_id,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "audit_sha256": sha256_file(audit_path),
        "design_sha256": sha256_file(schema_path),
        "exclusions_sha256": sha256_file(exclusions_path),
        "html_sha256": sha256_file(html_path),
        "hidden_key_sha256": sha256_file(args.key_out),
        "note": (
            "The hidden key hash is committed now. Freeze completed human ratings before "
            "opening or moving the key into the validation directory."
        ),
    }
    (args.out_dir / "pre-rating-commitment.json").write_text(
        json.dumps(commitment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("AAAI-27-oriented blinded human validation")
    print(f"  audit_id: {audit_id}")
    print(
        "  exact primary design: "
        f"n={primary_n}, reject H0 p<={args.p0:.2f} at "
        f">={int(design['critical_confirmations'])}/{primary_n} confirmations"
    )
    print(
        f"  achieved power at p={args.p1:.2f}: {100 * float(design['achieved_power_at_p1']):.1f}%"
    )
    print(
        f"  tasks: {len(key_rows)} = {primary_n} primary improvements + "
        f"{sentinel_n} degradation sentinels + {sentinel_n} tie sentinels"
    )
    print(
        "  eligible all8 population after exclusions: "
        f"{len(improvements)} improvements, {len(degradations)} degradations"
    )
    print(f"  old-audit whole questions excluded: {len(old_audit_questions)}")
    print(f"  explicit spoiled whole questions: {len(spoiled_questions)}")
    print(f"  HTML: {html_path}")
    print(f"  HIDDEN KEY — do not open before freeze: {args.key_out}")
    return 0


# ---------- freeze ----------


def freeze_command(args: argparse.Namespace) -> int:
    ratings = json.loads(args.ratings.read_text(encoding="utf-8"))
    design = json.loads(args.design.read_text(encoding="utf-8"))
    if ratings.get("human_audit_version") != VERSION:
        raise SystemExit(f"expected human_audit_version={VERSION}")
    if ratings.get("audit_id") != design.get("audit_id"):
        raise SystemExit("ratings/design audit_id mismatch")

    expected = int(design["sample"]["total_tasks"])
    rows = ratings.get("ratings")
    if not isinstance(rows, list) or len(rows) != expected:
        raise SystemExit(
            f"expected {expected} ratings, found {len(rows) if isinstance(rows, list) else 'invalid'}"
        )

    valid = {"A", "B", "same", "cannot_tell"}
    seen: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id"))
        if task_id in seen:
            raise SystemExit(f"duplicate task_id {task_id}")
        seen.add(task_id)
        if row.get("rating") not in valid:
            raise SystemExit(f"task {task_id} is not completely rated")

    reviewer = str(ratings.get("reviewer_id") or "").strip()
    if not reviewer:
        raise SystemExit("enter a non-identifying rater code such as R1 before export")

    frozen = {
        "human_audit_version": VERSION,
        "audit_id": ratings["audit_id"],
        "reviewer_id": reviewer,
        "frozen_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "complete": True,
        "ratings": rows,
    }
    payload = (json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    hash_path = args.out.with_suffix(args.out.suffix + ".sha256")
    hash_path.write_text(f"{digest}  {args.out.name}\n", encoding="utf-8")
    print(f"Frozen ratings: {args.out}")
    print(f"SHA-256: {digest}")
    print("Only now may the hidden decoding key be opened.")
    return 0


# ---------- score ----------


def rating_ab(value: str) -> int | None:
    return {"A": 1, "B": -1, "same": 0, "cannot_tell": None}[value]


def machine_ab(preference: str) -> int:
    return {"A": 1, "B": -1, "same": 0}[preference]


def score_command(args: argparse.Namespace) -> int:
    ratings_doc = json.loads(args.ratings.read_text(encoding="utf-8"))
    design = json.loads(args.design.read_text(encoding="utf-8"))
    key_rows = read_jsonl_objects(args.key)

    if ratings_doc.get("human_audit_version") != VERSION:
        raise SystemExit("wrong frozen ratings version")
    if ratings_doc.get("audit_id") != design.get("audit_id"):
        raise SystemExit("ratings/design audit_id mismatch")
    if len(key_rows) != int(design["sample"]["total_tasks"]):
        raise SystemExit("hidden key size does not match design")

    ratings = {str(row["task_id"]): str(row["rating"]) for row in ratings_doc["ratings"]}

    decoded: list[dict[str, object]] = []
    for key in key_rows:
        task_id = str(key["task_id"])
        human_value = ratings.get(task_id)
        if human_value is None:
            raise SystemExit(f"missing frozen rating for {task_id}")
        h = rating_ab(human_value)
        m = machine_ab(str(key["mechanical_preferred_response"]))
        decoded.append(
            {
                **key,
                "human_rating": human_value,
                "human_ab": h,
                "machine_ab": m,
                # cannot_tell is intentionally NOT counted as confirmation
                "confirmed": bool(h is not None and h == m),
            }
        )

    primary = [row for row in decoded if row["machine_stratum"] == "improvement"]
    n = len(primary)
    successes = sum(bool(row["confirmed"]) for row in primary)
    planned = design["primary_hypothesis"]["sample_size_calculation"]
    if n != int(planned["n"]):
        raise RuntimeError(f"primary n={n} does not match planned n={planned['n']}")

    p0 = float(planned["p0"])
    alpha = float(planned["alpha"])
    p_value = binomial_upper_tail(n, successes, p0)
    lower = clopper_pearson_lower_one_sided(successes, n, alpha)
    critical = int(planned["critical_confirmations"])
    reject = successes >= critical

    def sentinel_summary(label: str) -> dict[str, object]:
        rows = [row for row in decoded if row["machine_stratum"] == label]
        return {
            "n": len(rows),
            "machine_direction_confirmed": sum(bool(row["confirmed"]) for row in rows),
            "human_A": sum(row["human_rating"] == "A" for row in rows),
            "human_same": sum(row["human_rating"] == "same" for row in rows),
            "human_B": sum(row["human_rating"] == "B" for row in rows),
            "human_cannot_tell": sum(row["human_rating"] == "cannot_tell" for row in rows),
        }

    result = {
        "human_audit_version": VERSION,
        "audit_id": design["audit_id"],
        "reviewer_id": ratings_doc.get("reviewer_id"),
        "primary": {
            "n": n,
            "confirmations": successes,
            "confirmation_rate": successes / n,
            "p0": p0,
            "one_sided_exact_p_value": p_value,
            "one_sided_exact_lower_confidence_bound": lower,
            "confidence_level": 1.0 - alpha,
            "critical_confirmations": critical,
            "reject_null_at_prespecified_alpha": reject,
            "cannot_tell_counted_as_nonconfirmation": True,
            "cannot_tell": sum(row["human_rating"] == "cannot_tell" for row in primary),
            "same": sum(row["human_rating"] == "same" for row in primary),
            "opposite_direction": sum(
                row["human_ab"] is not None and row["human_ab"] == -row["machine_ab"]
                for row in primary
            ),
        },
        "sentinels": {
            "degradation": sentinel_summary("degradation"),
            "tie": sentinel_summary("tie"),
        },
        "by_primary_change_pattern": {},
        "scope_note": design["scope"],
    }

    patterns = sorted({tuple(row["changed_roles"]) for row in primary})
    for pattern in patterns:
        rows = [row for row in primary if tuple(row["changed_roles"]) == pattern]
        result["by_primary_change_pattern"]["+".join(pattern)] = {
            "n": len(rows),
            "confirmed": sum(bool(row["confirmed"]) for row in rows),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "aaai27-human-validation-report.json"
    txt_path = args.out_dir / "aaai27-human-validation-report.txt"
    tex_path = args.out_dir / "aaai27-human-validation-result.tex"

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "AAAI-27-oriented blinded human validation",
        f"Audit ID: {design['audit_id']}",
        f"Rater: {ratings_doc.get('reviewer_id')}",
        "",
        "Primary exact test:",
        f"  confirmations: {successes}/{n} ({100 * successes / n:.1f}%)",
        f"  H0: p <= {p0:.2f}",
        f"  one-sided exact p-value: {p_value:.6g}",
        f"  one-sided {(1 - alpha) * 100:.0f}% exact lower bound: {100 * lower:.1f}%",
        f"  pre-specified critical value: {critical}/{n}",
        f"  primary validation criterion met: {'YES' if reject else 'NO'}",
        f"  SAME on primary tasks: {result['primary']['same']}",
        f"  opposite-direction primary judgments: {result['primary']['opposite_direction']}",
        f"  CAN'T TELL on primary tasks: {result['primary']['cannot_tell']}",
        "",
        f"Degradation sentinels: {result['sentinels']['degradation']}",
        f"Tie sentinels: {result['sentinels']['tie']}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    criterion = "met" if reject else "was not met"
    tex = rf"""\paragraph{{Human-validation result.}}
The blinded rater confirmed {successes}/{n} ({100 * successes / n:.1f}\%) of the
pre-specified automatic completion improvements.  Under the exact one-sided test
of $H_0:p\leq {p0:.2f}$, $p={p_value:.4g}$; the one-sided
{100 * (1 - alpha):.0f}\% exact lower confidence bound was {100 * lower:.1f}\%.
The pre-specified validation criterion of at least {critical}/{n} confirmations
{criterion}.  Degradation and tie cases were included only as blinded descriptive
sentinels.
"""
    tex_path.write_text(tex, encoding="utf-8")

    print("\n".join(lines))
    print(f"\nwrote {txt_path}")
    print(f"wrote {json_path}")
    print(f"wrote {tex_path}")
    return 0


# ---------- anonymous supplementary ZIP ----------


def package_command(args: argparse.Namespace) -> int:
    required = [
        args.validation_dir / "aaai27-human-validation-audit.jsonl",
        args.validation_dir / "aaai27-human-validation-design.json",
        args.validation_dir / "aaai27-human-validation-exclusions.json",
        args.validation_dir / "pre-rating-commitment.json",
        args.validation_dir / "aaai27-human-validation-method.tex",
        args.validation_dir / "aaai27-human-validation-report.json",
        args.validation_dir / "aaai27-human-validation-report.txt",
        args.validation_dir / "aaai27-human-validation-result.tex",
        args.validation_dir / "aaai27-human-validation-ratings-frozen.json",
        args.validation_dir / "aaai27-human-validation-ratings-frozen.json.sha256",
        args.validation_dir / "aaai27-human-validation-key.jsonl",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "cannot package before freeze/score; missing:\n  "
            + "\n  ".join(str(path) for path in missing)
        )

    readme = """AAAI-27 blinded human-validation artifacts

This ZIP contains the frozen blind audit, pre-specified design and sample-size
calculation, exclusions, pre-rating cryptographic commitment, frozen human ratings,
decoding key, and exact-binomial analysis outputs.

The audit validates measurement of machine-detected changes. It does not replace
the full-experiment paired statistical analysis.

No external repository link is required to inspect these artifacts.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", readme)
        for path in required:
            archive.write(path, arcname=path.name)
        archive.write(Path(__file__), arcname="aaai27_human_validation.py")

    digest = sha256_file(args.out)
    sha_path = args.out.with_suffix(args.out.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {args.out.name}\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"SHA-256: {digest}")
    return 0


# ---------- CLI ----------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate/freeze/score/package the AAAI-27-oriented human audit"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export")
    export.add_argument("--workspace", type=Path, required=True)
    export.add_argument("--spoiled", type=Path)
    export.add_argument("--out-dir", type=Path, required=True)
    export.add_argument("--key-out", type=Path, required=True)
    export.add_argument("--seed", default=DEFAULT_SEED)
    export.add_argument("--p0", type=float, default=0.80)
    export.add_argument("--p1", type=float, default=0.95)
    export.add_argument("--alpha", type=float, default=0.05)
    export.add_argument("--power", type=float, default=0.80)
    export.add_argument(
        "--sentinel-fraction",
        type=float,
        default=0.10,
        help="unpowered degradation and tie sentinels as a fraction of primary n",
    )
    export.set_defaults(func=export_command)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--ratings", type=Path, required=True)
    freeze.add_argument("--design", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    freeze.set_defaults(func=freeze_command)

    score = sub.add_parser("score")
    score.add_argument("--ratings", type=Path, required=True)
    score.add_argument("--key", type=Path, required=True)
    score.add_argument("--design", type=Path, required=True)
    score.add_argument("--out-dir", type=Path, required=True)
    score.set_defaults(func=score_command)

    package = sub.add_parser("package")
    package.add_argument("--validation-dir", type=Path, required=True)
    package.add_argument("--out", type=Path, required=True)
    package.set_defaults(func=package_command)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
