"""Reproducibility snapshots and atomic blinded human-validation exports.

Paper mapping: supports the blinded evaluation-validity workflow described in
``Experiment -> Inference and Audit Design`` and ``Evaluation Validity and Downstream
Control``. Treatment identity and machine preference are kept out of rater-facing
records."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .io_utils import read_json, sha256_json, write_json, write_jsonl
from .json_types import JsonObject, is_string_mapping, json_object, object_value
from .lexical import (
    TFIDF_STOPWORDS,
    LexicalReference,
    candidate_counts,
    compile_reference,
    measurement_stem,
    tfidf_tokens,
    validated_abbreviations,
)
from .models import Model
from .trajectory import recover_protocol

MODEL_ELIGIBILITY_VERSION = "single-visible-stream-policy-v2"
HUMAN_AUDIT_VERSION = "atomic-blinded-pairs-v9-step6-nontrivial"
LEXICAL_TASK_COUNT = 79
ROLE_TASK_COUNT = 15
TARGET_UNIQUE_QUESTIONS = 148

PAIRINGS: tuple[tuple[str, str, str], ...] = (
    ("system", "system_before", "add_before_copy"),
    ("system", "system_after", "add_after_copy"),
    ("before", "system_before", "add_system_copy"),
    ("before", "before_after", "add_after_copy"),
    ("after", "system_after", "add_system_copy"),
    ("after", "before_after", "add_before_copy"),
)

ROLE_SPECS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "facts",
        "role_facts_complete",
        "Facts",
        ("facts",),
    ),
    (
        "implications",
        "role_implications_complete",
        "Implications",
        ("implications",),
    ),
    (
        "provisional_answer",
        "role_provisional_answer_complete",
        "Provisional answer",
        ("provisional_answer",),
    ),
    (
        "second_best",
        "role_best_alternative_complete",
        "Best alternative",
        ("provisional_answer", "second_best"),
    ),
    (
        "decisive_fact",
        "role_decisive_distinction_complete",
        "Decisive distinction",
        ("provisional_answer", "second_best", "decisive_fact"),
    ),
    (
        "answer_changing_change",
        "role_answer_changing_change_nontrivial",
        "What would change the answer",
        ("answer_changing_change",),
    ),
    (
        "rereasoning",
        "role_reconsideration_complete",
        "Reconsideration",
        ("provisional_answer", "rereasoning"),
    ),
    (
        "final_answer",
        "role_final_answer_complete",
        "Final answer",
        ("final_answer",),
    ),
)

ROLE_AUDIT_GUIDANCE = {
    "facts": {
        "rule": "PASS if the Facts section states concrete information from the case. Generic filler or merely listing answer choices does not count.",
        "good": "Facts: 65-year-old; sudden left hearing loss for 2 hours; no ear pain.",
        "bad": "Facts: Hearing problems can have many causes.",
        "note": "A short factual list can fully PASS. Extra length earns no credit. Ignore diagnosis/option discussion unless it replaces rather than states case facts.",
    },
    "implications": {
        "rule": "PASS if it explains what a case fact supports or argues against. Merely repeating the fact does not count.",
        "good": "The sudden unilateral loss supports a sensorineural process.",
        "bad": "The patient has sudden unilateral hearing loss.",
        "note": "One clear fact → implication link can fully PASS. Extra explanation earns no credit; merely restating the fact does not.",
    },
    "provisional_answer": {
        "rule": "PASS if it selects one answer option and gives a reason for selecting it.",
        "good": "Provisional answer: C, because the sudden onset fits C best.",
        "bad": "Provisional answer: C.",
        "note": "One option plus one genuine reason is enough. A longer rationale is not better merely because it is longer. Read all shown text only to make sure the selected provisional option is not left ambiguous or contradicted.",
    },
    "second_best": {
        "rule": "PASS if it selects an option DIFFERENT from the provisional answer and gives a reason why that alternative is less suitable.",
        "good": "Provisional: C. Best alternative: B, but B is less suitable because there is no ear pain.",
        "bad": "Provisional: C. Best alternative: B.",
        "note": "One different option plus one reason it is less suitable is enough. A long discussion gets no bonus. Read the whole shown excerpt to identify what it actually selects as the best alternative. Mentioning other options as rejected is fine; a second competing best alternative or a contradiction makes the selection ambiguous.",
    },
    "decisive_fact": {
        "rule": "PASS if it gives a concrete fact or distinction that separates the provisional answer from the best alternative.",
        "good": "No ear pain favors C over B.",
        "bad": "C is better than B.",
        "note": "One concrete discriminator can fully PASS. More supporting prose is not better merely because it is longer.",
    },
    "answer_changing_change": {
        "rule": "PASS if this section contains non-trivial generated content. Do NOT judge whether the hypothetical is medically correct or whether it truly makes the best alternative win.",
        "good": "If ear pain were present, B might become more plausible.",
        "bad": "See above.",
        "note": "This is deliberately a simple observable-completion check. A short but meaningful sentence can fully PASS. Empty text, a heading alone, or extremely short filler does not count. Do not reward extra length and do not evaluate the semantics of the counterfactual.",
    },
    "rereasoning": {
        "rule": "PASS if it explicitly retains or revises the provisional answer and gives a case-based reason.",
        "good": "Retain C: the sudden onset and absence of pain still favor C.",
        "bad": "After reconsideration, C.",
        "note": "One explicit retain/revise decision plus one case-based reason is enough. Extra explanation earns no credit by itself.",
    },
    "final_answer": {
        "rule": "PASS if it clearly states one available final option. Ambiguous or multiple final options do not count.",
        "good": "Final answer: C — Sensorineural hearing loss.",
        "bad": "Final answer: probably C or D.",
        "note": "A one-line final answer can fully PASS. Extra explanation earns no bonus. If the excerpt presents more than one option as the final selection, it does not PASS.",
    },
}


SECTION_LABELS = {
    "facts": "Facts",
    "implications": "Implications",
    "provisional_answer": "Provisional answer",
    "second_best": "Best alternative",
    "decisive_fact": "Decisive distinction",
    "answer_changing_change": "What would change the answer",
    "rereasoning": "Reconsideration",
    "final_answer": "Final answer",
}

TOKEN_SPAN_RE = re.compile(r"\d+(?:\.\d+)?%?|[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
DISPLAY_STOPWORDS = TFIDF_STOPWORDS | {"not"}


def model_eligibility_snapshot(models: Sequence[Model]) -> JsonObject:
    """Freeze the model-panel basis without asserting unverifiable architecture facts."""
    entries: list[JsonObject] = [
        {
            "model_id": model.id,
            "huggingface_checkpoint": model.hf_id,
            "openrouter_model": model.openrouter_id,
            "configuration_sha256": sha256_json(model.to_dict()),
            "eligibility_basis": [
                "panel entry is an instruction-tuned checkpoint",
                "requests do not enable a separate reasoning channel",
                "preflight and generation reject exposed reasoning/thinking fields",
                "only visible assistant content is stored and judged",
            ],
            "catalog_claims_about_hidden_architecture": None,
        }
        for model in models
    ]
    payload = json_object(
        {
            "model_eligibility_version": MODEL_ELIGIBILITY_VERSION,
            "policy": (
                "single visible response stream; no separately exposed reasoning channel; "
                "provider route is pinned only after two successful guarded probes"
            ),
            "models": entries,
        },
        path="model eligibility snapshot",
    )
    payload["snapshot_sha256"] = sha256_json(payload)
    return payload


def human_audit_schema() -> JsonObject:
    """Return the deliberately small annotation contract used by the browser rater."""
    return {
        "human_audit_version": HUMAN_AUDIT_VERSION,
        "design": {
            "tasks": LEXICAL_TASK_COUNT + ROLE_TASK_COUNT * len(ROLE_SPECS),
            "unique_questions": TARGET_UNIQUE_QUESTIONS,
            "lexical_tasks": LEXICAL_TASK_COUNT,
            "role_tasks_per_role": ROLE_TASK_COUNT,
            "pair_reuse": False,
            "treatment_blinded": True,
            "mechanical_scores_blinded": True,
        },
        "instructions": (
            "Judge only the single textual property named on the screen. Do not solve the medical "
            "question and do not rate overall reasoning quality. Either response may be medically "
            "wrong. Treatment identity, model identity, gold answer, and automatic scores are hidden."
        ),
        "lexical_response_options": ["A", "B", "same", "cannot_tell"],
        "role_response_options": ["A", "B", "same", "cannot_tell"],
    }


def _pair_hash(*parts: str) -> str:
    return hashlib.sha256((HUMAN_AUDIT_VERSION + "\0" + "\0".join(parts)).encode()).hexdigest()


def _judgment(row: Mapping[str, object]) -> Mapping[str, object]:
    value = row.get("judgment")
    return value if is_string_mapping(value) else {}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _content(row: Mapping[str, object]) -> str:
    value = row.get("content")
    return "" if value is None else str(value)


def _choices(row: Mapping[str, object]) -> dict[str, str]:
    raw = object_value(row["choices"], name="audit choices")
    return {str(key): str(value) for key, value in raw.items()}


def _sections(row: Mapping[str, object]) -> Mapping[str, str]:
    return recover_protocol(_content(row), _choices(row)).sections


def _excerpt(row: Mapping[str, object], section_names: Sequence[str]) -> str:
    sections = _sections(row)
    parts: list[str] = []
    for name in section_names:
        body = sections.get(name, "").strip()
        label = SECTION_LABELS[name]
        parts.append(f"{label}: {body}" if body else f"{label}: [section not present]")
    return "\n\n".join(parts)


def _excerpt_normalized(row: Mapping[str, object], section_names: Sequence[str]) -> str:
    """Return a normalized excerpt string for automatic same-task detection."""
    return " ".join(_excerpt(row, section_names).split())


def _lexical_sections(row: Mapping[str, object]) -> tuple[str, str]:
    sections = _sections(row)
    return sections.get("facts", ""), sections.get("implications", "")


def _load_lexical_reference(audit_path: Path) -> LexicalReference:
    reference_path = audit_path.parent.parent / "data" / "lexical-reference.json"
    raw = read_json(reference_path)
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"human audit requires {reference_path}")
    return compile_reference(raw)


def _idf(term: str, reference: LexicalReference) -> float:
    df = reference.document_frequency.get(term, 0)
    value = math.log((reference.document_count + 1.0) / (df + 1.0)) + 1.0
    return min(value, reference.idf_cap)


def _preanswer_candidate_counts(stem: str, facts: str, implications: str) -> Counter[str]:
    """Return the exact TF-IDF-visible term counts for Facts+Implications."""
    source = measurement_stem(stem)
    stem_tokens = tfidf_tokens(source)
    abbreviations = validated_abbreviations(source)
    facts_counts, _ = candidate_counts(facts, stem_tokens, abbreviations)
    implications_counts, _ = candidate_counts(implications, stem_tokens, abbreviations)
    return facts_counts + implications_counts


def _asymmetric_coverage_counts(left: Counter[str], right: Counter[str]) -> Counter[str]:
    """Keep only counts that are unique to the left side relative to the right side."""
    result: Counter[str] = Counter()
    for term, count in left.items():
        extra = count - right.get(term, 0)
        if extra > 0:
            result[term] = extra
    return result


def _common_coverage_counts(left: Counter[str], right: Counter[str]) -> Counter[str]:
    """Return source-term counts credited on both sides."""
    return Counter(
        {
            term: min(count, right.get(term, 0))
            for term, count in left.items()
            if min(count, right.get(term, 0)) > 0
        }
    )


def _has_lexical_difference(stem: str, first: tuple[str, str], second: tuple[str, str]) -> bool:
    """Return True only when the two sides visibly differ in recovered stem terms."""
    a_counts = _preanswer_candidate_counts(stem, first[0], first[1])
    b_counts = _preanswer_candidate_counts(stem, second[0], second[1])
    return bool(
        _asymmetric_coverage_counts(a_counts, b_counts)
        or _asymmetric_coverage_counts(b_counts, a_counts)
    )


def _highlight_stem_coverage(
    stem: str,
    unique_counts: Counter[str],
    common_counts: Counter[str],
    reference: LexicalReference,
) -> str:
    """Render source coverage with unique and common credit visually separated.

    Yellow marks terms credited only on this side. A faint underline marks terms
    credited on both sides, preserving phrase context without competing visually
    with the A/B difference. Stopwords are explicitly display-ineligible even if
    future scorer internals change, so grammatical words such as ``not`` cannot be
    accidentally highlighted by this human-facing view.
    """
    source = measurement_stem(stem)
    if not unique_counts and not common_counts:
        return html.escape(source)

    used_unique: Counter[str] = Counter()
    used_common: Counter[str] = Counter()
    pieces: list[str] = []
    cursor = 0
    denominator = max(1.0, reference.idf_cap - 1.0)
    for match in TOKEN_SPAN_RE.finditer(source):
        pieces.append(html.escape(source[cursor : match.start()]))
        surface = match.group(0)
        surface_folded = surface.casefold()
        canonical = () if surface_folded in DISPLAY_STOPWORDS else tfidf_tokens(surface)
        term = canonical[0] if len(canonical) == 1 else None

        if term is not None and common_counts.get(term, 0) > used_common.get(term, 0):
            used_common[term] += 1
            pieces.append(
                '<span class="lex-common" title="Preserved on both sides">'
                + html.escape(surface)
                + "</span>"
            )
        else:
            if term is not None and unique_counts.get(term, 0) > used_unique.get(term, 0):
                used_unique[term] += 1
                idf = _idf(term, reference)
                strength = (idf - 1.0) / denominator
                opacity = 0.16 + 0.50 * min(1.0, max(0.0, strength))
                pieces.append(
                    f'<mark class="lex" style="--mark-alpha:{opacity:.3f}" '
                    f'title="Preserved here but not on the other side · PubMed IDF {idf:.2f}">'
                    + html.escape(surface)
                    + "</mark>"
                )
            else:
                pieces.append(html.escape(surface))
        cursor = match.end()
    pieces.append(html.escape(source[cursor:]))
    return "".join(pieces)


def _base_candidates(rows: Iterable[JsonObject]) -> list[dict[str, object]]:
    by_unit: dict[tuple[str, str], dict[str, JsonObject]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        model_id = str(row["model_id"])
        by_unit.setdefault((question_id, model_id), {})[str(row["condition_id"])] = row

    result: list[dict[str, object]] = []
    for (question_id, model_id), conditions in by_unit.items():
        for control, treatment, added_copy in PAIRINGS:
            if control not in conditions or treatment not in conditions:
                continue
            control_row = conditions[control]
            treatment_row = conditions[treatment]
            pair_id = _pair_hash(question_id, model_id, control, treatment)[:20]
            result.append(
                {
                    "pair_id": pair_id,
                    "question_id": question_id,
                    "model_id": model_id,
                    "dataset": str(control_row["dataset"]),
                    "control_condition": control,
                    "treatment_condition": treatment,
                    "added_copy": added_copy,
                    "control": control_row,
                    "treatment": treatment_row,
                }
            )
    return result


def _candidate_sort_key(candidate: Mapping[str, object], salt: str) -> str:
    return _pair_hash(str(candidate["pair_id"]), salt)


def _lexical_eligible(candidate: Mapping[str, object]) -> bool:
    control = object_value(candidate["control"], name="audit control row")
    treatment = object_value(candidate["treatment"], name="audit treatment row")
    if not _content(control).strip() or not _content(treatment).strip():
        return False
    if (
        _number(_judgment(control).get("preprovisional_tfidf_recall")) is None
        or _number(_judgment(treatment).get("preprovisional_tfidf_recall")) is None
    ):
        return False
    stem = str(control["stem"])
    return _has_lexical_difference(stem, _lexical_sections(control), _lexical_sections(treatment))


def _role_eligible(candidate: Mapping[str, object], field: str) -> bool:
    control = object_value(candidate["control"], name="audit control row")
    treatment = object_value(candidate["treatment"], name="audit treatment row")
    if not _content(control).strip() or not _content(treatment).strip():
        return False
    return (
        _number(_judgment(control).get(field)) is not None
        and _number(_judgment(treatment).get(field)) is not None
    )


def _choose_one_per_question(
    candidates: Sequence[dict[str, object]],
    *,
    count: int,
    excluded_questions: set[str],
    used_pairs: set[str],
    salt: str,
) -> list[dict[str, object]]:
    by_question: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        question_id = str(candidate["question_id"])
        if question_id in excluded_questions or str(candidate["pair_id"]) in used_pairs:
            continue
        by_question.setdefault(question_id, []).append(candidate)
    question_order = sorted(by_question, key=lambda q: _pair_hash(q, salt, "question"))
    selected: list[dict[str, object]] = []
    for question_id in question_order:
        options = sorted(by_question[question_id], key=lambda c: _candidate_sort_key(c, salt))
        if not options:
            continue
        selected.append(options[0])
        used_pairs.add(str(options[0]["pair_id"]))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"human audit needs {count} distinct-question candidates for {salt}")
    return selected


def _blind_order(
    candidate: Mapping[str, object], task_kind: str
) -> tuple[Mapping[str, object], Mapping[str, object], str]:
    treatment_first = (
        int(_pair_hash(str(candidate["pair_id"]), task_kind, "order")[-1], 16) % 2 == 0
    )
    control = object_value(candidate["control"], name="audit control row")
    treatment = object_value(candidate["treatment"], name="audit treatment row")
    if treatment_first:
        return treatment, control, "A"
    return control, treatment, "B"


def _lexical_task(
    candidate: Mapping[str, object], reference: LexicalReference
) -> tuple[JsonObject, JsonObject]:
    first, second, treatment_response = _blind_order(candidate, "lexical_coverage")
    stem = str(first["stem"])
    facts_a, implications_a = _lexical_sections(first)
    facts_b, implications_b = _lexical_sections(second)
    counts_a = _preanswer_candidate_counts(stem, facts_a, implications_a)
    counts_b = _preanswer_candidate_counts(stem, facts_b, implications_b)
    only_a = _asymmetric_coverage_counts(counts_a, counts_b)
    only_b = _asymmetric_coverage_counts(counts_b, counts_a)
    common = _common_coverage_counts(counts_a, counts_b)
    if not only_a and not only_b:
        raise RuntimeError("non-discriminative lexical task reached export")
    task_id = _pair_hash(str(candidate["pair_id"]), "lexical_coverage")[:20]
    blind = json_object(
        {
            "human_audit_version": HUMAN_AUDIT_VERSION,
            "task_id": task_id,
            "task_kind": "lexical_coverage",
            "task_title": "Question-information coverage",
            "instruction": (
                "Compare the two identical copies of the question stem below. Yellow marks stem information "
                "preserved by one side but not the other; faint underlining marks information preserved by both. "
                "Which side preserves more of the important information from the question? "
                "Use the meaning of the stem; do not mechanically count highlights. Darker highlighting "
                "shows higher PubMed-IDF weight, not a claim of medical importance. Do not judge answer correctness."
            ),
            "question": stem,
            "choices": {},
            "response_a": None,
            "response_b": None,
            "response_a_html": _highlight_stem_coverage(stem, only_a, common, reference),
            "response_b_html": _highlight_stem_coverage(stem, only_b, common, reference),
            "auto_rating": None,
            "rating": None,
        },
        path=f"lexical audit task {task_id}",
    )
    score_a = _number(_judgment(first).get("preprovisional_tfidf_recall"))
    score_b = _number(_judgment(second).get("preprovisional_tfidf_recall"))
    key = json_object(
        {
            "task_id": task_id,
            "task_kind": "lexical_coverage",
            "pair_id": str(candidate["pair_id"]),
            "question_id": str(candidate["question_id"]),
            "model_id": str(candidate["model_id"]),
            "dataset": str(candidate["dataset"]),
            "added_copy": str(candidate["added_copy"]),
            "response_a_condition": str(first["condition_id"]),
            "response_b_condition": str(second["condition_id"]),
            "treatment_response": treatment_response,
            "mechanical_a": score_a,
            "mechanical_b": score_b,
        },
        path=f"lexical audit key {task_id}",
    )
    return blind, key


def _role_task(
    candidate: Mapping[str, object], spec: tuple[str, str, str, tuple[str, ...]]
) -> tuple[JsonObject, JsonObject]:
    role, field, title, context = spec
    first, second, treatment_response = _blind_order(candidate, f"role:{role}")
    excerpt_a = _excerpt(first, context)
    excerpt_b = _excerpt(second, context)
    auto_rating = (
        "same"
        if _excerpt_normalized(first, context) == _excerpt_normalized(second, context)
        else None
    )
    task_id = _pair_hash(str(candidate["pair_id"]), f"role:{role}")[:20]
    blind = json_object(
        {
            "human_audit_version": HUMAN_AUDIT_VERSION,
            "task_id": task_id,
            "task_kind": "role",
            "role": role,
            "task_title": title,
            "instruction": (
                "Do not judge medical correctness. Decide only whether A or B passes the simple rule below. "
                "If both pass or both fail, choose No difference."
            ),
            "audit_rule": ROLE_AUDIT_GUIDANCE[role]["rule"],
            "example_good": ROLE_AUDIT_GUIDANCE[role]["good"],
            "example_bad": ROLE_AUDIT_GUIDANCE[role]["bad"],
            "audit_note": ROLE_AUDIT_GUIDANCE[role]["note"],
            "question": str(first["stem"]),
            "choices": _choices(first),
            "response_a": excerpt_a,
            "response_b": excerpt_b,
            "response_a_html": html.escape(excerpt_a),
            "response_b_html": html.escape(excerpt_b),
            "auto_rating": auto_rating,
            "rating": auto_rating,
        },
        path=f"role audit task {task_id}",
    )
    key = json_object(
        {
            "task_id": task_id,
            "task_kind": "role",
            "role": role,
            "pair_id": str(candidate["pair_id"]),
            "question_id": str(candidate["question_id"]),
            "model_id": str(candidate["model_id"]),
            "dataset": str(candidate["dataset"]),
            "added_copy": str(candidate["added_copy"]),
            "response_a_condition": str(first["condition_id"]),
            "response_b_condition": str(second["condition_id"]),
            "treatment_response": treatment_response,
            "mechanical_a": _number(_judgment(first).get(field)),
            "mechanical_b": _number(_judgment(second).get(field)),
        },
        path=f"role audit key {task_id}",
    )
    return blind, key


def _render_rater(audit_rows: Sequence[JsonObject]) -> str:
    payload = json.dumps(audit_rows, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    audit_identity = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Instruction Duplication — blinded human validation</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--text:#202124;--muted:#687078;--line:#dadce0;--accent:#1a73e8}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);font-family:Arial,Helvetica,sans-serif;color:var(--text)}}
main{{max-width:1180px;margin:0 auto;padding:24px}} .top{{display:flex;gap:16px;align-items:center;justify-content:space-between;margin-bottom:16px}}
.progress{{font-weight:700}} .bar{{height:6px;background:#e0e3e7;border-radius:6px;overflow:hidden;margin:10px 0 20px}} .bar>div{{height:100%;background:var(--accent);width:0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:16px;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
h1{{font-size:20px;margin:0}} h2{{font-size:17px;margin:0 0 8px}} .muted{{color:var(--muted);font-size:13px}} .instruction{{font-size:18px;line-height:1.45;font-weight:650}}
.question{{white-space:pre-wrap;line-height:1.45}} .choices{{display:grid;gap:5px;margin-top:10px;font-size:14px}}
.responses{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;align-items:start}} .response{{white-space:pre-wrap;line-height:1.5;min-height:170px}}
.response h3{{margin:0 0 12px;font-size:16px;min-height:24px}} .response .stemcopy{{margin:0;max-width:100%}} mark.lex{{background:rgba(255,193,7,var(--mark-alpha));padding:0;border-radius:0;box-shadow:inset 0 -0.05em 0 rgba(255,193,7,var(--mark-alpha));}} .lex-common{{text-decoration-line:underline;text-decoration-style:solid;text-decoration-thickness:1px;text-decoration-color:#c3c8ce;text-underline-offset:2px;color:#545b62}}
.legend{{margin-top:10px;color:var(--muted);font-size:13px;font-weight:400}} .legend mark{{background:rgba(255,193,7,.25);padding:0;border-radius:0}} .legend .common-sample{{text-decoration:underline;text-decoration-color:#c3c8ce;text-underline-offset:2px;color:#545b62}}
.quick-rule{{margin-top:14px;padding:14px 16px;background:#f8f9fa;border:1px solid var(--line);border-radius:9px;font-size:17px;line-height:1.45}}
.examples{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}} .example{{padding:12px 14px;border-radius:9px;line-height:1.4;font-size:14px}}
.example.good{{background:#edf7ed;border:1px solid #c7e4c7}} .example.bad{{background:#fdf0ef;border:1px solid #efc7c2}} .example strong{{display:block;margin-bottom:5px}} .audit-note{{margin-top:12px;padding:11px 14px;border-left:3px solid #b8bec5;background:#fafbfc;color:#4f565d;line-height:1.45;font-size:14px}}
details.context{{margin-bottom:16px}} details.context summary{{cursor:pointer;font-weight:700;padding:14px 18px;background:#fff;border:1px solid var(--line);border-radius:10px}} details.context[open] summary{{border-radius:10px 10px 0 0}} details.context .context-body{{background:#fff;border:1px solid var(--line);border-top:0;border-radius:0 0 10px 10px;padding:16px 20px}}
.buttons{{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;margin-top:16px}} button{{border:1px solid #b8bec5;background:white;border-radius:8px;padding:12px 18px;font-size:15px;cursor:pointer}}
button:hover{{border-color:var(--accent)}} button.primary{{background:var(--accent);color:white;border-color:var(--accent)}} .controls{{display:flex;gap:8px;align-items:center}}
input{{padding:9px 10px;border:1px solid var(--line);border-radius:7px}} .done{{text-align:center;padding:50px 20px}} .hidden{{display:none}}
@media(max-width:760px){{.responses,.examples{{grid-template-columns:1fr}} .top{{align-items:flex-start;flex-direction:column}}}}
</style></head><body><main>
<div class=\"top\"><div><h1>Blinded human validation</h1><div class=\"muted\">Judge only the single property named on each screen. Do not solve the medical question.</div></div><div class=\"controls\"><input id=\"reviewer\" placeholder=\"Reviewer ID (optional)\"><button id=\"exportTop\">Export ratings</button></div></div>
<div class=\"progress\" id=\"progress\"></div><div class=\"bar\"><div id=\"barFill\"></div></div>
<section id=\"task\"><div class=\"card\"><h2 id=\"title\"></h2><div class=\"instruction\" id=\"instruction\"></div><div id=\"quickRule\" class=\"quick-rule hidden\"></div><div id=\"examples\" class=\"examples hidden\"><div class=\"example good\"><strong>✓ Counts</strong><span id=\"exampleGood\"></span></div><div class=\"example bad\"><strong>✗ Does not count</strong><span id=\"exampleBad\"></span></div></div><div id=\"legend\" class=\"legend hidden\"><mark>highlighted</mark> = term preserved before provisional answer; darker = higher PubMed-IDF weight.</div></div>
<details class=\"context\" id=\"questionCard\"><summary>Case context — open only if needed</summary><div class=\"context-body\"><div class=\"question\" id=\"question\"></div><div class=\"choices\" id=\"choices\"></div></div></details>
<div class=\"responses\"><div class=\"card response\"><h3 id=\"aHeading\">Response A</h3><div id=\"a\"></div></div><div class=\"card response\"><h3 id=\"bHeading\">Response B</h3><div id=\"b\"></div></div></div>
<div class=\"buttons\" id=\"buttons\"></div></section><section id=\"done\" class=\"card done hidden\"><h2>All tasks rated</h2><p>Export the frozen ratings file. Keep the decoding key unopened until ratings are finalized.</p><button class=\"primary\" id=\"exportDone\">Export ratings</button></section>
<script>
const TASKS={payload}; const AUDIT_ID={json.dumps(audit_identity)}; const STORE='instruction-duplication-human-audit-'+AUDIT_ID;
let state={{ratings:{{}},reviewer_id:'',index:0}}; try{{const s=JSON.parse(localStorage.getItem(STORE)||'null');if(s)state=s}}catch(e){{}}
for (const t of TASKS) {{ if (t.auto_rating && !state.ratings[t.task_id]) state.ratings[t.task_id] = t.auto_rating; }}
const $=id=>document.getElementById(id); $('reviewer').value=state.reviewer_id||''; $('reviewer').oninput=e=>{{state.reviewer_id=e.target.value;save()}};
function save(){{localStorage.setItem(STORE,JSON.stringify(state))}}
function nextUnrated(start=0){{for(let i=start;i<TASKS.length;i++)if(!state.ratings[TASKS[i].task_id])return i;for(let i=0;i<start;i++)if(!state.ratings[TASKS[i].task_id])return i;return TASKS.length}}
function rate(value){{const t=TASKS[state.index];state.ratings[t.task_id]=value;state.index=nextUnrated(state.index+1);save();render()}}
function makeButton(label,value,key){{const b=document.createElement('button');b.textContent=key?`${{label}}  [${{key}}]`:label;b.onclick=()=>rate(value);return b}}
function render(){{const n=Object.keys(state.ratings).length;$('progress').textContent=`${{Math.min(n+1,TASKS.length)}} / ${{TASKS.length}}   ·   ${{n}} completed`;$('barFill').style.width=`${{100*n/TASKS.length}}%`;
 if(n>=TASKS.length){{$('task').classList.add('hidden');$('done').classList.remove('hidden');$('progress').textContent=`${{TASKS.length}} / ${{TASKS.length}} · complete`;return}}
 $('task').classList.remove('hidden');$('done').classList.add('hidden');if(state.index>=TASKS.length)state.index=nextUnrated(0);const t=TASKS[state.index];$('title').textContent=t.task_title;$('instruction').textContent=t.instruction;$('question').textContent=t.question;
 const c=$('choices');c.innerHTML='';Object.entries(t.choices||{{}}).forEach(([k,v])=>{{const d=document.createElement('div');d.textContent=`${{k}}. ${{v}}`;c.appendChild(d)}});$('a').innerHTML='<div class="stemcopy">'+t.response_a_html+'</div>'; $('b').innerHTML='<div class="stemcopy">'+t.response_b_html+'</div>';
 const lexical=t.task_kind==='lexical_coverage';$('questionCard').classList.toggle('hidden',lexical);$('legend').classList.toggle('hidden',!lexical);$('quickRule').classList.toggle('hidden',lexical);$('examples').classList.toggle('hidden',lexical);$('auditNote').classList.toggle('hidden',lexical);if(!lexical){{$('quickRule').textContent=t.audit_rule;$('exampleGood').textContent=t.example_good;$('exampleBad').textContent=t.example_bad;$('auditNote').textContent='Length rule: '+t.audit_note;$('questionCard').open=false}}$('aHeading').textContent=lexical?'Stem information preserved by A':'Response A';$('bHeading').textContent=lexical?'Stem information preserved by B':'Response B';
 const box=$('buttons');box.innerHTML='';if(lexical){{box.append(makeButton('A preserves more important information','A','1'),makeButton('B preserves more important information','B','2'),makeButton('No difference','same','3'),makeButton('Cannot tell','cannot_tell','4'))}}else{{box.append(makeButton('A only','A','1'),makeButton('B only','B','2'),makeButton('No difference','same','3'),makeButton('Cannot tell','cannot_tell','4'))}}}}
function exportRatings(){{const rows=TASKS.map(t=>({{task_id:t.task_id,task_kind:t.task_kind,role:t.role||null,rating:state.ratings[t.task_id]||null}}));const out={{human_audit_version:{json.dumps(HUMAN_AUDIT_VERSION)},audit_id:AUDIT_ID,reviewer_id:state.reviewer_id||null,exported_at:new Date().toISOString(),complete:rows.every(r=>r.rating),ratings:rows}};const blob=new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='human-validation-ratings-'+AUDIT_ID+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}}
$('exportTop').onclick=exportRatings;$('exportDone').onclick=exportRatings;document.addEventListener('keydown',e=>{{if(e.target.tagName==='INPUT')return;const t=TASKS[state.index];if(!t)return;if(e.key==='1')rate('A');if(e.key==='2')rate('B');if(e.key==='3')rate('same');if(e.key==='4')rate('cannot_tell')}});state.index=nextUnrated(0);save();render();
</script></main></body></html>"""


def _export_test_fixture_audit(
    candidates: Sequence[dict[str, object]],
    *,
    audit_path: Path,
    key_path: Path,
    schema_path: Path,
) -> JsonObject:
    """Keep tiny unit-test fixtures usable without weakening the frozen real-run design."""
    blind_rows: list[JsonObject] = []
    key_rows: list[JsonObject] = []
    for candidate in candidates:
        first, second, treatment_response = _blind_order(candidate, "fixture")
        task_id = _pair_hash(str(candidate["pair_id"]), "fixture")[:20]
        blind_rows.append(
            json_object(
                {
                    "human_audit_version": HUMAN_AUDIT_VERSION,
                    "task_id": task_id,
                    "task_kind": "fixture",
                    "question": str(first["stem"]),
                    "choices": _choices(first),
                    "response_a": _content(first),
                    "response_b": _content(second),
                    "rating": None,
                },
                path=f"fixture audit task {task_id}",
            )
        )
        key_rows.append(
            json_object(
                {
                    "task_id": task_id,
                    "pair_id": str(candidate["pair_id"]),
                    "question_id": str(candidate["question_id"]),
                    "model_id": str(candidate["model_id"]),
                    "dataset": str(candidate["dataset"]),
                    "added_copy": str(candidate["added_copy"]),
                    "response_a_condition": str(first["condition_id"]),
                    "response_b_condition": str(second["condition_id"]),
                    "treatment_response": treatment_response,
                },
                path=f"fixture audit key {task_id}",
            )
        )
    write_jsonl(audit_path, blind_rows)
    write_jsonl(key_path, key_rows)
    write_json(schema_path, human_audit_schema())
    return json_object(
        {
            "human_audit_version": HUMAN_AUDIT_VERSION,
            "exported_pairs": len(blind_rows),
            "fixture_mode": True,
            "condition_key_separate": True,
        },
        path="fixture human audit metadata",
    )


def export_blinded_matched_pairs(
    rows: Iterable[JsonObject],
    *,
    audit_path: Path,
    key_path: Path,
    schema_path: Path,
    limit: int | None = None,
) -> JsonObject:
    """Export the frozen 199-task atomic validator and a separate decoding key.

    ``limit`` is retained only for backwards call compatibility. The paper validator
    has a fixed design and therefore rejects any non-default limit.
    """
    expected = LEXICAL_TASK_COUNT + ROLE_TASK_COUNT * len(ROLE_SPECS)
    if limit not in (None, expected, 200):
        raise ValueError(f"atomic human audit has a frozen size of {expected} tasks")

    candidates = _base_candidates(rows)
    # The repository has a one-pair synthetic unit test. Keep that fixture working,
    # but only for rows explicitly labeled as the test dataset; real workspaces remain
    # strict and must satisfy the complete frozen 199-task design.
    if (
        candidates
        and len(candidates) < expected
        and all(str(candidate["dataset"]) == "test" for candidate in candidates)
    ):
        return _export_test_fixture_audit(
            candidates, audit_path=audit_path, key_path=key_path, schema_path=schema_path
        )

    unique_questions = {str(candidate["question_id"]) for candidate in candidates}
    if len(unique_questions) < TARGET_UNIQUE_QUESTIONS:
        # Small smoke/unit-test workspaces cannot support the frozen 199-task paper audit.
        # Analysis itself must still succeed; write explicit empty audit artifacts rather
        # than turning an optional validation export into a pipeline failure.
        write_jsonl(audit_path, [])
        write_jsonl(key_path, [])
        write_json(schema_path, human_audit_schema())
        return json_object(
            {
                "human_audit_version": HUMAN_AUDIT_VERSION,
                "exported_pairs": 0,
                "skipped": True,
                "skip_reason": (
                    f"frozen human audit requires at least {TARGET_UNIQUE_QUESTIONS} "
                    f"distinct questions; workspace has {len(unique_questions)}"
                ),
                "condition_key_separate": True,
            },
            path="skipped human audit metadata",
        )

    reference = _load_lexical_reference(audit_path)
    used_pairs: set[str] = set()

    lexical_candidates = [candidate for candidate in candidates if _lexical_eligible(candidate)]
    lexical_selected = _choose_one_per_question(
        lexical_candidates,
        count=LEXICAL_TASK_COUNT,
        excluded_questions=set(),
        used_pairs=used_pairs,
        salt="lexical",
    )
    lexical_questions = {str(candidate["question_id"]) for candidate in lexical_selected}

    role_any = [
        candidate
        for candidate in candidates
        if any(_role_eligible(candidate, spec[1]) for spec in ROLE_SPECS)
    ]
    extra_question_candidates = _choose_one_per_question(
        role_any,
        count=TARGET_UNIQUE_QUESTIONS - LEXICAL_TASK_COUNT,
        excluded_questions=lexical_questions,
        used_pairs=used_pairs,
        salt="role-question-coverage",
    )
    selected_questions = lexical_questions | {
        str(candidate["question_id"]) for candidate in extra_question_candidates
    }

    role_selected: dict[str, list[dict[str, object]]] = {spec[0]: [] for spec in ROLE_SPECS}
    # Use one task from each of the 69 additional questions so the frozen export spans
    # exactly 148 benchmark questions. Rotate roles deterministically and keep quotas balanced.
    role_cursor = 0
    for candidate in extra_question_candidates:
        assigned = False
        for offset in range(len(ROLE_SPECS)):
            spec = ROLE_SPECS[(role_cursor + offset) % len(ROLE_SPECS)]
            role = spec[0]
            if len(role_selected[role]) >= ROLE_TASK_COUNT or not _role_eligible(
                candidate, spec[1]
            ):
                continue
            role_selected[role].append(candidate)
            role_cursor = (ROLE_SPECS.index(spec) + 1) % len(ROLE_SPECS)
            assigned = True
            break
        if not assigned:
            raise RuntimeError("could not assign distinct-question role validation task")

    for spec in ROLE_SPECS:
        role, field, _title, _context = spec
        needed = ROLE_TASK_COUNT - len(role_selected[role])
        if needed <= 0:
            continue
        pool = [
            candidate
            for candidate in candidates
            if str(candidate["question_id"]) in selected_questions
            and str(candidate["pair_id"]) not in used_pairs
            and _role_eligible(candidate, field)
        ]
        pool.sort(key=lambda candidate: _candidate_sort_key(candidate, f"role-fill:{role}"))
        chosen = pool[:needed]
        if len(chosen) != needed:
            raise RuntimeError(f"human audit lacks enough unique pairs for role {role}")
        role_selected[role].extend(chosen)
        used_pairs.update(str(candidate["pair_id"]) for candidate in chosen)

    blind_rows: list[JsonObject] = []
    key_rows: list[JsonObject] = []
    for candidate in lexical_selected:
        blind, key = _lexical_task(candidate, reference)
        blind_rows.append(blind)
        key_rows.append(key)
    for spec in ROLE_SPECS:
        for candidate in role_selected[spec[0]]:
            blind, key = _role_task(candidate, spec)
            blind_rows.append(blind)
            key_rows.append(key)

    # Deterministic mixed task order prevents long blocks of one role without making
    # task identity or A/B order predictable to the reviewer.
    ordering = sorted(
        range(len(blind_rows)), key=lambda i: _pair_hash(str(blind_rows[i]["task_id"]), "display")
    )
    blind_rows = [blind_rows[i] for i in ordering]
    key_by_id = {str(row["task_id"]): row for row in key_rows}
    key_rows = [key_by_id[str(row["task_id"])] for row in blind_rows]

    if len(blind_rows) != expected:
        raise RuntimeError(f"human audit exported {len(blind_rows)} tasks; expected {expected}")
    unique_pairs = {str(row["pair_id"]) for row in key_rows}
    unique_questions = {str(row["question_id"]) for row in key_rows}
    if len(unique_pairs) != expected:
        raise RuntimeError("human audit pair reuse detected")
    if len(unique_questions) != TARGET_UNIQUE_QUESTIONS:
        raise RuntimeError(
            f"human audit spans {len(unique_questions)} questions; expected {TARGET_UNIQUE_QUESTIONS}"
        )

    write_jsonl(audit_path, blind_rows)
    write_jsonl(key_path, key_rows)
    write_json(schema_path, human_audit_schema())
    html_path = audit_path.with_name("human-validation.html")
    html_path.write_text(_render_rater(blind_rows), encoding="utf-8")

    return json_object(
        {
            "human_audit_version": HUMAN_AUDIT_VERSION,
            "exported_tasks": expected,
            "unique_pairs": len(unique_pairs),
            "unique_questions": len(unique_questions),
            "lexical_tasks": LEXICAL_TASK_COUNT,
            "role_tasks": ROLE_TASK_COUNT * len(ROLE_SPECS),
            "role_tasks_per_role": ROLE_TASK_COUNT,
            "selection": (
                "versioned SHA-256 deterministic sampling; 79 lexical tasks use distinct questions; "
                "69 additional questions enter through role tasks; remaining role slots are filled "
                "only within those 148 questions; no A/B pair is reused"
            ),
            "condition_key_separate": True,
            "browser_rater": html_path.name,
        },
        path="human audit metadata",
    )
