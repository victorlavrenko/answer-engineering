"""Deterministic content, structure, and repair-oriented measurements.

Paper mapping: implements section completion, validated role completion, contrastive
discussion, premature commitment, final-answer accuracy, and the all-eight diagnostic
described in ``Experiment -> Measurements and Eligibility``. Lexical exposure is
delegated to :mod:`instruction_duplication.lexical`."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import TypedDict

from .answer_utils import ExtractedAnswer, extract_answer, normalize_text
from .facts import QuestionFacts, build_question_facts, score_fact_inventory
from .json_types import JsonObject
from .lexical import LexicalReference, content_tokens, score_anchor_recall, score_preanswer
from .manifest import JUDGE_VERSION
from .trajectory import (
    CONTENT_TAGS,
    VALID_DECISIONS,
    RecoveredProtocol,
    recover_protocol,
)
from .types import CellStatus, Question


class CommitmentEvent(TypedDict):
    """One literal answer commitment detected before the provisional answer."""

    option: str | None
    start: int
    evidence: str


class SectionDiagnostics(TypedDict):
    """Per-role visibility and content layers."""

    present: dict[str, bool]
    unique: dict[str, bool]
    nontrivial: dict[str, bool]
    schema_valid: dict[str, bool]
    present_count: int
    unique_count: int
    nontrivial_count: int
    substantive_count: int
    schema_valid_count: int


class Judgment(TypedDict):
    """Typed deterministic judgment payload shared with storage and analysis."""

    judge_version: str
    generation_usable: bool
    generation_status: str
    repair_endpoint_eligible: bool
    repair_endpoint_inapplicability_reason: str | None
    essential_fact_count: int
    fact_complexity: str
    fact_inventory_review_status: str
    protocol_scaffold_complete: float | None
    preanswer_discussion_complete: float | None
    visible_reasoning_scaffold_complete: float | None
    visible_preanswer_discussion_complete: float | None
    minimum_repair_scaffold: float | None
    contrastive_scaffold_complete: float | None
    correct_with_minimum_repair_scaffold: float | None
    protocol_format_score: float | None
    structure_score: float | None
    section_depth_score: float | None
    substantive_role_score: float | None
    contrastive_role_score: float | None
    visible_substantive_role_score: float | None
    visible_contrastive_role_score: float | None
    identified_role_count: int | None
    unique_role_count: int | None
    required_section_count: int | None
    nontrivial_section_count: int | None
    substantive_role_count: int | None
    validated_role_count: int | None
    validated_role_completeness_score: float | None
    contrastive_discussion_count: int | None
    contrastive_discussion_score: float | None
    validated_contrastive_discussion_count: int | None
    validated_contrastive_discussion_score: float | None
    provisional_answer_discussed: float | None
    best_alternative_discussed: float | None
    decisive_distinction_discussed: float | None
    answer_change_discussed: float | None
    reconsideration_discussed: float | None
    role_completeness_score: float | None
    all_roles_substantive: float | None
    all_roles_validated_complete: float | None
    validated_complete_role_scaffold: float | None
    roles_in_requested_order: float | None
    complete_role_scaffold: float | None
    role_facts_complete: float | None
    role_implications_complete: float | None
    role_provisional_answer_complete: float | None
    role_best_alternative_complete: float | None
    role_decisive_distinction_complete: float | None
    role_answer_changing_change_complete: float | None
    role_answer_changing_change_nontrivial: float | None
    role_reconsideration_complete: float | None
    role_final_answer_complete: float | None
    facts_tfidf_recall: float | None
    implications_tfidf_recall: float | None
    preanswer_tfidf_recall: float | None
    preprovisional_tfidf_recall: float | None
    preanswer_high_idf_tfidf_recall: float | None
    facts_implications_shared_tfidf_recall: float | None
    facts_implications_shared_high_idf_recall: float | None
    preanswer_tfidf_density_per_100_tokens: float | None
    preanswer_tfidf_token_count: float | None
    facts_anchor_recall: float | None
    implications_anchor_recall: float | None
    preanswer_anchor_recall: float | None
    shared_anchor_recall: float | None
    atomic_fact_coverage: float | None
    atomic_implication_trace_coverage: float | None
    hard_qualifier_fact_recall: float | None
    hard_qualifier_trace_recall: float | None
    fact_coverage_details: list[JsonObject]
    preprovisional_commitment_observed: float | None
    preprovisional_commitment_itt: float | None
    trajectory_complete: bool | None
    semantic_trajectory_complete: bool | None
    accuracy: int
    accuracy_parseable: bool
    final_option: str | None
    answer_extraction_status: str
    trajectory_errors: list[str]
    answer_candidates: list[str]
    trajectory_criteria: dict[str, bool]
    section_substantive: dict[str, bool]
    section_token_counts: dict[str, int]
    section_marker_counts: dict[str, int]
    section_present: dict[str, bool]
    section_unique: dict[str, bool]
    section_nontrivial: dict[str, bool]
    section_schema_valid: dict[str, bool]
    section_present_count: int | None
    section_unique_count: int | None
    section_nontrivial_count: int | None
    section_substantive_count: int | None
    section_schema_valid_count: int | None
    section_present_fraction: float | None
    section_unique_fraction: float | None
    section_nontrivial_fraction: float | None
    section_substantive_fraction: float | None
    section_schema_valid_fraction: float | None
    facts_choice_list_leakage: bool | None
    provisional_option: str | None
    second_best_option: str | None
    rereasoning_decision: str | None
    decision_consistent: bool
    strict_decision_consistent: bool
    early_commitments: list[CommitmentEvent]


CHOICE_LIST_MARKER_RE = re.compile(
    r"\b(?:option|choice|answer)\s*\(?((?-i:[A-Z]))\)?\b",
    re.IGNORECASE,
)
COUNTERFACTUAL_CUES = re.compile(
    r"\b(?:if|were|instead|without|absence|presence|would|become|"
    r"chang(?:e|ed|es|ing)|add(?:ed|ing|ition)|remov(?:e|ed|es|ing|al)|"
    r"replac(?:e|ed|es|ing|ement)|alter(?:ed|ing|ation)?)\b",
    re.IGNORECASE,
)
COUNTERFACTUAL_TASK_REWRITE_RE = re.compile(
    # Explicitly changing what the item asks is not a change to the patient's case.
    r"\b(?:rephras(?:e|ed|ing)|rewrit(?:e|ten|ing)|chang(?:e|ed|ing)|modif(?:y|ied|ying))\b"
    r"[^.;\n]{0,60}\b(?:stem|question)\b[^.;\n]{0,80}\b(?:ask|focus|word|phrasing|intent)\b"
    r"|\b(?:if\s+)?the\s+(?:stem|question)\s+(?:were\s+|was\s+)?"
    r"(?:rephrased|rewritten|changed|modified)\s+to\s+(?:ask|focus|word)\b"
    r"|\b(?:if\s+)?the\s+(?:stem|question)\s+(?:were\s+|was\s+)?"
    r"(?:asked|asking|focused)\b"
    r"|\b(?:shift|change|alter)\w*\b[^.;\n]{0,60}\b"
    r"(?:question(?:'s)?\s+(?:focus|intent)|focus\s+of\s+(?:the\s+)?question|"
    r"what\s+the\s+question\s+asks)\b"
    # Natural rewrites that escaped the older patterns: "question were changed to:\n"
    # followed by a new WH-task, or "question specifically asked about ...".
    r"|\b(?:if\s+)?(?:the\s+)?(?:stem|question)\s+"
    r"(?:were\s+|was\s+|is\s+|had\s+)?(?:rephrased|rewritten|changed|modified)\s+to\s*"
    r"[:,'\"“”\s-]*(?:ask\b|focus\b|word\b|which\b|what\b|who\b|when\b|where\b|how\b)"
    r"|\b(?:if\s+)?(?:the\s+)?(?:stem|question)\s+(?:specifically\s+)?"
    r"(?:asks?|asked|asking)\s+(?:about|for|which|what|who|when|where|how)\b"
    r"|\b(?:if\s+)?(?:the\s+)?(?:stem|question)\s+(?:had\s+)?stated\s*"
    r"[,:'\"“”\s-]+(?:which|what|who|when|where|how)\b"
    r"|\b(?:if\s+)?(?:the\s+)?(?:stem|question)\s+specified\s*"
    r"[,:'\"“”\s-]*(?:which|what|who|when|where|how)\b"
    # Rewriting an answer choice is likewise not a patient/case counterfactual.
    r"|\b(?:if\s+)?(?:option|choice)\s+\(?(?-i:[A-Z])\)?[^.;\n]{0,45}"
    r"\b(?:had\s+said|were\s+(?:changed|rewritten|rephrased)|"
    r"was\s+(?:changed|rewritten|rephrased))\b",
    re.IGNORECASE,
)

COUNTERFACTUAL_TARGET_RES = (
    # Causal target after an explicit make/favor construction.  Bare option letters
    # are allowed only in such target positions so pronouns (notably option I) do
    # not become accidental winners.
    re.compile(
        r"\b(?:make|making)\s+(?:the\s+best\s+alternative\s*)?"
        r"(?:(?:option|choice|answer)\s*)?\(?(?-i:((?-i:[A-Z])))\)?[.):\-]?[^;\n]{0,45}?"
        r"(?:the\s+)?(?:best|correct|most\s+likely|more\s+likely|most\s+appropriate|more\s+appropriate|preferred|better)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:would|could|might)\s+(?:then\s+)?(?:favor|favour|support|point\s+to)\s+"
        r"(?:(?:option|choice)\s*)?\(?(?-i:((?-i:[A-Z])))\)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:support|favor|favour)\s+(?:(?:option|choice)\s*)?\(?(?-i:((?-i:[A-Z])))\)?"
        r"\s+(?:over|rather\s+than)\s+(?:(?:option|choice)\s*)?\(?(?-i:[A-Z])\)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:answer|choice|diagnosis|treatment|selection)\s+would\s+"
        r"(?:then\s+)?(?:shift|change|switch)\s+to\s+"
        r"(?:(?:option|choice)\s*)?\(?(?-i:((?-i:[A-Z])))\)?\b",
        re.IGNORECASE,
    ),
    # Direct answer-target language is common in the requested counterfactual role.
    # ``What would change the answer to A is if ...`` already declares A as the
    # resulting answer; the hypothetical facts follow that declaration.
    re.compile(
        r"\bwhat\s+would\s+change\s+(?:the\s+)?(?:answer|choice|diagnosis|selection)\s+to\s+"
        r"(?:(?:option|choice)\s*)?\(?(?-i:((?-i:[A-Z])))\)?[.):\-]?\b",
        re.IGNORECASE,
    ),
    # Tight label -> modal -> winner predicate form. Keeping the span short avoids
    # attaching a later option's predicate to an earlier referenced option.
    re.compile(
        r"\b(?:option|choice|answer)\s*\(?(?-i:((?-i:[A-Z])))\)?[.):\-]?\s+"
        r"(?:would|could|might)\s+(?:then\s+)?(?:become|be|represent)\s+(?:the\s+)?"
        r"(?:best|correct|most\s+likely|more\s+likely|most\s+appropriate|more\s+appropriate|"
        r"preferred|better|more\s+suitable|most\s+suitable|most\s+significant|strongest|leading)\b",
        re.IGNORECASE,
    ),
    # A conventional option label (``C.`` / ``(C)`` / ``option C``) followed by
    # an explicit winning predicate. Do not cross another explicit option label;
    # otherwise ``choice C would become a feature, and choice A would be correct``
    # can incorrectly award C the predicate that belongs to A.
    re.compile(
        r"(?:\b(?:option|choice|answer)\s*\(?(?-i:((?-i:[A-Z])))\)?|"
        r"\((?-i:((?-i:[A-Z])))[.):\-]?|\b(?-i:((?-i:[A-Z])))[.)])"
        r"(?:(?!\b(?:option|choice|answer)\s*\(?(?-i:[A-Z])\)?\b)[^.;\n]){0,110}"
        r"\b(?:would|could|might)\s+(?:then\s+)?"
        r"(?:become|be|represent)\s+(?:the\s+)?"
        r"(?:best|correct|most\s+likely|more\s+likely|most\s+appropriate|more\s+appropriate|"
        r"preferred|better|more\s+suitable|most\s+suitable|most\s+significant|strongest|leading)\b",
        re.IGNORECASE,
    ),
    # Choice text is often followed by its option in parentheses: ``quetiapine (C)``
    # or preceded by a compact label: ``D (ectopic gastrin...)``.
    re.compile(
        r"\((?-i:((?-i:[A-Z])))\)|\b(?-i:((?-i:[A-Z])))\s*\([^.;\n)]{1,120}\)"
        r"[^.;\n]{0,100}\b(?:would|could|might)\s+(?:then\s+)?"
        r"(?:become|be|represent)\s+(?:the\s+)?"
        r"(?:best|correct|most\s+likely|more\s+likely|most\s+appropriate|more\s+appropriate|"
        r"preferred|better|more\s+suitable|most\s+suitable|most\s+significant|strongest|leading)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\((?-i:((?-i:[A-Z])))\)[^.;\n]{0,35}\b(?:the\s+)?"
        r"(?:best|correct|most\s+likely|most\s+appropriate|preferred)\s+"
        r"(?:answer|choice|diagnosis|treatment|option|approach)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:change|shift|switch)\s+(?:the\s+)?(?:answer|choice|diagnosis|selection)\s+"
        r"from\s+(?:(?:option|choice)\s*)?\(?(?-i:[A-Z])\)?[^.;\n]{0,25}?\bto\s+"
        r"(?:(?:option|choice)\s*)?\(?(?-i:((?-i:[A-Z])))\)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:answer|choice|diagnosis|selection)\s+(?:would|could|might)\s+"
        r"(?:then\s+)?(?:change|shift|switch)\s+from\s+"
        r"(?:(?:option|choice)\s*)?\(?(?-i:[A-Z])\)?[^.;\n]{0,25}?\bto\s+"
        r"(?:(?:option|choice)\s*)?\(?(?-i:((?-i:[A-Z])))\)?\b",
        re.IGNORECASE,
    ),
)

COUNTERFACTUAL_META_TOKENS = frozenset(
    {
        "actual",
        "another",
        "appropriate",
        "answer",
        "because",
        "become",
        "better",
        "case",
        "change",
        "changed",
        "choice",
        "correct",
        "current",
        "decisive",
        "different",
        "fact",
        "feature",
        "finding",
        "fit",
        "if",
        "information",
        "likely",
        "new",
        "other",
        "instead",
        "make",
        "more",
        "option",
        "present",
        "provided",
        "scenario",
        "smallest",
        "state",
        "stem",
        "substantive",
        "support",
        "supports",
        "suitable",
        "were",
        "why",
        "would",
    }
)
RATIONALE_CUES = re.compile(
    r"\b(?:because|since|given|due\s+to|therefore|thus|hence|explains?|fits?|supports?|"
    r"consistent\s+with|inconsistent\s+with|argues?\s+against|rules?\s+out|"
    r"less\s+likely|more\s+likely|whereas|however|despite|although|but)\b",
    re.IGNORECASE,
)
RATIONALE_META_TOKENS = frozenset(
    {
        "actual",
        "answer",
        "appear",
        "appears",
        "appropriate",
        "best",
        "candidate",
        "choice",
        "correct",
        "current",
        "currently",
        "decision",
        "explain",
        "explanation",
        "fit",
        "fits",
        "given",
        "likely",
        "lose",
        "loses",
        "losing",
        "lost",
        "make",
        "makes",
        "making",
        "match",
        "matches",
        "matching",
        "more",
        "most",
        "none",
        "option",
        "options",
        "other",
        "others",
        "plausible",
        "potentially",
        "preferred",
        "preference",
        "provisional",
        "rationale",
        "reason",
        "reasoning",
        "second",
        "select",
        "selected",
        "seem",
        "seems",
        "state",
        "stem",
        "suitable",
        "support",
        "supports",
        "valid",
        "viable",
        "why",
    }
)
CASE_REASONING_META_TOKENS = RATIONALE_META_TOKENS | frozenset(
    {
        "actual",
        "against",
        "align",
        "aligns",
        "check",
        "contrastive",
        "decisive",
        "distinguish",
        "distinguishes",
        "fact",
        "facts",
        "final",
        "initial",
        "key",
        "remain",
        "remains",
        "retain",
        "retained",
        "review",
        "reviewing",
        "revise",
        "revised",
        "rereasoning",
        "separate",
        "separates",
        "strongly",
        "unchanged",
    }
)

EXPLICIT_LABEL_COMMITMENT_RE = re.compile(
    r"\b(?:the\s+|my\s+)?(?:final\s+|correct\s+|best\s+|most\s+likely\s+|provisional\s+)?"
    r"(?:answer|choice|option|diagnosis|treatment|management|cause)\s*"
    r"(?:is|would\s+be|should\s+be|must\s+be|appears\s+to\s+be|seems\s+to\s+be|=|:)\s*"
    r"option\s*\(?((?-i:[A-Z]))\)?\b",
    re.IGNORECASE,
)
BARE_LABEL_COMMITMENT_RE = re.compile(
    r"\b(?:the\s+|my\s+)?(?:final\s+|correct\s+|best\s+|most\s+likely\s+|provisional\s+)?"
    r"(?:answer|choice|option|diagnosis|treatment|management|cause)\s*"
    r"(?:is|would\s+be|should\s+be|must\s+be|appears\s+to\s+be|seems\s+to\s+be|=|:)[ \t]*"
    r"\(?((?-i:[A-Z]))\)?\b",
    re.IGNORECASE,
)
DIRECT_OPTION_COMMITMENT_RE = re.compile(
    r"\b(?:option|choice)\s*\(?((?-i:[A-Z]))\)?\s+"
    r"(?:is|appears|seems|looks)\s+(?:to\s+be\s+)?(?:the\s+)?"
    r"(?:best|preferred|most\s+likely(?:\s+(?:answer|choice|candidate|diagnosis|treatment))?|"
    r"correct\s+(?:answer|choice|option|origin)|right\s+(?:answer|choice|option))",
    re.IGNORECASE,
)
CONCLUSIVE_OPTION_CORRECT_RE = re.compile(
    r"\b(?:therefore|thus|hence|so)\s*,?\s*(?:option|choice)\s*\(?((?-i:[A-Z]))\)?\s+"
    r"(?:is|appears|seems)\s+(?:to\s+be\s+)?(?:the\s+)?(?:correct|right)\b",
    re.IGNORECASE,
)
OPTION_RELATIVE_COMMITMENT_RE = re.compile(
    r"\b(?:option|choice)\s*\(?((?-i:[A-Z]))\)?\s+"
    r"(?:is|appears|seems|looks)\s+(?:to\s+be\s+)?(?:the\s+)?"
    r"(?:most\s+(?:suspect|appropriate|plausible|consistent)|"
    r"only\s+(?:plausible|viable|consistent)(?:\s+(?:answer|choice|option|candidate))?)\b",
    re.IGNORECASE,
)
BARE_CONCLUSION_COMMITMENT_RE = re.compile(
    r"\b(?:therefore|thus|hence|so)\s*,?\s*\(?((?-i:[A-Z]))\)?\s+"
    r"(?:is|appears|seems|looks)\s+(?:to\s+be\s+)?(?:the\s+)?"
    r"(?:correct|best|preferred|most\s+(?:likely|consistent|appropriate|plausible))\b",
    re.IGNORECASE,
)
EXPLICIT_SELECTION_COMMITMENT_RE = re.compile(
    r"\b(?:choose|select|pick|favor|favour|go\s+with)\s+"
    r"(?:option|choice)\s*\(?((?-i:[A-Z]))\)?\b",
    re.IGNORECASE,
)
BARE_SELECTION_COMMITMENT_RE = re.compile(
    r"\b(?:choose|select|pick|favor|favour|go\s+with)\s+"
    r"\(?((?-i:[A-Z]))\)?\b",
    re.IGNORECASE,
)

OPTION_AS_ANSWER_COMMITMENT_RE = re.compile(
    r"\b(?:supports?|favors?|favours?)\s+(?:option|choice)\s*\(?((?-i:[A-Z]))\)?\s+"
    r"as\s+(?:the\s+|a\s+)?(?:correct|best|preferred|most\s+likely|likely)\s+"
    r"(?:answer|choice|option)\b",
    re.IGNORECASE,
)
BARE_LABEL_ONLY_OPTION_RE = re.compile(
    r"\b\(?((?-i:[A-Z]))\)?\s+is\s+(?:the\s+)?only\s+"
    r"(?:answer|choice|option|candidate)\b",
    re.IGNORECASE,
)
SUPERLATIVE_LABEL_PREFIX_RE = re.compile(
    r"\b(?:the\s+)?(?:only|best|preferred|most\s+likely|most\s+appropriate|most\s+specific)\s+"
    r"(?:option|choice)\s*\(?((?-i:[A-Z]))\)?\b",
    re.IGNORECASE,
)
OPTION_POST_SUPERLATIVE_RE = re.compile(
    r"\b(?:option|choice)\s*\(?((?-i:[A-Z]))\)?(?:\s*[:—-]|\s+is)\s*"
    r"(?:the\s+)?(?:most\s+likely|most\s+appropriate|most\s+specific|best|preferred)\b",
    re.IGNORECASE,
)
BARE_LABEL_POST_SUPERLATIVE_RE = re.compile(
    r"\b\(?((?-i:[A-Z]))\)?\s*[:—-]\s*"
    r"(?:the\s+)?(?:most\s+likely|most\s+appropriate|most\s+specific|best|preferred)\b",
    re.IGNORECASE,
)
UNIQUE_CHOICE_DOMINANCE_RE = re.compile(
    r"\b(?:only\s+one|a\s+single)\s+(?:of\s+the\s+)?(?:option|choice)s?\b"
    r"[^.;\n<]{0,100}\b(?:correct(?:ly)?|best|appropriate|valid|fits?|matches?|aligns?)\b"
    r"[^<\n]{0,260}?"
    r"\b(?:option|choice)\s*\(?((?-i:[A-Z]))\)?\s+"
    r"(?:is|appears|seems|looks)\s+(?:to\s+be\s+)?"
    r"(?:consistent|correct|valid|appropriate|the\s+one|a\s+match|aligned)\b",
    re.IGNORECASE,
)
GENERIC_PREFERENCE_RES = (
    re.compile(
        r"(?<![.;\n<])[^.;\n<]{1,100}?\s+(?:is|appears|seems)\s+(?:to\s+be\s+)?(?:the\s+)?"
        r"(?:most\s+appropriate|most\s+likely|best|preferred)\s+(?:next\s+)?"
        r"(?:test|study|step|management|intervention|treatment|diagnosis|choice|option|answer|"
        r"finding|outcome|structure|cause|mechanism|approach|modality)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:points?|pointing)\s+(?:directly\s+|strongly\s+|immediately\s+)?"
        r"(?:to|toward|towards)\s+.{1,100}?\s+as\s+(?:a|the)\s+"
        r"(?:likely|most\s+likely|best|preferred|correct)\s+(?:answer|choice|option)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\w).{1,80}?\s+is\s+(?:the\s+)?only\s+(?:answer|choice|option)\s+that\b",
        re.IGNORECASE,
    ),
)
TARGET_TEXT_COMMITMENT_RE = re.compile(
    r"\b(?:the\s+)?(?:most\s+likely|most\s+appropriate|most\s+specific|best|preferred|correct)\s+"
    r"(?:[a-z][a-z-]*\s+){0,4}"
    r"(?:finding|feature|outcome|sign|structure|condition|statement|mechanism|cause|origin|"
    r"step|intervention|dose|presentation|manifestation|pathogen|complication|association|contributor|factor|etiology|"
    r"diagnosis|treatment|management|answer|choice|option)"
    r"(?:\s+(?:in|for|among)\s+[^.;\n<]{1,60}?)?\s+"
    r"(?:is|would\s+be|appears\s+to\s+be|seems\s+to\s+be|:)\s*"
    r"(.+?)(?:[.;\n<]|$)",
    re.IGNORECASE,
)
TEXT_COMMITMENT_RE = re.compile(
    r"\b(?:the\s+)?(?:final\s+|correct\s+|best\s+|most\s+likely\s+|provisional\s+)?"
    r"(?:answer|diagnosis|treatment|management)\s*"
    r"(?:is|would\s+be|should\s+be|must\s+be|appears\s+to\s+be|seems\s+to\s+be|=|:)\s*"
    r"(.+?)(?:[.;\n<]|$)",
    re.IGNORECASE,
)
CAUSE_COMMITMENT_RE = re.compile(
    r"\b(?:(?:the|correct|best|most\s+likely|primary|underlying)\s+)+cause\s*"
    r"(?:is|would\s+be|should\s+be|must\s+be|appears\s+to\s+be|seems\s+to\s+be|=|:)\s*"
    r"(.+?)(?:[.;\n<]|$)",
    re.IGNORECASE,
)


SEMANTIC_PROVISIONAL_BOUNDARY_RES = (
    re.compile(r"</?\s*provisional[_ -]*answer\b[^>]*>", re.IGNORECASE),
    re.compile(
        r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:step[ \t]*3[.:)\-]?[ \t]*)?"
        r"provisional[ _-]+answer\b"
    ),
    re.compile(
        r"\b(?:the|my)\s+provisional\s+answer\s+"
        r"(?:is|would\s+be|should\s+be|appears\s+to\s+be|seems\s+to\s+be|:)\b",
        re.IGNORECASE,
    ),
)
CHOICE_PREFERENCE_SUFFIX_RES = (
    re.compile(
        r"\s*(?:\([A-Za-z0-9]{1,12}\))?\)?\s+(?:is|appears|seems|looks)\s+(?:to\s+be\s+)?(?:the\s+|a\s+)?"
        r"(?:correct|best|preferred|most\s+likely|likely)\s+"
        r"(?:answer|choice|option|candidate|diagnosis|treatment|management|cause|finding|feature|"
        r"outcome|sign|structure|condition|statement|mechanism|origin|step|intervention|dose|"
        r"presentation|manifestation|pathogen|complication|association|contributor|factor|etiology)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*(?:\([A-Za-z0-9]{1,12}\))?\)?\s+as\s+(?:the\s+|a\s+)?(?:correct|best|preferred|most\s+likely|likely)\s+"
        r"(?:answer|choice|option|candidate|diagnosis|treatment|management|cause|finding|feature|"
        r"outcome|sign|structure|condition|statement|mechanism|origin|step|intervention|dose|"
        r"presentation|manifestation|pathogen|complication|association|contributor|factor|etiology)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*(?:\([A-Za-z0-9]{1,12}\))?\)?\s+(?:the\s+)?(?:most\s+likely|most\s+appropriate|most\s+specific|best|preferred)\s+"
        r"(?:[a-z][a-z-]*\s+){0,2}"
        r"(?:finding|feature|outcome|sign|structure|condition|statement|mechanism|cause|origin|"
        r"step|intervention|dose|presentation|manifestation|pathogen|complication|association|contributor|factor|etiology|"
        r"diagnosis|treatment|management|answer|choice|option)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*(?:\([A-Za-z0-9]{1,12}\))?\)?\s+(?:is|remains|appears|seems)\s+(?:to\s+be\s+)?(?:the\s+)?"
        r"(?:first[- ]line\b|(?:initial|first)\s+(?:[a-z][a-z-]*\s+){0,2}"
        r"(?:test|study|modality|step|management|treatment|intervention|choice|option|approach)\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*(?:\([A-Za-z0-9]{1,12}\))?\)?\s+(?:is\s+)?(?:the\s+)?only\s+(?:plausible\s+|viable\s+)?"
        r"(?:answer|choice|option|candidate)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*(?:\([A-Z]\))?\)?\s+(?:is|appears|seems|looks)\s+(?:to\s+be\s+)?(?:the\s+)?"
        r"most\s+(?:suspect|consistent|appropriate|plausible)\b",
        re.IGNORECASE,
    ),
)
CHOICE_PREFERENCE_PREFIX_RE = re.compile(
    r"(?:the\s+)?(?:correct|best|preferred|most\s+likely|provisional)\s+"
    r"(?:answer|choice|option|diagnosis|treatment|management|cause)\s*"
    r"(?:is|should\s+be|must\s+be|appears\s+to\s+be|seems\s+to\s+be|=|:)\s*"
    r"(?:the\s+)?$",
    re.IGNORECASE,
)


def _is_conditional_hypothetical(text: str, match: re.Match[str]) -> bool:
    """Exclude counterfactual option mentions without suppressing real preferences.

    Models often contrast alternatives before the provisional-answer marker with phrases
    such as ``if X were present, the answer would be B`` or ``a finding which would
    favor A``.  The selection verb itself can therefore look identical to a commitment.
    Inspect the containing clause, not just the regex match, while preserving genuine
    first-person preferences such as ``I would choose A``.
    """
    sentence_start = (
        max(
            text.rfind(".", 0, match.start()),
            text.rfind(";", 0, match.start()),
            text.rfind("\n", 0, match.start()),
        )
        + 1
    )
    prefix = text[sentence_start : match.start()]
    evidence = match.group(0).casefold()

    # A modal inside the commitment phrase is hypothetical only when the same clause
    # contains an explicit conditional antecedent.  ``The answer would be A`` remains
    # a commitment; ``if X were present, the answer would be A`` does not.
    if ("would" in evidence or "could" in evidence) and re.search(
        r"\b(?:if|unless)\b", prefix, re.IGNORECASE
    ):
        return True

    # Bare selection matches start at ``favor/choose/...`` and therefore omit a modal
    # immediately before the verb.  Relative/case-description subjects indicate a
    # hypothetical discriminator (``which would favor A``), whereas first-person
    # ``I/we would choose A`` is an actual preference and must still count.
    modal_prefix = re.search(
        r"\b(?P<subject>which|that|this|it|finding|feature|fact|change|scenario|"
        r"presence|absence)\s+(?:would|could|might|may)\s*$",
        prefix,
        re.IGNORECASE,
    )
    return modal_prefix is not None


def _semantic_provisional_start(recovered: RecoveredProtocol) -> int | None:
    """Locate the semantic start of provisional selection without forgiving structure.

    Structural compliance still requires the exact requested start marker.  This looser
    boundary exists only for the *timing* endpoint so a closing-tag typo or Markdown
    heading after a completed facts/implications discussion is not mislabeled as an
    early answer commitment.
    """
    candidates = [
        match.start()
        for pattern in SEMANTIC_PROVISIONAL_BOUNDARY_RES
        if (match := pattern.search(recovered.document)) is not None
    ]
    if recovered.provisional_start is not None:
        candidates.append(recovered.provisional_start)
    return min(candidates) if candidates else None


def _mask_lightweight_markup(text: str) -> str:
    """Mask Markdown emphasis without changing character offsets.

    Models frequently bold an answer name (``**option text**``).  Preference detection
    should be invariant to that presentation markup, while stored evidence and section
    offsets must still refer to the original response.
    """
    return re.sub(r"[*_`~]", " ", text)


def _interrogative_prompt_context(text: str, start: int, end: int) -> bool:
    """Return True when a preference-looking phrase is only restating a question.

    The timing endpoint is intended to detect a *declared preference*, not a copied
    prompt such as ``Question: what is the most appropriate next step?``.  Work
    clause-locally so a genuine answer in the following sentence is unaffected.
    """
    left = max(text.rfind("\n", 0, start), text.rfind(".", 0, start), text.rfind(";", 0, start)) + 1
    stops = [
        pos
        for pos in (
            text.find("\n", end),
            text.find(".", end),
            text.find(";", end),
            text.find("?", end),
        )
        if pos >= 0
    ]
    right = min(stops) + 1 if stops else min(len(text), end + 220)
    clause = text[left:right]
    folded = clause.casefold().strip()
    if "?" in clause:
        return True
    return bool(re.match(r"^(?:[-*#>\s]*)(?:key\s+)?question\s*[:—-]", folded))


def _hedged_choice_continuation(candidate: str, choice: str) -> bool:
    """Reject a choice mention that is explicitly framed as only a possibility.

    ``Diagnosis: Prerenal azotemia is possible, but ...`` is differential reasoning,
    not a firm answer commitment.  Strong superlative/selection wording is captured
    by other commitment patterns and remains eligible.
    """
    if not candidate.startswith(choice):
        return False
    remainder = candidate[len(choice) :].strip()
    return bool(
        re.match(
            r"^(?:is\s+)?(?:possible|plausible|a\s+possibility|a\s+consideration|less\s+likely|unlikely)\b|"
            r"^(?:could|may|might|can)\s+(?:be\s+)?(?:possible|plausible|considered|the\s+cause|a\s+cause)?\b|"
            r"^should\s+be\s+considered\b|^cannot\s+be\s+excluded\b",
            remainder,
            re.IGNORECASE,
        )
    )


def _choice_text_commitments(
    text: str,
    choices: Mapping[str, str],
    *,
    evidence_text: str | None = None,
) -> list[CommitmentEvent]:
    """Detect explicit preference phrasing where the choice text precedes the cue."""
    events: list[CommitmentEvent] = []
    source = evidence_text if evidence_text is not None else text
    for label, choice in choices.items():
        choice_text = choice.strip()
        if not choice_text:
            continue
        for occurrence in re.finditer(re.escape(choice_text), text, re.IGNORECASE):
            before = text[max(0, occurrence.start() - 140) : occurrence.start()]
            after = text[occurrence.end() : min(len(text), occurrence.end() + 140)]
            prefix = CHOICE_PREFERENCE_PREFIX_RE.search(before)
            suffix = next(
                (
                    pattern.match(after)
                    for pattern in CHOICE_PREFERENCE_SUFFIX_RES
                    if pattern.match(after)
                ),
                None,
            )
            if prefix is None and suffix is None:
                continue
            # Do not turn an explicit conditional alternative into the current answer.
            sentence_start = (
                max(
                    text.rfind(".", 0, occurrence.start()),
                    text.rfind(";", 0, occurrence.start()),
                    text.rfind("\n", 0, occurrence.start()),
                )
                + 1
            )
            sentence_prefix = text[sentence_start : occurrence.start()]
            suffix_text = suffix.group(0) if suffix is not None else ""
            if re.search(r"\b(?:if|unless)\b", sentence_prefix, re.IGNORECASE) and re.search(
                r"\b(?:would|could)\b", before[-80:] + suffix_text, re.IGNORECASE
            ):
                continue
            start = (
                prefix.start() + max(0, occurrence.start() - 140)
                if prefix is not None
                else occurrence.start()
            )
            end = occurrence.end() + (suffix.end() if suffix is not None else 0)
            events.append(
                {
                    "option": label,
                    "start": start,
                    "evidence": source[start:end].strip(),
                }
            )
    return events


def _commitments(text: str, choices: Mapping[str, str]) -> list[CommitmentEvent]:
    events: list[CommitmentEvent] = []
    scan_text = _mask_lightweight_markup(text)
    for pattern in (
        EXPLICIT_LABEL_COMMITMENT_RE,
        BARE_LABEL_COMMITMENT_RE,
        DIRECT_OPTION_COMMITMENT_RE,
        CONCLUSIVE_OPTION_CORRECT_RE,
        OPTION_RELATIVE_COMMITMENT_RE,
        BARE_CONCLUSION_COMMITMENT_RE,
        EXPLICIT_SELECTION_COMMITMENT_RE,
        BARE_SELECTION_COMMITMENT_RE,
        OPTION_AS_ANSWER_COMMITMENT_RE,
        BARE_LABEL_ONLY_OPTION_RE,
        SUPERLATIVE_LABEL_PREFIX_RE,
        OPTION_POST_SUPERLATIVE_RE,
        BARE_LABEL_POST_SUPERLATIVE_RE,
        UNIQUE_CHOICE_DOMINANCE_RE,
    ):
        for match in pattern.finditer(scan_text):
            if _is_conditional_hypothetical(text, match):
                continue
            if _interrogative_prompt_context(text, match.start(), match.end()):
                continue
            label = match.group(1).upper()
            if label in choices:
                events.append(
                    {
                        "option": label,
                        "start": match.start(),
                        "evidence": text[match.start() : match.end()],
                    }
                )
    normalized_choices = {label: normalize_text(value) for label, value in choices.items()}
    for pattern in (TEXT_COMMITMENT_RE, CAUSE_COMMITMENT_RE, TARGET_TEXT_COMMITMENT_RE):
        for match in pattern.finditer(scan_text):
            if _is_conditional_hypothetical(text, match):
                continue
            if _interrogative_prompt_context(text, match.start(), match.end()):
                continue
            evidence_prefix = normalize_text(
                match.group(0)[: max(0, match.group(0).find(match.group(1)))]
            )
            nearby_prefix = text[max(0, match.start() - 220) : match.start()]
            if re.fullmatch(
                r"(?:diagnosis|treatment|management|cause)", evidence_prefix
            ) and re.search(
                r"\b(?:differential(?:\s+diagnosis)?|answer\s+choices?|choices?\s+(?:include|represent)|"
                r"options?\s+(?:include|listed|above)|treatment\s+options?|alternative(?:s|\s+diagnoses)?)\b",
                nearby_prefix,
                re.IGNORECASE,
            ):
                continue
            candidate = normalize_text(match.group(1))
            candidate_without_article = re.sub(r"^(?:the|a|an)\s+", "", candidate)
            hits = [
                label
                for label, choice in normalized_choices.items()
                if choice
                and not _hedged_choice_continuation(candidate, choice)
                and not _hedged_choice_continuation(candidate_without_article, choice)
                and (
                    candidate == choice
                    or candidate.startswith(choice + " ")
                    or candidate_without_article == choice
                    or candidate_without_article.startswith(choice + " ")
                )
            ]
            if len(hits) == 1:
                events.append(
                    {
                        "option": hits[0],
                        "start": match.start(),
                        "evidence": text[match.start() : match.end()],
                    }
                )
    events.extend(_choice_text_commitments(scan_text, choices, evidence_text=text))
    for pattern in GENERIC_PREFERENCE_RES:
        for match in pattern.finditer(scan_text):
            if _is_conditional_hypothetical(text, match):
                continue
            if _interrogative_prompt_context(text, match.start(), match.end()):
                continue
            if re.search(
                r"\b(?:only\s+)?one\s+(?:is|appears|seems)\s+(?:to\s+be\s+)?(?:the\s+)?"
                r"(?:best|preferred|most\s+likely|most\s+appropriate)\b",
                match.group(0),
                re.IGNORECASE,
            ) or re.search(
                r"\b(?:the|an?)\s+(?:option|choice|answer)\s+that\b.{0,70}\b"
                r"(?:is|would\s+be)\s+(?:the\s+)?(?:best|preferred|most\s+likely|most\s+appropriate)\b",
                match.group(0),
                re.IGNORECASE,
            ):
                # A generic decision rule ("the option that matches X is best") does
                # not identify a preferred answer.  Do not infer identity from prior
                # medical reasoning; this timing endpoint deliberately stays mechanical.
                continue
            # An unresolved preference only counts when the prose explicitly says it
            # is an answer/choice/option.  ``Alcohol is the most likely cause`` in a
            # question whose actual choices are anatomical locations is clinical
            # reasoning, not early commitment to the multiple-choice answer.  Exact
            # choice-text preferences are recovered separately above.
            if not re.search(r"\b(?:answer|choice|option)\b", match.group(0), re.IGNORECASE):
                continue
            events.append(
                {
                    "option": None,
                    "start": match.start(),
                    "evidence": text[match.start() : match.end()].strip(),
                }
            )
    events.sort(key=lambda event: (int(event["start"]), str(event["option"])))
    deduplicated: list[CommitmentEvent] = []
    seen: set[tuple[int, str]] = set()
    for event in events:
        key = (int(event["start"]), str(event["option"]))
        if key not in seen:
            deduplicated.append(event)
            seen.add(key)
    return deduplicated


def _answer_from_recovered(
    recovered: RecoveredProtocol,
    choices: Mapping[str, str],
) -> ExtractedAnswer:
    """Resolve answer identity from the format-neutral final-answer role."""
    count = recovered.semantic_start_counts.get("final_answer", 0)
    if count == 0:
        return ExtractedAnswer(
            None,
            "missing_final_answer",
            recovered.sections.get("final_answer") or None,
            (),
        )
    option = recovered.semantic_final_option
    if option is None or option not in choices:
        return ExtractedAnswer(None, "invalid_label", None, ())

    body = recovered.sections["final_answer"]
    if body:
        body_answer = extract_answer(body, choices)
        if body_answer.option is not None and body_answer.option != option:
            return ExtractedAnswer(
                None,
                "conflicting",
                body,
                tuple(sorted({option, body_answer.option})),
            )
    return ExtractedAnswer(option, "parsed", body or None, (option,))


def extract_protocol_final_answer(raw: str, choices: Mapping[str, str]) -> ExtractedAnswer:
    """Extract answer identity from the clearly labeled final-answer role.

    The option may be expressed by label or unambiguous answer text. A body that
    clearly names a different option is treated as a genuine conflict; presentation
    syntax and punctuation do not affect accuracy.
    """
    return _answer_from_recovered(recover_protocol(raw, choices), choices)


def _token_count(text: str) -> int:
    return len(content_tokens(text))


def _informative(text: str, minimum_tokens: int) -> bool:
    """Reject empty, extremely short, and mechanically repetitive section filler."""
    tokens = content_tokens(text)
    if len(tokens) < minimum_tokens:
        return False
    counts = Counter(tokens)
    if len(counts) < min(4, minimum_tokens):
        return False
    most_common = max(counts.values())
    return most_common / len(tokens) <= 0.5


def _choice_residual_tokens(
    text: str,
    option: str | None,
    choices: Mapping[str, str],
) -> list[str]:
    """Return content tokens not explained by merely naming the selected choice."""
    tokens = list(content_tokens(text))
    if option is None or option not in choices:
        return tokens
    remaining_choice = content_tokens(choices[option])
    residual = tokens.copy()
    for token in remaining_choice:
        try:
            residual.remove(token)
        except ValueError:
            continue
    return residual


def _rationale_present(
    question: Question,
    text: str,
    option: str | None,
) -> bool:
    """Require explanatory material beyond an answer name or protocol boilerplate.

    A rationale can be useful without literally repeating a stem token or using a
    fixed connective such as ``because``.  Choice text is first subtracted so that
    long answer names cannot satisfy this role by themselves.  Short residuals must
    then either reconnect to the stem or contain an explicit rationale cue; a
    cue-free explanation is accepted only when it contains at least four distinct
    content terms outside generic answer-selection vocabulary.
    """
    residual = _choice_residual_tokens(text, option, question.choices)
    distinct = set(residual)
    if len(residual) < 2 or len(distinct) < 2:
        return False
    stem_terms = set(content_tokens(question.stem))
    if stem_terms.intersection(residual) or RATIONALE_CUES.search(text):
        return True
    specific = {token for token in distinct if token not in RATIONALE_META_TOKENS}
    return len(specific) >= 4


def _case_grounded_explanation(
    question: Question,
    text: str,
    options: tuple[str | None, ...],
    *,
    trajectory_references: tuple[str, ...] = (),
) -> bool:
    """Accept case-specific reasoning without requiring a literal stem repetition.

    Literal stem overlap remains the preferred signal. A later reasoning section may
    also paraphrase material already made explicit in the facts/implications trajectory,
    so non-boilerplate lexical continuity with those sections is accepted. Conceptual
    multiple-choice questions can instead be distinguished mainly through their answer
    choices; in that case the section must mention relevant choice material *and* add
    at least three non-boilerplate terms beyond merely restating choice names. These are
    specificity checks only and do not judge medical truth.
    """
    relevant = tuple(
        option for option in options if option is not None and option in question.choices
    )
    tokens = content_tokens(text)
    choice_tokens: list[str] = []
    for option in relevant:
        choice_tokens.extend(content_tokens(question.choices[option]))

    residual = list(tokens)
    for token in choice_tokens:
        try:
            residual.remove(token)
        except ValueError:
            continue
    residual_specific = {token for token in residual if token not in CASE_REASONING_META_TOKENS}
    if set(tokens).intersection(choice_tokens) and not residual_specific:
        return False

    # One shared generic token is too weak to establish case grounding. The
    # threshold is deliberately low enough for short stems while requiring a
    # visible connection beyond an incidental word match.
    if score_anchor_recall(question.stem, text) >= 0.08:
        return True

    text_specific = {token for token in tokens if token not in CASE_REASONING_META_TOKENS}
    for reference in trajectory_references:
        reference_specific = {
            token for token in content_tokens(reference) if token not in CASE_REASONING_META_TOKENS
        }
        if len(text_specific.intersection(reference_specific)) >= 2:
            return True

    if not relevant or not set(tokens).intersection(choice_tokens):
        return False
    return len(residual_specific) >= 3


def _explicit_counterfactual_winners(
    text: str,
    choices: Mapping[str, str],
) -> set[str]:
    """Return choices explicitly said to become the winning answer."""
    winners: set[str] = set()
    for pattern in COUNTERFACTUAL_TARGET_RES:
        for match in pattern.finditer(text):
            # Do not turn ``C would not be the best answer`` into a winner.
            if re.search(r"\b(?:not|never)\b", match.group(0), re.IGNORECASE):
                continue
            for group in match.groups():
                if group is not None and group.upper() in choices:
                    winners.add(group.upper())
    winning_suffix = re.compile(
        r"[^.;\n]{0,100}\b(?:would|could|might)\b[^.;\n]{0,70}\b"
        r"(?:best|correct|most\s+likely|more\s+likely|most\s+appropriate|more\s+appropriate|preferred|more\s+suitable|most\s+suitable|better\s+"
        r"(?:answer|choice|explanation|fit)|more\s+likely\s+(?:answer|diagnosis|choice))\b",
        re.IGNORECASE,
    )
    for option, choice in choices.items():
        if not choice.strip():
            continue
        for occurrence in re.finditer(re.escape(choice), text, re.IGNORECASE):
            after = text[occurrence.end() :]
            before = text[max(0, occurrence.start() - 80) : occurrence.start()]
            if winning_suffix.match(after):
                winners.add(option)
                break
            # Natural constructions often put the modal before the choice name:
            # ``would make striae albicantes the best answer``.
            if re.search(
                r"\b(?:would|could|might)\s+(?:then\s+)?make\b[^.;\n]{0,45}$", before, re.IGNORECASE
            ) and re.match(
                r"[^.;\n]{0,45}\b(?:the\s+)?(?:best|correct|most\s+likely|more\s+likely|most\s+appropriate|more\s+appropriate|preferred|better)\b",
                after,
                re.IGNORECASE,
            ):
                winners.add(option)
                break
    return winners


HUMAN_VALIDATED_COUNTERFACTUAL_JUDGE = "v1"
PAPER_VALIDATED_ROLE_AGGREGATE = "v1"

COUNTERFACTUAL_SCENARIO_SPLIT_RE = re.compile(
    # Do not split on the period in a compact option label such as "F. Release".
    r"(?<![A-Z]\.)(?<=[.!?])\s+|\n{2,}"
)
COUNTERFACTUAL_AMBIGUOUS_OR_RE = re.compile(
    r"\b(?-i:([A-Z]))\.\s*[^;\n]{0,120}\bor\s+"
    r"(?-i:([A-Z]))\.\s*[^;\n]{0,120}"
    r"\b(?:would|could|might)\b[^;\n]{0,80}"
    r"\b(?:best|preferred|most\s+likely|more\s+likely)\b",
    re.IGNORECASE,
)


def _counterfactual_scenarios(text: str) -> tuple[str, ...]:
    """Split alternative hypothetical branches without splitting option labels."""
    scenarios = tuple(
        part.strip() for part in COUNTERFACTUAL_SCENARIO_SPLIT_RE.split(text) if part.strip()
    )
    return scenarios or (text.strip(),)


def _normalized_choice_aliases(choice: str) -> tuple[str, ...]:
    """Return conservative aliases for detecting a named alternative in prose.

    The only normalization beyond the existing answer normalizer is removal of a
    leading determiner/possessive, allowing e.g. choice text ``His father`` to match
    ``father``.  We do not invent abbreviations or semantic synonyms here.
    """
    normalized = normalize_text(choice).casefold().strip()
    aliases: set[str] = {normalized} if normalized else set()
    tokens = normalized.split()
    while tokens and tokens[0] in {"the", "a", "an", "his", "her", "their"}:
        tokens = tokens[1:]
    if tokens:
        aliases.add(" ".join(tokens))
    return tuple(sorted((alias for alias in aliases if len(alias) >= 3), key=len, reverse=True))


def _natural_second_best_target(
    scenario: str,
    choices: Mapping[str, str],
    second_best_option: str,
) -> bool:
    """Recognize natural winner language missed by the strict option regexes."""
    normalized = normalize_text(scenario).casefold()
    aliases = _normalized_choice_aliases(choices[second_best_option])
    for alias in aliases:
        escaped = re.escape(alias)
        patterns = (
            # "would strongly suggest Crohn's disease over ulcerative colitis"
            rf"\b(?:would|could|might)\b.{{0,55}}\b"
            rf"(?:suggest|favor|favour|support|point\s+to)\s+(?:the\s+)?{escaped}\b",
            # "previous attempt would become the most significant risk factor"
            rf"\b{escaped}\b.{{0,110}}\b(?:would|could|might)\b.{{0,65}}\b"
            rf"(?:become|be|represent)\b.{{0,30}}\b"
            rf"(?:best|preferred|better|most\s+likely|more\s+likely|"
            rf"most\s+significant|strongest)\b",
            # "shift the diagnosis/preference towards Crohn's disease"
            rf"\b(?:shift|change|switch)\b.{{0,40}}\b"
            rf"(?:diagnosis|answer|preference)\b.{{0,20}}\b"
            rf"(?:to|toward|towards)\s+(?:the\s+)?{escaped}\b",
        )
        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns):
            return True

    # Safe anaphora: this role is evaluated together with a separately recovered,
    # declared best alternative.  Saying the "best alternative" would become/favor
    # the winner therefore identifies that recovered option without guessing medicine.
    return bool(
        re.search(
            r"\b(?:make|making)\s+(?:the\s+)?best\s+alternative\b.{0,100}\b"
            r"(?:best|preferred|correct|better|most\s+likely)\b",
            normalized,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:change|shift|switch)\s+(?:the\s+)?answer\s+to\s+"
            r"(?:favor|favour)\s+(?:the\s+)?best\s+alternative\b",
            normalized,
            re.IGNORECASE,
        )
        or re.search(
            r"\bbest\s+alternative\b.{0,90}\b(?:would|could|might)\b.{0,65}\b"
            r"(?:become|be)\b.{0,30}\b(?:best|preferred|better|most\s+likely)\b",
            normalized,
            re.IGNORECASE,
        )
    )


def _scenario_has_ambiguous_joint_winner(
    scenario: str,
    choices: Mapping[str, str],
    second_best_option: str,
) -> bool:
    """Reject a single hypothetical branch that offers two co-winners.

    Separate later branches are allowed.  For example, a response can first give a
    valid change that makes its stated alternative win and then discuss another
    hypothetical.  What fails is ``F or H would become the best answer`` inside the
    same proposed change.
    """
    for match in COUNTERFACTUAL_AMBIGUOUS_OR_RE.finditer(scenario):
        left, right = (group.upper() for group in match.groups())
        if (
            left in choices
            and right in choices
            and left != right
            and second_best_option in {left, right}
        ):
            return True
    return False


def _second_best_has_winning_scenario(
    text: str,
    choices: Mapping[str, str],
    second_best_option: str,
) -> bool:
    """Require at least one unambiguous hypothetical where the declared alternative wins."""
    for scenario in _counterfactual_scenarios(text):
        if _scenario_has_ambiguous_joint_winner(scenario, choices, second_best_option):
            continue
        winners = _explicit_counterfactual_winners(scenario, choices)
        other_winners = winners - {second_best_option}
        if second_best_option in winners and not other_winners:
            return True
        # If a strict regex found a *different* winner in this same branch, do not
        # override it with looser anaphoric/choice-text matching.
        if other_winners:
            continue
        if _natural_second_best_target(scenario, choices, second_best_option):
            return True
    return False


def _case_specific_counterfactual(
    question: Question,
    text: str,
    second_best_option: str | None,
) -> bool:
    """Judge the requested answer-changing counterfactual as an observable operation.

    PASS requires all of the following, without judging medical truth:
    1. a substantive hypothetical rather than boilerplate;
    2. a change to case content, not a rewrite of the task or answer choices;
    3. a valid, separately declared best alternative;
    4. at least one unambiguous hypothetical branch in which that exact alternative
       is said to become/furnish the winning answer; and
    5. concrete content beyond merely naming the alternative.

    This definition follows the atomic human-validation instruction.  In particular,
    ``X or Y would become the best answer`` does not complete the role when X is the
    declared alternative, while natural wording such as ``skip lesions would strongly
    suggest Crohn's disease over ulcerative colitis`` does complete it.
    """
    if not (_informative(text, 7) and COUNTERFACTUAL_CUES.search(text)):
        return False
    if COUNTERFACTUAL_TASK_REWRITE_RE.search(text):
        return False
    if second_best_option is None or second_best_option not in question.choices:
        return False
    if not _second_best_has_winning_scenario(text, question.choices, second_best_option):
        return False

    specific = {token for token in content_tokens(text) if token not in COUNTERFACTUAL_META_TOKENS}
    choice_terms = set(content_tokens(question.choices[second_best_option]))
    introduced = specific - choice_terms
    if len(introduced) < 2 and not (
        set(content_tokens(text)).intersection(choice_terms) and RATIONALE_CUES.search(text)
    ):
        return False

    # Keep the construct case-grounded.  New hypothetical findings are allowed and
    # often absent from the original stem, so strong literal overlap is not required;
    # the introduced-content test above supplies the complementary guard.
    return bool(
        score_anchor_recall(question.stem, text) >= 0.05
        or specific.intersection(choice_terms)
        or len(introduced) >= 3
    )


def _failure_judgment(instructed: bool, status: str, inventory: QuestionFacts) -> Judgment:
    eligible = inventory.applicable
    adverse = 1.0 if instructed else None
    all_question_zero = 0.0 if instructed else None
    zero = 0.0 if instructed and eligible else None
    empty_counts: dict[str, int] = dict.fromkeys(CONTENT_TAGS, 0) if instructed else {}
    empty_flags: dict[str, bool] = dict.fromkeys(CONTENT_TAGS, False) if instructed else {}
    return {
        "judge_version": JUDGE_VERSION,
        "generation_usable": False,
        "generation_status": status,
        "repair_endpoint_eligible": eligible,
        "repair_endpoint_inapplicability_reason": inventory.inapplicability_reason,
        "essential_fact_count": len(inventory.facts),
        "fact_complexity": inventory.complexity,
        "fact_inventory_review_status": "automatic_unreviewed",
        "protocol_scaffold_complete": all_question_zero,
        "preanswer_discussion_complete": all_question_zero,
        "visible_reasoning_scaffold_complete": all_question_zero,
        "visible_preanswer_discussion_complete": all_question_zero,
        "minimum_repair_scaffold": zero,
        "contrastive_scaffold_complete": zero,
        "correct_with_minimum_repair_scaffold": zero,
        "protocol_format_score": all_question_zero,
        "structure_score": all_question_zero,
        "section_depth_score": all_question_zero,
        "substantive_role_score": zero,
        "contrastive_role_score": zero,
        "visible_substantive_role_score": all_question_zero,
        "visible_contrastive_role_score": all_question_zero,
        "identified_role_count": 0 if instructed else None,
        "unique_role_count": 0 if instructed else None,
        "required_section_count": 0 if instructed else None,
        "nontrivial_section_count": 0 if instructed else None,
        "substantive_role_count": 0 if instructed else None,
        "validated_role_count": 0 if instructed else None,
        "validated_role_completeness_score": all_question_zero,
        "contrastive_discussion_count": 0 if instructed else None,
        "contrastive_discussion_score": all_question_zero,
        "validated_contrastive_discussion_count": 0 if instructed else None,
        "validated_contrastive_discussion_score": all_question_zero,
        "provisional_answer_discussed": all_question_zero,
        "best_alternative_discussed": all_question_zero,
        "decisive_distinction_discussed": all_question_zero,
        "answer_change_discussed": all_question_zero,
        "reconsideration_discussed": all_question_zero,
        "role_completeness_score": all_question_zero,
        "all_roles_substantive": all_question_zero,
        "all_roles_validated_complete": all_question_zero,
        "validated_complete_role_scaffold": all_question_zero,
        "roles_in_requested_order": all_question_zero,
        "complete_role_scaffold": all_question_zero,
        "role_facts_complete": all_question_zero,
        "role_implications_complete": all_question_zero,
        "role_provisional_answer_complete": all_question_zero,
        "role_best_alternative_complete": all_question_zero,
        "role_decisive_distinction_complete": all_question_zero,
        "role_answer_changing_change_complete": all_question_zero,
        "role_answer_changing_change_nontrivial": all_question_zero,
        "role_reconsideration_complete": all_question_zero,
        "role_final_answer_complete": all_question_zero,
        "facts_tfidf_recall": zero,
        "implications_tfidf_recall": zero,
        "preanswer_tfidf_recall": zero,
        "preprovisional_tfidf_recall": zero,
        "preanswer_high_idf_tfidf_recall": zero,
        "facts_implications_shared_tfidf_recall": zero,
        "facts_implications_shared_high_idf_recall": zero,
        "preanswer_tfidf_density_per_100_tokens": zero,
        "preanswer_tfidf_token_count": zero,
        "facts_anchor_recall": zero,
        "implications_anchor_recall": zero,
        "preanswer_anchor_recall": zero,
        "shared_anchor_recall": zero,
        "atomic_fact_coverage": zero,
        "atomic_implication_trace_coverage": zero,
        "hard_qualifier_fact_recall": zero,
        "hard_qualifier_trace_recall": zero,
        "fact_coverage_details": [],
        "preprovisional_commitment_observed": None,
        "preprovisional_commitment_itt": adverse,
        "trajectory_complete": False if instructed else None,
        "semantic_trajectory_complete": False if instructed else None,
        "accuracy": 0,
        "accuracy_parseable": False,
        "final_option": None,
        "answer_extraction_status": "generation_failure",
        "answer_candidates": [],
        "trajectory_errors": [f"generation status: {status}"],
        "trajectory_criteria": {},
        "section_substantive": dict(empty_flags),
        "section_token_counts": dict(empty_counts),
        "section_marker_counts": dict(empty_counts),
        "section_present": dict(empty_flags),
        "section_unique": dict(empty_flags),
        "section_nontrivial": dict(empty_flags),
        "section_schema_valid": dict(empty_flags),
        "section_present_count": 0 if instructed else None,
        "section_unique_count": 0 if instructed else None,
        "section_nontrivial_count": 0 if instructed else None,
        "section_substantive_count": 0 if instructed else None,
        "section_schema_valid_count": 0 if instructed else None,
        "section_present_fraction": all_question_zero,
        "section_unique_fraction": all_question_zero,
        "section_nontrivial_fraction": all_question_zero,
        "section_substantive_fraction": all_question_zero,
        "section_schema_valid_fraction": all_question_zero,
        "facts_choice_list_leakage": None,
        "provisional_option": None,
        "second_best_option": None,
        "rereasoning_decision": None,
        "decision_consistent": False,
        "strict_decision_consistent": False,
        "early_commitments": [],
    }


def _baseline_judgment(
    question: Question,
    status: str,
    raw: str,
    inventory: QuestionFacts,
) -> Judgment:
    answer = extract_answer(raw, question.choices)
    parseable = answer.status == "parsed" and answer.option is not None
    return {
        "judge_version": JUDGE_VERSION,
        "generation_usable": True,
        "generation_status": status,
        "repair_endpoint_eligible": inventory.applicable,
        "repair_endpoint_inapplicability_reason": inventory.inapplicability_reason,
        "essential_fact_count": len(inventory.facts),
        "fact_complexity": inventory.complexity,
        "fact_inventory_review_status": "automatic_unreviewed",
        "protocol_scaffold_complete": None,
        "preanswer_discussion_complete": None,
        "visible_reasoning_scaffold_complete": None,
        "visible_preanswer_discussion_complete": None,
        "minimum_repair_scaffold": None,
        "contrastive_scaffold_complete": None,
        "correct_with_minimum_repair_scaffold": None,
        "protocol_format_score": None,
        "structure_score": None,
        "section_depth_score": None,
        "substantive_role_score": None,
        "contrastive_role_score": None,
        "visible_substantive_role_score": None,
        "visible_contrastive_role_score": None,
        "identified_role_count": None,
        "unique_role_count": None,
        "required_section_count": None,
        "nontrivial_section_count": None,
        "substantive_role_count": None,
        "validated_role_count": None,
        "validated_role_completeness_score": None,
        "contrastive_discussion_count": None,
        "contrastive_discussion_score": None,
        "validated_contrastive_discussion_count": None,
        "validated_contrastive_discussion_score": None,
        "provisional_answer_discussed": None,
        "best_alternative_discussed": None,
        "decisive_distinction_discussed": None,
        "answer_change_discussed": None,
        "reconsideration_discussed": None,
        "role_completeness_score": None,
        "all_roles_substantive": None,
        "all_roles_validated_complete": None,
        "validated_complete_role_scaffold": None,
        "roles_in_requested_order": None,
        "complete_role_scaffold": None,
        "role_facts_complete": None,
        "role_implications_complete": None,
        "role_provisional_answer_complete": None,
        "role_best_alternative_complete": None,
        "role_decisive_distinction_complete": None,
        "role_answer_changing_change_complete": None,
        "role_answer_changing_change_nontrivial": None,
        "role_reconsideration_complete": None,
        "role_final_answer_complete": None,
        "facts_tfidf_recall": None,
        "implications_tfidf_recall": None,
        "preanswer_tfidf_recall": None,
        "preprovisional_tfidf_recall": None,
        "preanswer_high_idf_tfidf_recall": None,
        "facts_implications_shared_tfidf_recall": None,
        "facts_implications_shared_high_idf_recall": None,
        "preanswer_tfidf_density_per_100_tokens": None,
        "preanswer_tfidf_token_count": None,
        "facts_anchor_recall": None,
        "implications_anchor_recall": None,
        "preanswer_anchor_recall": None,
        "shared_anchor_recall": None,
        "atomic_fact_coverage": None,
        "atomic_implication_trace_coverage": None,
        "hard_qualifier_fact_recall": None,
        "hard_qualifier_trace_recall": None,
        "fact_coverage_details": [],
        "preprovisional_commitment_observed": None,
        "preprovisional_commitment_itt": None,
        "trajectory_complete": None,
        "semantic_trajectory_complete": None,
        "accuracy": int(bool(parseable and answer.option == question.gold)),
        "accuracy_parseable": parseable,
        "final_option": answer.option,
        "answer_extraction_status": answer.status,
        "answer_candidates": list(answer.candidates),
        "trajectory_errors": [],
        "trajectory_criteria": {},
        "section_substantive": {},
        "section_token_counts": {},
        "section_marker_counts": {},
        "section_present": {},
        "section_unique": {},
        "section_nontrivial": {},
        "section_schema_valid": {},
        "section_present_count": None,
        "section_unique_count": None,
        "section_nontrivial_count": None,
        "section_substantive_count": None,
        "section_schema_valid_count": None,
        "section_present_fraction": None,
        "section_unique_fraction": None,
        "section_nontrivial_fraction": None,
        "section_substantive_fraction": None,
        "section_schema_valid_fraction": None,
        "facts_choice_list_leakage": None,
        "provisional_option": None,
        "second_best_option": None,
        "rereasoning_decision": None,
        "decision_consistent": False,
        "strict_decision_consistent": False,
        "early_commitments": [],
    }


def _section_counts(recovered: RecoveredProtocol) -> dict[str, int]:
    sections = recovered.sections
    return {
        "facts": _token_count(sections["facts"]),
        "implications": _token_count(sections["implications"]),
        "provisional_answer": _token_count(sections["provisional_answer"]),
        "second_best": _token_count(sections["second_best"]),
        "decisive_fact": _token_count(sections["decisive_fact"]),
        "answer_changing_change": _token_count(sections["answer_changing_change"]),
        "rereasoning": _token_count(sections["rereasoning"]),
        "final_answer": _token_count(sections["final_answer"]),
    }


def _strict_decision_consistent(
    recovered: RecoveredProtocol,
    choices: Mapping[str, str],
) -> bool:
    return bool(
        recovered.decision in VALID_DECISIONS
        and recovered.provisional_option in choices
        and recovered.final_option in choices
        and (
            (
                recovered.decision == "retain"
                and recovered.final_option == recovered.provisional_option
            )
            or (
                recovered.decision == "revise"
                and recovered.final_option != recovered.provisional_option
            )
        )
    )


def _semantic_decision_consistent(
    recovered: RecoveredProtocol,
    choices: Mapping[str, str],
) -> bool:
    return bool(
        recovered.semantic_decision in VALID_DECISIONS
        and recovered.semantic_provisional_option in choices
        and recovered.semantic_final_option in choices
        and (
            (
                recovered.semantic_decision == "retain"
                and recovered.semantic_final_option == recovered.semantic_provisional_option
            )
            or (
                recovered.semantic_decision == "revise"
                and recovered.semantic_final_option != recovered.semantic_provisional_option
            )
        )
    )


def _facts_without_choice_list(text: str, choices: Mapping[str, str]) -> tuple[str, bool]:
    """Remove an explicit multi-option list while preserving preceding stem discussion."""
    matches = [
        match for match in CHOICE_LIST_MARKER_RE.finditer(text) if match.group(1).upper() in choices
    ]
    if len({match.group(1).upper() for match in matches}) < 2:
        return text, False
    return text[: matches[0].start()].rstrip(), True


def _preanswer_facts_discussed(
    question: Question,
    text: str,
) -> bool:
    """Score substantive engagement before selection without enforcing role purity.

    Full protocol compliance separately penalizes a Facts role that turns into an
    explicit choice list.  For the narrower timing construct, a fact-poor/topic-only
    MCQ can still show genuine pre-answer engagement by explaining the supplied topic
    and/or choices before selecting anything.  Require nontrivial content anchored to
    the actual stem, but leave answer preference to the commitment detector.
    """
    return bool(_informative(text, 4) and score_anchor_recall(question.stem, text) > 0)


DECISIVE_CONTRAST_CUES = re.compile(
    r"\b(?:versus|vs\.?|rather\s+than|whereas|while|unlike|compared\s+(?:with|to)|"
    r"distinguish(?:es|ed|ing)?|distinction|difference|differ(?:s|ed|ent)?|"
    r"more\s+likely\s+than|less\s+likely\s+than|favou?rs?\s+.+?\s+over)\b",
    re.IGNORECASE,
)


def _decisive_discussion_present(
    question: Question,
    text: str,
    provisional_option: str | None,
    second_best_option: str | None,
    *,
    trajectory_references: tuple[str, ...],
) -> bool:
    """Recognize a genuine attempt to articulate the decisive distinction.

    This endpoint measures whether the model actually performs the contrastive
    operation, not whether every medical statement is true.  Literal stem/choice
    grounding remains sufficient, but natural paraphrases such as ``imaging versus
    conservative management`` should not fail solely because they do not repeat the
    benchmark choice strings verbatim.
    """
    if not _informative(text, 3):
        return False
    if _case_grounded_explanation(
        question,
        text,
        (provisional_option, second_best_option),
        trajectory_references=trajectory_references,
    ):
        return True
    return bool(DECISIVE_CONTRAST_CUES.search(text) and _informative(text, 6))


def _substantive_sections(
    question: Question,
    recovered: RecoveredProtocol,
    lexical: Mapping[str, float],
    decision_consistent: bool,
    facts_choice_list_leakage: bool,
    answer: ExtractedAnswer,
) -> dict[str, bool]:
    sections = recovered.sections
    return {
        "facts": bool(
            _informative(sections["facts"], 4)
            and lexical["facts_anchor_recall"] > 0
            and not facts_choice_list_leakage
        ),
        "implications": bool(
            _informative(sections["implications"], 6)
            and _case_grounded_explanation(
                question,
                sections["implications"],
                tuple(question.choices),
                trajectory_references=(sections["facts"],),
            )
        ),
        "provisional_answer": bool(
            recovered.semantic_provisional_option in question.choices
            and _informative(sections["provisional_answer"], 4)
            and _rationale_present(
                question,
                sections["provisional_answer"],
                recovered.semantic_provisional_option,
            )
        ),
        "second_best": bool(
            recovered.semantic_second_best_option in question.choices
            and recovered.semantic_second_best_option != recovered.semantic_provisional_option
            and _informative(sections["second_best"], 5)
            and _rationale_present(
                question,
                sections["second_best"],
                recovered.semantic_second_best_option,
            )
        ),
        "decisive_fact": _decisive_discussion_present(
            question,
            sections["decisive_fact"],
            recovered.semantic_provisional_option,
            recovered.semantic_second_best_option,
            trajectory_references=(sections["facts"], sections["implications"]),
        ),
        "answer_changing_change": _case_specific_counterfactual(
            question,
            sections["answer_changing_change"],
            recovered.semantic_second_best_option,
        ),
        "rereasoning": bool(
            _informative(sections["rereasoning"], 5)
            and decision_consistent
            and _case_grounded_explanation(
                question,
                sections["rereasoning"],
                (
                    recovered.semantic_provisional_option,
                    recovered.semantic_second_best_option,
                    recovered.semantic_final_option,
                ),
                trajectory_references=(
                    sections["facts"],
                    sections["implications"],
                    sections["decisive_fact"],
                    sections["answer_changing_change"],
                ),
            )
        ),
        "final_answer": bool(
            recovered.semantic_final_option in question.choices and answer.status != "conflicting"
        ),
    }


def _mean_scores(values: Mapping[str, bool]) -> float:
    return sum(float(value) for value in values.values()) / len(values) if values else 0.0


NONTRIVIAL_MIN_TOKENS = {
    "facts": 3,
    "implications": 3,
    "provisional_answer": 3,
    "second_best": 3,
    "decisive_fact": 3,
    "answer_changing_change": 3,
    "rereasoning": 3,
    "final_answer": 2,
}


def _nontrivial_sections(recovered: RecoveredProtocol) -> dict[str, bool]:
    """Mark visible sections that contain meaningful generated material.

    This is intentionally easier than semantic role completion.  An explicit option
    selection is meaningful content even when the model omitted the requested
    rationale; the contrastive-role metrics score that missing rationale separately.
    Likewise an explicit retain/revise decision is meaningful even if its explanation
    is weak.  Other roles need a few non-boilerplate terms.
    """
    result: dict[str, bool] = {}
    option_by_role = {
        "provisional_answer": recovered.semantic_provisional_option,
        "second_best": recovered.semantic_second_best_option,
        "final_answer": recovered.semantic_final_option,
    }
    for tag in CONTENT_TAGS:
        if recovered.semantic_start_counts[tag] < 1:
            result[tag] = False
            continue
        if tag in option_by_role and option_by_role[tag] is not None:
            result[tag] = True
            continue
        if tag == "rereasoning" and recovered.semantic_decision in VALID_DECISIONS:
            result[tag] = True
            continue
        result[tag] = _informative(recovered.sections[tag], NONTRIVIAL_MIN_TOKENS[tag])
    return result


def _section_diagnostics(
    recovered: RecoveredProtocol,
    substantive: Mapping[str, bool],
    nontrivial: Mapping[str, bool],
) -> SectionDiagnostics:
    """Measure identifiable roles independently of presentation syntax.

    ``present`` and ``unique`` use the frozen format-neutral role recognizer.
    ``schema_valid`` is retained only as a serialization diagnostic when a model
    voluntarily emits the historical XML-like markers.
    """
    present = {tag: recovered.semantic_start_counts[tag] >= 1 for tag in CONTENT_TAGS}
    unique = {tag: recovered.semantic_start_counts[tag] == 1 for tag in CONTENT_TAGS}
    schema_valid = {tag: recovered.marker_schema_valid[tag] for tag in CONTENT_TAGS}
    return {
        "present": present,
        "unique": unique,
        "nontrivial": dict(nontrivial),
        "schema_valid": schema_valid,
        "present_count": sum(present.values()),
        "unique_count": sum(unique.values()),
        "nontrivial_count": sum(bool(nontrivial.get(tag, False)) for tag in CONTENT_TAGS),
        "substantive_count": sum(bool(substantive.get(tag, False)) for tag in CONTENT_TAGS),
        "schema_valid_count": sum(schema_valid.values()),
    }


def _section_depth_score(counts: Mapping[str, int]) -> float:
    """Score nontrivial discussion in every visible reasoning role.

    Role-specific caps prevent a single long section from compensating for empty or
    perfunctory roles. This measures discussion depth, not factual correctness.
    """
    target_tokens = {
        "facts": 48,
        "implications": 64,
        "provisional_answer": 32,
        "second_best": 32,
        "decisive_fact": 24,
        "answer_changing_change": 28,
        "rereasoning": 36,
    }
    return sum(
        min(1.0, counts.get(role, 0) / target) for role, target in target_tokens.items()
    ) / len(target_tokens)


def _trajectory_complete(recovered: RecoveredProtocol, choices: Mapping[str, str]) -> bool:
    """Require one identifiable role each, correct order, and semantic field values."""
    return bool(
        all(count == 1 for count in recovered.semantic_start_counts.values())
        and recovered.semantic_ordered_starts
        and all(text for tag, text in recovered.sections.items() if tag != "final_answer")
        and recovered.semantic_provisional_option in choices
        and recovered.semantic_second_best_option in choices
        and recovered.semantic_second_best_option != recovered.semantic_provisional_option
        and recovered.semantic_final_option in choices
        and recovered.semantic_decision in VALID_DECISIONS
    )


def _semantic_trajectory_complete(
    recovered: RecoveredProtocol,
    choices: Mapping[str, str],
) -> bool:
    return bool(
        all(count == 1 for count in recovered.semantic_start_counts.values())
        and recovered.semantic_ordered_starts
        and all(text for tag, text in recovered.sections.items() if tag != "final_answer")
        and recovered.semantic_provisional_option in choices
        and recovered.semantic_second_best_option in choices
        and recovered.semantic_second_best_option != recovered.semantic_provisional_option
        and recovered.semantic_final_option in choices
        and recovered.semantic_decision in VALID_DECISIONS
    )


def _protocol_format_score(
    recovered: RecoveredProtocol,
    choices: Mapping[str, str],
    *,
    trajectory_complete: bool,
    strict_decision_consistent: bool,
) -> float:
    """Score exact adherence to the requested marker and attribute syntax."""
    components = {
        "all_required_starts_once": all(count == 1 for count in recovered.start_counts.values()),
        "all_marker_schemas_valid": all(recovered.marker_schema_valid.values()),
        "requested_order": recovered.ordered_starts,
        "trajectory_complete": trajectory_complete,
        "provisional_option_valid": recovered.provisional_option in choices,
        "second_best_distinct": bool(
            recovered.second_best_option in choices
            and recovered.second_best_option != recovered.provisional_option
        ),
        "final_option_valid": recovered.final_option in choices,
        "rereasoning_decision_valid": recovered.decision in VALID_DECISIONS,
        "strict_decision_consistent": strict_decision_consistent,
    }
    return _mean_scores(components)


def _structure_score(
    recovered: RecoveredProtocol,
    choices: Mapping[str, str],
    *,
    semantic_trajectory_complete: bool,
    decision_consistent: bool,
) -> float:
    """Score visible reasoning topology independently of serialization syntax."""
    components = {
        "all_visible_roles_once": all(
            count == 1 for count in recovered.semantic_start_counts.values()
        ),
        "visible_role_order": recovered.semantic_ordered_starts,
        "all_reasoning_sections_nonempty": bool(
            all(text for tag, text in recovered.sections.items() if tag != "final_answer")
        ),
        "semantic_trajectory_complete": semantic_trajectory_complete,
        "provisional_option_valid": recovered.semantic_provisional_option in choices,
        "second_best_distinct": bool(
            recovered.semantic_second_best_option in choices
            and recovered.semantic_second_best_option != recovered.semantic_provisional_option
        ),
        "final_option_valid": recovered.semantic_final_option in choices,
        "rereasoning_decision_valid": recovered.semantic_decision in VALID_DECISIONS,
        "decision_consistent": decision_consistent,
    }
    return _mean_scores(components)


def _contrastive_role_score(
    substantive: Mapping[str, bool],
    *,
    decision_consistent: bool,
) -> float:
    components = {
        "second_best": substantive.get("second_best", False),
        "decisive_fact": substantive.get("decisive_fact", False),
        "answer_changing_change": substantive.get("answer_changing_change", False),
        "rereasoning": substantive.get("rereasoning", False),
        "decision_consistent": decision_consistent,
    }
    return _mean_scores(components)


def _instructed_judgment(
    question: Question,
    status: str,
    raw: str,
    lexical_reference: LexicalReference,
    inventory: QuestionFacts,
) -> Judgment:
    recovered = recover_protocol(raw, question.choices)
    sections = recovered.sections
    coverage_facts, facts_choice_list_leakage = _facts_without_choice_list(
        sections["facts"], question.choices
    )
    lexical = score_preanswer(
        question.stem,
        sections["facts"],
        sections["implications"],
        lexical_reference,
    )
    fact_coverage = score_fact_inventory(
        inventory,
        coverage_facts,
        sections["implications"],
    )
    semantic_provisional_start = _semantic_provisional_start(recovered)
    preprovisional = (
        recovered.document[:semantic_provisional_start]
        if semantic_provisional_start is not None
        else recovered.document
    )
    commitments = _commitments(preprovisional, question.choices)
    counts = _section_counts(recovered)
    strict_decision_consistent = _strict_decision_consistent(recovered, question.choices)
    decision_consistent = _semantic_decision_consistent(recovered, question.choices)
    answer = _answer_from_recovered(recovered, question.choices)
    substantive = _substantive_sections(
        question,
        recovered,
        lexical,
        decision_consistent,
        facts_choice_list_leakage,
        answer,
    )
    nontrivial = _nontrivial_sections(recovered)
    # Paper-facing validated completion uses seven semantic role checks plus a
    # deliberately simpler Step-6 observable: non-trivial generated content.
    # The strict semantic Step-6 result remains exported as exploratory.
    validated_substantive = dict(substantive)
    validated_substantive["answer_changing_change"] = bool(nontrivial["answer_changing_change"])
    validated_role_count = sum(bool(value) for value in validated_substantive.values())
    validated_role_completeness_score = validated_role_count / len(CONTENT_TAGS)
    all_roles_validated_complete = bool(all(validated_substantive.values()))
    section_diagnostics = _section_diagnostics(recovered, substantive, nontrivial)
    trajectory_complete = _trajectory_complete(recovered, question.choices)
    semantic_trajectory_complete = _semantic_trajectory_complete(recovered, question.choices)
    contrastive_complete = bool(
        semantic_trajectory_complete
        and substantive["second_best"]
        and substantive["decisive_fact"]
        and substantive["answer_changing_change"]
        and substantive["rereasoning"]
        and decision_consistent
    )
    minimum_scaffold = bool(
        semantic_trajectory_complete
        and all(substantive.values())
        and decision_consistent
        and not commitments
    )
    visible_reasoning_scaffold_complete = bool(
        semantic_trajectory_complete
        and all(substantive.values())
        and decision_consistent
        and not commitments
    )
    preanswer_discussion_complete = bool(
        all(
            recovered.semantic_start_counts[tag] == 1
            for tag in ("facts", "implications", "provisional_answer")
        )
        and recovered.semantic_preanswer_ordered
        and recovered.semantic_provisional_option in question.choices
        and _preanswer_facts_discussed(question, sections["facts"])
        and substantive["implications"]
        and substantive["provisional_answer"]
        and not commitments
    )
    visible_preanswer_discussion_complete = bool(
        all(
            recovered.semantic_start_counts[tag] == 1
            for tag in ("facts", "implications", "provisional_answer")
        )
        and recovered.semantic_preanswer_ordered
        and recovered.semantic_provisional_option in question.choices
        and _preanswer_facts_discussed(question, sections["facts"])
        and substantive["implications"]
        and substantive["provisional_answer"]
        and not commitments
    )
    protocol_format_score = _protocol_format_score(
        recovered,
        question.choices,
        trajectory_complete=trajectory_complete,
        strict_decision_consistent=strict_decision_consistent,
    )
    structure_score = _structure_score(
        recovered,
        question.choices,
        semantic_trajectory_complete=semantic_trajectory_complete,
        decision_consistent=decision_consistent,
    )
    substantive_role_score = _mean_scores(substantive)
    contrastive_role_score = _contrastive_role_score(
        substantive, decision_consistent=decision_consistent
    )
    section_depth_score = _section_depth_score(counts)
    section_denominator = float(len(CONTENT_TAGS))
    section_present_fraction = section_diagnostics["present_count"] / section_denominator
    section_unique_fraction = section_diagnostics["unique_count"] / section_denominator
    section_nontrivial_fraction = section_diagnostics["nontrivial_count"] / section_denominator
    section_substantive_fraction = section_diagnostics["substantive_count"] / section_denominator
    section_schema_valid_fraction = section_diagnostics["schema_valid_count"] / section_denominator
    identified_role_count = section_diagnostics["present_count"]
    unique_role_count = section_diagnostics["unique_count"]
    required_section_count = section_diagnostics["present_count"]
    nontrivial_section_count = section_diagnostics["nontrivial_count"]
    substantive_role_count = section_diagnostics["substantive_count"]
    contrastive_discussion_components = (
        substantive["provisional_answer"],
        substantive["second_best"],
        substantive["decisive_fact"],
        substantive["answer_changing_change"],
        substantive["rereasoning"],
    )
    contrastive_discussion_count = sum(bool(value) for value in contrastive_discussion_components)
    contrastive_discussion_score = contrastive_discussion_count / len(
        contrastive_discussion_components
    )
    validated_contrastive_discussion_components = (
        validated_substantive["provisional_answer"],
        validated_substantive["second_best"],
        validated_substantive["decisive_fact"],
        validated_substantive["answer_changing_change"],
        validated_substantive["rereasoning"],
    )
    validated_contrastive_discussion_count = sum(
        bool(value) for value in validated_contrastive_discussion_components
    )
    validated_contrastive_discussion_score = validated_contrastive_discussion_count / len(
        validated_contrastive_discussion_components
    )
    role_completeness_score = section_substantive_fraction
    all_roles_substantive = bool(all(substantive.values()))
    roles_in_requested_order = bool(recovered.semantic_ordered_starts)
    complete_role_scaffold = bool(
        all_roles_substantive
        and all(section_diagnostics["unique"].values())
        and roles_in_requested_order
    )
    validated_complete_role_scaffold = bool(
        all_roles_validated_complete
        and all(section_diagnostics["unique"].values())
        and roles_in_requested_order
    )
    parseable = answer.status == "parsed" and answer.option is not None
    eligible = inventory.applicable
    minimum_value = float(minimum_scaffold) if eligible else None
    contrastive_value = float(contrastive_complete) if eligible else None

    def eligible_lexical(name: str) -> float | None:
        return lexical[name] if eligible else None

    return {
        "judge_version": JUDGE_VERSION,
        "generation_usable": True,
        "generation_status": status,
        "repair_endpoint_eligible": eligible,
        "repair_endpoint_inapplicability_reason": inventory.inapplicability_reason,
        "essential_fact_count": len(inventory.facts),
        "fact_complexity": inventory.complexity,
        "fact_inventory_review_status": "automatic_unreviewed",
        "protocol_scaffold_complete": float(minimum_scaffold),
        "preanswer_discussion_complete": float(preanswer_discussion_complete),
        "visible_reasoning_scaffold_complete": float(visible_reasoning_scaffold_complete),
        "visible_preanswer_discussion_complete": float(visible_preanswer_discussion_complete),
        "minimum_repair_scaffold": minimum_value,
        "contrastive_scaffold_complete": contrastive_value,
        "correct_with_minimum_repair_scaffold": (
            float(bool(minimum_scaffold and parseable and answer.option == question.gold))
            if eligible
            else None
        ),
        "protocol_format_score": protocol_format_score,
        "structure_score": structure_score,
        "section_depth_score": section_depth_score,
        "substantive_role_score": substantive_role_score if eligible else None,
        "contrastive_role_score": contrastive_role_score if eligible else None,
        "visible_substantive_role_score": substantive_role_score,
        "visible_contrastive_role_score": contrastive_role_score,
        "identified_role_count": identified_role_count,
        "unique_role_count": unique_role_count,
        "required_section_count": required_section_count,
        "nontrivial_section_count": nontrivial_section_count,
        "substantive_role_count": substantive_role_count,
        "validated_role_count": validated_role_count,
        "validated_role_completeness_score": validated_role_completeness_score,
        "contrastive_discussion_count": contrastive_discussion_count,
        "contrastive_discussion_score": contrastive_discussion_score,
        "validated_contrastive_discussion_count": validated_contrastive_discussion_count,
        "validated_contrastive_discussion_score": validated_contrastive_discussion_score,
        "provisional_answer_discussed": float(substantive["provisional_answer"]),
        "best_alternative_discussed": float(substantive["second_best"]),
        "decisive_distinction_discussed": float(substantive["decisive_fact"]),
        "answer_change_discussed": float(substantive["answer_changing_change"]),
        "reconsideration_discussed": float(substantive["rereasoning"]),
        "role_completeness_score": role_completeness_score,
        "all_roles_substantive": float(all_roles_substantive),
        "all_roles_validated_complete": float(all_roles_validated_complete),
        "validated_complete_role_scaffold": float(validated_complete_role_scaffold),
        "roles_in_requested_order": float(roles_in_requested_order),
        "complete_role_scaffold": float(complete_role_scaffold),
        "role_facts_complete": float(substantive["facts"]),
        "role_implications_complete": float(substantive["implications"]),
        "role_provisional_answer_complete": float(substantive["provisional_answer"]),
        "role_best_alternative_complete": float(substantive["second_best"]),
        "role_decisive_distinction_complete": float(substantive["decisive_fact"]),
        "role_answer_changing_change_complete": float(substantive["answer_changing_change"]),
        "role_answer_changing_change_nontrivial": float(
            validated_substantive["answer_changing_change"]
        ),
        "role_reconsideration_complete": float(substantive["rereasoning"]),
        "role_final_answer_complete": float(substantive["final_answer"]),
        "trajectory_criteria": {
            "trajectory_structure": trajectory_complete,
            "semantic_trajectory_structure": semantic_trajectory_complete,
            "visible_reasoning_scaffold": visible_reasoning_scaffold_complete,
            "all_sections_substantive": all(substantive.values()),
            "facts_exclude_explicit_choice_list": not facts_choice_list_leakage,
            "contrastive_check": contrastive_complete,
            "decision_and_final_consistency": bool(
                decision_consistent and substantive["final_answer"]
            ),
        },
        "preprovisional_commitment_observed": float(bool(commitments)),
        "preprovisional_commitment_itt": float(bool(commitments)),
        "trajectory_complete": semantic_trajectory_complete,
        "semantic_trajectory_complete": semantic_trajectory_complete,
        "section_substantive": substantive,
        "section_token_counts": counts,
        "section_marker_counts": dict(recovered.start_counts),
        "section_present": section_diagnostics["present"],
        "section_unique": section_diagnostics["unique"],
        "section_nontrivial": section_diagnostics["nontrivial"],
        "section_schema_valid": section_diagnostics["schema_valid"],
        "section_present_count": section_diagnostics["present_count"],
        "section_unique_count": section_diagnostics["unique_count"],
        "section_nontrivial_count": section_diagnostics["nontrivial_count"],
        "section_substantive_count": section_diagnostics["substantive_count"],
        "section_schema_valid_count": section_diagnostics["schema_valid_count"],
        "section_present_fraction": section_present_fraction,
        "section_unique_fraction": section_unique_fraction,
        "section_nontrivial_fraction": section_nontrivial_fraction,
        "section_substantive_fraction": section_substantive_fraction,
        "section_schema_valid_fraction": section_schema_valid_fraction,
        "facts_choice_list_leakage": facts_choice_list_leakage,
        "provisional_option": recovered.semantic_provisional_option,
        "second_best_option": recovered.semantic_second_best_option,
        "rereasoning_decision": recovered.semantic_decision,
        "decision_consistent": decision_consistent,
        "strict_decision_consistent": strict_decision_consistent,
        "final_option": answer.option,
        "accuracy": int(bool(parseable and answer.option == question.gold)),
        "accuracy_parseable": parseable,
        "answer_extraction_status": answer.status,
        "answer_candidates": list(answer.candidates),
        "trajectory_errors": list(recovered.errors),
        "early_commitments": commitments,
        "facts_tfidf_recall": eligible_lexical("facts_tfidf_recall"),
        "implications_tfidf_recall": eligible_lexical("implications_tfidf_recall"),
        "preanswer_tfidf_recall": eligible_lexical("preanswer_tfidf_recall"),
        "preprovisional_tfidf_recall": eligible_lexical("preanswer_tfidf_recall"),
        "preanswer_high_idf_tfidf_recall": eligible_lexical("preanswer_high_idf_tfidf_recall"),
        "facts_implications_shared_tfidf_recall": eligible_lexical(
            "facts_implications_shared_tfidf_recall"
        ),
        "facts_implications_shared_high_idf_recall": eligible_lexical(
            "facts_implications_shared_high_idf_recall"
        ),
        "preanswer_tfidf_density_per_100_tokens": eligible_lexical(
            "preanswer_tfidf_density_per_100_tokens"
        ),
        "preanswer_tfidf_token_count": eligible_lexical("preanswer_tfidf_token_count"),
        "facts_anchor_recall": eligible_lexical("facts_anchor_recall"),
        "implications_anchor_recall": eligible_lexical("implications_anchor_recall"),
        "preanswer_anchor_recall": eligible_lexical("preanswer_anchor_recall"),
        "shared_anchor_recall": eligible_lexical("shared_anchor_recall"),
        "atomic_fact_coverage": fact_coverage["atomic_fact_coverage"],
        "atomic_implication_trace_coverage": fact_coverage["atomic_implication_trace_coverage"],
        "hard_qualifier_fact_recall": fact_coverage["hard_qualifier_fact_recall"],
        "hard_qualifier_trace_recall": fact_coverage["hard_qualifier_trace_recall"],
        "fact_coverage_details": fact_coverage["fact_coverage_details"],
    }


def judge(
    question: Question,
    status: str,
    raw: str | None,
    condition_id: str,
    lexical_reference: LexicalReference,
    inventory: QuestionFacts | None = None,
) -> Judgment:
    """Compute versioned deterministic measurements for one cell."""
    effective_inventory = inventory or build_question_facts(question)
    instructed = condition_id != "zero"
    if status != CellStatus.COMPLETED.value or raw is None or not raw.strip():
        return _failure_judgment(instructed, status, effective_inventory)
    if not instructed:
        return _baseline_judgment(question, status, raw, effective_inventory)
    return _instructed_judgment(
        question,
        status,
        raw,
        lexical_reference,
        effective_inventory,
    )
