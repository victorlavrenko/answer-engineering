"""Automatic atomic-fact inventory and qualifier-preservation diagnostics."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypedDict

from .json_types import (
    JsonObject,
    boolean_value,
    integer_value,
    is_object_sequence,
    json_object,
    object_value,
)
from .types import Question

FACT_INVENTORY_VERSION = "automatic-clause-facts-v3"

TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?%?")
NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*(?:%|mg|mcg|g|kg|mm|cm|ml|l|mmhg|bpm|"
    r"years?|months?|weeks?|days?|hours?|minutes?))?\b",
    re.IGNORECASE,
)
QUESTION_START_RE = re.compile(
    r"^(?:what|which|who|when|where|why|how|select|choose|identify|"
    r"the most likely|the best|the next)\b",
    re.IGNORECASE,
)
CAPITALIZED_QUESTION_CLAUSE_RE = re.compile(
    r"\s+(?:What|Which|Who|When|Where|Why|How|Select|Choose|Identify)\b"
)
CONDITIONAL_QUESTION_CLAUSE_RE = re.compile(
    r"^(?:if|when)\b.*?[,;:]\s*(?:what|which|how)\b",
    re.IGNORECASE,
)
TOPIC_ONLY_RE = re.compile(
    r"^(?:concerning|regarding|with regard to|with respect to|in relation to)\b",
    re.IGNORECASE,
)
META_PREFIX_RE = re.compile(
    r"^(?:based on|according to)\b",
    re.IGNORECASE,
)
SOURCE_NUMBER_RE = re.compile(r"^\d+\s*\.\s*$")
INCOMPLETE_PROMPT_RE = re.compile(
    r"(?:[:;]|\b(?:is|are|was|were|seen in|following(?: except)?|except))\s*$",
    re.IGNORECASE,
)
COORDINATE_SPLIT_RE = re.compile(
    r"(?:,\s*|\s+)(?:and|but|while|whereas)\s+"
    r"(?=(?:he|she|it|they|the patient|the newborn|the mother|the patient|"
    r"is|are|was|were|has|have|had|does|do|did|can|could|will|would|should|"
    r"may|might|appears|becomes|starts|reports|states|says|notes|denies|"
    r"takes|works|shows|reveals|no|not|without)\b)",
    re.IGNORECASE,
)
INFERENCE_CUE_RE = re.compile(
    r"\b(?:support|suggest|favor|favour|argue|against|exclude|rule out|"
    r"consistent|inconsistent|imply|indicate|because|therefore|thus|due to|"
    r"distinguish|differentiat|point to|weakens?)\b|→",
    re.IGNORECASE,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "patient",
    "patients",
    "she",
    "that",
    "the",
    "their",
    "then",
    "there",
    "this",
    "to",
    "was",
    "were",
    "which",
    "who",
    "with",
}
NEGATIONS = {"no", "not", "without", "denies", "denied", "absent", "never", "neither"}
LATERALITY = {"left", "right", "bilateral", "unilateral", "ipsilateral", "contralateral"}
TIMING = {
    "sudden",
    "acute",
    "chronic",
    "gradual",
    "recurrent",
    "persistent",
    "intermittent",
    "hours",
    "hour",
    "days",
    "day",
    "weeks",
    "week",
    "months",
    "month",
    "years",
    "year",
    "minutes",
    "minute",
    "onset",
    "duration",
}
DIRECTION = {
    "high",
    "higher",
    "elevated",
    "increased",
    "rising",
    "low",
    "lower",
    "decreased",
    "reduced",
    "positive",
    "negative",
    "normal",
    "abnormal",
    "prolonged",
    "shortened",
    "widened",
    "narrowed",
}


class FactCoverage(TypedDict):
    atomic_fact_coverage: float | None
    atomic_implication_trace_coverage: float | None
    hard_qualifier_fact_recall: float | None
    hard_qualifier_trace_recall: float | None
    fact_coverage_details: list[JsonObject]


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for raw in TOKEN_RE.findall(text)
        if len(token := raw.casefold().strip("-'_")) > 1 and token not in STOPWORDS
    )


def _hard_features(text: str) -> dict[str, tuple[str, ...]]:
    tokens = set(_tokens(text))
    values = {
        "numbers": tuple(
            sorted(match.group(0).casefold().replace(" ", "") for match in NUMBER_RE.finditer(text))
        ),
        "negation": tuple(sorted(tokens & NEGATIONS)),
        "laterality": tuple(sorted(tokens & LATERALITY)),
        "timing": tuple(sorted(tokens & TIMING)),
        "direction": tuple(sorted(tokens & DIRECTION)),
    }
    return {key: value for key, value in values.items() if value}


def _strip_question_clause(sentence: str) -> str:
    """Return the factual/scenario prefix before an explicit question clause.

    Capitalization is useful here: source questions normally introduce their final
    interrogative with ``Which``/``What`` while ordinary relative clauses use lower
    case ``which``/``who``.  Lower-case tails are accepted only for explicit
    conditional question forms such as ``If ..., which ...``.
    """
    if QUESTION_START_RE.match(sentence):
        return ""
    capitalized = CAPITALIZED_QUESTION_CLAUSE_RE.search(sentence)
    if capitalized is not None:
        return sentence[: capitalized.start()].rstrip(" ,;:")
    conditional = CONDITIONAL_QUESTION_CLAUSE_RE.match(sentence)
    if conditional is not None:
        tail = re.search(r"[,;:]\s*(?:what|which|how)\b", conditional.group(0), re.IGNORECASE)
        if tail is not None:
            return sentence[: tail.start()].rstrip(" ,;:")
    return sentence


def _standalone_fact_candidate(text: str) -> bool:
    candidate = text.strip(" ,")
    if not candidate or SOURCE_NUMBER_RE.fullmatch(candidate):
        return False
    if QUESTION_START_RE.match(candidate) or TOPIC_ONLY_RE.match(candidate):
        return False
    if META_PREFIX_RE.match(candidate):
        return False
    if INCOMPLETE_PROMPT_RE.search(candidate):
        return False
    return bool(_tokens(candidate))


def _fact_clauses(stem: str) -> list[str]:
    """Extract deterministic fact/scenario units without parsing answer prompts as facts.

    The extractor is deliberately conservative.  It retains declarative clinical
    sentences and conditional scenario premises, removes the explicit interrogative
    tail, and only splits coordinated clauses when the second half visibly starts a
    new predication.  Topic labels and sentence-completion prompts are inapplicable
    to fact-repair endpoints but remain in every other part of the experiment.
    """
    normalized = " ".join(stem.split())
    if not normalized or TOPIC_ONLY_RE.match(normalized):
        return []

    sentences = re.split(r"(?<=[.!?])\s+|\s*;\s*", normalized)
    facts: list[str] = []
    for raw_sentence in sentences:
        sentence = _strip_question_clause(raw_sentence.strip())
        if not _standalone_fact_candidate(sentence):
            continue
        clauses = COORDINATE_SPLIT_RE.split(sentence)
        for raw_clause in clauses:
            clause = raw_clause.strip(" ,")
            if _standalone_fact_candidate(clause):
                facts.append(clause)
    return facts


@dataclass(frozen=True, slots=True)
class FactUnit:
    """One deterministic fact-sized span from a question stem."""

    fact_id: str
    source_text: str
    weight: int
    hard_features: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hard_features",
            MappingProxyType({key: tuple(value) for key, value in self.hard_features.items()}),
        )

    def to_dict(self) -> JsonObject:
        return {
            "fact_id": self.fact_id,
            "source_text": self.source_text,
            "weight": self.weight,
            "hard_features": {key: list(value) for key, value in self.hard_features.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FactUnit:
        raw_features = object_value(value.get("hard_features"), name="fact.hard_features")
        features: dict[str, tuple[str, ...]] = {}
        for key, raw in raw_features.items():
            if not is_object_sequence(raw) or not all(isinstance(item, str) for item in raw):
                raise ValueError(f"fact hard feature {key!r} must be a string list")
            features[key] = tuple(str(item) for item in raw)
        return cls(
            fact_id=str(value["fact_id"]),
            source_text=str(value["source_text"]),
            weight=integer_value(value["weight"], name="fact.weight"),
            hard_features=features,
        )


@dataclass(frozen=True, slots=True)
class QuestionFacts:
    """The automatic fact inventory and applicability decision for one question."""

    question_id: str
    applicable: bool
    inapplicability_reason: str | None
    facts: tuple[FactUnit, ...]

    @property
    def complexity(self) -> str:
        count = len(self.facts)
        if count == 0:
            return "inapplicable"
        if count == 1:
            return "one_fact"
        if count == 2:
            return "two_facts"
        return "three_or_more_facts"

    def to_dict(self) -> JsonObject:
        return {
            "fact_inventory_version": FACT_INVENTORY_VERSION,
            "question_id": self.question_id,
            "automatic_unreviewed": True,
            "repair_endpoint_eligible": self.applicable,
            "inapplicability_reason": self.inapplicability_reason,
            "essential_fact_count": len(self.facts),
            "fact_complexity": self.complexity,
            "facts": [fact.to_dict() for fact in self.facts],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> QuestionFacts:
        if value.get("fact_inventory_version") != FACT_INVENTORY_VERSION:
            raise ValueError("fact inventory version does not match the current judge")
        raw_facts = value.get("facts")
        if not is_object_sequence(raw_facts):
            raise ValueError("fact inventory facts must be a list")
        return cls(
            question_id=str(value["question_id"]),
            applicable=boolean_value(
                value["repair_endpoint_eligible"],
                name="fact inventory repair_endpoint_eligible",
            ),
            inapplicability_reason=(
                None
                if value.get("inapplicability_reason") is None
                else str(value["inapplicability_reason"])
            ),
            facts=tuple(
                FactUnit.from_dict(object_value(item, name=f"facts[{index}]"))
                for index, item in enumerate(raw_facts)
            ),
        )


def build_question_facts(question: Question) -> QuestionFacts:
    clauses = _fact_clauses(question.stem)
    facts = tuple(
        FactUnit(
            fact_id=f"{question.id}:F{index:02d}",
            source_text=clause,
            weight=2 if _hard_features(clause) else 1,
            hard_features=_hard_features(clause),
        )
        for index, clause in enumerate(clauses, 1)
    )
    reason = None if facts else "stem_has_no_standalone_fact_outside_the_question"
    return QuestionFacts(question.id, bool(facts), reason, facts)


def build_fact_inventory(questions: Iterable[Question]) -> list[QuestionFacts]:
    return [build_question_facts(question) for question in questions]


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(
        sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()
    )


def _feature_present(kind: str, value: str, text: str) -> bool:
    normalized = text.casefold()
    if kind == "numbers":
        return value in re.sub(r"\s+", "", normalized)
    return value in set(_tokens(normalized))


def _direction_conflict(required: Sequence[str], text: str) -> bool:
    present = set(_tokens(text)) & DIRECTION
    high = {"high", "higher", "elevated", "increased", "rising", "positive"}
    low = {"low", "lower", "decreased", "reduced", "negative"}
    required_set = set(required)
    return bool((required_set & high and present & low) or (required_set & low and present & high))


def _fact_match(fact: FactUnit, text: str) -> tuple[bool, float, list[str], int, int]:
    source = set(_tokens(fact.source_text))
    target = set(_tokens(text))
    matched = sorted(source & target)
    recall = len(matched) / len(source) if source else 0.0
    threshold = 0.55 if len(source) >= 5 else 0.67
    lexical = bool(source and (recall >= threshold or len(matched) >= 5))
    qualifier_total = 0
    qualifier_present = 0
    conflict = False
    for kind, values in fact.hard_features.items():
        qualifier_total += len(values)
        qualifier_present += sum(_feature_present(kind, value, text) for value in values)
        if kind == "direction" and _direction_conflict(values, text):
            conflict = True
    qualifiers_ok = qualifier_present == qualifier_total and not conflict
    return lexical and qualifiers_ok, recall, matched, qualifier_present, qualifier_total


def score_fact_inventory(
    inventory: QuestionFacts,
    facts_text: str,
    implications_text: str,
) -> FactCoverage:
    """Score atomic representation, inference tracing, and hard qualifiers."""
    if not inventory.applicable:
        return {
            "atomic_fact_coverage": None,
            "atomic_implication_trace_coverage": None,
            "hard_qualifier_fact_recall": None,
            "hard_qualifier_trace_recall": None,
            "fact_coverage_details": [],
        }
    represented_weight = 0
    traced_weight = 0
    total_weight = sum(fact.weight for fact in inventory.facts)
    fact_qualifiers_present = 0
    trace_qualifiers_present = 0
    qualifiers_total = 0
    details: list[JsonObject] = []
    fact_sentences = _sentences(facts_text) or (facts_text,)
    implication_sentences = _sentences(implications_text) or (implications_text,)
    for fact in inventory.facts:
        fact_candidates = [(_fact_match(fact, sentence), sentence) for sentence in fact_sentences]
        best_fact, fact_evidence = max(fact_candidates, key=lambda item: item[0][1])
        implication_candidates = [
            (_fact_match(fact, sentence), sentence)
            for sentence in implication_sentences
            if INFERENCE_CUE_RE.search(sentence)
        ]
        best_trace: tuple[bool, float, list[str], int, int]
        trace_evidence: str
        if implication_candidates:
            best_trace, trace_evidence = max(implication_candidates, key=lambda item: item[0][1])
        else:
            best_trace, trace_evidence = (False, 0.0, list[str](), 0, best_fact[4]), ""
        represented_weight += fact.weight if best_fact[0] else 0
        traced_weight += fact.weight if best_trace[0] else 0
        fact_qualifiers_present += best_fact[3]
        trace_qualifiers_present += best_trace[3]
        qualifiers_total += best_fact[4]
        details.append(
            json_object(
                {
                    "fact_id": fact.fact_id,
                    "source_text": fact.source_text,
                    "weight": fact.weight,
                    "represented_in_facts": best_fact[0],
                    "facts_lexical_recall": best_fact[1],
                    "facts_matched_tokens": best_fact[2],
                    "facts_evidence": fact_evidence,
                    "traced_in_implications": best_trace[0],
                    "implications_lexical_recall": best_trace[1],
                    "implications_matched_tokens": best_trace[2],
                    "implications_evidence": trace_evidence or None,
                },
                path=f"fact coverage {fact.fact_id}",
            )
        )
    return {
        "atomic_fact_coverage": represented_weight / total_weight if total_weight else 0.0,
        "atomic_implication_trace_coverage": traced_weight / total_weight if total_weight else 0.0,
        "hard_qualifier_fact_recall": (
            fact_qualifiers_present / qualifiers_total if qualifiers_total else 1.0
        ),
        "hard_qualifier_trace_recall": (
            trace_qualifiers_present / qualifiers_total if qualifiers_total else 1.0
        ),
        "fact_coverage_details": details,
    }


def question_qc(question: Question, inventory: QuestionFacts) -> JsonObject:
    """Return transparent review flags without guessing medical correctness."""
    malformed_gold_fields = False
    try:
        decoded: object = json.loads(question.gold_raw)
        decoded_object = object_value(decoded, name="question.gold_raw")
        malformed_gold_fields = bool(decoded_object.get("malformed"))
    except (TypeError, ValueError):
        malformed_gold_fields = False
    flags: list[str] = []
    if not inventory.applicable:
        flags.append("repair_endpoints_inapplicable_topic_or_choice_dependent_stem")
    if malformed_gold_fields:
        flags.append("one_or_more_secondary_gold_fields_were_malformed")
    if len(_tokens(question.stem)) < 5:
        flags.append("very_short_stem")
    return json_object(
        {
            "question_id": question.id,
            "dataset": question.dataset,
            "stem": question.stem,
            "choices": dict(question.choices),
            "normalized_gold": question.gold,
            "normalized_gold_text": question.gold_text,
            "gold_source": question.gold_source,
            "gold_conflict_detected": False,
            "repair_endpoint_eligible": inventory.applicable,
            "essential_fact_count": len(inventory.facts),
            "fact_complexity": inventory.complexity,
            "review_priority": "targeted" if flags else "routine",
            "review_flags": flags,
            "reviewed_by_clinician": False,
            "clinician_disposition": None,
            "clinician_notes": None,
        },
        path=f"question QC {question.id}",
    )
