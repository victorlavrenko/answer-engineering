"""Deterministic lexical exposure measurements for visible pre-answer reasoning.

Paper mapping: implements ``Pre-provisional TF-IDF recall`` in ``Experiment ->
Measurements and Eligibility``: the numerator is PubMed-IDF-weighted stem content
recovered in Facts + Implications, capped by stem frequency; the denominator is the
total weighted content of the source stem."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from types import MappingProxyType

from .io_utils import sha256_json
from .json_types import JsonObject, json_object, number_value, object_value
from .pubmed_idf import (
    PUBMED_COMMIT,
    PUBMED_EXPECTED_SHA256,
    PUBMED_REFERENCE_NAME,
    PUBMED_REFERENCE_URL,
    PUBMED_REFERENCE_VERSION,
    fixture_document_frequencies,
    read_document_frequencies,
)
from .types import Question

LEXICAL_VERSION = "pubmed-idf-lexical-exposure-v11"
ABBREVIATION_VERSION = "reviewed-medical-abbreviations-v1"
IDF_FORMULA = "log((N + 1) / (df + 1)) + 1"
IDF_CAP = 10.0
HIGH_IDF_THRESHOLD = 6.0

# TF-IDF should let corpus frequency downweight ordinary domain words such as
# "patient" rather than deleting them. Only high-frequency grammatical words
# that carry little stem-specific content are removed here.
TFIDF_STOPWORDS = {
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
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "in",
    "is",
    "it",
    "its",
    "may",
    "might",
    "of",
    "on",
    "or",
    "she",
    "should",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "were",
    "which",
    "who",
    "whom",
    "will",
    "with",
    "would",
}

# The polarity-aware anchor metric remains as a complementary diagnostic. It is
# intentionally more selective than TF-IDF and gives explicit extra weight to
# qualifiers that are easy to lose while paraphrasing a clinical stem.
ANCHOR_STOPWORDS = TFIDF_STOPWORDS | {
    "answer",
    "best",
    "diagnosis",
    "following",
    "likely",
    "most",
    "next",
    "patient",
    "patients",
    "question",
    "step",
}
NEGATIONS = {"no", "not", "never", "without", "denies", "denied", "absent", "absence"}
LATERALITY = {"left", "right", "bilateral", "ipsilateral", "contralateral"}
TIMING = {
    "acute",
    "chronic",
    "sudden",
    "gradual",
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
}
TOKEN_RE = re.compile(r"\d+(?:\.\d+)?%?|[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,7}\b")
EXPLICIT_ABBREVIATION_RE = re.compile(
    r"(?P<long>(?:[A-Za-z][A-Za-z0-9-]*[\s-]+){1,7}[A-Za-z][A-Za-z0-9-]*)"
    r"\s*\((?P<short>[A-Z][A-Z0-9]{1,7})\)"
)
SOURCE_ITEM_PREFIX_RE = re.compile(r"^\s*\d+\s*[.)][\s\u200b]*")

_IRREGULAR_CANONICAL = {
    "children": "child",
    "feet": "foot",
    "men": "man",
    "teeth": "tooth",
    "women": "woman",
}

# Only deliberately reviewed, common medical expansions are included. Values
# contain the canonical content terms produced by this module's tokenizer; a
# one-character initial such as the B in GBS is therefore intentionally absent.
# The mapping is symmetric at matching time and is never inferred from arbitrary
# word initials.
ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "aids": ("acquired", "immunodeficiency", "syndrome"),
    "copd": ("chronic", "obstructive", "pulmonary", "disease"),
    "csf": ("cerebrospinal", "fluid"),
    "dka": ("diabetic", "ketoacidosis"),
    "ecg": ("electrocardiogram",),
    "ekg": ("electrocardiogram",),
    "hiv": ("human", "immunodeficiency", "virus"),
    "mi": ("myocardial", "infarction"),
    "mri": ("magnetic", "resonance", "imaging"),
    "fasd": ("fetal", "alcohol", "spectrum", "disorder"),
    "gbs": ("group", "streptococcus"),
}


@dataclass(frozen=True, slots=True)
class LexicalReference:
    """Validated in-memory TF-IDF reference reused across all judged cells."""

    document_count: int
    document_frequency: Mapping[str, int]
    high_idf_threshold: float
    idf_cap: float


def _normalize_lexical_text(text: str) -> str:
    """Normalize presentation punctuation before deterministic tokenization."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
    normalized = normalized.replace("\N{LEFT SINGLE QUOTATION MARK}", "'")
    for dash in ("\N{HYPHEN}", "\N{NON-BREAKING HYPHEN}", "\N{EN DASH}", "\N{EM DASH}"):
        normalized = normalized.replace(dash, "-")
    return normalized.replace("-", " ")


def normalize_token(token: str) -> str:
    """Apply conservative token normalization without broad stemming."""
    normalized = token.strip("-'_")
    if len(normalized) > 2 and normalized.endswith("'s"):
        normalized = normalized[:-2]
    return normalized


def _raw_tokens(text: str) -> tuple[str, ...]:
    """Tokenize normalized text with a bounded cache for repeated section scoring."""
    return tuple(
        token
        for raw in TOKEN_RE.findall(_normalize_lexical_text(text))
        if (token := normalize_token(raw)) and len(token) > 1
    )


def _canonical_term(token: str) -> str:
    """Collapse only conservative English inflections used for lexical matching."""
    irregular = _IRREGULAR_CANONICAL.get(token)
    if irregular is not None:
        return irregular
    if len(token) > 4 and token.endswith("ies") and token[-4] not in "aeiou":
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("xes", "zes", "ches", "shes", "sses")):
        return token[:-2]
    if (
        len(token) > 3
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
        and not token.replace(".", "").isdigit()
    ):
        return token[:-1]
    return token


def content_tokens(text: str) -> tuple[str, ...]:
    """Return the selective token stream used by polarity-aware anchor diagnostics."""
    return tuple(token for token in _raw_tokens(text) if token not in ANCHOR_STOPWORDS)


@lru_cache(maxsize=4096)
def tfidf_tokens(text: str) -> tuple[str, ...]:
    """Return canonicalized content terms for deterministic TF-IDF scoring."""
    return tuple(
        _canonical_term(token) for token in _raw_tokens(text) if token not in TFIDF_STOPWORDS
    )


def anchor_tokens(text: str) -> tuple[str, ...]:
    """Return conservative canonical terms plus local polarity anchors."""
    tokens = tuple(_canonical_term(token) for token in content_tokens(text))
    anchors = list(tokens)
    for index, token in enumerate(tokens):
        if token in NEGATIONS:
            anchors.extend(f"NEG::{following}" for following in tokens[index + 1 : index + 4])
    return tuple(anchors)


def _anchor_weight(token: str) -> float:
    plain = token.removeprefix("NEG::")
    if token.startswith("NEG::") or plain in NEGATIONS or plain.replace(".", "").isdigit():
        return 2.0
    if plain in LATERALITY or plain in TIMING or plain.endswith("%"):
        return 1.5
    return 1.0


def _smooth_idf(document_count: int, document_frequency: int) -> float:
    if document_count < 1 or not 0 <= document_frequency <= document_count:
        raise ValueError("invalid TF-IDF document frequency")
    return math.log((document_count + 1.0) / (document_frequency + 1.0)) + 1.0


def measurement_stem(stem: str) -> str:
    """Remove source item numbering that is presentation metadata, not stem content."""
    return SOURCE_ITEM_PREFIX_RE.sub("", stem, count=1)


def _reference_document(
    questions: Iterable[Question],
    *,
    document_count: int,
    document_frequency: Mapping[str, int],
    reference_scope: str,
    source: JsonObject,
) -> JsonObject:
    """Freeze the corpus frequencies needed by the selected stem terms."""
    materialized = list(questions)
    if not materialized:
        raise ValueError("TF-IDF reference requires at least one question")
    selected_terms = sorted(
        {
            term
            for question in materialized
            for term in tfidf_tokens(measurement_stem(question.stem))
        }
    )
    frequencies = {term: int(document_frequency.get(term, 0)) for term in selected_terms}
    return json_object(
        {
            "lexical_version": LEXICAL_VERSION,
            "reference_scope": reference_scope,
            "document_count": document_count,
            "idf_formula": IDF_FORMULA,
            "idf_cap": IDF_CAP,
            "sublinear_term_frequency": True,
            "document_frequency": frequencies,
            "high_idf_threshold": HIGH_IDF_THRESHOLD,
            "selected_terms_hash": sha256_json(selected_terms),
            "source": source,
            "matching": {
                "unicode_normalization": "NFKC+casefold",
                "leading_source_item_number_ignored": True,
                "conservative_inflection": True,
                "possessive_normalization": True,
                "hyphen_as_boundary": True,
                "validated_abbreviation_version": ABBREVIATION_VERSION,
                "validated_abbreviations": sorted(ABBREVIATIONS),
                "explicit_parenthetical_abbreviations": True,
                "arbitrary_initialism_inference": False,
                "embedding_or_prefix_matching": False,
            },
            "anchor_weighting": {
                "ordinary": 1.0,
                "laterality_or_timing": 1.5,
                "numeric_or_polarity": 2.0,
            },
        },
        path="lexical reference",
    )


def build_reference(questions: Iterable[Question]) -> JsonObject:
    """Build an explicit fixture reference for unit tests and fake runs."""
    materialized = list(questions)
    selected_terms = {
        term for question in materialized for term in tfidf_tokens(measurement_stem(question.stem))
    }
    document_count, frequencies = fixture_document_frequencies(selected_terms)
    return _reference_document(
        materialized,
        document_count=document_count,
        document_frequency=frequencies,
        reference_scope="deterministic_test_fixture",
        source={"kind": "fixture", "scientific_use": False},
    )


def selected_terms_hash(questions: Iterable[Question]) -> str:
    """Return the identity of exact canonical terms required by a question set."""
    terms = sorted(
        {term for question in questions for term in tfidf_tokens(measurement_stem(question.stem))}
    )
    return sha256_json(terms)


def build_pubmed_reference(
    questions: Iterable[Question],
    vocabulary_path: Path,
) -> JsonObject:
    """Build the production reference from immutable global PubMed frequencies."""
    materialized = list(questions)
    selected_terms = {
        term for question in materialized for term in tfidf_tokens(measurement_stem(question.stem))
    }
    document_count, frequencies = read_document_frequencies(vocabulary_path, selected_terms)
    return _reference_document(
        materialized,
        document_count=document_count,
        document_frequency=frequencies,
        reference_scope="global_pubmed_abstracts_2010_2024",
        source={
            "kind": "published_document_frequency_table",
            "name": PUBMED_REFERENCE_NAME,
            "version": PUBMED_REFERENCE_VERSION,
            "commit": PUBMED_COMMIT,
            "url": PUBMED_REFERENCE_URL,
            "sha256": PUBMED_EXPECTED_SHA256,
        },
    )


def compile_reference(reference: Mapping[str, object]) -> LexicalReference:
    """Validate a persisted reference once and compile it for repeated scoring."""
    if reference.get("lexical_version") != LEXICAL_VERSION:
        raise ValueError("lexical reference version does not match the current scorer")
    document_count = int(number_value(reference.get("document_count"), name="document_count"))
    raw_frequencies = object_value(reference.get("document_frequency"), name="document_frequency")
    frequencies: dict[str, int] = {}
    for term, raw_frequency in raw_frequencies.items():
        if isinstance(raw_frequency, bool) or not isinstance(raw_frequency, int):
            raise ValueError(f"document frequency for {term!r} is not an integer")
        if not 0 <= raw_frequency <= document_count:
            raise ValueError(f"document frequency for {term!r} is outside the corpus range")
        frequencies[term] = raw_frequency
    high = number_value(reference.get("high_idf_threshold"), name="high_idf_threshold")
    idf_cap = number_value(reference.get("idf_cap"), name="idf_cap")
    return LexicalReference(
        document_count=document_count,
        document_frequency=MappingProxyType(frequencies),
        high_idf_threshold=high,
        idf_cap=idf_cap,
    )


@cache
def _stem_tfidf_counts(stem: str) -> tuple[tuple[str, int], ...]:
    return tuple(Counter(tfidf_tokens(measurement_stem(stem))).items())


def _term_idf(term: str, reference: LexicalReference) -> float:
    return min(
        _smooth_idf(reference.document_count, reference.document_frequency.get(term, 0)),
        reference.idf_cap,
    )


def _contains_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    if not phrase or len(phrase) > len(tokens):
        return False
    width = len(phrase)
    target = tuple(phrase)
    return any(
        tuple(tokens[index : index + width]) == target for index in range(len(tokens) - width + 1)
    )


@lru_cache(maxsize=4096)
def _explicit_abbreviation_phrases(text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return only abbreviation pairs explicitly written as ``long form (ABC)``.

    The parenthetical notation supplies the semantic pairing. Initials are checked
    against the shortest matching suffix so unrelated preceding prose is never
    incorporated into the expansion.
    """
    pairs: dict[str, tuple[str, ...]] = {}
    for match in EXPLICIT_ABBREVIATION_RE.finditer(unicodedata.normalize("NFKC", text)):
        abbreviation = match.group("short")
        words = re.findall(r"[A-Za-z0-9]+", match.group("long"))
        for width in range(1, min(8, len(words)) + 1):
            window = words[-width:]
            initials = "".join(word[0].upper() for word in window if word)
            if initials != abbreviation:
                continue
            canonical = tuple(
                _canonical_term(token)
                for word in window
                if (token := normalize_token(word.casefold()))
                and len(token) > 1
                and token not in TFIDF_STOPWORDS
            )
            if canonical:
                pairs[abbreviation.casefold()] = canonical
            break
    return tuple(sorted(pairs.items()))


def _validated_abbreviations(reference_text: str) -> dict[str, tuple[str, ...]]:
    """Combine the reviewed dictionary with pairs explicitly supplied in the stem."""
    result = dict(ABBREVIATIONS)
    result.update(_explicit_abbreviation_phrases(reference_text))
    return result


def validated_abbreviations(reference_text: str) -> dict[str, tuple[str, ...]]:
    """Expose the validated abbreviation map for reproducibility-audit rendering."""
    return _validated_abbreviations(reference_text)


def _abbreviation_credit(reference_text: str, candidate_text: str) -> Counter[str]:
    """Credit only reviewed or explicitly paired abbreviations."""
    reference_tokens = anchor_tokens(reference_text)
    candidate_tokens = anchor_tokens(candidate_text)
    candidate_acronyms = {item.casefold() for item in ACRONYM_RE.findall(candidate_text)}
    credit: Counter[str] = Counter()
    for abbreviation, expansion in _validated_abbreviations(reference_text).items():
        abbreviation_term = _canonical_term(abbreviation)
        expansion_terms = tuple(_canonical_term(token) for token in expansion)
        if abbreviation in candidate_acronyms and _contains_phrase(
            reference_tokens, expansion_terms
        ):
            credit.update(expansion_terms)
        if abbreviation_term in reference_tokens and _contains_phrase(
            candidate_tokens, expansion_terms
        ):
            credit[abbreviation_term] += 1
    return credit


def _candidate_counts(
    candidate: str,
    stem_tokens: Sequence[str],
    abbreviations: Mapping[str, tuple[str, ...]],
) -> tuple[Counter[str], int]:
    raw = [token for token in _raw_tokens(candidate) if token not in TFIDF_STOPWORDS]
    canonical = [_canonical_term(token) for token in raw]
    counts = Counter(canonical)
    stem_terms = set(stem_tokens)
    for abbreviation, expansion in abbreviations.items():
        abbreviation_term = _canonical_term(abbreviation)
        expansion_terms = tuple(_canonical_term(token) for token in expansion)
        if abbreviation_term in stem_terms and _contains_phrase(canonical, expansion_terms):
            counts[abbreviation_term] += 1
        if abbreviation_term in canonical and _contains_phrase(stem_tokens, expansion_terms):
            for term in expansion_terms:
                counts[term] += 1
    return counts, len(canonical)


def candidate_counts(
    candidate: str,
    stem_tokens: Sequence[str],
    abbreviations: Mapping[str, tuple[str, ...]],
) -> tuple[Counter[str], int]:
    """Expose exact candidate-term counting for reproducibility-audit rendering."""
    return _candidate_counts(candidate, stem_tokens, abbreviations)


def _tfidf_weight(count: int, idf: float) -> float:
    return (1.0 + math.log(float(count))) * idf if count > 0 else 0.0


def _weighted_recall(
    stem_counts: Mapping[str, int],
    candidate_counts: Mapping[str, int],
    reference: LexicalReference,
    *,
    minimum_idf: float | None = None,
) -> tuple[float, float]:
    total = 0.0
    credited = 0.0
    for term, stem_count in stem_counts.items():
        idf = _term_idf(term, reference)
        if minimum_idf is not None and idf < minimum_idf:
            continue
        total += _tfidf_weight(stem_count, idf)
        credited += _tfidf_weight(min(stem_count, candidate_counts.get(term, 0)), idf)
    return (credited / total if total else 0.0, credited)


def _shared_tfidf_recall(
    stem_counts: Mapping[str, int],
    facts_counts: Mapping[str, int],
    implications_counts: Mapping[str, int],
    reference: LexicalReference,
    *,
    minimum_idf: float | None = None,
) -> float:
    total = 0.0
    credited = 0.0
    for term, stem_count in stem_counts.items():
        idf = _term_idf(term, reference)
        if minimum_idf is not None and idf < minimum_idf:
            continue
        total += _tfidf_weight(stem_count, idf)
        shared_count = min(
            stem_count,
            facts_counts.get(term, 0),
            implications_counts.get(term, 0),
        )
        credited += _tfidf_weight(shared_count, idf)
    return credited / total if total else 0.0


def _anchor_recall(stem: str, output: str) -> float:
    stem_counts = Counter(anchor_tokens(stem))
    output_counts = Counter(anchor_tokens(output))
    reference_credit = _abbreviation_credit(stem, output)
    stem_terms = set(stem_counts)
    for term, frequency in reference_credit.items():
        if term in stem_terms:
            output_counts[term] += frequency
    total = 0.0
    credited = 0.0
    for token, frequency in stem_counts.items():
        weight = _anchor_weight(token) * frequency
        total += weight
        credited += _anchor_weight(token) * min(frequency, output_counts.get(token, 0))
    return credited / total if total else 0.0


def score_anchor_recall(stem: str, output: str) -> float:
    """Return the independent polarity-aware lexical anchor recall diagnostic."""
    return _anchor_recall(measurement_stem(stem), output)


def _shared_anchor_recall(stem: str, facts: str, implications: str) -> float:
    stem_counts = Counter(anchor_tokens(measurement_stem(stem)))
    facts_counts = Counter(anchor_tokens(facts))
    implications_counts = Counter(anchor_tokens(implications))
    total = 0.0
    credited = 0.0
    for token, frequency in stem_counts.items():
        total += _anchor_weight(token) * frequency
        shared = min(frequency, facts_counts.get(token, 0), implications_counts.get(token, 0))
        credited += _anchor_weight(token) * shared
    return credited / total if total else 0.0


def score_preanswer(
    stem: str,
    facts: str,
    implications: str,
    reference: LexicalReference,
) -> dict[str, float]:
    """Measure TF-IDF lexical exposure and polarity-aware anchor preservation."""
    scored_stem = measurement_stem(stem)
    stem_tokens = tfidf_tokens(scored_stem)
    stem_counts = dict(_stem_tfidf_counts(scored_stem))
    abbreviations = _validated_abbreviations(scored_stem)
    facts_counts, facts_token_count = _candidate_counts(facts, stem_tokens, abbreviations)
    implications_counts, implications_token_count = _candidate_counts(
        implications, stem_tokens, abbreviations
    )
    preanswer_counts = facts_counts + implications_counts

    facts_tfidf, _facts_credit = _weighted_recall(stem_counts, facts_counts, reference)
    implications_tfidf, _implications_credit = _weighted_recall(
        stem_counts, implications_counts, reference
    )
    preanswer_tfidf, preanswer_credit = _weighted_recall(stem_counts, preanswer_counts, reference)
    high_idf, _high_credit = _weighted_recall(
        stem_counts,
        preanswer_counts,
        reference,
        minimum_idf=reference.high_idf_threshold,
    )
    preanswer_token_count = facts_token_count + implications_token_count
    density = 100.0 * preanswer_credit / preanswer_token_count if preanswer_token_count else 0.0
    return {
        "facts_tfidf_recall": facts_tfidf,
        "implications_tfidf_recall": implications_tfidf,
        "preanswer_tfidf_recall": preanswer_tfidf,
        "preanswer_high_idf_tfidf_recall": high_idf,
        "facts_implications_shared_tfidf_recall": _shared_tfidf_recall(
            stem_counts,
            facts_counts,
            implications_counts,
            reference,
        ),
        "facts_implications_shared_high_idf_recall": _shared_tfidf_recall(
            stem_counts,
            facts_counts,
            implications_counts,
            reference,
            minimum_idf=reference.high_idf_threshold,
        ),
        "preanswer_tfidf_density_per_100_tokens": density,
        "preanswer_tfidf_token_count": float(preanswer_token_count),
        "facts_anchor_recall": _anchor_recall(scored_stem, facts),
        "implications_anchor_recall": _anchor_recall(scored_stem, implications),
        "preanswer_anchor_recall": _anchor_recall(scored_stem, f"{facts} {implications}"),
        "shared_anchor_recall": _shared_anchor_recall(scored_stem, facts, implications),
    }
