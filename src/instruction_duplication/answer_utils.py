"""Strict benchmark normalization and conservative answer extraction."""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

from .json_types import object_mapping, object_sequence

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABEL_PATTERN = r"[A-Z]"
ANSWER_CUE_PATTERN = (
    r"(?:final\s+answer|answer|best\s+(?:answer|choice)|correct\s+(?:answer|choice|option)|"
    r"false\s+(?:statement|answer)|diagnosis|response)"
)
ANSWER_COPULA_PATTERN = r"(?:is|would\s+be|appears\s+to\s+be|seems\s+to\s+be)?\s*(?:[:=]\s*)?"
EXPLICIT_OPTION_LABEL_RE = re.compile(
    rf"\b{ANSWER_CUE_PATTERN}\s*{ANSWER_COPULA_PATTERN}"
    rf"option\s*[\[(]?({LABEL_PATTERN})[\])]?(?=\b|\s|[.,;:!?<])",
    re.IGNORECASE,
)
EXPLICIT_BARE_LABEL_RE = re.compile(
    rf"\b{ANSWER_CUE_PATTERN}\s*{ANSWER_COPULA_PATTERN}"
    rf"[\[(]?((?-i:[A-Z]))[\])]?(?=\b|\s|[.,;:!?<])",
    re.IGNORECASE,
)
EXPLICIT_LOWER_BARE_LABEL_RE = re.compile(
    rf"\b{ANSWER_CUE_PATTERN}\s*{ANSWER_COPULA_PATTERN}"
    r"(?:\(((?-i:[a-z]))\)|((?-i:[a-z]))[.)]|((?-i:[a-z]))\s*$)",
    re.IGNORECASE,
)
FINAL_ANSWER_CUE_RE = re.compile(r"\bfinal\s+answer\b", re.IGNORECASE)
LINE_DECISION_CUE_RE = re.compile(
    r"(?im)^\s*(?:(?:therefore|thus|hence)\s*,?\s*|conclusion\s*:\s*)?"
    r"(?:the\s+)?(?:answer|best\s+(?:answer|choice)|correct\s+(?:answer|choice|option)|"
    r"false\s+(?:statement|answer)|most\s+(?:appropriate|accurate|likely)\s+"
    r"(?:answer|choice|statement|option|diagnosis|management|treatment))\b"
)
BOXED_RE = re.compile(rf"\\boxed\s*\{{\s*\(?({LABEL_PATTERN})\)?\s*\}}", re.IGNORECASE)
XMLISH_OPTION_ATTR_RE = re.compile(
    rf"<(?:answer|choice|correct_answer)\b[^<>]*\boption\s*=\s*[\"']?"
    rf"({LABEL_PATTERN})[\"']?[^<>]*>",
    re.IGNORECASE,
)
XMLISH_LABEL_BODY_RE = re.compile(
    rf"<(?:answer|correct_answer)>\s*\(?({LABEL_PATTERN})\)?\s*</(?:answer|correct_answer)>",
    re.IGNORECASE,
)
TERMINAL_LABEL_RE = re.compile(
    rf"(?:^|\n)\s*(?:final\s+answer\s*[:=]?\s*)?[\[(]?({LABEL_PATTERN})[\])]?[.)]?\s*$",
    re.IGNORECASE,
)
TERMINAL_OPTION_LABEL_RE = re.compile(
    rf"(?:^|\n)\s*(?:option|choice)\s*[\[(]?({LABEL_PATTERN})[\])]?[.)]?\s*$",
    re.IGNORECASE,
)
LEADING_LABELED_CHOICE_RE = re.compile(
    rf"^\s*(?:option\s*)?[\[(]?({LABEL_PATTERN})[\])]?[.):\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
TRAILING_LABELED_CHOICE_RE = re.compile(
    r"\b((?-i:[A-Z]))[.):\-]\s*(.+?)\s*$",
    re.DOTALL,
)


class ChoiceNormalizationError(ValueError):
    """Raised when a choice encoding is ambiguous or lossy."""


def normalize_text(text: str) -> str:
    """Normalize Unicode text without discarding non-English letters."""
    normalized = (
        unicodedata.normalize("NFKC", str(text))
        .casefold()
        .replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
    )
    normalized = re.sub(r"\b(?:option|choice|answer)\s+[a-z]\b", " ", normalized)
    characters = [
        character if character.isalnum() or unicodedata.category(character).startswith("L") else " "
        for character in normalized
    ]
    return " ".join("".join(characters).split())


def _maybe_deserialize(value: object) -> object:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{(\"'":
        return value
    try:
        parsed_json: object = json.loads(text)
    except (ValueError, json.JSONDecodeError, TypeError):
        parsed_json = value
    if parsed_json != value:
        return parsed_json
    try:
        parsed_literal: object = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        return value
    if parsed_literal != value:
        return parsed_literal
    return value


def _flatten(value: object) -> list[object]:
    value = _maybe_deserialize(value)
    mapping = object_mapping(value)
    if mapping is not None:
        truthy: list[object] = [
            key
            for key, item in mapping.items()
            if item is True or str(item).strip().casefold() in {"true", "1", "yes"}
        ]
        if truthy:
            return truthy
        result: list[object] = []
        for key in (
            "label",
            "labels",
            "answer",
            "answers",
            "correct",
            "correct_answer",
            "text",
            "value",
        ):
            if key in mapping:
                result.extend(_flatten(mapping[key]))
        return result or list(mapping.values())
    sequence = object_sequence(value)
    if sequence is not None:
        return [candidate for item in sequence for candidate in _flatten(item)]
    return [value]


def _parse_label(value: object, *, numeric_zero_based: bool | None = None) -> str | None:
    text = str(value).strip()
    letter = re.fullmatch(
        rf"(?:OPTION|CHOICE|ANSWER)?[_\s-]*({LABEL_PATTERN})(?:[).:]?)",
        text,
        re.IGNORECASE,
    )
    if letter:
        return letter.group(1).upper()
    numbered = re.fullmatch(
        r"(?:OPTION|CHOICE|ANSWER)[_\s-]*(\d+)(?:[).:]?)",
        text,
        re.IGNORECASE,
    )
    if numbered:
        number = int(numbered.group(1)) - 1
        return LABELS[number] if 0 <= number < len(LABELS) else None
    if re.fullmatch(r"\d+", text) and numeric_zero_based is not None:
        number = int(text) if numeric_zero_based else int(text) - 1
        return LABELS[number] if 0 <= number < len(LABELS) else None
    return None


def _add_choice(result: dict[str, str], label: object, text: object) -> None:
    if text is None:
        return
    parsed_label = _parse_label(label)
    value = " ".join(str(text).split())
    if parsed_label is None or not value:
        return
    if parsed_label in result:
        raise ChoiceNormalizationError(f"duplicate or colliding choice label {parsed_label}")
    result[parsed_label] = value


def _ordered(result: Mapping[str, str]) -> dict[str, str]:
    ordered = dict(sorted(result.items()))
    expected = list(LABELS[: len(ordered)])
    if list(ordered) != expected:
        raise ChoiceNormalizationError(
            f"choice labels must resolve to contiguous A..{expected[-1] if expected else 'A'}"
        )
    normalized = [normalize_text(text) for text in ordered.values()]
    if any(not text for text in normalized) or len(normalized) != len(set(normalized)):
        raise ChoiceNormalizationError("choice texts are empty or duplicate after normalization")
    return ordered


def canonicalize_choices(value: object) -> dict[str, str]:
    """Normalize common benchmark option encodings without silent overwrites."""
    value = _maybe_deserialize(value)
    result: dict[str, str] = {}
    mapping = object_mapping(value)
    if mapping is not None:
        if "label" in mapping and "text" in mapping:
            labels = object_sequence(mapping["label"])
            texts = object_sequence(mapping["text"])
            if labels is None or texts is None:
                raise ChoiceNormalizationError("parallel label/text fields must both be arrays")
            if len(labels) != len(texts):
                raise ChoiceNormalizationError("parallel label/text arrays have different lengths")
            for parallel_label, parallel_text in zip(labels, texts, strict=True):
                _add_choice(result, parallel_label, parallel_text)
            return _ordered(result)

        keys = [str(key).strip() for key in mapping]
        numeric = [int(key) for key in keys if key.isdigit()]
        zero_based = bool(numeric and 0 in numeric)
        for index, (key, mapped_text) in enumerate(mapping.items()):
            mapped_label = _parse_label(key, numeric_zero_based=zero_based)
            if mapped_label is None:
                mapped_label = LABELS[index]
            _add_choice(result, mapped_label, mapped_text)
        return _ordered(result)

    sequence = object_sequence(value)
    if sequence is not None:
        for index, item in enumerate(sequence):
            item_mapping = object_mapping(item)
            if item_mapping is not None:
                label: object = next(
                    (
                        item_mapping[key]
                        for key in ("label", "option", "key", "id")
                        if key in item_mapping
                    ),
                    LABELS[index],
                )
                text: object = next(
                    (
                        item_mapping[key]
                        for key in ("text", "value", "answer", "content")
                        if key in item_mapping
                    ),
                    None,
                )
                _add_choice(result, label, text)
            else:
                item_text = " ".join(str(item).split())
                labeled = LEADING_LABELED_CHOICE_RE.match(item_text)
                _add_choice(
                    result,
                    labeled.group(1) if labeled else LABELS[index],
                    labeled.group(2) if labeled else item_text,
                )
        return _ordered(result)

    if isinstance(value, str):
        text = value.strip()
        matches = list(
            re.finditer(
                rf"(?:^|[\n;|]\s*|\s{{2,}})({LABEL_PATTERN})\s*[).:\-]\s*",
                text,
                re.IGNORECASE,
            )
        )
        if len(matches) >= 2:
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                _add_choice(result, match.group(1), text[match.end() : end].strip(" \t\n;|"))
            return _ordered(result)
        numbered = list(re.finditer(r"(?:^|\n)\s*\d+\s*[).:\-]\s*", text))
        if len(numbered) >= 2:
            for index, match in enumerate(numbered):
                end = numbered[index + 1].start() if index + 1 < len(numbered) else len(text)
                _add_choice(result, LABELS[index], text[match.end() : end].strip())
            return _ordered(result)
    return {}


def canonical_gold_labels(
    value: object,
    choices: Mapping[str, str],
    *,
    index_base: int | None = None,
) -> set[str]:
    """Resolve labels without inferring numeric-index conventions from field names."""
    if index_base not in (None, 0, 1):
        raise ValueError("index_base must be 0, 1, or None")
    labels = list(choices)
    normalized_choices = {label: normalize_text(text) for label, text in choices.items()}
    candidates: set[str] = set()
    for item in _flatten(value):
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, int) or (isinstance(item, str) and item.strip().isdigit()):
            if index_base is None:
                raise ValueError("numeric gold value requires an explicit index_base")
            offset = int(item) - index_base
            if 0 <= offset < len(labels):
                candidates.add(labels[offset])
            continue
        text = str(item).strip()
        if not text:
            continue
        direct = _parse_label(text)
        if direct is not None and direct in choices:
            candidates.add(direct)
            continue
        if re.fullmatch(
            rf"\s*(?:option\s*)?{LABEL_PATTERN}(?:\s*[,;/|]\s*(?:option\s*)?{LABEL_PATTERN})+\s*",
            text,
            re.IGNORECASE,
        ):
            candidates.update(
                match.group(1).upper()
                for match in re.finditer(rf"(?:option\s*)?({LABEL_PATTERN})", text, re.IGNORECASE)
                if match.group(1).upper() in choices
            )
            continue
        norm = normalize_text(text)
        exact_text = {label for label, choice in normalized_choices.items() if norm == choice}
        if exact_text:
            candidates.update(exact_text)
            continue
        prefixed = LEADING_LABELED_CHOICE_RE.match(text)
        if prefixed:
            label = prefixed.group(1).upper()
            if label in choices and normalize_text(prefixed.group(2)) == normalized_choices[label]:
                candidates.add(label)
    return candidates


@dataclass(frozen=True, slots=True)
class ExtractedAnswer:
    """A final answer and its deterministic extraction diagnostics."""

    option: str | None
    status: str
    evidence: str | None
    candidates: tuple[str, ...]


type AnswerCandidate = tuple[int, str, str]


def _choice_text_at_start(text: str, choices: Mapping[str, str]) -> str | None:
    normalized = normalize_text(text)
    matches: list[str] = []
    for label, choice in choices.items():
        choice_norm = normalize_text(choice)
        if choice_norm and (
            normalized == choice_norm
            or normalized.startswith((choice_norm + " because", choice_norm + " due to"))
        ):
            matches.append(label)
    return matches[0] if len(matches) == 1 else None


def _strip_markdown_presentation(text: str) -> str:
    """Remove common Markdown decoration without changing answer-bearing text."""
    lines = [re.sub(r"^\s{0,3}#{1,6}\s*", "", line) for line in text.splitlines()]
    cleaned = "\n".join(lines)
    return re.sub(r"(?:\*\*|__|~~|`+)", "", cleaned)


def _answer_region(text: str) -> tuple[str, bool]:
    decision_cues = [
        *FINAL_ANSWER_CUE_RE.finditer(text),
        *LINE_DECISION_CUE_RE.finditer(text),
    ]
    region_start = max((match.start() for match in decision_cues), default=0)
    return text[region_start:], bool(decision_cues)


def _explicit_answer_candidates(region: str, choices: Mapping[str, str]) -> list[AnswerCandidate]:
    candidates: list[AnswerCandidate] = []
    for match in EXPLICIT_LOWER_BARE_LABEL_RE.finditer(region):
        raw_label = next(group for group in match.groups() if group is not None)
        label = raw_label.upper()
        if label in choices:
            candidates.append((match.start(), label, match.group(0)))

    for pattern in (
        EXPLICIT_OPTION_LABEL_RE,
        EXPLICIT_BARE_LABEL_RE,
        BOXED_RE,
        XMLISH_OPTION_ATTR_RE,
        XMLISH_LABEL_BODY_RE,
    ):
        for match in pattern.finditer(region):
            label = match.group(1).upper()
            if label in choices:
                candidates.append((match.start(), label, match.group(0)))
    return candidates


def _decision_line_candidates(region: str, choices: Mapping[str, str]) -> list[AnswerCandidate]:
    candidates: list[AnswerCandidate] = []
    offset = 0
    for line in region.splitlines(keepends=True):
        visible = line.strip()
        labeled_line = LEADING_LABELED_CHOICE_RE.match(visible)
        if labeled_line is not None:
            label = labeled_line.group(1).upper()
            if label in choices:
                body = normalize_text(labeled_line.group(2))
                expected = normalize_text(choices[label])
                if body == expected or body.startswith(expected + " "):
                    candidates.append((offset, label, visible))
        offset += len(line)
    return candidates


def _last_line_candidates(region: str, choices: Mapping[str, str]) -> list[AnswerCandidate]:
    last_nonempty = next(
        (line.strip() for line in reversed(region.splitlines()) if line.strip()),
        "",
    )
    visible = last_nonempty.strip("*` ")
    labeled = LEADING_LABELED_CHOICE_RE.match(visible)
    if labeled and labeled.group(1).upper() in choices:
        label = labeled.group(1).upper()
        body = normalize_text(labeled.group(2))
        expected = normalize_text(choices[label])
        if not body or body == expected or body.startswith(expected + " because"):
            return [(region.rfind(last_nonempty), label, last_nonempty)]
        return []

    text_label = _choice_text_at_start(visible, choices)
    if text_label:
        return [(region.rfind(last_nonempty), text_label, last_nonempty)]

    trailing = TRAILING_LABELED_CHOICE_RE.search(last_nonempty)
    if trailing is None or trailing.group(1) not in choices:
        return []
    label = trailing.group(1)
    body = normalize_text(trailing.group(2))
    expected = normalize_text(choices[label])
    if body != expected and not body.startswith(expected + " "):
        return []
    return [(region.rfind(last_nonempty) + trailing.start(), label, trailing.group(0))]


def _terminal_label_candidates(region: str, choices: Mapping[str, str]) -> list[AnswerCandidate]:
    candidates: list[AnswerCandidate] = []
    for pattern in (TERMINAL_LABEL_RE, TERMINAL_OPTION_LABEL_RE):
        match = pattern.search(region)
        if match and match.group(1).upper() in choices:
            candidates.append((match.start(), match.group(1).upper(), match.group(0)))
    return candidates


def _resolve_answer_candidates(candidates: list[AnswerCandidate]) -> ExtractedAnswer:
    if not candidates:
        return ExtractedAnswer(None, "unparseable", None, ())
    candidates.sort(key=lambda item: item[0])
    final_labels = {item[1] for item in candidates}
    if len(final_labels) > 1:
        return ExtractedAnswer(
            None,
            "conflicting",
            candidates[-1][2],
            tuple(sorted(final_labels)),
        )
    selected = candidates[-1]
    return ExtractedAnswer(
        selected[1],
        "parsed",
        selected[2],
        tuple(sorted(final_labels)),
    )


def extract_answer(
    text: str,
    choices: Mapping[str, str],
    *,
    allow_terminal_label: bool = True,
) -> ExtractedAnswer:
    """Extract an answer from an explicit final region or concise terminal line."""
    stripped = _strip_markdown_presentation(text).strip()
    if not stripped:
        return ExtractedAnswer(None, "unparseable", None, ())

    region, has_decision_cue = _answer_region(stripped)
    candidates = _explicit_answer_candidates(region, choices)
    if has_decision_cue:
        candidates.extend(_decision_line_candidates(region, choices))
    candidates.extend(_last_line_candidates(region, choices))
    if allow_terminal_label:
        candidates.extend(_terminal_label_candidates(region, choices))
    return _resolve_answer_candidates(candidates)
