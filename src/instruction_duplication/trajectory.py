"""Format-neutral extraction of the visible reasoning trajectory.

Paper mapping: implements section recovery used by ``Experiment -> Measurements and
Eligibility``. The experiment asks for eight named headings without privileging a
presentation syntax; this module recognizes ordinary text, numbering, Markdown,
XML-like headings, and a frozen set of role-title aliases solely to recover section
boundaries for the deterministic measurements."""

from __future__ import annotations

import html
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .answer_utils import normalize_text
from .protocol import SECTION_GUIDANCE

CONTENT_TAGS = (
    "facts",
    "implications",
    "provisional_answer",
    "second_best",
    "decisive_fact",
    "answer_changing_change",
    "rereasoning",
    "final_answer",
)
VALID_DECISIONS = {"retain", "revise"}

OUTER_CODE_FENCE_RE = re.compile(
    r"\A```(?:xml|markdown|md|text)?[ \t]*\r?\n(?P<document>.*)\r?\n```\Z",
    re.IGNORECASE | re.DOTALL,
)
OPTION_ATTR_RE = re.compile(r"\boption\s*=\s*['\"]\s*([A-Z])\s*['\"]", re.IGNORECASE)
DECISION_ATTR_RE = re.compile(
    r"\bdecision\s*=\s*['\"]\s*([A-Za-z]+)\s*['\"]",
    re.IGNORECASE,
)
XML_TAG_RE = re.compile(
    r"</?(?:[^\W\d]|_)[\w.:-]*(?:\s+[^<>]*?)?\s*/?>",
    re.DOTALL | re.UNICODE,
)
MARKDOWN_FENCE_RE = re.compile(r"```(?:xml|markdown|md|text)?", re.IGNORECASE)
START_TAG_PATTERNS = {
    tag: re.compile(rf"<{tag}\b(?P<attrs>[^<>]*)>", re.IGNORECASE) for tag in CONTENT_TAGS
}

_SEMANTIC_LABELS = {
    "facts": r"(?:case[ _-]+|stem[ _-]+)?facts?",
    "implications": r"implications?|interpretation",
    "provisional_answer": r"provisional[ _-]+answer|initial[ _-]+answer",
    "second_best": (r"best[ _-]+alternative|second[ _-]+best(?:[ _-]+answer)?|runner[ _-]*up"),
    "decisive_fact": r"decisive[ _-]+distinction|decisive[ _-]+fact|key[ _-]+distinction",
    "answer_changing_change": (
        # Conservative unnumbered synonyms. These all explicitly describe an
        # answer-changing/counterfactual role; vague "what would change?" is handled
        # only by the numbered Step-6 fallback below.
        r"answer[ _-]+changing[ _-]+(?:change|scenario)|"
        r"answer[ _-]+change|"
        r"counterfactual(?:[ _-]+(?:change|scenario|test))?|"
        r"what[ _-]+would[ _-]+change[ _-]+(?:the[ _-]+)?(?:answer|question)"
        r"(?:[ _-]+to[ _-]+(?:(?:(?:option|choice)[ _-]+)?[A-Z]"
        r"(?:[ _-]*\([^\)\r\n]{1,60}\))?|[^()\r\n:]{1,60}[ _-]*\([A-Z]\)))?|"
        r"what[ _-]+(?:would[ _-]+need[ _-]+to|needs?[ _-]+to)[ _-]+change|"
        r"what[ _-]+change[ _-]+would[ _-]+(?:change|alter|switch)[ _-]+"
        r"(?:the[ _-]+)?answer(?:[ _-]+to[ _-]+(?:(?:option|choice)[ _-]+)?[A-Z])?|"
        r"how[ _-]+(?:could|would)[ _-]+(?:the[ _-]+)?answer[ _-]+change|"
        r"what[ _-]+would[ _-]+make[ _-]+(?:the[ _-]+)?"
        r"(?:best[ _-]+alternative|second[ _-]+best(?:[ _-]+answer)?|alternative|"
        r"(?:(?:option|choice)[ _-]+)?[A-Z])"
        r"[ _-]+(?:the[ _-]+)?(?:best[ _-]+answer|best|correct|win)|"
        r"how[ _-]+to[ _-]+make[ _-]+(?:the[ _-]+)?"
        r"(?:best[ _-]+alternative|second[ _-]+best(?:[ _-]+answer)?|alternative|"
        r"(?:(?:option|choice)[ _-]+)?[A-Z])[ _-]+(?:best|correct|win)|"
        r"(?:smallest[ _-]+)?change[ _-]+(?:needed|required)[ _-]+"
        r"(?:to[ _-]+(?:change|alter|switch)[ _-]+(?:the[ _-]+)?answer|"
        r"for[ _-]+(?:the[ _-]+)?"
        r"(?:best[ _-]+alternative|second[ _-]+best(?:[ _-]+answer)?|alternative|"
        r"(?:(?:option|choice)[ _-]+)?[A-Z])"
        r"[ _-]+to[ _-]+(?:win|be[ _-]+correct|become[ _-]+best))|"
        r"change[ _-]+that[ _-]+would[ _-]+make[ _-]+(?:the[ _-]+)?"
        r"(?:best[ _-]+alternative|second[ _-]+best(?:[ _-]+answer)?|alternative|"
        r"(?:(?:option|choice)[ _-]+)?[A-Z])[ _-]+(?:win|correct|best)|"
        r"conditions?[ _-]+for[ _-]+(?:the[ _-]+)?"
        r"(?:best[ _-]+alternative|second[ _-]+best(?:[ _-]+answer)?|alternative|"
        r"(?:(?:option|choice)[ _-]+)?[A-Z])"
        r"[ _-]+to[ _-]+(?:win|be[ _-]+correct|become[ _-]+best)|"
        r"when[ _-]+would[ _-]+(?:the[ _-]+)?"
        r"(?:best[ _-]+alternative|second[ _-]+best(?:[ _-]+answer)?|alternative|"
        r"(?:(?:option|choice)[ _-]+)?[A-Z])"
        r"[ _-]+(?:win|be[ _-]+correct|become[ _-]+best)"
    ),
    "rereasoning": (
        r"reconsideration|re[ _-]?reasoning|rereasoning|re[ _-]?evaluation|final[ _-]+check"
    ),
    "final_answer": r"final[ _-]+answer",
}
SEMANTIC_HEADING_PATTERNS = {
    tag: re.compile(
        rf"(?im)^[ \t]*(?:\#{{1,6}}[ \t]*)?(?:[-+*][ \t]+)?"
        rf"(?:\*\*|__)?"
        rf"(?:(?:step[ \t]*)?{index}[.:)\-][ \t]*)?"
        rf"(?:\*\*|__)?(?:{_SEMANTIC_LABELS[tag]})(?:\?)?(?:\*\*|__)?[ \t]*"
        rf"(?:[:\u2014\u2013-][ \t]*|(?=\r?$))"
    )
    for index, tag in enumerate(CONTENT_TAGS, start=1)
}

# Some models preserve the requested role number while paraphrasing its title.
# Because explicit "6." / "Step 6" is itself a strong boundary signal, this
# fallback may safely accept a somewhat wider family than the unnumbered aliases.
# Body prose without an explicit Step-6 marker is not recovered by this fallback.
STEP6_NUMBERED_HEADING_FALLBACK_RE = re.compile(
    r"(?im)^[ \t]*(?:\#{1,6}[ \t]*)?(?:[-+*][ \t]+)?"
    r"(?:\*\*|__)?(?:step[ \t]*)?6[.:)\-][ \t]*(?:\*\*|__)?"
    r"(?:"
    r"what[ _-]+would[ _-]+change(?:[ _-]+(?:the[ _-]+)?(?:answer|question))?"
    r"(?:[ _-]+to[ _-]+[^:\r\n\u2014\u2013-]{1,80})?|"
    r"what[ _-]+(?:would[ _-]+need[ _-]+to|needs?[ _-]+to)[ _-]+change|"
    r"what[ _-]+would[ _-]+make[ _-]+[^:\r\n\u2014\u2013-]{1,80}|"
    r"how[ _-]+to[ _-]+make[ _-]+[^:\r\n\u2014\u2013-]{1,80}|"
    r"(?:smallest[ _-]+)?change[ _-]+(?:needed|required)"
    r"(?:[ _-]+[^:\r\n\u2014\u2013-]{1,80})?|"
    r"counterfactual(?:[ _-]+(?:change|scenario|test))?"
    r")"
    r"(?:\?)?(?:\*\*|__)?[ \t]*(?:[:\u2014\u2013-][ \t]*|(?=\r?$))"
)
CLOSING_TAG_PATTERNS = {
    tag: re.compile(
        rf"</\s*{tag}\b(?P<attrs>[^<>]*)>",
        re.IGNORECASE,
    )
    for tag in CONTENT_TAGS
}
SEMANTIC_OPTION_RES = (
    # A section body often starts directly with ``B. Choice text`` or ``- **B. ...**``.
    re.compile(
        r"(?m)^[ \t]*(?:[-+*][ \t]+)?(?:\*\*|__|`)?\s*(?-i:([A-Z]))\s*[.):\-]",
        re.IGNORECASE,
    ),
    re.compile(r"\boption\s*=\s*['\"]\s*([A-Z])\s*['\"]", re.IGNORECASE),
    # Context-specific selectors must precede the generic ``option B`` fallback;
    # otherwise phrases such as ``best alternative to option C could be B`` are
    # incorrectly captured as C.
    re.compile(
        r"\b(?:best\s+alternative|second[- ]best(?:\s+answer)?|runner[- ]?up)"
        r"(?:\s+to\s+(?:option|choice)\s*[A-Z])?\s*"
        r"(?:is|would\s+be|could\s+be|appears\s+to\s+be|seems\s+to\s+be|=|:)?\s*"
        r"[:=]?\s*[\"'(]*(?-i:([A-Z]))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:provisional|initial|final|best|correct|selected|most\s+likely|likely|preferred|favou?red)\s+"
        r"(?:answer|choice|option|diagnosis|treatment|management|response)\s*"
        r"(?:is|would\s+be|could\s+be|appears\s+to\s+be|seems\s+to\s+be|=|:)?\s*"
        r"[:=]?\s*[\"'(]*(?-i:([A-Z]))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:answer|choice|option)\s+selected\s*(?:is|=|:)?\s*[\"'(]*(?-i:([A-Z]))\b",
        re.IGNORECASE,
    ),
    re.compile(r"\\boxed\s*\{?\s*([A-Z])\s*\}?", re.IGNORECASE),
    # Last resort: an explicit option/choice label anywhere in the role body.
    re.compile(r"\b(?:option|choice)\s*[\"'(]*([A-Z])\b", re.IGNORECASE),
)

SEMANTIC_DECISION_RE = re.compile(
    r"\bdecision\s*=\s*['\"]\s*(retain|revise)\s*['\"]"
    r"|\bdecision\s+(?:is\s+)?(?:to\s+)?(retain|revise)\b"
    r"|\b(retain|revise)\b"
    r"|\b(retaining|revising|maintaining|retained|revised|maintained)\b"
    r"|\b(i|we)\s+(?:would\s+|will\s+|now\s+)?(retain|keep|maintain|revise|change|switch)\b"
    r"|\b(?:provisional\s+answer|answer|conclusion)[^\n]{0,140}?\b"
    r"(retained|kept|maintained|unchanged|revised|changed|switched|remains|remain|stands)\b"
    r"|\b(?:provisional\s+answer|initial\s+answer|initial\s+choice)[^\n]{0,140}?\b"
    r"(?:is\s+)?(still\s+best|still\s+correct|still\s+most\s+likely)\b"
    r"|\b(?:therefore|thus|overall|on\s+balance|after\s+reconsideration|upon\s+reconsideration)?[^\n]{0,80}?\b"
    r"(switching|changing|revising)\s+(?:the\s+)?(?:answer|choice|provisional\s+answer)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RecoveredProtocol:
    """Locally recovered protocol content, independent of serialization syntax."""

    document: str
    sections: Mapping[str, str]
    start_counts: Mapping[str, int]
    marker_schema_valid: Mapping[str, bool]
    semantic_start_counts: Mapping[str, int]
    ordered_starts: bool
    preanswer_ordered: bool
    semantic_ordered_starts: bool
    semantic_preanswer_ordered: bool
    provisional_option: str | None
    second_best_option: str | None
    final_option: str | None
    decision: str | None
    semantic_provisional_option: str | None
    semantic_second_best_option: str | None
    semantic_final_option: str | None
    semantic_decision: str | None
    provisional_start: int | None
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sections", MappingProxyType(dict(self.sections)))
        object.__setattr__(self, "start_counts", MappingProxyType(dict(self.start_counts)))
        object.__setattr__(
            self,
            "marker_schema_valid",
            MappingProxyType(dict(self.marker_schema_valid)),
        )
        object.__setattr__(
            self,
            "semantic_start_counts",
            MappingProxyType(dict(self.semantic_start_counts)),
        )


def normalize_outer_code_fence(raw: str) -> str:
    """Strip outer whitespace and one conventional whole-response code fence."""
    candidate = raw.strip()
    match = OUTER_CODE_FENCE_RE.fullmatch(candidate)
    return candidate if match is None else match.group("document").strip()


def _plain_text(value: str) -> str:
    """Flatten markup while preserving visible text and decoded entities."""
    without_tags = XML_TAG_RE.sub(" ", value)
    without_fences = MARKDOWN_FENCE_RE.sub(" ", without_tags)
    without_presentation = re.sub(r"(?:\*\*|__|~~|`+)", "", without_fences)
    return " ".join(html.unescape(without_presentation).split())


def _strip_guidance_echo(tag: str, value: str) -> str:
    """Remove an exact leading copy of the protocol's section description.

    Some models copy a section's natural-language requirement immediately after
    its marker and then provide the actual response. Treating that fixed prompt
    text as generated evidence would contaminate both substantive and lexical
    measurements, so only an exact normalized prefix is removed.
    """
    guidance = " ".join(SECTION_GUIDANCE[tag].split())
    if value == guidance:
        return ""
    prefix = guidance + " "
    return value[len(prefix) :].lstrip() if value.startswith(prefix) else value


def _first_match(
    matches: Mapping[str, tuple[re.Match[str], ...]], tag: str
) -> re.Match[str] | None:
    values = matches[tag]
    return values[0] if values else None


def _segment_end(
    document: str,
    matches: Mapping[str, tuple[re.Match[str], ...]],
    start: re.Match[str],
) -> int:
    """End a semantic section at the next visible role marker."""
    later_starts = [
        match.start()
        for role_matches in matches.values()
        for match in role_matches
        if match.start() > start.start()
    ]
    return min(later_starts) if later_starts else len(document)


def _option(match: re.Match[str] | None, choices: Mapping[str, str]) -> str | None:
    if match is None:
        return None
    attr = OPTION_ATTR_RE.search(match.group("attrs"))
    if attr is None:
        return None
    option = attr.group(1).upper()
    return option if option in choices else None


def _normalized_choice_tokens(
    label: str, choice: str, choices: Mapping[str, str]
) -> tuple[str, ...]:
    """Normalize benchmark choice text for semantic matching, including known export debris."""
    cleaned = _plain_text(choice)
    terminal = re.search(r"\s*\(([A-Z])\)\s*$", cleaned)
    if terminal is not None and terminal.group(1).upper() != label.upper():
        cleaned = cleaned[: terminal.start()].rstrip()
    tokens = normalize_text(cleaned).split()
    # Some MedQA exports in this corpus contain a literal terminal ``rn`` where a
    # line break was intended (e.g. ``Fludarabinern`` / ``Vincristinern``). Only
    # activate that repair when it is visibly a question-level encoding pattern.
    rn_artifact = sum(normalize_text(value).endswith("rn") for value in choices.values()) >= 2
    if rn_artifact:
        tokens = [
            token[:-2] if len(token) > 4 and token.endswith("rn") else token for token in tokens
        ]
    return tuple(token for token in tokens if token)


def _light_stem(token: str) -> str:
    """Very small lexical normalization for choice-text matching."""
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _choice_from_fragment(fragment: str, choices: Mapping[str, str]) -> str | None:
    """Resolve one uniquely indicated choice inside a short role-local fragment."""
    explicit: set[str] = set()
    for pattern in (
        # Bare option letters are conventionally uppercase. Keeping these two
        # patterns case-sensitive prevents abbreviations such as ``e.g.`` from
        # becoming a spurious option E.
        re.compile(r"(?<![A-Za-z])([A-Z])(?:\s*[.):\-]|\s+(?=\())"),
        re.compile(r"\(([A-Z])\)"),
        # Explicit option/choice words safely permit lower-case labels.
        re.compile(r"\b(?:option|choice)\s+([A-Z])\b", re.IGNORECASE),
    ):
        explicit.update(
            match.group(1).upper()
            for match in pattern.finditer(fragment)
            if match.group(1).upper() in choices
        )
    if len(explicit) == 1:
        return next(iter(explicit))

    normalized_fragment = normalize_text(fragment)
    exact: list[str] = []
    for label, choice in choices.items():
        tokens = _normalized_choice_tokens(label, choice, choices)
        if not tokens or set(tokens) <= {"n", "a"}:
            continue
        phrase = " ".join(tokens)
        if phrase and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized_fragment):
            exact.append(label)
    if len(exact) == 1:
        return exact[0]

    fragment_tokens = {_light_stem(token) for token in normalized_fragment.split()}
    scored: list[tuple[float, int, str]] = []
    for label, choice in choices.items():
        choice_tokens = {
            _light_stem(token) for token in _normalized_choice_tokens(label, choice, choices)
        }
        if not choice_tokens or choice_tokens <= {"n", "a"}:
            continue
        overlap = len(choice_tokens & fragment_tokens) / len(choice_tokens)
        scored.append((overlap, len(choice_tokens), label))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < 0.8:
        return None
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    # Short one-token answers need exact lexical identity; multi-token answers can
    # tolerate inflectional variation but must clearly beat the runner-up.
    if scored[0][1] == 1 and scored[0][0] < 1.0:
        return None
    if runner_up >= scored[0][0] - 0.15:
        return None
    return scored[0][2]


def _choice_from_salient_text(text: str, choices: Mapping[str, str]) -> str | None:
    """Resolve a choice from multiple choice-specific lexical anchors.

    Long benchmark choices are often paraphrased rather than copied verbatim.  This
    fallback stays purely lexical: it only uses tokens that occur in one answer
    choice and requires several such anchors to agree.  It therefore recognizes
    ``CD15 ... CD30`` for a choice whose wording is ``cells staining positive for
    CD15 and CD30`` without performing any medical synonym inference.  A role that
    mentions several competing choices does not pass unless one choice clearly
    dominates the other choices' unique-token evidence.
    """
    fragment_tokens = {_light_stem(token) for token in normalize_text(_plain_text(text)).split()}
    if not fragment_tokens:
        return None

    per_choice: dict[str, set[str]] = {}
    token_owners: dict[str, set[str]] = {}
    for label, choice in choices.items():
        tokens = {_light_stem(token) for token in _normalized_choice_tokens(label, choice, choices)}
        tokens = {
            token
            for token in tokens
            if token
            not in {
                "a",
                "an",
                "the",
                "of",
                "and",
                "or",
                "with",
                "for",
                "to",
                "in",
                "on",
                "is",
                "are",
                "cell",
                "answer",
                "choice",
                "option",
            }
            and (len(token) >= 3 or any(ch.isdigit() for ch in token))
        }
        per_choice[label] = tokens
        for token in tokens:
            token_owners.setdefault(token, set()).add(label)

    scored: list[tuple[int, float, str]] = []
    for label, tokens in per_choice.items():
        distinctive = {token for token in tokens if token_owners.get(token) == {label}}
        if not distinctive:
            continue
        matched = distinctive & fragment_tokens
        count = len(matched)
        coverage = count / len(distinctive)
        # One anchor is only sufficient when the choice itself has one highly
        # distinctive content token.  Otherwise require at least two unique anchors.
        single_strong = (
            count == 1
            and len(distinctive) == 1
            and next(iter(matched), "")
            and (len(next(iter(matched))) >= 8 or any(ch.isdigit() for ch in next(iter(matched))))
        )
        if count >= 2 or single_strong:
            scored.append((count, coverage, label))

    if not scored:
        return None
    scored.sort(reverse=True)
    top_count, top_coverage, top_label = scored[0]
    if len(scored) == 1:
        return top_label
    runner_count, runner_coverage, _ = scored[1]
    if top_count >= runner_count + 2:
        return top_label
    if top_count > runner_count and top_coverage >= runner_coverage + 0.35:
        return top_label
    return None


def _choice_from_selected_clause(
    visible: str,
    choices: Mapping[str, str],
    role: str | None,
) -> str | None:
    """Map a conclusive role-local clause to one unique benchmark choice."""
    if role == "second_best":
        cue = re.compile(
            r"\b(?:best\s+alternative|second[- ]best|runner[- ]?up|"
            r"(?:an?\s+)?(?:(?:plausible|reasonable|good|strong|less\s+suitable\s+but)\s+)?"
            r"alternative(?:\s+(?:answer|choice|diagnosis|treatment|option))?\s+"
            r"(?:is|would\s+be|could\s+be|seems|appears)|"
            r"(?:a\s+)?different\s+(?:answer|choice|diagnosis|treatment|option)\s+"
            r"(?:that\s+)?could\s+be\s+considered\s+(?:is|as))\b",
            re.IGNORECASE,
        )
    elif role == "final_answer":
        cue = re.compile(
            r"\b(?:final\s+answer|final\s+choice|therefore|thus|conclusion)\b", re.IGNORECASE
        )
    else:
        cue = re.compile(
            r"\b(?:provisional\s+answer|initial\s+answer|best\s+answer|best\s+choice|"
            r"best\s+treatment|most\s+likely|most\s+appropriate|most\s+(?:directly\s+)?related|correct\s+answer|"
            r"correct\s+choice|only\s+plausible|preferred\s+answer|"
            r"(?:seems|appears)\s+(?:to\s+be\s+)?(?:a\s+)?(?:reasonable|appropriate|prudent)\s+"
            r"(?:choice|option|approach)|seems\s+prudent)\b",
            re.IGNORECASE,
        )

    clauses = re.split(r"(?<=[.!?])\s+|\n+", visible)
    candidates: list[tuple[int, str]] = []
    offset = 0
    for clause in clauses:
        stripped = clause.strip()
        if not stripped:
            offset += len(clause) + 1
            continue
        # The role heading itself is not evidence that the model selected a choice.
        # Only role-local generated language (or an explicit leading option label,
        # handled by ``_semantic_option``) can create a commitment.
        if not cue.search(stripped):
            offset += len(clause) + 1
            continue
        selected = _choice_from_fragment(stripped, choices)
        if selected is not None:
            candidates.append((offset, selected))
        offset += len(clause) + 1
    return candidates[-1][1] if candidates else None


def _semantic_option(
    text: str,
    choices: Mapping[str, str],
    role: str | None = None,
) -> str | None:
    """Recover the option actually selected inside one reasoning role.

    Selection syntax is treated as presentation.  The parser first looks for
    role-local conclusive statements (including choice text plus a parenthesized
    label), choosing the *last* conclusive statement so an explicit self-correction
    inside the provisional section is respected.  It then falls back to conservative
    generic option syntax.
    """
    visible = _plain_text(text)
    contextual: list[tuple[int, str]] = []

    # A role's own explicit declaration is higher-confidence than incidental option
    # discussion later in the same section.  This prevents ``Best alternative: A``
    # from becoming B merely because the rationale explains why B is stronger, and
    # prevents ordinary prose articles from competing with declared option labels.
    explicit_role: list[tuple[int, str]] = []
    leading_role_option: str | None = None
    if role in {"provisional_answer", "second_best", "final_answer"}:
        leading_match = re.match(
            r"(?s)^\s*(?:[-+*]\s+)?(?:\*\*|__|`)?\s*"
            r"(?:(?:option|choice)\s+)?(?-i:([A-Z]))(?:\s*[.):\-]|\s+(?=\())",
            visible,
            re.IGNORECASE,
        )
        if leading_match is not None and leading_match.group(1).upper() in choices:
            leading_role_option = leading_match.group(1).upper()
        else:
            unpunctuated = re.match(
                r"(?s)^\s*(?:[-+*]\s+)?(?:\*\*|__|`)?\s*(?-i:([A-Z]))\s+",
                visible,
            )
            if unpunctuated is not None:
                candidate = unpunctuated.group(1).upper()
                if (
                    candidate in choices
                    and _choice_from_fragment(visible[:240], choices) == candidate
                ):
                    leading_role_option = candidate
        if leading_role_option is None:
            compact_role_label = re.match(
                r"(?s)^\s*(?:[-+*]\s+)?(?:\*\*|__|`)?\s*(?-i:([A-Z]))\s+"
                r"(?:Why\b|Best\s+explanation\b)",
                visible,
                re.IGNORECASE,
            )
            if compact_role_label is not None and compact_role_label.group(1).upper() in choices:
                leading_role_option = compact_role_label.group(1).upper()

        if leading_role_option is None and role == "provisional_answer":
            # Some compact model formats omit punctuation but write e.g.
            # ``C Best describes ...``.  Restrict this special case to the literal
            # ``Best describes`` phrase so sentence-initial articles such as
            # ``A best alternative ...`` cannot become option A.
            compact_best = re.match(
                r"(?s)^\s*(?:[-+*]\s+)?(?:\*\*|__|`)?\s*(?-i:([A-Z]))\s+Best\s+describes\b",
                visible,
                re.IGNORECASE,
            )
            if compact_best is not None and compact_best.group(1).upper() in choices:
                leading_role_option = compact_best.group(1).upper()
    if leading_role_option is None:
        leading_role_negated = False
    elif role == "second_best":
        # A second-best choice is often explicitly described as incorrect or less
        # suitable; that is the rationale for it being *second* best, not a rejection
        # of the role declaration.  Only invalidate the leading label when the text
        # directly says it is not the alternative or explicitly corrects to another.
        escaped = re.escape(leading_role_option)
        leading_role_negated = bool(
            re.search(
                rf"(?s)^\s*(?:[-+*]\s+)?(?:(?:option|choice)\s+)?{escaped}[.):]?"
                r"[^.;\n]{0,120}\b(?:is|would\s+be|seems|appears)?\s*not\s+(?:the\s+)?"
                r"(?:best\s+alternative|second[- ]best|runner[- ]?up|viable\s+alternative)\b",
                visible[:240],
                re.IGNORECASE,
            )
            or re.search(
                r"\b(?:wait|correction|correcting|instead|actually\s+the\s+(?:best|correct)\s+alternative)\b",
                visible[:220],
                re.IGNORECASE,
            )
        )
    else:
        leading_role_negated = bool(
            re.search(
                r"\b(?:wait\b[^.;\n]{0,30}\breconsider|no\b[^.;\n]{0,30}\breconsider|"
                r"reconsidering|re-evaluating|not\s+(?:correct|the\s+(?:best|next|right|safest)|appropriate)|"
                r"incorrect|correction|instead\b|actually\s+the\s+(?:best|correct|provisional)|"
                r"therefore\b[^.;\n]{0,80}\b(?:best|correct|provisional)\s+(?:answer|choice))\b",
                visible[:220],
                re.IGNORECASE,
            )
        )

    if role == "provisional_answer":
        role_decl = re.compile(
            r"\b(?:(?:provisional|initial)\s+(?:best\s+)?(?:answer|choice|diagnosis|treatment)\s*"
            r"(?:(?:is\s*:?)|=|:|of)?\s*|"
            r"best\s+(?:answer|choice|diagnosis|treatment)\s*(?:(?:is\s*:?)|=|:)?\s*|"
            r"(?:revise|change|correct)\w*\s+(?:the\s+)?(?:provisional|initial)\s+(?:answer|choice)\s+to\s+|"
            r"(?:i|we)\s+(?:now\s+)?(?:select|choose)\s+)"
            r"(?:actually\s+)?(?:\*\*|__|`)?\s*[\"']*\s*(?:(?:option|choice)\s+)?\(?(?-i:([A-Z]))(?![A-Za-z])\)?[.):\-]?"
            r"(?:[^.;\n]{0,90}\bas\s+(?:the\s+)?(?:provisional|initial)\s+(?:answer|choice))?",
            re.IGNORECASE,
        )
    elif role == "second_best":
        role_decl = re.compile(
            r"\b(?:best\s+alternative|second[- ]best(?:\s+answer)?|runner[- ]?up|alternative\s+(?:answer|choice|diagnosis|treatment|option))"
            r"(?:"
            r"\s+to\s+[^\n]{1,120}?\b(?:is|would\s+be|could\s+be|could\s+be\s+considered\s+as|seems\s+to\s+be|appears\s+to\s+be)\s+"
            r"|\s*(?:is|would\s+be|could\s+be|=|:)\s*"
            r")"
            r"(?:actually\s+|still\s+)?(?:\*\*|__|`)?\s*[\"']*\s*"
            r"(?:(?:option|choice)\s+)?\(?(?-i:([A-Z]))(?![A-Za-z])\)?[.):\-]?",
            re.IGNORECASE,
        )
    elif role == "final_answer":
        role_decl = re.compile(
            r"\b(?:final\s+(?:answer|choice)|selected\s+(?:answer|choice|option))\s*"
            r"(?:(?:is|=|:)\s*)?(?:actually\s+)?(?:\*\*|__|`)?\s*[\"']*\s*"
            r"(?:(?:option|choice)\s+)?\(?(?-i:([A-Z]))(?![A-Za-z])\)?[.):\-]?",
            re.IGNORECASE,
        )
    else:
        role_decl = None
    if role_decl is not None:
        for match in role_decl.finditer(visible):
            option = match.group(1).upper()
            if option in choices:
                explicit_role.append((match.start(), option))

    if role == "second_best":
        selectors = r"(?:best\s+alternative|second[- ]best(?:\s+answer)?|runner[- ]?up)"
    elif role == "final_answer":
        selectors = r"(?:final\s+answer|final\s+choice|selected\s+answer|correct\s+answer)"
    else:
        selectors = (
            r"(?:(?:provisional|initial)\s+(?:answer|choice|diagnosis|treatment)|"
            r"(?:best|correct|selected|preferred|most\s+likely|most\s+appropriate|least\s+likely|least\s+appropriate|most\s+reasonable)\s+"
            r"(?:[a-z][a-z'\N{RIGHT SINGLE QUOTATION MARK}-]*\s+){0,10}(?:answer|choice|option|diagnosis|treatment|management|"
            r"cause|explanation|phase|approach|position|step)?)"
        )

    # Selector followed directly by an option label, allowing descriptive words
    # between the selector and the copula ("most likely cause of ... is D").
    direct = re.compile(
        rf"\b{selectors}\b[^.;\n]{{0,150}}?\b(?:is|would\s+be|could\s+be|should\s+be|could\s+be\s+considered\s+as|seems\s+to\s+be|appears\s+to\s+be|=|:)\s*:?[ \t]*"
        r"(?:(?:option|choice)\s*)?\(?(?-i:([A-Z]))\)?[.):\-]?\b",
        re.IGNORECASE,
    )
    for match in direct.finditer(visible):
        option = match.group(1).upper()
        if option in choices:
            contextual.append((match.start(), option))

    # Selector followed by answer text with an explicit parenthesized option.
    parenthesized = re.compile(
        rf"\b{selectors}\b[^.;\n]{{0,180}}?\((?-i:([A-Z]))\)",
        re.IGNORECASE,
    )
    for match in parenthesized.finditer(visible):
        option = match.group(1).upper()
        if option in choices:
            contextual.append((match.start(), option))

    if role in {None, "provisional_answer"}:
        # Natural variants put the preference after the labeled choice:
        # ``Early acceleration (C) seems to be the most appropriate choice``.
        suffix = re.compile(
            r"(?:\b(?-i:([A-Z]))[.):\-]|\((?-i:([A-Z]))\))[^.;\n]{0,100}?"
            r"\b(?:is|seems|appears)\s+(?:to\s+be\s+)?(?:(?:the|a|an)\s+)?"
            r"(?:best|better|correct|most\s+likely|most\s+appropriate|preferred)\s+"
            r"(?:answer|choice|option|diagnosis|treatment|approach|phase)\b",
            re.IGNORECASE,
        )
        for match in suffix.finditer(visible):
            option = next(group for group in match.groups() if group is not None).upper()
            if option in choices:
                contextual.append((match.start(), option))
        label_first = re.compile(
            r"(?<![A-Za-z])(?-i:([A-Z]))[.):]?\s+"
            r"(?:is|would\s+be|seems\s+to\s+be|appears\s+to\s+be)\s+(?:the\s+)?"
            r"(?:best|better|correct|most\s+likely|most\s+appropriate|preferred)\s+"
            r"(?:answer|choice|option|diagnosis|treatment|approach)\b",
            re.IGNORECASE,
        )
        for match in label_first.finditer(visible):
            option = match.group(1).upper()
            if option in choices:
                contextual.append((match.start(), option))

        labeled_evaluation = re.compile(
            r"(?<![A-Za-z])(?-i:([A-Z]))[.):]\s*[^.;\n]{0,120}?"
            r"\b(?:seems|appears|is)\s+(?:to\s+be\s+)?(?:(?:the|a|an)\s+)?"
            r"(?:plausible|reasonable|appropriate|likely|compelling|better|best|preferred|universally\s+applicable)\b",
            re.IGNORECASE,
        )
        for match in labeled_evaluation.finditer(visible):
            option = match.group(1).upper()
            if option in choices:
                contextual.append((match.start(), option))

        parenthesized_candidate = re.compile(
            r"\((?-i:([A-Z]))\)[^.;\n]{0,70}\b"
            r"(?:is|remains?|seems|appears)\s+(?:to\s+be\s+)?(?:a|the)?\s*"
            r"(?:strong|leading|plausible|reasonable|compelling)\s+(?:candidate|choice|diagnosis|answer)\b",
            re.IGNORECASE,
        )
        for match in parenthesized_candidate.finditer(visible):
            option = match.group(1).upper()
            if option in choices:
                contextual.append((match.start(), option))

        parenthesized_evaluation = re.compile(
            r"\((?-i:([A-Z]))\)[^.;\n]{0,100}\b"
            r"(?:is|remains?|seems|appears)\s+(?:to\s+be\s+)?(?:(?:the|a|an)\s+)?"
            r"(?:best|correct|preferred|most\s+likely|(?:condition\s+)?least\s+likely|most\s+appropriate|"
            r"least\s+appropriate|most\s+accurate|most\s+reasonable|strong\s+candidate|"
            r"leading\s+candidate|plausible\s+candidate|reasonable\s+candidate)\b",
            re.IGNORECASE,
        )
        for match in parenthesized_evaluation.finditer(visible):
            option = match.group(1).upper()
            if option in choices:
                contextual.append((match.start(), option))

        labeled_direct_answer = re.compile(
            r"(?<![A-Za-z])(?-i:([A-Z]))[.):]?\s*[^.;\n]{0,120}?"
            r"\b(?:directly\s+(?:answers?|addresses?|fits?)|best\s+(?:answers?|fits?))\b",
            re.IGNORECASE,
        )
        for match in labeled_direct_answer.finditer(visible):
            option = match.group(1).upper()
            if option in choices:
                contextual.append((match.start(), option))

        # When a bare option letter is not punctuated (``B seems universally
        # applicable``), scan only actual choice labels.  A generic ``[A-Z]`` regex
        # can otherwise capture the capital G in ``Given this`` and skip the real B.
        contextual.extend(
            (match.start(), option)
            for option in choices
            for match in re.finditer(
                rf"(?<![A-Za-z]){re.escape(option)}(?![A-Za-z])[^.;\n]{{0,80}}?"
                r"(?:seems|appears|is)\s+(?:to\s+be\s+)?(?:(?:the|a|an)\s+)?"
                r"(?:universally\s+applicable|best|better|preferred|most\s+appropriate)|"
                rf"(?<![A-Za-z]){re.escape(option)}(?![A-Za-z])[^.;\n]{{0,100}}?directly\s+(?:answers?|addresses?|fits?)",
                visible,
                re.IGNORECASE,
            )
        )

        role_after_label = re.compile(
            r"(?<![A-Za-z])(?-i:([A-Z]))[.):]\s*[^.;\n]{0,150}?"
            r"\b(?:this\s+is\s+)?(?:the\s+)?(?:provisional|initial)\s+(?:best\s+)?(?:answer|choice)\b",
            re.IGNORECASE,
        )
        for match in role_after_label.finditer(visible):
            option = match.group(1).upper()
            if option in choices:
                contextual.append((match.start(), option))

        better_fit = re.compile(
            r"\b(?:a\s+)?(?:better|best)\s+fit\s+(?:seems|appears)\s+to\s+be\s+"
            r"(?:(?:option|choice)\s+)?(?-i:([A-Z]))[.):]?\b",
            re.IGNORECASE,
        )
        for match in better_fit.finditer(visible):
            option = match.group(1).upper()
            if option in choices:
                contextual.append((match.start(), option))

    if role == "provisional_answer":
        # Some responses make a clear natural-language selection without repeating
        # the literal role name, e.g. ``Among the provided options, Fludarabine is
        # a purine analog used in CLL``.  Accept this only when the clause uniquely
        # maps to an offered choice and contains a positive applicability predicate.
        among_options = re.compile(
            r"\bamong\s+(?:the\s+)?(?:provided|given|listed|available)\s+(?:options|choices)\s*[:,]?\s*"
            r"([^.;\n]{1,180})",
            re.IGNORECASE,
        )
        for match in among_options.finditer(visible):
            fragment = match.group(1)
            if re.search(
                r"\b(?:is|are|used|indicated|recommended|appropriate|fits?|matches?|supports?|"
                r"consistent\s+with|treats?|treatment|therapy|diagnosis)\b",
                fragment,
                re.IGNORECASE,
            ):
                selected = _choice_from_fragment(fragment, choices)
                if selected is not None:
                    contextual.append((match.start(), selected))

        given_labeled_option = re.compile(
            r"\bgiven\s+(?:the\s+)?(?:options|choices)\s+(?:provided|given|listed|available)?\s*[:,]?\s*"
            r"(?:[-+*]\s*)?(?-i:([A-Z]))[.):]",
            re.IGNORECASE,
        )
        for match in given_labeled_option.finditer(visible):
            option = match.group(1).upper()
            if option in choices:
                contextual.append((match.start(), option))

        given_options = re.compile(
            r"\bgiven\s+(?:the\s+)?(?:options|choices)\s+(?:provided|given|listed|available)?\s*[:,]?\s*"
            r"([^.;\n]{1,220})",
            re.IGNORECASE,
        )
        for match in given_options.finditer(visible):
            fragment = match.group(1)
            if re.search(
                r"\b(?:is|are|used|indicated|recommended|appropriate|fits?|matches?|supports?|"
                r"consistent\s+with|treats?|treatment|therapy|diagnosis)\b",
                fragment,
                re.IGNORECASE,
            ):
                selected = _choice_from_fragment(fragment, choices)
                if selected is not None:
                    contextual.append((match.start(), selected))

    if role == "second_best":
        # Respect an explicit correction from an initially repeated provisional
        # option to a genuinely different fallback: ``if we have to choose another,
        # it would be D``.
        choose_another = re.compile(
            r"\b(?:if\s+(?:we|i)\s+(?:have|had)\s+to\s+choose\s+another|"
            r"another\s+(?:choice|option)\s+would\s+be)\b[^.;\n]{0,45}?"
            r"(?:(?:option|choice)\s+)?(?-i:([A-Z]))[.):]?",
            re.IGNORECASE,
        )
        for match in choose_another.finditer(visible):
            option = match.group(1).upper()
            if option in choices:
                contextual.append((match.start(), option))

        strong_alternative = re.compile(
            r"\b(?:a\s+)?(?:strong|plausible|reasonable|viable|closest|main)\s+alternative"
            r"(?:\s+to\s+[^.;\n]{1,100}?)?\s+"
            r"(?:is|would\s+be|could\s+be|could\s+be\s+considered\s+as|seems\s+to\s+be|appears\s+to\s+be)"
            r"[^.;\n]{0,120}?\((?-i:([A-Z]))\)",
            re.IGNORECASE,
        )
        for match in strong_alternative.finditer(visible):
            option = match.group(1).upper()
            if option in choices:
                contextual.append((match.start(), option))

        target_re = re.compile(
            r"\bbest\s+alternative(?:\s+to\s+[^.;\n]{1,90}?)?\s+"
            r"(?:is|would\s+be|could\s+be|could\s+be\s+considered\s+as|"
            r"seems\s+to\s+be|appears\s+to\s+be)\s+([^.;\n]{1,140})",
            re.IGNORECASE,
        )
        for match in target_re.finditer(visible):
            selected = _choice_from_fragment(match.group(1), choices)
            if selected is not None:
                contextual.append((match.start(), selected))
        label_first_alternative = re.compile(
            r"(?<![A-Za-z])(?-i:([A-Z]))[.):]?\s+"
            r"(?:is|would\s+be|could\s+be|could\s+be\s+considered\s+as|seems\s+to\s+be|appears\s+to\s+be)\s+"
            r"(?:the\s+)?(?:best\s+alternative|second[- ]best(?:\s+answer)?|runner[- ]?up)\b",
            re.IGNORECASE,
        )
        for match in label_first_alternative.finditer(visible):
            option = match.group(1).upper()
            if option in choices:
                contextual.append((match.start(), option))

    if role == "final_answer":
        boxed = re.compile(r"\\boxed\s*\{?\s*([A-Z])\s*\}?", re.IGNORECASE)
        for match in boxed.finditer(text):
            option = match.group(1).upper()
            if option in choices:
                contextual.append((len(visible) + match.start() + 1, option))

    clause_choice = _choice_from_selected_clause(visible, choices, role)
    explicit_leading_rejection = False
    if leading_role_option is not None:
        escaped_leading = re.escape(leading_role_option)
        explicit_leading_rejection = bool(
            re.search(
                rf"(?s)^\s*(?:[-+*]\s+)?(?:\*\*|__|`)?\s*{escaped_leading}[.):]?\s*"
                r"[^.;\n]{0,120}\bis\s+not\s+(?:the\s+)?"
                r"(?:best|correct|appropriate|preferred|most\s+appropriate)\b",
                visible,
                re.IGNORECASE,
            )
        )
    if explicit_leading_rejection and clause_choice not in {None, leading_role_option}:
        return clause_choice
    if clause_choice is not None and not contextual:
        # Clause-to-choice mapping is a fallback.  It must not override an explicit
        # labeled/selector relation already recovered from the same role; doing so
        # can turn a generic phrase such as ``benign condition`` into the wrong
        # benchmark option even when the model later writes ``B. ... seems plausible``.
        contextual.append((len(visible), clause_choice))

    # A leading label is normally authoritative, but models sometimes explicitly
    # repair it inside the same provisional role.  Respect a later conclusive choice
    # only when it follows an unmistakable correction/re-evaluation cue; ordinary
    # discussion of alternatives still cannot override the role declaration.
    correction_positions = [
        match.start()
        for match in re.finditer(
            r"\b(?:wait|correction|correcting|upon\s+closer\s+reflection|on\s+closer\s+reflection|"
            r"upon\s+reconsideration|after\s+reconsideration|re[- ]?evaluat(?:e|ing|ed)|reconsider(?:ing)?|"
            r"this\s+contradicts?\s+the\s+facts|cannot\s+be\s+correct|between\s+these|"
            r"(?<![A-Za-z])(?-i:[A-Z])[.):]?[^.;\n]{0,80}\bis\s+(?:the\s+)?better\s+choice)\b",
            visible,
            re.IGNORECASE,
        )
    ]
    if leading_role_option is not None:
        escaped_leading = re.escape(leading_role_option)
        correction_positions.extend(
            match.start()
            for match in re.finditer(
                rf"(?<![A-Za-z]){escaped_leading}[.):]?[^.;\n]{{0,120}}?"
                r"\b(?:is|would\s+be|seems\s+to\s+be)?\s*not\s+(?:the\s+)?"
                r"(?:best|correct|appropriate|preferred|most\s+appropriate)\b",
                visible,
                re.IGNORECASE,
            )
        )
    if correction_positions:
        last_correction = correction_positions[-1]
        repaired = [
            item
            for item in (*explicit_role, *contextual)
            if item[0] >= last_correction and item[1] in choices
        ]
        if repaired:
            repaired.sort(key=lambda item: item[0])
            if leading_role_option is None or repaired[-1][1] != leading_role_option:
                return repaired[-1][1]

    if explicit_role:
        explicit_role.sort(key=lambda item: item[0])
        return explicit_role[-1][1]

    if leading_role_option is not None and not leading_role_negated:
        return leading_role_option

    if contextual:
        contextual.sort(key=lambda item: item[0])
        return contextual[-1][1]

    for pattern in SEMANTIC_OPTION_RES:
        matches = list(pattern.finditer(visible))
        if matches:
            match = matches[-1]
            option = match.group(1).upper()
            if option in choices:
                return option

    normalized = " ".join(visible.casefold().split())
    hits = [
        option
        for option, choice in choices.items()
        if choice.strip() and " ".join(choice.casefold().split()) in normalized
    ]
    if len(hits) == 1:
        return hits[0]

    # Salient-token recovery is a conservative lexical fallback, not a license to
    # infer a choice merely because its medical terms were discussed.  Require a
    # role-appropriate selection relation somewhere in the body.
    if role == "provisional_answer":
        salient_allowed = bool(
            re.search(
                r"\b(?:provisional\s+(?:answer|choice)|best\s+(?:answer|choice|treatment|diagnosis)|"
                r"most\s+likely\s+(?:diagnosis|explanation|answer|choice|cause)|"
                r"most\s+appropriate\s+(?:answer|choice|treatment|management|step)|"
                r"strong\s+consideration|expected\s+(?:additional\s+)?(?:feature|finding|manifestation)\s+would\s+be|(?:is|seems|appears)\s+(?:to\s+be\s+)?(?:the\s+)?"
                r"(?:most\s+likely|best|preferred)\b|therefore[^.;\n]{0,80}\b(?:answer|choice)\b)",
                visible,
                re.IGNORECASE,
            )
        )
    elif role == "second_best":
        salient_allowed = bool(
            re.search(
                r"\b(?:best\s+alternative|second[- ]best|runner[- ]?up|alternative\s+(?:answer|choice|diagnosis|treatment)|is\s+the\s+alternative)\b",
                visible,
                re.IGNORECASE,
            )
        )
    else:
        salient_allowed = True
    if salient_allowed:
        salient = _choice_from_salient_text(visible, choices)
        if salient is not None:
            return salient
    return None


def _decision(match: re.Match[str] | None) -> str | None:
    if match is None:
        return None
    attr = DECISION_ATTR_RE.search(match.group("attrs"))
    if attr is None:
        return None
    value = attr.group(1).casefold()
    return value if value in VALID_DECISIONS else None


def _marker_schema_valid(
    tag: str,
    matches: tuple[re.Match[str], ...],
    choices: Mapping[str, str],
) -> bool:
    """Require one requested opening marker with only its instructed attribute.

    Semantic recovery deliberately tolerates headings and reversed slashes.  This
    validator does not: it represents whether an Answer Engineering controller can
    address the requested slot without first repairing the serialization itself.
    Whitespace and either conventional quote character are accepted, but missing,
    duplicate, placeholder, or extra attributes are not.
    """
    if len(matches) != 1:
        return False
    attrs = matches[0].group("attrs")
    if tag in {
        "facts",
        "implications",
        "decisive_fact",
        "answer_changing_change",
    }:
        return not attrs.strip()
    if tag in {"provisional_answer", "second_best", "final_answer"}:
        option_match = re.fullmatch(
            r"\s+option\s*=\s*(['\"])\s*([A-Z])\s*\1\s*",
            attrs,
            re.IGNORECASE,
        )
        return bool(option_match and option_match.group(2).upper() in choices)
    if tag == "rereasoning":
        decision_match = re.fullmatch(
            r"\s+decision\s*=\s*(['\"])\s*([A-Za-z]+)\s*\1\s*",
            attrs,
            re.IGNORECASE,
        )
        return bool(decision_match and decision_match.group(2).casefold() in VALID_DECISIONS)
    raise AssertionError(f"unhandled protocol tag: {tag}")


def _reconsideration_selected_option(
    text: str,
    choices: Mapping[str, str],
) -> str | None:
    """Recover a choice that the reconsideration itself clearly concludes on.

    This is deliberately narrower than generic option mention recovery: an option
    must be tied to a conclusion cue such as reaffirming, confirming, remaining the
    most likely answer, becoming the more specific choice, or being explicitly
    supported as correct. References to alternatives without a conclusion do not
    select an option.
    """
    visible = _plain_text(text)
    candidates: list[tuple[int, str]] = []

    direct_patterns = (
        re.compile(
            r"\b(?:reaffirm(?:ing|ed)?|confirm(?:ing|ed)?|retain(?:ing|ed)?)\s+"
            r"(?:(?:the\s+)?(?:provisional|initial|final)\s+(?:answer|choice)\s*)?"
            r"(?:(?:option|choice)\s*)?\(?(?-i:([A-Z]))\)?[.):\-]?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:support|favor|favour|confirm|point)\w*[^.;\n]{0,80}"
            r"(?:\b(?:option|choice)\s+\(?(?-i:([A-Z]))\)?\b|"
            r"\((?-i:([A-Z]))\)|(?<![A-Za-z])(?-i:([A-Z]))[.):])"
            r"[^.;\n]{0,50}(?:correct|best|most\s+likely|most\s+appropriate)?",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:the\s+)?(?:correct|better|best|final)\s+(?:answer|choice|option|diagnosis|treatment)\s+"
            r"(?:is|would\s+be|becomes?)\s+(?:(?:option|choice)\s+)?\(?(?-i:([A-Z]))\)?[.):]?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:option|choice)\s+\(?(?-i:([A-Z]))\)?\b[^.;\n]{0,80}"
            r"(?:stands?\s+(?:out\s+)?as|emerges?\s+as)\s+(?:(?:the|a|an)\s+)?"
            r"(?:best|correct|most\s+likely|most\s+appropriate|most\s+plausible|plausible|"
            r"reasonable|compelling|preferred|strongest|fundamental)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:option|choice)\s+\(?(?-i:([A-Z]))\)?\b[^.;\n]{0,80}"
            r"(?:remains?|is\s+still|still\s+(?:seems|appears)|seems|appears|is)\s+"
            r"(?:to\s+be\s+)?(?:(?:the|a|an)\s+)?"
            r"(?:best|correct|most\s+likely|most\s+appropriate|more\s+appropriate|"
            r"more\s+specific|most\s+accurate|compelling|preferred|strongest)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?<![A-Za-z])(?-i:([A-Z]))[.):]?\s+"
            r"(?:remains?|is\s+still|still\s+(?:seems|appears)|seems|appears|is)\s+"
            r"(?:to\s+be\s+)?(?:(?:the|a|an)\s+)?"
            r"(?:best|correct|most\s+likely|most\s+appropriate|most\s+accurate|most\s+plausible|"
            r"plausible|likely|reasonable|compelling|better|accurate|appropriate|preferred|strongest)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?<![A-Za-z])(?-i:([A-Z]))[.):]?\s+stands?\s+out\s+as\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:option|choice)\s+\(?(?-i:([A-Z]))\)?\b[^.;\n]{0,100}"
            r"(?:remains?\s+(?:(?:the|a|an)\s+)?(?:superior|strong\s+candidate)|"
            r"(?:is|seems|appears)\s+(?:to\s+be\s+)?(?:(?:the|a|an)\s+)?"
            r"(?:compelling|accurate|appropriate|plausible|reasonable|ideal|superior)|"
            r"(?:best|still\s+best)\s+(?:aligns?|fits?)|"
            r"directly\s+addresses?[^.;\n]{0,55}\b(?:accurate|appropriate|significant|best)\b)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?<![A-Za-z])(?-i:([A-Z]))[.):]?\s+"
            r"(?:remains?\s+(?:(?:the|a|an)\s+)?(?:superior|strong\s+candidate)|"
            r"(?:is|seems|appears)\s+(?:to\s+be\s+)?(?:(?:the|a|an)\s+)?"
            r"(?:compelling|accurate|appropriate|plausible|reasonable|ideal|superior))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:reinforc(?:e|es|ed|ing)|reaffirm(?:s|ed|ing)?)\s+"
            r"(?:the\s+)?(?:choice|answer|selection)\s+(?:of\s+)?"
            r"(?:(?:option|choice)\s+)?\(?(?-i:([A-Z]))\)?[.):]?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bmakes?\s+(?:(?:option|choice)\s+)?(?-i:([A-Z]))[.):]?"
            r"[^.;\n]{0,90}?(?:(?:the|a|an)\s+)?"
            r"(?:compelling|best|preferred|most\s+likely|most\s+appropriate|strong)\s+"
            r"(?:answer|choice|diagnosis|treatment|option)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bmak(?:e|es|ing)\s+(?:(?:option|choice)\s+)?(?-i:([A-Z]))[.):]?"
            r"[^.;\n]{0,100}?\b(?:the\s+)?(?:correct|best|most\s+appropriate|most\s+likely)\s+"
            r"(?:course\s+of\s+action|answer|choice|option|diagnosis|treatment|step)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:the\s+)?(?:provisional|initial)\s+(?:answer|choice)\s+(?:of|is|:)\s*"
            r"(?:(?:option|choice)\s+)?\(?(?-i:([A-Z]))\)?[.):]?"
            r"[^.;\n]{0,120}\b(?:remains?|seems|appears|is)\s+(?:to\s+be\s+)?"
            r"(?:(?:the|a|an)\s+)?(?:best|correct|most\s+likely|most\s+appropriate|"
            r"most\s+accurate|most\s+plausible|plausible|reasonable|compelling|"
            r"most\s+comprehensive\s+explanation|comprehensive\s+explanation)\b",
            re.IGNORECASE,
        ),
    )
    for pattern in direct_patterns:
        for match in pattern.finditer(visible):
            option = next((group for group in match.groups() if group is not None), None)
            if option is None:
                continue
            option = option.upper()
            if option in choices:
                candidates.append((match.start(), option))

    # Exact choice text tied locally to a conclusion predicate covers prose such as
    # ``SSSS remains the most likely diagnosis`` when the benchmark answer text is
    # actually written out, while still rejecting a bare mention.
    normalized = normalize_text(visible)
    for option, choice in choices.items():
        choice_norm = " ".join(_normalized_choice_tokens(option, choice, choices))
        if not choice_norm or choice_norm in {"n a", "na"}:
            continue
        for match in re.finditer(rf"(?<!\w){re.escape(choice_norm)}(?!\w)", normalized):
            after = normalized[match.end() : match.end() + 130]
            label = re.escape(option.casefold())
            if re.match(
                rf"(?:\s+{label})?\s+(?:remains?|is\s+still|still\s+(?:seems|appears)|seems|appears|is)\s+"
                r"(?:to\s+be\s+)?(?:(?:the|a|an)\s+)?(?:best|correct|most\s+likely|most\s+appropriate|"
                r"more\s+appropriate|more\s+specific|most\s+accurate|most\s+plausible|plausible|likely|"
                r"reasonable|compelling|better|best\s+fit|strong\s+candidate|stands?(?:\s+out)?|superior|ideal|most\s+logical|most\s+comprehensive\s+explanation|"
                r"comprehensive\s+explanation|preferred|strongest)\b"
                r"|(?:\s+" + label + r")?\s+(?:best|better|most\s+plausibly)\s+explains?\b"
                r"|(?:\s+" + label + r")?\s+stands?\s+out\b",
                after,
            ):
                candidates.append((match.start(), option))

    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _reconsideration_supports_option(
    text: str,
    choices: Mapping[str, str],
    option: str,
) -> bool:
    """Whether reconsideration positively concludes on a particular option.

    This is a local semantic relation test, not a final-answer shortcut.  The target
    option (or enough of its answer text) must actually occur in a clause/sentence
    containing affirmative evaluative language.  Thus ``final == provisional`` by
    itself earns no credit, while natural prose such as ``neuroblastoma remains a
    strong candidate`` or ``Option B is the most suitable choice`` does.
    """
    if option not in choices:
        return False
    visible = _plain_text(text)
    normalized = normalize_text(visible)
    if len(normalized.split()) < 5:
        return False

    option_lower = option.casefold()
    choice_tokens = _normalized_choice_tokens(option, choices[option], choices)
    choice_norm = " ".join(choice_tokens)
    other_tokens = {
        token
        for other_option, other_choice in choices.items()
        if other_option != option
        for token in _normalized_choice_tokens(other_option, other_choice, choices)
    }
    distinctive_tokens = {
        token for token in choice_tokens if len(token) >= 6 and token not in other_tokens
    }

    # Work sentence/bullet-locally so praise for one alternative cannot be attached
    # to another option mentioned elsewhere in the reconsideration.
    # Split before lexical normalization so sentence punctuation remains available
    # as a semantic boundary.  Normalizing the whole paragraph first erases the
    # punctuation and can let a negation about a later rejected alternative veto
    # a positive conclusion in the preceding sentence.
    chunks = [
        normalized_chunk
        for chunk in re.split(r"(?:[.;!?]\s+|\n+|\s+-\s+)", visible)
        if (normalized_chunk := normalize_text(chunk.strip()))
    ]
    positive = re.compile(
        r"\b(?:"
        r"(?:remains?|is|seems|appears|continues?|stands?)\b[^,;]{0,70}\b"
        r"(?:correct|accurate|appropriate|plausible|likely|reasonable|compelling|valid|beneficial|necessary|"
        r"ideal|superior|suitable|relevant|supported|best|strong\s+candidate|most\s+likely|"
        r"most\s+appropriate|most\s+plausible|most\s+accurate|most\s+logical|most\s+relevant)|"
        r"(?:best|most\s+likely|most\s+appropriate|most\s+plausible|most\s+logical|most\s+suitable)"
        r"\b[^,;]{0,40}\b(?:answer|choice|diagnosis|treatment|intervention|explanation|location)|"
        r"(?:support|favor|favour|confirm|reinforce|suggest|point)s?\b|"
        r"(?:best|most\s+plausibly)\s+(?:explains?|fits?|aligns?)\b|"
        r"(?:still\s+)?(?:points?|aligns?|fits?)\s+(?:(?:best|most\s+strongly|strongly)\s+)?(?:with|toward|towards|to)\b|"
        r"takes?\s+precedence\b|should\s+be\s+(?:strongly\s+)?considered\b|"
        r"(?:indeed\s+)?stands?\s+out\b|(?:still\s+)?lean(?:s|ing)?\s+(?:toward|towards)\b|"
        r"(?:is|remains?)\s+(?:the\s+)?preferred\b|make(?:s)?\s+it\s+(?:the\s+)?preferred\b|"
        r"seems?\s+more\s+directly\s+related\b|provides?\b[^,;]{0,45}\b(?:relevant|specific|strong|direct)\b|"
        r"directly\s+addresses?\b|makes?\s+it\s+(?:highly\s+)?(?:suitable|appropriate|compelling|preferred)\b|"
        r"(?:most\s+)?supported\s+(?:statement|choice|answer|diagnosis)\b|indeed\s+correct\b"
        r")",
        re.IGNORECASE,
    )
    negative = re.compile(
        r"\b(?:not|less\s+likely|unlikely|incorrect|wrong|weaker|less\s+appropriate|"
        r"does\s+not|cannot|can't|would\s+not)\b",
        re.IGNORECASE,
    )

    label_refs = (
        re.compile(rf"\b(?:option|choice)\s+{re.escape(option_lower)}\b"),
        re.compile(rf"\({re.escape(option_lower)}\)"),
        re.compile(rf"(?:^|\s){re.escape(option_lower)}[.):](?:\s|$)"),
    )
    for chunk in chunks:
        has_label = any(pattern.search(chunk) for pattern in label_refs)
        has_choice = bool(
            choice_norm
            and len(choice_tokens) >= 1
            and re.search(rf"(?<!\w){re.escape(choice_norm)}(?!\w)", chunk)
        )
        # For long answer choices, allow a distinctive content-word subset because
        # prose often shortens benchmark choices (e.g. ``high-dose glucocorticoids``).
        if not has_choice and len(choice_tokens) >= 3:
            hits = sum(1 for token in choice_tokens if re.search(rf"\b{re.escape(token)}\b", chunk))
            has_choice = hits >= min(3, len(choice_tokens))
        if not has_choice and distinctive_tokens:
            has_choice = any(
                re.search(rf"\b{re.escape(token)}\b", chunk) for token in distinctive_tokens
            )
        if not (has_label or has_choice):
            continue
        if not positive.search(chunk):
            continue
        # A negation only vetoes when it occurs close to the target reference; this
        # avoids rejecting ``other options are not appropriate; B is best``.
        ref_positions: list[int] = []
        for pattern in label_refs:
            ref_positions.extend(m.start() for m in pattern.finditer(chunk))
        if has_choice and choice_norm:
            ref_positions.extend(m.start() for m in re.finditer(re.escape(choice_norm), chunk))
        veto = False
        for nmatch in negative.finditer(chunk):
            if any(abs(nmatch.start() - pos) <= 55 for pos in ref_positions):
                veto = True
                break
        if not veto:
            return True
    return False


def _semantic_decision(
    text: str,
    choices: Mapping[str, str] | None = None,
    provisional_option: str | None = None,
    final_option: str | None = None,
) -> str | None:
    """Recover the meaning of the reconsideration decision, not its exact wording.

    The protocol asks the model to state whether the provisional answer is retained
    or revised.  Natural outputs frequently say ``remains the best answer``, ``no
    revision is needed``, or ``the initial approach still seems appropriate``.
    Those are semantically explicit retain decisions.  Negated revision phrases are
    handled before the generic ``revise`` token so ``no reason to revise`` cannot be
    misread as a revision.
    """
    normalized = " ".join(text.casefold().split())
    if re.search(
        r"\b(?:no\s+(?:reason|need|basis)\s+to\s+(?:revise|change|switch)|"
        r"no\s+(?:revision|change|changes)\s+(?:(?:is|are)\s+)?(?:needed|necessary|warranted)|"
        r"(?:revision|change)\s+is\s+not\s+(?:needed|necessary|warranted)|"
        r"no\s+changes?\s+to\s+(?:the\s+)?(?:provisional|initial)\s+(?:answer|choice)(?:\s*\([a-z]\))?\s+(?:is|are)\s+(?:needed|necessary|warranted)|"
        r"no\s+(?:new\s+)?(?:information|evidence|fact|finding)s?[^.;\n]{0,80}"
        r"(?:suggest|support|justify|warrant)s?\s+(?:a\s+)?(?:revision|change|switch)|"
        r"no\s+(?:new\s+)?(?:information|evidence|fact|finding)s?[^.;\n]{0,80}"
        r"(?:would|should|could)\s+(?:suggest|support|justify|warrant)\s+"
        r"(?:revising|changing|switching))\b",
        normalized,
    ):
        return "retain"

    if re.search(
        r"\b(?:reaffirm(?:ing|ed)?|confirm(?:ing|ed)?)\s+(?:the\s+)?(?:provisional|initial)\s+(?:answer|choice)\b",
        normalized,
    ):
        return "retain"

    if re.search(
        r"\b(?:the\s+)?(?:initial|original)\s+consideration\b[^.;\n]{0,120}\bstands?\b",
        normalized,
    ):
        return "retain"

    if re.search(
        r"\b(?:(?:the\s+)?(?:provisional|initial|original)\s+(?:answer|choice|conclusion)\s+"
        r"(?:holds?|stands?|remains?\s+(?:supported|valid))|"
        r"no\s+(?:new\s+)?(?:facts?|evidence|findings?|information)\s+"
        r"(?:alter|change|undermine)s?\s+(?:this|the|my|our)?\s*(?:conclusion|answer|choice))\b",
        normalized,
    ):
        return "retain"

    if re.search(
        r"(?:^|[.;:\n])\s*[-*]?\s*(?:retain|keep|maintain)\s+"
        r"(?:(?:the\s+)?(?:provisional|initial)\s+(?:answer|choice)(?:\s+(?:of|as|is))?\s*|"
        r"(?:(?:option|choice)\s+)?[a-z](?:[.):]|\b))",
        normalized,
    ) or re.search(
        r"\b(?:no\s+change\s+(?:in|to)\s+(?:the\s+)?(?:provisional|initial)\s+(?:answer|choice)|"
        r"(?:provisional|initial)\s+(?:answer|choice)[^.;\n]{0,50}\bshould\s+be\s+retained)\b",
        normalized,
    ):
        return "retain"

    direct_role_action = re.search(
        r"(?:^|[.;:\n])\s*[-*]?\s*(retain|keep|maintain|revise|change|switch)\s+"
        r"(?:the\s+)?(?:provisional|initial)\s+(?:answer|choice)\b",
        normalized,
    )
    if direct_role_action is not None:
        action = direct_role_action.group(1)
        return "retain" if action in {"retain", "keep", "maintain"} else "revise"

    # An explicit first-person or decision-level action is stronger than any
    # later implicit wording. In particular, ``I revise my provisional answer``
    # must not be overridden because the revised option is later called
    # ``more appropriate``.
    explicit_action = re.search(
        r"\b(?:decision\s+(?:is\s+)?(?:to\s+)?|(?:i|we)\s+(?:would\s+|will\s+|now\s+)?)"
        r"(retain|keep|maintain|revise|change|switch)\b",
        normalized,
    )
    if explicit_action is not None:
        action = explicit_action.group(1)
        return "retain" if action in {"retain", "keep", "maintain"} else "revise"

    # Explicit answer-level retain language.  Requiring an answer-like subject avoids
    # interpreting statements such as ``the murmur remains the most specific clue``
    # as a decision about the provisional answer.
    answer_subject = (
        r"(?:provisional|initial)\s+(?:answer|choice|diagnosis|treatment|approach)"
        r"|(?:answer|choice|diagnosis|treatment|approach)"
    )
    if re.search(
        rf"\b(?:{answer_subject})\b[^.;\n]{{0,100}}\b"
        r"(?:remains?|stands|is\s+still|still\s+(?:seems|appears)|(?:seems|appears)\s+(?:to\s+be\s+)?|continues?\s+to\s+be|"
        r"is|is\s+(?:strongly\s+)?supported)\b[^.;\n]{0,80}\b"
        r"(?:best|correct|most\s+likely|most\s+appropriate|appropriate|accurate|"
        r"reasonable|rational|supported|compelling|fitting|fit|strongest\s+fit|choice|answer|diagnosis|treatment)?\b",
        normalized,
    ):
        return "retain"

    if re.search(
        r"\b(?:facts?|evidence|findings?|presentation|clinical\s+picture|original\s+facts?)"
        r"[^.;\n]{0,90}\b(?:still\s+|strongly\s+)?(?:support|favor|favour|confirm)\w*"
        r"[^.;\n]{0,60}\b(?:the\s+)?(?:provisional|initial)\s+(?:answer|choice|diagnosis|treatment)\b",
        normalized,
    ):
        return "retain"

    # If the actual provisional option/choice is explicitly named, accept the same
    # retain predicates even when the author omits the words ``provisional answer``.
    if choices is not None and provisional_option is not None and provisional_option in choices:
        raw_label = re.escape(provisional_option.upper())
        # At a sentence/list boundary an uppercase bare label followed immediately
        # by a retain predicate is unambiguous enough to accept (``A remains the
        # most accurate answer``). This is evaluated on raw text so article "a"
        # cannot collide with option A.
        if re.search(
            rf"(?<![A-Za-z]){raw_label}\s+remains?\s+(?:the\s+)?"
            r"(?:best|correct|most\s+likely|most\s+appropriate|most\s+accurate|accurate|appropriate|"
            r"reasonable|preferred|strongest)\b",
            text,
        ):
            return "retain"

        label = re.escape(provisional_option.casefold())
        choice = " ".join(choices[provisional_option].casefold().split())
        # In case-folded text a bare option A/I is indistinguishable from ordinary
        # words. Require option punctuation/parentheses, or use the full answer text.
        subjects = [rf"(?:(?:option|choice)\s+{label}\b|\({label}\)|\b{label}[.)])"]
        if choice:
            subjects.append(re.escape(choice))
        for subject in subjects:
            if re.search(
                rf"(?:{subject})[^.;\n]{{0,80}}\b"
                r"(?:remains?|is\s+still|still\s+(?:seems|appears)|(?:seems|appears)\s+(?:to\s+be\s+)?|continues?\s+to\s+be)\b"
                r"[^.;\n]{0,70}\b(?:best|correct|most\s+likely|appropriate|accurate|"
                r"reasonable|rational|supported|compelling|fitting|fit|strongest\s+fit|answer|choice|diagnosis|treatment)\b",
                normalized,
            ) or re.search(
                r"\b(?:still\s+)?(?:support|favor|favour|confirm|point)\w*"
                r"[^.;\n]{0,70}" + rf"(?:{subject})",
                normalized,
            ):
                return "retain"

    # A direct option-labelled evaluative relation is also a genuine conclusion.
    # Keep this on raw visible text: normalize_text intentionally strips leading
    # option labels, which would otherwise erase the subject in forms such as
    # ``Option J makes it highly suitable``.
    if (
        choices is not None
        and provisional_option is not None
        and provisional_option in choices
        and final_option is not None
        and final_option in choices
    ):
        for target in (final_option, provisional_option):
            target_label = re.escape(target.upper())
            if re.search(
                rf"\b(?:option|choice)\s+{target_label}\b[^.;\n]{{0,90}}\b"
                r"makes?\s+it\s+(?:highly\s+)?(?:suitable|appropriate|compelling|preferred)\b",
                text,
                re.IGNORECASE,
            ):
                return "retain" if target == provisional_option else "revise"

    # The protocol asks for a real reconsideration decision, not a magic keyword.
    # If the section itself clearly concludes on an option, infer retain/revise by
    # comparing that concluded option with the recovered provisional answer. The
    # judge separately verifies consistency with the final answer.
    if choices is not None and provisional_option is not None and provisional_option in choices:
        concluded = _reconsideration_selected_option(text, choices)
        if concluded == provisional_option:
            return "retain"
        if (
            concluded in choices
            and final_option is not None
            and final_option in choices
            and concluded == final_option
        ):
            return "revise"
        if (
            final_option is not None
            and final_option in choices
            and _reconsideration_supports_option(text, choices, final_option)
        ):
            return "retain" if final_option == provisional_option else "revise"
        if (
            final_option is not None
            and final_option in choices
            and final_option == provisional_option
            and re.search(
                r"\b(?:provisional\s+answer\s+(?:is\s+)?confirmed|"
                r"(?:therefore|thus)[^.;\n]{0,60}provisional\s+answer\s+(?:is\s+)?confirmed|"
                r"(?:remains?|is\s+still|still\s+(?:seems|appears))\s+(?:the\s+)?"
                r"(?:best|correct|most\s+likely|most\s+appropriate|most\s+accurate)\s+"
                r"(?:answer|choice|diagnosis|treatment|explanation))\b",
                normalized,
            )
        ):
            return "retain"

    if (
        choices is not None
        and provisional_option is not None
        and provisional_option in choices
        and final_option is not None
        and final_option in choices
    ):
        token_count = len(normalize_text(text).split())
        continuity = re.search(
            r"\b(?:upon\s+reconsideration|after\s+reconsideration|reconsidering|re-evaluating|"
            r"revisiting|original\s+facts?|initial\s+analysis|original\s+analysis)\b",
            normalized,
        )
        retain_signal = re.search(
            r"\b(?:still\s+(?:points?|supports?|favors?|favours?)|continues?\s+to\s+(?:support|favor|favour|point)|"
            r"reinforc(?:e|es|ed|ing)|strongly\s+(?:supports?|suggests?)|supports?\s+(?:this|the)\s+choice|"
            r"supports?\s+(?:the\s+)?(?:initial|original)\s+(?:conclusion|impression)|"
            r"remains?\s+(?:the\s+)?(?:most\s+likely|most\s+plausible|best|paramount)|"
            r"(?:the\s+)?most\s+likely\s+(?:condition|diagnosis|answer|choice|explanation)\b|"
            r"focus\s+should\s+remain|(?:initial|original)\s+(?:analysis|conclusion|impression|consideration)\s+stands?)\b",
            normalized,
        )
        explicit_continuity_retain = re.search(
            r"\b(?:support(?:s|ed|ing)?|reinforc(?:e|es|ed|ing)|confirm(?:s|ed|ing)?)\s+"
            r"(?:the\s+)?(?:initial|original)\s+(?:conclusion|impression|answer|choice)\b",
            normalized,
        )
        if (
            token_count >= 8
            and continuity
            and (retain_signal or explicit_continuity_retain)
            and final_option == provisional_option
        ):
            return "retain"

    match = SEMANTIC_DECISION_RE.search(text)
    if match is None:
        return None
    values = [group.casefold() for group in match.groups() if group is not None]
    action = next(
        (
            value
            for value in values
            if value
            in {
                "retain",
                "revise",
                "retaining",
                "revising",
                "maintaining",
                "retained",
                "revised",
                "maintained",
                "keep",
                "change",
                "switch",
                "maintain",
                "kept",
                "unchanged",
                "changed",
                "switched",
                "remains",
                "remain",
                "stands",
                "still best",
                "still correct",
                "still most likely",
                "switching",
                "changing",
            }
        ),
        None,
    )
    if action in {
        "retain",
        "retaining",
        "keep",
        "maintain",
        "maintaining",
        "retained",
        "kept",
        "maintained",
        "unchanged",
        "remains",
        "remain",
        "stands",
        "still best",
        "still correct",
        "still most likely",
    }:
        return "retain"
    if action in {
        "revise",
        "revising",
        "change",
        "switch",
        "revised",
        "changed",
        "switched",
        "switching",
        "changing",
    }:
        return "revise"
    return None


def _semantic_marker_strength(document: str, match: re.Match[str]) -> int:
    """Rank likely section headings above body-internal role labels.

    A presentation marker is strong only when it behaves like a boundary.  In
    particular, ``### **4. Best Alternative**`` is a heading, while
    ``**Best alternative: A. ...**`` is answer content and must not truncate the
    section merely because it is bold.
    """
    marker = match.group(0)
    stripped = marker.lstrip()
    if stripped.startswith("<"):
        return 6
    line_end = document.find("\n", match.end())
    if line_end < 0:
        line_end = len(document)
    remainder = document[match.end() : line_end].strip()
    if "#" in marker or re.search(r"\b(?:step[ \t]*)?\d+[.:)\-]", marker, re.IGNORECASE):
        return 5
    if ("**" in marker or "__" in marker) and not remainder:
        return 4
    return 3 if not remainder else 1


def _semantic_matches(
    document: str,
    exact: Mapping[str, tuple[re.Match[str], ...]],
) -> dict[str, tuple[re.Match[str], ...]]:
    """Return format-tolerant role boundaries without false body-label duplicates."""
    recovered: dict[str, tuple[re.Match[str], ...]] = {}
    for tag in CONTENT_TAGS:
        headings = tuple(SEMANTIC_HEADING_PATTERNS[tag].finditer(document))
        if tag == "answer_changing_change":
            # Merge the broader numbered fallback without double-counting a line
            # already recognized by the ordinary semantic-heading pattern.
            all_headings = (*headings, *STEP6_NUMBERED_HEADING_FALLBACK_RE.finditer(document))
            headings = tuple(
                match
                for _, match in sorted(
                    {(match.start(), match.end()): match for match in all_headings}.items()
                )
            )
        combined = sorted((*exact[tag], *headings), key=lambda match: match.start())
        if combined:
            best_strength = max(_semantic_marker_strength(document, match) for match in combined)
            recovered[tag] = tuple(
                match
                for match in combined
                if _semantic_marker_strength(document, match) == best_strength
            )
            continue
        # A reversed opening slash can still visibly name the requested role. Use
        # one such boundary only when no ordinary label for that role exists.
        closings = tuple(CLOSING_TAG_PATTERNS[tag].finditer(document))
        recovered[tag] = closings[:1]
    return recovered


def _local_semantic_body(
    document: str,
    matches: Mapping[str, tuple[re.Match[str], ...]],
    start: re.Match[str],
) -> str:
    """Visible text immediately owned by one candidate marker.

    This deliberately stops at the next *candidate* role marker, not merely the
    next marker selected for the final trajectory.  It is used only to decide
    which repeated eight-role scaffold is the substantive one.  A blank template
    therefore cannot steal the content of a later filled template.
    """
    end = _segment_end(document, matches, start)
    return _plain_text(document[start.end() : end])


def _candidate_sequence_score(
    document: str,
    semantic_matches: Mapping[str, tuple[re.Match[str], ...]],
    tag: str,
    match: re.Match[str],
    choices: Mapping[str, str],
) -> float:
    """Score one candidate role boundary for coherent-sequence recovery.

    Heading syntax is only a boundary-confidence signal.  Most of the score comes
    from whether the candidate actually owns generated content and, for answer
    roles, a recoverable semantic field.  The tiny position term breaks otherwise
    exact ties in favor of the later scaffold, which is the conventional pattern
    when a model prints a template and then a filled response.
    """
    body = _strip_guidance_echo(
        tag,
        _local_semantic_body(document, semantic_matches, match),
    )
    body_tokens = len(normalize_text(body).split())
    score = float(_semantic_marker_strength(document, match) * 10)
    score += min(30.0, float(body_tokens))
    if body_tokens >= 3:
        score += 8.0
    if tag in {"provisional_answer", "second_best", "final_answer"}:
        if _semantic_option(body, choices, tag) is not None:
            score += 12.0
    elif tag == "rereasoning" and _semantic_decision(body, choices) in VALID_DECISIONS:
        score += 12.0
    if document:
        score += (match.start() / len(document)) * 0.001
    return score


def _coherent_semantic_sequence(
    document: str,
    semantic_matches: Mapping[str, tuple[re.Match[str], ...]],
    choices: Mapping[str, str],
) -> dict[str, re.Match[str]] | None:
    """Choose one ordered eight-role trajectory when repeated scaffolds exist.

    The old recovery chose the first strongest marker for each role independently.
    That fails on two common natural outputs: an empty eight-heading template
    followed by a filled response, and role-like subheadings inside an earlier
    section (for example ``Key distinction`` inside Implications).  Here the eight
    boundaries are selected jointly.  If no complete ordered sequence exists we
    fall back to the legacy per-role recovery so malformed outputs remain
    diagnosable rather than being silently repaired.
    """
    if any(not semantic_matches[tag] for tag in CONTENT_TAGS):
        return None
    if all(len(semantic_matches[tag]) == 1 for tag in CONTENT_TAGS):
        only = {tag: semantic_matches[tag][0] for tag in CONTENT_TAGS}
        starts = [only[tag].start() for tag in CONTENT_TAGS]
        return only if starts == sorted(starts) else None

    # Dynamic programming over role order.  Each state stores the best cumulative
    # score ending at a concrete marker plus the chosen prefix.  Candidate counts
    # are tiny in practice, but this remains O(sum n_i*n_{i-1}) rather than
    # enumerating the Cartesian product of repeated templates.
    first_tag = CONTENT_TAGS[0]
    states: list[tuple[re.Match[str], float, tuple[re.Match[str], ...]]] = [
        (
            match,
            _candidate_sequence_score(document, semantic_matches, first_tag, match, choices),
            (match,),
        )
        for match in semantic_matches[first_tag]
    ]

    for tag in CONTENT_TAGS[1:]:
        next_states: list[tuple[re.Match[str], float, tuple[re.Match[str], ...]]] = []
        for match in semantic_matches[tag]:
            valid = [state for state in states if state[0].start() < match.start()]
            if not valid:
                continue
            previous = max(valid, key=lambda item: item[1])
            next_states.append(
                (
                    match,
                    previous[1]
                    + _candidate_sequence_score(document, semantic_matches, tag, match, choices),
                    (*previous[2], match),
                )
            )
        if not next_states:
            return None
        states = next_states

    selected = max(states, key=lambda item: item[1])[2]
    return dict(zip(CONTENT_TAGS, selected, strict=True))


def _selected_segment_end(
    document: str,
    selected: Mapping[str, re.Match[str] | None],
    semantic_matches: Mapping[str, tuple[re.Match[str], ...]],
    tag: str,
    start: re.Match[str],
) -> int:
    """End one selected role without swallowing an abandoned duplicate scaffold.

    Usually the next boundary is the next marker selected for the recovered
    trajectory. A special case occurs when an output prints a complete empty
    template and then fills only a suffix of it. The optimal trajectory can then
    use an early prefix (for example Facts/Implications) and a later filled suffix
    (Provisional answer through Final answer). In that case, ending the prefix at
    the *later* selected marker would incorrectly absorb the intervening empty
    template and bridge prose.

    Therefore, if the immediate next role has an earlier duplicate marker between
    this role and the selected next marker, treat that earlier marker as a boundary
    only when it is at least as strong a presentation marker as the selected one.
    This preserves protection against weak body-internal role phrases while making
    genuine numbered/Markdown/XML duplicate templates segment correctly.
    """
    later = [
        match.start()
        for match in selected.values()
        if match is not None and match.start() > start.start()
    ]
    end = min(later) if later else len(document)

    try:
        index = CONTENT_TAGS.index(tag)
    except ValueError as exc:
        raise AssertionError(f"unknown protocol role: {tag}") from exc
    if index + 1 >= len(CONTENT_TAGS):
        return end

    next_tag = CONTENT_TAGS[index + 1]
    next_selected = selected.get(next_tag)
    if next_selected is None or next_selected.start() <= start.start():
        return end

    selected_strength = _semantic_marker_strength(document, next_selected)
    intervening = [
        candidate.start()
        for candidate in semantic_matches[next_tag]
        if (
            start.start() < candidate.start() < next_selected.start()
            and _semantic_marker_strength(document, candidate) >= selected_strength
        )
    ]
    if intervening:
        end = min(end, min(intervening))
    return end


def recover_protocol(raw: str, choices: Mapping[str, str]) -> RecoveredProtocol:
    """Recover ordered role content independently of the chosen heading syntax."""
    document = normalize_outer_code_fence(raw)
    matches = {tag: tuple(START_TAG_PATTERNS[tag].finditer(document)) for tag in CONTENT_TAGS}
    semantic_matches = _semantic_matches(document, matches)
    first = {tag: _first_match(matches, tag) for tag in CONTENT_TAGS}
    coherent = _coherent_semantic_sequence(document, semantic_matches, choices)
    semantic_first = (
        coherent
        if coherent is not None
        else {tag: _first_match(semantic_matches, tag) for tag in CONTENT_TAGS}
    )

    recovered: dict[str, str] = {}
    for tag in CONTENT_TAGS:
        start = semantic_first[tag]
        if start is None:
            recovered[tag] = ""
            continue
        end = (
            _selected_segment_end(
                document,
                semantic_first,
                semantic_matches,
                tag,
                start,
            )
            if coherent is not None
            else _segment_end(document, semantic_matches, start)
        )
        visible = _plain_text(document[start.end() : end])
        recovered[tag] = _strip_guidance_echo(tag, visible)

    start_counts = {tag: len(matches[tag]) for tag in CONTENT_TAGS}
    marker_schema_valid = {
        tag: _marker_schema_valid(tag, matches[tag], choices) for tag in CONTENT_TAGS
    }
    semantic_start_counts = {tag: len(semantic_matches[tag]) for tag in CONTENT_TAGS}
    starts = [match.start() for tag in CONTENT_TAGS if (match := first[tag]) is not None]
    ordered = len(starts) == len(CONTENT_TAGS) and starts == sorted(starts)
    semantic_starts = [
        match.start() for tag in CONTENT_TAGS if (match := semantic_first[tag]) is not None
    ]
    semantic_ordered = len(semantic_starts) == len(CONTENT_TAGS) and semantic_starts == sorted(
        semantic_starts
    )
    preanswer_tags = ("facts", "implications", "provisional_answer")
    preanswer_starts = [
        match.start() for tag in preanswer_tags if (match := first[tag]) is not None
    ]
    preanswer_ordered = len(preanswer_starts) == len(preanswer_tags) and preanswer_starts == sorted(
        preanswer_starts
    )
    semantic_preanswer_starts = [
        match.start() for tag in preanswer_tags if (match := semantic_first[tag]) is not None
    ]
    semantic_preanswer_ordered = len(semantic_preanswer_starts) == len(
        preanswer_tags
    ) and semantic_preanswer_starts == sorted(semantic_preanswer_starts)

    provisional = _option(first["provisional_answer"], choices)
    second = _option(first["second_best"], choices)
    final = _option(first["final_answer"], choices)
    decision = _decision(first["rereasoning"])

    # Parse semantic choices from the generated role body only.  The heading text
    # names the requested role; it is not evidence that the model actually selected
    # an option. XML option attributes are already recovered above by ``_option``.
    semantic_provisional = provisional or _semantic_option(
        recovered["provisional_answer"], choices, "provisional_answer"
    )
    semantic_second = second or _semantic_option(recovered["second_best"], choices, "second_best")
    semantic_final = final or _semantic_option(recovered["final_answer"], choices, "final_answer")
    semantic_decision = decision or _semantic_decision(
        recovered["rereasoning"], choices, semantic_provisional, semantic_final
    )

    errors: list[str] = []
    for tag in CONTENT_TAGS:
        count = semantic_start_counts[tag]
        if count == 0:
            errors.append(f"missing reasoning role: {tag}")
        elif count > 1:
            errors.append(f"duplicate reasoning role: {tag} ({count})")
    if not semantic_ordered:
        errors.append("reasoning roles are missing, duplicated, or out of order")
    if semantic_provisional is None:
        errors.append("provisional answer option is absent or invalid")
    if semantic_second is None:
        errors.append("best alternative option is absent or invalid")
    elif semantic_provisional is not None and semantic_second == semantic_provisional:
        errors.append("best alternative must differ from provisional answer")
    if semantic_final is None:
        errors.append("final answer option is absent or invalid")
    if semantic_decision is None:
        errors.append("reconsideration must say retain or revise")
    errors.extend(
        f"recovered {tag} content is empty"
        for tag in CONTENT_TAGS
        if tag != "final_answer" and not recovered[tag]
    )

    provisional_match = semantic_first["provisional_answer"]
    return RecoveredProtocol(
        document=document,
        sections=recovered,
        start_counts=start_counts,
        marker_schema_valid=marker_schema_valid,
        semantic_start_counts=semantic_start_counts,
        ordered_starts=ordered,
        preanswer_ordered=preanswer_ordered,
        semantic_ordered_starts=semantic_ordered,
        semantic_preanswer_ordered=semantic_preanswer_ordered,
        provisional_option=provisional,
        second_best_option=second,
        final_option=final,
        decision=decision,
        semantic_provisional_option=semantic_provisional,
        semantic_second_best_option=semantic_second,
        semantic_final_option=semantic_final,
        semantic_decision=semantic_decision,
        provisional_start=provisional_match.start() if provisional_match is not None else None,
        errors=tuple(errors),
    )
