"""Lark-based parser for the markdown rule domain-specific language.

Purpose:
    Parse authored markdown rules, validate block/section structure, expand
    template variants, and build typed rule abstract-syntax-tree objects.

Architectural role:
    Main source-text ingestion boundary inside the rules subsystem.

Current architecture notes:
    This parser is logically well-scoped, though the resulting
    abstract-syntax-tree still embeds shared engine match-tree contracts.

"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal, Protocol

from lark import Lark, Token, Tree
from lark.exceptions import UnexpectedInput

from answer_engineering.config.engine_defaults import (
    MatchDefaults,
    PolicyDefaults,
    RuleDefaults,
    ScopeDefaults,
)
from answer_engineering.engine.proposal.match_tree.match_tree import (
    MatchAll,
    MatchAndThen,
    MatchAny,
    MatchNot,
    MatchTerm,
    MatchTree,
)
from answer_engineering.rules.matching.options import ResolvedMatchOptions
from answer_engineering.rules.parse.ast import (
    AfterRuleAST,
    AvoidEditSpecAST,
    AvoidRuleAST,
    ForceRuleAST,
    MatchOptionsAST,
    ReplaceRuleAST,
    RuleAST,
    RulesetAST,
    ScopeAST,
    SetMatchAST,
    register_ruleset_text_parser,
)
from answer_engineering.rules.parse.errors import (
    RulesSyntaxError,
)
from answer_engineering.rules.parse.match_options import (
    resolve_match_options,
)

Mode = Literal["any", "all", "none", "incomplete"]


@dataclass(frozen=True, slots=True)
class _RuleBlock:
    """Intermediate parsed rule block before abstract-syntax-tree construction.

    Purpose:
        Hold one markdown rule heading together with its normalized
        section-to-bullets mapping after grammar parsing but before rule-family
        conversion.

    """

    title: str
    sections: dict[str, list[str]]
    section_match_options: dict[str, MatchOptionsAST]


@dataclass(frozen=True, slots=True)
class _TemplateBullet:
    """Parsed representation of one bullet that may participate in template.

    Purpose:
        Hold the concrete variant values extracted from one bullet together with
        the template-dimension marker, if any.

    """

    values: tuple[str, ...]
    dimension: int | None


@dataclass(frozen=True, slots=True)
class _SectionHeaderInlineValue:
    """Normalized section-header value with optional inline content.

    Purpose:
        Carry the section name parsed from a markdown-like rule header together
        with any value written on the same line.

    Architectural role:
        Parser-internal value object used before abstract-syntax-tree
        construction. It prevents the grammar transformer from passing loosely
        shaped tuples through later parsing code.

    Inputs (architectural provenance):
        Values are produced from Lark parse results for rule section headers.

    Outputs (downstream usage):
        Supplies normalized section names and inline values to rule-block
        assembly.

    Invariants/constraints:
        The object remains tuple-unpack compatible only as a parser convenience.
        Downstream abstract-syntax-tree code should consume named fields where
        practical.

    """

    section_name: str
    inline_value: str | None
    modifiers: tuple[str, ...] = ()

    def __iter__(self):
        """Yield ``(section_name, inline_value)`` for tuple-style unpacking."""
        yield self.section_name
        yield self.inline_value


class ConditionedTerms:
    """Canonical operator-keyed term container.

    Purpose:
        Normalize section terms into ``any``, ``all``, ``none``, and
        ``incomplete`` buckets for later match-expression construction.

    Architectural role:
        Parser-owned helper value that bridges normalized section data to
        match-tree builders.

    """

    __slots__ = ("_terms_by_mode",)
    MODES: tuple[Mode, ...] = ("any", "all", "none", "incomplete")

    def __init__(
        self,
        *,
        section_values: Mapping[str, list[str]] | None = None,
        label: str | None = None,
        default_op: Mode = "any",
        terms_by_mode: Mapping[Mode, Iterable[str]] | None = None,
    ) -> None:
        """Normalize section values into operator-keyed term buckets.

        Purpose:
            Convert parsed section entries into the canonical include/exclude
            buckets consumed by condition builders.

        Architectural role:
            Parser-normalization constructor between raw markdown section syntax
            and the typed rule abstract-syntax-tree construction path.

        Inputs (architectural provenance):
            Receives section values parsed from authored rules, including
            entries that may carry operator-specific meaning.

        Outputs (downstream usage):
            Stores normalized term groups used when building guard, trigger, and
            match conditions.

        Invariants/constraints:
            Terms must be grouped without changing their authored text meaning.
            Later compiler stages should not need to re-interpret raw section
            syntax.

        """
        resolved: dict[Mode, tuple[str, ...]] = {
            "any": tuple(),
            "all": tuple(),
            "none": tuple(),
            "incomplete": tuple(),
        }
        if terms_by_mode is not None:
            for mode in resolved:
                resolved[mode] = tuple(terms_by_mode.get(mode, ()))
        if section_values is not None and label is not None:
            default_terms: tuple[str, ...] = tuple()
            section_modes: tuple[tuple[str, Mode | None], ...] = (
                (label, None),
                (f"{label}:any", "any"),
                (f"{label}:all", "all"),
                (f"{label}:none", "none"),
                (f"{label}:incomplete", "incomplete"),
            )
            for section_key, op in section_modes:
                fetched = tuple(section_values.get(section_key, ()))
                if not fetched:
                    continue
                if op is None:
                    default_terms = fetched
                else:
                    resolved[op] = fetched + resolved[op]
            if default_terms:
                resolved[default_op] = default_terms + resolved[default_op]
        self._terms_by_mode = resolved

    def terms(self, mode: Mode) -> tuple[str, ...]:
        """Return the stored term tuple for one condition mode.

        Purpose:
            Expose one normalized operator bucket from ``ConditionedTerms``
            without leaking or mutating the container's internal mapping
            structure.

        Architectural role:
            Parser-owned accessor used by gate and avoid-expression builders.

        Inputs (architectural provenance):
            Called by parser helpers that are assembling structural match-tree
            expressions.

        Outputs (downstream usage):
            Returns the tuple stored for one of the canonical modes ``any``,
            ``all``, ``none``, or ``incomplete``.

        Invariants/constraints:
            Lookups preserve the canonical normalization performed at
            construction time.

        """
        return self._terms_by_mode[mode]

    def items(self) -> tuple[tuple[Mode, tuple[str, ...]], ...]:
        """Return all normalized condition buckets in canonical mode order.

        Purpose:
            Provide deterministic iteration over ``ConditionedTerms`` so
            expression builders can assemble match trees without caring about
            the internal dictionary representation.

        Architectural role:
            Parser-owned accessor used by generic term-to-expression helpers.

        Outputs (downstream usage):
            Returns ``(mode, terms)`` pairs in the fixed order declared by
            ``MODES``.

        Invariants/constraints:
            Every canonical mode appears exactly once even when its term tuple
            is empty.

        """
        return tuple((mode, self._terms_by_mode[mode]) for mode in self.MODES)


@dataclass(frozen=True, slots=True)
class _RuleMods:
    """Parsed rule-header payload.

    Purpose:
        Hold the normalized rule kind, modifier list, and textual target
        extracted from one markdown rule heading before family-specific
        abstract-syntax-tree construction.

    Architectural role:
        Parser-owned intermediate value between header parsing and rule-family
        conversion.

    Inputs (architectural provenance):
        Produced by ``MarkdownRulesParser._mods`` from the authored ``##
        Kind(mods): target`` header syntax.

    Outputs (downstream usage):
        Consumed immediately by ``MarkdownRulesParser._to_rule``.

    """

    kind: str
    modifiers: tuple[str, ...]
    target: str

    def __iter__(self):
        """Yield ``(kind, modifiers, target)`` for unpacking.

        Purpose:
            Let parser internals unpack the parsed rule-header payload directly
            at the point where rule-family dispatch happens.

        Architectural role:
            Tiny convenience method on a parser-owned intermediate value.

        Outputs (downstream usage):
            Used by ``MarkdownRulesParser._to_rule`` and similar internal call
            sites.

        Invariants/constraints:
            Always yields exactly three values in header-field order.

        """
        yield self.kind
        yield self.modifiers
        yield self.target


class RulesetTreeParser(Protocol):
    """Protocol for the underlying grammar parser object.

    Purpose:
        Describe the minimal tree-parser capability ``MarkdownRulesParser``
        depends on from its concrete Lark-backed parser.

    Architectural role:
        Small collaboration contract inside the parser boundary.

    Inputs (architectural provenance):
        Implemented by the concrete parser object created during
        ``MarkdownRulesParser`` construction.

    Outputs (downstream usage):
        Lets the higher-level parser own grammar orchestration without tying
        every use site to the concrete parser type.

    """

    def parse(self, text: str) -> Tree[Token]:
        """Parse source text into a syntax tree.

        Purpose:
            Declare the parser method required by ``MarkdownRulesParser`` for
            tree construction.

        """
        raise NotImplementedError


class MarkdownRulesParser:
    """Markdown-to-abstract-syntax-tree parser for the rules language.

    Purpose:
        Own grammar loading, syntax checking, template expansion, block
        normalization, and final conversion into ``RulesetAST``.

    Architectural role:
        Main behavior-owning parser object in the rules boundary.

    Invariants/constraints:
        Non-empty lines that are not part of recognized rule syntax are rejected
        rather than silently ignored.

    """

    def __init__(self) -> None:
        """Load the grammar file and construct the concrete markdown parser.

        Purpose:
            Build the reusable parser object that turns authored
            answer-engineering rules markdown into the repository
            abstract-syntax-tree.

        Architectural role:
            Construction boundary for the rules parse subsystem. It owns grammar
            loading and parser initialization so call sites do not depend on
            Lark setup details.

        Inputs (architectural provenance):
            Reads the packaged grammar resource that defines the supported
            ruleset language.

        Outputs (downstream usage):
            Stores the initialized parser used by `parse` for notebook rulesets,
            compiled policies, and tests.

        Invariants/constraints:
            Parser construction should fail early if the grammar cannot be
            loaded or is invalid. Successful instances should be ready to parse
            many rulesets.

        """
        grammar_path = Path(__file__).with_name("grammar") / "ae_rules.lark"
        self._parser: RulesetTreeParser = Lark(
            grammar_path.read_text(encoding="utf-8"),
            parser="lalr",
            propagate_positions=True,
            maybe_placeholders=False,
        )

    def parse(self, text: str) -> RulesetAST:
        """Parse markdown rules document into canonical abstract-syntax-tree.

        Purpose:
            Run syntax parsing, reject stray unparsed lines, expand templates,
            and convert normalized blocks into typed rule abstract-syntax-tree
            objects.

        """
        try:
            tree = self._parser.parse(text)
        except UnexpectedInput as exc:
            raise _syntax_error(text, exc) from exc

        _raise_on_unparsed_lines(tree)

        blocks = self._blocks_from_tree(tree)
        blocks = self._expand_template_blocks(blocks)
        rules = tuple(
            rule
            for idx, block in enumerate(blocks)
            if (rule := self._to_rule(block, idx)) is not None
        )
        return RulesetAST(rules)

    def _blocks_from_tree(self, tree: Tree[Token]) -> list[_RuleBlock]:
        """Convert the grammar tree into normalized rule blocks.

        Purpose:
            Walk the Lark parse tree, extract each rule heading, normalize
            section names, and collect inline and bullet values into
            parser-owned ``_RuleBlock`` objects.

        Architectural role:
            Structural extraction helper inside ``MarkdownRulesParser``.

        Inputs (architectural provenance):
            Receives the parse tree produced by the underlying grammar parser
            after syntax validation.

        Outputs (downstream usage):
            Returns the normalized block list later consumed by template
            expansion and final abstract-syntax-tree conversion.

        Invariants/constraints:
            Only recognized rule items become blocks; unrelated tree shapes are
            skipped.

        """
        blocks: list[_RuleBlock] = []
        for item in tree.children:
            if not isinstance(item, Tree) or item.data != "item":
                continue
            first = item.children[0] if item.children else None
            if not isinstance(first, Tree) or first.data != "block":
                continue
            heading = ""
            sections: dict[str, list[str]] = {}
            section_match_options: dict[str, MatchOptionsAST] = {}
            for child in first.children:
                if isinstance(child, Tree) and child.data == "heading":
                    heading = _token_value(child, "HEADING")
                elif isinstance(child, Tree) and child.data == "section":
                    sec_name = _token_value(child, "SECTION_HEADER")
                    header = self._section_header_and_inline_value(sec_name)
                    norm = header.section_name
                    section_base = norm.split(":", 1)[0]
                    if header.modifiers and section_base not in {
                        "prefix",
                        "postfix",
                        "prompt",
                        "connector",
                    }:
                        raise RulesSyntaxError(
                            (
                                "Unsupported modifier "
                                f"'{header.modifiers[0]}' in "
                                f"{section_base.title()} section. This section "
                                "does not support matching modifiers."
                            ),
                            line=1,
                            column=1,
                            snippet=sec_name,
                        )
                    section_match_options[norm] = _parse_match_options(
                        header.modifiers,
                        context=f"{norm} section",
                        allow_word=True,
                    )
                    section_match_options[section_base] = section_match_options[
                        norm
                    ]
                    inline_value = header.inline_value
                    values: list[str] = []
                    if inline_value:
                        values.append(inline_value)
                    for sub in child.children:
                        if isinstance(sub, Tree) and sub.data == "bullets":
                            values.extend(_bullets(sub))
                    sections[norm] = values
            if heading:
                blocks.append(
                    _RuleBlock(
                        title=heading,
                        sections=sections,
                        section_match_options=section_match_options,
                    )
                )
        return blocks

    def _expand_template_blocks(
        self, blocks: list[_RuleBlock]
    ) -> list[_RuleBlock]:
        expanded: list[_RuleBlock] = []
        for block in blocks:
            expanded.extend(self._expand_single_template_block(block))
        return expanded

    def _expand_single_template_block(
        self, block: _RuleBlock
    ) -> list[_RuleBlock]:
        parsed_sections: dict[str, list[_TemplateBullet]] = {}
        dimension_variant_counts: dict[int, int] = {}
        for section_name, bullets in block.sections.items():
            parsed_bullets: list[_TemplateBullet] = []
            for bullet in bullets:
                parsed = self._split_template_variants(bullet)
                if any(not item.strip() for item in parsed.values):
                    raise RulesSyntaxError(
                        "Invalid rules syntax",
                        line=1,
                        column=1,
                        snippet=bullet,
                    )
                if parsed.dimension is not None:
                    existing_count = dimension_variant_counts.get(
                        parsed.dimension
                    )
                    if existing_count is None:
                        dimension_variant_counts[parsed.dimension] = len(
                            parsed.values
                        )
                    elif existing_count != len(parsed.values):
                        raise RulesSyntaxError(
                            "Invalid rules syntax",
                            line=1,
                            column=1,
                            snippet=bullet,
                        )
                parsed_bullets.append(parsed)
            parsed_sections[section_name] = parsed_bullets

        if not dimension_variant_counts:
            sections: dict[str, list[str]] = {}
            for section_name, bullets in parsed_sections.items():
                sections[section_name] = [
                    parsed.values[0] for parsed in bullets
                ]
            return [
                _RuleBlock(
                    title=block.title,
                    sections=sections,
                    section_match_options=block.section_match_options,
                )
            ]

        out: list[_RuleBlock] = []
        dimensions = tuple(sorted(dimension_variant_counts.keys()))
        index_ranges = [
            range(dimension_variant_counts[dim]) for dim in dimensions
        ]
        for combo in product(*index_ranges):
            selected_index = dict(zip(dimensions, combo, strict=True))
            sections: dict[str, list[str]] = {}
            for section_name, bullets in parsed_sections.items():
                concrete: list[str] = []
                for parsed in bullets:
                    if parsed.dimension is None:
                        concrete.append(parsed.values[0])
                        continue
                    concrete.append(
                        parsed.values[selected_index[parsed.dimension]]
                    )
                sections[section_name] = concrete
            out.append(
                _RuleBlock(
                    title=block.title,
                    sections=sections,
                    section_match_options=block.section_match_options,
                )
            )
        return out

    def _split_template_variants(self, text: str) -> _TemplateBullet:
        """Parse one bullet for template-variant expansion markers.

        Purpose:
            Recognize ``|`` / ``||`` style template dimensions, validate
            consistent run lengths, and return the concrete segment values for
            later cartesian expansion.

        """
        escaped = False
        runs: list[int] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if escaped:
                escaped = False
                i += 1
                continue
            if ch == "\\":
                escaped = True
                i += 1
                continue
            if ch != "|":
                i += 1
                continue

            j = i
            while j < len(text) and text[j] == "|":
                j += 1
            runs.append(j - i)
            i = j

        if not runs:
            return _TemplateBullet(
                values=(_unescape_template_text(text).strip(),),
                dimension=None,
            )
        if len(set(runs)) != 1:
            raise RulesSyntaxError(
                "Invalid rules syntax",
                line=1,
                column=1,
                snippet=text,
            )

        dimension = runs[0]
        segments = [
            _unescape_template_text(part).strip()
            for part in re.split(rf"(?<!\\)\|{{{dimension}}}", text)
        ]
        is_template = len(segments) > 1
        return _TemplateBullet(
            values=tuple(segments), dimension=dimension if is_template else None
        )

    def _section_header_and_inline_value(
        self, header: str
    ) -> _SectionHeaderInlineValue:
        left, right = header.partition(":")[::2]
        inline_value = right.strip() or None
        left_stripped = left.strip()
        open_idx = left_stripped.find("(")
        close_idx = left_stripped.rfind(")")
        if open_idx != -1 and close_idx > open_idx:
            label_raw = left_stripped[:open_idx]
            mods_raw = left_stripped[open_idx + 1 : close_idx]
        else:
            label_raw = left_stripped
            mods_raw = ""
        if not label_raw.strip():
            return _SectionHeaderInlineValue(
                left_stripped.lower().replace(" ", "_"), inline_value
            )
        label = label_raw.strip().lower().replace(" ", "_")
        raw_mods = tuple(
            s.strip().lower() for s in mods_raw.split(",") if s.strip()
        )
        mode_mods = {"any", "all", "none", "incomplete", "partial", "missing"}
        normalized_mode = tuple(
            "incomplete" if mod in {"partial", "missing"} else mod
            for mod in raw_mods
            if mod in mode_mods
        )
        match_mods = tuple(mod for mod in raw_mods if mod not in mode_mods)
        if len(set(normalized_mode)) > 1:
            raise RulesSyntaxError(
                "Contradictory section set-mode modifiers",
                line=1,
                column=1,
                snippet=header,
            )
        section_name = (
            f"{label}:{normalized_mode[0]}" if normalized_mode else label
        )
        return _SectionHeaderInlineValue(
            section_name, inline_value, modifiers=match_mods
        )

    def _to_rule(self, block: _RuleBlock, idx: int):
        """Convert one normalized rule block into the matching rule-family AST.

        Purpose:
            Resolve fire policy, scope, section semantics, and family-specific
            fields for replace, after, avoid, and force rules.

        """
        mods_payload = _mods(block.title)
        kind = mods_payload.kind
        mods = mods_payload.modifiers
        target = mods_payload.target
        if not kind:
            return None
        _reject_item_modifiers_in_non_matching_sections(block)
        (
            fire_mods,
            match_modifiers,
            behavior_mods,
        ) = _partition_rule_modifiers(
            kind=kind, modifiers=mods, title=block.title
        )
        rule_match_options = _parse_match_options(
            match_modifiers,
            context=f"{kind.title()} rule header",
            allow_word=True,
        )

        default_fire: Literal["once", "repeat"] = (
            RuleDefaults().fire_repeat
            if kind == "avoid"
            else RuleDefaults().fire_once
        )
        fire: Literal["once", "repeat"]
        if "repeat" in fire_mods:
            fire = RuleDefaults().fire_repeat
        elif "once" in fire_mods:
            fire = RuleDefaults().fire_once
        else:
            fire = default_fire
        scope_bullets = block.sections.get("scope", [])
        scope = _to_scope(scope_bullets)
        digest = hashlib.sha1(block.title.encode("utf-8")).hexdigest()[:8]
        rule_id = f"mdr_{idx}_{kind}_{digest}"

        replace_prefix_terms = _section_terms_by_operator(
            block, "prefix", default_op=MatchDefaults().replace_prefix_match
        )
        after_prefix_terms = _section_terms_by_operator(
            block, "prefix", default_op=MatchDefaults().after_prefix_match
        )
        avoid_prefix_terms = _section_terms_by_operator(
            block, "prefix", default_op=MatchDefaults().avoid_prefix_match
        )
        avoid_prompt_terms = _section_terms_by_operator(
            block, "prompt", default_op="all"
        )
        avoid_postfix_terms = _section_terms_by_operator(
            block, "postfix", default_op=MatchDefaults().avoid_postfix_match
        )
        if kind == "replace":
            replace_gate = _build_gate_expression(
                terms=replace_prefix_terms,
                section_key="prefix",
                rule_match_options=rule_match_options,
                section_match_options=block.section_match_options.get(
                    "prefix", MatchOptionsAST()
                ),
            )
            return ReplaceRuleAST(
                rule_id=rule_id,
                fire=fire,
                scope=scope,
                match_options=rule_match_options,
                target=target,
                candidates=tuple(block.sections.get("with", [])),
                gate=SetMatchAST(expression=replace_gate),
            )
        if kind == "after":
            after_gate = _build_gate_expression(
                terms=after_prefix_terms,
                section_key="prefix",
                rule_match_options=rule_match_options,
                section_match_options=block.section_match_options.get(
                    "prefix", MatchOptionsAST()
                ),
            )
            return AfterRuleAST(
                rule_id=rule_id,
                fire=fire,
                scope=scope,
                match_options=rule_match_options,
                target=target,
                candidates=tuple(block.sections.get("add", [])),
                gate=SetMatchAST(expression=after_gate),
                wait_for_closing_parenthesis=(
                    _parse_after_wait_for_closing_parenthesis(
                        block.sections.get("options", [])
                    )
                ),
            )
        if kind == "avoid":
            connector_terms = tuple(block.sections.get("connector", []))
            return AvoidRuleAST(
                rule_id=rule_id,
                fire=fire,
                scope=scope,
                match_options=rule_match_options,
                target=target,
                edit=AvoidEditSpecAST(behavior_mods),
                guard_expression=_build_avoid_expression(
                    before_terms=avoid_prefix_terms,
                    prompt_terms=avoid_prompt_terms,
                    connector_terms=connector_terms,
                    overlap_terms=avoid_postfix_terms,
                    rule_match_options=rule_match_options,
                    section_match_options=block.section_match_options,
                ),
                connector_terms=connector_terms,
                fallback=tuple(block.sections.get("fallback", [])),
                options=_parse_options(block.sections.get("options", [])),
            )
        if kind == "force":
            return ForceRuleAST(
                rule_id=rule_id,
                fire=fire,
                scope=scope,
                match_options=rule_match_options,
                target=target,
                add=tuple(block.sections.get("add", [])),
            )
        return None


def _syntax_error(text: str, exc: UnexpectedInput) -> RulesSyntaxError:
    line = exc.line or 1
    column = exc.column or 1
    lines = text.splitlines() or [""]
    snippet = lines[line - 1] if 0 < line <= len(lines) else ""
    return RulesSyntaxError(
        "Invalid rules syntax", line=line, column=column, snippet=snippet
    )


def _raise_on_unparsed_lines(tree: Tree[Token]) -> None:
    for child in tree.children:
        if (
            not isinstance(child, Tree)
            or child.data != "ignored"
            or not child.children
        ):
            continue
        token = child.children[0]
        if isinstance(token, Token) and str(token).strip():
            raise RulesSyntaxError(
                "Invalid rules syntax",
                line=token.line or 1,
                column=token.column or 1,
                snippet=str(token),
            )


def _unescape_template_text(text: str) -> str:
    """Unescape template text, preserving a trailing unmatched backslash."""
    out: list[str] = []
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        out.append(ch)
    if escaped:
        out.append("\\")
    return "".join(out)


def _token_value(tree: Tree[Token], token_type: str) -> str:
    """Return the first token value of ``token_type`` from ``tree``."""
    for child in tree.children:
        if isinstance(child, Token) and child.type == token_type:
            return str(child)
    return ""


def _bullets(tree: Tree[Token]) -> list[str]:
    """Extract bullet text values from a parsed bullets node."""
    values: list[str] = []
    for child in tree.children:
        if isinstance(child, Token) and child.type == "BULLET":
            values.append(str(child)[1:].strip())
    return values


def _section_terms_by_operator(
    block: _RuleBlock,
    label: str,
    default_op: Literal["any", "all", "none", "incomplete"],
) -> ConditionedTerms:
    return ConditionedTerms(
        section_values=block.sections,
        label=label,
        default_op=default_op,
    )


def _to_scope(bullets: list[str]) -> ScopeAST:
    """Parse scope bullets into canonical ``ScopeAST`` form."""
    if not bullets:
        return ScopeAST(
            kind="whole_doc", n=0, casefold=ScopeDefaults().casefold
        )
    low = [b.lower() for b in bullets]
    for b in low:
        normalized = " ".join(b.split())
        if normalized in {
            "all",
            "from beginning",
            "from the beginning",
            "from the start",
        }:
            return ScopeAST(
                kind="whole_doc", n=0, casefold=ScopeDefaults().casefold
            )
        if "clause" in b:
            n = int(b.split()[0]) if b.split() and b.split()[0].isdigit() else 1
            return ScopeAST(
                kind="tail_clauses", n=n, casefold=ScopeDefaults().casefold
            )
        if "sentence" in b:
            n = int(b.split()[0]) if b.split() and b.split()[0].isdigit() else 1
            return ScopeAST(
                kind="tail_sentences",
                n=n,
                casefold=ScopeDefaults().casefold,
            )
        if "char" in b:
            tokens = [t for t in re.split(r"\s+", b) if t.isdigit()]
            n = int(tokens[0]) if tokens else ScopeDefaults().tail_chars
            return ScopeAST(
                kind="tail_chars", n=n, casefold=ScopeDefaults().casefold
            )
    return ScopeAST(kind="whole_doc", n=0, casefold=ScopeDefaults().casefold)


def _mods(title: str) -> _RuleMods:
    """Parse a rule heading into normalized kind, modifiers, and target text.

    Purpose:
        Interpret the compact rule-heading syntax used by authored markdown
        rules.

    Architectural role:
        Parser helper at the boundary between rule-language surface syntax and
        the normalized abstract-syntax-tree consumed by the compiler.

    Inputs (architectural provenance):
        Receives the raw heading text captured from a parsed rule block.

    Outputs (downstream usage):
        Returns the rule kind, normalized modifier set, and remaining target
        text used to build the rule abstract-syntax-tree.

    Invariants/constraints:
        Modifier parsing should stay syntax-only. Semantic validation and option
        precedence belong to compiler and match-option layers.

    """
    if not title.lstrip().startswith("##"):
        return _RuleMods("", (), "")
    _, _, remainder = title.partition("##")
    head, sep, target_raw = remainder.partition(":")
    if not sep:
        return _RuleMods("", (), "")
    head = head.strip()
    if not head:
        return _RuleMods("", (), "")
    open_idx = head.find("(")
    close_idx = head.rfind(")")
    if open_idx != -1 and close_idx > open_idx:
        kind_raw = head[:open_idx]
        mods_raw = head[open_idx + 1 : close_idx]
    else:
        kind_raw = head
        mods_raw = ""
    kind = kind_raw.strip().lower()
    if not kind:
        return _RuleMods("", (), "")
    mods = tuple(
        modifier.strip().lower()
        for modifier in mods_raw.split(",")
        if modifier.strip()
    )
    target = target_raw.strip()
    return _RuleMods(kind=kind, modifiers=mods, target=target)


_CASE_SENSITIVE_ALIASES = {
    "case-sensitive",
    "respect-case",
    "respect case",
    "casefold-false",
    "casefold=false",
}
_CASE_INSENSITIVE_ALIASES = {
    "case-insensitive",
    "ignore-case",
    "ignore case",
    "casefold",
    "casefold-true",
    "casefold=true",
}
_WORD_ALIASES = {"word", "whole-word", "whole word"}


def _parse_match_options(
    modifiers: Sequence[str],
    *,
    context: str,
    allow_word: bool,
) -> MatchOptionsAST:
    casefold: bool | None = None
    word: bool | None = None
    for mod in modifiers:
        normalized = " ".join(mod.lower().split())
        if normalized in _CASE_SENSITIVE_ALIASES:
            if casefold is True:
                raise RulesSyntaxError(
                    f"Contradictory modifiers in {context}",
                    line=1,
                    column=1,
                    snippet=mod,
                )
            casefold = False
            continue
        if normalized in _CASE_INSENSITIVE_ALIASES:
            if casefold is False:
                raise RulesSyntaxError(
                    f"Contradictory modifiers in {context}",
                    line=1,
                    column=1,
                    snippet=mod,
                )
            casefold = True
            continue
        if normalized in _WORD_ALIASES:
            if not allow_word:
                raise RulesSyntaxError(
                    f"Unsupported modifier '{mod}' in {context}",
                    line=1,
                    column=1,
                    snippet=mod,
                )
            word = True
            continue
        raise RulesSyntaxError(
            (
                f"Unsupported modifier '{mod}' in {context}. "
                "Supported matching modifiers: case-sensitive, "
                "case-insensitive, word"
            ),
            line=1,
            column=1,
            snippet=mod,
        )
    return MatchOptionsAST(casefold=casefold, word=word)


def _partition_rule_modifiers(
    *, kind: str, modifiers: tuple[str, ...], title: str
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    fire_mods: list[str] = []
    match_mods: list[str] = []
    behavior_mods: list[str] = []
    for mod in modifiers:
        normalized = " ".join(mod.lower().split())
        if normalized in {"once", "repeat"}:
            fire_mods.append(normalized)
            continue
        if (
            normalized in _CASE_SENSITIVE_ALIASES
            or normalized in _CASE_INSENSITIVE_ALIASES
            or normalized in _WORD_ALIASES
        ):
            match_mods.append(normalized)
            continue
        behavior_mods.append(normalized)
    if len(set(fire_mods)) > 1:
        raise RulesSyntaxError(
            "Contradictory fire modifiers in rule header",
            line=1,
            column=1,
            snippet=title,
        )
    if kind in {"replace", "after", "avoid"}:
        bad = (
            [
                m
                for m in behavior_mods
                if not _is_supported_avoid_behavior_modifier(m)
            ]
            if kind == "avoid"
            else list(behavior_mods)
        )
        if bad:
            raise RulesSyntaxError(
                (
                    f"Unsupported modifier '{bad[0]}' in "
                    f"{kind.title()} rule header"
                ),
                line=1,
                column=1,
                snippet=title,
            )
    elif match_mods:
        raise RulesSyntaxError(
            f"Unsupported matching modifiers in {kind.title()} rule header",
            line=1,
            column=1,
            snippet=title,
        )
    return tuple(fire_mods), tuple(match_mods), tuple(behavior_mods)


def _is_supported_avoid_behavior_modifier(modifier: str) -> bool:
    if modifier in {
        "postfix",
        "prefix clause",
        "matched prefix clause",
        "clause containing anchor to scope end",
        "clause_containing_anchor_to_scope_end",
        "everything",
        "all",
        "last clause",
        "last sentence",
    }:
        return True
    if re.fullmatch(r"\d+\s+(?:last\s+)?clauses?", modifier):
        return True
    return bool(re.fullmatch(r"\d+\s+(?:last\s+)?sentences?", modifier))


def _resolve_match_options(
    *,
    item: MatchOptionsAST,
    section: MatchOptionsAST,
    rule: MatchOptionsAST,
) -> ResolvedMatchOptions:
    return resolve_match_options(
        item=item,
        section=section,
        rule=rule,
        defaults=MatchDefaults(),
    )


def _parse_item_modifier_and_expression(
    value: str, *, context: str
) -> tuple[str, MatchOptionsAST]:
    m = re.match(r"^\((?P<mods>[^)]*)\)\s+(?P<text>.+)$", value.strip())
    if m is None:
        return value, MatchOptionsAST()
    mods = tuple(
        part.strip().lower()
        for part in m.group("mods").split(",")
        if part.strip()
    )
    return m.group("text").strip(), _parse_match_options(
        mods, context=context, allow_word=True
    )


def _reject_item_modifiers_in_non_matching_sections(block: _RuleBlock) -> None:
    non_matching_sections = {
        "with",
        "add",
        "fallback",
        "scope",
        "options",
        "rewrite",
    }
    for section, values in block.sections.items():
        base = section.split(":", 1)[0]
        if base not in non_matching_sections:
            continue
        for value in values:
            if re.match(r"^\([^)]*\)\s+.+$", value.strip()):
                raise RulesSyntaxError(
                    (
                        f"Unsupported item modifier in {base.title()} section. "
                        "This section does not support matching modifiers."
                    ),
                    line=1,
                    column=1,
                    snippet=value,
                )


def _build_gate_expression(
    *,
    terms: ConditionedTerms,
    section_key: str,
    rule_match_options: MatchOptionsAST,
    section_match_options: MatchOptionsAST,
) -> MatchTree | None:
    """Build a generic gate match expression from conditioned terms.

    Purpose:
        Convert ``any`` / ``all`` / ``none`` sections into one structural
        match-tree expression used by replace and after rules, with semantic
        markers aligned to authored section names for downstream telemetry.

    """
    normalized_section = section_key.casefold()
    marker_section = (
        normalized_section
        if normalized_section in {"prefix", "postfix", "prompt"}
        else None
    )

    def _marker_for_mode(mode: Mode) -> str | None:
        if marker_section is None:
            return None
        return f"{marker_section}_{mode}"

    handlers: dict[Mode, Callable[[tuple[str, ...]], MatchTree | None]] = {
        "all": lambda values: _combine_all(
            [
                _build_match_term(
                    expr,
                    marker=_marker_for_mode("all"),
                    context=f"{section_key} item",
                    rule_match_options=rule_match_options,
                    section_match_options=section_match_options,
                )
                for expr in values
            ]
        ),
        "any": lambda values: MatchAny(
            tuple(
                _build_match_term(
                    expr,
                    marker=_marker_for_mode("any"),
                    context=f"{section_key} item",
                    rule_match_options=rule_match_options,
                    section_match_options=section_match_options,
                )
                for expr in values
            )
        ),
        "none": lambda values: MatchNot(
            MatchAny(
                tuple(
                    _build_match_term(
                        expr,
                        marker=_marker_for_mode("none"),
                        context=f"{section_key} item",
                        rule_match_options=rule_match_options,
                        section_match_options=section_match_options,
                    )
                    for expr in values
                )
            )
        ),
        "incomplete": lambda _values: None,
    }
    nodes: list[MatchTree] = []
    for mode, mode_terms in terms.items():
        if not mode_terms:
            continue
        built = handlers[mode](mode_terms)
        if built is not None:
            nodes.append(built)
    return _combine_all(nodes)


def _build_avoid_expression(
    *,
    before_terms: ConditionedTerms,
    prompt_terms: ConditionedTerms,
    connector_terms: tuple[str, ...],
    overlap_terms: ConditionedTerms,
    rule_match_options: MatchOptionsAST,
    section_match_options: Mapping[str, MatchOptionsAST],
) -> MatchTree | None:
    """Build the ordered guard expression used by avoid rules.

    Purpose:
        Combine prompt-side requirements, ordered prefix terms, connector terms,
        and postfix overlap terms into the structural match-tree encoding
        expected by avoid-rule helpers and the compiler.

    """
    ordered_right = _ordered_avoid_tail(
        connector_terms=connector_terms,
        overlap_terms=overlap_terms,
        rule_match_options=rule_match_options,
        section_match_options=section_match_options,
    )
    ordered_nodes: list[MatchTree] = []
    if ordered_right is not None:
        for mode, mode_terms in before_terms.items():
            if not mode_terms:
                continue
            built = _build_ordered_prefix_condition(
                mode=mode,
                values=mode_terms,
                right=ordered_right,
                rule_match_options=rule_match_options,
                section_match_options=section_match_options,
            )
            if built is not None:
                ordered_nodes.append(built)
        if not ordered_nodes:
            ordered_nodes.append(ordered_right)

    before_match_nodes = [
        _build_negated_condition(
            section="prefix",
            mode=mode,
            values=mode_terms,
            rule_match_options=rule_match_options,
            section_match_options=section_match_options,
        )
        for mode, mode_terms in before_terms.items()
        if mode_terms
    ]
    answer_side = _combine_all(
        [
            node
            for node in (*before_match_nodes, _combine_all(ordered_nodes))
            if node is not None
        ]
    )
    prompt_nodes: list[MatchTree] = []
    for mode, mode_terms in prompt_terms.items():
        if not mode_terms:
            continue
        built = _build_section_condition(
            section="prompt",
            mode=mode,
            values=mode_terms,
            rule_match_options=rule_match_options,
            section_match_options=section_match_options,
        )
        if built is not None:
            prompt_nodes.append(built)
    prompt_side = _combine_all(prompt_nodes)
    if prompt_side is None:
        return answer_side
    if answer_side is None:
        return prompt_side
    return MatchAndThen(
        prompt_side, answer_side, marker="prompt_answer_boundary"
    )


def _combine_all(nodes: Sequence[MatchTree]) -> MatchTree | None:
    """Return ``None``, the single node, or a conjunction over many nodes.

    Purpose:
        Keep match-tree builders compact while preserving the smallest truthful
        structural representation.

    """
    if not nodes:
        return None
    if len(nodes) == 1:
        return nodes[0]
    return MatchAll(tuple(nodes))


def _ordered_avoid_tail(
    *,
    connector_terms: tuple[str, ...],
    overlap_terms: ConditionedTerms,
    rule_match_options: MatchOptionsAST,
    section_match_options: Mapping[str, MatchOptionsAST],
) -> MatchTree | None:
    """Build the ordered connector/postfix tail for an avoid expression.

    Purpose:
        Encode connector terms and overlap terms into a left-to-right
        ``MatchAndThen`` chain used by ordered avoid matching.

    """
    ordered_nodes: list[MatchTree] = []
    if connector_terms:
        ordered_nodes.append(
            _build_connector_condition(
                connector_terms,
                rule_match_options=rule_match_options,
                section_match_options=section_match_options,
            )
        )
    for mode, mode_terms in overlap_terms.items():
        if not mode_terms:
            continue
        built = _build_section_condition(
            section="postfix",
            mode=mode,
            values=mode_terms,
            rule_match_options=rule_match_options,
            section_match_options=section_match_options,
        )
        if built is not None:
            ordered_nodes.append(built)
    if not ordered_nodes:
        return None
    chain = ordered_nodes[0]
    for node in ordered_nodes[1:]:
        chain = MatchAndThen(chain, node)
    return chain


def _build_connector_condition(
    values: tuple[str, ...],
    *,
    rule_match_options: MatchOptionsAST,
    section_match_options: Mapping[str, MatchOptionsAST],
) -> MatchTree:
    return MatchAny(
        tuple(
            _build_match_term(
                expr,
                marker="connector",
                context="connector item",
                rule_match_options=rule_match_options,
                section_match_options=section_match_options.get(
                    "connector", MatchOptionsAST()
                ),
            )
            for expr in values
        )
    )


def _build_ordered_prefix_condition(
    *,
    mode: Mode,
    values: tuple[str, ...],
    right: MatchTree,
    rule_match_options: MatchOptionsAST,
    section_match_options: Mapping[str, MatchOptionsAST],
) -> MatchTree | None:
    if mode == "all":
        return _combine_all(
            tuple(
                MatchAndThen(
                    _build_match_term(
                        expr,
                        marker="prefix_all",
                        context="prefix item",
                        rule_match_options=rule_match_options,
                        section_match_options=section_match_options.get(
                            "prefix", MatchOptionsAST()
                        ),
                    ),
                    right,
                )
                for expr in values
            )
        )
    if mode == "any":
        return MatchAny(
            tuple(
                MatchAndThen(
                    _build_match_term(
                        expr,
                        marker="prefix_any",
                        context="prefix item",
                        rule_match_options=rule_match_options,
                        section_match_options=section_match_options.get(
                            "prefix", MatchOptionsAST()
                        ),
                    ),
                    right,
                )
                for expr in values
            )
        )
    return None


def _marked_terms(
    *,
    section: str,
    mode: str,
    values: tuple[str, ...],
    rule_match_options: MatchOptionsAST,
    section_match_options: Mapping[str, MatchOptionsAST],
) -> tuple[MatchTerm, ...]:
    return tuple(
        _build_match_term(
            expr,
            marker=f"{section}_{mode}",
            context=f"{section} item",
            rule_match_options=rule_match_options,
            section_match_options=section_match_options.get(
                section, MatchOptionsAST()
            ),
        )
        for expr in values
    )


def _build_negated_condition(
    *,
    section: str,
    mode: Mode,
    values: tuple[str, ...],
    rule_match_options: MatchOptionsAST,
    section_match_options: Mapping[str, MatchOptionsAST],
) -> MatchTree | None:
    if mode == "none":
        return MatchNot(
            MatchAny(
                _marked_terms(
                    section=section,
                    mode="none",
                    values=values,
                    rule_match_options=rule_match_options,
                    section_match_options=section_match_options,
                )
            )
        )
    if mode == "incomplete":
        return MatchNot(
            MatchAll(
                _marked_terms(
                    section=section,
                    mode="incomplete",
                    values=values,
                    rule_match_options=rule_match_options,
                    section_match_options=section_match_options,
                )
            )
        )
    return None


def _build_section_condition(
    *,
    section: str,
    mode: Mode,
    values: tuple[str, ...],
    rule_match_options: MatchOptionsAST,
    section_match_options: Mapping[str, MatchOptionsAST],
) -> MatchTree | None:
    if mode == "all":
        return _combine_all(
            _marked_terms(
                section=section,
                mode="all",
                values=values,
                rule_match_options=rule_match_options,
                section_match_options=section_match_options,
            )
        )
    if mode == "any":
        return MatchAny(
            _marked_terms(
                section=section,
                mode="any",
                values=values,
                rule_match_options=rule_match_options,
                section_match_options=section_match_options,
            )
        )
    if mode in {"none", "incomplete"}:
        return _build_negated_condition(
            section=section,
            mode=mode,
            values=values,
            rule_match_options=rule_match_options,
            section_match_options=section_match_options,
        )
    return None


def _build_match_term(
    raw_value: str,
    *,
    marker: str | None,
    context: str,
    rule_match_options: MatchOptionsAST,
    section_match_options: MatchOptionsAST,
) -> MatchTerm:
    expression, item_match_options = _parse_item_modifier_and_expression(
        raw_value, context=context
    )
    resolved = _resolve_match_options(
        item=item_match_options,
        section=section_match_options,
        rule=rule_match_options,
    )
    return MatchTerm(expression, marker=marker, match_options=resolved)


def _parse_options(bullets: list[str]) -> dict[str, float | int]:
    """Parse numeric option bullets into normalized option keys.

    Purpose:
        Convert authored aliases such as beam width, token count, and minimum
        probability ratio into the canonical option names consumed by avoid-rule
        compilation.

    """
    out: dict[str, float | int] = {}
    for b in bullets:
        if ":" not in b:
            continue
        k, v = b.split(":", 1)
        key = k.strip().lower()
        raw = v.strip()
        try:
            val: float | int
            val = int(raw) if raw.isdigit() else float(raw)
        except ValueError:
            continue
        if key in {
            "num_beams",
            "beams",
            "beam_width",
            "width",
            "k",
            "trajectories",
            "max trajectories",
            "max_trajectories",
        }:
            out["probe_num_beams"] = int(val)
        elif key in {
            "max_new_tokens",
            "probe_max_new_tokens",
            "tokens",
            "token_count",
        }:
            out["probe_max_new_tokens"] = int(val)
        elif key in {"min probability ratio", "min_prob_ratio_to_best"}:
            out["min_prob_ratio_to_best"] = float(val)
        elif key == "skip":
            out["skip"] = int(val)
    return out


def _parse_after_wait_for_closing_parenthesis(bullets: list[str]) -> bool:
    """Parse the after-rule option that controls closing-parenthesis waiting.

    Purpose:
        Support both regime-style and explicit boolean option spellings while
        falling back to policy defaults when the option is absent.

    """
    wait_for_closing = PolicyDefaults().after_wait_for_closing_parenthesis
    for bullet in bullets:
        if ":" not in bullet:
            continue
        key_raw, value_raw = bullet.split(":", 1)
        key = key_raw.strip().lower().replace("_", " ")
        value = " ".join(value_raw.strip().lower().replace("_", " ").split())

        if key in {"regime", "fire regime", "parenthesis regime"}:
            if value in {
                "fire after closing",
                "wait for closing",
                "wait",
                "default",
            }:
                wait_for_closing = True
            elif value in {
                "don't wait for closing",
                "dont wait for closing",
                "no wait",
            }:
                wait_for_closing = False
            continue

        if key not in {"wait for closing", "wait for closing parenthesis"}:
            continue
        if value in {"true", "yes", "1", "on"}:
            wait_for_closing = True
        elif value in {"false", "no", "0", "off"}:
            wait_for_closing = False
    return wait_for_closing


def _parse_ruleset_text(source: str) -> Sequence[RuleAST]:
    """Parser registration hook for ``RulesetAST`` text construction.

    Purpose:
        Provide the canonical function registered with the abstract-syntax-tree
        layer so text-based ``RulesetAST`` construction reuses
        ``MarkdownRulesParser``.

    """
    return MarkdownRulesParser().parse(source).rules


register_ruleset_text_parser(_parse_ruleset_text)
