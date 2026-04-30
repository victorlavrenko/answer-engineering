"""Typed abstract-syntax-tree values and helpers for the rules language.

Define immutable rule-language objects produced by parsing and consumed by plan
compilation.

Architectural role:
    Source-representation layer inside the rules parsing and compilation
    boundary.

Inputs (architectural provenance):
    Constructed by ``MarkdownRulesParser`` from markdown rule text.

Outputs (downstream usage):
    Consumed by ``FullPlanCompiler`` and by abstract-syntax-tree-side helper
    accessors that derive ordered rule terms.

Current architecture notes:
    This layer is meaningfully owned by the rules boundary, but its
    match-expression field types still come from ``engine.proposal.match_tree``.

"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from answer_engineering.engine.proposal.match_tree.match_tree import (
    MatchAll,
    MatchAndThen,
    MatchAny,
    MatchNot,
    MatchTerm,
    MatchTree,
)

FireMode = Literal["once", "repeat"]
_RulesetTextParser = Callable[[str], Sequence["RuleAST"]]
_ruleset_text_parser: _RulesetTextParser | None = None


def register_ruleset_text_parser(parser: _RulesetTextParser) -> None:
    """Register the parser hook used by ``RulesetAST`` text construction.

    Purpose:
        Attach the current markdown-to-abstract-syntax-tree parser so
        ``RulesetAST(source_text)`` can parse text through one canonical
        registration point.

    Architectural role:
        Small integration seam between the abstract-syntax-tree value layer and
        the parser implementation.

    Invariants/constraints:
        The registered callable must return rule abstract-syntax-tree objects
        compatible with this module's dataclasses.

    """
    global _ruleset_text_parser
    _ruleset_text_parser = parser


@dataclass(frozen=True, slots=True)
class MatchOptionsAST:
    """Partial matching options declared in rule syntax.

    ``None`` means the option is not declared at this abstract-syntax-tree
    location and should inherit from the next outer level.

    Todo:
        Extend this model deliberately if the domain-specific language adds
        richer match modes such as regex matching.

    """

    casefold: bool | None = None
    word: bool | None = None


@dataclass(frozen=True, slots=True)
class ScopeAST:
    """Immutable AST representation of one rule scope declaration.

    Purpose:
        Carry the parsed scope kind, numeric extent, and casefold policy before
        compilation.

    Architectural role:
        Source-level scope value object in the rules abstract-syntax-tree layer.

    Outputs (downstream usage):
        Consumed by ``FullPlanCompiler._scope`` to produce executable
        ``ScopeSpec`` values.

    Invariants/constraints:
        ``kind`` names the logical scope family and ``n`` carries its parsed
        extent, even when whole-document scope uses ``0`` as a sentinel.

    """

    kind: Literal["tail_chars", "tail_sentences", "tail_clauses", "whole_doc"]
    n: int
    casefold: bool = False


@dataclass(frozen=True, slots=True)
class SetMatchAST:
    """abstract-syntax-tree wrapper for an optional set-style match expression.

    Purpose:
        Hold the parsed match tree for prefix or gate conditions and expose
        simple term-oriented views when the tree has a recognized shape.

    Architectural role:
        Rules-abstract-syntax-tree adapter between generic match-tree
        expressions and rule-family helpers.

    Outputs (downstream usage):
        Consumed by the compiler as a raw expression and by convenience
        properties such as ``any_of`` and ``none_of``.

    Current architecture notes:
        The wrapper is rules-owned, but the underlying expression type still
        comes from the engine match-tree contract.

    """

    expression: MatchTree | None = None

    @property
    def any_of(self) -> Sequence[str]:
        """Return the first simple OR-group of term expressions, if present.

        Purpose:
            Project a recognized ``MatchAny`` subtree into a plain tuple of
            terms for callers that need the simple domain-specific language view
            rather than the full expression tree.

        """
        if self.expression is None:
            return ()
        nodes = _all_children(self.expression)
        for node in nodes:
            if isinstance(node, MatchAny) and all(
                isinstance(child, MatchTerm) for child in node.children
            ):
                terms: list[str] = []
                for child in node.children:
                    if isinstance(child, MatchTerm):
                        terms.append(child.expression)
                return tuple(terms)
        return ()

    @property
    def all_of(self) -> Sequence[str]:
        """Return every leaf term expression reachable in this set match.

        Purpose:
            Provide the broadest simple term view of the stored expression by
            collecting all ``MatchTerm`` leaves reachable under the current
            tree.

        Architectural role:
            abstract-syntax-tree-side convenience accessor used when later
            compiler or helper code wants a flat all-of style term tuple instead
            of the structural match tree.

        Invariants/constraints:
            Returned terms preserve traversal order and do not mutate the
            underlying expression.

        """
        if self.expression is None:
            return ()
        nodes = _all_children(self.expression)
        terms: list[str] = []
        for node in nodes:
            if isinstance(node, MatchTerm):
                terms.append(node.expression)
        return tuple(terms)

    @property
    def none_of(self) -> Sequence[str]:
        """Return the negated OR-group encoded in this set match, if present.

        Purpose:
            Recognize the common ``none`` shape ``MatchNot(MatchAny(...))`` and
            expose its leaf terms as a plain tuple for callers that need the
            authored domain-specific language view.

        Architectural role:
            abstract-syntax-tree-side convenience accessor that bridges
            structural negation in the match tree back to the simpler none-of
            vocabulary used by rule authors.

        Invariants/constraints:
            Returns an empty tuple when the stored expression is absent or does
            not match the recognized negated-any shape.

        """
        if self.expression is None:
            return ()
        nodes = _all_children(self.expression)
        for node in nodes:
            if (
                isinstance(node, MatchNot)
                and isinstance(node.child, MatchAny)
                and all(
                    isinstance(child, MatchTerm)
                    for child in node.child.children
                )
            ):
                terms: list[str] = []
                for child in node.child.children:
                    if isinstance(child, MatchTerm):
                        terms.append(child.expression)
                return tuple(terms)
        return ()


@dataclass(frozen=True, slots=True)
class RuleAST:
    """Common base value for parsed rule-family abstract-syntax-tree objects.

    Purpose:
        Hold the fields shared by every parsed markdown rule: stable parsed rule
        id, fire mode, and optional scope.

    Architectural role:
        Base source-representation contract for rule-family abstract-syntax-tree
        dataclasses.

    """

    rule_id: str
    fire: FireMode
    scope: ScopeAST | None = None
    match_options: MatchOptionsAST = field(default_factory=MatchOptionsAST)


@dataclass(frozen=True, slots=True)
class ReplaceRuleAST(RuleAST):
    """Parsed abstract-syntax-tree for one replace rule.

    Purpose:
        Represent a rule that rewrites a matched span with one or more static
        candidate replacements.

    Outputs (downstream usage):
        Consumed by the compiler to build a replace-style ``RulePlan`` with
        match-span targeting and rewrite candidates.

    """

    target: str = ""
    candidates: Sequence[str] = ()
    gate: SetMatchAST | None = None


@dataclass(frozen=True, slots=True)
class AfterRuleAST(RuleAST):
    """Parsed abstract-syntax-tree for one after rule.

    Purpose:
        Represent a rule that inserts candidate text after a recognized anchor
        phrase within the active scope.

    Outputs (downstream usage):
        Consumed by the compiler to build an after-anchor ``RulePlan``.

    Invariants/constraints:
        ``wait_for_closing_parenthesis`` preserves the rule-family option that
        delays firing when the anchor appears inside an unfinished
        parenthetical.

    """

    target: str = ""
    candidates: Sequence[str] = ()
    gate: SetMatchAST | None = None
    wait_for_closing_parenthesis: bool = True


@dataclass(frozen=True, slots=True, init=False)
class AvoidEditSpecAST:
    """Parsed edit-behavior specification for an avoid rule.

    Purpose:
        Normalize avoid-rule modifiers such as postfix, prefix clause, last
        sentence, and last clause into one canonical edit-behavior value.

    Architectural role:
        Avoid-rule modifier normalizer in the rules abstract-syntax-tree layer.

    Inputs (architectural provenance):
        Built either from explicit normalized fields or by parsing markdown
        modifiers attached to an avoid-rule header.

    Invariants/constraints:
        Exactly one edit behavior is resolved for one avoid rule, with optional
        sentence or clause counts attached only for the matching behavior
        family.

    """

    kind: Literal[
        "everything",
        "postfix",
        "tail_sentences",
        "tail_clauses",
        "prefix_clause",
    ] = "everything"
    n_sentences: int | None = None
    n_clauses: int | None = None

    def __init__(
        self,
        modifiers: tuple[str, ...] = tuple(),
        *,
        kind: Literal[
            "everything",
            "postfix",
            "tail_sentences",
            "tail_clauses",
            "prefix_clause",
        ]
        | None = None,
        n_sentences: int | None = None,
        n_clauses: int | None = None,
    ) -> None:
        """Resolve explicit values or modifiers into one avoid-edit behavior.

        Purpose:
            Provide one construction path for avoid-edit specifications authored
            either as explicit fields or as parsed modifier collections.

        Architectural role:
            abstract-syntax-tree normalization constructor in the rules parse
            layer. It turns flexible surface syntax into one immutable behavior
            object for the compiler.

        Inputs (architectural provenance):
            Receives optional explicit replacement values, behavior mode, and
            modifiers produced by the markdown parser.

        Outputs (downstream usage):
            Stores the canonical avoid-edit mode and replacement data consumed
            by the rules compiler and proposal generation.

        Invariants/constraints:
            Construction must reject incompatible or ambiguous behavior
            descriptions so compiled rules never have to resolve parser-level
            modifier conflicts.

        """
        resolved_kind: Literal[
            "everything",
            "postfix",
            "tail_sentences",
            "tail_clauses",
            "prefix_clause",
        ] = "everything"
        resolved_n_sentences: int | None = None
        resolved_n_clauses: int | None = None

        if kind is not None:
            resolved_kind = kind
            resolved_n_sentences = n_sentences
            resolved_n_clauses = n_clauses
        else:
            for modifier in modifiers:
                normalized = " ".join(modifier.lower().split())
                if normalized == "postfix":
                    resolved_kind = "postfix"
                    break
                if normalized in {
                    "prefix clause",
                    "matched prefix clause",
                    "clause containing anchor to scope end",
                    "clause_containing_anchor_to_scope_end",
                }:
                    resolved_kind = "prefix_clause"
                    break
                if normalized in {"everything", "all"}:
                    resolved_kind = "everything"
                    break
                if normalized == "last clause":
                    resolved_kind = "tail_clauses"
                    resolved_n_clauses = 1
                    break
                if normalized == "last sentence":
                    resolved_kind = "tail_sentences"
                    resolved_n_sentences = 1
                    break
                clause_match = re.fullmatch(
                    r"(\d+)\s+(?:last\s+)?clauses?", normalized
                )
                if clause_match is not None:
                    resolved_kind = "tail_clauses"
                    resolved_n_clauses = int(clause_match.group(1))
                    break
                sentence_match = re.fullmatch(
                    r"(\d+)\s+(?:last\s+)?sentences?", normalized
                )
                if sentence_match is not None:
                    resolved_kind = "tail_sentences"
                    resolved_n_sentences = int(sentence_match.group(1))
                    break

        object.__setattr__(self, "kind", resolved_kind)
        object.__setattr__(self, "n_sentences", resolved_n_sentences)
        object.__setattr__(self, "n_clauses", resolved_n_clauses)


@dataclass(frozen=True, slots=True)
class AvoidRuleAST(RuleAST):
    """Parsed abstract-syntax-tree for one avoid rule.

    Purpose:
        Represent a rule that detects an unsafe trajectory pattern and redirects
        it with fallback candidates plus edit-behavior metadata.

    Architectural role:
        Avoid-family source representation in the rules abstract-syntax-tree
        layer.

    Outputs (downstream usage):
        Consumed by the compiler to build guard expressions, fallback
        candidates, probing policy options, and edit-target behavior.

    """

    target: str = ""
    edit: AvoidEditSpecAST = field(default_factory=AvoidEditSpecAST)
    guard_expression: MatchTree | None = None
    connector_terms: Sequence[str] = ()
    fallback: Sequence[str] = ()
    options: dict[str, float | int] = field(default_factory=lambda: {})

    @property
    def required_before_all(self) -> Sequence[str]:
        """Return required pre-anchor terms that must all appear.

        Purpose:
            Recover the simple domain-specific language view used by compiler
            logic that derives prefix-anchor behavior from the full guard
            expression.

        """
        return _ordered_before_terms(self.guard_expression)[0]

    @property
    def required_before_any(self) -> Sequence[str]:
        """Return pre-anchor alternatives that can satisfy the avoid prefix.

        Purpose:
            Recover the OR-group of prefix terms from the structural avoid-rule
            guard expression so compiler and debugging code can speak in the
            simpler authored-rule vocabulary.

        Architectural role:
            Avoid-rule abstract-syntax-tree convenience accessor layered on top
            of the shared match-tree representation.

        Invariants/constraints:
            The returned tuple reflects only ordered pre-any terms and does not
            include the required-all prefix terms recovered by the companion
            accessor.

        """
        return _ordered_before_terms(self.guard_expression)[1]

    @property
    def required_before_incomplete(self) -> Sequence[str]:
        """Return prefix terms whose absence marks the pattern incomplete.

        Purpose:
            Recognize the ``incomplete`` encoding used in avoid-rule guard
            expressions and expose the missing required terms as a simple tuple
            for downstream compiler or diagnostics code.

        Architectural role:
            Avoid-rule abstract-syntax-tree convenience accessor for one
            specific structural guard pattern.

        Invariants/constraints:
            Returns an empty tuple when the stored guard expression does not
            encode the incomplete-prefix pattern.

        """
        if self.guard_expression is None:
            return ()
        for node in _all_children(self.guard_expression):
            if (
                isinstance(node, MatchNot)
                and isinstance(node.child, MatchAll)
                and all(
                    isinstance(child, MatchTerm)
                    for child in node.child.children
                )
            ):
                terms: list[str] = []
                for child in node.child.children:
                    if isinstance(child, MatchTerm):
                        terms.append(child.expression)
                return tuple(terms)
        return ()

    @property
    def required_after_any(self) -> Sequence[str]:
        """Return tail-term alternatives that may satisfy the avoid tail.

        Purpose:
            Recover the simple any-of tail terms from the structural ordered
            avoid expression after connector handling so later code can reason
            about authored overlap alternatives directly.

        Architectural role:
            Avoid-rule abstract-syntax-tree convenience accessor layered over
            ordered-tail match-tree helpers.

        Invariants/constraints:
            Returned values exclude connector markers and preserve the
            ordered-tail interpretation used by avoid compilation.

        """
        return _ordered_after_terms(self.guard_expression)[0]

    @property
    def required_after_all(self) -> Sequence[str]:
        """Return tail terms that must all appear after the connector.

        Purpose:
            Recover the all-of portion of the ordered avoid tail from the
            structural match-tree encoding so later code can refer back to
            authored post-tail requirements directly.

        Architectural role:
            Avoid-rule abstract-syntax-tree convenience accessor for
            post-connector all-of terms.

        Invariants/constraints:
            Returned values exclude connector markers and do not include any-of
            overlap alternatives returned by the companion accessor.

        """
        return _ordered_after_terms(self.guard_expression)[1]


@dataclass(frozen=True, slots=True)
class ForceRuleAST(RuleAST):
    """Parsed abstract-syntax-tree for one force rule.

    Purpose:
        Represent a rule that enforces one or more added statements across the
        active scope.

    Outputs (downstream usage):
        Consumed by the compiler to build scope-wide force candidates.

    """

    target: str = ""
    add: Sequence[str] = ()


@dataclass(frozen=True, slots=True)
class RulesetAST:
    """Immutable container for one parsed ruleset.

    Purpose:
        Hold the ordered sequence of parsed rule abstract-syntax-tree objects
        produced from one markdown rules document.

    Architectural role:
        Top-level source representation handed from the parser to the compiler.

    """

    rules: Sequence[RuleAST]


def _all_children(tree: MatchTree) -> tuple[MatchTree, ...]:
    """Return ``MatchAll`` children, else return the node as a one-item tuple.

    Purpose:
        Normalize downstream helper logic so callers can treat top-level
        conjunctions and single nodes uniformly.

    """
    if isinstance(tree, MatchAll):
        return tree.children
    return (tree,)


def _ordered_before_terms(
    tree: MatchTree | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract ordered before-side term groups from an avoid guard expression.

    Purpose:
        Recover ``required_before_all`` and ``required_before_any`` from the
        structural match-tree encoding used by avoid rules.

    Invariants/constraints:
        Terms that move into the OR-group are removed from the all-of result so
        the two returned groups remain semantically distinct.

    """
    if tree is None:
        return (), ()
    root_nodes = _all_children(tree)
    before_all: list[str] = []
    before_any: list[str] = []
    any_group_seen = False
    for node in root_nodes:
        if isinstance(node, MatchAny):
            terms: list[str] = []
            for child in node.children:
                if isinstance(child, MatchAndThen) and isinstance(
                    child.left, MatchTerm
                ):
                    terms.append(child.left.expression)
            if terms:
                any_group_seen = True
                before_any.extend(terms)
        elif isinstance(node, MatchAndThen) and isinstance(
            node.left, MatchTerm
        ):
            before_all.append(node.left.expression)
    if not any_group_seen:
        return tuple(before_all), ()
    else:
        grouped = set(before_any)
        before_all = [term for term in before_all if term not in grouped]
    return tuple(before_all), tuple(before_any)


def _ordered_after_terms(
    tree: MatchTree | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract ordered tail terms from an avoid guard expression.

    Purpose:
        Recover the any-of and all-of post-connector term groups used by
        avoid-rule convenience accessors.

    Invariants/constraints:
        Connector markers are treated as a structural boundary and are not
        returned as semantic overlap terms.

    """
    if tree is None:
        return (), ()
    root_nodes = (
        tree.children if isinstance(tree, MatchAny) else _all_children(tree)
    )
    right: MatchTree | None = next(
        (node.right for node in root_nodes if isinstance(node, MatchAndThen)),
        tree if isinstance(tree, MatchAndThen) else None,
    )
    if right is None:
        right = tree
    tail_nodes = _ordered_sequence_nodes(right)
    connector_index = next(
        (
            idx
            for idx, node in enumerate(tail_nodes)
            if _node_has_leaf_marker(node, "connector")
        ),
        None,
    )
    if connector_index is not None:
        tail_nodes = tail_nodes[connector_index + 1 :]
    any_after: tuple[str, ...] = ()
    all_after: list[str] = []
    for node in tail_nodes:
        if isinstance(node, MatchAny) and all(
            isinstance(child, MatchTerm) for child in node.children
        ):
            any_terms: list[str] = []
            for child in node.children:
                if isinstance(child, MatchTerm):
                    any_terms.append(child.expression)
            any_after = tuple(any_terms)
        elif isinstance(node, MatchAll) and all(
            isinstance(child, MatchTerm) for child in node.children
        ):
            for child in node.children:
                if isinstance(child, MatchTerm):
                    all_after.append(child.expression)
        elif isinstance(node, MatchTerm):
            all_after.append(node.expression)
    return any_after, tuple(all_after)


def _ordered_sequence_nodes(node: MatchTree | None) -> tuple[MatchTree, ...]:
    """Flatten a left-to-right ``MatchAndThen`` chain into ordered nodes.

    Purpose:
        Preserve sequential match semantics while giving later helper logic a
        simple iterable representation of the chain.

    Architectural role:
        Low-level abstract-syntax-tree helper used by ordered-term extraction
        for avoid-rule analysis.

    Inputs (architectural provenance):
        Called by other abstract-syntax-tree helpers that inspect ordered avoid
        expressions.

    Outputs (downstream usage):
        Returns nodes in traversal order for later marker and term-group
        inspection.

    Invariants/constraints:
        ``None`` yields an empty tuple and non-chain nodes are returned as
        singletons.

    """
    if node is None:
        return ()
    if isinstance(node, MatchAndThen):
        return (
            *_ordered_sequence_nodes(node.left),
            *_ordered_sequence_nodes(node.right),
        )
    return (node,)


def _node_has_leaf_marker(node: MatchTree, marker: str) -> bool:
    """Report whether a node or simple leaf carries the given marker.

    Purpose:
        Support structural helper logic that needs to detect connector or other
        marker-bearing terms inside compact match-tree shapes.

    """
    if isinstance(node, MatchTerm):
        return node.marker == marker
    if isinstance(node, MatchAny):
        return any(
            isinstance(child, MatchTerm) and child.marker == marker
            for child in node.children
        )
    if isinstance(node, MatchAll):
        return any(
            isinstance(child, MatchTerm) and child.marker == marker
            for child in node.children
        )
    return False
