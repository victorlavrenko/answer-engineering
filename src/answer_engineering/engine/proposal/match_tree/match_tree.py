"""Boolean match-tree nodes for rule guard evaluation.

Purpose:
    Define the recursive guard-expression model used to search scoped text,
    preserve telemetry, and express conjunction, disjunction, negation, and
    sequence constraints.

Architectural role:
    Core guard-expression semantic boundary beneath proposal precheck.

Architectural direction:
    Keep this module as the authoritative recursive guard-expression model and
    prevent guard semantics from drifting into ad hoc helper behavior.

Why this matters:
    This seam is central to guard correctness, explainability, and future guard
    language growth.

What better would look like:
    Higher layers depend on clear match-tree semantics without reimplementing
    guard meaning in parallel helpers.

How improvement can be recognized:
    - Clearer dependency on match-tree semantics from higher proposal layers
    - Fewer ad hoc guard-evaluation shortcuts outside this boundary
    - Easier explanation of guard behavior and telemetry provenance

Open constraint:
    Future guard-language growth may require representation adjustments.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from answer_engineering.config.engine_defaults import MatchDefaults
from answer_engineering.engine.pipeline import text_patterns
from answer_engineering.rules.matching.options import ResolvedMatchOptions


@dataclass(frozen=True, slots=True, order=True)
class Span:
    """Half-open character span used by match-tree evaluation.

    Purpose:
        Represent one text match in a compact, orderable form.

    Architectural role:
        Shared primitive for guard telemetry and match results.

    Inputs (architectural provenance):
        Produced by term matching and by composite nodes that merge child spans.

    Outputs (downstream usage):
        Consumed by guard telemetry flattening, overlap analysis, and debugging.

    """

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class NodeTelemetry:
    """Recursive telemetry tree for one match-tree evaluation.

    Purpose:
        Preserve per-node match status, spans, marker information, and children
        so higher layers can explain why a guard matched or failed.

    Architectural role:
        Internal diagnostic representation emitted by ``MatchTree`` nodes.

    Inputs (architectural provenance):
        Built by concrete match-tree nodes during ``evaluate``.

    Outputs (downstream usage):
        Consumed by guard-flattening logic and avoid-overlap checks.

    """

    node_type: str
    marker: str | None
    matched: bool
    spans: tuple[Span, ...]
    children: tuple[NodeTelemetry, ...] = ()
    expression: str | None = None


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Top-level result of evaluating one match-tree node.

    Purpose:
        Package the boolean outcome, resolved spans, and explanatory telemetry
        for a guard-expression evaluation step.

    Architectural role:
        Core result type shared by all concrete ``MatchTree`` implementations.

    Inputs (architectural provenance):
        Constructed by concrete nodes after evaluating their text or child
        nodes.

    Outputs (downstream usage):
        Consumed by guard matching and downstream telemetry processing.

    """

    matched: bool
    spans: tuple[Span, ...]
    telemetry: NodeTelemetry


class MatchTree(ABC):
    """Abstract base class for compiled guard-expression nodes.

    Purpose:
        Define the common protocol for evaluating, normalizing, debugging, and
        extracting ordered-overlap subtrees from guard expressions.

    Architectural role:
        Stable semantic boundary between compiled rule guards and runtime guard
        evaluation.

    Inputs (architectural provenance):
        Implemented by concrete nodes created during rule compilation.

    Outputs (downstream usage):
        Produces ``MatchResult`` values and normalized tree structure consumed
        by guard matching.

    """

    marker: str | None

    @abstractmethod
    def fingerprint(self) -> str:
        """Return the semantic identity fingerprint for this node.

        Purpose:
            Encode semantic matching structure deterministically while excluding
            telemetry-only metadata such as node markers.

        """

    @abstractmethod
    def evaluate(self, text: str, *, casefold: bool = True) -> MatchResult:
        """Evaluate this guard-expression node against runtime text.

        Purpose:
            Produce the boolean match result, resolved spans, and telemetry for
            this node.

        """

    @abstractmethod
    def normalize(self) -> MatchTree:
        """Return this node in canonical constructor-normalized form.

        Policy:
            Normalization is constructor-owned. ``MatchAll`` and ``MatchAny``
            flatten nested unmarked peers in ``__post_init__``; other nodes are
            identity-like.

        """

    @abstractmethod
    def to_debug_string(self) -> str:
        """Render this node in a stable human-readable debug form.

        Purpose:
            Support telemetry, diagnostics, and tests that need a readable tree
            representation.

        """

    @abstractmethod
    def ordered_overlap_subtree(self) -> MatchTree | None:
        """Return the subtree relevant to ordered-overlap analysis.

        Purpose:
            Let higher-level overlap checks preserve only the structure that
            matters for sequence-sensitive matching.

        """


@dataclass(frozen=True, slots=True)
class MatchTerm(MatchTree):
    """Leaf guard node that matches one phrase against text.

    Purpose:
        Search for occurrences of ``expression`` and emit spans plus term-level
        telemetry.

    Architectural role:
        Smallest executable unit of the guard-expression tree.

    Inputs (architectural provenance):
        Created by rule compilation from literal guard terms.

    Outputs (downstream usage):
        Supplies leaf ``MatchResult`` objects consumed by composite match nodes.

    """

    expression: str
    marker: str | None = None
    match_options: ResolvedMatchOptions | None = None

    def fingerprint(self) -> str:
        """Return the semantic fingerprint of this term.

        Purpose:
            Encode term identity from expression text only and ignore telemetry
            marker metadata by design.

        """
        defaults = MatchDefaults()
        options = self.match_options or ResolvedMatchOptions(
            casefold=defaults.casefold,
            word=defaults.word,
        )
        return (
            f"term({len(self.expression)}:{self.expression}|"
            f"casefold={options.casefold}|word={options.word})"
        )

    def evaluate(self, text: str, *, casefold: bool = True) -> MatchResult:
        """Evaluate the term by searching the input text for ``expression``.

        Purpose:
            Convert raw span matches into ``Span`` objects and leaf telemetry.

        Notes:
            ``casefold`` is retained as a compatibility fallback for callers
            that still evaluate trees without explicit ``match_options`` on each
            ``MatchTerm``. When ``self.match_options`` is present, it is the
            authoritative runtime configuration.

        Architectural role:
            Leaf execution step in match-tree evaluation.

        Inputs (architectural provenance):
            Receives runtime guard text plus the caller's case-fold policy.

        Outputs (downstream usage):
            Returns a ``MatchResult`` used by guard matching or composite nodes.

        """
        defaults = MatchDefaults()
        options = self.match_options or ResolvedMatchOptions(
            casefold=casefold,
            word=defaults.word,
        )
        spans = tuple(
            Span(start, end)
            for start, end in text_patterns.find_spans(
                text, self.expression, match_options=options
            )
        )
        telemetry = NodeTelemetry(
            node_type="MatchTerm",
            marker=self.marker,
            matched=bool(spans),
            spans=spans,
            expression=self.expression,
        )
        return MatchResult(
            matched=bool(spans), spans=spans, telemetry=telemetry
        )

    def normalize(self) -> MatchTree:
        """Return the canonical form of a term node.

        Purpose:
            Signal that leaf terms are already normalized and do not require
            structural rewriting.

        Architectural role:
            Canonicalization hook required by the ``MatchTree`` interface.

        Inputs (architectural provenance):
            Called from constructors of composite nodes during tree
            normalization.

        Outputs (downstream usage):
            Returns ``self`` unchanged for use in normalized trees.

        """
        return self

    def to_debug_string(self) -> str:
        """Return the literal term text used in debug rendering.

        Purpose:
            Provide a concise textual representation of a leaf expression.

        Architectural role:
            Debug-formatting hook for match-tree nodes.

        Inputs (architectural provenance):
            Reads the node's stored ``expression``.

        Outputs (downstream usage):
            String is consumed by diagnostics and telemetry formatting.

        """
        return self.expression

    def ordered_overlap_subtree(self) -> MatchTree | None:
        """Return the ordered-overlap-relevant subtree for this node.

        Purpose:
            Indicate that a plain term contributes no special ordered-overlap
            subtree on its own.

        Architectural role:
            Interface hook used by higher-level overlap analysis.

        Inputs (architectural provenance):
            Called when ordered-overlap extraction walks the tree.

        Outputs (downstream usage):
            Returns ``None`` to signal no specialized subtree.

        """
        return None


@dataclass(frozen=True, slots=True)
class MatchAll(MatchTree):
    """Composite guard node requiring all child expressions to match.

    Purpose:
        Evaluate conjunction semantics and merge child spans when every child
        succeeds.

    Architectural role:
        Boolean-AND node within the guard-expression tree.

    Inputs (architectural provenance):
        Built by compilation from all-match group expressions.

    Outputs (downstream usage):
        Emits a combined ``MatchResult`` and conjunction telemetry.

    """

    children: tuple[MatchTree, ...]
    marker: str | None = None

    def __post_init__(self) -> None:
        """Normalize and flatten nested unmarked conjunction children.

        Purpose:
            Enforce constructor-owned normalization so equivalent conjunction
            trees have the same shape.

        Architectural role:
            Invariant-establishing constructor hook for ``MatchAll``.

        Inputs (architectural provenance):
            Uses the initially supplied child nodes from rule compilation.

        Outputs (downstream usage):
            Mutates the frozen dataclass during construction to store normalized
            children.

        Invariants/constraints:
            A conjunction must contain at least one child.

        """
        if not self.children:
            raise ValueError("MatchAll nodes must contain at least one child")
        flattened: list[MatchTree] = []
        for child in self.children:
            normalized_child = child.normalize()
            if (
                isinstance(normalized_child, MatchAll)
                and normalized_child.marker is None
            ):
                flattened.extend(normalized_child.children)
            else:
                flattened.append(normalized_child)
        object.__setattr__(self, "children", tuple(flattened))

    def fingerprint(self) -> str:
        """Return the semantic fingerprint of this conjunction.

        Purpose:
            Encode conjunction structure and ordered child semantics while
            ignoring marker metadata.

        """
        children = ",".join(child.fingerprint() for child in self.children)
        return f"all({children})"

    def evaluate(self, text: str, *, casefold: bool = True) -> MatchResult:
        """Evaluate all children and succeed only when every child matches.

        Purpose:
            Implement conjunction semantics and merge and deduplicate child
            spans when the whole conjunction matches.

        Architectural role:
            Composite evaluation step within the match-tree runtime.

        Inputs (architectural provenance):
            Receives runtime guard text and case-fold policy.

        Outputs (downstream usage):
            Returns a combined ``MatchResult`` with conjunction telemetry.

        """
        child_results = tuple(
            child.evaluate(text, casefold=casefold) for child in self.children
        )
        matched = all(child.matched for child in child_results)
        spans = (
            _dedup_spans(
                span for child in child_results for span in child.spans
            )
            if matched
            else ()
        )
        telemetry = NodeTelemetry(
            node_type="MatchAll",
            marker=self.marker,
            matched=matched,
            spans=spans,
            children=tuple(child.telemetry for child in child_results),
        )
        return MatchResult(matched=matched, spans=spans, telemetry=telemetry)

    def normalize(self) -> MatchTree:
        """Return the already-normalized conjunction node.

        Purpose:
            Expose the canonicalized constructor result through the common tree
            interface.

        Architectural role:
            Canonicalization hook for composite nodes.

        Inputs (architectural provenance):
            Called by parent constructors and normalization walks.

        Outputs (downstream usage):
            Returns ``self`` because ``__post_init__`` already normalized
            children.

        """
        return self

    def to_debug_string(self) -> str:
        """Render the conjunction in a readable debug form.

        Purpose:
            Provide a deterministic string representation of the normalized
            child tree.

        Architectural role:
            Debug-formatting helper for telemetry and tests.

        Inputs (architectural provenance):
            Uses the stored normalized child nodes and optional marker.

        Outputs (downstream usage):
            Returns a string describing the conjunction structure.

        """
        return (
            "ALL("
            + ", ".join(child.to_debug_string() for child in self.children)
            + ")"
        )

    def ordered_overlap_subtree(self) -> MatchTree | None:
        """Return the subtree relevant to ordered-overlap analysis.

        Purpose:
            Preserve conjunction structure when searching for
            ordered-overlap-sensitive components beneath this node.

        Architectural role:
            Overlap-analysis hook for conjunction nodes.

        Inputs (architectural provenance):
            Called by guard-overlap logic walking normalized trees.

        Outputs (downstream usage):
            Returns either a reduced subtree or ``None`` depending on children.

        """
        actionable_children = [
            child_overlap
            for child in self.children
            if (child_overlap := child.ordered_overlap_subtree())
        ]
        if not actionable_children:
            return None
        if len(actionable_children) == 1:
            return actionable_children[0]
        return MatchAll(tuple(actionable_children))


@dataclass(frozen=True, slots=True)
class MatchAny(MatchTree):
    """Composite guard node requiring at least one child expression to match.

    Purpose:
        Evaluate disjunction semantics and collect spans from matching children.

    Architectural role:
        Boolean-OR node within the guard-expression tree.

    Inputs (architectural provenance):
        Built by compilation from any-match group expressions.

    Outputs (downstream usage):
        Emits a ``MatchResult`` and disjunction telemetry for guard matching.

    """

    children: tuple[MatchTree, ...]
    marker: str | None = None

    def __post_init__(self) -> None:
        """Normalize and flatten nested unmarked disjunction children.

        Purpose:
            Establish a canonical disjunction shape at construction time.

        Architectural role:
            Invariant hook for ``MatchAny`` construction.

        Inputs (architectural provenance):
            Uses children supplied by the compiler.

        Outputs (downstream usage):
            Stores normalized children for later evaluation and debugging.

        """
        if not self.children:
            raise ValueError("MatchAny nodes must contain at least one child")
        flattened: list[MatchTree] = []
        for child in self.children:
            normalized_child = child.normalize()
            if (
                isinstance(normalized_child, MatchAny)
                and normalized_child.marker is None
            ):
                flattened.extend(normalized_child.children)
            else:
                flattened.append(normalized_child)
        object.__setattr__(self, "children", tuple(flattened))

    def fingerprint(self) -> str:
        """Return the semantic fingerprint of this disjunction.

        Purpose:
            Encode disjunction structure and ordered child semantics while
            ignoring marker metadata.

        """
        children = ",".join(child.fingerprint() for child in self.children)
        return f"any({children})"

    def evaluate(self, text: str, *, casefold: bool = True) -> MatchResult:
        """Evaluate children and succeed when any child matches.

        Purpose:
            Implement disjunction semantics and collect spans from matching
            branches.

        Architectural role:
            Composite evaluation step within the match-tree runtime.

        Inputs (architectural provenance):
            Receives runtime guard text and case-fold policy.

        Outputs (downstream usage):
            Returns a disjunction ``MatchResult`` with branch telemetry.

        """
        child_results = tuple(
            child.evaluate(text, casefold=casefold) for child in self.children
        )
        matched = any(child.matched for child in child_results)
        spans = (
            _dedup_spans(
                span for child in child_results for span in child.spans
            )
            if matched
            else ()
        )
        telemetry = NodeTelemetry(
            node_type="MatchAny",
            marker=self.marker,
            matched=matched,
            spans=spans,
            children=tuple(child.telemetry for child in child_results),
        )
        return MatchResult(matched=matched, spans=spans, telemetry=telemetry)

    def normalize(self) -> MatchTree:
        """Return the canonicalized disjunction node.

        Purpose:
            Expose the constructor-normalized ``MatchAny`` through the common
            tree API.

        Architectural role:
            Canonicalization hook for disjunction nodes.

        Inputs (architectural provenance):
            Called by normalization walks in parent nodes.

        Outputs (downstream usage):
            Returns ``self`` because normalization happened during construction.

        """
        return self

    def to_debug_string(self) -> str:
        """Render the disjunction in a readable debug form.

        Purpose:
            Produce a stable string representation for diagnostics and tests.

        Architectural role:
            Debug-formatting helper for ``MatchAny``.

        Inputs (architectural provenance):
            Uses normalized children and optional marker.

        Outputs (downstream usage):
            Returns the formatted disjunction string.

        """
        return (
            "ANY("
            + ", ".join(child.to_debug_string() for child in self.children)
            + ")"
        )

    def ordered_overlap_subtree(self) -> MatchTree | None:
        """Return any ordered-overlap-relevant subtree under the disjunction.

        Purpose:
            Help overlap analysis preserve only the branch structure relevant to
            ordered matching semantics.

        Architectural role:
            Overlap-analysis hook for disjunction nodes.

        Inputs (architectural provenance):
            Called when ordered-overlap extraction walks the tree.

        Outputs (downstream usage):
            Returns a reduced subtree or ``None``.

        """
        actionable_children = [
            child_overlap
            for child in self.children
            if (child_overlap := child.ordered_overlap_subtree())
        ]
        if not actionable_children:
            return None
        if len(actionable_children) == 1:
            return actionable_children[0]
        return MatchAny(tuple(actionable_children))


@dataclass(frozen=True, slots=True)
class MatchNot(MatchTree):
    """Unary guard node that negates one child expression.

    Purpose:
        Invert the child's match outcome while preserving child telemetry for
        later explanation.

    Architectural role:
        Boolean-NOT node within the guard-expression tree.

    Inputs (architectural provenance):
        Built by compilation from negated guard syntax.

    Outputs (downstream usage):
        Emits a negated ``MatchResult`` used by guard evaluation and overlap
        analysis.

    """

    child: MatchTree
    marker: str | None = None

    def __post_init__(self) -> None:
        """Normalize the child node stored beneath this negation.

        Purpose:
            Ensure negation wraps the child's canonical form rather than an
            arbitrary pre-normalized subtree.

        Architectural role:
            Constructor-owned invariant hook for ``MatchNot``.

        Inputs (architectural provenance):
            Uses the child supplied by compilation.

        Outputs (downstream usage):
            Stores the normalized child for later evaluation.

        """
        object.__setattr__(self, "child", self.child.normalize())

    def fingerprint(self) -> str:
        """Return the semantic fingerprint of this negation.

        Purpose:
            Encode negation structure from child semantics while ignoring marker
            metadata.

        """
        return f"not({self.child.fingerprint()})"

    def evaluate(self, text: str, *, casefold: bool = True) -> MatchResult:
        """Evaluate the child and invert its match outcome.

        Purpose:
            Implement negation semantics while preserving child telemetry for
            downstream explanation.

        Architectural role:
            Unary evaluation step inside the match-tree runtime.

        Inputs (architectural provenance):
            Receives runtime guard text and case-fold policy.

        Outputs (downstream usage):
            Returns a negated ``MatchResult``.

        """
        child = self.child.evaluate(text, casefold=casefold)
        telemetry = NodeTelemetry(
            node_type="MatchNot",
            marker=self.marker,
            matched=not child.matched,
            spans=(),
            children=(child.telemetry,),
        )
        return MatchResult(
            matched=not child.matched, spans=(), telemetry=telemetry
        )

    def normalize(self) -> MatchTree:
        """Return the canonicalized negation node.

        Purpose:
            Expose the normalized ``MatchNot`` through the shared interface.

        Architectural role:
            Canonicalization hook for negation nodes.

        Inputs (architectural provenance):
            Called by parent nodes during normalization.

        Outputs (downstream usage):
            Returns ``self`` because child normalization already happened in
            ``__post_init__``.

        """
        return self

    def to_debug_string(self) -> str:
        """Render the negation in a readable debug form.

        Purpose:
            Provide a stable string representation of the negated child
            expression.

        Architectural role:
            Debug-formatting helper for ``MatchNot``.

        Inputs (architectural provenance):
            Uses the stored child node and marker.

        Outputs (downstream usage):
            Returns the formatted negation string.

        """
        return f"NOT({self.child.to_debug_string()})"

    def ordered_overlap_subtree(self) -> MatchTree | None:
        """Return the ordered-overlap-relevant subtree under this negation.

        Purpose:
            Preserve child structure for overlap analysis without adding extra
            negation semantics there.

        Architectural role:
            Overlap-analysis hook for negation nodes.

        Inputs (architectural provenance):
            Called by ordered-overlap extraction walks.

        Outputs (downstream usage):
            Returns the child's relevant subtree or ``None``.

        """
        return None


@dataclass(frozen=True, slots=True)
class MatchAndThen(MatchTree):
    """Ordered composite guard node requiring child matches in sequence.

    Purpose:
        Enforce ordering constraints between child matches and preserve only
        spans consistent with that order.

    Architectural role:
        Specialized ordered-overlap node for guards that care about sequence.

    Inputs (architectural provenance):
        Built by compilation from ordered group syntax.

    Outputs (downstream usage):
        Emits ordered ``MatchResult`` values and telemetry used by guard
        matching.

    """

    left: MatchTree
    right: MatchTree
    marker: str | None = None

    def __post_init__(self) -> None:
        """Normalize and validate ordered-match children at construction time.

        Purpose:
            Establish the canonical ordered sequence used by ``MatchAndThen``
            evaluation.

        Architectural role:
            Invariant hook for ordered composite guard nodes.

        Inputs (architectural provenance):
            Uses ordered children supplied by rule compilation.

        Outputs (downstream usage):
            Stores normalized children for ordered evaluation.

        """
        object.__setattr__(self, "left", self.left.normalize())
        object.__setattr__(self, "right", self.right.normalize())

    def fingerprint(self) -> str:
        """Return the semantic fingerprint of this ordered sequence.

        Purpose:
            Encode left-to-right ordered semantics while ignoring marker
            metadata.

        """
        return f"and_then({self.left.fingerprint()},{self.right.fingerprint()})"

    def evaluate(self, text: str, *, casefold: bool = True) -> MatchResult:
        """Evaluate children while enforcing left-to-right ordering of matches.

        Purpose:
            Succeed only when each child matches after the previous one,
            producing spans consistent with ordered guard semantics.

        Architectural role:
            Ordered composite evaluation step in the match-tree runtime.

        Inputs (architectural provenance):
            Receives runtime guard text and case-fold policy.

        Outputs (downstream usage):
            Returns an ordered ``MatchResult`` with telemetry for overlap and
            debug logic.

        """
        left = self.left.evaluate(text, casefold=casefold)
        right = self.right.evaluate(text, casefold=casefold)
        candidate_pairs = [
            Span(lspan.start, rspan.end)
            for lspan in left.spans
            for rspan in right.spans
            if lspan.end <= rspan.start
        ]

        # Guard nested ordered chains from selecting a middle
        # disjunct when another
        # disjunct already occurs before the chain anchor. This prevents
        # `(a ANDTHEN (b1 OR b2)) ANDTHEN c` from matching `b2 a b1 c`.
        if (
            candidate_pairs
            and isinstance(self.left, MatchAndThen)
            and isinstance(self.left.right, MatchAny)
        ):
            left_anchor = self.left.left.evaluate(text, casefold=casefold)
            middle_disjunction = self.left.right.evaluate(
                text, casefold=casefold
            )
            if left_anchor.spans:
                earliest_anchor_start = min(
                    anchor.start for anchor in left_anchor.spans
                )
                if any(
                    mid.start < earliest_anchor_start
                    for mid in middle_disjunction.spans
                ):
                    candidate_pairs = []

        spans = _dedup_spans(candidate_pairs)
        telemetry = NodeTelemetry(
            node_type="MatchAndThen",
            marker=self.marker,
            matched=bool(spans),
            spans=spans,
            children=(left.telemetry, right.telemetry),
        )
        return MatchResult(
            matched=bool(spans), spans=spans, telemetry=telemetry
        )

    def normalize(self) -> MatchTree:
        """Return the canonicalized ordered composite node.

        Purpose:
            Expose the constructor-normalized ordered sequence through the
            common tree interface.

        Architectural role:
            Canonicalization hook for ordered nodes.

        Inputs (architectural provenance):
            Called by normalization walks in parent nodes.

        Outputs (downstream usage):
            Returns ``self`` once child normalization has completed.

        """
        return self

    def to_debug_string(self) -> str:
        """Render the ordered composite in a readable debug form.

        Purpose:
            Provide a stable textual description of the sequence constraint
            represented by this node.

        Architectural role:
            Debug-formatting helper for ordered guard nodes.

        Inputs (architectural provenance):
            Uses normalized children and optional marker.

        Outputs (downstream usage):
            Returns a string for diagnostics and tests.

        """
        left_debug = self.left.to_debug_string()
        right_debug = self.right.to_debug_string()
        return f"AND_THEN({left_debug} -> {right_debug})"

    def ordered_overlap_subtree(self) -> MatchTree | None:
        """Return the ordered subtree represented by this node.

        Purpose:
            Surface the exact ordered structure that matters for overlap
            analysis.

        Architectural role:
            Ordered-overlap hook for sequence-sensitive guard nodes.

        Inputs (architectural provenance):
            Called by overlap analysis walking the tree.

        Outputs (downstream usage):
            Returns ``self`` or a reduced ordered subtree as appropriate.

        """
        return self.right


def _dedup_spans(spans: Iterable[Span]) -> tuple[Span, ...]:
    """Return spans without duplicates while preserving deterministic order.

    Purpose:
        Prevent composite match nodes from reporting repeated spans when
        multiple children contribute the same region.

    Architectural role:
        Small utility in match-result assembly.

    Inputs (architectural provenance):
        Receives spans emitted by child evaluations.

    Outputs (downstream usage):
        Returns a normalized span tuple used in ``MatchResult`` values.

    """
    return tuple(sorted(set(spans), key=lambda span: (span.start, span.end)))
