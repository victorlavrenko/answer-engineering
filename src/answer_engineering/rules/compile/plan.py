"""Executable plan schema for compiled rules.

These immutable values are emitted by the rules compiler and consumed by the
runtime. The schema is rules-owned, although some leaf fields still reference
shared engine types such as ``MatchTree`` and ``PatchOp``.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from answer_engineering.config.engine_defaults import (
    MatchDefaults,
    PolicyDefaults,
    ScopeDefaults,
)
from answer_engineering.config.inference_defaults import ProbeDefaults
from answer_engineering.engine.proposal.match_tree.match_tree import (
    MatchTree,
)
from answer_engineering.engine.runtime.runtime_primitives import (
    PatchOp,
)
from answer_engineering.rules.matching.options import ResolvedMatchOptions


@dataclass(frozen=True, slots=True)
class ScopeSpec:
    """Compiled description of the document window a rule may inspect.

    Purpose:
        Represent the runtime scope policy produced by the rules compiler.

    Architectural role:
        Compiler-to-runtime contract used by guard matching, edit targeting, and
        candidate generation.

    Inputs (architectural provenance):
        Constructed from authored rule scope modifiers or default scope
        settings.

    Outputs (downstream usage):
        Consumed by StepContext view construction to derive guard and edit
        windows.

    Invariants/constraints:
        Scope kinds define how `n` and `max_chars` are interpreted. Case
        handling and delimiter inclusion must remain explicit because downstream
        matching depends on the resolved window semantics.

    """

    kind: Literal[
        "whole_doc", "tail_clauses", "tail_chars", "tail_sentences"
    ] = "tail_chars"
    n: int | None = None
    max_chars: int | None = None
    casefold: bool = ScopeDefaults().casefold
    include_leading_delimiter: bool = False


@dataclass(frozen=True, slots=True)
class AnchorQuerySpec:
    """Compiled instructions for locating one named anchor.

    Purpose:
        Describe how runtime code should find an authored anchor phrase inside
        the active guard or edit view.

    Architectural role:
        Anchor-resolution contract between the parser/compiler and proposal
        logic.

    Inputs (architectural provenance):
        Built from rule-language anchor declarations and resolved match options.

    Outputs (downstream usage):
        Consumed by proposal prechecks and target-span computation.

    Invariants/constraints:
        `anchor_id` identifies the anchor for later target references. Match
        mode controls first/last selection, and phrase-level options override
        defaults only for the associated phrase.

    """

    anchor_id: str
    match_phrase_any: tuple[str, ...]
    match_mode: Literal["first", "last"] = "last"
    match_phrase_options: tuple[tuple[str, ResolvedMatchOptions], ...] = ()
    match_options: ResolvedMatchOptions = field(
        default_factory=lambda: ResolvedMatchOptions(
            casefold=MatchDefaults().casefold,
            word=MatchDefaults().word,
        )
    )


@dataclass(frozen=True, slots=True)
class EditTargetSpec:
    """Compiled plan for turning scope and anchors into an edit span.

    Purpose:
        Preserve the authored edit-target semantics after parsing so runtime
        code can compute an absolute patch span without reinterpreting rule
        text.

    Architectural role:
        Targeting contract consumed by proposal prechecks before candidates are
        converted into patch proposals.

    Inputs (architectural provenance):
        Constructed from rule target clauses, anchor references, and target
        defaults.

    Outputs (downstream usage):
        Consumed by `_compute_target_span` to produce the span later used by
        patching and conflict resolution.

    Invariants/constraints:
        Anchor-dependent target kinds require a matching resolved anchor id. The
        `include_anchor` flag must be preserved exactly because it changes the
        patch span and therefore conflict behavior.

    """

    kind: Literal[
        "match_span",
        "after_anchor_to_scope_end",
        "clause_containing_anchor_to_scope_end",
        "after_anchor_to_sentence_end",
        "after_anchor_to_clause_end",
        "scope_entire",
    ] = "scope_entire"
    anchor_id: str | None = None
    include_anchor: bool = False


@dataclass(frozen=True, slots=True)
class GuardSpec:
    """Compiled guard predicate required before a rule may fire.

    Purpose:
        Carry the match-tree expression that decides whether a rule is eligible
        in the current guard scope.

    Architectural role:
        Guard contract between compiled rules and proposal precheck logic.

    Inputs (architectural provenance):
        Built from rule-language guard expressions or left empty for unguarded
        rules.

    Outputs (downstream usage):
        Consumed by runtime matching before anchors, targets, and candidates are
        evaluated.

    Invariants/constraints:
        A `None` expression means there is no additional guard predicate, not
        that matching failed.

    """

    expression: MatchTree | None = None


@dataclass(frozen=True, slots=True)
class SpanSelectorSpec:
    """Compiled selector plan for match-derived spans.

    Purpose:
        Describe span discovery for rules whose edit location is selected from a
        phrase or regex match rather than a separately named anchor.

    Architectural role:
        Span-selection contract used by proposal logic for selector-style rules.

    Inputs (architectural provenance):
        Constructed from authored selector clauses and their phrase or regex
        match configuration.

    Outputs (downstream usage):
        Consumed when proposal prechecks derive the absolute span to edit.

    Invariants/constraints:
        Exactly the selector kind determines whether the span is the match
        itself, the scope, or text after the match. Include-anchor behavior must
        remain explicit because it changes the resulting patch range.

    """

    kind: Literal[
        "match_span",
        "after_match_to_scope_end",
        "scope_entire",
        "from_match_to_next_clause_end",
    ]
    match_phrase_any: tuple[str, ...] = ()
    match_regex: str | None = None
    include_anchor: bool = False


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One compiled rewrite, insert, or fallback candidate.

    The value is rules-owned, but its ``op`` still uses the shared engine-side
    ``PatchOp`` enum.

    """

    op: PatchOp
    text: str
    kind: Literal["static", "generated", "fallback"] = "static"
    priority: int = 0
    label: str = ""
    candidate_id: str = ""
    logprob: float | None = None


@dataclass(frozen=True, slots=True)
class DecisionPolicySpec:
    """Compiled runtime policy for validating and ranking candidates.

    Purpose:
        Carry per-rule decisions about probability thresholds, probing budget,
        skipped tokens, no-op allowance, and validation behavior.

    Architectural role:
        Policy contract connecting rule compilation to scoring, probing, and
        candidate gating.

    Inputs (architectural provenance):
        Constructed from authored rule policy clauses merged with engine
        defaults.

    Outputs (downstream usage):
        Consumed by proposal, scoring, and selection stages when deciding which
        candidate can become an accepted edit.

    Invariants/constraints:
        Defaults must match central configuration values so compiled rules
        remain deterministic across parser and runtime entrypoints.

    """

    min_prob_ratio_to_best: float | None = None
    skip_tokens: int = PolicyDefaults().skip_tokens
    probe_num_beams: int = ProbeDefaults().num_beams
    probe_max_new_tokens: int = ProbeDefaults().max_new_tokens
    allow_noop: bool = True
    validation_for_all: bool = PolicyDefaults().validation_for_all


@dataclass(frozen=True, slots=True)
class FirePolicySpec:
    """Compiled repeatability policy for one rule.

    Purpose:
        State whether a rule may fire once or repeatedly across a generation
        run.

    Architectural role:
        Runtime-control contract used by orchestration to prevent unintended
        repeated interventions.

    Inputs (architectural provenance):
        Built from authored fire policy or compiler defaults.

    Outputs (downstream usage):
        Consumed by runtime bookkeeping when deciding whether a previously fired
        rule remains eligible.

    Invariants/constraints:
        Mode values are intentionally narrow so orchestration can treat fire
        policy as a closed set.

    """

    mode: Literal["once", "repeat"]


@dataclass(frozen=True, slots=True)
class RulePlan:
    """Executable plan for one compiled rule.

    A ``RulePlan`` bundles the scope, guard, anchors, target behavior,
    candidates, and runtime policies needed to execute one authored rule.

    """

    rule_id: str
    name: str = ""
    scope: ScopeSpec = field(default_factory=ScopeSpec)
    guard_scope: ScopeSpec | None = None
    edit_scope: ScopeSpec | None = None
    guard: GuardSpec | None = None
    anchors: Sequence[AnchorQuerySpec] = field(default_factory=tuple)
    target: EditTargetSpec = field(default_factory=EditTargetSpec)
    selector: SpanSelectorSpec | None = None
    candidates: Sequence[CandidateSpec] = field(default_factory=tuple)
    policy: DecisionPolicySpec = field(default_factory=DecisionPolicySpec)
    fire: FirePolicySpec = field(
        default_factory=lambda: FirePolicySpec(mode="once")
    )
    wait_for_closing_parenthesis: bool = (
        PolicyDefaults().after_wait_for_closing_parenthesis
    )

    def effective_guard_scope(self) -> ScopeSpec:
        """Return the explicit guard scope, or fall back to the rule scope.

        Purpose:
            Materialize the scope in which guard phrases should be evaluated.

        Architectural role:
            Compile-plan accessor that makes authored scope inheritance explicit
            before rule execution.

        Inputs (architectural provenance):
            Reads the parsed rule-level scope and optional guard-specific scope
            stored on the immutable plan.

        Outputs (downstream usage):
            Returns the scope consumed by guard compilation and fingerprinting.

        Invariants/constraints:
            Guard scope falls back to the rule scope only when no guard override
            exists; callers should not repeat that inheritance logic.

        """
        return self.guard_scope or self.scope

    def effective_edit_scope(self) -> ScopeSpec:
        """Return the explicit edit scope, or fall back to the guard scope.

        Purpose:
            Materialize the scope in which replacement or avoidance edits are
            allowed to operate.

        Architectural role:
            Compile-plan accessor that centralizes scope inheritance for edit
            planning.

        Inputs (architectural provenance):
            Reads the parsed rule scope, optional guard scope, and optional edit
            scope stored on the immutable plan.

        Outputs (downstream usage):
            Returns the scope consumed by compiled edit specifications.

        Invariants/constraints:
            Edit scope falls back through the effective guard scope, which
            itself may fall back to the rule scope. Callers should treat this
            method as the single source of truth for edit-scope inheritance.

        """
        return self.edit_scope or self.effective_guard_scope()


@dataclass(frozen=True, slots=True)
class PlanIR:
    """Compiled ruleset payload handed from compiler to runtime.

    Purpose:
        Bundle all executable rule plans and the schema version that describes
        their runtime interpretation.

    Architectural role:
        Top-level compiler/runtime boundary for compiled rules.

    Inputs (architectural provenance):
        Produced by the rule compiler after parsing, resolving defaults, and
        building per-rule plans.

    Outputs (downstream usage):
        Consumed by `CompiledRules`, stream sessions, orchestration, and tests
        that inspect compiled rule behavior.

    Invariants/constraints:
        `plan_version` must change when runtime interpretation of the plan
        schema changes in a non-trivial way.

    """

    rules: Sequence[RulePlan]
    plan_version: str = "core-full-v2"
