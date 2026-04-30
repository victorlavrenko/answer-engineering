"""Compiler from rules abstract syntax trees to executable immutable plan.

Convert parsed markdown rule objects into the normalized plan schema consumed by
runtime execution.

Architectural role:
    Compilation stage between the rules abstract-syntax-tree layer and the
    engine's executable plan consumers.

Current architecture notes:
    The compiler is the right owner for plan assembly, but some leaf contracts
    it emits still come from engine-side shared types such as ``PatchOp``.

"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import singledispatchmethod

from answer_engineering.config.engine_defaults import (
    MatchDefaults,
    PolicyDefaults,
    ScopeDefaults,
)
from answer_engineering.config.inference_defaults import ProbeDefaults
from answer_engineering.engine.proposal.match_tree.match_tree import (
    MatchAll,
    MatchAndThen,
    MatchAny,
    MatchNot,
    MatchTerm,
    MatchTree,
)
from answer_engineering.engine.runtime.runtime_primitives import (
    PatchOp,
)
from answer_engineering.rules.compile.plan import (
    AnchorQuerySpec,
    CandidateSpec,
    DecisionPolicySpec,
    EditTargetSpec,
    FirePolicySpec,
    GuardSpec,
    PlanIR,
    RulePlan,
    ScopeSpec,
)
from answer_engineering.rules.matching.options import ResolvedMatchOptions
from answer_engineering.rules.parse.ast import (
    AfterRuleAST,
    AvoidRuleAST,
    ForceRuleAST,
    ReplaceRuleAST,
    RuleAST,
    RulesetAST,
    ScopeAST,
    SetMatchAST,
)


@dataclass(frozen=True, slots=True)
class FullPlanCompiler:
    """Compiler for the full markdown-rules feature set.

    Purpose:
        Assemble executable ``RulePlan`` values from parsed rule-family
        abstract-syntax-tree objects and package them as one ``PlanIR``.

    Architectural role:
        Main behavior-owning compiler object in the rules boundary.

    """

    def compile(self, ast: RulesetAST) -> PlanIR:
        """Compile one parsed ruleset into canonical executable plan IR.

        Purpose:
            Preserve parsed rule order while converting every
            abstract-syntax-tree rule into its runtime ``RulePlan``
            representation.

        """
        rules = [self._compile_rule(r) for r in ast.rules]
        return PlanIR(rules=tuple(rules), plan_version="core-full-v2")

    def stable_rule_id(self, rule: RuleAST) -> str:
        """Derive a deterministic compiled rule id from one parsed rule.

        Purpose:
            Keep the authored rule id while adding a short content fingerprint
            so compiled plan ids stay stable across repeated compilation of the
            same rule payload.

        """
        digest = hashlib.sha1(
            _rule_fingerprint_payload(rule).encode("utf-8")
        ).hexdigest()[:10]
        return f"ng-{rule.rule_id}-{digest}"

    @singledispatchmethod
    def _compile_rule(self, rule: RuleAST) -> RulePlan:
        """Compile one parsed rule through the family-specific dispatch path.

        Purpose:
            Provide the common dispatch entry used by ``compile`` while giving
            unsupported rule types a deterministic placeholder plan instead of
            crashing.

        """
        rid = self.stable_rule_id(rule)
        return RulePlan(rule_id=rid, name="unsupported")

    @_compile_rule.register
    def _(self, rule: ReplaceRuleAST) -> RulePlan:
        """Compile a replace-rule AST through the singledispatch entry point.

        Purpose:
            Route ``ReplaceRuleAST`` values from the generic compile dispatch
            into the replace-rule compiler without forcing callers to branch on
            rule family themselves.

        Architectural role:
            Family-specific singledispatch arm inside ``FullPlanCompiler``.

        Inputs (architectural provenance):
            Invoked by ``_compile_rule`` when the parsed abstract-syntax-tree
            object is a replace rule.

        Outputs (downstream usage):
            Returns the executable ``RulePlan`` assembled by
            ``_compile_replace_rule``.

        """
        return self._compile_replace_rule(rule, self.stable_rule_id(rule))

    @_compile_rule.register
    def _(self, rule: AfterRuleAST) -> RulePlan:
        """Compile an after-rule AST through the singledispatch entry point.

        Purpose:
            Route ``AfterRuleAST`` values from the generic compile dispatch into
            the after-rule compiler while keeping rule-family branching inside
            the compiler boundary.

        Architectural role:
            Family-specific singledispatch arm inside ``FullPlanCompiler``.

        Inputs (architectural provenance):
            Invoked by ``_compile_rule`` when the parsed abstract-syntax-tree
            object is an after rule.

        Outputs (downstream usage):
            Returns the executable ``RulePlan`` assembled by
            ``_compile_after_rule``.

        """
        return self._compile_after_rule(rule, self.stable_rule_id(rule))

    @_compile_rule.register
    def _(self, rule: ForceRuleAST) -> RulePlan:
        """Compile a force-rule AST through the singledispatch entry point.

        Purpose:
            Route ``ForceRuleAST`` values from generic compiler dispatch into
            the force-rule compilation path while preserving one canonical entry
            point for callers.

        Architectural role:
            Family-specific singledispatch arm inside ``FullPlanCompiler``.

        Inputs (architectural provenance):
            Invoked by ``_compile_rule`` when the parsed abstract-syntax-tree
            object is a force rule.

        Outputs (downstream usage):
            Returns the executable ``RulePlan`` assembled by
            ``_compile_force_rule``.

        """
        return self._compile_force_rule(rule, self.stable_rule_id(rule))

    @_compile_rule.register
    def _(self, rule: AvoidRuleAST) -> RulePlan:
        """Compile an avoid-rule AST through the singledispatch entry point.

        Purpose:
            Route ``AvoidRuleAST`` values from generic compiler dispatch into
            the avoid-rule compilation path, where edit behavior, guards,
            fallbacks, and probing options are assembled.

        Architectural role:
            Family-specific singledispatch arm inside ``FullPlanCompiler``.

        Inputs (architectural provenance):
            Invoked by ``_compile_rule`` when the parsed abstract-syntax-tree
            object is an avoid rule.

        Outputs (downstream usage):
            Returns the executable ``RulePlan`` assembled by
            ``_compile_avoid_rule``.

        """
        return self._compile_avoid_rule(rule, self.stable_rule_id(rule))

    def _compile_replace_rule(self, rule: ReplaceRuleAST, rid: str) -> RulePlan:
        """Compile one replace-rule AST into a replace-style executable plan.

        Purpose:
            Build the full replace-rule runtime contract: effective scope, match
            anchor query, match-span target, static rewrite candidates, optional
            gate guard, and fire policy.

        Architectural role:
            Replace-family compilation routine inside ``FullPlanCompiler``.

        Inputs (architectural provenance):
            Receives one parsed ``ReplaceRuleAST`` plus the stable compiled rule
            id already assigned for this compilation pass.

        Outputs (downstream usage):
            Produces a ``RulePlan`` consumed by proposal and runtime execution
            code.

        """
        scope = _scope(rule.scope)
        return RulePlan(
            rule_id=rid,
            name=f"replace:{rule.target}",
            scope=scope,
            guard_scope=scope,
            target=EditTargetSpec(kind="match_span", anchor_id="match"),
            anchors=(
                AnchorQuerySpec(
                    anchor_id="match",
                    match_phrase_any=(rule.target,),
                    match_mode="last",
                    match_options=_resolved_rule_match_options(rule),
                ),
            ),
            candidates=tuple(
                CandidateSpec(
                    candidate_id=f"rewrite_{i + 1}",
                    op=PatchOp.REPLACE,
                    text=c,
                    kind="static",
                    priority=10,
                    label=c,
                )
                for i, c in enumerate(rule.candidates)
            ),
            guard=_build_gate_guard(rule.gate),
            policy=DecisionPolicySpec(),
            fire=FirePolicySpec(mode=rule.fire),
        )

    def _compile_after_rule(self, rule: AfterRuleAST, rid: str) -> RulePlan:
        """Compile one after-rule AST into an after-anchor executable plan.

        Purpose:
            Build after-anchor targeting, normalized anchor phrase variants,
            insertion candidates, optional gate guard, and parenthesis-wait
            policy for an after rule.

        """
        scope = _scope(rule.scope)
        return RulePlan(
            rule_id=rid,
            name=f"after:{rule.target}",
            scope=scope,
            guard_scope=scope,
            target=EditTargetSpec(
                kind="after_anchor_to_scope_end", anchor_id="anchor"
            ),
            anchors=(
                AnchorQuerySpec(
                    anchor_id="anchor",
                    match_phrase_any=_after_anchor_phrases(rule.target),
                    match_mode="last",
                    match_options=_resolved_rule_match_options(rule),
                ),
            ),
            candidates=tuple(
                CandidateSpec(
                    candidate_id=f"insert_{i + 1}",
                    op=PatchOp.REPLACE,
                    text=c,
                    kind="static",
                    priority=10,
                    label=c,
                )
                for i, c in enumerate(rule.candidates)
            ),
            guard=_build_gate_guard(rule.gate),
            policy=DecisionPolicySpec(),
            fire=FirePolicySpec(mode=rule.fire),
            wait_for_closing_parenthesis=rule.wait_for_closing_parenthesis,
        )

    def _compile_force_rule(self, rule: ForceRuleAST, rid: str) -> RulePlan:
        """Compile one force-rule AST into a scope-wide force plan.

        Purpose:
            Convert force additions into compiled static candidates and attach
            the rule's effective scope and fire policy so runtime code can
            enforce the statement across the intended region.

        Architectural role:
            Force-family compilation routine inside ``FullPlanCompiler``.

        Inputs (architectural provenance):
            Receives one parsed ``ForceRuleAST`` plus the stable compiled rule
            id already assigned for this compilation pass.

        Outputs (downstream usage):
            Produces a ``RulePlan`` consumed by later proposal and apply-side
            execution logic.

        """
        scope = _scope(rule.scope)
        return RulePlan(
            rule_id=rid,
            name=f"force:{rule.target}",
            scope=scope,
            guard_scope=scope,
            target=EditTargetSpec(kind="scope_entire"),
            candidates=tuple(
                CandidateSpec(
                    candidate_id=f"force_{i + 1}",
                    op=PatchOp.REPLACE,
                    text=c,
                    kind="static",
                    priority=10,
                    label=c,
                )
                for i, c in enumerate(rule.add)
            ),
            policy=DecisionPolicySpec(),
            fire=FirePolicySpec(mode=rule.fire),
        )

    def _compile_avoid_rule(self, rule: AvoidRuleAST, rid: str) -> RulePlan:
        """Compile one avoid-rule AST into an avoid-style executable plan.

        Purpose:
            Translate avoid-rule guard semantics, edit behavior, fallback
            candidates, and probing-related numeric options into one normalized
            runtime ``RulePlan``.

        Architectural role:
            Avoid-family compilation routine inside ``FullPlanCompiler``.

        Inputs (architectural provenance):
            Receives one parsed ``AvoidRuleAST`` plus the stable compiled rule
            id already assigned for this compilation pass.

        Outputs (downstream usage):
            Produces a ``RulePlan`` consumed by proposal, probing, scoring, and
            apply-side runtime logic.

        Current architecture notes:
            This routine is the main place where rule-language avoid semantics
            are converted into engine-facing execution contracts.

        """
        scope = _scope(rule.scope)
        edit_scope, anchors, target = _avoid_edit_behavior(rule, scope)
        return RulePlan(
            rule_id=rid,
            name=f"avoid:{rule.target}",
            scope=scope,
            guard_scope=scope,
            edit_scope=edit_scope,
            guard=GuardSpec(expression=rule.guard_expression),
            anchors=anchors,
            target=target,
            candidates=tuple(
                CandidateSpec(
                    candidate_id=f"fallback_{i + 1}",
                    op=PatchOp.REPLACE,
                    text=txt,
                    kind="fallback",
                    priority=10 - i,
                    label=f"fallback_{i + 1}",
                )
                for i, txt in enumerate(rule.fallback)
            ),
            policy=DecisionPolicySpec(
                min_prob_ratio_to_best=(
                    float(rule.options["min_prob_ratio_to_best"])
                    if "min_prob_ratio_to_best" in rule.options
                    else None
                ),
                skip_tokens=int(
                    rule.options.get("skip", PolicyDefaults().skip_tokens)
                ),
                probe_num_beams=max(
                    1,
                    int(
                        rule.options.get(
                            "probe_num_beams", ProbeDefaults().num_beams
                        )
                    ),
                ),
                probe_max_new_tokens=max(
                    0,
                    int(
                        rule.options.get(
                            "probe_max_new_tokens",
                            ProbeDefaults().max_new_tokens,
                        )
                    ),
                ),
                allow_noop=True,
            ),
            fire=FirePolicySpec(mode=rule.fire),
        )


def _scope(scope: ScopeAST | None) -> ScopeSpec:
    """Convert an AST scope declaration into executable ``ScopeSpec`` form.

    Purpose:
        Normalize whole-document, tail-character, tail-clause, and tail-sentence
        scopes while filling in default casefold behavior when needed.

    """
    if scope is None:
        return ScopeSpec(kind="whole_doc", casefold=ScopeDefaults().casefold)
    if scope.kind == "tail_chars":
        return ScopeSpec(
            kind="tail_chars", max_chars=scope.n, casefold=scope.casefold
        )
    if scope.kind == "tail_clauses":
        return ScopeSpec(
            kind="tail_clauses", n=scope.n, casefold=scope.casefold
        )
    if scope.kind == "tail_sentences":
        return ScopeSpec(
            kind="tail_sentences", n=scope.n, casefold=scope.casefold
        )
    return ScopeSpec(kind="whole_doc", casefold=scope.casefold)


def _rule_fingerprint_payload(rule: RuleAST) -> str:
    """Build the text payload used for deterministic compiled-rule IDs.

    Purpose:
        Convert the semantic fields of one compiled rule plan into a stable
        payload for identity hashing.

    Architectural role:
        Compiler-side identity boundary used to make rule IDs reproducible
        across runs and independent of incidental object identity.

    Inputs (architectural provenance):
        Receives the parsed and normalized rule plan plus compiler-derived
        guards, edits, scopes, and options.

    Outputs (downstream usage):
        Returns a text payload that is hashed into the compiled rule
        fingerprint.

    Invariants/constraints:
        Only semantic rule content should participate. Formatting or ordering
        that is not meaningful to runtime behavior must be normalized before
        hashing.

    """
    payload: dict[str, object]
    if isinstance(rule, ReplaceRuleAST):
        payload = {
            "kind": "replace",
            "rule_id": rule.rule_id,
            "fire": rule.fire,
            "scope": _scope_fingerprint(rule.scope),
            "target": rule.target,
            "candidates": tuple(rule.candidates),
            "gate": _set_match_fingerprint(rule.gate),
        }
    elif isinstance(rule, AfterRuleAST):
        payload = {
            "kind": "after",
            "rule_id": rule.rule_id,
            "fire": rule.fire,
            "scope": _scope_fingerprint(rule.scope),
            "target": rule.target,
            "candidates": tuple(rule.candidates),
            "gate": _set_match_fingerprint(rule.gate),
            "wait_for_closing_parenthesis": rule.wait_for_closing_parenthesis,
        }
    elif isinstance(rule, AvoidRuleAST):
        payload = {
            "kind": "avoid",
            "rule_id": rule.rule_id,
            "fire": rule.fire,
            "scope": _scope_fingerprint(rule.scope),
            "target": rule.target,
            "edit": {
                "kind": rule.edit.kind,
                "n_sentences": rule.edit.n_sentences,
                "n_clauses": rule.edit.n_clauses,
            },
            "guard_expression": (
                None
                if rule.guard_expression is None
                else rule.guard_expression.fingerprint()
            ),
            "connector_terms": tuple(rule.connector_terms),
            "fallback": tuple(rule.fallback),
            "options": _stable_options_items(rule.options),
        }
    elif isinstance(rule, ForceRuleAST):
        payload = {
            "kind": "force",
            "rule_id": rule.rule_id,
            "fire": rule.fire,
            "scope": _scope_fingerprint(rule.scope),
            "target": rule.target,
            "add": tuple(rule.add),
        }
    else:
        payload = {
            "kind": "unsupported",
            "rule_type": type(rule).__name__,
            "rule_id": rule.rule_id,
            "fire": rule.fire,
            "scope": _scope_fingerprint(rule.scope),
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _after_anchor_phrases(target: str) -> tuple[str, ...]:
    """Generate the anchor phrase variants used by after rules.

    Purpose:
        Expand authored after-rule anchors into the phrase forms that can
        trigger post-anchor edit behavior.

    Architectural role:
        Compiler helper that keeps anchor expansion separate from runtime
        matching and proposal construction.

    Inputs (architectural provenance):
        Receives parsed rule-plan sections and marker-scoped phrase options.

    Outputs (downstream usage):
        Returns normalized anchor phrases consumed by guard/edit compilation.

    Invariants/constraints:
        Expansion must preserve authored matching options while avoiding
        duplicate semantic anchors in the compiled representation.

    """
    bases = [target]
    if target and not target.startswith("("):
        bases.extend((f"({target})", f"({target}"))
    phrases: list[str] = []
    for base in bases:
        phrases.extend((f"{base}.", f"{base},", base))
    return tuple(dict.fromkeys(phrases))


def _build_gate_guard(gate: SetMatchAST | None) -> GuardSpec | None:
    """Wrap a parsed gate expression in `GuardSpec` form when present.

    Purpose:
        Convert the optional parsed gate condition attached to a rule into the
        guard object expected by executable rule plans.

    Architectural role:
        Compile-boundary adapter between parse abstract-syntax-tree condition
        syntax and runtime guard evaluation.

    Inputs (architectural provenance):
        Receives a parsed gate node from a rule abstract-syntax-tree after
        family-specific compile dispatch has selected the rule plan shape.

    Outputs (downstream usage):
        Returns a `GuardSpec` consumed by proposal guards, or `None` when the
        rule has no authored gate.

    Invariants/constraints:
        The helper should not evaluate the guard. It only preserves authored
        semantics in the immutable plan representation.

    """
    if gate is None:
        return None
    return GuardSpec(expression=gate.expression)


def _avoid_edit_behavior(
    rule: AvoidRuleAST,
    scope: ScopeSpec,
) -> tuple[ScopeSpec, tuple[AnchorQuerySpec, ...], EditTargetSpec]:
    if rule.edit.kind == "postfix":
        postfix_phrase_options = _phrase_options_for_markers(
            tree=rule.guard_expression,
            marker_prefixes=("postfix_all", "postfix_any"),
            fallback_options=_resolved_rule_match_options(rule),
        )
        postfix_terms = tuple(
            dict.fromkeys(term for term, _ in postfix_phrase_options)
        )
        if not postfix_terms:
            return (scope, (), EditTargetSpec(kind="scope_entire"))
        return (
            scope,
            (
                AnchorQuerySpec(
                    anchor_id="postfix_match",
                    match_phrase_any=postfix_terms,
                    match_mode="last",
                    match_phrase_options=postfix_phrase_options,
                    match_options=_resolved_rule_match_options(rule),
                ),
            ),
            EditTargetSpec(
                kind="clause_containing_anchor_to_scope_end",
                anchor_id="postfix_match",
                include_anchor=True,
            ),
        )
    if rule.edit.kind == "tail_sentences":
        n = rule.edit.n_sentences or 1
        return (
            ScopeSpec(kind="tail_sentences", n=n, casefold=scope.casefold),
            (),
            EditTargetSpec(kind="scope_entire"),
        )
    if rule.edit.kind == "tail_clauses":
        n = rule.edit.n_clauses or 1
        return (
            ScopeSpec(
                kind="tail_clauses",
                n=n,
                casefold=scope.casefold,
                include_leading_delimiter=True,
            ),
            (),
            EditTargetSpec(kind="scope_entire"),
        )
    if rule.edit.kind == "prefix_clause":
        prefix_phrase_options = _phrase_options_for_markers(
            tree=rule.guard_expression,
            marker_prefixes=("prefix_all", "prefix_any"),
            fallback_options=_resolved_rule_match_options(rule),
        )
        prefix_terms = tuple(
            dict.fromkeys(term for term, _ in prefix_phrase_options)
        )
        if not prefix_terms:
            return (scope, (), EditTargetSpec(kind="scope_entire"))
        return (
            scope,
            (
                AnchorQuerySpec(
                    anchor_id="prefix_match",
                    match_phrase_any=prefix_terms,
                    match_mode="last",
                    match_phrase_options=prefix_phrase_options,
                    match_options=_resolved_rule_match_options(rule),
                ),
            ),
            EditTargetSpec(
                kind="clause_containing_anchor_to_scope_end",
                anchor_id="prefix_match",
                include_anchor=True,
            ),
        )
    return (scope, (), EditTargetSpec(kind="scope_entire"))


def _scope_fingerprint(scope: ScopeAST | None) -> dict[str, object] | None:
    """Return semantic scope content as a stable dictionary payload.

    Purpose:
        Represent a compiled scope in a deterministic, hashable-friendly shape
        used for rule identity construction.

    Architectural role:
        Fingerprinting helper at the compile boundary. It prevents rule IDs from
        depending on object identity or incidental dataclass formatting.

    Inputs (architectural provenance):
        Receives the parsed or compiled scope object attached to one rule
        family.

    Outputs (downstream usage):
        Returns a dictionary payload folded into stable rule-id generation and
        telemetry-facing provenance.

    Invariants/constraints:
        The payload should include only semantic scope fields. Cosmetic source
        formatting and transient compiler state must not affect the fingerprint.

    """
    if scope is None:
        return None
    return {
        "kind": scope.kind,
        "n": scope.n,
        "casefold": scope.casefold,
    }


def _set_match_fingerprint(gate: SetMatchAST | None) -> str | None:
    """Return semantic gate fingerprint derived from match-tree nodes.

    Purpose:
        Normalize a set-match expression into stable structural data for
        compiled rule identity.

    Architectural role:
        Compiler fingerprint helper for condition trees. It keeps identity
        generation aligned with rule semantics rather than parser object layout.

    Inputs (architectural provenance):
        Receives match-tree nodes produced by the parser and carried through the
        abstract-syntax-tree.

    Outputs (downstream usage):
        Returns a deterministic payload used by stable rule-id construction.

    Invariants/constraints:
        Equivalent authored match trees should produce the same fingerprint even
        when source whitespace or parser object identities differ.

    """
    if gate is None or gate.expression is None:
        return None
    return gate.expression.fingerprint()


def _stable_options_items(
    options: dict[str, float | int],
) -> tuple[tuple[str, float | int], ...]:
    """Return avoid options in deterministic key order for hashing payloads."""
    return tuple(sorted(options.items(), key=lambda item: item[0]))


def _resolved_rule_match_options(rule: RuleAST) -> ResolvedMatchOptions:
    defaults = MatchDefaults()
    return ResolvedMatchOptions(
        casefold=(
            rule.match_options.casefold
            if rule.match_options.casefold is not None
            else defaults.casefold
        ),
        word=(
            rule.match_options.word
            if rule.match_options.word is not None
            else defaults.word
        ),
    )


def _phrase_options_for_markers(
    *,
    tree: MatchTree | None,
    marker_prefixes: tuple[str, ...],
    fallback_options: ResolvedMatchOptions,
) -> tuple[tuple[str, ResolvedMatchOptions], ...]:
    """Recover marker-scoped phrase options for anchor lookup.

    Purpose:
        Reassociate parsed marker positions with the phrase-level matching
        options that were authored around those markers.

    Architectural role:
        Compile-side bridge between template parsing and runtime anchor lookup.
        It keeps marker extraction from losing local match-option overrides.

    Inputs (architectural provenance):
        Receives parsed marker metadata and phrase option records from the rule
        abstract-syntax-tree.

    Outputs (downstream usage):
        Returns ordered `(phrase, options)` pairs consumed when anchor terms are
        compiled for runtime matching.

    Invariants/constraints:
        The returned pairs must preserve authored order and option scope so
        anchor lookup observes the same case, word-boundary, and matching policy
        that the rule author wrote.

    """
    if tree is None:
        return tuple()
    phrase_options: list[tuple[str, ResolvedMatchOptions]] = []
    for node in _iter_match_terms(tree):
        marker = node.marker or ""
        if not any(marker.startswith(prefix) for prefix in marker_prefixes):
            continue
        phrase_options.append(
            (node.expression, node.match_options or fallback_options)
        )
    return tuple(dict.fromkeys(phrase_options))


def _iter_match_terms(tree: MatchTree) -> tuple[MatchTerm, ...]:
    if isinstance(tree, MatchTerm):
        return (tree,)
    children: list[MatchTree] = []
    if isinstance(tree, MatchAll | MatchAny):
        children.extend(tree.children)
    if isinstance(tree, MatchNot):
        children.append(tree.child)
    if isinstance(tree, MatchAndThen):
        children.append(tree.left)
        children.append(tree.right)
    out: list[MatchTerm] = []
    for child in children:
        out.extend(_iter_match_terms(child))
    return tuple(out)
