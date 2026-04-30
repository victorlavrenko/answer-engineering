"""Core plan-runner defaults used across Answer Engineering.

Purpose:
    Collect runtime constants that shape orchestration, proposal retries, and
    default execution limits.

Architectural role:
    Configuration boundary for engine-level policy. The module keeps shared
    defaults out of orchestration code so runtime behavior is easier to audit
    and tune.

Inputs (architectural provenance):
    Values are authored as repository constants and imported by engine
    collaborators that need deterministic defaults.

Outputs (downstream usage):
    Exposes constants consumed by runtime construction, plan execution, and test
    fixtures that assert default behavior.

Invariants/constraints:
    Defaults should remain simple immutable values. Policy-specific explanation
    belongs here rather than at each call site that consumes the value.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SetMatchDefault = Literal["any", "all", "none", "incomplete"]


@dataclass(frozen=True, slots=True)
class ScopeDefaults:
    """Scope-level default parameters used when rules omit explicit scope.

    Purpose:
        Provide canonical fallback values for guard/edit scope construction,
        including tail length, sentence/clause limits, and case-fold behavior.

    Architectural role:
        Configuration value object shared by compiler-side rule normalization
        and runtime view-building code.

    Inputs (architectural provenance):
        Referenced when compiled rules or runtime helpers need a scope attribute
        but the source ruleset did not specify one.

    Outputs (downstream usage):
        Supplies concrete defaults that become part of effective scope specs
        used by guard matching and proposal targeting.

    Invariants/constraints:
        Values are immutable process defaults; they do not carry per-run state.

    """

    tail_chars: int = 800
    sentences: int = 1
    clauses: int = 1
    casefold: bool = True


@dataclass(frozen=True, slots=True)
class RuleDefaults:
    """Rule-level defaults for generic firing and rewrite behavior.

    Purpose:
        Centralize default policy literals for fire-once/repeat behavior and the
        avoid-rewrite mode used when rules omit these options.

    Architectural role:
        Shared constant bundle consumed during rule compilation and
        normalization.

    Inputs (architectural provenance):
        Read by parser/compiler code when source markdown does not state the
        corresponding rule policy explicitly.

    Outputs (downstream usage):
        Produces normalized rule settings that later control proposal generation
        and repeat-cycle handling.

    Invariants/constraints:
        The fields encode canonical literals expected by downstream rule logic.

    """

    fire_once: Literal["once"] = "once"
    fire_repeat: Literal["repeat"] = "repeat"
    avoid_rewrite: Literal["everything"] = "everything"


@dataclass(frozen=True, slots=True)
class PolicyDefaults:
    """Global policy defaults applied to runtime rule execution.

    Purpose:
        Define fallback values for token skipping, parenthesis-handling after
        inserts, and validation defaults when no explicit policy is provided.

    Architectural role:
        Immutable defaults bundle bridging rule normalization and runtime
        proposal logic.

    Inputs (architectural provenance):
        Consulted by compilation/runtime setup when building effective
        generation policy from partial user-specified configuration.

    Outputs (downstream usage):
        Feeds effective policy flags consumed by proposal generation and
        validation decisions.

    Invariants/constraints:
        Fields are process-wide defaults and must remain deterministic.

    """

    skip_tokens: int = 0
    after_wait_for_closing_parenthesis: bool = True
    validation_for_all: bool = True


@dataclass(frozen=True, slots=True)
class MatchDefaults:
    """Default matching semantics for guard operators and text-matcher options.

    Purpose:
        Provide canonical fallback set operators and engine-level fallback
        matching options (case folding and whole-word mode).

    Architectural role:
        Configuration record used while translating parsed rule syntax into
        effective match-tree expectations.

    Inputs (architectural provenance):
        Used when a ruleset omits explicit set-match directives for one of the
        supported guard families.

    Outputs (downstream usage):
        Supplies normalized match modes that shape guard evaluation in proposal
        prechecks.

    Invariants/constraints:
        Each field must remain one of the compiler-recognized set-match
        literals.

    """

    replace_prefix_match: SetMatchDefault = "any"
    after_prefix_match: SetMatchDefault = "any"
    avoid_prefix_match: SetMatchDefault = "all"
    avoid_postfix_match: SetMatchDefault = "all"
    casefold: bool = True
    word: bool = False


__all__ = [
    "MatchDefaults",
    "PolicyDefaults",
    "RuleDefaults",
    "ScopeDefaults",
    "SetMatchDefault",
]
