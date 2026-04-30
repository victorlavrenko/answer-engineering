"""Parse/compile helpers for resolving authored match-option overrides.

Purpose:
    Convert parser-level option objects into executable matching policy by
    applying item, section, rule, and default precedence.

Architectural role:
    Narrow boundary between authored domain-specific language syntax and
    compiled runtime matching options.

Inputs (architectural provenance):
    Receives `MatchOptionsAST` values produced by the parser plus compiler
    defaults from configuration.

Outputs (downstream usage):
    Produces `ResolvedMatchOptions` values consumed by compiled match
    specifications.

Invariants/constraints:
    Local authored values override broader values only when explicitly set. The
    runtime should never need to inspect parser abstract-syntax-tree option
    objects.

"""

from __future__ import annotations

from answer_engineering.config.engine_defaults import MatchDefaults
from answer_engineering.rules.matching.options import ResolvedMatchOptions
from answer_engineering.rules.parse.ast import MatchOptionsAST


def resolve_match_options(
    *,
    item: MatchOptionsAST,
    section: MatchOptionsAST,
    rule: MatchOptionsAST,
    defaults: MatchDefaults,
) -> ResolvedMatchOptions:
    """Resolve item>section>rule>default match-option precedence.

    Purpose:
        Merge match-option overrides from the most local authored scope back to
        the global defaults.

    Architectural role:
        Rule-language semantics helper used while compiling parsed rule
        structures into executable match specifications.

    Inputs (architectural provenance):
        Receives `MatchOptions` values parsed from an item, its section, its
        rule, and the compiler defaults.

    Outputs (downstream usage):
        Returns the effective `MatchOptions` consumed by compiled phrase, guard,
        and anchor matching.

    Invariants/constraints:
        Local values override broader values only when explicitly set. Defaults
        are the final fallback so compiled rules never need to resolve option
        precedence during runtime matching.

    """
    raw_casefold = (
        item.casefold
        if item.casefold is not None
        else section.casefold
        if section.casefold is not None
        else rule.casefold
        if rule.casefold is not None
        else defaults.casefold
    )
    raw_word = (
        item.word
        if item.word is not None
        else section.word
        if section.word is not None
        else rule.word
        if rule.word is not None
        else defaults.word
    )
    return ResolvedMatchOptions(
        casefold=bool(raw_casefold), word=bool(raw_word)
    )
