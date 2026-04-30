"""Matching-option contracts shared by rule compilation and runtime matching.

Purpose:
    Define the resolved match-policy shape consumed after authored rule options
    have been parsed and defaulted.

Architectural role:
    Boundary module between the rules domain-specific language and text-matching
    implementation. It keeps runtime matchers independent of parser-specific
    option objects.

Inputs (architectural provenance):
    Receives normalized option values from parse/compile helpers rather than
    directly from raw markdown rules.

Outputs (downstream usage):
    Exposes immutable option containers passed to phrase, guard, and anchor
    matching code.

Invariants/constraints:
    Objects here should represent executable matching policy, not partially
    resolved authored syntax.

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResolvedMatchOptions:
    """Fully resolved matching options consumed by runtime text matchers.

    Purpose:
        Carry the effective case and word-boundary behavior after item, section,
        rule, and default precedence has already been applied.

    Architectural role:
        Runtime-facing value object at the parse/compile-to-matching boundary.
        It prevents matchers from depending on authored override hierarchy.

    Inputs (architectural provenance):
        Constructed by compiler-side option resolution from parsed
        domain-specific language options and repository defaults.

    Outputs (downstream usage):
        Passed into compiled phrase, guard, and anchor matchers wherever
        effective text-matching policy is required.

    Invariants/constraints:
        Values must be complete booleans. Runtime matching code should never
        receive `None` or need to re-run option precedence.

    Todo:
        Extend this model deliberately if the domain-specific language adds
        richer match modes, such as explicit regex-mode selection semantics.

    """

    casefold: bool
    word: bool
