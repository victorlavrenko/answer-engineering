"""Canonical extension points for parser, compiler, and runtime integration.

Purpose:
    Re-export the small set of collaboration interfaces that external code may
    use to plug into Answer Engineering without importing deep implementation
    modules.

Architectural role:
    Public extension boundary spanning rules parsing, plan compilation,
    candidate proposal, scoring, and telemetry sinks.

Inputs (architectural provenance):
    Imports stable protocol or implementation entry points from their owning
    subsystems.

Outputs (downstream usage):
    Provides a single import surface for integrations, examples, notebooks, and
    downstream packages that need extension hooks.

Invariants/constraints:
    This module should expose boundary objects only. Adding convenience aliases
    here should be treated as an API decision, not as an implementation
    shortcut.

"""

from __future__ import annotations

from answer_engineering.engine.proposal.candidates.base import (
    CandidateProvider,
)
from answer_engineering.engine.scoring.base import Scorer
from answer_engineering.engine.telemetry.events.event_sink import (
    RuntimeEventSink,
)
from answer_engineering.rules.compile.compiler import (
    FullPlanCompiler,
)
from answer_engineering.rules.parse.parser import (
    MarkdownRulesParser,
)

__all__ = [
    "CandidateProvider",
    "FullPlanCompiler",
    "MarkdownRulesParser",
    "RuntimeEventSink",
    "Scorer",
]
