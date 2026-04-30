"""Answer Engineering rules public package.

Purpose:
    Expose the narrow stable rules-language surface used by external callers and
    downstream packages.

Architectural role:
    Public package boundary for rules parsing and compilation contracts.

Exports:
    Only symbols intentionally designated as downstream-consumable rules
    contracts are exposed here.

Boundary note:
    This package provides one canonical rules import surface and decouples
    downstream packages from parser/compiler module layout.

"""

from answer_engineering.rules.compile.compiled_rules import CompiledRules
from answer_engineering.rules.compile.compiler import FullPlanCompiler
from answer_engineering.rules.parse.errors import RulesSyntaxError
from answer_engineering.rules.parse.parser import MarkdownRulesParser

__all__ = [
    "CompiledRules",
    "FullPlanCompiler",
    "MarkdownRulesParser",
    "RulesSyntaxError",
]
