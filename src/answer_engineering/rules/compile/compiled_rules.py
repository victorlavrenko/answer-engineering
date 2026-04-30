"""Convenience entry point that parses and compiles one markdown ruleset in one.

Purpose:
    Expose the simplest public construction path from markdown text to
    executable rule plan.

Architectural role:
    Outer convenience facade for the rules boundary.

"""

from __future__ import annotations

from dataclasses import dataclass

from answer_engineering.rules.compile.compiler import (
    FullPlanCompiler,
)
from answer_engineering.rules.compile.plan import PlanIR
from answer_engineering.rules.parse.parser import (
    MarkdownRulesParser,
)


@dataclass(frozen=True, slots=True, init=False)
class CompiledRules:
    """Compiled Answer Engineering rules ready for runtime use.

    Parse authored Markdown rules and compile them into the internal plan
    consumed by rule-enabled generation. Users normally receive this object from
    subruns or pass Markdown rules to policy/runtime helpers that compile it for
    them.

    .. note::
        Keep authored rules as Markdown for review, but pass compiled rules to
        repeated generation runs when you want to avoid recompiling the same
        bundle.

    Examples:
        ```python
        rules = CompiledRules(\"\"\"
        Replace:
            Design from scratch
        With:
            Consider reusing an existing library
        \"\"\")

        result = runtime.generate(
            request,
            policy=policy,
            rules=rules,
        )
        ```

    Attributes:
        text: Authored rules text supplied by the caller.
        plan: Compiled rule plan used internally by generation.

    Runtime behavior:
        Construction parses and compiles the rule text immediately. Generation
        then treats the compiled object as an opaque rule bundle.

    Architectural role:
        Public rule-compilation boundary between authored Markdown and runtime
        orchestration.

    Consumes:
        Markdown rule text written by users, notebooks, or reproduction
        fixtures.

    Produces:
        A compiled plan consumed by rule-triggering, proposal, probing,
        selection, and patching internals.

    Invariants:
        Callers should never hold a half-compiled rules object. Parser or
        compiler failures prevent construction.

    Developer Notes:
        Keep this class as the user-facing constructor for rich rule
        compilation. Parser, compiler, and plan internals may move as boundaries
        are cleaned up, but callers should not need to import those internals
        for ordinary use.

    Todo:
        Continue splitting parser/compiler internals without expanding the
        public API unnecessarily. Improve diagnostics for syntax errors while
        preserving a simple constructor path.

    See Also:
        :class:`~answer_engineering.GenerationPolicy`
        :meth:`~answer_engineering.GenerationRuntime.generate`
        :class:`~answer_engineering.rules.MarkdownRulesParser`
        :class:`~answer_engineering.rules.FullPlanCompiler`

    """

    rules_markdown: str
    plan: PlanIR

    def __init__(self, text: str) -> None:
        """Parse and compile authored markdown into the public rules value.

        Purpose:
            Run the package-standard markdown parser and full-plan compiler in
            the order expected by runtime execution.

        Args:
            text: Authored markdown rules text supplied by notebooks, policies,
                tests, or reproduction experiment definitions.

        Architectural role:
            Construction boundary between human-authored rules and executable
            rule plans.

        Inputs (architectural provenance):
            Receives markdown rules text from the public rules surface.

        Outputs (downstream usage):
            Stores the source text and compiled ``PlanIR`` on the instance for
            runtime orchestration and reporting.

        Invariants/constraints:
            The method does not accept pre-parsed or partially compiled input.
            Parser or compiler failure prevents construction, so callers never
            hold a half-compiled rules object.

        """
        ast = MarkdownRulesParser().parse(text)
        plan = FullPlanCompiler().compile(ast)
        object.__setattr__(self, "rules_markdown", text)
        object.__setattr__(self, "plan", plan)


__all__ = ["CompiledRules"]
