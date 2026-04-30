"""Exception types for the markdown-rules parser.

Purpose:
    Define structured parse-time failures that can be surfaced back to rule
    authors with source-location context.

"""

from __future__ import annotations


class RulesSyntaxError(Exception):
    """Structured parse failure for invalid markdown rule syntax.

    Purpose:
        Carry a human-readable error plus line, column, and snippet context so
        callers can map parser failures back to the authored rule text.

    """

    def __init__(
        self, message: str, *, line: int, column: int, snippet: str
    ) -> None:
        """Store parse context and compose the final syntax-error message.

        Purpose:
            Preserve the user-facing rule-source context needed to explain why
            markdown parsing failed.

        Architectural role:
            Error-construction boundary between low-level parser failures and
            authored ruleset diagnostics shown to notebooks or command-line
            callers.

        Inputs (architectural provenance):
            Receives the base message plus optional source line, line number,
            column, and parser detail collected while parsing a rules document.

        Outputs (downstream usage):
            Initializes the exception with a formatted message and stores
            structured context fields for tests and diagnostic rendering.

        Invariants/constraints:
            The formatted message should remain readable without requiring
            callers to inspect parser internals.

        """
        self.line = line
        self.column = column
        self.snippet = snippet
        super().__init__(
            f"{message} at line {line}, column {column}: {snippet}"
        )
