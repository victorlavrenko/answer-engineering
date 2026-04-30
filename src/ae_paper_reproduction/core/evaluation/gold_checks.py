"""Parse and evaluate the lightweight gold-expression language.

Purpose:
    Compile textual gold rules into a small boolean program and evaluate
    generated answers against that program using token and logical operators
    such as AND, OR, and NOT.

Architectural role:
    Core evaluation module for gold-expression parsing and execution.

Inputs (architectural provenance):
    Consumes gold strings from dataset rows and generated answer text from model
    execution.

Outputs (downstream usage):
    Compiled gold programs and boolean pass/fail judgments consumed by
    evaluation results.

Invariants/constraints:
    Parsing and evaluation must preserve the gold language semantics used by
    stored datasets so the same gold string yields the same result across runs.

"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_TOKEN_RE = re.compile(
    r"""
    \s*(
        \(|\)|
        \bAND\b|\band\b|
        \bOR\b|\bor\b|
        \bNOT\b|\bnot\b|
        "[^"]*"|
        '[^']*'|
        [^()\s]+
    )\s*
    """,
    re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class GoldExpr:
    """Typed wrapper for one authored gold-expression string.

    Purpose:
        Distinguish validated gold-check expressions from ordinary strings in
        evaluation code.

    Architectural role:
        Public value object for the small gold-expression domain-specific
        language used by reproduction checks and notebooks.

    Inputs (architectural provenance):
        Constructed from authored gold lines after parsing or validation.

    Outputs (downstream usage):
        Passed to answer-checking helpers that evaluate model output against the
        authored expression.

    Invariants/constraints:
        The wrapped text should remain the original authored expression so error
        messages and reports stay traceable.

    """

    raw: str

    def __str__(self) -> str:
        """Return the original authored gold-expression text."""
        return self.raw


class _Node:
    """Base type for private gold-expression abstract-syntax-tree nodes."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class _Term(_Node):
    """Leaf abstract-syntax-tree node for one quoted or bare search term."""

    text: str


@dataclass(frozen=True, slots=True)
class _Not(_Node):
    """Unary abstract-syntax-tree node that negates one child expression."""

    child: _Node


@dataclass(frozen=True, slots=True)
class _Bin(_Node):
    """Binary AST node representing an AND or OR combination."""

    op: str  # "and" | "or"
    left: _Node
    right: _Node


def _tokenize(expr: str) -> list[str]:
    """Tokenize one gold expression string into parser symbols."""
    out: list[str] = []
    i = 0
    while i < len(expr):
        m = _TOKEN_RE.match(expr, i)
        if not m:
            raise ValueError(
                f"Cannot tokenize gold expression at position {i}: "
                f"{expr[i : i + 20]!r}"
            )
        tok = m.group(1)
        i = m.end()
        out.append(tok)
    return out


def _unquote(s: str) -> str:
    """Strip matching single/double quote delimiters from one parsed gold."""
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


class _Parser:
    """Parse tokenized gold expressions into a small boolean AST.

    Purpose:
        Maintain parser position over the token stream and produce the syntax
        tree used by the gold evaluator.

    Architectural role:
        Private recursive-descent parser for the gold-expression language.

    Inputs (architectural provenance):
        Consumes the token list produced by `_tokenize`.

    Outputs (downstream usage):
        Internal abstract-syntax-tree nodes consumed by compilation and
        evaluation helpers.

    Invariants/constraints:
        Parser state should advance monotonically through one token stream and
        should reject malformed expressions.

    """

    def __init__(self, toks: list[str]):
        """Initialize parser state for one token stream."""
        self.toks = toks
        self.i = 0

    def _peek(self) -> str | None:
        """Return the next token without consuming it."""
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _eat(self, expected: str | None = None) -> str:
        """Consume and return the next token, optionally enforcing."""
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of gold expression")
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected!r}, got {tok!r}")
        self.i += 1
        return tok

    def parse(self) -> _Node:
        """Parse the full token stream into one root expression node."""
        node = self._parse_or()
        if self._peek() is not None:
            raise ValueError(f"Unexpected token {self._peek()!r} at end")
        return node

    def _parse_or(self) -> _Node:
        """Parse the OR-precedence layer of the gold-expression grammar."""
        node = self._parse_and()
        while True:
            tok = self._peek()
            if tok is not None and tok.lower() == "or":
                self._eat()
                node = _Bin("or", node, self._parse_and())
            else:
                return node

    def _parse_and(self) -> _Node:
        """Parse the AND-precedence layer of the gold-expression grammar."""
        node = self._parse_not()
        while True:
            tok = self._peek()
            if tok is not None and tok.lower() == "and":
                self._eat()
                node = _Bin("and", node, self._parse_not())
            else:
                return node

    def _parse_not(self) -> _Node:
        """Parse the NOT-precedence layer of the gold-expression grammar."""
        tok = self._peek()
        if tok is not None and tok.lower() == "not":
            self._eat()
            return _Not(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> _Node:
        """Parse one atomic term or parenthesized subexpression."""
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of gold expression")
        if tok == "(":
            self._eat("(")
            node = self._parse_or()
            self._eat(")")
            return node
        self._eat()
        term = _unquote(tok)
        if term == "":
            raise ValueError("Empty substring term is not allowed")
        return _Term(term)


def compile_gold(lines: Iterable[str]) -> list[GoldExpr]:
    """Compile authored gold lines into normalized `GoldExpr` values.

    Purpose:
        Validate and normalize the textual gold-expression domain-specific
        language before answers are evaluated.

    Architectural role:
        Evaluation boundary between authored benchmark metadata and executable
        answer checks.

    Inputs (architectural provenance):
        Receives raw gold-expression strings from cases, fixtures, or notebook
        inputs.

    Outputs (downstream usage):
        Returns typed expressions consumed by `check_gold` and report
        generation.

    Invariants/constraints:
        Invalid expressions should fail during compilation rather than during
        later answer evaluation.

    """
    exprs: list[GoldExpr] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("-"):
            s = s[1:].strip()
        if not s:
            continue
        exprs.append(GoldExpr(s))
    return exprs


def _eval(node: _Node, text: str) -> bool:
    """Evaluate one compiled gold-expression node against normalized answer."""
    if isinstance(node, _Term):
        return node.text.lower() in text
    if isinstance(node, _Not):
        return not _eval(node.child, text)
    if isinstance(node, _Bin):
        if node.op == "and":
            return _eval(node.left, text) and _eval(node.right, text)
        if node.op == "or":
            return _eval(node.left, text) or _eval(node.right, text)
    raise TypeError(f"Unexpected node type: {type(node)}")


def check_gold(
    answer_text: str, gold: str
) -> tuple[bool, list[tuple[str, bool]]]:
    """Evaluate answer text against one authored gold expression string.

    Purpose:
        Decide whether a generated answer satisfies the benchmark's
        gold-expression criteria.

    Architectural role:
        Evaluation helper at the boundary between model output text and
        structured correctness checks.

    Inputs (architectural provenance):
        Receives answer text from runtime output and an authored expression from
        the benchmark case definition.

    Outputs (downstream usage):
        Returns the boolean correctness signal consumed by aggregation and
        reporting.

    Invariants/constraints:
        Matching semantics should remain consistent with `compile_gold` so
        notebook and batch evaluations produce the same result.

    """
    text = answer_text.lower()
    exprs = compile_gold(gold.splitlines())
    results: list[tuple[str, bool]] = []
    overall = True
    for expr in exprs:
        toks = _tokenize(expr.raw)
        ast = _Parser(toks).parse()
        ok = _eval(ast, text)
        results.append((expr.raw, ok))
        overall = overall and ok
    return overall, results
