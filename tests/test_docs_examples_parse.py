from __future__ import annotations

import re
from pathlib import Path

from answer_engineering.rules.compile.compiler import (
    FullPlanCompiler,
)
from answer_engineering.rules.parse.parser import (
    MarkdownRulesParser,
)

_MARKED_BLOCK = re.compile(
    r"<!-- ae-example -->\s*```ae-rules\n(.*?)\n```", re.DOTALL
)


def _examples(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return [block.strip() for block in _MARKED_BLOCK.findall(text)]


def test_marked_examples_in_docs_parse_and_compile() -> None:
    parser = MarkdownRulesParser()
    compiler = FullPlanCompiler()
    for path in ("docs/users/writing-rules.md",):
        examples = _examples(path)
        assert examples, f"no marked examples found in {path}"
        for block in examples:
            ast = parser.parse(block)
            assert ast.rules
            plan = compiler.compile(ast)
            assert plan.rules

    for block in _examples("README.md"):
        ast = parser.parse(block)
        assert ast.rules
        plan = compiler.compile(ast)
        assert plan.rules
