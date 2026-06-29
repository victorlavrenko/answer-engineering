from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import answer_engineering
from ae_paper_reproduction.api import (
    Dataset,
    NotebookSubruns,
    Subrun,
    Summary,
)
from answer_engineering import (
    CompiledRules,
    GenerationPolicy,
    GenerationRequest,
    GenerationRuntime,
)
from answer_engineering.rules.parse.parser import MarkdownRulesParser


def test_top_level_version_matches_installed_metadata() -> None:
    try:
        expected_version = version("answer-engineering")
    except PackageNotFoundError:
        expected_version = "0+unknown"

    assert answer_engineering.__version__ == expected_version


def test_public_api_imports_and_minimal_calls() -> None:
    assert answer_engineering.__version__

    rules = MarkdownRulesParser().parse(
        "## Replace (once): x\n\nWith:\n\n* y\n"
    )
    compiled = CompiledRules("## Replace (once): x\n\nWith:\n\n* y\n")
    assert len(rules.rules) == 1
    assert compiled.rules_markdown.startswith("## Replace")

    assert Dataset is not None
    assert GenerationRuntime is not None
    assert Summary is not None
    assert NotebookSubruns is not None
    assert Subrun is not None


def test_top_level_public_api_has_canonical_runtime_surface() -> None:
    exported = set(answer_engineering.__all__)

    assert "GenerationRuntime" in exported
    assert "GenerationRequest" in exported
    assert "GenerationPolicy" in exported
    assert "CompiledRules" in exported
    assert "RuntimeEngine" not in exported

    request = GenerationRequest(question="What is SSNHL?")
    policy = GenerationPolicy(
        rules=CompiledRules("## Replace (once): x\n\nWith:\n\n* y\n")
    )

    assert request.question == "What is SSNHL?"
    assert policy.rules is not None
