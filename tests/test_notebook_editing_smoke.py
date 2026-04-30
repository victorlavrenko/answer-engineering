from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from ae_paper_reproduction.core.planning.notebook_extractor import (
    NotebookRulesetSpec,
    extract_answer_engineering_rulesets_from_ipynb,
    extract_answer_engineering_subruns_from_ipynb,
)
from ae_paper_reproduction.core.planning.subruns import (
    SubrunDefinition,
)
from answer_engineering.inference.decode.session_orchestration import (
    ExecutionSession,
)
from answer_engineering.rules.compile.compiled_rules import (
    CompiledRules,
)
from tests._support.core_helpers import create_step_snapshot
from tests._support.runtime_harness import configure_runtime_scoring
from tests.core._scoring_stubs import GenerationRuntimeStub


def _first_rules_markdown(ipynb_path: Path) -> str:
    for ruleset in extract_answer_engineering_rulesets_from_ipynb(ipynb_path):
        if ruleset.rules_markdown.lstrip().startswith("##"):
            return ruleset.rules_markdown
    raise ValueError("Could not find rule markdown in notebook rulesets")


def test_notebook_extraction_skips_intro_and_rules_edit_applies(
    tmp_path: Path,
) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Answer Engineering Rules\n",
                    "This is a notebook intro and not the rules section.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "source": [
                    "# Answer Engineering Rules\n",
                    "## Replace (once): sensorineural hearing loss\n",
                    "\n",
                    "With:\n",
                    "\n",
                    "- SSNHL\n",
                ],
            },
        ]
    }
    ipynb_path = tmp_path / "demo.ipynb"
    ipynb_path.write_text(json.dumps(notebook), encoding="utf-8")

    rules_md = _first_rules_markdown(ipynb_path)

    engine = ExecutionSession(plan=CompiledRules(rules_md).plan)
    configure_runtime_scoring(
        engine,
        generation_runtime=GenerationRuntimeStub.loaded_runtime(),
        require_model_scoring=True,
    )

    decision = engine.execute_step(
        create_step_snapshot(
            snapshot_text="Findings support sensorineural hearing loss.",
            token_index=0,
        )
    )

    assert "Replace (once): sensorineural hearing loss" in rules_md
    assert decision.changed, (
        "expected a rule edit to apply from notebook-extracted markdown"
    )
    assert "SSNHL" in decision.final_text


def test_notebook_extraction_ignores_visual_separators_inside_rules(
    tmp_path: Path,
) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Answer Engineering Rules\n",
                    "## Replace: sensorineural hearing loss\n",
                    "\n",
                    "With:\n",
                    "\n",
                    "- sudden sensorineural hearing loss\n",
                    "\n",
                    "---\n",
                    "\n",
                    "## After: sudden sensorineural hearing loss\n",
                    "\n",
                    "Add:\n",
                    "\n",
                    "- This condition requires urgent treatment.\n",
                ],
            }
        ]
    }
    ipynb_path = tmp_path / "demo.ipynb"
    ipynb_path.write_text(json.dumps(notebook), encoding="utf-8")

    rulesets = extract_answer_engineering_rulesets_from_ipynb(ipynb_path)

    assert rulesets[0].rules_markdown == (
        "## Replace: sensorineural hearing loss\n\n"
        "With:\n\n"
        "- sudden sensorineural hearing loss\n\n"
        "## After: sudden sensorineural hearing loss\n\n"
        "Add:\n\n"
        "- This condition requires urgent treatment.\n"
    )
    assert rulesets[0].system_prompt is None


def test_notebook_extraction_reads_system_prompt_before_rules(
    tmp_path: Path,
) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Answer Engineering Rules\n",
                    "\n",
                    "## Run: full\n",
                    "\n",
                    "## System Prompt\n",
                    "\n",
                    "You are a terse clinical assistant.\n",
                    "Do not add markdown tables.\n",
                    "\n",
                    "## Replace: hearing loss\n",
                    "\n",
                    "With:\n",
                    "\n",
                    "- HL\n",
                ],
            }
        ]
    }
    ipynb_path = tmp_path / "demo.ipynb"
    ipynb_path.write_text(json.dumps(notebook), encoding="utf-8")

    [ruleset] = extract_answer_engineering_rulesets_from_ipynb(ipynb_path)
    assert ruleset.system_prompt == (
        "You are a terse clinical assistant.\nDo not add markdown tables.\n"
    )
    assert (
        ruleset.rules_markdown == "## Replace: hearing loss\n\nWith:\n\n- HL\n"
    )


def test_notebook_extraction_parses_multiple_rulesets_and_subruns(
    tmp_path: Path,
) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Answer Engineering Rules\n",
                    "\n",
                    "## Run: baseline\n",
                    "\n",
                    "- orl-ssnhl-acute\n",
                    "- orl-conductive-acute\n",
                ],
            },
            {
                "cell_type": "markdown",
                "source": [
                    "# Answer Engineering Rules\n",
                    "\n",
                    "## Run: full\n",
                    "\n",
                    "- orl-ssnhl-acute\n",
                    "- orl-conductive-acute\n",
                    "\n",
                    "## Replace: sensorineural hearing loss\n",
                    "\n",
                    "With:\n",
                    "\n",
                    "- sudden sensorineural hearing loss\n",
                ],
            },
        ]
    }
    ipynb_path = tmp_path / "demo.ipynb"
    ipynb_path.write_text(json.dumps(notebook), encoding="utf-8")

    rulesets = extract_answer_engineering_rulesets_from_ipynb(ipynb_path)
    assert [ruleset.ruleset_name for ruleset in rulesets] == [
        "baseline",
        "full",
    ]
    assert rulesets[0].case_types == ("orl-ssnhl-acute", "orl-conductive-acute")
    assert rulesets[0].rules_markdown == ""
    assert "Replace: sensorineural hearing loss" in rulesets[1].rules_markdown

    subruns = [
        (ruleset, case_type)
        for ruleset in rulesets
        for case_type in (ruleset.case_types or (None,))
    ]
    assert [
        f"{ruleset.ruleset_name}-{case_type or 'all'}"
        for ruleset, case_type in subruns
    ] == [
        "baseline-orl-ssnhl-acute",
        "baseline-orl-conductive-acute",
        "full-orl-ssnhl-acute",
        "full-orl-conductive-acute",
    ]


def test_notebook_extraction_defaults_missing_run_name_and_case_type_to_all(
    tmp_path: Path,
) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Answer Engineering Rules\n",
                    "\n",
                    "## Run:\n",
                    "\n",
                    "## Replace: hearing loss\n",
                    "\n",
                    "With:\n",
                    "\n",
                    "- HL\n",
                ],
            }
        ]
    }
    ipynb_path = tmp_path / "demo.ipynb"
    ipynb_path.write_text(json.dumps(notebook), encoding="utf-8")

    rulesets = extract_answer_engineering_rulesets_from_ipynb(ipynb_path)
    assert rulesets[0].ruleset_name == "notebook-cell-0"
    assert rulesets[0].case_types == ()

    subruns = extract_answer_engineering_subruns_from_ipynb(ipynb_path)
    assert len(subruns) == 1
    assert subruns[0][1] is None
    assert (
        f"{subruns[0][0].ruleset_name}-{subruns[0][1] or 'all'}"
        == "notebook-cell-0-all"
    )


def test_notebook_extraction_prefers_colab_runtime_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Answer Engineering Rules\n",
                    "## Replace (once): cat\n",
                    "\n",
                    "With:\n",
                    "\n",
                    "- dog\n",
                ],
            }
        ]
    }

    class _FakeColabMessage:
        @staticmethod
        def blocking_request(
            _name: str, request: str = "", timeout_sec: int = 5
        ) -> dict[str, object]:
            assert request == ""
            assert timeout_sec == 5
            return {"ipynb": runtime_notebook}

    google_module = types.ModuleType("google")
    colab_module = types.ModuleType("google.colab")
    colab_module._message = _FakeColabMessage  # pyright: ignore[reportAttributeAccessIssue]
    google_module.colab = colab_module  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)

    file_notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Answer Engineering Rules\n",
                    "This notebook should not be selected.\n",
                ],
            }
        ]
    }
    ipynb_path = tmp_path / "demo.ipynb"
    ipynb_path.write_text(json.dumps(file_notebook), encoding="utf-8")

    rules_md = _first_rules_markdown(ipynb_path)

    assert "Replace (once): cat" in rules_md


def test_notebook_extraction_falls_back_to_file_when_runtime_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Answer Engineering Rules\n",
                    "## Replace (once): hearing loss\n",
                    "\n",
                    "With:\n",
                    "\n",
                    "- HL\n",
                ],
            }
        ]
    }
    ipynb_path = tmp_path / "demo.ipynb"
    ipynb_path.write_text(json.dumps(notebook), encoding="utf-8")

    rules_md = _first_rules_markdown(ipynb_path)

    assert "Replace (once): hearing loss" in rules_md


def test_subrun_definition_preserves_explicit_empty_system_prompt() -> None:
    ruleset = NotebookRulesetSpec(
        (
            "# Answer Engineering Rules\n\n"
            "## Run: empty\n\n"
            "- orl\n\n"
            "## System Prompt\n\n"
            "---\n"
        ),
        cell_index=0,
        source_hint="demo.ipynb",
    )

    definition = SubrunDefinition(
        ruleset=ruleset,
        case_type="orl",
        index=0,
        notebook_path="demo.ipynb",
        mode="reasoning",
    )

    assert ruleset.system_prompt == ""
    assert definition.system_prompt == ""


def test_notebook_extraction_preserves_explicit_empty_system_prompt(
    tmp_path: Path,
) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Answer Engineering Rules\n",
                    "\n",
                    "## Run: empty\n",
                    "\n",
                    "- orl-ssnhl-acute\n",
                    "- orl-conductive-acute\n",
                    "\n",
                    "## System Prompt\n",
                    "\n",
                    "---\n",
                ],
            }
        ]
    }
    ipynb_path = tmp_path / "demo.ipynb"
    ipynb_path.write_text(json.dumps(notebook), encoding="utf-8")

    [ruleset] = extract_answer_engineering_rulesets_from_ipynb(ipynb_path)

    assert ruleset.system_prompt == ""
    assert ruleset.rules_markdown == ""


def test_notebook_ruleset_accepts_appendix_and_exploratory_paper_roles() -> (
    None
):
    appendix = NotebookRulesetSpec(
        (
            "# Answer Engineering Rules\n"
            "## Run: appendix-baseline\n"
            "## Mode: baseline\n"
            "## Paper Role: appendix\n"
            "## Variant: appendix-baseline\n"
            "- orl\n"
        ),
        cell_index=0,
        source_hint="demo.ipynb",
    )
    exploratory = NotebookRulesetSpec(
        (
            "# Answer Engineering Rules\n"
            "## Run: exploratory-trajectory\n"
            "## Mode: trajectory\n"
            "## Paper Role: exploratory\n"
            "## Variant: exploratory-trajectory\n"
            "- orl\n"
        ),
        cell_index=1,
        source_hint="demo.ipynb",
    )

    assert appendix.paper_role == "appendix"
    assert appendix.paper_variant == "appendix-baseline"
    assert exploratory.paper_role == "exploratory"
    assert exploratory.paper_variant == "exploratory-trajectory"
