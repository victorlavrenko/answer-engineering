from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ae_paper_reproduction.telemetry.paper_metrics import (
    NA_TEX,
    render_paper_metrics_tex,
)
from ae_paper_reproduction.telemetry.telemetry_types import SubrunTelemetry


def _fake_row(*, case_id: str, ok: bool, runtime_sec: float) -> object:
    return SimpleNamespace(
        id=case_id,
        edited_ok=ok,
        edited_runtime_sec=runtime_sec,
        baseline_runtime_sec=runtime_sec,
        ae_telemetry=None,
    )


def _fake_subrun(
    *,
    mode: str,
    scope: str,
    paper_role: str = "primary",
    paper_variant: str | None = None,
    rows: list[object],
) -> SubrunTelemetry:
    variant = paper_variant or mode
    return cast(
        SubrunTelemetry,
        SimpleNamespace(
            mode=mode,
            ruleset_name=variant,
            paper_role=paper_role,
            paper_variant=variant,
            case_type_filter=scope,
            answer_rows=tuple(rows),
        ),
    )


def _extract_macro(tex: str, macro: str) -> str:
    match = re.search(rf"\\newcommand\{{\\{macro}\}}\{{([^}}]+)\}}", tex)
    assert match is not None, macro
    return match.group(1)


def test_three_mode_fixture_emits_complete_canonical_grid() -> None:
    summaries = [
        _fake_subrun(
            mode="baseline",
            scope="orl-ssnhl-acute",
            rows=[
                _fake_row(case_id="s1", ok=False, runtime_sec=2.0),
                _fake_row(case_id="s2", ok=True, runtime_sec=2.0),
            ],
        ),
        _fake_subrun(
            mode="baseline",
            scope="orl-conductive-acute",
            rows=[
                _fake_row(case_id="c1", ok=False, runtime_sec=2.0),
                _fake_row(case_id="c2", ok=False, runtime_sec=2.0),
            ],
        ),
        _fake_subrun(
            mode="reasoning",
            scope="orl-ssnhl-acute",
            rows=[
                _fake_row(case_id="s1", ok=True, runtime_sec=4.0),
                _fake_row(case_id="s2", ok=False, runtime_sec=4.0),
            ],
        ),
        _fake_subrun(
            mode="reasoning",
            scope="orl-conductive-acute",
            rows=[
                _fake_row(case_id="c1", ok=True, runtime_sec=4.0),
                _fake_row(case_id="c2", ok=False, runtime_sec=4.0),
            ],
        ),
        _fake_subrun(
            mode="trajectory",
            scope="orl-ssnhl-acute",
            rows=[
                _fake_row(case_id="s1", ok=True, runtime_sec=3.0),
                _fake_row(case_id="s2", ok=True, runtime_sec=3.0),
            ],
        ),
        _fake_subrun(
            mode="trajectory",
            scope="orl-conductive-acute",
            rows=[
                _fake_row(case_id="c1", ok=False, runtime_sec=3.0),
                _fake_row(case_id="c2", ok=True, runtime_sec=3.0),
            ],
        ),
    ]

    tex = render_paper_metrics_tex(subrun_telemetry=summaries)

    assert _extract_macro(tex, "SSNHLBaselineAcceptedPct") == "50.0\\%"
    assert _extract_macro(tex, "SSNHLBaselineAcceptedRaw") == "50.0"
    assert _extract_macro(tex, "SSNHLBaselineSlowdownX") == "0.50$\\times$"
    assert _extract_macro(tex, "ConductiveBaselineSlowdownX") == "0.50$\\times$"
    assert _extract_macro(tex, "SSNHLReasoningSlowdownX") == "1.00$\\times$"
    assert (
        _extract_macro(tex, "ConductiveReasoningSlowdownX") == "1.00$\\times$"
    )
    assert (
        _extract_macro(tex, "CombinedTrajectoryBalancedAccuracyPct")
        == "75.0\\%"
    )
    assert _extract_macro(tex, "SSNHLBaselineDeltaPP") == "+0.0 pp"
    assert _extract_macro(tex, "ConductiveBaselineDeltaPP") == "-50.0 pp"

    assert _extract_macro(tex, "ConductiveReplaceAfterAcceptedPct") == NA_TEX
    assert _extract_macro(tex, "ConductiveReplaceAfterSlowdownX") == NA_TEX
    assert (
        _extract_macro(tex, "CombinedReplaceAfterBalancedAccuracyPct") == NA_TEX
    )

    legacy_macros = (
        "SSNHLBaselineAccepted",
        "BaselineSSNHLAccuracy",
        "BaselineConductiveAccuracy",
        "BaselineBalancedAccuracy",
        "ReasoningBalancedAccuracy",
        "TrajectoryBalancedAccuracy",
        "BaselineSlowdown",
        "ReasoningSlowdown",
        "TrajectorySlowdown",
        "ConductiveTrajectoryImproved",
    )

    for macro in legacy_macros:
        assert f"\\newcommand{{\\{macro}}}" not in tex


def test_missing_required_modes_fail_loudly() -> None:
    with pytest.raises(ValueError, match="mode='reasoning'"):
        render_paper_metrics_tex(
            subrun_telemetry=(
                _fake_subrun(
                    mode="baseline",
                    scope="orl-ssnhl-acute",
                    rows=[_fake_row(case_id="s1", ok=True, runtime_sec=2.0)],
                ),
                _fake_subrun(
                    mode="baseline",
                    scope="orl-conductive-acute",
                    rows=[_fake_row(case_id="c1", ok=True, runtime_sec=2.0)],
                ),
                _fake_subrun(
                    mode="trajectory",
                    scope="orl-ssnhl-acute",
                    paper_variant="trajectory",
                    rows=[_fake_row(case_id="s1", ok=True, runtime_sec=3.0)],
                ),
                _fake_subrun(
                    mode="trajectory",
                    scope="orl-conductive-acute",
                    paper_variant="trajectory",
                    rows=[_fake_row(case_id="c1", ok=True, runtime_sec=3.0)],
                ),
            ),
        )


def test_missing_baseline_mode_fails_loudly() -> None:
    with pytest.raises(ValueError, match="mode='baseline'"):
        render_paper_metrics_tex(
            subrun_telemetry=(
                _fake_subrun(
                    mode="reasoning",
                    scope="orl-ssnhl-acute",
                    rows=[_fake_row(case_id="s1", ok=True, runtime_sec=4.0)],
                ),
                _fake_subrun(
                    mode="reasoning",
                    scope="orl-conductive-acute",
                    rows=[_fake_row(case_id="c1", ok=True, runtime_sec=4.0)],
                ),
                _fake_subrun(
                    mode="trajectory",
                    scope="orl-ssnhl-acute",
                    paper_variant="trajectory",
                    rows=[_fake_row(case_id="s1", ok=True, runtime_sec=3.0)],
                ),
                _fake_subrun(
                    mode="trajectory",
                    scope="orl-conductive-acute",
                    paper_variant="trajectory",
                    rows=[_fake_row(case_id="c1", ok=True, runtime_sec=3.0)],
                ),
            ),
        )


def test_missing_trajectory_mode_fails_loudly() -> None:
    with pytest.raises(ValueError, match="mode='trajectory'"):
        render_paper_metrics_tex(
            subrun_telemetry=(
                _fake_subrun(
                    mode="baseline",
                    scope="orl-ssnhl-acute",
                    rows=[_fake_row(case_id="s1", ok=True, runtime_sec=2.0)],
                ),
                _fake_subrun(
                    mode="baseline",
                    scope="orl-conductive-acute",
                    rows=[_fake_row(case_id="c1", ok=True, runtime_sec=2.0)],
                ),
                _fake_subrun(
                    mode="reasoning",
                    scope="orl-ssnhl-acute",
                    rows=[_fake_row(case_id="s1", ok=True, runtime_sec=4.0)],
                ),
                _fake_subrun(
                    mode="reasoning",
                    scope="orl-conductive-acute",
                    rows=[_fake_row(case_id="c1", ok=True, runtime_sec=4.0)],
                ),
            ),
        )


def test_generated_tex_golden_contains_expected_macros_and_main_refs() -> None:
    generated_tex = Path("docs/paper/generated/paper-metrics.tex").read_text(
        encoding="utf-8"
    )
    for expected in (
        "\\SSNHLReplaceAfterAcceptedPct",
        "\\ConductiveReplaceAfterAcceptedPct",
        "\\CombinedReplaceAfterBalancedAccuracyPct",
    ):
        assert expected in generated_tex

    defined = set(
        re.findall(
            r"\\\\newcommand\{\\\\([A-Za-z][A-Za-z0-9]*)\}",
            generated_tex,
        )
    )
    main_tex = Path("docs/paper/main.tex").read_text(encoding="utf-8")
    assert "& --" not in main_tex
    assert "-- \\\\" not in main_tex

    used = set(re.findall(r"\\\\([A-Za-z][A-Za-z0-9]*)\{\}", main_tex))
    ignored = {
        "PaperEvalN",
        "ldots",
        "paragraph",
        "cmark",
        "xmark",
        "fancyhf",
        "partialmark",
    }
    missing = sorted(
        name for name in used if name not in defined and name not in ignored
    )
    assert missing == []


def test_duplicate_primary_variant_per_scope_fails_loudly() -> None:
    with pytest.raises(ValueError, match="Duplicate paper metadata key"):
        render_paper_metrics_tex(
            subrun_telemetry=(
                _fake_subrun(
                    mode="baseline",
                    scope="orl-ssnhl-acute",
                    paper_role="primary",
                    paper_variant="baseline",
                    rows=[_fake_row(case_id="s1", ok=True, runtime_sec=1.0)],
                ),
                _fake_subrun(
                    mode="baseline",
                    scope="orl-ssnhl-acute",
                    paper_role="primary",
                    paper_variant="baseline",
                    rows=[_fake_row(case_id="s1", ok=True, runtime_sec=1.0)],
                ),
            )
        )
