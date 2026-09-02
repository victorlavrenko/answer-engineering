from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    path = ROOT / "tools" / "aaai27_human_validation.py"
    spec = importlib.util.spec_from_file_location("aaai27_human_validation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_default_design_is_calculated_not_hardcoded() -> None:
    tool = load_tool()
    design = tool.exact_design(p0=0.80, p1=0.95, alpha=0.05, power=0.80)
    assert design["n"] == 30
    assert design["critical_confirmations"] == 28
    assert 0.80 < design["achieved_power_at_p1"] < 0.82
    assert design["null_tail_at_critical"] < 0.05


def test_exact_inference_round_trip() -> None:
    tool = load_tool()
    p = tool.binomial_upper_tail(30, 28, 0.80)
    assert abs(p - 0.04417898515199706) < 1e-12
    lower = tool.clopper_pearson_lower_one_sided(28, 30, 0.05)
    assert 0.80 < lower < 0.81


def test_blinded_html_has_one_click_decision_and_valid_js(tmp_path: Path) -> None:
    tool = load_tool()
    html = tool.render_html(
        [
            {
                "task_id": "t1",
                "question": "Example question",
                "choices": {"A": "Alpha", "B": "Beta"},
                "criteria": [
                    {
                        "role": "facts",
                        "title": "Facts",
                        "rule": "PASS if a concrete fact is stated.",
                        "good": "Age 65.",
                        "bad": "Many causes exist.",
                        "response_a": "Age 65.",
                        "response_b": "General discussion.",
                    }
                ],
            }
        ],
        "audit-test",
    )
    assert "LEFT / A [1]" in html
    assert "SAME [2]" in html
    assert "RIGHT / B [3]" in html
    assert "CAN'T TELL [4]" in html
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    assert scripts
    node = shutil.which("node")
    if node:
        js = tmp_path / "validator.js"
        js.write_text("\n".join(scripts), encoding="utf-8")
        subprocess.run([node, "--check", str(js)], check=True, capture_output=True, text=True)


def test_highlight_ui_is_recorded_and_js_valid(tmp_path: Path) -> None:
    tool = load_tool()
    html = tool.render_html(
        [
            {
                "task_id": "t1",
                "question": "Patient drinks whiskey nightly.",
                "choices": {"A": "Alpha", "B": "Beta"},
                "criteria": [
                    {
                        "role": "facts",
                        "title": "Facts",
                        "rule": "PASS if concrete case facts are stated.",
                        "good": "whiskey nightly",
                        "bad": "many causes exist",
                        "response_a": "Facts: whiskey nightly.",
                        "response_b": "Facts: many causes exist.",
                        "source_coverage_a_html": '<mark class="lex" style="--mark-alpha:.5">whiskey</mark>',
                        "source_coverage_b_html": "whiskey",
                        "source_coverage_note": "Visual aid only.",
                    }
                ],
            }
        ],
        "audit-test",
    )
    assert "Case terms preserved by LEFT / A" in html
    assert "Case terms preserved by RIGHT / B" in html
    assert "mark.lex" in html
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    assert scripts
    node = shutil.which("node")
    if node:
        js = tmp_path / "validator.js"
        js.write_text("\n".join(scripts), encoding="utf-8")
        subprocess.run([node, "--check", str(js)], check=True, capture_output=True, text=True)
