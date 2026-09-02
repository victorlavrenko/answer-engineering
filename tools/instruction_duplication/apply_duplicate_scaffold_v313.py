#!/usr/bin/env python3
"""Upgrade a 3.0.12 checkout with duplicate-scaffold trajectory recovery."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SCORE_OLD = "    body = _local_semantic_body(document, semantic_matches, match)\n    body_tokens = len(normalize_text(body).split())\n"
SCORE_NEW = "    body = _strip_guidance_echo(\n        tag,\n        _local_semantic_body(document, semantic_matches, match),\n    )\n    body_tokens = len(normalize_text(body).split())\n"
SEGMENT_OLD = 'def _selected_segment_end(\n    document: str,\n    selected: Mapping[str, re.Match[str]],\n    start: re.Match[str],\n) -> int:\n    """End a recovered section at the next boundary in the selected trajectory."""\n    later = [match.start() for match in selected.values() if match.start() > start.start()]\n    return min(later) if later else len(document)\n'
SEGMENT_NEW = 'def _selected_segment_end(\n    document: str,\n    selected: Mapping[str, re.Match[str]],\n    semantic_matches: Mapping[str, tuple[re.Match[str], ...]],\n    tag: str,\n    start: re.Match[str],\n) -> int:\n    """End one selected role without swallowing an abandoned duplicate scaffold.\n\n    Usually the next boundary is the next marker selected for the recovered\n    trajectory. A special case occurs when an output prints a complete empty\n    template and then fills only a suffix of it. The optimal trajectory can then\n    use an early prefix (for example Facts/Implications) and a later filled suffix\n    (Provisional answer through Final answer). In that case, ending the prefix at\n    the *later* selected marker would incorrectly absorb the intervening empty\n    template and bridge prose.\n\n    Therefore, if the immediate next role has an earlier duplicate marker between\n    this role and the selected next marker, treat that earlier marker as a boundary\n    only when it is at least as strong a presentation marker as the selected one.\n    This preserves protection against weak body-internal role phrases while making\n    genuine numbered/Markdown/XML duplicate templates segment correctly.\n    """\n    later = [match.start() for match in selected.values() if match.start() > start.start()]\n    end = min(later) if later else len(document)\n\n    try:\n        index = CONTENT_TAGS.index(tag)\n    except ValueError as exc:\n        raise AssertionError(f"unknown protocol role: {tag}") from exc\n    if index + 1 >= len(CONTENT_TAGS):\n        return end\n\n    next_tag = CONTENT_TAGS[index + 1]\n    next_selected = selected.get(next_tag)\n    if next_selected is None or next_selected.start() <= start.start():\n        return end\n\n    selected_strength = _semantic_marker_strength(document, next_selected)\n    intervening = [\n        candidate.start()\n        for candidate in semantic_matches[next_tag]\n        if (\n            start.start() < candidate.start() < next_selected.start()\n            and _semantic_marker_strength(document, candidate) >= selected_strength\n        )\n    ]\n    if intervening:\n        end = min(end, min(intervening))\n    return end\n'
CALL_OLD = "            _selected_segment_end(document, semantic_first, start)\n            if coherent is not None\n"
CALL_NEW = "            _selected_segment_end(\n                document,\n                semantic_first,\n                semantic_matches,\n                tag,\n                start,\n            )\n            if coherent is not None\n"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_trajectory(text: str) -> str:
    if "without swallowing an abandoned duplicate scaffold" in text:
        return text
    if "STEP6_NUMBERED_HEADING_FALLBACK_RE" not in text:
        raise RuntimeError("trajectory.py does not contain the 3.0.12 Step-6 heading refinement")
    text = replace_once(text, SCORE_OLD, SCORE_NEW, "guidance-clean candidate scoring")
    text = replace_once(text, SEGMENT_OLD, SEGMENT_NEW, "duplicate-scaffold segment boundary")
    text = replace_once(text, CALL_OLD, CALL_NEW, "recover_protocol selected-segment call")
    return text


def patch_version(text: str) -> str:
    if '__version__ = "3.0.13"' in text:
        return text
    if '__version__ = "3.0.12"' not in text:
        raise RuntimeError("expected package version 3.0.12")
    return text.replace('__version__ = "3.0.12"', '__version__ = "3.0.13"', 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()

    trajectory = root / "src/instruction_duplication/trajectory.py"
    init = root / "src/instruction_duplication/__init__.py"
    if not trajectory.is_file() or not init.is_file():
        raise SystemExit("run this from the instruction-duplication repository root")

    trajectory_text = trajectory.read_text(encoding="utf-8")
    init_text = init.read_text(encoding="utf-8")

    if (
        '__version__ = "3.0.13"' in init_text
        and "without swallowing an abandoned duplicate scaffold" in trajectory_text
    ):
        print("3.0.13 duplicate-scaffold recovery is already applied")
        return 0

    if '__version__ = "3.0.12"' not in init_text:
        raise SystemExit(
            "this overlay expects the current 3.0.12 checkout; do not apply it to an older version"
        )

    new_trajectory = patch_trajectory(trajectory_text)
    new_init = patch_version(init_text)

    trajectory_backup = trajectory.with_suffix(".py.v312.bak")
    init_backup = init.with_suffix(".py.v312.bak")
    if not trajectory_backup.exists():
        shutil.copy2(trajectory, trajectory_backup)
    if not init_backup.exists():
        shutil.copy2(init, init_backup)

    trajectory.write_text(new_trajectory, encoding="utf-8")
    init.write_text(new_init, encoding="utf-8")

    pkg = root / "PKG-INFO"
    if pkg.is_file():
        pkg_text = pkg.read_text(encoding="utf-8")
        if "Version: 3.0.12" in pkg_text:
            backup = pkg.with_suffix(".v312.bak")
            if not backup.exists():
                shutil.copy2(pkg, backup)
            pkg.write_text(
                pkg_text.replace("Version: 3.0.12", "Version: 3.0.13", 1),
                encoding="utf-8",
            )

    print("applied 3.0.13 duplicate-scaffold trajectory recovery")
    print("prompts/generation and role-content criteria are unchanged")
    print("existing generations need deterministic rejudging/reanalysis only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
