#!/usr/bin/env python3
# Apply the 3.0.12 Step-6 heading-recognition refinement.
# Presentation-layer only: the non-trivial-content threshold, semantic
# counterfactual diagnostic, prompts, and generation code are unchanged.

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

OLD_ENTRY = """    "answer_changing_change": (
        r"answer[ _-]+changing[ _-]+change|what[ _-]+would[ _-]+change[ _-]+the[ _-]+answer|"
        r"answer[ _-]+change|counterfactual[ _-]+change"
    ),"""

NEW_ENTRY = r"""    "answer_changing_change": (
        # Conservative unnumbered synonyms. These all explicitly describe an
        # answer-changing/counterfactual role; vague "what would change?" is handled
        # only by the numbered Step-6 fallback below.
        r"answer[ _-]+changing[ _-]+(?:change|scenario)|"
        r"answer[ _-]+change|"
        r"counterfactual(?:[ _-]+(?:change|scenario|test))?|"
        r"what[ _-]+would[ _-]+change[ _-]+(?:the[ _-]+)?(?:answer|question)"
        r"(?:[ _-]+to[ _-]+(?:(?:(?:option|choice)[ _-]+)?[A-Z]"
        r"(?:[ _-]*\([^\)\r\n]{1,60}\))?|[^()\r\n:]{1,60}[ _-]*\([A-Z]\)))?|"
        r"what[ _-]+(?:would[ _-]+need[ _-]+to|needs?[ _-]+to)[ _-]+change|"
        r"what[ _-]+change[ _-]+would[ _-]+(?:change|alter|switch)[ _-]+"
        r"(?:the[ _-]+)?answer(?:[ _-]+to[ _-]+(?:(?:option|choice)[ _-]+)?[A-Z])?|"
        r"how[ _-]+(?:could|would)[ _-]+(?:the[ _-]+)?answer[ _-]+change|"
        r"what[ _-]+would[ _-]+make[ _-]+(?:the[ _-]+)?"
        r"(?:best[ _-]+alternative|second[ _-]+best(?:[ _-]+answer)?|alternative|"
        r"(?:(?:option|choice)[ _-]+)?[A-Z])"
        r"[ _-]+(?:the[ _-]+)?(?:best[ _-]+answer|best|correct|win)|"
        r"how[ _-]+to[ _-]+make[ _-]+(?:the[ _-]+)?"
        r"(?:best[ _-]+alternative|second[ _-]+best(?:[ _-]+answer)?|alternative|"
        r"(?:(?:option|choice)[ _-]+)?[A-Z])[ _-]+(?:best|correct|win)|"
        r"(?:smallest[ _-]+)?change[ _-]+(?:needed|required)[ _-]+"
        r"(?:to[ _-]+(?:change|alter|switch)[ _-]+(?:the[ _-]+)?answer|"
        r"for[ _-]+(?:the[ _-]+)?"
        r"(?:best[ _-]+alternative|second[ _-]+best(?:[ _-]+answer)?|alternative|"
        r"(?:(?:option|choice)[ _-]+)?[A-Z])"
        r"[ _-]+to[ _-]+(?:win|be[ _-]+correct|become[ _-]+best))|"
        r"change[ _-]+that[ _-]+would[ _-]+make[ _-]+(?:the[ _-]+)?"
        r"(?:best[ _-]+alternative|second[ _-]+best(?:[ _-]+answer)?|alternative|"
        r"(?:(?:option|choice)[ _-]+)?[A-Z])[ _-]+(?:win|correct|best)|"
        r"conditions?[ _-]+for[ _-]+(?:the[ _-]+)?"
        r"(?:best[ _-]+alternative|second[ _-]+best(?:[ _-]+answer)?|alternative|"
        r"(?:(?:option|choice)[ _-]+)?[A-Z])"
        r"[ _-]+to[ _-]+(?:win|be[ _-]+correct|become[ _-]+best)|"
        r"when[ _-]+would[ _-]+(?:the[ _-]+)?"
        r"(?:best[ _-]+alternative|second[ _-]+best(?:[ _-]+answer)?|alternative|"
        r"(?:(?:option|choice)[ _-]+)?[A-Z])"
        r"[ _-]+(?:win|be[ _-]+correct|become[ _-]+best)"
    ),"""

PATTERN_BLOCK_ANCHOR = """SEMANTIC_HEADING_PATTERNS = {
    tag: re.compile(
        rf"(?im)^[ \\t]*(?:\\#{{1,6}}[ \\t]*)?(?:[-+*][ \\t]+)?"
        rf"(?:\\*\\*|__)?"
        rf"(?:(?:step[ \\t]*)?{index}[.:)\\-][ \\t]*)?"
        rf"(?:\\*\\*|__)?(?:{_SEMANTIC_LABELS[tag]})(?:\\?)?(?:\\*\\*|__)?[ \\t]*"
        rf"(?:[:\\u2014\\u2013-][ \\t]*|(?=\\r?$))"
    )
    for index, tag in enumerate(CONTENT_TAGS, start=1)
}
"""

FALLBACK_BLOCK = r"""
# Some models preserve the requested role number while paraphrasing its title.
# Because explicit "6." / "Step 6" is itself a strong boundary signal, this
# fallback may safely accept a somewhat wider family than the unnumbered aliases.
# Body prose without an explicit Step-6 marker is not recovered by this fallback.
STEP6_NUMBERED_HEADING_FALLBACK_RE = re.compile(
    r"(?im)^[ \t]*(?:\#{1,6}[ \t]*)?(?:[-+*][ \t]+)?"
    r"(?:\*\*|__)?(?:step[ \t]*)?6[.:)\-][ \t]*(?:\*\*|__)?"
    r"(?:"
    r"what[ _-]+would[ _-]+change(?:[ _-]+(?:the[ _-]+)?(?:answer|question))?"
    r"(?:[ _-]+to[ _-]+[^:\r\n\u2014\u2013-]{1,80})?|"
    r"what[ _-]+(?:would[ _-]+need[ _-]+to|needs?[ _-]+to)[ _-]+change|"
    r"what[ _-]+would[ _-]+make[ _-]+[^:\r\n\u2014\u2013-]{1,80}|"
    r"how[ _-]+to[ _-]+make[ _-]+[^:\r\n\u2014\u2013-]{1,80}|"
    r"(?:smallest[ _-]+)?change[ _-]+(?:needed|required)"
    r"(?:[ _-]+[^:\r\n\u2014\u2013-]{1,80})?|"
    r"counterfactual(?:[ _-]+(?:change|scenario|test))?"
    r")"
    r"(?:\?)?(?:\*\*|__)?[ \t]*(?:[:\u2014\u2013-][ \t]*|(?=\r?$))"
)
"""

OLD_MATCHES = """        headings = tuple(SEMANTIC_HEADING_PATTERNS[tag].finditer(document))
        combined = sorted((*exact[tag], *headings), key=lambda match: match.start())
"""

NEW_MATCHES = """        headings = tuple(SEMANTIC_HEADING_PATTERNS[tag].finditer(document))
        if tag == "answer_changing_change":
            # Merge the broader numbered fallback without double-counting a line
            # already recognized by the ordinary semantic-heading pattern.
            all_headings = (*headings, *STEP6_NUMBERED_HEADING_FALLBACK_RE.finditer(document))
            headings = tuple(
                match
                for _, match in sorted(
                    {
                        (match.start(), match.end()): match
                        for match in all_headings
                    }.items()
                )
            )
        combined = sorted((*exact[tag], *headings), key=lambda match: match.start())
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_trajectory(text: str) -> str:
    if "STEP6_NUMBERED_HEADING_FALLBACK_RE" in text:
        return text
    text = replace_once(text, OLD_ENTRY, NEW_ENTRY, "Step-6 semantic-label entry")
    text = replace_once(
        text,
        PATTERN_BLOCK_ANCHOR,
        PATTERN_BLOCK_ANCHOR + FALLBACK_BLOCK,
        "semantic-heading pattern block",
    )
    text = replace_once(text, OLD_MATCHES, NEW_MATCHES, "_semantic_matches heading merge")
    return text


def patch_version(text: str) -> str:
    if '__version__ = "3.0.12"' in text:
        return text
    if '__version__ = "3.0.11"' not in text:
        raise RuntimeError("expected package version 3.0.11")
    return text.replace('__version__ = "3.0.11"', '__version__ = "3.0.12"', 1)


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
        '__version__ = "3.0.12"' in init_text
        and "STEP6_NUMBERED_HEADING_FALLBACK_RE" in trajectory_text
    ):
        print("3.0.12 Step-6 heading refinement is already applied")
        return 0

    if '__version__ = "3.0.11"' not in init_text:
        raise SystemExit(
            "this overlay expects your current 3.0.11 checkout; do not apply it to an older version"
        )

    # Compute every transformed file before writing anything.
    new_trajectory = patch_trajectory(trajectory_text)
    new_init = patch_version(init_text)

    trajectory_backup = trajectory.with_suffix(".py.v311.bak")
    init_backup = init.with_suffix(".py.v311.bak")
    if not trajectory_backup.exists():
        shutil.copy2(trajectory, trajectory_backup)
    if not init_backup.exists():
        shutil.copy2(init, init_backup)

    trajectory.write_text(new_trajectory, encoding="utf-8")
    init.write_text(new_init, encoding="utf-8")

    pkg = root / "PKG-INFO"
    if pkg.is_file():
        pkg_text = pkg.read_text(encoding="utf-8")
        if "Version: 3.0.11" in pkg_text:
            pkg_backup = pkg.with_suffix(".v311.bak")
            if not pkg_backup.exists():
                shutil.copy2(pkg, pkg_backup)
            pkg.write_text(
                pkg_text.replace("Version: 3.0.11", "Version: 3.0.12", 1),
                encoding="utf-8",
            )

    print("applied 3.0.12 Step-6 heading-recognition refinement")
    print(
        "non-trivial-content rule, semantic counterfactual diagnostic, prompts, and generation are unchanged"
    )
    print("existing generations need rejudging/reanalysis only; no regeneration is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
