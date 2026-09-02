#!/usr/bin/env python3
"""Apply the human-validated 3.0.10 counterfactual-judge refinement.

This is deliberately an in-place source overlay rather than a new generator.  It
changes only deterministic judging code and the package version, so existing model
generations can be rejudged without regeneration.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

BASE_JUDGE_GIT_BLOB_SHA1 = "c8c783e40dda70be00a4baaabf229d7184da8317"
BASE_INIT_GIT_BLOB_SHA1 = "fb88e7eb0f21c580b8118ff8c5f6096269e2259b"
PATCH_MARKER = 'HUMAN_VALIDATED_COUNTERFACTUAL_JUDGE = "v1"'


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


REWRITE_BLOCK = r"""COUNTERFACTUAL_TASK_REWRITE_RE = re.compile(
    # Explicitly changing what the item asks is not a change to the patient's case.
    r"\b(?:rephras(?:e|ed|ing)|rewrit(?:e|ten|ing)|chang(?:e|ed|ing)|modif(?:y|ied|ying))\b"
    r"[^.;\n]{0,60}\b(?:stem|question)\b[^.;\n]{0,80}\b(?:ask|focus|word|phrasing|intent)\b"
    r"|\b(?:if\s+)?the\s+(?:stem|question)\s+(?:were\s+|was\s+)?"
    r"(?:rephrased|rewritten|changed|modified)\s+to\s+(?:ask|focus|word)\b"
    r"|\b(?:if\s+)?the\s+(?:stem|question)\s+(?:were\s+|was\s+)?"
    r"(?:asked|asking|focused)\b"
    r"|\b(?:shift|change|alter)\w*\b[^.;\n]{0,60}\b"
    r"(?:question(?:'s)?\s+(?:focus|intent)|focus\s+of\s+(?:the\s+)?question|"
    r"what\s+the\s+question\s+asks)\b"
    # Natural rewrites that escaped the older patterns: "question were changed to:\n"
    # followed by a new WH-task, or "question specifically asked about ...".
    r"|\b(?:if\s+)?(?:the\s+)?(?:stem|question)\s+"
    r"(?:were\s+|was\s+|is\s+|had\s+)?(?:rephrased|rewritten|changed|modified)\s+to\s*"
    r"[:,'\"“”\s-]*(?:ask\b|focus\b|word\b|which\b|what\b|who\b|when\b|where\b|how\b)"
    r"|\b(?:if\s+)?(?:the\s+)?(?:stem|question)\s+(?:specifically\s+)?"
    r"(?:asks?|asked|asking)\s+(?:about|for|which|what|who|when|where|how)\b"
    r"|\b(?:if\s+)?(?:the\s+)?(?:stem|question)\s+(?:had\s+)?stated\s*"
    r"[,:'\"“”\s-]+(?:which|what|who|when|where|how)\b"
    r"|\b(?:if\s+)?(?:the\s+)?(?:stem|question)\s+specified\s*"
    r"[,:'\"“”\s-]*(?:which|what|who|when|where|how)\b"
    # Rewriting an answer choice is likewise not a patient/case counterfactual.
    r"|\b(?:if\s+)?(?:option|choice)\s+\(?(?-i:[A-Z])\)?[^.;\n]{0,45}"
    r"\b(?:had\s+said|were\s+(?:changed|rewritten|rephrased)|"
    r"was\s+(?:changed|rewritten|rephrased))\b",
    re.IGNORECASE,
)
"""

COUNTERFACTUAL_REPLACEMENT = r'''HUMAN_VALIDATED_COUNTERFACTUAL_JUDGE = "v1"

COUNTERFACTUAL_SCENARIO_SPLIT_RE = re.compile(
    # Do not split on the period in a compact option label such as "F. Release".
    r"(?<![A-Z]\.)(?<=[.!?])\s+|\n{2,}"
)
COUNTERFACTUAL_AMBIGUOUS_OR_RE = re.compile(
    r"\b(?-i:([A-Z]))\.\s*[^;\n]{0,120}\bor\s+"
    r"(?-i:([A-Z]))\.\s*[^;\n]{0,120}"
    r"\b(?:would|could|might)\b[^;\n]{0,80}"
    r"\b(?:best|preferred|most\s+likely|more\s+likely)\b",
    re.IGNORECASE,
)


def _counterfactual_scenarios(text: str) -> tuple[str, ...]:
    """Split alternative hypothetical branches without splitting option labels."""
    scenarios = tuple(
        part.strip() for part in COUNTERFACTUAL_SCENARIO_SPLIT_RE.split(text) if part.strip()
    )
    return scenarios or (text.strip(),)


def _normalized_choice_aliases(choice: str) -> tuple[str, ...]:
    """Return conservative aliases for detecting a named alternative in prose.

    The only normalization beyond the existing answer normalizer is removal of a
    leading determiner/possessive, allowing e.g. choice text ``His father`` to match
    ``father``.  We do not invent abbreviations or semantic synonyms here.
    """
    normalized = normalize_text(choice).casefold().strip()
    aliases = {normalized} if normalized else set()
    tokens = normalized.split()
    while tokens and tokens[0] in {"the", "a", "an", "his", "her", "their"}:
        tokens = tokens[1:]
    if tokens:
        aliases.add(" ".join(tokens))
    return tuple(sorted((alias for alias in aliases if len(alias) >= 3), key=len, reverse=True))


def _natural_second_best_target(
    scenario: str,
    choices: Mapping[str, str],
    second_best_option: str,
) -> bool:
    """Recognize natural winner language missed by the strict option regexes."""
    normalized = normalize_text(scenario).casefold()
    aliases = _normalized_choice_aliases(choices[second_best_option])
    for alias in aliases:
        escaped = re.escape(alias)
        patterns = (
            # "would strongly suggest Crohn's disease over ulcerative colitis"
            rf"\b(?:would|could|might)\b.{{0,55}}\b"
            rf"(?:suggest|favor|favour|support|point\s+to)\s+(?:the\s+)?{escaped}\b",
            # "previous attempt would become the most significant risk factor"
            rf"\b{escaped}\b.{{0,110}}\b(?:would|could|might)\b.{{0,65}}\b"
            rf"(?:become|be|represent)\b.{{0,30}}\b"
            rf"(?:best|preferred|better|most\s+likely|more\s+likely|"
            rf"most\s+significant|strongest)\b",
            # "shift the diagnosis/preference towards Crohn's disease"
            rf"\b(?:shift|change|switch)\b.{{0,40}}\b"
            rf"(?:diagnosis|answer|preference)\b.{{0,20}}\b"
            rf"(?:to|toward|towards)\s+(?:the\s+)?{escaped}\b",
        )
        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns):
            return True

    # Safe anaphora: this role is evaluated together with a separately recovered,
    # declared best alternative.  Saying the "best alternative" would become/favor
    # the winner therefore identifies that recovered option without guessing medicine.
    return bool(
        re.search(
            r"\b(?:make|making)\s+(?:the\s+)?best\s+alternative\b.{0,100}\b"
            r"(?:best|preferred|correct|better|most\s+likely)\b",
            normalized,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:change|shift|switch)\s+(?:the\s+)?answer\s+to\s+"
            r"(?:favor|favour)\s+(?:the\s+)?best\s+alternative\b",
            normalized,
            re.IGNORECASE,
        )
        or re.search(
            r"\bbest\s+alternative\b.{0,90}\b(?:would|could|might)\b.{0,65}\b"
            r"(?:become|be)\b.{0,30}\b(?:best|preferred|better|most\s+likely)\b",
            normalized,
            re.IGNORECASE,
        )
    )


def _scenario_has_ambiguous_joint_winner(
    scenario: str,
    choices: Mapping[str, str],
    second_best_option: str,
) -> bool:
    """Reject a single hypothetical branch that offers two co-winners.

    Separate later branches are allowed.  For example, a response can first give a
    valid change that makes its stated alternative win and then discuss another
    hypothetical.  What fails is ``F or H would become the best answer`` inside the
    same proposed change.
    """
    for match in COUNTERFACTUAL_AMBIGUOUS_OR_RE.finditer(scenario):
        left, right = (group.upper() for group in match.groups())
        if left in choices and right in choices and left != right and second_best_option in {left, right}:
            return True
    return False


def _second_best_has_winning_scenario(
    text: str,
    choices: Mapping[str, str],
    second_best_option: str,
) -> bool:
    """Require at least one unambiguous hypothetical where the declared alternative wins."""
    for scenario in _counterfactual_scenarios(text):
        if _scenario_has_ambiguous_joint_winner(scenario, choices, second_best_option):
            continue
        winners = _explicit_counterfactual_winners(scenario, choices)
        other_winners = winners - {second_best_option}
        if second_best_option in winners and not other_winners:
            return True
        # If a strict regex found a *different* winner in this same branch, do not
        # override it with looser anaphoric/choice-text matching.
        if other_winners:
            continue
        if _natural_second_best_target(scenario, choices, second_best_option):
            return True
    return False


def _case_specific_counterfactual(
    question: Question,
    text: str,
    second_best_option: str | None,
) -> bool:
    """Judge the requested answer-changing counterfactual as an observable operation.

    PASS requires all of the following, without judging medical truth:
    1. a substantive hypothetical rather than boilerplate;
    2. a change to case content, not a rewrite of the task or answer choices;
    3. a valid, separately declared best alternative;
    4. at least one unambiguous hypothetical branch in which that exact alternative
       is said to become/furnish the winning answer; and
    5. concrete content beyond merely naming the alternative.

    This definition follows the atomic human-validation instruction.  In particular,
    ``X or Y would become the best answer`` does not complete the role when X is the
    declared alternative, while natural wording such as ``skip lesions would strongly
    suggest Crohn's disease over ulcerative colitis`` does complete it.
    """
    if not (_informative(text, 7) and COUNTERFACTUAL_CUES.search(text)):
        return False
    if COUNTERFACTUAL_TASK_REWRITE_RE.search(text):
        return False
    if second_best_option is None or second_best_option not in question.choices:
        return False
    if not _second_best_has_winning_scenario(text, question.choices, second_best_option):
        return False

    specific = {token for token in content_tokens(text) if token not in COUNTERFACTUAL_META_TOKENS}
    choice_terms = set(content_tokens(question.choices[second_best_option]))
    introduced = specific - choice_terms
    if len(introduced) < 2 and not (
        set(content_tokens(text)).intersection(choice_terms) and RATIONALE_CUES.search(text)
    ):
        return False

    # Keep the construct case-grounded.  New hypothetical findings are allowed and
    # often absent from the original stem, so strong literal overlap is not required;
    # the introduced-content test above supplies the complementary guard.
    return bool(
        score_anchor_recall(question.stem, text) >= 0.05
        or specific.intersection(choice_terms)
        or len(introduced) >= 3
    )
'''


def patch_judge(text: str) -> str:
    if PATCH_MARKER in text:
        return text

    start = text.index("COUNTERFACTUAL_TASK_REWRITE_RE = re.compile(")
    end = text.index("COUNTERFACTUAL_TARGET_RES = (", start)
    text = text[:start] + REWRITE_BLOCK + "\n" + text[end:]

    start = text.index("def _case_specific_counterfactual(")
    end = text.index("def _failure_judgment(", start)
    text = text[:start] + COUNTERFACTUAL_REPLACEMENT + "\n\n" + text[end:]
    return text


def bump_version(text: str) -> str:
    if '__version__ = "3.0.10"' in text:
        return text
    old = '__version__ = "3.0.9"'
    if old not in text:
        raise RuntimeError("expected package version 3.0.9; refusing to guess a version edit")
    return text.replace(old, '__version__ = "3.0.10"', 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--force", action="store_true", help="allow a non-pristine 3.0.9 judge.py")
    args = parser.parse_args()

    root = args.root.resolve()
    judge_path = root / "src" / "instruction_duplication" / "judge.py"
    init_path = root / "src" / "instruction_duplication" / "__init__.py"
    pkg_info_path = root / "PKG-INFO"
    if not judge_path.is_file() or not init_path.is_file():
        raise SystemExit(f"not an instruction-duplication source checkout: {root}")

    judge_bytes = judge_path.read_bytes()
    init_bytes = init_path.read_bytes()
    judge_text = judge_bytes.decode("utf-8")
    init_text = init_bytes.decode("utf-8")

    if PATCH_MARKER in judge_text and '__version__ = "3.0.10"' in init_text:
        print("human-validated judge 3.0.10 is already applied")
        return 0

    judge_sha = git_blob_sha1(judge_bytes)
    init_sha = git_blob_sha1(init_bytes)
    if not args.force:
        if judge_sha != BASE_JUDGE_GIT_BLOB_SHA1:
            raise SystemExit(
                "judge.py does not match the frozen 3.0.9 base; refusing to overwrite local edits.\n"
                f"expected git blob {BASE_JUDGE_GIT_BLOB_SHA1}, got {judge_sha}\n"
                "Re-run with --force only after reviewing your local judge.py changes."
            )
        if init_sha != BASE_INIT_GIT_BLOB_SHA1:
            raise SystemExit(
                "__init__.py does not match the frozen 3.0.9 base; refusing to guess version state.\n"
                f"expected git blob {BASE_INIT_GIT_BLOB_SHA1}, got {init_sha}"
            )

    patched_judge = patch_judge(judge_text)
    patched_init = bump_version(init_text)

    shutil.copy2(judge_path, judge_path.with_suffix(".py.v309.bak"))
    shutil.copy2(init_path, init_path.with_suffix(".py.v309.bak"))
    judge_path.write_text(patched_judge, encoding="utf-8")
    init_path.write_text(patched_init, encoding="utf-8")

    if pkg_info_path.is_file():
        pkg = pkg_info_path.read_text(encoding="utf-8")
        if "Version: 3.0.9" in pkg:
            shutil.copy2(pkg_info_path, pkg_info_path.with_suffix(".v309.bak"))
            pkg_info_path.write_text(
                pkg.replace("Version: 3.0.9", "Version: 3.0.10", 1), encoding="utf-8"
            )

    print("applied human-validated counterfactual judge; package version is now 3.0.10")
    print("generation code and prompts were not changed; existing generations can be rejudged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
