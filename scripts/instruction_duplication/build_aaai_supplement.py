#!/usr/bin/env python3
"""Build an anonymized AAAI supplementary code/data ZIP.

This helper is for double-blind review. It copies the reproducibility source tree
and a frozen experiment workspace, strips author-identifying metadata from small
text files, scans every copied file for known identifiers, excludes ordinary
VCS/build caches, and fails if identifiers remain. It does not modify the source
repository or the original run.

Paper mapping: Reproducibility, Limitations, and Responsible Use.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "pyproject.toml",
    "requirements-dev.lock",
    "requirements-research.lock",
    "setup.cfg",
)
ROOT_DIRS = ("src", "tests", "docs", "examples", "scripts")
RUN_DIRS = ("config", "data", "results", "state")
RUN_FILES = ("manifest.json",)

EXCLUDED_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    "PKG-INFO",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
MAX_IN_MEMORY_SCRUB_BYTES = 16 * 1024 * 1024

# Keep this list narrow and inspect the resulting ZIP manually. These strings are
# identifiers, not scientific content.
IDENTIFYING_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Victor Lavrenko", re.IGNORECASE), "Anonymous Author"),
    (re.compile(r"victorlavrenko", re.IGNORECASE), "anonymous-author"),
    (re.compile(r"victor@peacetech\.vc", re.IGNORECASE), "anonymous@example.invalid"),
    (re.compile(r"PeaceTech VC", re.IGNORECASE), "Anonymous Institution"),
)
IDENTIFYING_BYTES = tuple(
    needle.casefold().encode("utf-8")
    for needle in ("Victor Lavrenko", "victorlavrenko", "victor@peacetech.vc", "PeaceTech VC")
)
TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".toml",
    ".cfg",
    ".ini",
    ".json",
    ".jsonl",
    ".csv",
    ".yaml",
    ".yml",
    ".tex",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ignored(path: Path) -> bool:
    return any(part in EXCLUDED_NAMES for part in path.parts) or path.suffix in EXCLUDED_SUFFIXES


def copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    for path in source.rglob("*"):
        rel = path.relative_to(source)
        if ignored(rel) or path.is_dir():
            continue
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def scrub_text(path: Path) -> None:
    """Scrub small text metadata; large artifacts are scanned but never rewritten in memory."""
    if path.stat().st_size > MAX_IN_MEMORY_SCRUB_BYTES:
        return
    if path.suffix.casefold() not in TEXT_SUFFIXES and path.name not in {"LICENSE", "MANIFEST.in"}:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    for pattern, replacement in IDENTIFYING_PATTERNS:
        text = pattern.sub(replacement, text)
    # OpenRouter attribution is not experimental content and can identify the authors.
    text = text.replace(
        "https://github.com/anonymous-author/answer-engineering",
        "https://example.invalid/anonymous-supplement",
    )
    path.write_text(text, encoding="utf-8")


def contains_identifier(path: Path) -> bool:
    """Scan any file, including SQLite/large JSONL, without loading it all into memory."""
    overlap = 128
    tail = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            folded = (tail + chunk).lower()
            if any(needle in folded for needle in IDENTIFYING_BYTES):
                return True
            tail = folded[-overlap:]
    return False


def assert_anonymous(root: Path) -> None:
    hits: list[str] = []
    for path in root.rglob("*"):
        if path.is_dir() or ignored(path.relative_to(root)):
            continue
        if contains_identifier(path):
            hits.append(path.relative_to(root).as_posix())
    if hits:
        raise SystemExit(
            "Anonymous-package check failed; identifying strings remain in:\n"
            + "\n".join(hits)
            + "\nRemove/scrub those files or update the narrow scrub rules after manual inspection."
        )


def write_manifest(root: Path) -> None:
    rows = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_readme(root: Path) -> None:
    text = (
        "# Anonymous supplementary code and data\n\n"
        "This archive accompanies a double-blind submission. Author-identifying metadata "
        "has been scrubbed; the scientific source and frozen experiment records are otherwise "
        "copied from the reproduction materials.\n\n"
        "## Reanalysis\n\n"
        "Create a Python environment, install `requirements-research.lock` and the package, "
        "then run:\n\n"
        "```bash\n"
        "python -m pip install -r requirements-research.lock\n"
        "python -m pip install -e .\n"
        "instruction-duplication status --workspace paper-run\n"
        "instruction-duplication judge --workspace paper-run\n"
        "instruction-duplication analyze --workspace paper-run\n"
        "```\n\n"
        "Rejudging and analysis use stored generations and do not require provider credentials. "
        "New generation requires provider credentials and may not reproduce hosted-provider "
        "outputs byte-for-byte.\n\n"
        "See `docs/reproducibility.md`, `docs/paper-run-parameters.md`, and "
        "`docs/experiment.md` for the experimental contract and exact paper-run settings.\n\n"
        "Human-validation ratings are intentionally not asserted by this package until that "
        "audit is finalized.\n"
    )
    (root / "ANONYMOUS_SUPPLEMENT_README.md").write_text(text, encoding="utf-8")


def build(repo: Path, run: Path, output: Path) -> None:
    repo = repo.resolve()
    run = run.resolve()
    output = output.resolve()
    if not (repo / "pyproject.toml").is_file():
        raise SystemExit(f"Not a repository root: {repo}")
    if not (run / "manifest.json").is_file():
        raise SystemExit(f"Not an experiment workspace: {run}")

    with tempfile.TemporaryDirectory(prefix="aaai27-supplement-") as temp_name:
        root = Path(temp_name) / "supplement"
        root.mkdir()

        for name in ROOT_FILES:
            source = repo / name
            if source.is_file():
                shutil.copy2(source, root / name)
        for name in ROOT_DIRS:
            copy_tree(repo / name, root / name)

        paper_run = root / "paper-run"
        paper_run.mkdir()
        for name in RUN_FILES:
            shutil.copy2(run / name, paper_run / name)
        for name in RUN_DIRS:
            copy_tree(run / name, paper_run / name)

        # The treatment/mechanical decoding key is not needed for reproducing the automatic
        # paper results and is withheld until human ratings are finalized.
        for relative in (
            "paper-run/results/blinded-matched-pair-key.jsonl",
            "paper-run/results/human-audit-schema.json",
        ):
            candidate = root / relative
            if candidate.exists():
                candidate.unlink()

        for path in list(root.rglob("*")):
            if path.is_file():
                scrub_text(path)

        make_readme(root)
        assert_anonymous(root)
        write_manifest(root)

        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root))

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.1f} MiB)")
    print(f"SHA-256: {sha256(output)}")
    print("Inspect the ZIP manually before uploading it to a double-blind review system.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--run", type=Path, required=True, help="frozen paper workspace")
    parser.add_argument("--output", type=Path, required=True, help="output ZIP path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.repo, args.run, args.output)
