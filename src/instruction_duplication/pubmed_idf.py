"""Pinned PubMed document frequencies used by the lexical exposure metric."""

from __future__ import annotations

import csv
import gzip
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import httpx

from .io_utils import sha256_file

PUBMED_COMMIT = "53db991afc251782106cd817a1c3fa47a4d41781"
PUBMED_REFERENCE_URL = (
    "https://raw.githubusercontent.com/berenslab/llm-excess-vocab/"
    f"{PUBMED_COMMIT}/results/yearly-counts.csv.gz"
)
PUBMED_REFERENCE_NAME = "berenslab-pubmed-document-frequency-2010-2024"
PUBMED_REFERENCE_VERSION = "pubmed-global-idf-v3"
PUBMED_EXPECTED_SHA256 = "e42e37c9ad5abc4e098e0ea02558399b8557d85332bf350942c6cb9bda9d9d93"
PUBMED_EXPECTED_DOCUMENTS = 15_103_887
PUBMED_MINIMUM_ROWS = 300_000

type ProgressSink = Callable[[str], None]


def _ignore_progress(_message: str) -> None:
    return None


def default_cache_path() -> Path:
    """Return a platform-appropriate cache path without placing data in a run."""
    configured = os.environ.get("INSTRUCTION_DUPLICATION_CACHE_DIR")
    if configured:
        base = Path(configured).expanduser()
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "instruction-duplication" / "pubmed" / "yearly-counts.csv.gz"


def _validate_hash(path: Path) -> None:
    digest = sha256_file(path)
    if digest != PUBMED_EXPECTED_SHA256:
        raise RuntimeError(
            "PubMed document-frequency file has SHA-256 "
            f"{digest}, expected {PUBMED_EXPECTED_SHA256}"
        )


def ensure_pubmed_vocabulary(
    *,
    source_path: Path | None = None,
    cache_path: Path | None = None,
    timeout_seconds: float = 120.0,
    progress: ProgressSink | None = None,
) -> Path:
    """Return the verified immutable PubMed table, downloading it once if needed."""
    emit: ProgressSink = progress or _ignore_progress
    if source_path is not None:
        source = source_path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        _validate_hash(source)
        emit(f"[idf] using verified local PubMed vocabulary: {source}")
        return source

    destination = cache_path or default_cache_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            _validate_hash(destination)
        except RuntimeError:
            emit("[idf] cached PubMed vocabulary failed verification; downloading it again")
        else:
            emit(f"[idf] using cached PubMed vocabulary: {destination}")
            return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    emit("[idf] downloading the pinned PubMed document-frequency table (about 5.6 MB)")
    try:
        with httpx.stream(
            "GET",
            PUBMED_REFERENCE_URL,
            timeout=timeout_seconds,
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(128 * 1024):
                    if chunk:
                        handle.write(chunk)
        _validate_hash(temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    emit(f"[idf] cached verified PubMed vocabulary: {destination}")
    return destination


def read_document_frequencies(
    path: Path,
    selected_terms: Iterable[str],
    *,
    verify_published_source: bool = True,
) -> tuple[int, dict[str, int]]:
    """Read document frequencies only for requested terms from the compressed table."""
    if verify_published_source:
        _validate_hash(path)
    terms = {term.casefold().strip() for term in selected_terms if term.strip()}
    found: dict[str, int] = {}
    document_count: int | None = None
    row_count = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or fieldnames[0] != "word" or len(fieldnames) < 10:
            raise ValueError("PubMed vocabulary has an incompatible header")
        years = fieldnames[1:]
        for row in reader:
            row_count += 1
            word = str(row.get("word") or "").casefold()
            counts = [int(float(row.get(year) or 0)) for year in years]
            if word in terms:
                found[word] = sum(counts)
            if not word or word in {"total", "totals", "__total__", "all"}:
                document_count = sum(counts)
    if document_count is None:
        raise ValueError("PubMed vocabulary has no terminal abstract-count row")
    if verify_published_source:
        if row_count < PUBMED_MINIMUM_ROWS:
            raise ValueError("PubMed vocabulary is unexpectedly small")
        if document_count != PUBMED_EXPECTED_DOCUMENTS:
            raise ValueError(
                f"PubMed vocabulary contains {document_count:,} abstracts; "
                f"expected {PUBMED_EXPECTED_DOCUMENTS:,}"
            )
    return document_count, {term: found.get(term, 0) for term in sorted(terms)}


def fixture_document_frequencies(
    selected_terms: Iterable[str],
    values: Mapping[str, int] | None = None,
    *,
    document_count: int = 1_000_000,
) -> tuple[int, dict[str, int]]:
    """Build a small explicit reference for tests and fake end-to-end runs."""
    supplied = {key.casefold(): int(value) for key, value in (values or {}).items()}
    terms = {term.casefold().strip() for term in selected_terms if term.strip()}
    return document_count, {term: supplied.get(term, 1_000) for term in sorted(terms)}
