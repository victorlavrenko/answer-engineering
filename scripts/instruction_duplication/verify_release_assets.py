#!/usr/bin/env python3
"""Verify the frozen release assets for the Instruction Duplication preprint."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import tarfile
import tempfile
from pathlib import Path

PRIMARY = "instruction-duplication-paper-run.tar.gz"
AE = "instruction-placement-results.jsonl.gz"
CHECKSUMS = "SHA256SUMS"
EXPECTED_HASHES = {
    PRIMARY: "b4094a39981c427406aa925d0a3b8d9bfcaf1dadeecf4acacfa73903e86b7eec",
    AE: "c004b8b83691b5fdc4aada06481259c5e64237ad6fe80a82c8291965c8c37897",
}
EXPECTED_AE = {
    "trajectory-editing-orl-ssnhl-acute": {
        "system-only": 842,
        "duplicated-after-query": 971,
        "improved": 154,
        "degraded": 25,
    },
    "trajectory-editing-orl-conductive-acute": {
        "system-only": 786,
        "duplicated-after-query": 738,
        "improved": 153,
        "degraded": 201,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(answer: str, gold: str) -> bool:
    text = answer.lower()
    expression = gold.strip().removeprefix("-").strip().lower()
    if expression == "steroid":
        return "steroid" in text
    if expression == "conductive and not senso and not snhl":
        return "conductive" in text and "senso" not in text and "ssnhl" not in text
    raise AssertionError(f"unexpected gold expression: {gold!r}")


def exact_two_sided_mcnemar(improved: int, degraded: int) -> float:
    n = improved + degraded
    high = max(improved, degraded)
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(high, n + 1)) / (2**n))


def verify_checksums(root: Path) -> None:
    observed = {}
    for name in (PRIMARY, AE):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = sha256(path)
        if observed[name] != EXPECTED_HASHES[name]:
            raise RuntimeError(f"{name}: SHA-256 mismatch: {observed[name]}")
    manifest = root / CHECKSUMS
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        listed[name.strip().lstrip("*")] = digest
    if listed != EXPECTED_HASHES:
        raise RuntimeError(f"SHA256SUMS differs from frozen manifest: {listed}")
    print("checksums=ok")


def verify_primary(root: Path) -> None:
    archive = root / PRIMARY
    with tempfile.TemporaryDirectory(prefix="instruction-duplication-release-") as tmp:
        temp = Path(tmp)
        with tarfile.open(archive, "r:gz") as tf:
            members = tf.getmembers()
            for member in members:
                resolved = (temp / member.name).resolve()
                if temp.resolve() not in resolved.parents and resolved != temp.resolve():
                    raise RuntimeError(f"unsafe archive member: {member.name}")
            tf.extractall(temp, filter="data")
        bundle = temp / "instruction-duplication-paper-run"
        workspace = bundle / "paper-run"
        required = [
            bundle / "README.md",
            workspace / "manifest.json",
            workspace / "data" / "questions.jsonl",
            workspace / "data" / "dataset-audit.json",
            workspace / "data" / "generation-schedule.json",
            workspace / "results" / "cells-and-judgments.jsonl",
            bundle / "human-validation",
            bundle / "robustness-analysis",
            bundle / "measurement-validation",
        ]
        missing = [str(p.relative_to(bundle)) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f"primary archive missing: {missing}")
        cells = workspace / "results" / "cells-and-judgments.jsonl"
        count = sum(1 for line in cells.open(encoding="utf-8") if line.strip())
        if count != 16800:
            raise RuntimeError(f"expected 16800 frozen cells, found {count}")
        questions = workspace / "data" / "questions.jsonl"
        qcount = sum(1 for line in questions.open(encoding="utf-8") if line.strip())
        if qcount != 300:
            raise RuntimeError(f"expected 300 primary questions, found {qcount}")
        print(f"primary=ok cells={count} questions={qcount}")


def verify_ae(root: Path) -> None:
    with gzip.open(root / AE, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or rows[0].get("record_type") != "metadata":
        raise RuntimeError("AE artifact is missing its metadata row")
    data = rows[1:]
    if len(data) != 4000:
        raise RuntimeError(f"expected 4000 AE results, found {len(data)}")
    groups: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    questions: dict[str, str] = {}
    for row in data:
        if row.get("record_type") != "result":
            raise RuntimeError("unexpected AE row type")
        if evaluate(str(row["answer"]), str(row["gold"])) is not bool(row["ok"]):
            raise RuntimeError(f"stored AE score mismatch for {row['case_id']}")
        subrun = str(row["subrun"])
        condition = str(row["condition"])
        case_id = str(row["case_id"])
        groups.setdefault((subrun, condition), {})[case_id] = row
        qhash = str(row["question_sha256"])
        question = str(row["question"])
        prior = questions.setdefault(qhash, question)
        if prior != question:
            raise RuntimeError("question hash maps to multiple question texts")
    if len(questions) != 2000:
        raise RuntimeError(f"expected 2000 unique AE questions, found {len(questions)}")
    for subrun, expected in EXPECTED_AE.items():
        baseline = groups[(subrun, "system-only")]
        duplicate = groups[(subrun, "duplicated-after-query")]
        if len(baseline) != 1000 or len(duplicate) != 1000 or baseline.keys() != duplicate.keys():
            raise RuntimeError(f"pair integrity failure for {subrun}")
        base_ok = sum(bool(v["ok"]) for v in baseline.values())
        dup_ok = sum(bool(v["ok"]) for v in duplicate.values())
        improved = sum(not bool(baseline[k]["ok"]) and bool(duplicate[k]["ok"]) for k in baseline)
        degraded = sum(bool(baseline[k]["ok"]) and not bool(duplicate[k]["ok"]) for k in baseline)
        observed = {
            "system-only": base_ok,
            "duplicated-after-query": dup_ok,
            "improved": improved,
            "degraded": degraded,
        }
        if observed != expected:
            raise RuntimeError(f"AE aggregate mismatch for {subrun}: {observed}")
        p = exact_two_sided_mcnemar(improved, degraded)
        print(
            f"ae={subrun} {base_ok}/1000->{dup_ok}/1000 "
            f"improved={improved} degraded={degraded} p={p:.16g}"
        )
    print(f"ae=ok rows={len(data)} unique_questions={len(questions)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path, help="directory containing the three release assets")
    return parser.parse_args()


def main() -> int:
    root = parse_args().release_dir.resolve()
    verify_checksums(root)
    verify_primary(root)
    verify_ae(root)
    print("instruction_duplication_release=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
