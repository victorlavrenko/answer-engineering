"""Workspace paths, artifact integrity validation, and process locking."""

from __future__ import annotations

import csv
import os
import subprocess
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .audit import MODEL_ELIGIBILITY_VERSION
from .environment import runtime_environment
from .facts import FACT_INVENTORY_VERSION
from .io_utils import read_json, read_jsonl, sha256_json
from .json_types import object_value
from .lexical import LEXICAL_VERSION, selected_terms_hash
from .manifest import ExperimentManifest
from .models import SMOKE_PROFILE_ID
from .protocol import CONDITIONS, PROTOCOL_HASH
from .schedule import SCHEDULE_VERSION
from .types import Question

TRANSPORT_COMPATIBLE_PACKAGE_SERIES = (
    frozenset({"3.0.1", "3.0.2", "3.0.3"}),
    frozenset({"3.0.4", "3.0.5", "3.0.6", "3.0.7", "3.0.8", "3.0.9"}),
)


def _package_versions_generation_compatible(stored: str, current: str) -> bool:
    """Allow exact versions plus explicitly generation-compatible transport patch series."""
    return stored == current or any(
        stored in series and current in series for series in TRANSPORT_COMPATIBLE_PACKAGE_SERIES
    )


def _generation_runtime_environment_matches(
    stored: Mapping[str, object],
    current: Mapping[str, object],
    package_version: str,
) -> bool:
    """Allow only the package-version delta for transport-compatible 3.0.x patches."""
    if stored == current:
        return True
    if not _package_versions_generation_compatible(package_version, __version__):
        return False
    stored_distributions = stored.get("distributions")
    current_distributions = current.get("distributions")
    try:
        stored_versions = object_value(stored_distributions, name="stored distributions")
        current_versions = object_value(current_distributions, name="current distributions")
    except ValueError:
        return False
    stored_package = stored_versions.get("instruction-duplication")
    current_package = current_versions.get("instruction-duplication")
    if (
        not isinstance(stored_package, str)
        or not isinstance(current_package, str)
        or not _package_versions_generation_compatible(stored_package, current_package)
    ):
        return False
    normalized = dict(stored)
    normalized_distributions = dict(stored_versions)
    normalized_distributions["instruction-duplication"] = current_package
    normalized["distributions"] = normalized_distributions
    return normalized == current


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path

    @property
    def directory(self) -> Path:
        return self.root

    @property
    def config_directory(self) -> Path:
        return self.root / "config"

    @property
    def data_directory(self) -> Path:
        return self.root / "data"

    @property
    def state_directory(self) -> Path:
        return self.root / "state"

    @property
    def results_directory(self) -> Path:
        return self.root / "results"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def questions(self) -> Path:
        return self.data_directory / "questions.jsonl"

    @property
    def models(self) -> Path:
        return self.config_directory / "models.json"

    @property
    def dataset_audit(self) -> Path:
        return self.data_directory / "dataset-audit.json"

    @property
    def environment(self) -> Path:
        return self.config_directory / "environment.json"

    @property
    def lexical_reference(self) -> Path:
        return self.data_directory / "lexical-reference.json"

    @property
    def fact_inventory(self) -> Path:
        return self.data_directory / "fact-inventory.jsonl"

    @property
    def question_qc(self) -> Path:
        return self.data_directory / "question-qc.jsonl"

    @property
    def generation_schedule(self) -> Path:
        return self.data_directory / "generation-schedule.json"

    @property
    def model_eligibility(self) -> Path:
        return self.config_directory / "model-eligibility.json"

    @property
    def database(self) -> Path:
        return self.state_directory / "run.sqlite3"

    @property
    def routes(self) -> Path:
        return self.config_directory / "routes.json"

    @property
    def preflight_log(self) -> Path:
        return self.config_directory / "preflight.json"

    @property
    def analysis(self) -> Path:
        return self.results_directory / "analysis.json"

    @property
    def report(self) -> Path:
        return self.results_directory / "report.txt"

    @property
    def paper_report(self) -> Path:
        return self.results_directory / "paper-report.txt"

    @property
    def model_effects(self) -> Path:
        return self.results_directory / "model-effects.csv"

    @property
    def model_effect_summary(self) -> Path:
        return self.results_directory / "model-effect-summary.csv"

    @property
    def human_audit(self) -> Path:
        return self.results_directory / "blinded-matched-pair-audit.jsonl"

    @property
    def human_audit_key(self) -> Path:
        return self.results_directory / "blinded-matched-pair-key.jsonl"

    @property
    def human_audit_schema(self) -> Path:
        return self.results_directory / "human-audit-schema.json"

    @property
    def cells_export(self) -> Path:
        return self.results_directory / "cells-and-judgments.jsonl"

    @property
    def attempts_export(self) -> Path:
        return self.results_directory / "attempts.jsonl"

    @property
    def lock_path(self) -> Path:
        return self.root / ".lock"

    def create(self) -> None:
        for path in (
            self.root,
            self.config_directory,
            self.data_directory,
            self.state_directory,
            self.results_directory,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> ExperimentManifest:
        value = read_json(self.manifest)
        if not isinstance(value, dict):
            raise ValueError("manifest must be a JSON object")
        return ExperimentManifest.from_dict(value)

    def require_stored_run_integrity(
        self, *, require_runtime_environment: bool = False
    ) -> ExperimentManifest:
        """Validate frozen run artifacts without imposing current prompt semantics.

        Judging and analysis operate on immutable stored questions and generations.  A
        newer measurement implementation may therefore audit a run produced by an
        older protocol version.  Generation/resume uses :meth:`require_generation_integrity`
        and remains strict about the current protocol and condition definitions.
        """
        required = (
            self.manifest,
            self.questions,
            self.models,
            self.dataset_audit,
            self.environment,
            self.database,
        )
        missing = [str(path.relative_to(self.root)) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("workspace is not prepared; missing " + ", ".join(missing))

        manifest = self.load_manifest()
        questions = read_jsonl(self.questions)
        models = read_json(self.models)
        environment = read_json(self.environment)
        if not isinstance(models, list):
            raise ValueError("models.json must contain a list")
        environment_matches_manifest = (
            isinstance(environment, dict) and sha256_json(environment) == manifest.environment_hash
        )
        checks = {
            "questions_hash": sha256_json(questions) == manifest.questions_hash,
            "question_ids": tuple(str(row.get("id")) for row in questions) == manifest.question_ids,
            "models_hash": sha256_json(models) == manifest.models_hash,
            "environment_hash": environment_matches_manifest,
        }
        if require_runtime_environment:
            checks["runtime_environment"] = (
                environment_matches_manifest and environment == runtime_environment()
            )
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                "workspace frozen artifacts failed integrity validation: " + ", ".join(failed)
            )
        return manifest

    def require_generation_integrity(
        self, *, require_runtime_environment: bool = False
    ) -> ExperimentManifest:
        """Require stored artifacts plus the current generation semantics."""
        manifest = self.require_stored_run_integrity(require_runtime_environment=False)
        checks = {
            "protocol_hash": manifest.protocol_hash == PROTOCOL_HASH,
            "condition_hash": manifest.condition_hash
            == sha256_json([condition.to_dict() for condition in CONDITIONS]),
            "smoke_profile_id": manifest.smoke_profile_id == SMOKE_PROFILE_ID,
        }
        if require_runtime_environment:
            environment = read_json(self.environment)
            checks["runtime_environment"] = isinstance(
                environment, dict
            ) and _generation_runtime_environment_matches(
                environment,
                runtime_environment(),
                manifest.package_version,
            )
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                "workspace generation semantics changed: "
                + ", ".join(failed)
                + "; use the software that prepared this workspace to resume generation"
            )
        return manifest

    def require_prepared(self, *, require_runtime_environment: bool = True) -> ExperimentManifest:
        """Require generation-compatible software and current generation-support artifacts.

        Judge and analysis versions are intentionally *not* part of generation compatibility:
        a measurement-only release may rejudge or resume an older frozen generation workspace
        without changing prompts, routes, schedules, or already-completed responses.
        """
        manifest = self.require_generation_integrity(
            require_runtime_environment=require_runtime_environment
        )
        checks = {
            "package_version": _package_versions_generation_compatible(
                manifest.package_version, __version__
            ),
            "lexical_version": manifest.lexical_version == LEXICAL_VERSION,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                "workspace software or measurement identity changed: "
                + ", ".join(failed)
                + "; generation/resume requires a workspace prepared by this version"
            )
        if not self.lexical_reference.is_file():
            raise RuntimeError("workspace is not prepared; missing data/lexical-reference.json")
        required_scientific = (
            self.fact_inventory,
            self.question_qc,
            self.generation_schedule,
            self.model_eligibility,
        )
        missing_scientific = [
            str(path.relative_to(self.root)) for path in required_scientific if not path.is_file()
        ]
        if missing_scientific:
            raise RuntimeError(
                "workspace is not prepared; missing " + ", ".join(missing_scientific)
            )
        lexical = read_json(self.lexical_reference)
        if not isinstance(lexical, dict) or lexical.get("lexical_version") != LEXICAL_VERSION:
            raise RuntimeError("lexical-reference.json is missing or incompatible")
        questions = [Question.from_dict(row) for row in read_jsonl(self.questions)]
        if lexical.get("selected_terms_hash") != selected_terms_hash(questions):
            raise RuntimeError("lexical-reference.json does not match the selected questions")

        fact_rows = read_jsonl(self.fact_inventory)
        if [str(row.get("question_id")) for row in fact_rows] != list(manifest.question_ids) or any(
            row.get("fact_inventory_version") != FACT_INVENTORY_VERSION for row in fact_rows
        ):
            raise RuntimeError("fact-inventory.jsonl does not match the selected questions")
        qc_rows = read_jsonl(self.question_qc)
        if [str(row.get("question_id")) for row in qc_rows] != list(manifest.question_ids):
            raise RuntimeError("question-qc.jsonl does not match the selected questions")

        schedule = read_json(self.generation_schedule)
        schedule_cells = schedule.get("cells") if isinstance(schedule, dict) else None
        expected_cells = len(manifest.question_ids) * len(manifest.models) * len(CONDITIONS)
        if (
            not isinstance(schedule, dict)
            or schedule.get("schedule_version") != SCHEDULE_VERSION
            or not isinstance(schedule_cells, list)
            or schedule.get("cell_count") != expected_cells
            or schedule.get("schedule_hash") != sha256_json(schedule_cells)
        ):
            raise RuntimeError("generation-schedule.json failed integrity validation")

        model_eligibility = read_json(self.model_eligibility)
        if not isinstance(model_eligibility, dict):
            raise RuntimeError("model-eligibility.json is not a JSON object")
        eligibility_hash = model_eligibility.get("snapshot_sha256")
        unhashed_eligibility = dict(model_eligibility)
        unhashed_eligibility.pop("snapshot_sha256", None)
        if model_eligibility.get(
            "model_eligibility_version"
        ) != MODEL_ELIGIBILITY_VERSION or eligibility_hash != sha256_json(unhashed_eligibility):
            raise RuntimeError("model-eligibility.json failed integrity validation")
        return manifest

    @staticmethod
    def _windows_pid_alive(pid: int) -> bool:
        """Check a PID without using ``os.kill(pid, 0)`` on Windows."""
        if pid == os.getpid():
            return True
        try:
            result = subprocess.run(
                ["tasklist.exe", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=5.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            # If Windows cannot answer safely, preserve the lock rather than
            # deleting one that may belong to a live process.
            return True
        if result.returncode != 0:
            return True
        expected_pid = str(pid)
        return any(
            len(row) >= 2 and row[1].strip() == expected_pid
            for row in csv.reader(result.stdout.splitlines())
        )

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            return Workspace._windows_pid_alive(pid)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    @contextmanager
    def lock(self) -> Generator[None, None, None]:
        """Prevent concurrent mutation while recovering stale crash locks safely."""
        self.root.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        for _ in range(2):
            try:
                descriptor = os.open(self.lock_path, flags, 0o600)
                break
            except FileExistsError as exc:
                text = self.lock_path.read_text(encoding="utf-8", errors="replace")
                try:
                    pid = int(text.split("pid=", 1)[1].splitlines()[0])
                except (IndexError, ValueError):
                    pid = -1
                if not self._pid_alive(pid):
                    self.lock_path.unlink(missing_ok=True)
                    continue
                raise RuntimeError(f"workspace is locked by process {pid}") from exc
        else:
            raise RuntimeError("could not acquire workspace lock")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()}\n")
                handle.flush()
                os.fsync(handle.fileno())
            yield
        finally:
            self.lock_path.unlink(missing_ok=True)
