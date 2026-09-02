"""Release-invariant checks that require only runtime dependencies."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
SOURCE = SOURCE_ROOT / "instruction_duplication"
sys.path.insert(0, str(SOURCE_ROOT))


def _smoke_profile() -> None:
    from instruction_duplication.models import (
        MODELS,
        SMOKE_PROFILE_ID,
        SMOKE_PROFILE_SCHEMA_VERSION,
    )
    from instruction_duplication.protocol import PROTOCOL_HASH

    public_path = ROOT / "data" / "smoke-profile-2026-08-05.json"
    package_path = SOURCE / public_path.name
    public = json.loads(public_path.read_text(encoding="utf-8"))
    packaged = json.loads(package_path.read_text(encoding="utf-8"))
    if public != packaged:
        raise RuntimeError("public and packaged smoke profiles differ")
    if public.get("schema_version") != SMOKE_PROFILE_SCHEMA_VERSION:
        raise RuntimeError("smoke profile schema version is stale")
    if public.get("profile_id") != SMOKE_PROFILE_ID:
        raise RuntimeError("smoke profile identity does not match the model configuration")
    if public["completed_cells"] != 1680:
        raise RuntimeError("smoke profile must contain 1,680 completed cells")
    provenance = public.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("smoke profile provenance is missing")
    if provenance.get("active_protocol_hash") != PROTOCOL_HASH:
        raise RuntimeError("smoke profile is not approved for the active protocol")
    if provenance.get("compatibility") != "provisional_cross_protocol":
        raise RuntimeError("unexpected smoke profile compatibility status")
    if provenance.get("source_protocol_hash") is not None:
        raise RuntimeError("legacy smoke profile must not invent an unavailable protocol hash")
    source_archive_hash = provenance.get("source_archive_sha256")
    if not isinstance(source_archive_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_archive_hash
    ):
        raise RuntimeError("smoke profile source archive hash is invalid")

    profile_models = public["models"]
    for model in MODELS:
        profile = profile_models[model.id]
        observed = profile["output_tokens"]["max"]
        if model.smoke_observed_max_output_tokens != observed:
            raise RuntimeError(f"observed maximum mismatch for {model.id}")
        if model.request_output_tokens != profile["request_output_tokens"]:
            raise RuntimeError(f"request ceiling mismatch for {model.id}")
        if model.request_output_tokens < observed * 1.25:
            raise RuntimeError(f"request ceiling lacks 25% headroom for {model.id}")
        if model.request_output_tokens % 256:
            raise RuntimeError(f"request ceiling is not a 256-token boundary for {model.id}")
        if model.request_output_tokens > model.provider_output_capability:
            raise RuntimeError(f"request ceiling exceeds provider capability for {model.id}")


def _metadata_and_sources() -> None:
    from instruction_duplication import __version__

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if project["project"].get("dynamic") != ["version"]:
        raise RuntimeError("package version must have one dynamic source")
    if project["project"].get("requires-python") != ">=3.12.13":
        raise RuntimeError("the package must require Python 3.12.13 or newer")
    if project["tool"]["ruff"].get("target-version") != "py312":
        raise RuntimeError("Ruff must target Python 3.12")
    if project["tool"]["mypy"].get("python_version") != "3.12":
        raise RuntimeError("mypy must target Python 3.12")
    if project["tool"]["mypy"].get("strict") is not True:
        raise RuntimeError("mypy strict mode must remain enabled")
    if project["tool"]["pyright"].get("pythonVersion") != "3.12":
        raise RuntimeError("Pyright must target Python 3.12")
    if project["tool"]["pyright"].get("typeCheckingMode") != "strict":
        raise RuntimeError("Pyright strict mode must remain enabled")
    if not __version__:
        raise RuntimeError("package version must be non-empty")
    for forbidden in (
        "PROTOCOL.txt",
        "setup.cfg",
        "PKG-INFO",
    ):
        if (ROOT / forbidden).exists():
            raise RuntimeError(f"source tree contains a packaging by-product: {forbidden}")
    if (SOURCE_ROOT / "instruction_duplication.egg-info").exists():
        raise RuntimeError("egg-info must not be committed to the source tree")
    if not (SOURCE / "protocol.py").exists():
        raise RuntimeError("protocol source is missing")


def _locks_and_actions() -> None:
    for name in ("requirements-research.lock", "requirements-dev.lock"):
        lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
        requirements = [line for line in lines if line and not line.startswith("#")]
        if not requirements or any("==" not in line for line in requirements):
            raise RuntimeError(f"{name} contains an unpinned direct requirement")
    workflow = (ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    action_refs = re.findall(r"uses:\s*[^@\s]+@([^\s]+)", workflow)
    if not action_refs or any(not re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs):
        raise RuntimeError("GitHub Actions must be pinned to exact 40-character SHAs")


def main() -> int:
    from instruction_duplication import __version__

    _smoke_profile()
    _metadata_and_sources()
    _locks_and_actions()
    print(f"release_invariants=ok smoke_cells=1680 version={__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
