"""Deterministic runtime-environment provenance for research workspaces."""

from __future__ import annotations

import platform
import sys
from importlib import metadata

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .json_types import JsonObject

ROOT_DISTRIBUTIONS = ("instruction-duplication", "httpx", "datasets", "huggingface-hub")


def _active_requirements(distribution: metadata.Distribution) -> list[str]:
    names: list[str] = []
    for raw in distribution.requires or ():
        requirement = Requirement(raw)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            names.append(str(canonicalize_name(requirement.name)))
    return names


def runtime_environment() -> JsonObject:
    """Return the exact active dependency closure and interpreter identity."""
    pending: list[str] = [str(canonicalize_name(name)) for name in ROOT_DISTRIBUTIONS]
    seen: set[str] = set()
    versions: dict[str, str | None] = {}
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
            continue
        versions[name] = distribution.version
        pending.extend(dep for dep in _active_requirements(distribution) if dep not in seen)
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "distributions": dict(sorted(versions.items())),
    }
