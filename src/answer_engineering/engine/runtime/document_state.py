"""Immutable runtime document snapshot model.

Purpose:
    Represent the authoritative runtime document snapshot text and version
    lineage passed between stages.

Architectural role:
    Core runtime snapshot contract passed across orchestration stages as the
    source of truth for one document version.

Inputs:
    Created by engine entry/orchestration startup and replaced with new
    snapshots after successful patch application in other modules.

Outputs:
    Consumed by proposal, scoring, apply, and telemetry stages to validate
    version consistency and produce deterministic traces.

"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def _compute_version_id(text: str) -> str:
    """Compute the short stable runtime version fingerprint used to track."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True, init=False)
class DocumentState:
    """Immutable document snapshot with version lineage metadata.

    Purpose:
        Hold text plus stable version identifier and history for one immutable
        runtime snapshot state.

    Architectural role:
        Data-plane snapshot contract shared between orchestration queue messages
        and patching/runtime services.

    Inputs:
        Constructed from prior ``DocumentState`` and applied patch outcomes, or
        initialized from engine entry text.

    Outputs:
        Passed through orchestrator messages and telemetry events as canonical
        source for text/version identity.

    """

    text: str
    version_id: str
    history: tuple[str, ...] = field(default_factory=tuple)

    def __init__(
        self,
        text: str,
        version_id: str | None = None,
        history: tuple[str, ...] = tuple(),
    ) -> None:
        """Build an immutable document snapshot and derive ``version_id`` when.

        Purpose:
            Normalize the text, version identifier, and lineage tuple stored on
            the authoritative runtime snapshot shared by proposal, scoring, and
            apply code.

        Architectural role:
            Constructor for the core runtime value object representing one
            document version.

        Inputs:
            ``text`` is the authoritative document body; ``version_id`` can be
            injected by callers that already computed or persisted it;
            ``history`` carries prior version ids.

        Invariants:
            ``version_id`` always corresponds to ``text`` when callers do not
            override it explicitly.

        Non-ownership:
            Mutation algorithms and patch-application ordering occur outside
            this module; this type only models snapshots.

        """
        object.__setattr__(self, "text", text)
        object.__setattr__(
            self, "version_id", version_id or _compute_version_id(text)
        )
        object.__setattr__(self, "history", history)
