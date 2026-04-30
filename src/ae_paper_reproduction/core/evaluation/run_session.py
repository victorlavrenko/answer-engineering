"""Define identifiers and comparison helpers for a single evaluation run.

Purpose:
    Materialize run-level and subrun-level identity values and compute case-type
    comparison rows between two accuracy reports.

Architectural role:
    Small support module between evaluation reports and run summarization.

Inputs (architectural provenance):
    Consumes timestamps, run tags, and accuracy reports produced during
    experiment execution.

Outputs (downstream usage):
    Stable run identifiers and case-type comparison rows consumed by planning
    and summary code.

Invariants/constraints:
    Identifiers should be reproducible from their inputs, and comparison rows
    should only reflect already-computed reports.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RunSession:
    """Store run-level identity inputs for a reproduction execution.

    Purpose:
        Keep the timestamp and optional tag needed to derive the stable run id
        shared by all subruns in one session.

    Architectural role:
        Lightweight run-identity value object between planning and summary code.

    Inputs (architectural provenance):
        Constructed at the start of one reproduction run.

    Outputs (downstream usage):
        Run-level identity values consumed by subrun naming and summary
        builders.

    Invariants/constraints:
        The same `RunSession` should be reused for all subruns in one run if a
        shared run id is desired.

    """

    now: datetime
    run_tag: str | None

    @property
    def run_id(self) -> str:
        """Build the stable run identifier for this reproduction session.

        Purpose:
            Combine the stored timestamp and optional tag into the canonical run
            id used by subrun sessions and reporting artifacts.

        Architectural role:
            Derived identifier accessor on the run-session value object.

        Inputs (architectural provenance):
            Reads the stored session timestamp and optional tag.

        Outputs (downstream usage):
            A run id string consumed by subrun naming and summaries.

        Invariants/constraints:
            The returned id must remain stable for the lifetime of this run
            session.

        """
        stamp = self.now.strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}-{self.run_tag}" if self.run_tag else stamp


@dataclass(frozen=True, slots=True)
class SubrunSession:
    """Store the naming inputs for one subrun within a run session.

    Purpose:
        Pair the run session identity with a subrun index and ruleset name so
        the canonical subrun id can be derived consistently.

    Architectural role:
        Subrun-level identity value object used during planning and reporting.

    Inputs (architectural provenance):
        Constructed from one run session together with subrun-specific naming
        values.

    Outputs (downstream usage):
        Subrun identity data consumed by planning and summaries.

    Invariants/constraints:
        The stored index and ruleset name should match the executed subrun being
        described.

    """

    index: int
    ruleset_name: str

    @property
    def subrun_id(self) -> str:
        """Build the canonical identifier for this subrun session.

        Purpose:
            Combine the parent run id, subrun index, and ruleset name into the
            stable subrun id used across results, telemetry, and artifacts.

        Architectural role:
            Derived identifier accessor on the subrun-session value object.

        Inputs (architectural provenance):
            Reads the stored run-session, index, and ruleset name.

        Outputs (downstream usage):
            A subrun id string consumed across the reproduction pipeline.

        Invariants/constraints:
            The returned id must be stable and unique within its parent run.

        """
        slug = "".join(
            ch.lower() if ch.isalnum() else "-" for ch in self.ruleset_name
        ).strip("-")
        while "--" in slug:
            slug = slug.replace("--", "-")
        if not slug:
            slug = f"ruleset-{self.index:03d}"
        return f"{self.index:03d}-{slug}"
