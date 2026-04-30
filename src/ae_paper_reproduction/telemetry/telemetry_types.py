"""Typed telemetry transport models and serialization helpers.

Purpose:
    Define typed rows, contexts, manifests, and serializers that bridge runtime
    telemetry and reproduction reporting/publication surfaces.

Architectural role:
    Typed transport-contract boundary between runtime telemetry outputs and
    reproduction artifact/report pipelines.

Architectural direction:
    Keep transport contracts stable while making serialization and publication
    concerns easier to reason about independently.

Why this matters:
    This file currently combines row/context transport contracts with several
    JSON serialization transforms used by reporting and publication paths,
    creating a high-concentration but functional boundary.

What better would look like:
    Downstream reporting can rely on stable row/context contracts without
    requiring broad schema/serialization coupling in one place.

How improvement can be recognized:
    - Stable transport types with clearer serialization ownership
    - Fewer unrelated schema and formatting changes coupled together
    - Easier explanation of runtime-to-artifact data transformations

Open constraint:
    Schema modeling must remain aligned with actual downstream reporting needs.

"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Protocol

from huggingface_hub import CommitOperationAdd

from ae_paper_reproduction.core.aggregation.comparison_results import (
    GroupComparisonRow,
)
from ae_paper_reproduction.core.aggregation.rule_stats import (
    AggregatedRunStats,
)
from ae_paper_reproduction.core.evaluation.reports import (
    RulesetEvaluationResult,
    RunOutcomeTransitions,
)
from ae_paper_reproduction.core.planning.notebook_extractor import (
    GenerationMode,
    PaperRole,
)
from ae_paper_reproduction.core.planning.subruns import SubrunResult
from answer_engineering.telemetry import (
    CandidateTelemetrySnapshot as RuntimeCandidateTelemetrySnapshot,
)
from answer_engineering.telemetry import (
    ConditionTelemetrySnapshot as RuntimeConditionTelemetrySnapshot,
)
from answer_engineering.telemetry import (
    RuleTelemetrySnapshot as RuntimeRuleTelemetrySnapshot,
)
from answer_engineering.telemetry import (
    RuntimeTelemetrySnapshot,
)

SCHEMA_VERSION = "1"
type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
type RowValue = JsonValue
type TelemetryRow = dict[str, RowValue]
type TelemetryRows = list[TelemetryRow]


def _serialize_condition(
    condition: RuntimeConditionTelemetrySnapshot,
) -> JsonValue:
    """Serialize one condition snapshot into a JSON-safe mapping.

    Purpose:
        Carry out the specific telemetry types transformation or helper step
        represented by this function while keeping the surrounding boundary code
        small and predictable.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    """
    return {
        "condition_id": condition.condition_id,
        "node_path": condition.node_path,
        "node_type": condition.node_type,
        "debug_expression": condition.debug_expression,
        "matched": condition.matched,
        "seen": condition.seen,
    }


def _serialize_candidate(
    candidate: RuntimeCandidateTelemetrySnapshot,
) -> JsonValue:
    """Serialize one candidate snapshot into a JSON-safe mapping.

    Purpose:
        Carry out the specific telemetry types transformation or helper step
        represented by this function while keeping the surrounding boundary code
        small and predictable.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    """
    return {
        "kind": candidate.kind,
        "candidate_id": candidate.candidate_id,
        "label": candidate.label,
        "chosen": candidate.chosen,
    }


def _serialize_rule(rule: RuntimeRuleTelemetrySnapshot) -> JsonValue:
    """Serialize one rule telemetry snapshot into a JSON-safe mapping.

    Purpose:
        Carry out the specific telemetry types transformation or helper step
        represented by this function while keeping the surrounding boundary code
        small and predictable.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    """
    return {
        "rule_id": rule.rule_id,
        "rule_name": rule.rule_name,
        "evaluations": rule.evaluations,
        "applied": rule.applied,
        "trigger_firings": rule.trigger_firings,
        "proposals_generated": rule.proposals_generated,
        "generated_candidates_considered": rule.generated_candidates_considered,
        "fallback_candidates_considered": rule.fallback_candidates_considered,
        "static_candidates_considered": rule.static_candidates_considered,
        "noop_candidates_generated": rule.noop_candidates_generated,
        "conditions": {
            condition.condition_id: _serialize_condition(condition)
            for condition in rule.conditions
        },
        "candidate_choices": {
            f"{candidate.kind}:{candidate.candidate_id}": _serialize_candidate(
                candidate
            )
            for candidate in rule.candidate_choices
        },
    }


def serialize_runtime_telemetry(
    telemetry: RuntimeTelemetrySnapshot | None,
) -> dict[str, JsonValue] | None:
    """Convert a runtime telemetry snapshot into the JSON-safe artifact payload.

    Purpose:
        Flatten runtime events, counters, applied-decision metadata, and
        optional per-rule telemetry into the dictionary form stored in answer
        rows and published artifacts.

    Outputs:
        A JSON-safe mapping, or ``None`` when no runtime telemetry was captured.

    Ownership:
        Owned by
        ``answer_engineering.telemetry.representation.telemetry_types``.

    """
    if telemetry is None:
        return None
    payload: dict[str, JsonValue] = {
        "events": [event.serialize() for event in telemetry.events],
    }
    if telemetry.runtime_sec is not None:
        payload["runtime_sec"] = telemetry.runtime_sec
    payload["rules_triggered_count"] = telemetry.rules_triggered_count
    payload["rules_applied_count"] = telemetry.rules_applied_count
    payload["applied_decisions"] = telemetry.applied_decisions
    payload["decision_limit_reached"] = telemetry.decision_limit_reached
    if telemetry.rules:
        payload["rules"] = {
            rule.rule_id: _serialize_rule(rule) for rule in telemetry.rules
        }
    return payload


@dataclass(frozen=True, slots=True, init=False)
class AnswerTelemetryRow:
    """Immutable answer-level row used by telemetry reports and artifact.

    Purpose:
        Hold the question, gold label, baseline/edited outcomes, runtimes, and
        optional runtime telemetry for one evaluated answer in a stable field
        layout shared by row serializers and report builders.

    Invariants:
        Instances are frozen transport values and should not be mutated after
        construction.

    Ownership:
        Owned by
        ``answer_engineering.telemetry.representation.telemetry_types``.

    """

    run_id: str | None
    group_run_id: str | None
    subrun_id: str | None
    id: str
    case_type: str
    question: str
    gold: str
    baseline_ok: bool | None
    edited_ok: bool
    baseline_answer: str | None
    edited_answer: str
    baseline_runtime_sec: float | None
    edited_runtime_sec: float | None
    ae_telemetry: RuntimeTelemetrySnapshot | None

    def __init__(
        self, *, ctx: SubrunContext, result: RulesetEvaluationResult
    ) -> None:
        """Build an ``AnswerTelemetryRow`` from one subrun evaluation result.

        Purpose:
            Copy the identifiers, answer fields, correctness flags, runtime, and
            captured answer-engineering telemetry from a subrun result into the
            row shape used by downstream reporting.

        """
        object.__setattr__(self, "run_id", None)
        object.__setattr__(self, "group_run_id", ctx.group_run_id)
        object.__setattr__(self, "subrun_id", ctx.subrun_id)
        object.__setattr__(self, "id", str(result.id or ""))
        object.__setattr__(self, "case_type", str(result.case_type or ""))
        object.__setattr__(self, "question", str(result.question or ""))
        object.__setattr__(self, "gold", str(result.gold or ""))
        object.__setattr__(self, "baseline_ok", None)
        object.__setattr__(self, "edited_ok", bool(result.ok))
        object.__setattr__(self, "baseline_answer", None)
        object.__setattr__(self, "edited_answer", str(result.answer or ""))
        object.__setattr__(self, "baseline_runtime_sec", None)
        object.__setattr__(self, "edited_runtime_sec", result.runtime_sec)
        object.__setattr__(self, "ae_telemetry", result.ae_telemetry)

    def to_serialized_row(self) -> TelemetryRow:
        """Convert this row into the flat payload stored in telemetry artifacts.

        Purpose:
            Serialize ``ae_telemetry`` through ``serialize_runtime_telemetry``
            while leaving the remaining dataclass fields in their exported row
            names.

        """
        row: TelemetryRow = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "ae_telemetry"
        }
        row["ae_telemetry"] = serialize_runtime_telemetry(self.ae_telemetry)
        return row


type AnswerTelemetryRows = Sequence[AnswerTelemetryRow]


class ArtifactHubApi(Protocol):
    """Minimal artifact-backend protocol used by telemetry publication.

    Purpose:
        Define just the repository and commit operations that the telemetry
        representation layer needs so publication code can work with a concrete
        Hugging Face client or a test double.

    Ownership:
        Owned by
        ``answer_engineering.telemetry.representation.telemetry_types``.

    """

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        private: bool,
        token: str,
        exist_ok: bool,
    ) -> object:
        """Create or ensure the remote artifact repository.

        Purpose:
            Carry out the specific telemetry types transformation or helper step
            represented by this function while keeping the surrounding boundary
            code small and predictable.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        raise NotImplementedError

    def create_commit(
        self,
        repo_id: str,
        operations: Iterable[CommitOperationAdd],
        *,
        commit_message: str,
        token: str,
        repo_type: str,
    ) -> object:
        """Create the remote commit that publishes telemetry artifacts.

        Purpose:
            Carry out the specific telemetry types transformation or helper step
            represented by this function while keeping the surrounding boundary
            code small and predictable.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        raise NotImplementedError

    def upload_file(
        self,
        *,
        path_or_fileobj: str | bytes,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        token: str,
    ) -> object:
        """Upload one file into the remote artifact repository.

        Purpose:
            Push the generated artifact files and metadata through the
            configured publication backend.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable row model for run-level metadata shared by telemetry artifacts.

    Purpose:
        Bundle the contextual identifiers, paths, and aggregate values that
        multiple reporting or upload steps need to share consistently.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    """

    run_id: str
    created_at_utc: str
    code_commit_sha: str
    dataset_id: str
    split: str
    case_type_filter: str | None
    n_eval_requested: int
    n_eval_actual: int
    model_id: str
    max_new_tokens: int
    compute_baseline: bool
    baseline_accuracy: float
    edited_accuracy: float
    delta_accuracy: float
    applied_decisions_total: int
    decision_limit_reached: bool
    rules_triggered_count: int
    rules_applied_count: int
    run_tag: str | None
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True, init=False)
class ArtifactManifest:
    """Immutable row model for the paths of generated artifacts for one.

    Purpose:
        Describe the generated artifact set in one serializable object so
        publication code can reason about bundle contents explicitly.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    """

    report_md_path: str
    rules_original_md_path: str
    rules_with_stats_md_path: str

    def __init__(
        self,
        artifacts: EvaluationArtifactFiles | None = None,
        *,
        report_md_path: str | None = None,
        rules_original_md_path: str | None = None,
        rules_with_stats_md_path: str | None = None,
    ) -> None:
        """Build the artifact manifest from canonical artifact file paths.

        Purpose:
            Normalize the set of generated artifact paths into the manifest
            object used for publishing and reproducibility checks.

        Architectural role:
            Telemetry value constructor at the boundary between local files and
            remote artifact metadata.

        Inputs (architectural provenance):
            Receives canonical paths produced by reporting and
            artifact-generation workflows.

        Outputs (downstream usage):
            Stores manifest fields consumed by upload code, JSON serialization,
            and paper reproduction records.

        Invariants/constraints:
            Paths should represent already-known artifacts. The constructor
            should normalize metadata but not generate or upload files.

        """
        if artifacts is None:
            if (
                report_md_path is None
                or rules_original_md_path is None
                or rules_with_stats_md_path is None
            ):
                raise TypeError(
                    "ArtifactManifest requires artifacts or explicit paths"
                )
            object.__setattr__(self, "report_md_path", report_md_path)
            object.__setattr__(
                self, "rules_original_md_path", rules_original_md_path
            )
            object.__setattr__(
                self, "rules_with_stats_md_path", rules_with_stats_md_path
            )
            return
        object.__setattr__(self, "report_md_path", str(artifacts.run_report_md))
        object.__setattr__(
            self, "rules_original_md_path", str(artifacts.rules_original_md)
        )
        object.__setattr__(
            self,
            "rules_with_stats_md_path",
            str(artifacts.rules_with_stats_md),
        )


@dataclass(frozen=True, slots=True)
class RunCaseTypeStatsRow:
    """Immutable row model for run-level case-type summary statistics.

    Purpose:
        Carry one already-shaped reporting row so downstream serializers and
        table renderers can operate on stable field names.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    Invariants:
        Instances are transport-oriented values and should remain stable after
        construction.

    """

    case_type: str
    n_cases: int
    baseline_accuracy: float
    edited_accuracy: float
    delta_accuracy: float

    def to_row(self) -> TelemetryRow:
        """Convert the run-level case-type stats row into a flat mapping.

        Purpose:
            Project the current value into the row or snapshot form consumed by
            downstream reporting code.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SubrunCaseTypeStatsRow:
    """Immutable row model for subrun-level case-type summary statistics.

    Purpose:
        Carry one already-shaped reporting row so downstream serializers and
        table renderers can operate on stable field names.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    Invariants:
        Instances are transport-oriented values and should remain stable after
        construction.

    """

    case_type: str
    n_cases: int
    accuracy: float
    delta_accuracy_vs_anchor: float

    def to_row(self) -> TelemetryRow:
        """Convert the subrun-level case-type stats row into a flat mapping.

        Purpose:
            Project the current value into the row or snapshot form consumed by
            downstream reporting code.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuleStatsRow:
    """Immutable row model for per-rule statistics in telemetry reports.

    Purpose:
        Carry one already-shaped reporting row so downstream serializers and
        table renderers can operate on stable field names.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    Invariants:
        Instances are transport-oriented values and should remain stable after
        construction.

    """

    run_id: str | None
    group_run_id: str | None
    subrun_id: str | None
    rule_id: str
    rule_name: str
    evaluations: int
    applied: int
    applied_rate: float
    candidate_kind: str
    candidate_id: str
    candidate_label: str
    candidate_chosen: int
    candidate_chosen_rate: float
    condition_section: str
    condition_id: str
    condition_operator: str
    condition_expression: str
    condition_matched: int
    condition_seen: int
    condition_match_rate: float

    def to_row(self) -> TelemetryRow:
        """Convert the per-rule stats row into a flat mapping.

        Purpose:
            Project the current value into the row or snapshot form consumed by
            downstream reporting code.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SubrunCaseStatsRow:
    """Immutable row model for per-case statistics in a subrun report.

    Purpose:
        Carry one already-shaped reporting row so downstream serializers and
        table renderers can operate on stable field names.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    Invariants:
        Instances are transport-oriented values and should remain stable after
        construction.

    """

    group_run_id: str
    subrun_id: str
    case_type: str
    n_cases: int
    accuracy: float
    delta_accuracy_vs_anchor: float

    def to_row(self) -> TelemetryRow:
        """Convert the per-case stats row into a flat mapping.

        Purpose:
            Project the current value into the row or snapshot form consumed by
            downstream reporting code.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return asdict(self)


@dataclass(frozen=True, slots=True, init=False)
class SubrunSummaryRow:
    """Immutable row model for one subrun summary row.

    Purpose:
        Carry one already-shaped reporting row so downstream serializers and
        table renderers can operate on stable field names.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    Invariants:
        Instances are transport-oriented values and should remain stable after
        construction.

    """

    group_run_id: str
    subrun_id: str
    ruleset_name: str
    system_prompt: str
    created_at_utc: str
    code_commit_sha: str
    dataset_id: str
    split: str
    case_type_filter: str | None
    n_eval_requested: int
    n_eval_actual: int
    model_id: str
    max_new_tokens: int
    accuracy: float
    anchor_subrun_id: str | None
    anchor_accuracy: float | None
    delta_accuracy: float | None
    applied_decisions_total: int
    decision_limit_reached: bool
    rules_triggered_count: int
    rules_applied_count: int
    run_tag: str | None
    schema_version: str
    report_md_path: str
    rules_original_md_path: str
    rules_with_stats_md_path: str

    def __init__(self, subrun: SubrunTelemetry) -> None:
        """Build a subrun summary row from the richer `SubrunTelemetry` object.

        Purpose:
            Offer a convenience construction path that derives this value
            directly from upstream runtime or evaluation inputs.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        values = {
            **asdict(subrun.ctx),
            **asdict(subrun.artifact_manifest),
        }
        for field in fields(self):
            object.__setattr__(self, field.name, values[field.name])

    def to_row(self) -> TelemetryRow:
        """Convert the subrun summary row into a flat mapping.

        Purpose:
            Project the current value into the row or snapshot form consumed by
            downstream reporting code.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return asdict(self)


@dataclass(frozen=True, slots=True, init=False)
class GroupSummaryRow:
    """Immutable row model for one group summary row.

    Purpose:
        Carry one already-shaped reporting row so downstream serializers and
        table renderers can operate on stable field names.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    Invariants:
        Instances are transport-oriented values and should remain stable after
        construction.

    """

    group_run_id: str
    created_at_utc: str
    code_commit_sha: str
    dataset_id: str
    split: str
    case_type_filter: str | None
    n_eval_requested: int
    n_eval_actual: int
    model_id: str
    max_new_tokens: int
    run_tag: str | None
    schema_version: str
    group_report_md_path: str

    def __init__(
        self,
        context: GroupRunContext | None = None,
        *,
        group_report_md_path: str,
        group_run_id: str | None = None,
        created_at_utc: str | None = None,
        code_commit_sha: str | None = None,
        dataset_id: str | None = None,
        split: str | None = None,
        case_type_filter: str | None = None,
        n_eval_requested: int | None = None,
        n_eval_actual: int | None = None,
        model_id: str | None = None,
        max_new_tokens: int | None = None,
        run_tag: str | None = None,
        schema_version: str | None = None,
    ) -> None:
        """Build a group summary row from canonical group context inputs.

        Purpose:
            Normalize one grouped reporting row for telemetry summaries and
            paper artifact tables.

        Architectural role:
            Report-row constructor between aggregation data and serialized
            telemetry output.

        Inputs (architectural provenance):
            Receives group identifiers, counts, scores, and contextual metadata
            from the aggregation/reporting pipeline.

        Outputs (downstream usage):
            Stores a row consumed by JSON, markdown, and TeX renderers.

        Invariants/constraints:
            Derived row fields should be computed once during construction so
            renderers can remain presentation-only.

        """
        if context is not None:
            for key, value in asdict(context).items():
                object.__setattr__(self, key, value)
        else:
            required_values = {
                "group_run_id": group_run_id,
                "created_at_utc": created_at_utc,
                "code_commit_sha": code_commit_sha,
                "dataset_id": dataset_id,
                "split": split,
                "n_eval_requested": n_eval_requested,
                "n_eval_actual": n_eval_actual,
                "model_id": model_id,
                "max_new_tokens": max_new_tokens,
                "schema_version": schema_version,
            }
            if any(value is None for value in required_values.values()):
                raise TypeError(
                    "GroupSummaryRow requires context or explicit fields"
                )
            for key, value in required_values.items():
                object.__setattr__(self, key, value)
            object.__setattr__(self, "case_type_filter", case_type_filter)
            object.__setattr__(self, "run_tag", run_tag)
        object.__setattr__(self, "group_report_md_path", group_report_md_path)

    def to_row(self) -> TelemetryRow:
        """Convert the group summary row into a flat mapping.

        Purpose:
            Project the current value into the row or snapshot form consumed by
            downstream reporting code.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunTelemetry:
    """Immutable row model for the full telemetry bundle for one run.

    Purpose:
        Group the telemetry records that belong to one reporting scope so
        downstream code can move them as one coherent unit.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    """

    ctx: RunContext
    artifact_manifest: ArtifactManifest
    rule_rows: Sequence[RuleStatsRow]
    case_rows: Sequence[RunCaseTypeStatsRow]
    outcome_transitions: RunOutcomeTransitions
    answer_rows: Sequence[AnswerTelemetryRow]


@dataclass(frozen=True, slots=True, init=False)
class SubrunTelemetry:
    """Immutable telemetry bundle for one evaluated subrun.

    Purpose:
        Keep the subrun context, generated artifact manifest, rule rows,
        case-level rows, and answer-level rows together so publication and
        summary code can treat one subrun as a single reporting unit.

    Ownership:
        Owned by
        ``answer_engineering.telemetry.representation.telemetry_types``.

    """

    ctx: SubrunContext
    artifact_manifest: ArtifactManifest
    rule_rows: Sequence[RuleStatsRow]
    case_rows: Sequence[SubrunCaseStatsRow]
    answer_rows: Sequence[AnswerTelemetryRow]

    def __init__(
        self,
        *,
        ctx: SubrunContext,
        run_stats: AggregatedRunStats,
        case_type_stats_rows: Sequence[SubrunCaseTypeStatsRow],
        artifact_manifest: ArtifactManifest,
        eval_results: Iterable[RulesetEvaluationResult],
    ) -> None:
        """Build the subrun telemetry bundle from context, evaluation rows, and.

        Purpose:
            Derive rule rows, case rows, and answer rows from upstream subrun
            context and stats so callers construct this bundle through the
            truthful production inputs.

        """
        object.__setattr__(self, "ctx", ctx)
        object.__setattr__(self, "artifact_manifest", artifact_manifest)
        object.__setattr__(
            self,
            "rule_rows",
            tuple(_build_rule_rows(ctx=ctx, run_stats=run_stats)),
        )
        object.__setattr__(
            self,
            "case_rows",
            tuple(
                SubrunCaseStatsRow(
                    group_run_id=ctx.group_run_id,
                    subrun_id=ctx.subrun_id,
                    case_type=row.case_type,
                    n_cases=row.n_cases,
                    accuracy=row.accuracy,
                    delta_accuracy_vs_anchor=row.delta_accuracy_vs_anchor,
                )
                for row in case_type_stats_rows
            ),
        )
        object.__setattr__(
            self,
            "answer_rows",
            _build_subrun_answer_rows(ctx=ctx, eval_results=eval_results),
        )

    @property
    def subrun_id(self) -> str:
        """Return the stable subrun id stored in this telemetry bundle.

        Purpose:
            Carry out the specific telemetry types transformation or helper step
            represented by this function while keeping the surrounding boundary
            code small and predictable.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return self.ctx.subrun_id

    @property
    def ruleset_name(self) -> str:
        """Return the ruleset name associated with this subrun telemetry bundle.

        Purpose:
            Carry out the specific telemetry types transformation or helper step
            represented by this function while keeping the surrounding boundary
            code small and predictable.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return self.ctx.ruleset_name

    @property
    def case_type_filter(self) -> str | None:
        """Return the case-type filter associated with this subrun telemetry.

        Purpose:
            Carry out the specific telemetry types transformation or helper step
            represented by this function while keeping the surrounding boundary
            code small and predictable.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return self.ctx.case_type_filter

    @property
    def mode(self) -> GenerationMode:
        """Return the explicit generation mode for this subrun telemetry."""
        return self.ctx.mode

    @property
    def paper_role(self) -> PaperRole | None:
        """Return paper reporting role metadata for this subrun telemetry."""
        return self.ctx.paper_role

    @property
    def paper_variant(self) -> str | None:
        """Return paper reporting variant metadata for this subrun telemetry."""
        return self.ctx.paper_variant

    @property
    def group_run_id(self) -> str:
        """Return the parent group-run id associated with this subrun.

        Purpose:
            Carry out the specific telemetry types transformation or helper step
            represented by this function while keeping the surrounding boundary
            code small and predictable.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return self.ctx.group_run_id

    @property
    def accuracy(self) -> float:
        """Return the edited accuracy stored in this subrun telemetry bundle.

        Purpose:
            Carry out the specific telemetry types transformation or helper step
            represented by this function while keeping the surrounding boundary
            code small and predictable.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return self.ctx.accuracy

    @property
    def delta_accuracy(self) -> float:
        """Return the delta accuracy stored in this subrun telemetry bundle.

        Purpose:
            Carry out the specific telemetry types transformation or helper step
            represented by this function while keeping the surrounding boundary
            code small and predictable.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        if self.ctx.delta_accuracy is None:
            return 0.0
        return self.ctx.delta_accuracy

    @property
    def report_md_path(self) -> str:
        """Return the markdown report path for this subrun telemetry bundle.

        Purpose:
            Carry out the specific telemetry types transformation or helper step
            represented by this function while keeping the surrounding boundary
            code small and predictable.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return self.artifact_manifest.report_md_path

    def to_row(self) -> TelemetryRow:
        """Convert the subrun telemetry bundle into its summary row form.

        Purpose:
            Project the current value into the row or snapshot form consumed by
            downstream reporting code.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return SubrunSummaryRow(self).to_row()


def _build_rule_rows(
    *, ctx: RunContext | SubrunContext, run_stats: AggregatedRunStats
) -> list[RuleStatsRow]:
    """Build per-rule reporting rows from aggregated runtime statistics.

    Purpose:
        Project aggregated rule counters and rates into the ``RuleStatsRow``
        objects consumed by report renderers and artifact serializers.

    """
    rule_rows_value: list[RuleStatsRow] = []
    for rule in run_stats.rules:
        for condition in rule.conditions or (None,):
            for candidate in rule.candidate_choices or (None,):
                condition_seen = (
                    int(condition.seen) if condition is not None else 0
                )
                condition_matched = (
                    int(condition.matched) if condition is not None else 0
                )
                condition_rate = (
                    (condition_matched / condition_seen)
                    if condition_seen
                    else 0.0
                )
                candidate_chosen = (
                    int(candidate.chosen) if candidate is not None else 0
                )
                candidate_rate = (
                    (candidate_chosen / int(rule.evaluations))
                    if int(rule.evaluations)
                    else 0.0
                )
                rule_rows_value.append(
                    RuleStatsRow(
                        run_id=ctx.run_id
                        if isinstance(ctx, RunContext)
                        else None,
                        group_run_id=(
                            None
                            if isinstance(ctx, RunContext)
                            else ctx.group_run_id
                        ),
                        subrun_id=(
                            None
                            if isinstance(ctx, RunContext)
                            else ctx.subrun_id
                        ),
                        rule_id=rule.rule_id,
                        rule_name=rule.rule_name,
                        evaluations=rule.evaluations,
                        applied=rule.applied,
                        applied_rate=(
                            (rule.applied / rule.evaluations)
                            if rule.evaluations
                            else 0.0
                        ),
                        candidate_kind=(
                            candidate.kind if candidate is not None else ""
                        ),
                        candidate_id=(
                            candidate.candidate_id
                            if candidate is not None
                            else ""
                        ),
                        candidate_label=(
                            candidate.label if candidate is not None else ""
                        ),
                        candidate_chosen=candidate_chosen,
                        candidate_chosen_rate=candidate_rate,
                        condition_section=(
                            condition.node_path if condition is not None else ""
                        ),
                        condition_id=(
                            condition.condition_id
                            if condition is not None
                            else ""
                        ),
                        condition_operator=(
                            condition.node_type if condition is not None else ""
                        ),
                        condition_expression=(
                            condition.debug_expression
                            if condition is not None
                            else ""
                        ),
                        condition_matched=condition_matched,
                        condition_seen=condition_seen,
                        condition_match_rate=condition_rate,
                    )
                )
    return rule_rows_value


def _build_subrun_answer_rows(
    *,
    ctx: SubrunContext,
    eval_results: Iterable[RulesetEvaluationResult],
) -> AnswerTelemetryRows:
    """Build the answer-level telemetry rows for one subrun.

    Purpose:
        Materialize one ``AnswerTelemetryRow`` per evaluation result while
        preserving the surrounding subrun context needed by downstream reports.

    """
    answer_rows_value: list[AnswerTelemetryRow] = []
    for result in eval_results:
        answer_rows_value.append(AnswerTelemetryRow(ctx=ctx, result=result))
    return tuple(answer_rows_value)


@dataclass(frozen=True, slots=True)
class EvaluationArtifactFiles:
    """Paths to the files generated for one evaluated run or subrun.

    Purpose:
        Keep the markdown report, serialized rows, and related local artifact
        paths together so publication code can build manifests without
        reconstructing filenames in multiple places.

    Ownership:
        Owned by
        ``answer_engineering.telemetry.representation.telemetry_types``.

    """

    run_report_md: Path
    rules_original_md: Path
    rules_with_stats_md: Path
    run_summary_json: Path
    answers_json: Path

    def upload_files(self) -> dict[str, Path]:
        """Upload the evaluated subrun's generated files to the artifact store.

        Purpose:
            Push the generated artifact files and metadata through the
            configured publication backend.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return {
            "run_report.md": self.run_report_md,
            "rules_original.md": self.rules_original_md,
            "rules_with_stats.md": self.rules_with_stats_md,
            "run_summary.json": self.run_summary_json,
            "answers.json": self.answers_json,
        }


@dataclass(frozen=True, slots=True)
class GroupArtifactFiles:
    """Paths to the files generated for one grouped telemetry report.

    Purpose:
        Hold the group-level markdown report, summary tables, paper TeX
        fragments, and related outputs that are published together for a grouped
        evaluation run.

    Ownership:
        Owned by
        ``answer_engineering.telemetry.representation.telemetry_types``.

    """

    group_report_md: Path
    group_summary_json: Path
    subruns_json: Path
    comparisons_json: Path
    paper_metrics_json: Path
    generated_tex_dir: Path

    def upload_files(self) -> dict[str, Path]:
        """Upload the group's generated files to the artifact store.

        Purpose:
            Push the generated artifact files and metadata through the
            configured publication backend.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return {
            "group_report.md": self.group_report_md,
            "group_summary.json": self.group_summary_json,
            "subruns.json": self.subruns_json,
            "comparisons.json": self.comparisons_json,
            "paper_metrics.json": self.paper_metrics_json,
        }

    def generated_files(self) -> dict[str, Path]:
        """Return the generated files that belong to this group artifact bundle.

        Purpose:
            Carry out the specific telemetry types transformation or helper step
            represented by this function while keeping the surrounding boundary
            code small and predictable.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        return {
            "paper-metrics.tex": self.generated_tex_dir / "paper-metrics.tex",
        }


@dataclass(frozen=True, slots=True, init=False)
class GroupRunContext:
    """Immutable row model for group-level context derived from many subresults.

    Purpose:
        Bundle the contextual identifiers, paths, and aggregate values that
        multiple reporting or upload steps need to share consistently.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    """

    group_run_id: str
    created_at_utc: str
    code_commit_sha: str
    dataset_id: str
    split: str
    case_type_filter: str | None
    n_eval_requested: int
    n_eval_actual: int
    model_id: str
    max_new_tokens: int
    run_tag: str | None
    schema_version: str = SCHEMA_VERSION

    def __init__(
        self,
        *,
        group_run_id: str,
        created_at_utc: str,
        code_commit_sha: str,
        subresults: Iterable[SubrunResult],
        default_max_new_tokens: int,
        run_tag: str | None = None,
    ) -> None:
        """Build the group-run context from the collected subrun results.

        Purpose:
            Offer a convenience construction path that derives this value
            directly from upstream runtime or evaluation inputs.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        materialized = tuple(subresults)
        if not materialized:
            raise ValueError("subresults must not be empty")
        first = materialized[0]
        dataset_metadata = first.subrun.dataset.metadata()
        object.__setattr__(self, "group_run_id", group_run_id)
        object.__setattr__(self, "created_at_utc", created_at_utc)
        object.__setattr__(self, "code_commit_sha", code_commit_sha)
        object.__setattr__(self, "dataset_id", dataset_metadata["dataset_id"])
        object.__setattr__(self, "split", dataset_metadata["split"])
        object.__setattr__(self, "case_type_filter", None)
        object.__setattr__(
            self,
            "n_eval_requested",
            sum(subresult.n_eval_requested for subresult in materialized),
        )
        object.__setattr__(
            self,
            "n_eval_actual",
            sum(subresult.n_eval_actual for subresult in materialized),
        )
        object.__setattr__(self, "model_id", first.subrun.model.model_id)
        object.__setattr__(self, "max_new_tokens", default_max_new_tokens)
        object.__setattr__(self, "run_tag", run_tag)
        object.__setattr__(self, "schema_version", SCHEMA_VERSION)


@dataclass(frozen=True, slots=True, init=False)
class SubrunContext:
    """Immutable row model for subrun-level context derived from one subrun.

    Purpose:
        Bundle the contextual identifiers, paths, and aggregate values that
        multiple reporting or upload steps need to share consistently.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    """

    group_run_id: str
    subrun_id: str
    ruleset_name: str
    mode: GenerationMode
    paper_role: PaperRole | None
    paper_variant: str | None
    system_prompt: str
    created_at_utc: str
    code_commit_sha: str
    dataset_id: str
    split: str
    case_type_filter: str | None
    n_eval_requested: int
    n_eval_actual: int
    model_id: str
    max_new_tokens: int
    accuracy: float
    anchor_subrun_id: str | None
    anchor_accuracy: float | None
    delta_accuracy: float | None
    applied_decisions_total: int
    decision_limit_reached: bool
    rules_triggered_count: int
    rules_applied_count: int
    run_tag: str | None
    schema_version: str = SCHEMA_VERSION

    def __init__(
        self,
        *,
        group_context: GroupRunContext,
        subresult: SubrunResult,
        anchor_subrun: SubrunResult,
        run_stats: AggregatedRunStats,
        default_max_new_tokens: int,
        created_at_utc: str,
        code_commit_sha: str,
        run_tag: str | None = None,
    ) -> None:
        """Build the subrun context from one subrun result object.

        Purpose:
            Offer a convenience construction path that derives this value
            directly from upstream runtime or evaluation inputs.

        Architectural role:
            Serialization and reporting-model helper inside the downstream
            telemetry representation boundary.

        Inputs:
            Runtime telemetry snapshots, evaluation outputs, and run-context
            metadata prepared by the engine and repro layers.

        Outputs:
            Serialized rows, manifest-friendly records, and typed containers
            used by reporting and artifact publication.

        Ownership:
            Owned by
            `answer_engineering.telemetry.representation.telemetry_types` within
            the downstream telemetry representation boundary.

        """
        is_anchor_subrun = (
            subresult.subrun.subrun_id == anchor_subrun.subrun.subrun_id
        )
        anchor_accuracy = (
            None if is_anchor_subrun else anchor_subrun.report.accuracy
        )
        delta_accuracy = (
            None
            if is_anchor_subrun
            else subresult.report.accuracy - anchor_subrun.report.accuracy
        )
        dataset_metadata = subresult.subrun.dataset.metadata()
        object.__setattr__(self, "group_run_id", group_context.group_run_id)
        object.__setattr__(self, "subrun_id", subresult.subrun.subrun_id)
        object.__setattr__(
            self,
            "ruleset_name",
            getattr(subresult.subrun, "ruleset_name", subresult.subrun.name),
        )
        object.__setattr__(self, "mode", subresult.subrun.mode)
        object.__setattr__(self, "paper_role", subresult.subrun.paper_role)
        object.__setattr__(
            self, "paper_variant", subresult.subrun.paper_variant
        )
        object.__setattr__(
            self, "system_prompt", subresult.subrun.system_prompt
        )
        object.__setattr__(self, "created_at_utc", created_at_utc)
        object.__setattr__(self, "code_commit_sha", code_commit_sha)
        object.__setattr__(self, "dataset_id", dataset_metadata["dataset_id"])
        object.__setattr__(self, "split", dataset_metadata["split"])
        object.__setattr__(self, "case_type_filter", subresult.subrun.case_type)
        object.__setattr__(self, "n_eval_requested", subresult.n_eval_requested)
        object.__setattr__(self, "n_eval_actual", subresult.n_eval_actual)
        object.__setattr__(self, "model_id", subresult.subrun.model.model_id)
        object.__setattr__(self, "max_new_tokens", default_max_new_tokens)
        object.__setattr__(self, "accuracy", subresult.report.accuracy)
        object.__setattr__(
            self,
            "anchor_subrun_id",
            None if is_anchor_subrun else anchor_subrun.subrun.subrun_id,
        )
        object.__setattr__(self, "anchor_accuracy", anchor_accuracy)
        object.__setattr__(self, "delta_accuracy", delta_accuracy)
        object.__setattr__(
            self, "applied_decisions_total", run_stats.applied_decisions
        )
        object.__setattr__(
            self, "decision_limit_reached", run_stats.decision_limit_reached
        )
        object.__setattr__(
            self,
            "rules_triggered_count",
            sum(1 for rule in run_stats.rules if rule.evaluations > 0),
        )
        object.__setattr__(
            self,
            "rules_applied_count",
            sum(1 for rule in run_stats.rules if rule.applied > 0),
        )
        object.__setattr__(self, "run_tag", run_tag)
        object.__setattr__(self, "schema_version", SCHEMA_VERSION)


@dataclass(frozen=True, slots=True)
class GroupTelemetry:
    """Immutable row model for the full telemetry bundle for one comparison.

    Purpose:
        Group the telemetry records that belong to one reporting scope so
        downstream code can move them as one coherent unit.

    Architectural role:
        Serialization and reporting-model helper inside the downstream telemetry
        representation boundary.

    Inputs:
        Runtime telemetry snapshots, evaluation outputs, and run-context
        metadata prepared by the engine and repro layers.

    Outputs:
        Serialized rows, manifest-friendly records, and typed containers used by
        reporting and artifact publication.

    Ownership:
        Owned by `answer_engineering.telemetry.representation.telemetry_types`
        within the downstream telemetry representation boundary.

    """

    group_run_id: str
    group_row: GroupSummaryRow
    comparison_rows: tuple[GroupComparisonRow, ...]
