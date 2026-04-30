"""Merge runtime telemetry into rule-level reproduction statistics.

Purpose:
    Transform per-generation runtime telemetry snapshots into merged rule,
    condition, and candidate aggregates, and annotate rules markdown with run
    statistics.

Architectural role:
    Authoritative aggregation boundary from runtime telemetry to
    reproduction-facing rule statistics.

Architectural direction:
    Preserve authoritative aggregation semantics while reducing concentration of
    unrelated reporting-adjacent concerns in one module.

Why this matters:
    The current concentration is functional but heavy for a central aggregation
    seam.

What better would look like:
    Aggregation policy becomes easier to explain and modify without changing
    unrelated reporting pathways.

How improvement can be recognized:
    - Clearer separation between aggregation policy and report formatting
    - Lower extension cost for new aggregate metrics
    - Fewer cross-module edits for aggregation-only changes

Open constraint:
    Aggregation structure should continue to follow actual experiment and report
    requirements.

"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from answer_engineering.rules import (
    FullPlanCompiler,
    MarkdownRulesParser,
    RulesSyntaxError,
)
from answer_engineering.telemetry import RuntimeTelemetrySnapshot


@dataclass(frozen=True, slots=True)
class RuleConditionAggregate:
    """Store merged counters for one rule condition across a subrun.

    Purpose:
        Represent how often a specific condition was seen, matched, and matched
        while its rule fired after telemetry from many generations has been
        merged together.

    Architectural role:
        Condition-level aggregate in the telemetry aggregation boundary.

    Inputs (architectural provenance):
        Constructed during runtime telemetry merging from per-generation
        condition telemetry.

    Outputs (downstream usage):
        A stable condition summary consumed by rule-level reports and markdown
        annotation.

    Invariants/constraints:
        A condition aggregate is keyed by one condition identity and should not
        mix counts from different condition definitions.

    """

    condition_id: str
    node_path: str
    node_type: str
    debug_expression: str
    matched: int
    seen: int
    matched_generations: int
    matched_fired_generations: int
    matched_while_fired: int

    @property
    def section(self) -> str:
        """Return the section key for matching this condition to rules markdown.

        Purpose:
            Expose the section key used when matching this condition back to
            rules markdown.

        Architectural role:
            Read-only derived view over a merged rule condition aggregate.

        Inputs (architectural provenance):
            Computed from the condition aggregate stored on this object.

        Outputs (downstream usage):
            A normalized scalar used by markdown annotation and debug renderers.

        Invariants/constraints:
            The derived value must stay consistent with the underlying condition
            fields.

        """
        return self.node_path

    @property
    def operator(self) -> str:
        """Return the normalized condition operator label for reporting.

        Purpose:
            Expose the normalized condition operator label for reporting.

        Architectural role:
            Read-only derived view over a merged rule condition aggregate.

        Inputs (architectural provenance):
            Computed from the condition aggregate stored on this object.

        Outputs (downstream usage):
            A normalized scalar used by markdown annotation and debug renderers.

        Invariants/constraints:
            The derived value must stay consistent with the underlying condition
            fields.

        """
        return self.node_type

    @property
    def expression(self) -> str:
        """Return the human-readable condition expression used in annotations.

        Purpose:
            Expose the human-readable condition expression used in annotations.

        Architectural role:
            Read-only derived view over a merged rule condition aggregate.

        Inputs (architectural provenance):
            Computed from the condition aggregate stored on this object.

        Outputs (downstream usage):
            A normalized scalar used by markdown annotation and debug renderers.

        Invariants/constraints:
            The derived value must stay consistent with the underlying condition
            fields.

        """
        return self.debug_expression


@dataclass(frozen=True, slots=True)
class RuleCandidateAggregate:
    """Store merged counters for one candidate choice under a rule.

    Purpose:
        Represent how often a specific candidate option was chosen and in how
        many generations it appeared after telemetry has been merged across a
        subrun.

    Architectural role:
        Candidate-level aggregate nested under one merged rule summary.

    Inputs (architectural provenance):
        Constructed from candidate-choice telemetry emitted during runtime
        execution.

    Outputs (downstream usage):
        A stable candidate summary consumed by human-readable rule statistics
        and debug renderers.

    Invariants/constraints:
        Counts on one aggregate must refer to one candidate identity and kind.

    """

    kind: str
    candidate_id: str
    label: str
    chosen: int
    chosen_generations: int


@dataclass(frozen=True, slots=True)
class RuleAggregate:
    """Store merged counters for one rule across a whole subrun.

    Purpose:
        Collect evaluation counts, application counts, and nested condition and
        candidate aggregates for one rule after many telemetry items have been
        merged together.

    Architectural role:
        Rule-level aggregate produced by telemetry merging.

    Inputs (architectural provenance):
        Constructed from many per-generation rule telemetry records.

    Outputs (downstream usage):
        A stable rule summary consumed by report renderers and markdown
        annotation.

    Invariants/constraints:
        Each aggregate should represent one rule identity across one merged run
        summary.

    """

    rule_id: str
    rule_name: str
    evaluations: int
    applied: int
    fired_generations: int
    total_generations: int
    conditions: tuple[RuleConditionAggregate, ...]
    candidate_choices: tuple[RuleCandidateAggregate, ...]

    @classmethod
    def merge(cls, rules: Sequence[RuleAggregate]) -> RuleAggregate:
        """Merge many rule aggregates into one canonical rule summary.

        Purpose:
            Combine per-run rule counters and examples into the aggregate view
            used by paper reporting.

        Architectural role:
            Aggregation boundary between raw telemetry-derived rows and stable
            report objects.

        Inputs (architectural provenance):
            Receives compatible rule aggregates produced from individual runs or
            intermediate merged sections.

        Outputs (downstream usage):
            Returns one canonical aggregate consumed by tables, markdown
            summaries, and reproduction artifacts.

        Invariants/constraints:
            Merging must preserve rule identity, sum counters deterministically,
            and avoid inventing examples that were not present in the source
            aggregates.

        """
        if not rules:
            raise ValueError("Cannot merge empty RuleAggregate sequence")
        total_generations = rules[0].total_generations
        for rule in rules[1:]:
            if rule.total_generations != total_generations:
                raise ValueError(
                    "Inconsistent total_generations across rule stats"
                )
        merged_conditions: dict[
            tuple[str, str, str, str], RuleConditionAggregate
        ] = {}
        for condition in (
            condition for rule in rules for condition in rule.conditions
        ):
            key = (
                condition.condition_id,
                condition.node_path,
                condition.node_type,
                condition.debug_expression,
            )
            prior = merged_conditions.get(key)
            if prior is None:
                merged_conditions[key] = condition
                continue
            merged_conditions[key] = RuleConditionAggregate(
                condition_id=condition.condition_id,
                node_path=condition.node_path,
                node_type=condition.node_type,
                debug_expression=condition.debug_expression,
                matched=prior.matched + condition.matched,
                seen=prior.seen + condition.seen,
                matched_generations=(
                    prior.matched_generations + condition.matched_generations
                ),
                matched_fired_generations=(
                    prior.matched_fired_generations
                    + condition.matched_fired_generations
                ),
                matched_while_fired=(
                    prior.matched_while_fired + condition.matched_while_fired
                ),
            )
        merged_candidates: dict[
            tuple[str, str, str], RuleCandidateAggregate
        ] = {}
        for candidate in (
            candidate for rule in rules for candidate in rule.candidate_choices
        ):
            key = (candidate.kind, candidate.candidate_id, candidate.label)
            prior = merged_candidates.get(key)
            if prior is None:
                merged_candidates[key] = candidate
                continue
            merged_candidates[key] = RuleCandidateAggregate(
                kind=candidate.kind,
                candidate_id=candidate.candidate_id,
                label=candidate.label,
                chosen=prior.chosen + candidate.chosen,
                chosen_generations=(
                    prior.chosen_generations + candidate.chosen_generations
                ),
            )
        fired_generations_sum = sum(rule.fired_generations for rule in rules)
        fired_generations = min(fired_generations_sum, total_generations)
        merged_conditions_capped = tuple(
            RuleConditionAggregate(
                condition_id=condition.condition_id,
                node_path=condition.node_path,
                node_type=condition.node_type,
                debug_expression=condition.debug_expression,
                matched=condition.matched,
                seen=condition.seen,
                matched_generations=min(
                    condition.matched_generations, total_generations
                ),
                matched_fired_generations=min(
                    condition.matched_fired_generations, fired_generations
                ),
                matched_while_fired=condition.matched_while_fired,
            )
            for condition in sorted(
                merged_conditions.values(),
                key=lambda item: (
                    item.node_path,
                    item.node_type,
                    item.debug_expression,
                    item.condition_id,
                ),
            )
        )
        merged_candidates_capped = tuple(
            RuleCandidateAggregate(
                kind=candidate.kind,
                candidate_id=candidate.candidate_id,
                label=candidate.label,
                chosen=candidate.chosen,
                chosen_generations=min(
                    candidate.chosen_generations, fired_generations
                ),
            )
            for candidate in sorted(
                merged_candidates.values(),
                key=lambda item: (item.kind, item.candidate_id, item.label),
            )
        )
        return cls(
            rule_id=rules[0].rule_id,
            rule_name=rules[0].rule_name,
            evaluations=sum(r.evaluations for r in rules),
            applied=sum(r.applied for r in rules),
            fired_generations=fired_generations,
            total_generations=total_generations,
            conditions=merged_conditions_capped,
            candidate_choices=merged_candidates_capped,
        )

    @classmethod
    def from_merged_rule(
        cls, rule_data: _MergedRule, *, total_generations: int
    ) -> RuleAggregate:
        """Materialize one aggregate from merged mutable counters.

        Purpose:
            Freeze accumulated mutable aggregation state into the immutable
            report value used by downstream rendering.

        Architectural role:
            Constructor-like boundary between internal counting machinery and
            public aggregate rows.

        Inputs (architectural provenance):
            Receives canonical rule identity, accumulated counters, match
            summaries, and example collections from the aggregation pass.

        Outputs (downstream usage):
            Returns a `RuleAggregate` consumed by run summaries and report
            renderers.

        Invariants/constraints:
            The method should be the point where mutable counting state is
            normalized, not a place where new telemetry semantics are inferred.

        """
        return cls(
            rule_id=rule_data.rule_id,
            rule_name=rule_data.rule_name,
            evaluations=rule_data.evaluations,
            applied=rule_data.applied,
            fired_generations=rule_data.fired_generations,
            total_generations=total_generations,
            conditions=tuple(
                RuleConditionAggregate(
                    condition_id=condition.condition_id,
                    node_path=condition.node_path,
                    node_type=condition.node_type,
                    debug_expression=condition.debug_expression,
                    matched=condition.matched,
                    seen=condition.seen,
                    matched_generations=condition.matched_generations,
                    matched_fired_generations=(
                        condition.matched_fired_generations
                    ),
                    matched_while_fired=condition.matched_while_fired,
                )
                for condition in sorted(
                    rule_data.conditions.values(),
                    key=lambda x: (
                        x.node_path,
                        x.node_type,
                        x.debug_expression,
                    ),
                )
            ),
            candidate_choices=tuple(
                RuleCandidateAggregate(
                    kind=candidate.kind,
                    candidate_id=candidate.candidate_id,
                    label=candidate.label,
                    chosen=candidate.chosen,
                    chosen_generations=candidate.chosen_generations,
                )
                for candidate in sorted(
                    rule_data.candidate_choices.values(),
                    key=lambda x: (x.kind, x.candidate_id, x.label),
                )
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class AggregatedRunStats:
    """Store the merged telemetry summary for an entire evaluated subrun.

    Purpose:
        Collect the total applied-decision counts, decision-limit status, and
        merged rule aggregates for one run so downstream reporting code can
        treat telemetry as one coherent object.

    Architectural role:
        Top-level telemetry summary record in the aggregation boundary.

    Inputs (architectural provenance):
        Constructed from the `TelemetryItem` sequence extracted from one
        evaluated subrun.

    Outputs (downstream usage):
        Merged run statistics consumed by summary builders and artifact
        renderers.

    Invariants/constraints:
        The contained rule aggregates must all originate from the same evaluated
        subrun.

    """

    applied_decisions: int
    decision_limit_reached: bool
    rules: tuple[RuleAggregate, ...]

    def __init__(self, items: Sequence[TelemetryItem]) -> None:
        """Merge telemetry items into one canonical run-level summary.

        Purpose:
            Build the top-level aggregate view for a run from rule, case, and
            section telemetry inputs.

        Architectural role:
            Reporting constructor that turns many telemetry-derived records into
            the stable object consumed by notebooks and paper artifacts.

        Inputs (architectural provenance):
            Receives run metadata, aggregate counters, rule aggregates, and
            optional section-level breakdowns from aggregation code.

        Outputs (downstream usage):
            Stores normalized aggregate data for markdown, JSON, and TeX
            reporting.

        Invariants/constraints:
            Constructor normalization should make the object internally
            consistent so renderers can remain formatting-only.

        """
        merged_rules: dict[str, _MergedRule] = {}
        applied_decisions = 0
        decision_limit_reached = False
        for item in items:
            applied_decisions += item.applied_decisions
            decision_limit_reached = (
                decision_limit_reached or item.decision_limit_reached
            )
            for stats in item.rules:
                bucket = merged_rules.setdefault(
                    stats.rule_id,
                    _MergedRule(
                        rule_id=stats.rule_id,
                        rule_name=stats.rule_name,
                        conditions={},
                        candidate_choices={},
                    ),
                )
                bucket.evaluations += stats.evaluations
                item_applied = stats.applied
                bucket.applied += item_applied
                if item_applied > 0:
                    bucket.fired_generations += 1
                for condition in stats.conditions:
                    key = (
                        f"{condition.node_path}:{condition.node_type}:"
                        f"{condition.debug_expression}"
                    )
                    merged_condition = bucket.conditions.setdefault(
                        key,
                        _MergedCondition(
                            condition_id=condition.condition_id,
                            node_path=condition.node_path,
                            node_type=condition.node_type,
                            debug_expression=condition.debug_expression,
                        ),
                    )
                    this_matched = condition.matched
                    merged_condition.matched += this_matched
                    merged_condition.seen += condition.seen
                    if this_matched > 0:
                        merged_condition.matched_generations += 1
                        if item_applied > 0:
                            merged_condition.matched_fired_generations += 1
                            merged_condition.matched_while_fired += this_matched
                for candidate in stats.candidate_choices:
                    key = f"{candidate.kind}:{candidate.candidate_id}"
                    merged_candidate = bucket.candidate_choices.setdefault(
                        key,
                        _MergedCandidate(
                            kind=candidate.kind,
                            candidate_id=candidate.candidate_id,
                            label=candidate.label,
                        ),
                    )
                    candidate_chosen = candidate.chosen
                    merged_candidate.chosen += candidate_chosen
                    if candidate_chosen > 0:
                        merged_candidate.chosen_generations += 1
        rules = tuple(
            RuleAggregate.from_merged_rule(
                rule_data,
                total_generations=len(items),
            )
            for rule_data in sorted(
                merged_rules.values(), key=lambda x: x.rule_id
            )
        )
        object.__setattr__(self, "applied_decisions", applied_decisions)
        object.__setattr__(
            self, "decision_limit_reached", decision_limit_reached
        )
        object.__setattr__(self, "rules", rules)


@dataclass(frozen=True, slots=True)
class ConditionTelemetry:
    """Represent one condition snapshot from runtime telemetry.

    Purpose:
        Capture the raw per-generation counters for a single condition before
        they are merged across the run.

    Architectural role:
        Input row type at the edge of telemetry aggregation.

    Inputs (architectural provenance):
        Produced from runtime telemetry snapshots emitted by the engine.

    Outputs (downstream usage):
        A condition telemetry item consumed by merge helpers.

    Invariants/constraints:
        This type should remain a faithful copy of runtime telemetry rather than
        a derived summary.

    """

    condition_id: str
    node_path: str
    node_type: str
    debug_expression: str
    matched: int
    seen: int


@dataclass(frozen=True, slots=True)
class CandidateTelemetry:
    """Represent one candidate-choice snapshot from runtime telemetry.

    Purpose:
        Capture the raw per-generation counters for one candidate option before
        run-level aggregation combines them.

    Architectural role:
        Input row type at the edge of telemetry aggregation.

    Inputs (architectural provenance):
        Produced from runtime telemetry snapshots emitted by the engine.

    Outputs (downstream usage):
        A candidate telemetry item consumed by merge helpers.

    Invariants/constraints:
        This type should remain a faithful copy of runtime telemetry rather than
        a derived summary.

    """

    kind: str
    candidate_id: str
    label: str
    chosen: int


@dataclass(frozen=True, slots=True)
class RuleTelemetry:
    """Represent one rule snapshot from runtime telemetry.

    Purpose:
        Capture the per-generation counters and nested condition/candidate
        telemetry for one rule before aggregation merges many snapshots
        together.

    Architectural role:
        Input row type for rule-level telemetry aggregation.

    Inputs (architectural provenance):
        Produced from runtime telemetry snapshots emitted during answer
        generation.

    Outputs (downstream usage):
        A rule telemetry item consumed by `TelemetryItem` and merge helpers.

    Invariants/constraints:
        One instance should correspond to one rule in one generation snapshot.

    """

    rule_id: str
    rule_name: str
    evaluations: int
    applied: int
    conditions: tuple[ConditionTelemetry, ...]
    candidate_choices: tuple[CandidateTelemetry, ...]


@dataclass(frozen=True, slots=True, init=False)
class TelemetryItem:
    """Represent one telemetry snapshot in reproduction-friendly form.

    Purpose:
        Normalize the runtime telemetry emitted for one generated answer into
        immutable rule-level records that can later be merged across the whole
        subrun.

    Architectural role:
        Adapter type between engine telemetry snapshots and reproduction
        aggregation code.

    Inputs (architectural provenance):
        Constructed from raw engine runtime telemetry snapshots attached to
        evaluation results.

    Outputs (downstream usage):
        Per-generation telemetry items consumed by run-level merge functions.

    Invariants/constraints:
        A telemetry item must preserve the rule, condition, and candidate
        identities emitted by the runtime.

    """

    applied_decisions: int
    decision_limit_reached: bool
    rules: tuple[RuleTelemetry, ...]

    def __init__(self, raw: RuntimeTelemetrySnapshot) -> None:
        """Convert one runtime snapshot into a reproduction telemetry item.

        Purpose:
            Copy the rule-level structure and counters from an engine
            `RuntimeTelemetrySnapshot` into immutable reproduction-facing
            telemetry records.

        Architectural role:
            Constructor at the boundary between engine telemetry and
            reproduction aggregation.

        Inputs (architectural provenance):
            Consumes one raw runtime telemetry snapshot.

        Outputs (downstream usage):
            A normalized `TelemetryItem` ready for later merging.

        Invariants/constraints:
            This conversion should preserve counts and identifiers rather than
            reinterpret their semantics.

        """
        object.__setattr__(self, "applied_decisions", raw.applied_decisions)
        object.__setattr__(
            self, "decision_limit_reached", raw.decision_limit_reached
        )
        object.__setattr__(
            self,
            "rules",
            tuple(
                RuleTelemetry(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    evaluations=rule.evaluations,
                    applied=rule.applied,
                    conditions=tuple(
                        ConditionTelemetry(
                            condition_id=condition.condition_id,
                            node_path=condition.node_path,
                            node_type=condition.node_type,
                            debug_expression=condition.debug_expression,
                            matched=condition.matched,
                            seen=condition.seen,
                        )
                        for condition in rule.conditions
                    ),
                    candidate_choices=tuple(
                        CandidateTelemetry(
                            kind=candidate.kind,
                            candidate_id=candidate.candidate_id,
                            label=candidate.label,
                            chosen=candidate.chosen,
                        )
                        for candidate in rule.candidate_choices
                    ),
                )
                for rule in raw.rules
            ),
        )


@dataclass(slots=True)
class _MergedCondition:
    """Hold mutable merge state for condition aggregation.

    Purpose:
        Accumulate counters while many telemetry items are being merged before
        the final immutable aggregate objects are materialized.

    Architectural role:
        Private helper record used only inside telemetry merging.

    Inputs (architectural provenance):
        Created and mutated by the merge helpers in this module.

    Outputs (downstream usage):
        Intermediate state consumed by the final aggregate constructors.

    Invariants/constraints:
        Private merge state should not leak outside this module.

    """

    condition_id: str
    node_path: str
    node_type: str
    debug_expression: str
    matched: int = 0
    seen: int = 0
    matched_generations: int = 0
    matched_fired_generations: int = 0
    matched_while_fired: int = 0


@dataclass(slots=True)
class _MergedCandidate:
    """Hold mutable merge state for candidate aggregation.

    Purpose:
        Accumulate counters while many telemetry items are being merged before
        the final immutable aggregate objects are materialized.

    Architectural role:
        Private helper record used only inside telemetry merging.

    Inputs (architectural provenance):
        Created and mutated by the merge helpers in this module.

    Outputs (downstream usage):
        Intermediate state consumed by the final aggregate constructors.

    Invariants/constraints:
        Private merge state should not leak outside this module.

    """

    kind: str
    candidate_id: str
    label: str
    chosen: int = 0
    chosen_generations: int = 0


def _make_conditions() -> dict[str, _MergedCondition]:
    return {}


def _make_candidate_choices() -> dict[str, _MergedCandidate]:
    return {}


@dataclass(slots=True)
class _MergedRule:
    """Hold mutable merge state for rule aggregation.

    Purpose:
        Accumulate counters while many telemetry items are being merged before
        the final immutable aggregate objects are materialized.

    Architectural role:
        Private helper record used only inside telemetry merging.

    Inputs (architectural provenance):
        Created and mutated by the merge helpers in this module.

    Outputs (downstream usage):
        Intermediate state consumed by the final aggregate constructors.

    Invariants/constraints:
        Private merge state should not leak outside this module.

    """

    rule_id: str
    rule_name: str
    evaluations: int = 0
    applied: int = 0
    fired_generations: int = 0
    conditions: dict[str, _MergedCondition] = field(
        default_factory=_make_conditions
    )
    candidate_choices: dict[str, _MergedCandidate] = field(
        default_factory=_make_candidate_choices
    )


def annotate_rules_with_run_stats(
    rules_markdown: str, run_stats: AggregatedRunStats
) -> str:
    """Annotate rules markdown with merged runtime statistics.

    Purpose:
        Parse the original rules markdown, attach merged run-level condition and
        candidate statistics where possible, and return a human-readable
        annotated version for artifacts.

    Architectural role:
        Presentation helper at the end of telemetry aggregation.

    Inputs (architectural provenance):
        Consumes original rules markdown and the merged run statistics for a
        subrun.

    Outputs (downstream usage):
        Annotated markdown consumed by report and artifact writers.

    Invariants/constraints:
        If structural matching cannot be done safely, the caller should receive
        a fallback annotation rather than broken markdown.

    """
    input_contains_annotations = _contains_ae_annotations(rules_markdown)
    cleaned_rules = _strip_existing_ae_annotations(rules_markdown)
    stats_by_id = {rule.rule_id: rule for rule in run_stats.rules}
    merged_stats_by_name = _merge_rule_aggregates_by_normalized_name(
        run_stats.rules
    )
    try:
        parsed = MarkdownRulesParser().parse(cleaned_rules)
        plan = FullPlanCompiler().compile(parsed)
    except (RulesSyntaxError, TypeError, ValueError):
        return _annotate_rules_fallback(cleaned_rules, run_stats)
    lines = cleaned_rules.splitlines()
    heading_indices = [
        index
        for index, line in enumerate(lines)
        if line.strip().startswith("## ")
    ]
    heading_titles_by_index = {
        index: title
        for index, line in enumerate(lines)
        if (title := _heading_title_from_line(line)) is not None
    }
    if heading_indices and not heading_titles_by_index:
        return _annotate_rules_fallback(cleaned_rules, run_stats)

    compiled_name_to_merged_stats: dict[str, RuleAggregate] = {}
    compiled_name_to_rule_ids: dict[str, list[str]] = defaultdict(list)
    compiled_name_to_canonical: dict[str, str] = {}
    for rule in plan.rules:
        normalized_name = _normalize_heading_title(rule.name)
        compiled_name_to_rule_ids[normalized_name].append(rule.rule_id)
        compiled_name_to_canonical.setdefault(normalized_name, rule.name)
        normalized_stats = merged_stats_by_name.get(normalized_name)
        direct_stats = stats_by_id.get(rule.rule_id)
        if normalized_stats is not None:
            compiled_name_to_merged_stats[normalized_name] = normalized_stats
        elif direct_stats is not None:
            compiled_name_to_merged_stats[normalized_name] = direct_stats

    output: list[str] = []
    current_rule_stats: RuleAggregate | None = None
    current_section = ""
    for index, line in enumerate(lines):
        stripped = line.strip()
        heading_title = heading_titles_by_index.get(index)
        if heading_title is not None:
            normalized_heading = _normalize_heading_title(heading_title)
            current_rule_stats = compiled_name_to_merged_stats.get(
                normalized_heading
            ) or merged_stats_by_name.get(normalized_heading)
            heading_rule_ids = compiled_name_to_rule_ids.get(
                normalized_heading, []
            )
            if heading_rule_ids:
                canonical_name = compiled_name_to_canonical.get(
                    normalized_heading, normalized_heading
                )
                output.append(
                    f"// ae-rule-id: {heading_rule_ids[0]} "
                    f"canonical={canonical_name}"
                )
            else:
                output.append(
                    f"// ae-rule-id: authored canonical={heading_title}"
                )
            if current_rule_stats is not None:
                output.extend(_render_human_rule_summary(current_rule_stats))
            current_section = ""
        parsed_section = _parse_section_line(stripped)
        if parsed_section is not None:
            current_section = parsed_section.key
        output.append(
            _annotate_line_inline(
                line=line,
                current_section=current_section,
                stats=current_rule_stats,
            )
        )

    if not input_contains_annotations:
        output.extend(_render_run_level_summary(run_stats))
    result = "\n".join(output)
    return f"{result}\n" if result else ""


def _annotate_rules_fallback(
    rules_markdown: str, run_stats: AggregatedRunStats
) -> str:
    """Render fallback annotation when structural rule matching is unavailable.

    Purpose:
        Append run-level and per-rule summaries to the raw rules markdown
        without relying on parser-based structural placement.

    Architectural role:
        Private degradation path for telemetry annotation.

    Inputs (architectural provenance):
        Consumes original rules markdown and merged run statistics.

    Outputs (downstream usage):
        A readable fallback markdown report.

    Invariants/constraints:
        Fallback output should remain readable even when exact
        condition-to-markdown alignment fails.

    """
    lines = rules_markdown.splitlines()
    stats_by_name = _merge_rule_aggregates_by_raw_name(run_stats.rules)
    stats_by_name_normalized = _merge_rule_aggregates_by_normalized_name(
        run_stats.rules
    )
    stats_by_id = {rule.rule_id: rule for rule in run_stats.rules}
    out: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            title = stripped[3:].strip()
            current_rule = stats_by_name.get(title) or stats_by_id.get(title)
            if current_rule is None:
                current_rule = stats_by_name_normalized.get(
                    _normalize_heading_title(title)
                )
            if current_rule is not None:
                out.extend(_render_human_rule_summary(current_rule))
        out.append(line)
    out.extend(_render_run_level_summary(run_stats))
    return "\n".join(out)


def _normalize_candidate_label(value: str) -> str:
    """Normalize a candidate label for stable human-facing reporting.

    Purpose:
        Convert a candidate label for stable human-facing reporting.

    Architectural role:
        Private presentation helper inside telemetry aggregation.

    Inputs (architectural provenance):
        Consumes already-computed telemetry aggregates or markdown fragments.

    Outputs (downstream usage):
        A normalized string or list of strings used while building annotated
        reports.

    Invariants/constraints:
        These helpers should format existing information only and must not
        change aggregate counts.

    """
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _render_candidate_stats(
    *,
    candidate_id: str,
    label: str,
    text: str,
    ordinal: int,
    stats: RuleAggregate | None,
) -> str:
    """Render compact statistics for one candidate choice aggregate.

    Purpose:
        Build compact statistics for one candidate choice aggregate.

    Architectural role:
        Private presentation helper inside telemetry aggregation.

    Inputs (architectural provenance):
        Consumes already-computed telemetry aggregates or markdown fragments.

    Outputs (downstream usage):
        A normalized string or list of strings used while building annotated
        reports.

    Invariants/constraints:
        These helpers should format existing information only and must not
        change aggregate counts.

    """
    if stats is None:
        return ""
    normalized_label = _normalize_candidate_label(label)
    normalized_text = _normalize_candidate_label(text)
    candidate = next(
        (c for c in stats.candidate_choices if c.candidate_id == candidate_id),
        None,
    )
    if candidate is None:
        candidate = next(
            (c for c in stats.candidate_choices if c.label == label and label),
            None,
        )
    if candidate is None:
        candidate = next(
            (
                c
                for c in stats.candidate_choices
                if _normalize_candidate_label(c.label)
                in {normalized_label, normalized_text}
            ),
            None,
        )
    if candidate is None:
        fallback_label = f"fallback_{ordinal}"
        candidate = next(
            (c for c in stats.candidate_choices if c.label == fallback_label),
            None,
        )
    if candidate is None or candidate.chosen_generations <= 0:
        return ""
    chosen = candidate.chosen
    chosen_generations = candidate.chosen_generations
    rate = (
        (chosen_generations / stats.fired_generations)
        if stats.fired_generations
        else 0.0
    )
    avg_when_chosen = (
        (chosen / chosen_generations) if chosen_generations else 0.0
    )
    return (
        "// ae-stats: "
        f"chosen={chosen_generations}/{stats.fired_generations or 1} "
        f"({rate:.1%}) avg_hits_when_chosen={avg_when_chosen:.2f} "
        f"total_hits={chosen}"
    )


def _normalize_heading_title(title: str) -> str:
    """Normalize a markdown heading title before matching telemetry to sections.

    Purpose:
        Convert a markdown heading title before matching telemetry to sections.

    Architectural role:
        Private presentation helper inside telemetry aggregation.

    Inputs (architectural provenance):
        Consumes already-computed telemetry aggregates or markdown fragments.

    Outputs (downstream usage):
        A normalized string or list of strings used while building annotated
        reports.

    Invariants/constraints:
        These helpers should format existing information only and must not
        change aggregate counts.

    """
    lowered = re.sub(r"\s+", " ", title.strip().lower())
    without_mode = re.sub(r"\s*\([^)]*\)", "", lowered)
    normalized = re.sub(
        r"^(replace|avoid|after|force)\s*:?\s*", r"\1: ", without_mode
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _heading_title_from_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("## "):
        return None
    title = stripped[3:].strip()
    return title if title else None


def _render_human_rule_summary(rule: RuleAggregate) -> list[str]:
    """Render compact human-readable summary lines for one merged rule.

    Purpose:
        Build compact human-readable summary lines for one merged rule.

    Architectural role:
        Private presentation helper inside telemetry aggregation.

    Inputs (architectural provenance):
        Consumes already-computed telemetry aggregates or markdown fragments.

    Outputs (downstream usage):
        A normalized string or list of strings used while building annotated
        reports.

    Invariants/constraints:
        These helpers should format existing information only and must not
        change aggregate counts.

    """
    fired_rate = (
        (rule.fired_generations / rule.total_generations)
        if rule.total_generations
        else 0.0
    )
    avg_repeat = (
        (rule.applied / rule.fired_generations)
        if rule.fired_generations
        else 0.0
    )
    lines: list[str] = [
        (
            "// ae-stats: "
            f"fired={rule.fired_generations}/{rule.total_generations} "
            f"({fired_rate:.1%}) total_applications={rule.applied} "
            f"avg_repeat_when_fired={avg_repeat:.2f}"
        )
    ]
    top_candidates = sorted(
        (
            candidate
            for candidate in rule.candidate_choices
            if candidate.chosen_generations > 0
        ),
        key=lambda item: (
            -item.chosen,
            item.label.casefold(),
            item.candidate_id,
        ),
    )[:3]
    if top_candidates:
        joined = ", ".join(
            f"{_safe_candidate_label(candidate)} {candidate.chosen}"
            for candidate in top_candidates
        )
        lines.append(f"// ae-stats: chosen candidates: {joined}")
    top_terms = _top_condition_terms(
        rule.conditions, total=rule.total_generations
    )
    if top_terms:
        top_terms_joined = ", ".join(top_terms)
        lines.append(f"// ae-stats: top trigger terms: {top_terms_joined}")
    return lines


def _safe_candidate_label(candidate: RuleCandidateAggregate) -> str:
    raw = (candidate.label or "").strip()
    if not raw or _looks_like_internal_candidate_id(raw):
        return candidate.candidate_id
    return raw


def _looks_like_internal_candidate_id(value: str) -> bool:
    return bool(
        re.match(r"^(probe|rewrite|insert|force|candidate)_\d+$", value)
    )


@dataclass(frozen=True, slots=True)
class ParsedSectionLine:
    key: str
    value: str | None


def _parse_section_line(stripped_line: str) -> ParsedSectionLine | None:
    match = re.match(
        (
            r"^(Prompt|Prefix|Connector|Postfix|With|Fallback|Add)"
            r"(?:\s*\([^)]*\))?:\s*(.*?)\s*$"
        ),
        stripped_line,
    )
    if match is None:
        return None
    value = match.group(2).strip()
    return ParsedSectionLine(
        key=match.group(1).casefold(),
        value=value if value else None,
    )


def _annotate_line_inline(
    *,
    line: str,
    current_section: str,
    stats: RuleAggregate | None,
) -> str:
    if stats is None:
        return line
    parsed_section = _parse_section_line(line.strip())
    if (
        parsed_section is not None
        and parsed_section.value is not None
        and parsed_section.key in {"with", "fallback", "add"}
    ):
        annotation = _render_candidate_stats(
            candidate_id=parsed_section.value,
            label=parsed_section.value,
            text=parsed_section.value,
            ordinal=1,
            stats=stats,
        )
        return f"{line} {annotation}".rstrip() if annotation else line
    if (
        parsed_section is not None
        and parsed_section.value is not None
        and parsed_section.key in {"prefix", "postfix", "prompt", "connector"}
    ):
        condition_annotation = _render_condition_annotation(
            section=parsed_section.key,
            text=parsed_section.value,
            stats=stats,
        )
        return (
            f"{line} {condition_annotation}".rstrip()
            if condition_annotation
            else line
        )
    bullet_match = re.match(r"^(\s*-\s+)(.+?)\s*$", line)
    if bullet_match is None:
        return line
    bullet_text = bullet_match.group(2)
    if current_section in {"with", "fallback", "add"}:
        annotation = _render_candidate_stats(
            candidate_id=bullet_text,
            label=bullet_text,
            text=bullet_text,
            ordinal=1,
            stats=stats,
        )
        return f"{line} {annotation}".rstrip() if annotation else line
    if current_section in {"prefix", "postfix", "prompt", "connector"}:
        condition_annotation = _render_condition_annotation(
            section=current_section,
            text=bullet_text,
            stats=stats,
        )
        return (
            f"{line} {condition_annotation}".rstrip()
            if condition_annotation
            else line
        )
    return line


def _render_condition_annotation(
    *,
    section: str,
    text: str,
    stats: RuleAggregate,
) -> str:
    template_annotation = _render_template_condition_annotation(
        section=section,
        text=text,
        stats=stats,
    )
    if template_annotation:
        return template_annotation
    condition = _find_matching_leaf_condition(stats.conditions, section, text)
    if condition is None or condition.matched_generations <= 0:
        return ""
    return (
        f"// ae-stats: matched={condition.matched_generations}/"
        f"{stats.total_generations}"
    )


def _find_matching_leaf_condition(
    conditions: Iterable[RuleConditionAggregate],
    section: str,
    bullet_text: str,
) -> RuleConditionAggregate | None:
    normalized_text = _normalize_candidate_label(bullet_text)
    candidates = [
        condition
        for condition in conditions
        if condition.node_path == section
        and not _is_structural_condition(condition)
    ]
    for condition in candidates:
        if (
            _normalize_candidate_label(condition.debug_expression)
            == normalized_text
        ):
            return condition
    return None


def _render_template_condition_annotation(
    *,
    section: str,
    text: str,
    stats: RuleAggregate,
) -> str:
    alternatives = _extract_template_alternatives(text)
    if alternatives is None or len(alternatives) <= 1:
        return ""
    rendered: list[str] = []
    found_any = False
    for alternative in alternatives:
        condition = _find_matching_leaf_condition(
            stats.conditions, section, alternative
        )
        matched_generations = condition.matched_generations if condition else 0
        if matched_generations > 0:
            found_any = True
        rendered.append(
            f"{alternative} {matched_generations}/{stats.total_generations}"
        )
    if not found_any:
        return ""
    joined = ", ".join(rendered)
    return f"// ae-stats: {joined}"


def _extract_template_alternatives(text: str) -> tuple[str, ...] | None:
    escaped = False
    runs: list[int] = []
    index = 0
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if char != "|":
            index += 1
            continue
        run_end = index
        while run_end < len(text) and text[run_end] == "|":
            run_end += 1
        runs.append(run_end - index)
        index = run_end
    if not runs:
        return None
    if len(set(runs)) != 1:
        return None
    delimiter_width = runs[0]
    parts = re.split(rf"(?<!\\)\|{{{delimiter_width}}}", text)
    alternatives = tuple(part.replace("\\|", "|").strip() for part in parts)
    if len(alternatives) <= 1:
        return None
    return alternatives


def _is_structural_condition(condition: RuleConditionAggregate) -> bool:
    expr = condition.debug_expression.strip()
    structural_node_types = {
        "MatchAll",
        "MatchAny",
        "MatchAndThen",
        "MatchNot",
    }
    return (
        condition.node_path.startswith("guard")
        or not expr
        or expr == condition.node_type
        or condition.node_type in structural_node_types
        or expr.casefold() in {"matchall", "matchany", "matchandthen"}
    )


def _top_condition_terms(
    conditions: Iterable[RuleConditionAggregate], *, total: int
) -> list[str]:
    top = sorted(
        (
            condition
            for condition in conditions
            if condition.matched_generations > 0
            and not _is_structural_condition(condition)
        ),
        key=lambda item: (
            -item.matched_generations,
            item.debug_expression.casefold(),
        ),
    )[:3]
    return [
        f"{condition.debug_expression} {condition.matched_generations}/{total}"
        for condition in top
    ]


def _merge_rule_aggregates_by_raw_name(
    rules: Sequence[RuleAggregate],
) -> dict[str, RuleAggregate]:
    grouped: dict[str, list[RuleAggregate]] = defaultdict(list)
    for rule in rules:
        if rule.rule_name:
            grouped[rule.rule_name].append(rule)
    return {name: RuleAggregate.merge(group) for name, group in grouped.items()}


def _merge_rule_aggregates_by_normalized_name(
    rules: Sequence[RuleAggregate],
) -> dict[str, RuleAggregate]:
    grouped: dict[str, list[RuleAggregate]] = defaultdict(list)
    for rule in rules:
        if rule.rule_name:
            grouped[_normalize_heading_title(rule.rule_name)].append(rule)
    return {name: RuleAggregate.merge(group) for name, group in grouped.items()}


def _render_run_level_summary(run_stats: AggregatedRunStats) -> list[str]:
    if not run_stats.rules:
        return list()
    fired_sorted = sorted(
        run_stats.rules,
        key=lambda item: (
            -item.fired_generations,
            -item.applied,
            item.rule_name,
        ),
    )[:3]
    repeat_sorted = sorted(
        (
            item
            for item in run_stats.rules
            if item.fired_generations > 0 and item.applied > 0
        ),
        key=lambda item: (
            -(item.applied / item.fired_generations),
            item.rule_name,
        ),
    )[:3]
    fallback_used = sorted(
        (
            item
            for item in run_stats.rules
            if any(
                candidate.kind == "fallback"
                and candidate.chosen_generations > 0
                for candidate in item.candidate_choices
            )
        ),
        key=lambda item: (-item.fired_generations, item.rule_name),
    )[:3]
    lines = ["", "## Rule activity summary"]
    lines.append("- most active rules by fired generations:")
    for rule in fired_sorted:
        lines.append(
            f"  - {rule.rule_name}: "
            f"{rule.fired_generations}/{rule.total_generations}"
        )
    lines.append("- highest repeat burden:")
    for rule in repeat_sorted:
        repeat = rule.applied / rule.fired_generations
        lines.append(f"  - {rule.rule_name}: {repeat:.2f}")
    lines.append("- fallback actually used:")
    if fallback_used:
        for rule in fallback_used:
            lines.append(
                f"  - {rule.rule_name}: "
                f"{rule.fired_generations} fired generations"
            )
    else:
        lines.append("  - none")
    lines.append("// ae-stats: run-summary")
    lines.append(
        f"//   applied_decisions={run_stats.applied_decisions} "
        f"decision_limit_reached="
        f"{str(run_stats.decision_limit_reached).lower()}"
    )
    return lines


def _strip_existing_ae_annotations(rules_markdown: str) -> str:
    lines = rules_markdown.splitlines()
    output: list[str] = []

    for line in lines:
        stripped = line.strip()

        if stripped == "## Rule activity summary":
            break

        if stripped.startswith("// ae-rule-id:") or stripped.startswith(
            "// ae-stats:"
        ):
            continue
        if stripped.startswith("//   applied_decisions="):
            continue

        output.append(re.sub(r"\s+// ae-stats:.*$", "", line))

    result = "\n".join(output)
    return f"{result}\n" if result else ""


def _contains_ae_annotations(rules_markdown: str) -> bool:
    for line in rules_markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("// ae-rule-id:") or stripped.startswith(
            "// ae-stats:"
        ):
            return True
        if re.search(r"\s+// ae-stats:.*$", line):
            return True
    return False


__all__ = [
    "AggregatedRunStats",
    "RuleAggregate",
    "RuleCandidateAggregate",
    "RuleConditionAggregate",
    "annotate_rules_with_run_stats",
]
