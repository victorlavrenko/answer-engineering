"""Render verbose debug views of aggregated rule telemetry.

Purpose:
    Format merged rule statistics into developer-oriented text that exposes
    condition trees, candidate choices, and aggregate counters more explicitly
    than the human-facing summary renderer.

Architectural role:
    Debug-only rendering module in the aggregation boundary.

Inputs (architectural provenance):
    Consumes merged rule aggregates produced by telemetry aggregation.

Outputs (downstream usage):
    Verbose textual reports used for investigation and manual inspection.

Invariants/constraints:
    These renderers should remain downstream views over already-computed stats
    rather than recomputing aggregation logic.

"""

from __future__ import annotations

from ae_paper_reproduction.core.aggregation.rule_stats import (
    AggregatedRunStats,
    RuleAggregate,
)


def render_debug_rule_summary(rule: RuleAggregate) -> str:
    """Render a verbose debug view for one merged rule aggregate.

    Purpose:
        Expand one merged rule into multi-line text that exposes raw counters,
        condition details, and candidate choice information for manual
        inspection.

    Architectural role:
        Debug renderer layered on top of telemetry aggregation outputs.

    Inputs (architectural provenance):
        Consumes one merged rule aggregate.

    Outputs (downstream usage):
        A list of debug-friendly text lines for one rule.

    Invariants/constraints:
        This formatter should expose internal detail more aggressively than the
        human-facing summary renderer.

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
    parts: list[str] = [
        "// ae-debug:"
        f" fired in {rule.fired_generations}/{rule.total_generations}"
        " generations "
        f"({fired_rate:.1%}), average repeat: {avg_repeat:.2f}"
    ]
    for candidate in rule.candidate_choices:
        parts.append(
            "//   candidate "
            f"kind={candidate.kind} id={candidate.candidate_id} "
            f"label={candidate.label!r} chosen={candidate.chosen} "
            "chosen_generations="
            f"{candidate.chosen_generations}"
        )
    for condition in rule.conditions:
        parts.append(
            "//   condition "
            f"section={condition.section} op={condition.operator} "
            f"expression={condition.expression!r} "
            f"matched={condition.matched} seen={condition.seen} "
            f"matched_generations={condition.matched_generations} "
            f"matched_fired_generations={condition.matched_fired_generations} "
            f"matched_while_fired={condition.matched_while_fired}"
        )
    return "\n".join(parts)


def render_debug_rules_report(run_stats: AggregatedRunStats) -> str:
    """Render a full debug report for merged run statistics.

    Purpose:
        Walk the merged run-level telemetry summary and produce a verbose text
        report across all rules for troubleshooting and validation.

    Architectural role:
        Top-level debug formatter for telemetry aggregation outputs.

    Inputs (architectural provenance):
        Consumes one `AggregatedRunStats` object.

    Outputs (downstream usage):
        A complete text report suitable for debugging or manual review.

    Invariants/constraints:
        The report should reflect the merged stats exactly rather than
        recomputing them differently.

    """
    lines: list[str] = []
    for rule in run_stats.rules:
        lines.append(
            f"// ae-rule-id: {rule.rule_id} rule_name={rule.rule_name}"
        )
        lines.append(render_debug_rule_summary(rule))
    return "\n".join(lines)


__all__ = ["render_debug_rule_summary", "render_debug_rules_report"]
