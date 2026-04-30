"""Define the aggregation boundary for reproduction outputs.

Purpose:
    Convert per-subrun evaluation and runtime telemetry outputs into comparison
    rows, merged rule statistics, and reporting-ready structures.

Architectural role:
    Downstream aggregation package between evaluation results and reporting
    artifacts.

Inputs (architectural provenance):
    Consumes evaluation results and per-generation telemetry produced by subrun
    execution.

Outputs (downstream usage):
    Aggregated run statistics, comparison rows, and helper renderers used by
    summary and reporting code.

Invariants/constraints:
    Aggregation should stay downstream of evaluation and should not reach back
    into planning or model execution.

"""
