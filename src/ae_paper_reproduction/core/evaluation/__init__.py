"""Define the evaluation boundary for reproduction runs.

Purpose:
    Collect the types and pure computations that turn generated answers into
    correctness judgments, accuracy summaries, and comparison reports.

Architectural role:
    Domain package for correctness assessment and report construction.

Inputs (architectural provenance):
    Consumes dataset rows, generated answers, and gold-matching programs
    produced by planning and execution stages.

Outputs (downstream usage):
    Evaluation records, accuracy reports, and comparison summaries consumed by
    aggregation and session summarization.

Invariants/constraints:
    Evaluation code should stay deterministic and should not own model execution
    or artifact publishing.

"""
