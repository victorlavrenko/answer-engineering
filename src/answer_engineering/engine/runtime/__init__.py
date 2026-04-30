"""Runtime domain-model package.

Purpose:
    Own immutable runtime value types and runtime-local helpers shared across
    proposal, scoring, patching, and orchestration flows.

Owns now:
    - Immutable runtime value records (document/view/alignment/decision-facing).
    - Scoped text-view construction and scope-window derivation helpers.
    - Tokenizer-offset and text/token alignment helpers.
    - Runtime primitives reused across proposal/scoring/patching orchestration.

Does not own:
    - Patch application algorithms.
    - Proposal-planning policy.
    - Selection/conflict-resolution policy.

"""
