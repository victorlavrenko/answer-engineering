"""Canonical patch payload and hash helpers.

Purpose:
    Normalize payload text and construct stable patch bytes/hashes used for
    deterministic identity, events, and caching.

Architectural role:
    Runtime patch identity utility shared by proposal and apply stages.

"""

from __future__ import annotations

from hashlib import sha1

from answer_engineering.engine.patching import patcher
from answer_engineering.engine.runtime.runtime_types import (
    PatchOp,
    SpanAbs,
)

normalize_insert_payload = patcher.normalize_insert_payload
normalize_replace_payload = patcher.normalize_replace_payload


def canonicalize_payload(
    *,
    op: PatchOp,
    payload: str | None,
    text: str,
    span_abs: SpanAbs | None,
    apply_spacing: bool = True,
) -> str | None:
    """Normalize proposal payload text according to operation semantics.

    Purpose:
        Convert raw patch payload fields into the canonical textual
        representation used for application, hashing, and telemetry identity.

    Architectural role:
        Patching boundary between proposal-specific payloads and
        operation-neutral patch records.

    Inputs (architectural provenance):
        Receives a patch operation and the proposed text payload emitted by
        proposal construction.

    Outputs (downstream usage):
        Returns canonical payload text consumed by patch serialization and patch
        application.

    Invariants/constraints:
        Canonicalization must preserve semantic edit intent while removing
        incidental representation differences that would destabilize hashes or
        comparisons.

    """
    if op in {PatchOp.NOOP, PatchOp.DELETE}:
        return None
    if span_abs is None:
        raise ValueError("span_abs required for payload canonicalization")
    value = payload or ""
    start, end = span_abs
    if not apply_spacing:
        return value
    if op == PatchOp.REPLACE:
        return normalize_replace_payload(text, start, end, value)
    if op == PatchOp.INSERT_BEFORE:
        return normalize_insert_payload(text, start, value)
    if op == PatchOp.INSERT_AFTER:
        return normalize_insert_payload(text, end, value)
    return value


def patch_bytes(
    op: PatchOp, span_abs: SpanAbs | None, payload_norm: str | None
) -> bytes:
    """Serialize a canonical patch tuple used for hashing and event identity.

    Purpose:
        Produce the stable byte representation of one canonical patch.

    Architectural role:
        Identity boundary for patch hashes, telemetry references, and
        deterministic comparisons.

    Inputs (architectural provenance):
        Receives canonical patch fields after payload normalization.

    Outputs (downstream usage):
        Returns UTF-8 bytes consumed by `patch_hash` and any identity-sensitive
        diagnostics.

    Invariants/constraints:
        Field order and separators are part of the identity contract and should
        only change with an intentional patch-identity migration.

    """
    span = "-" if span_abs is None else f"{span_abs[0]}:{span_abs[1]}"
    payload = "" if payload_norm is None else payload_norm
    return f"op={op.value}|span={span}|payload={payload}".encode()


def patch_hash(payload_bytes: bytes) -> str:
    """Return SHA1 hash for canonical patch bytes.

    Purpose:
        Compute a compact deterministic identifier for a canonical patch.

    Architectural role:
        Patch identity helper used by proposal, telemetry, and
        conflict-resolution paths.

    Inputs (architectural provenance):
        Receives canonical patch fields that have already passed payload
        normalization.

    Outputs (downstream usage):
        Returns the hexadecimal SHA1 digest used as a stable patch identifier.

    Invariants/constraints:
        Hashing is identity-oriented, not security-oriented. Callers must
        canonicalize semantic fields before invoking this helper.

    """
    return sha1(payload_bytes).hexdigest()
