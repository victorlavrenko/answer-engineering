"""Core immutable runtime value types.

Purpose:
    Define runtime-owned value contracts for scoped text views, alignment, and
    decision-facing outputs passed between proposal, scoring, patching, and
    orchestration stages.

Architectural role:
    Shared runtime domain-model layer for the editing pipeline.

Architectural direction:
    Keep this module as the authoritative runtime value boundary while
    continuing to tighten ownership splits between runtime, proposal, and
    patching value concerns.

Why this matters:
    Value ownership is mostly explicit, but some seams remain transitional
    across neighboring packages.

What better would look like:
    Higher layers depend on stable runtime values through narrow semantics
    rather than duplicating adjacent value-shaping logic.

How improvement can be recognized:
    - Clearer value ownership boundaries across runtime/proposal/patching
    - Fewer adapter-like value conversions across neighboring layers
    - Stronger agreement between declared contracts and actual usage

Open constraint:
    Value contracts must remain responsive to evolving execution and telemetry
    requirements.

"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from answer_engineering.engine.patching.proposals import (
    PatchProposal,
)
from answer_engineering.engine.runtime import (
    document_state as _document_state,
)
from answer_engineering.engine.runtime import (
    runtime_primitives as _runtime_primitives,
)
from answer_engineering.engine.runtime import scope
from answer_engineering.engine.runtime.runtime_primitives import (
    SpanAbs,
)
from answer_engineering.rules.compile.plan import (
    ScopeSpec,
)

DocumentState = _document_state.DocumentState
PatchOp = _runtime_primitives.PatchOp


@dataclass(frozen=True, slots=True)
class TokenCharAlignment:
    """Align one generated token index to its character span in assistant-.

    Purpose:
        Preserve decode-time token-to-text correspondence for later span and
        token-boundary conversions.

    Invariants:
        ``token_index`` and ``[char_start, char_end)`` refer to the same text
        view used when decode/alignment state was produced.

    """

    token_index: int
    char_start: int
    char_end: int
    piece_text: str


@dataclass(frozen=True, slots=True, init=False)
class TextView:
    """Scoped document slice with absolute-coordinate mapping metadata.

    Purpose:
        Represent the exact text surface used for guard matching and edit-target
        resolution while preserving where that slice sits in the document
        version it came from.

    Architectural role:
        Runtime-owned scoped projection value shared across scope building,
        proposal logic, and telemetry.

    Invariants:
        View-relative spans are valid only against ``text`` from this object and
        map to absolute coordinates via ``abs_start``/``abs_end``.

    Ownership:
        Owned by ``answer_engineering.engine.runtime``.

    """

    document: DocumentState
    abs_start: int
    abs_end: int

    def __init__(
        self,
        doc: DocumentState | None = None,
        spec: ScopeSpec | None = None,
        *,
        abs_start: int | None = None,
        abs_end: int | None = None,
    ) -> None:
        """Build one scoped runtime view from canonical document inputs.

        Purpose:
            Bind a document snapshot, an absolute scope, and the corresponding
            relative text view into a single coordinate-conversion object.

        Architectural role:
            Constructor boundary between document state and rule/proposal logic.
            It gives downstream stages a scoped text projection without losing
            the source document version or absolute-coordinate provenance.

        Inputs (architectural provenance):
            Receives the canonical document text, source version id, absolute
            scope, and scoped visible text from runtime view construction.

        Outputs (downstream usage):
            Stores the view data used by matching, guard evaluation, proposal
            building, and absolute span conversion.

        Invariants/constraints:
            Relative indexes are meaningful only inside the stored scope.
            Absolute conversions must preserve half-open span semantics and
            refer to the stored document version.

        """
        if doc is not None and spec is not None:
            raw, scoped_abs_start = scope.resolve_scoped_text(doc, spec)
            object.__setattr__(self, "document", doc)
            object.__setattr__(self, "abs_start", scoped_abs_start)
            object.__setattr__(self, "abs_end", scoped_abs_start + len(raw))
            return
        if doc is None or abs_start is None or abs_end is None:
            raise TypeError(
                "TextView requires doc + spec or doc + absolute boundaries"
            )
        object.__setattr__(self, "document", doc)
        object.__setattr__(self, "abs_start", abs_start)
        object.__setattr__(self, "abs_end", abs_end)

    @property
    def text(self) -> str:
        """Return the canonical text slice for this scoped view."""
        return self.document.text[self.abs_start : self.abs_end]

    @property
    def base_version_id(self) -> str:
        """Return the source document version identifier for this view."""
        return self.document.version_id

    def to_abs(self, i: int) -> int:
        """Convert a view-relative index to an absolute document index."""
        if i < 0:
            raise ValueError("view offset must be >= 0")
        return self.abs_start + i

    def to_abs_span(self, i0: int, i1: int) -> SpanAbs:
        """Convert a view-relative half-open span to absolute coordinates."""
        if i0 < 0 or i1 < i0:
            raise ValueError("invalid view span")
        return (self.to_abs(i0), self.to_abs(i1))


@dataclass(frozen=True, slots=True)
class Match:
    """Accepted guard match with both absolute and view-relative span.

    Purpose:
        Carry resolved match spans plus captures/provenance so later proposal
        stages can build anchors and edit targets without re-running matching.

    Invariants:
        ``span_abs`` and ``span_view`` identify the same match occurrence in the
        ``TextView`` used by guard evaluation.

    """

    span_abs: SpanAbs
    span_view: tuple[int, int]
    captures: dict[str, str] = field(default_factory=lambda: {})
    provenance: dict[str, str] = field(default_factory=lambda: {})


@dataclass(frozen=True, slots=True)
class Anchor:
    """Named absolute span produced by anchor resolution for proposal targeting.

    Purpose:
        Carry resolved document coordinates referenced by edit-target placement
        logic.

    """

    anchor_id: str
    abs_start: int
    abs_end: int


@dataclass(frozen=True, slots=True)
class AppliedPatch:
    """Immutable applied-patch record for decision and reporting surfaces.

    Purpose:
        Preserve the committed proposal together with the resulting version id
        so final decision assembly can report what changed without re-running
        patching logic.

    """

    patch_id: str
    proposal: PatchProposal
    new_version_id: str


class DecisionSource(Protocol):
    """Protocol for runtime result objects that can be normalized into.

    Purpose:
        Define the minimum read-only surface (final document, applied patches,
        runtime events) that ``Decision`` can consume from orchestration
        outputs.

    """

    @property
    def final_doc(self) -> DocumentState:
        """Return the final document state produced by this decision source."""
        raise NotImplementedError

    @property
    def applied_patches(self) -> Sequence[AppliedPatch]:
        """Return applied patches in the order they were committed."""
        raise NotImplementedError

    @property
    def events(self) -> Sequence[object]:
        """Return structured runtime events associated with this result."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True, init=False)
class Decision:
    """Normalized final-run output value returned from one orchestration.

    Purpose:
        Provide one immutable output shape that collapses multiple caller result
        shapes into one normalized final-run record.

    Outputs:
        API, decode-session, tests, and reproduction tooling consume this value
        as the run's final text/change summary plus applied patch/event history.

    Non-ownership:
        ``Decision`` is not the full orchestrator domain object; it is the
        normalized output projection for callers.

    Todo:
        Target:
            Centralize runtime completion/result assembly behind one boundary so
            orchestrator internals and API-facing output shaping do not drift.

        Boundary note:
            ``Decision`` currently bridges multiple callers (decode session,
            orchestrator results, tests) that each touch result-shaping
            concerns.

    """

    final_text: str
    applied_patches: tuple[AppliedPatch, ...] = field(default_factory=tuple)
    events: tuple[object, ...] = field(default_factory=tuple)
    changed: bool = False

    def __init__(
        self,
        source: DecisionSource | None = None,
        *,
        final_text: str | None = None,
        applied_patches: Iterable[AppliedPatch] = tuple(),
        events: Iterable[object] = tuple(),
        changed: bool | None = None,
    ) -> None:
        """Build a normalized decision from a runtime result or explicit fields.

        Purpose:
            Support both direct construction from a runtime decision source and
            explicit reconstruction from already-materialized decision fields.

        Architectural role:
            Normalization constructor for the decision facade exposed to
            telemetry and reporting code.

        Inputs (architectural provenance):
            Runtime-result mode reads final document, applied patches, and
            events from a decision source. Explicit mode receives those fields
            from upstream code that has already unpacked them.

        Outputs (downstream usage):
            Stores a canonical final document, applied-patch sequence, and event
            sequence consumed by result assembly and diagnostics.

        Invariants/constraints:
            Callers must provide a complete construction mode. Stored fields
            should represent one coherent runtime decision, not a mix of
            unrelated runs.

        """
        if source is not None:
            if final_text is not None:
                raise TypeError(
                    "final_text cannot be provided when source is set"
                )
            final_text = source.final_doc.text
            applied_patches = source.applied_patches
            events = source.events
            changed = (
                bool(source.applied_patches) if changed is None else changed
            )
        elif final_text is None:
            raise TypeError("Decision requires either source or final_text")

        normalized_patches = tuple(applied_patches)
        object.__setattr__(self, "final_text", final_text)
        object.__setattr__(self, "applied_patches", normalized_patches)
        object.__setattr__(self, "events", tuple(events))
        object.__setattr__(
            self,
            "changed",
            bool(normalized_patches) if changed is None else changed,
        )


class TokenAlignedTextView(Protocol):
    """Protocol for runtime views that expose generated-token alignment.

    Purpose:
        Describe the text-view capabilities needed by probing and proposal code
        when character spans must be related back to generated token positions.

    Architectural role:
        Structural boundary between generic runtime text views and token-aware
        inference helpers.

    Inputs (architectural provenance):
        Implemented by runtime view objects constructed from model-generated
        text and tokenizer alignment metadata.

    Outputs (downstream usage):
        Lets probing-prefix code read scoped text, absolute coordinates, and
        token alignment without depending on a concrete view class.

    Invariants/constraints:
        Character and token coordinates must describe the same underlying
        generated document snapshot.

    """

    assistant_visible_text: str
    generated_token_ids: list[int]
    generated_token_alignment: list[TokenCharAlignment]


__all__ = [
    "Anchor",
    "AppliedPatch",
    "Decision",
    "DocumentState",
    "Match",
    "PatchOp",
    "PatchProposal",
    "SpanAbs",
    "TextView",
    "TokenCharAlignment",
    "TokenAlignedTextView",
]
