"""Candidate-provider interfaces and static candidate source.

Purpose:
    Define proposal-facing request/provision records and the provider boundary
    used by ProposalPlanner to obtain candidate sets.

Architectural role:
    Proposal-internal extension seam between planning and concrete candidate
    sources.

Current architecture notes:
    CandidateProvider is a real proposal boundary. CandidateRequest still
    carries more proposal/precheck detail than an eventual tighter boundary
    should.

Architectural direction:
    Candidate providers should remain narrow sourcing components rather than
    broad decision-making engines.

Why this matters:
    Broad provider responsibilities increase extension cost and blur ownership
    boundaries.

What better would look like:
    Providers supply candidate material through simple contracts while shared
    proposal logic handles global decisions.

How improvement can be recognized:
    - Less duplicated decision logic
    - Fewer lifecycle responsibilities in providers

Open constraint:
    Provider boundaries should remain flexible enough to support new
    candidate-generation experiments.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from answer_engineering.engine.pipeline.context import (
    StepContext,
)
from answer_engineering.engine.proposal.proposal_logic import (
    GenerationPrecheck,
)
from answer_engineering.engine.telemetry.events.event_sink import (
    RuntimeEventSink,
)
from answer_engineering.rules.compile.plan import (
    CandidateSpec,
)


@dataclass(frozen=True, slots=True)
class CandidateRequest:
    """Provider request for one candidate-serving decision.

    Purpose:
        Bundle the active step context, optional proposal precheck, and event
        sink so providers receive one canonical request object.

    Architectural role:
        Input record for the proposal-side CandidateProvider boundary.

    Inputs (architectural provenance):
        Built by ProposalPlanner.generate after proposal precheck and before
        provider provisioning.

    Outputs (downstream usage):
        Consumed by static and avoid providers to produce CandidateProvision
        values.

    Architectural TODO:
        Consider narrowing this record once precheck details no longer need to
        cross the provider seam directly.

    """

    ctx: StepContext
    precheck: GenerationPrecheck | None = None
    event_sink: RuntimeEventSink | None = None


@dataclass(frozen=True, slots=True)
class CandidateProvision:
    """Atomic provider output for one candidate-serving decision.

    Purpose:
        Return the effective StepContext to use for proposal generation together
        with the candidate specs supplied by the provider.

    Architectural role:
        Output record of the proposal-side CandidateProvider boundary.

    """

    ctx: StepContext
    candidates: tuple[CandidateSpec, ...]


class CandidateProvider(ABC):
    """Proposal-side source of candidate edit specifications.

    Purpose:
        Standardize how ProposalPlanner asks different backends whether they
        support a step and what candidates they can provide.

    Architectural role:
        Stable collaboration boundary between proposal planning and concrete
        candidate sources.

    Inputs (architectural provenance):
        Implementations receive StepContext for support checks and
        CandidateRequest for provisioning.

    Outputs (downstream usage):
        Emit CandidateProvision values consumed by shared proposal logic.

    Invariants/constraints:
        Providers own candidate sourcing only. Shared proposal materialization
        stays outside this interface.

    """

    @abstractmethod
    def supports(self, ctx: StepContext) -> bool:
        """Return whether this provider should handle the active step context.

        Purpose:
            Let ProposalPlanner route a rule evaluation to the correct candidate
            source.

        """

    @abstractmethod
    def provide(self, request: CandidateRequest) -> CandidateProvision:
        """Provide candidate specs for one proposal request.

        Purpose:
            Return the provider-specific candidate set and any context
            adjustments required before shared proposal logic runs.

        """


@dataclass(slots=True)
class StaticCandidatesProvider(CandidateProvider):
    """Provider that serves rule-declared static candidates.

    Purpose:
        Return compiled rule candidates for rule families that do not require
        model- backed probing.

    Architectural role:
        Concrete CandidateProvider implementation for replace/after/force-style
        rules.

    """

    name_prefixes: tuple[str, ...]

    def supports(self, ctx: StepContext) -> bool:
        """Return whether a rule name belongs to the static-candidate family.

        Purpose:
            Select the provider for rule prefixes such as replace, after, and
            force that do not require probe generation.

        Architectural role:
            Routing predicate used by planner provider selection.

        Inputs (architectural provenance):
            Receives the active ``StepContext`` from the proposal planner.

        Outputs (downstream usage):
            Boolean determines whether ``provide`` should be called on this
            provider.

        """
        return any(
            ctx.rule.name.startswith(prefix) for prefix in self.name_prefixes
        )

    def provide(self, request: CandidateRequest) -> CandidateProvision:
        """Convert rule-declared candidates into a canonical static provision.

        Purpose:
            Repackage parsed candidate declarations as ``CandidateSpec`` objects
            tagged as static, preserving labels and priorities from the compiled
            rule.

        Architectural role:
            Non-generative candidate source inside the proposal subsystem.

        Inputs (architectural provenance):
            Receives the compiled rule through ``CandidateRequest.ctx``.

        Outputs (downstream usage):
            Returns a ``CandidateProvision`` consumed by shared
            proposal-generation logic.

        """
        ctx = request.ctx
        return CandidateProvision(
            ctx=ctx,
            candidates=tuple(
                CandidateSpec(
                    op=candidate.op,
                    text=candidate.text,
                    kind="static",
                    priority=candidate.priority,
                    label=candidate.label,
                    candidate_id=candidate.candidate_id,
                )
                for candidate in ctx.rule.candidates
            ),
        )
