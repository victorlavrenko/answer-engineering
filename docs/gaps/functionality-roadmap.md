# Functionality Roadmap

This document describes capability gaps between the current system and the intended architecture.

Each item represents a structural improvement rather than a minor feature.

## Architectural convergence context

This roadmap tracks capability gaps, but each capability gap also creates architectural boundary pressure.

For this repository stage, capability progress and architectural convergence should be read together:

- capability expansion may require boundary adjustments
- boundary adjustments should improve ownership clarity and extension cost
- the target is not one fixed final architecture, but observable convergence in subsystem contracts

The sections below therefore describe capability deltas and their architectural consequences.

## Capability gap: Causal trajectory repair

Current:

Violations are handled near the generation frontier.

Target:

The system should identify earlier causal mistakes and repair them instead of repeatedly constraining downstream generation.

Why it matters:

Many late failures originate from earlier incorrect framing decisions.

Architectural consequence:

Trajectory-repair capability will pressure decode/orchestration boundaries and should be evaluated with explicit ownership of causal-state handling.

Next steps:

-   violation-to-cause attribution
-   upstream repair generation
-   repair scoring
-   continuation from corrected trajectory

------------------------------------------------------------------------

## Capability gap: Alternative trajectory tracking

Current:

    Only the selected trajectory is preserved.

Target:

    Multiple valid trajectories should be observable.

Why it matters:

    Enables transparency and evaluation of decision paths.

Architectural consequence:

    Alternative trajectory support will pressure runtime state, telemetry contracts, and 
    comparison/reporting seams, so boundary evolution should remain explicit.

Next steps:

-   trajectory branching representation
-   branch selection policies
-   branch comparison metrics

------------------------------------------------------------------------

## Capability gap: Self-hosting architecture

Current:

    Parts of the codebase still violate internal conventions.

Target:

    The system should generate and refactor code that complies with its own architectural rules.

Why it matters:

    Enables continuous architectural improvement and trust in generated systems.

Architectural consequence:

    Self-hosting capabilities will pressure convention-boundary enforcement surfaces and
    should converge through clearer contracts, not ad hoc cross-layer coupling.

Next steps:

-   boundary-aware code generation
-   convention compliance checks
-   automated refactoring assistance
