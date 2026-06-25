"""Inference defaults shared by the public generation runtime and the probing.

Purpose:
    Collect immutable default values for ordinary generation and short probe
    generation.

Architectural role:
    Configuration leaf module under the inference boundary.

Inputs (architectural provenance):
    Read by GenerationPolicy construction and probing-generation helpers when
    callers do not override those values.

Outputs (downstream usage):
    Supplies concrete numeric defaults for decode length, beam count, diversity,
    and optional grouped-beam loading behavior.

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationDefaults:
    """Default text-generation limits for main runtime generation.

    Purpose:
        Provide the fallback ``max_new_tokens`` value used when constructing the
        normal generation configuration for inference.

    Architectural role:
        Small immutable configuration object shared by runtime bootstrapping and
        generation adapters.

    Inputs (architectural provenance):
        Read when higher-level APIs or notebooks do not specify an explicit
        maximum generation length.

    Outputs (downstream usage):
        Supplies a concrete generation cap to model runtime setup.

    Invariants/constraints:
        This object stores defaults only; it does not own per-request overrides.

    """

    max_new_tokens: int = 512


@dataclass(frozen=True, slots=True)
class ProbeDefaults:
    """Default settings for short probe generation used by avoid-candidate.

    Purpose:
        Collect beam-search and token-limit defaults for probe expansion,
        including grouped-beam settings and optional preloading of the custom
        search backend.

    Architectural role:
        Configuration value object for probing runtime setup inside the proposal
        candidate subsystem.

    Inputs (architectural provenance):
        Consumed when avoid-candidate probing is configured without explicit
        probe parameters from the caller.

    Outputs (downstream usage):
        Produces the effective probing configuration passed to beam-generation
        helpers and runtime loaders.

    Invariants/constraints:
        Settings are immutable defaults; per-run probing state lives elsewhere.

    """

    num_beams: int = 10
    max_new_tokens: int = 8
    beams_per_group: int = 3
    diversity_penalty: float = 2.0
    use_group_beam_search: bool = True
    group_beam_search_repo_id: str = "lavrenko/group-beam-search-v5-cache-fix"
    group_beam_search_revision: str | None = None
    preload_group_beam_search: bool = False


@dataclass(frozen=True, slots=True)
class StreamRenderingDefaults:
    """Default console-rendering policy values for streaming generation output.

    Purpose:
        Centralize the rendering-policy defaults used by decode and
        assistant-visible text synchronization.

    """

    wrap_width: int = 80
    retractable_tail_chars: int = 80
    min_emit_chars: int = 10
    debug_prefix: str = "[AE]"


__all__ = [
    "GenerationDefaults",
    "ProbeDefaults",
    "StreamRenderingDefaults",
]
