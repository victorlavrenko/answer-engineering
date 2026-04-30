"""Probe cache record types and cache-key formatting helpers.

Owns:
    - ``ProbeCacheKey`` identity fields for reusable probe-set requests.
    - ``ProbeCacheCandidate`` replay payload records.
    - String cache-key formatting helpers.

Does not own:
    - In-memory cache lifecycle or lookup/store flow (owned by
      ``ProbeRuntime.pivot_cache``).

"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProbeCacheKey:
    """Immutable identity for one reusable probe-set request.

    Purpose:
        Capture the fields that make one probe-generation result reusable across
        repeated calls with the same effective prefix and generation controls.

    Architectural role:
        Value object for probing cache lookup.

    Inputs:
        Built by probing runtime code from a resolved probe prefix snapshot,
        span start, and beam-generation settings.

    Outputs:
        Used to index cached probe candidates and avoid unnecessary
        regeneration.

    """

    abs_start: int
    prefix_hash: str
    probe_num_beams: int
    probe_max_new_tokens: int

    def cache_key_id(self) -> str:
        """Return the stable string cache key id for this probe key."""
        return (
            f"{self.abs_start}:{self.prefix_hash}:"
            f"{self.probe_num_beams}:{self.probe_max_new_tokens}"
        )


@dataclass(frozen=True, slots=True)
class ProbeCacheCandidate:
    """Minimal cached replay payload for one generated probe candidate.

    Purpose:
        Preserve only the generated text and score needed to replay candidates
        from cache without rerunning model generation.

    Architectural role:
        Value object stored inside the probing cache.

    Inputs:
        Built from raw probing results after beam generation and candidate
        normalization.

    Outputs:
        Consumed by probing runtime code when reconstructing probe results or
        adapting cached candidates into downstream-facing forms.

    """

    text: str
    logprob: float
