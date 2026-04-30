"""Live probing runtime orchestration and probe-set reuse.

Owns now:
    - Prefix snapshot construction for probe requests.
    - Cache lookup/store lifecycle via ``pivot_cache``.
    - Miss-triggered probing generation.
    - Probing debug event emission policy.
    - Adaptation into current proposal-facing output shape.

Current boundary leak:
    The runtime still depends on proposal-layer types (`StepContext`,
    `GenerationPrecheck`, `CandidateSpec`) and therefore is not yet a closed
    probing-native API.

"""

from dataclasses import dataclass, field

import torch

from answer_engineering.engine.pipeline.context import StepContext
from answer_engineering.engine.pipeline.events import AvoidProbeSetGenerated
from answer_engineering.engine.proposal.proposal_logic import GenerationPrecheck
from answer_engineering.engine.runtime.runtime_types import PatchOp
from answer_engineering.engine.telemetry.events.event_sink import (
    DebugEventEmitter,
)
from answer_engineering.inference.model_types import (
    GenerationRuntimeProtocol,
)
from answer_engineering.inference.probing.generation.probe_generator import (
    ProbeResult,
)
from answer_engineering.inference.probing.prefix.request_prefix import (
    ProbeRequestPrefix,
)
from answer_engineering.inference.probing.runtime.probe_cache import (
    ProbeCacheCandidate,
    ProbeCacheKey,
)
from answer_engineering.inference.prompting import prompt_prefix
from answer_engineering.rules.compile.plan import CandidateSpec


@dataclass(frozen=True, slots=True)
class ProbePrefixSnapshot:
    """Canonical snapshot of how a probe prefix was constructed.

    Purpose:
        Preserve the prompt ids, full prefix ids, and whether generated-token
        alignment contributed to the final prefix used for probing and cache
        lookup.

    Architectural role:
        Small probing-runtime value object shared by prefix construction, cache
        identity, and debug output.

    """

    prompt_ids: tuple[int, ...]
    prefix_ids: tuple[int, ...]
    used_generated_alignment: bool


@dataclass(slots=True)
class ProbeRuntime:
    """Runtime owner for live probe generation and probe-set reuse.

    Purpose:
        Build probe-prefix snapshots, reuse or populate probe cache entries,
        emit debug telemetry, and adapt probe outputs into current
        proposal-facing candidates.

    Architectural role:
        Main behavior-owning runtime object in the probing subsystem.

    Ownership:
        Owns ``pivot_cache`` lifecycle, probe-set reuse behavior, and probing
        debug-emission policy.

    Current architecture notes:
        The runtime still depends on proposal-layer types (``StepContext``,
        ``GenerationPrecheck``, ``CandidateSpec``) and therefore is not yet a
        closed probing-native API.

    Architectural direction:
        The probing subsystem is expected to evolve toward a more self-contained
        boundary with clearer ownership of probing logic.

    Why this matters:
        Subsystems that depend heavily on proposal-layer structures are harder
        to extend independently.

    What better would look like:
        New probing strategies could be added without increasing proposal-layer
        coupling.

    How improvement can be recognized:
        - Fewer proposal-layer imports
        - Narrower boundary types
        - Reduced adaptation logic

    Open constraint:
        The final boundary should remain responsive to future probing
        experiments.

    """

    generation_runtime: GenerationRuntimeProtocol | None = None
    pivot_cache: dict[ProbeCacheKey, tuple[ProbeCacheCandidate, ...]] = field(
        default_factory=lambda: {}
    )
    trajectory_debug: bool = False
    debug_emitter: DebugEventEmitter = field(default_factory=DebugEventEmitter)

    def avoid_stream_key(
        self, ctx: StepContext, *, abs_start: int
    ) -> ProbeCacheKey:
        """Build the reusable probe-set cache identity for one edit start.

        Purpose:
            Create the structured key that decides whether avoid probing can
            reuse an existing candidate set or must generate a new one.

        Architectural role:
            Probing-runtime identity boundary. It owns cache-key construction
            rather than leaving proposal providers to infer probe equivalence.

        Inputs (architectural provenance):
            Receives rule identity, edit start, prefix snapshot, generation
            parameters, and document context from the avoid proposal path.

        Outputs (downstream usage):
            Returns an avoid-stream key consumed by probe caches and proposal
            serving code.

        Invariants/constraints:
            Include every input that affects generated candidates and exclude
            values that are diagnostic-only. Otherwise cache reuse becomes
            either unsafe or too fragmented.

        """
        prefix_snapshot = self._build_probe_prefix_snapshot(
            ctx, abs_start=abs_start
        )
        return ProbeCacheKey(
            abs_start=abs_start,
            prefix_hash=prompt_prefix.stable_prefix_fingerprint(
                prefix_snapshot.prefix_ids
            ),
            probe_num_beams=ctx.rule.policy.probe_num_beams,
            probe_max_new_tokens=ctx.rule.policy.probe_max_new_tokens,
        )

    def generate(
        self, ctx: StepContext, *, precheck: GenerationPrecheck | None = None
    ) -> list[CandidateSpec]:
        """Run probing end-to-end for one step and return proposal-facing.

        Flow:
            1) resolve/reuse proposal precheck and validate span/runtime,
            2) apply avoid-floor adjustment when present,
            3) build probe-prefix snapshot,
            4) lookup cache by probe identity,
            5) on miss run probe generation and populate cache,
            6) emit probe debug/telemetry events,
            7) adapt cached probe candidates to ``CandidateSpec`` outputs.

        """
        self.debug_emitter.event_sink = ctx.event_sink
        pre = precheck if precheck is not None else GenerationPrecheck(ctx)
        if pre.span is None:  # TODO: probably debug log this case
            return list()
        if self.generation_runtime is None:
            return list()
        matched_observations = tuple(
            obs for obs in pre.guard_observations if obs.matched
        )
        if self.trajectory_debug:
            matched_str = ", ".join(
                f"expression:{obs.node_type}:{obs.debug_expression}"
                for obs in matched_observations
            )
            if not matched_str:
                matched_str = "none"
            observations: list[str] = []
            for obs in pre.guard_observations:
                match_flag = "1" if obs.matched else "0"
                observations.append(
                    "expression:"
                    f"{obs.node_type}:{obs.debug_expression}={match_flag}"
                )
            observations_str = ", ".join(observations)
            noop_reason = pre.noop_reason or ""
            self._debug(
                "PROBE_RULE_TRIGGER "
                f"rule_id={ctx.rule.rule_id} rule_name={ctx.rule.name} "
                f"matched=[{matched_str}] "
                f"noop_reason={noop_reason}".rstrip(),
            )
            self._debug(f"PROBE_GUARD_MATCHES all=[{observations_str}]")
        edit_span = pre.span
        if ctx.avoid_edit_floor_abs_start is not None:
            floored_start = max(edit_span[0], ctx.avoid_edit_floor_abs_start)
            if floored_start < edit_span[1]:
                self._debug(
                    "PROBE_SCOPE_FLOOR "
                    f"rule_id={ctx.rule.rule_id} "
                    f"old_start={edit_span[0]} "
                    f"new_start={floored_start}",
                )
                edit_span = (floored_start, edit_span[1])
        prefix_snapshot = self._build_probe_prefix_snapshot(
            ctx, abs_start=edit_span[0]
        )
        prompt_ids = list(prefix_snapshot.prompt_ids)
        prefix_ids = list(prefix_snapshot.prefix_ids)
        tok = self.generation_runtime.text_codec()
        if self.trajectory_debug:
            assistant_prefix_ids_len = len(prefix_ids) - len(prompt_ids)
            used_generated_alignment = prefix_snapshot.used_generated_alignment
            decoded_tail = tok.decode(
                prefix_ids[-80:], skip_special_tokens=True
            )
            built_prefix_text = tok.decode(prefix_ids, skip_special_tokens=True)
            escaped_decoded_tail = decoded_tail.encode("unicode_escape").decode(
                "ascii"
            )
            escaped_prefix_text = built_prefix_text.encode(
                "unicode_escape"
            ).decode("ascii")
            self._debug(
                "PROBE_PREFIX\n"
                f"  prompt_ids_len={len(prompt_ids)}\n"
                f"  assistant_prefix_ids_len={assistant_prefix_ids_len}\n"
                f"  full_prefix_ids_len={len(prefix_ids)}\n"
                f"  used_generated_alignment={used_generated_alignment}\n"
                f'  decoded_tail="{escaped_decoded_tail}"\n'
                f'  prefix_build_text="{escaped_prefix_text}"',
            )
            guard_view_text = ctx.guard_view.text.encode(
                "unicode_escape"
            ).decode("ascii")
            self._debug(
                "PROBE_GUARD_SCOPE "
                f"abs_start={ctx.guard_view.abs_start} "
                f"abs_end={ctx.guard_view.abs_end} "
                f"len_chars={len(ctx.guard_view.text)} "
                f"text={guard_view_text}",
            )

        prefix_hash = prompt_prefix.stable_prefix_fingerprint(prefix_ids)
        key = ProbeCacheKey(
            abs_start=edit_span[0],
            prefix_hash=prefix_hash,
            probe_num_beams=ctx.rule.policy.probe_num_beams,
            probe_max_new_tokens=ctx.rule.policy.probe_max_new_tokens,
        )
        cached = self.pivot_cache.get(key)
        self._debug(
            "PROBE_CALL "
            f"rule_id={ctx.rule.rule_id} rule_name={ctx.rule.name} "
            f"abs_start={edit_span[0]} prefix_ids_len={len(prefix_ids)}",
        )
        if cached is None:
            self._debug("PROBE_CACHE miss")
            try:
                probe = ProbeResult(
                    self.generation_runtime,
                    prefix_ids=prefix_ids,
                    num_beams=ctx.rule.policy.probe_num_beams,
                    max_new_tokens=ctx.rule.policy.probe_max_new_tokens,
                    abs_start=edit_span[0],
                    doc_text=ctx.doc.text,
                    trajectory_debug=ctx.trajectory_debug,
                    event_sink=ctx.event_sink,
                )
            except (
                RuntimeError,
                ValueError,
                TypeError,
                torch.cuda.OutOfMemoryError,
            ):
                return list()
            cached = tuple(
                ProbeCacheCandidate(
                    text=c.text,
                    logprob=float(getattr(c, "logprob_sum", float("nan"))),
                )
                for c in probe.beams
                if c.text
            )
            self.pivot_cache[key] = cached

            if ctx.event_sink is not None:
                ctx.event_sink.emit(
                    AvoidProbeSetGenerated(
                        rule_id=ctx.rule.rule_id,
                        cache_key=key.cache_key_id(),
                        generated_count=len(cached),
                        probe_budget=ctx.rule.policy.probe_num_beams,
                    )
                )
        else:
            self._debug("PROBE_CACHE hit")
        return [
            CandidateSpec(
                op=PatchOp.REPLACE,
                text=candidate.text,
                kind="generated",
                priority=100 - idx,
                label=f"probe_{idx + 1}",
                candidate_id=f"probe_{idx + 1}",
                logprob=candidate.logprob,
            )
            for idx, candidate in enumerate(cached)
        ]

    def _build_probe_prefix_snapshot(
        self, ctx: StepContext, *, abs_start: int
    ) -> ProbePrefixSnapshot:
        """Construct the canonical probe-prefix snapshot used by cache and."""
        runtime = self.generation_runtime
        if runtime is None:
            return ProbePrefixSnapshot(
                prompt_ids=tuple(),
                prefix_ids=tuple(ctx.doc.text[:abs_start].encode("utf-8")),
                used_generated_alignment=False,
            )
        tok = runtime.text_codec()
        prompt_ids = (
            [int(token_id.item()) for token_id in ctx.step.prompt_ids[0]]
            if ctx.step.prompt_ids is not None
            else []
        )
        prefix_build = ProbeRequestPrefix(
            tok=tok,
            prompt_ids=prompt_ids,
            doc_text=ctx.doc.text,
            abs_start=abs_start,
            generated_ids=ctx.step.generated_ids,
            generated_token_alignment=ctx.step.generated_token_alignment,
        )
        return ProbePrefixSnapshot(
            prompt_ids=tuple(prompt_ids),
            prefix_ids=tuple(prefix_build.prefix_ids),
            used_generated_alignment=prefix_build.used_generated_alignment,
        )

    def _debug(self, msg: str) -> None:
        """Emit a probing debug message when trajectory debugging is enabled.

        Purpose:
            Keep ProbeRuntime-specific debug routing local so probing code can
            report cache, prefix, and generation decisions without duplicating
            debug-policy checks.

        Architectural role:
            Small telemetry helper inside probing runtime orchestration and the
            central seam for ProbeRuntime debug-emission policy.

        Inputs:
            Receives probing-generated debug text from cache, prefix, and
            generation flow branches.

        Outputs:
            May forward the message to the configured debug emitter; otherwise
            produces no effect.

        """
        if not self.trajectory_debug:
            return
        self.debug_emitter.emit(msg)
