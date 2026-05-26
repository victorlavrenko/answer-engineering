# Trajectory-Control Taxonomy

This document defines the current terminology used for Answer Engineering (AE)
trajectory control. It is intended to keep paper language, documentation, and
code-facing discussions aligned without overstating what the current runtime
implements.

Read this document as a taxonomy for the current system, not as a claim that AE
is a new model architecture, a new decoding algorithm, or a complete formal
verification system.

---

## Runtime control surface at a glance

```text
Prompt / scaffold control
            ↓
Trajectory control (Answer Engineering)
            ↓
Workflow / guardrail control
```

AE occupies the trajectory-control layer. It operates on the evolving visible
trajectory during generation rather than only before generation (prompting) or
after generation (workflow policies and guardrails).

Current runtime flow:

```text
Prompt → Decode → Trigger → Intervention → Rebuild → Continue
```

This summarizes the current execution character of Answer Engineering.

---

## 1. Scope

AE is a runtime layer for protocol-constrained generation. It observes the
visible assistant trajectory during decoding, applies authored rule logic, and
continues generation from the resulting visible prefix.

The current runtime is best described as:

- model-backed greedy decoding
- with optional deterministic rule intervention
- with runtime text edits
- with telemetry over rule execution and applied decisions
- with prefix / KV-cache rebuild after visible edits

It should not be described as:

- retraining
- fine-tuning
- hidden-state steering
- generic post-processing
- a general constrained-decoding framework
- a complete branch-aware search runtime
- a causal-repair system in the strong future sense

---

## 2. Core terms

| Term | Meaning in AE | Current runtime mechanism | Telemetry / observable signal |
| --- | --- | --- | --- |
| **Trajectory** | The evolving visible assistant answer produced during one generation call. | The generated assistant text and token ids managed by the decode state. | Final `GenerationResult.text`; runtime snapshot attached as `GenerationResult.ae_telemetry`. |
| **Trajectory control** | Runtime intervention on the evolving visible answer before generation is complete. | The decode loop generates a token, optionally runs rule logic, edits visible text if needed, rebuilds state, and continues. | Runtime events, triggered/applied rule counts, applied decisions. |
| **Trajectory editing** | A concrete change to already generated visible text. | Patch operations such as replace-style edits applied by the engine. | Applied decisions and candidate-level telemetry when rules fire. |
| **Repair** | A trajectory edit whose purpose is to make the visible answer more protocol-compliant. | Candidate proposal, scoring, selection, and patch application. | Applied decisions; per-rule and per-candidate telemetry snapshots. |
| **Intervention** | Any runtime action that changes or constrains the trajectory because a rule fired. | Rule execution through the compiled plan and `PlanRunner`. | Rule-triggered and rule-applied counts. |
| **Protocol signal** | A visible cue or rule-relevant condition indicating that the answer may need enforcement or repair. | Rule anchors, guards, scopes, and match logic in compiled rules. | Condition / rule telemetry where available. |
| **Causal span** | A span suspected to be the upstream cause of a later protocol violation. | Not a full implemented current capability. Current replace-style edits can target spans, but general cause-directed repair is a future direction. | Not yet a stable telemetry concept. |
| **Branch** | An alternative possible continuation or repair path. | Not represented as a persistent first-class runtime object today. Beam-style probing may generate alternatives for candidate selection, but the selected trajectory is what persists. | Candidate alternatives may be visible in proposal/scoring telemetry; branch telemetry is future work. |
| **Rollback** | Returning to an earlier prefix or target position to generate/select a valid continuation. | Local rollback / candidate probing where supported by the intervention pipeline. | Candidate generation and applied-decision telemetry. |
| **Forcing** | Inserting or ensuring required protocol text appears in the trajectory. | `Force` / `After`-style rule behavior currently compiles into replace-oriented executable plans over derived spans. | Applied decision showing inserted or selected protocol text. |
| **Avoidance** | Preventing or repairing prohibited text. | `Avoid` rules detect disallowed text and route through proposal / patch logic. | Triggered and applied rule events for the relevant rule. |

---

## 3. Mapping to code-level concepts

The following mappings are intentionally approximate and documentation-facing.
They identify where a reader should look, not a frozen internal API contract.

| Taxonomy concept | Code-facing concept |
| --- | --- |
| Generation call | `GenerationRuntime.generate(request, policy)` |
| Runtime policy | `GenerationPolicy` |
| Input request | `GenerationRequest` |
| Runtime result | `GenerationResult` |
| Rule set | `CompiledRules` |
| Decode loop | `GreedyDecoder.decode()` |
| Per-call session | `StreamSession.run()` |
| Rule execution | `ExecutionSession.apply_step(...)` and `PlanRunner` |
| Compiled executable rule plan | `PlanIR` / `RulePlan` |
| Rule families | `Replace`, `After`, `Avoid`, `Force` |
| Patch operation vocabulary | `REPLACE`, `INSERT_BEFORE`, `INSERT_AFTER`, `DELETE`, `NOOP` |
| Runtime telemetry | `RuntimeTelemetrySnapshot` and related telemetry snapshot types |
| Reporting serialization | `ae_paper_reproduction.telemetry.telemetry_types.serialize_runtime_telemetry(...)` |

---

## 4. Relation to nearby control surfaces

### Constrained decoding

Constrained decoding usually restricts the next-token search space or grammar at
decoding time. AE does not currently claim to be a general constrained-decoding
algorithm. Its current mechanism is rule-triggered runtime intervention on the
visible trajectory, including edits to already generated text and continuation
from the repaired prefix.

### Guardrails

Guardrails often validate, block, or rewrite model outputs at API or
post-processing boundaries. AE is closer to an online intervention runtime: it
can act while the answer is being generated, records intervention telemetry, and
continues generation after edits. It is not merely final-output filtering.

### Validation

Validation checks whether some generated content satisfies a condition. AE uses
validation-like signals, but the distinctive runtime behavior is that a detected
condition can lead to candidate generation, scoring, selection, patching, and
continued decoding.

### Expert systems

AE can encode protocol knowledge outside model weights, but it is not a classic
expert system. The language model still supplies fluent generation and model
likelihood scoring; authored rules supply localized protocol constraints and
repair opportunities.

### Prompting

Prompting influences the model through the initial context. AE can also insert
or repair visible text after generation begins. Prompting and AE are therefore
complementary rather than equivalent.

---

## 5. Non-goals and non-claims

The current AE system does not claim that:

- rule-guided generation is always correct
- authored protocol rules are complete
- telemetry proves semantic correctness by itself
- the runtime formally verifies medical, legal, or financial advice
- edits preserve every hidden model state from the pre-edit continuation
- current implementation maintains persistent alternative branches
- current implementation performs general causal diagnosis of all failures
- current implementation is the only or best control surface for LLM output

The narrower claim is that AE provides an inspectable runtime mechanism for
localized, rule-triggered, protocol-oriented intervention during generation.

---

## 6. Reviewer-risk clarifications

### Risk: AE is mistaken for generic guardrails

Clarification: AE is not just a final validator or post-hoc rewrite layer. The
current runtime can intervene during token-by-token decoding, edit visible text,
rebuild the prefix/cache state, and continue generation from the repaired
trajectory.

### Risk: AE is mistaken for constrained decoding

Clarification: AE does not primarily define a token-level admissible grammar. It
uses authored rules to detect protocol-relevant situations and apply selected
trajectory edits. The control surface is the visible generated trajectory, not
only the next-token candidate set.

### Risk: AE overclaims causal repair

Clarification: causal repair is a direction, not a fully implemented current
capability. Current edits may repair earlier spans, but persistent branch
tracking, general upstream-cause identification, and history-preserving causal
continuation remain future work.

### Risk: AE is mistaken for hidden-state steering

Clarification: current AE changes visible text and rebuilds decode state from the
edited prefix. It does not directly modify model weights or hidden activations.

---

## 7. Recommended terminology

Use these phrases for current behavior:

- runtime trajectory control
- visible-prefix trajectory editing
- rule-triggered intervention
- protocol-oriented repair
- deterministic rule execution
- edit-triggered prefix / cache rebuild
- runtime telemetry over interventions

Avoid these phrases unless explicitly discussing future work or non-goals:

- full causal repair
- branch-aware decoding runtime
- formal verification
- hidden-state editing
- complete constrained decoding
- guaranteed correctness

---

## 8. Relationship to vision docs

This taxonomy describes current terminology and current implementation claims.
The trajectory-control vision document describes future directions such as
causal trajectory repair, preserved alternative branches, tokens of doubt, and
more selective history repair.

When the two documents differ in tone, this taxonomy should be treated as the
safer source for current paper terminology and reviewer-facing claims.
