# Current Architecture

This document describes the **current repository architecture** as it exists in code today. It is intentionally descriptive rather than aspirational.

The repository currently contains two Python packages:

- `answer_engineering`: the main runtime and rule system
- `ae_paper_reproduction`: the notebook- and paper-oriented reproduction
  package built on top of the runtime

The architectural center of gravity is the `answer_engineering` package. The reproduction package is a separate application layer for experiments, reports, and paper artifacts.

---
## Architectural maturity model

This repository currently operates with two different architectural maturity levels.

The inner runtime (`answer_engineering`) has relatively stable boundaries, a narrow public API, and explicit subsystem responsibilities.

The surrounding experiment and evaluation layers (`ae_paper_reproduction`) are currently more flexible and are more shaped by ongoing research workflows.

This difference is expected at the current stage of system evolution.

The architecture is better understood primarily as:

- partially converged
- intentionally transitional
- actively evolving

not as primarily:

- accidental
- legacy-constrained

---


## 1. Public package boundaries

The root public import surface of `answer_engineering` is intentionally narrow.

Today it exports:

- `GenerationRuntime`
- `GenerationRequest`
- `GenerationPolicy`
- `GenerationResult`
- `CompiledRules`

This is the main external entrypoint for callers that want to execute one rule-guided generation run.

There is also a dedicated public telemetry surface at `answer_engineering.telemetry` for downstream code that consumes runtime telemetry snapshots and selected event records.

---

## 2. Main package layout inside `answer_engineering`

The main package is currently divided into these functional areas.

### `rules/`

Owns the Markdown domain-specific language and compilation pipeline.

Current responsibilities:

- parse rules Markdown into abstract-syntax-tree values
- compile abstract-syntax-tree values into executable `PlanIR`
- expose the convenience boundary `CompiledRules`

Current supported rule families:

- `Replace`
- `After`
- `Avoid`
- `Force`

The compiler emits `RulePlan` / `PlanIR` objects that are later consumed by runtime execution.

### `inference/`

Owns model-backed generation and the decode-side runtime.

Current responsibilities:

- materialize tokenizer and model resources
- build prompts and input ids
- run the greedy decode loop
- bridge decode state to the core plan runner
- package the final `GenerationResult`

The central runtime class is `GenerationRuntime`.
Per-call execution is coordinated by `StreamSession`.
Token-by-token decoding is owned by `GreedyDecoder`.

### `engine/`

Owns rule execution against the current text under construction.

Current responsibilities:

- proposal generation
- guard evaluation and match logic
- candidate scoring
- conflict resolution / selection
- patch application
- runtime telemetry event generation

The orchestration center is `PlanRunner` together with stage modules under `engine/orchestration/stages/`.

Supporting subareas currently include:

- `proposal/`
- `scoring/`
- `selection/`
- `patching/`
- `runtime/`
- `pipeline/`
- `telemetry/`

### `config/`

Owns typed defaults and policy defaults used by runtime and rules.

### `infra/`

Currently small. It contains console output support used by streaming and debug flows.

---

## 3. Runtime execution boundary

The current public execution entrypoint is:

`GenerationRuntime.generate(request, policy)`

At a high level the execution stack is:

1. `GenerationRuntime.generate(...)`
2. `StreamSession.run()`
3. prompt construction and input-id preparation
4. `GreedyDecoder.decode()`
5. optional rule execution via `ExecutionSession` and `PlanRunner`
6. telemetry aggregation
7. `GenerationResult`

This means the runtime is not only a post-processing layer.
The engine can intervene while generation is in progress.

---

## 4. Current rule-execution architecture

The rule system is compiled before execution.

Current flow:

1. authored Markdown rules
2. `MarkdownRulesParser`
3. abstract-syntax-tree values such as `ReplaceRuleAST`, `AfterRuleAST`, `AvoidRuleAST`,
   `ForceRuleAST`
4. `FullPlanCompiler`
5. executable `PlanIR`
6. `PlanRunner` during runtime

The compiled plan currently carries:

- scope
- anchors
- edit target behavior
- guard expressions
- candidate specs
- fire policy
- per-rule execution policy

---

## 5. Current edit model

The runtime currently operates on text edits expressed through shared patch operations.

Current patch operation vocabulary:

- `REPLACE`
- `INSERT_BEFORE`
- `INSERT_AFTER`
- `DELETE`
- `NOOP`

In practice, the currently compiled rule families mostly materialize as replace-style runtime behavior, including `After` and `Force`, which are compiled into replace-oriented plans over derived target spans.

Patch application and proposal normalization are owned by the `engine` package, not by `inference`.

---

## 6. Current telemetry architecture

Telemetry is currently an engine-owned concern.

Current layers:

- event emission during proposal / scoring / apply flow
- event sinks under `engine.telemetry.events`
- aggregation under `engine.telemetry.aggregation`
- immutable snapshots under `engine.telemetry.snapshots`

Decode-side execution records events when rules are enabled, then aggregates them into a `RuntimeTelemetrySnapshot` that is attached to the public `GenerationResult`.

---

## 7. Current reproduction package role

`ae_paper_reproduction` is currently a separate package that builds on top of `answer_engineering`.

It currently owns application-level concerns such as:

- notebook extraction
- subrun planning
- dataset integration
- evaluation sessions
- report assembly
- paper tables and experiment telemetry exports

It is not the runtime itself.
It is the experiment / analysis layer around the runtime.

---

## 8. Architectural character of the current codebase

The current architecture is strongest in the main runtime package:

- narrow public API
- explicit subpackages
- deterministic control orientation
- typed dataclasses and explicit contracts

The reproduction package is broader and more application-shaped. It is useful and functional, but currently less architecturally tight than the core runtime package.

---

## Why the current architecture is suboptimal

The current architecture is not considered optimal yet.

Its structure reflects experiment sequencing: capabilities were added in research exploration order, and boundaries followed immediate research needs rather than a single preplanned module map.

Evaluation workflows also influenced design. The surrounding experiment stack had to support rapid iteration for notebook-driven and report-oriented validation, which shaped subsystem seams around workflow milestones.

Reproducibility requirements further shaped boundaries. Components were organized to preserve stable experiment execution and comparison behavior even when that added transitional adapters.

This is primarily the result of research-driven development, not production legacy constraints.

---

## How architectural improvement is evaluated

Improvement can be recognized by observable signals such as:

- fewer modules requiring simultaneous edits for one capability
- clearer subsystem ownership boundaries
- reduced cross-layer imports
- simpler explanation of system structure
- stronger alignment between code and documentation
- more stable public runtime interfaces

Architectural improvement here does not require one frozen end-state design. It requires measurable convergence in boundaries, contracts, and explainability while preserving flexibility for ongoing research and evaluation needs.

---

## 9. What this document does not claim

This document does **not** claim that the architecture is already fully converged or convention-perfect.

In particular:

- some boundaries are still heavier than ideal
- some internals remain in active refactor
- the reproduction package is still a moving target

Those topics belong in gap documents, not in this current-state description.
