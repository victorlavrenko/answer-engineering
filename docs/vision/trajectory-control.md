# Trajectory Control Vision

This document describes the **trajectory-control direction** of Answer Engineering.

It is intentionally aspirational, but it starts from the mechanisms that already exist.

---

## Architectural evolution constraints

Trajectory control is central to why the architecture remains intentionally open in selected areas.

Today, the core runtime boundary is relatively disciplined, while parts of the surrounding experimentation and reporting stack remain more research-shaped.

That is expected for the current stage: trajectory-control capabilities are still being validated through experiment workflows, and architectural boundaries should continue to adapt to research outcomes, operational signals, and customer demand validated by revenue.

So this vision should be read as property-oriented direction, not a fixed final module map.

---

## 1. Core idea

Generation should not be treated as an untouchable stream of tokens. It should be possible to intervene in it using explicit, inspectable, protocol-shaped control.

In this project, trajectory control means:

- observe the evolving answer
- detect protocol-relevant situations
- generate or select corrective actions
- apply those actions deterministically
- continue generation from the repaired state

The goal is not just prettier text. 
The goal is bounded, auditable, protocol-constrained output.

---

## 2. Current starting point

The current system already has real trajectory-control mechanisms. Today it can already do all of the following:

- compile authored rules into executable plans
- evaluate rules during generation
- generate proposals deterministically
- score proposals
- resolve conflicts deterministically
- apply text edits during generation
- rebuild runtime state and continue

Current rule families provide the current control vocabulary:

- `Replace`
- `After`
- `Avoid`
- `Force`

So the vision is not starting from zero.
It is extending an existing intervention runtime.

---

## 3. Immediate conceptual target

The long-term target is not merely “more rules”. The target is a richer control model in which the system can reason about:

- where a violation appeared
- what earlier commitment caused it
- which repairs are valid
- whether multiple trajectories remain plausible
- how uncertainty should be surfaced

---

## 4. Causal trajectory repair

This is the most important future extension.

### Problem

Many violations show up late, but their true cause is earlier.

Examples of the general pattern:

- a bad diagnosis frame leads to a bad treatment sentence
- a misleading sales framing leads to a later compliance breach
- a wrong architectural assumption leads to a bad code block much later

In these cases, repeatedly constraining the frontier may be inferior to repairing the earlier cause.

### Target behavior

The system should eventually be able to:

- identify likely upstream causes of a late violation
- propose a minimal earlier repair
- preserve as much valid downstream text as possible
- resume generation from the corrected trajectory

### Why this matters

This is not mainly a cache optimization idea. It is a **causal repair** idea.

The aim is to fix the cause, not just the symptom.

---

## 5. Alternative trajectories

Today the runtime preserves the selected trajectory. In the future, the system should be able to retain awareness that multiple valid continuations or repairs existed.

Potential future capabilities:

- explicit branch representation
- branch comparison
- branch-aware scoring and selection
- branch-aware telemetry

This would make the system more transparent and easier to evaluate.

---

## 6. Tokens of doubt

A more speculative future capability is explicit uncertainty signaling.

The idea is that if multiple materially different but valid trajectories existed, the system may preserve that fact rather than hiding it behind one apparently certain output.

Possible future forms include:

- explicit uncertainty markers
- branch-aware metadata
- special tokens or supervision schemes representing alternative valid
  trajectories

This is not implemented today.
It is a research direction.

---

## 7. Partial-history editing

Another future direction is more selective history repair.

Today the runtime can apply local text edits and then rebuild the visible prefix / cache state to continue generation. That is already useful.

The future extension is more ambitious:

- repair an earlier causal span
- preserve unaffected later material where valid
- avoid treating all later text as equally disposable

This again is not just efficiency work.
It is about better semantic repair.

---

## 8. Multi-rule and protocol-level control

A mature trajectory-control system should eventually do more than fire single isolated rules.

It should support:

- interaction between rules
- cause-directed repair before blind downstream search
- protocol-level reasoning over multiple valid actions
- richer explanations of why one path was chosen

The current engine already has deterministic proposal ordering and conflict resolution. The vision is to extend that toward more global control over trajectories.

---

## 9. Domain significance

These ideas matter most in domains where a response must remain both useful and constrained.

Especially important domains include:

- medicine
- finance and insurance
- regulated sales
- legal and compliance communication
- code generation and architecture guidance
- safety-critical operations

In such domains, the best output is often not the most verbose or the most creative one.
It is the most protocol-faithful usable one.

---

## 10. Relationship to self-hosting

One long-term implication is that the system should eventually help generate and repair code and architecture in its own style.

That means trajectory control is not only for prose or clinical answers. It is also relevant to:

- code generation
- architecture-aware refactoring
- convention-compliant self-improvement

The project should eventually be able to become a customer of its own methods.

---

## 11. What is already true vs not yet true

Already true today:

- runtime intervention during generation
- deterministic proposal / conflict behavior
- replace-style repair mechanisms
- avoid-style safety intervention
- runtime rebuild and continuation after edits

Not yet true today:

- causal trajectory repair
- preserved alternative trajectory structure
- tokens of doubt
- fine-grained history-preserving continuation after upstream repair

That is the correct current baseline for this vision.
