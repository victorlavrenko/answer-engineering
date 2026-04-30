# Convention Compliance Gaps

This document describes the gap between the repository’s stated conventions and its current implementation state.

It is not a theory document.
It is a practical reading of where convergence is still incomplete.

The main convention sources today are:

- `conventions/pyguide.md`
- `conventions/pyguide-extension.md`

---

## 1. What is already aligned

The repository already complies well with several convention themes.

### Package naming is mostly meaningful

The main package layout avoids vague ownership buckets such as `utils` or `common`.

Important top-level areas are named by real responsibility:

- `rules`
- `inference`
- `engine`
- `telemetry`
- `config`

This is already strong alignment with the package-ownership conventions.

### The public surface is intentionally narrow

The root public API of `answer_engineering` is small and explicit. That matches the convention preference for one truthful canonical path.

### Typed dataclass contracts are used heavily

The repository already leans strongly on:

- typed immutable value objects
- explicit dataclass contracts
- narrow public boundary objects

That is real compliance, not aspirational prose only.

### Tooling and tests already exist

The repository already has:

- pyright configuration
- ruff
- pylint
- tests
- golden tests
- architecture-oriented tests

So convention enforcement is not merely verbal.

---

## 2. Where convergence is still incomplete

The remaining gap is not “no conventions”.
It is **partial convergence**.

### Core runtime vs reproduction maturity gap

The main runtime package aligns more strongly with the conventions than the reproduction package.

The current practical reality is:

- `answer_engineering` is closer to the intended style
- `ae_paper_reproduction` contains more application-shaped and transitional
  structure

This is the largest current compliance asymmetry in the repository.

### Boundary cleanliness is not yet uniform

The repository has clear public boundaries, but internal seams are not yet equally clean everywhere.

Practical signs of incomplete convergence include:

- some modules that still feel heavier than one clean ownership boundary
- areas where architecture is still being re-decided by refactors
- documentation that is ahead of final code convergence

### Single-source-of-truth discipline is not yet fully resolved everywhere

The conventions strongly prefer canonical upstream objects and discourage unnecessary flattening or duplicated state.

The repository clearly aims in that direction, but the current codebase still needs continued audit and cleanup to make that discipline uniformly true, especially outside the tightest core runtime paths.

### Refactor completion is uneven

The repository has already undergone major structural changes. That improves the architecture, but it also means some areas are still in post-refactor consolidation rather than final resting shape.

---

## 3. Current high-value compliance targets

These are the most important practical convergence targets.

### Boundary tightening in the reproduction layer

The reproduction package should continue moving toward:

- clearer ownership per module
- less transitional structure
- cleaner separation between planning, execution, aggregation, and export

### Stronger single-source-of-truth enforcement

Continued work is needed to reduce:

- duplicated conceptual state
- unnecessary flattening
- field-by-field transfer where a canonical container should exist

### Continued import-boundary grooming

The conventions prefer imports that preserve subsystem ownership and stable boundaries.

The current repository already reflects that goal in many places, but import patterns should continue to be reviewed as architecture converges.

### Documentation / codebase alignment

The repository already has a large documentation set, but the documentation structure itself still needs convergence so that it reflects current truth, vision, and gaps cleanly.

---

## Convergence gaps reflect experimentation, not disorder

Most convention gaps originate from experiment-driven development.

They do not primarily originate from:

- negligence
- lack of standards
- uncontrolled growth

These gaps are expected during research-heavy phases and should narrow as subsystem ownership stabilizes.

---

## 4. What would be inaccurate to say

It would be inaccurate to say:

- the repository ignores its conventions
- the repository is mostly unconstrained
- the codebase has no architectural discipline

None of those are true.

It would also be inaccurate to say:

- the repository is already fully converged to the convention ideal

That is also not true.

---

## 5. Most accurate summary

The most accurate summary today is:

- the conventions are real
- major parts of the codebase already follow them well
- the core runtime is more converged than the reproduction layer
- the repository is still in active architectural grooming
- the remaining work is convergence work, not invention of discipline from
  scratch

That is the correct current compliance picture.
