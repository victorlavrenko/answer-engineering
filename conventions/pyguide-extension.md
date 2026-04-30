# Greenfield Extensions to Google Python Style Guide

Purpose: this document extends Google’s Python Style Guide for a greenfield, architecture-first, strongly typed codebase.

This is a **human-readable diff**, not a replacement. It only states:

* what is added beyond Google’s guide
* what is tightened
* what is reinterpreted for greenfield work
* what is ignored because it exists for compatibility or does not fit this project’s goals

If a Google section is not mentioned here, follow Google’s guide as written.

---

## 1. Scope and interpretation

### 1.1 Greenfield assumption

This project assumes:

* no backward compatibility obligations unless explicitly documented
* freedom to redesign APIs and boundaries when architecture is wrong
* preference for deletion over compatibility shims
* preference for converging on one correct path over preserving multiple paths

### 1.2 Architectural priority

When Google’s guide optimizes for broad applicability but this project needs stronger guarantees, prefer:

* invariant safety
* deterministic lifecycle
* explicit ownership
* explicit collaboration boundaries
* strong typing
* locality of behavior
* architectural legibility through naming

### 1.3 Tool specificity

Google’s guide may name particular tools. This project does **not** treat tool specificity as a problem.

Where useful, this project may require **more** tools, not fewer, provided each tool enforces a distinct class of guarantees.

---

## 2.1 Lint

### Extension

Linting is a **multi-tool gate**, not a single-tool concern.

Required categories of enforcement:

* formatting
* linting
* type checking
* import and dependency boundary enforcement
* architectural pattern checks
* dead code and unused symbol checks
* tests

### Project rule

A greenfield codebase must use enough tools to enforce the required guarantees. Tool overlap is acceptable when it materially improves confidence.

### Recommended gate

At minimum, each completed migration step or feature branch must pass:

* formatter
* linter
* type checker
* test suite

When architectural rules are encoded in additional tools, those tools are part of the required gate as well.

---

## 2.2 Imports

### 2.2 General import policy — modified

Keep Google’s emphasis on explicit imports and clear provenance.

### Extension

Imports must also preserve **architectural boundaries**.

Imports are not only a readability concern. They are a dependency graph and must reflect subsystem ownership.

### Project rule

Use imports to expose and preserve architectural layering.

Prefer:

* imports from stable public boundary modules
* imports that make the dependency direction obvious
* imports that avoid reaching into implementation internals of another subsystem

Avoid:

* importing from deep internal modules of another subsystem when a boundary object already exists
* import patterns that couple callers to incidental implementation layout
* import shortcuts that erase architectural ownership

### Greenfield strengthening

If a module repeatedly needs deep imports into another subsystem, treat that as a design signal:

* either the boundary is wrong
* or the public API of that subsystem is incomplete

### 2.2.4.1 Exemptions — ignored

Ignore this Google subsection entirely for this project.

The compatibility-specific exemption discussion is not part of this guide extension.

---

## 2.3 Packages

### Extension

Package structure must reflect **stable subsystem boundaries**, not convenience buckets.

### Project rule

A package name must answer: what architectural boundary or capability area lives here?

Allowed examples:

* `engine`
* `inference`
* `reproduction`
* `telemetry`
* `selection`
* `config`

Forbidden examples unless literally correct:

* `utils`
* `helpers`
* `common`
* `misc`
* `impl`
* `new`
* `old`
* `v2`
* `tmp`

### Greenfield strengthening

Creating a vague package to avoid deciding ownership is forbidden.

---

## 2.6 Nested / local / inner definitions

### Extension

Nested functions and local helper definitions are allowed only when their locality is semantically real.

### Project rule

Use nested definitions only when at least one of the following is true:

* closure semantics are essential
* the behavior is truly local to one operation
* exposing the definition at module or class scope would make the API less truthful

Do not use nested definitions to hide reusable logic, suppress naming decisions, or avoid giving a helper its proper architectural home.

---

## 2.13 Properties

### Extension

Keep Google’s cheap-and-unsurprising property rule.

### Greenfield strengthening

Properties must not hide:

* lifecycle transitions
* remote I/O
* cross-boundary synchronization
* cache rebuilds with important semantics
* policy selection
* domain state transitions

If the operation is semantically significant, expose it as a method with a verb.

### Rationale

A property should behave like stable state exposure, not a disguised operation boundary.

---

## 2.19 Power features

### Extension

Do not treat all advanced language features as one category.

### Project rule

Classify advanced features by boundary and purpose:

#### Forbidden in domain and application logic unless explicitly justified

* magic that obscures control flow
* hidden mutation via metaprogramming
* runtime code generation for ordinary business logic
* reflection-heavy branching used instead of explicit interfaces

#### Allowed in infrastructure or framework layers when isolated and documented

* disciplined metaprogramming
* code generation at representation or tooling boundaries
* framework integration patterns that remain explicit at the public boundary

### Greenfield strengthening

The question is not “is this feature powerful?” but “does it preserve explicit behavior at this boundary?”

---

## 2.20 Modern Python / compatibility-era guidance

### Extension

Ignore compatibility guidance that exists primarily for old Python baselines or transitional migration states.

### Project rule

Target one modern Python baseline.

Use current syntax directly unless a specific retained construct has a clear operational benefit in the current toolchain.

### Greenfield strengthening

Do not carry compatibility-era rules forward into a clean-slate codebase without a present-day justification.

---

## 2.21 Type annotated code

### Extension

This project adopts a stricter typed-code standard than Google’s general baseline.

### Project rule

Require annotations for:

* all public functions and methods
* all nontrivial internal functions and methods
* constructors
* members of abstract base classes, protocols, and other explicit interface types
* class attributes when inference is ambiguous
* module-level state when inference is ambiguous or architectural meaning benefits from explicit type

### `Any`

`Any` is an escape hatch, not a normal design tool.

Allowed only when one of the following is true:

* boundary to an inherently dynamic third-party API
* unavoidable typing hole in the ecosystem
* temporary migration block explicitly marked for removal

When `Any` is used, prefer to contain it at the smallest possible boundary.

### Interfaces and abstraction boundaries

Prefer explicit collaboration contracts where multiple components interact across a stable boundary.

Choose the narrowest truthful form that matches the design:

* a concrete type when no abstraction boundary is needed
* an abstract base class when the project benefits from a nominal shared interface, shared semantics, controlled inheritance, or reusable default behavior
* a protocol only when structural substitution is genuinely the correct model and nominal inheritance would add unnecessary ceremony

Do not introduce abstraction layers mechanically.

Do not introduce protocols merely because a caller currently uses a subset of methods.

### Greenfield strengthening

Typing is not only for public API polish. It is part of architectural control.

---

## 3.8 Comments and docstrings

### Extension

Documentation must distinguish between three different concerns:

* public API contract
* architectural rationale
* local implementation commentary

### Project rule

Docstrings should primarily document:

* what the boundary does
* required invariants
* preconditions and postconditions when not obvious
* ownership or lifecycle expectations when relevant
* the component's architectural role in this codebase
* where each meaningful input originates architecturally (caller, subsystem, or lifecycle stage)
* how outputs are consumed by downstream components in this codebase

Do not use comments as a substitute for bad naming or bad structure.

### Step-2 template (required for new/updated production docstrings)

When writing or updating docstrings in this codebase, use this structure:

* `Purpose:` concise statement of what the boundary does
* `Architectural role:` where this component sits in the system design
* `Inputs (architectural provenance):` what inputs are accepted and which
  caller/subsystem/lifecycle stage supplies them
* `Outputs (downstream usage):` what is returned/emitted and which
  caller/subsystem consumes it
* `Invariants/constraints:` required safety or lifecycle rules when relevant

### Greenfield strengthening

Architecture-heavy code benefits from explicit rationale for boundaries, not from large volumes of descriptive prose about obvious mechanics.

---

## 3.15 Accessors

### Extension

Keep Google’s rejection of trivial getters and setters.

### Greenfield strengthening

Explicit methods are still preferred when the operation represents:

* a state transition
* invariant validation
* synchronization
* policy application
* event emission
* materialization of external resources

### Project rule

Use plain attribute exposure or cheap properties for stable state.
Use methods for meaningful operations.

---

## 3.16 Naming

### Extension

Google’s naming rules are necessary but not sufficient. This project adds **architectural naming**.

### Project rule

Names must reveal at least one of the following:

* the object’s semantic role
* the collaboration boundary it belongs to
* whether it owns behavior, lifecycle, or data
* whether it changes over time

### Required preference order

When naming, prefer this order:

1. domain term
2. architectural role
3. lifecycle/state term
4. representation term
5. implementation term

### Forbidden naming patterns unless literally correct

* `utils`
* `helper`
* `common`
* `misc`
* `manager`
* `processor`
* `handler`
* `controller`
* `wrapper`
* `adapter`
* `thing`
* `data`
* `info`
* migration markers such as `legacy`, `new`, `old`, `v2`, `final`, `compat`

### Suffix discipline

Use suffixes only when they materially narrow meaning.

Approved semantic suffix families include:

* `Runtime`
* `Session`
* `State`
* `Context`
* `Request`
* `Result`
* `Report`
* `Snapshot`
* `Record`
* `Row`
* `Spec`
* `Policy`
* `Defaults`
* `Sink`
* `Provider`
* `Loader`
* `Publisher`
* `Compiler`
* `Parser`
* `Resolver`
* `Emitter`
* `Formatter`

A suffix is not decoration. It must state object kind or responsibility.

---

## 3.16.3 File naming

### Extension

File names must follow the **primary concept or primary boundary object** defined inside.

### Project rule

Allowed patterns:

* one main concept per file
* family files for one closely related architectural family

Forbidden patterns unless literally correct:

* historical status in file names
* vague catch-all files
* file names that exist only to hold leftovers

### Greenfield strengthening

Delete obsolete code; do not immortalize migration state in names.

---

## 3.18 Function length

### Extension

Function size must be judged primarily by **coherence of responsibility**, not by line count alone.

### Project rule

A function is too large when it mixes:

* multiple architectural phases
* multiple ownership domains
* representation-boundary logic with core domain logic
* orchestration and low-level detail that should live elsewhere

### Greenfield strengthening

A short function can still be architecturally wrong. A longer function may be acceptable if it represents one coherent boundary operation.

---

## 3.19 Typing rules — strengthened

### 3.19.1 General typed-code rule

Require full typing discipline for production code except at narrowly contained dynamic boundaries.

### 3.19.7 Ignoring types

Type-ignore usage must be:

* narrow in scope
* justified with a reason
* treated as technical debt unless the boundary is inherently dynamic

### 3.19.9 Containers and immutability

Google’s tuple/list distinction is extended as follows:

#### Canonical rule

* accept flexible read-only input as `Sequence[T]`
* store stable immutable owned state as `tuple[T, ...]`
* use `Iterable[T]` when only iteration is required
* require `list[T]` only when mutation is part of the contract
* when constructing a sequence that will not be mutated, create it as a
  `tuple` directly instead of building a temporary `list` first

#### Rationale

The important design rule is not only annotation correctness but ownership and mutability semantics.

### 3.19.11 Compatibility-era string typing discussion — simplified

For this project, the effective rule is simply:

* use `str` for text
* use `bytes` for binary data

Ignore older compatibility discussion.

### 3.19.12 Typing imports

Keep this Google subsection as-is.

### 3.19.13 Conditional imports

Treat repeated `TYPE_CHECKING` import workarounds as a design smell.

If they appear frequently, revisit package boundaries, dependency direction, or module decomposition.

---

## 4. Object construction and lifecycle

### Extension

This section is added beyond Google’s guide.

### 4.1 Construction ownership priority

Object creation must follow this order:

1. prefer a rich unnamed constructor (`__init__`)
2. if one constructor would be semantically unclear, use a named constructor on the same class
3. construction by another class is a last resort and should require explicit justification

### 4.2 Constructor-first validity

Constructors must establish:

* object identity
* required domain state
* invariant validity

Objects must be usable immediately after construction.

Do not allow partially initialized domain objects.

### 4.3 Named constructors

Named constructors are allowed only when they express a distinct semantic construction mode, such as:

* loading from external state
* constructing from serialized form
* resolving runtime resources

Do not create named constructors that merely forward to the primary constructor without adding semantic meaning.

### 4.4 Materialization vs construction

Construction defines what the object is.
Materialization loads external state or heavy resources.

Materialization may hydrate, load, or resolve resources.
It must not define identity or complete missing invariants.

---

## 5. Flattening and data ownership

### Extension

This section is added beyond Google’s guide.

### 5.1 Canonical container rule

When a concept has multiple related arguments, define one canonical container object and pass that object through the system unchanged.

### 5.2 No flattening of owned data

Within domain and application logic, do not unpack one object field-by-field merely to satisfy another object or function.

Repeated field transfer is a design smell and must be treated as suspect until proven otherwise.

### 5.3 Representation boundary exception

Flattening is allowed only at representation boundaries, including:

* serialization
* persistence
* telemetry
* logging
* network transport
* external API payloads
* tabular reporting

### 5.4 Prefer storing passed objects

If lifecycle semantics permit, the receiving object should store the canonical passed object directly rather than copying its fields.

---

## 6. Runtime and infrastructure boundaries

### Extension

This section is added beyond Google’s guide.

### 6.1 Runtime objects

Runtime objects represent behavior-owning execution environments.

They may coordinate:

* models
* tokenizers or codecs
* devices
* execution policies
* runtime configuration
* execution lifecycle

### 6.2 Runtime behavior over raw dependency exposure

Prefer behavior-first exposure:

* `runtime.generate(request)`
* `runtime.text_codec()`

Avoid exposing raw implementation details solely for callers to orchestrate the runtime externally.

### 6.3 Subobjects

Subobjects may be exposed only when they represent a stable architectural role and their lifecycle remains owned by the parent object.

---

## 7. Collaboration boundaries and interface types

### Extension

This section is added beyond Google’s guide.

### 7.1 Choose the smallest architecturally meaningful boundary

Do not default to passing the largest available coordinating object.
Do not default to passing raw dependencies.

Pass the smallest boundary that represents a real stable role.

### 7.2 Avoid mechanical capability slicing

Do not create abstractions that exist only because one caller currently needs one method.

A valid abstraction boundary must represent:

* a stable concept
* a named responsibility
* a reusable collaboration role

This rule applies equally to concrete boundary objects, abstract base classes, and protocols.

### 7.3 Interface naming

Names for abstraction boundaries must describe collaboration roles, not uncertainty or accidental capability slices.

Avoid names such as:

* `RuntimeLike`
* `TokenizerLike`
* `PartialRuntime`
* `HasEncode`

Avoid `Supports*` names unless the interface truly represents a small, standard capability-style contract rather than an architectural role.

Prefer names such as:

* `TextCodec`
* `RulesParser`
* `RuntimeEventSink`
* `CandidateProvider`

When an abstract base class is the primary public abstraction, it should normally own the shortest stable role name.

When a protocol is used, it should also be named by role, not by uncertainty, method presence, or scaffold terminology.

### 7.4 Interface type selection

Choose abstraction mechanisms in this order:

1. use a concrete type when no abstraction boundary is needed
2. use an abstract base class when the boundary is nominal, stable, shared across implementations, or benefits from inherited semantics or default behavior
3. use a protocol only when structural substitution is genuinely the correct model and nominal inheritance would make the design less truthful

Do not introduce a protocol mechanically when an abstract base class or concrete type is the clearer dependency.

Do not use abstract base classes or protocols merely to mirror one implementation one method at a time.

### 7.5 Abstract base classes

Prefer abstract base classes when the project needs:

* a nominal shared interface
* inherited default behavior
* shared validation or helper logic
* controlled extension through subclassing
* a stable public abstraction that should be explicit in the type hierarchy

Abstract base classes should not exist merely as empty scaffolding.

If an abstract base class is only a nominal shell with no semantic or behavioral value, prefer either a concrete type or, when structurally appropriate, a protocol.

### 7.6 Protocol member implementation workaround

Protocol boundaries are structural contracts and must not use nominal abstraction markers.

Required rules:

* never decorate protocol members with `@abstractmethod`
* never mix `Protocol` and `ABC` in the same class base list
* keep protocol member docstrings, and when a concrete method body is required for tooling, use `raise NotImplementedError`

Canonical pattern:

```python
from typing import Protocol


class TextCodec(Protocol):

    def encode(self, text: str) -> bytes:
        """Encode text into tokens."""
        raise NotImplementedError
```

Rationale:

* preserves structural typing semantics
* avoids architectural ambiguity between protocol and abstract-base contracts
* keeps lint/type tooling compliant without introducing nominal inheritance

---

## 8. Encapsulation and state ownership

### Extension

This section is added beyond Google’s guide.

### 8.1 Private state by default for invariant-bearing objects

For domain objects, runtime objects, and lifecycle owners with invariants, internal state is private by default.

### 8.2 Public state is acceptable for simple value carriers

For simple immutable value objects or transparent data carriers, public attributes are acceptable when they do not bypass invariants or hide important behavior.

### 8.3 Intent-based mutation

If an object owns lifecycle state or invariants, state transitions should be expressed as explicit methods rather than arbitrary public mutation.

This extends Google’s accessor guidance by distinguishing simple data carriers from invariant-bearing objects.

---

## 9. Immutability

### Extension

This section is added beyond Google’s guide.

### 9.1 Immutability by default

Immutability is strongly preferred for value objects and domain entities whose identity does not evolve.
Lifecycle controllers and runtime state objects are expected to be mutable.

### 9.2 Mutable objects must justify mutation

Mutation is allowed only when the object truly represents evolving lifecycle state, such as a run, session, workflow, or job.

### 9.3 Value objects

Prefer immutable value objects that validate themselves on construction and compare by value.

---

## 10. Configuration ownership

### Extension

This section is added beyond Google’s guide.

### 10.1 Centralize environment and provider keys

Provider or environment integration keys must not be embedded as scattered string literals throughout domain or application logic.

Define them once in typed defaults/configuration carriers and reference those carriers.

### 10.2 Defaults carriers

Use `*Defaults` objects for subsystem default values when those defaults have architectural meaning or are shared across a boundary.

---

## 11. Migration and convergence policy

### Extension

This section is added beyond Google’s guide.

### 11.1 No parallel architecture paths without explicit justification

Once a replacement boundary exists, remove pre-migration compatibility shims and obsolete paths unless an external compatibility consumer is explicitly documented.

### 11.2 Prefer convergence over adapters

When two internal parts of the system are incompatible, 
prefer redesign and convergence over adding long-lived adapters inside the same subsystem.
Adapters are appropriate at subsystem or external integration boundaries.

### 11.3 Validation cadence

After each migration phase or major architectural step, run the required quality gate and record outcomes.

---

## Appendix A — Naming summary additions

This appendix supplements Google naming with architecture-specific naming.

### A.1 Object kind signaling

A class name should make it obvious whether the object is primarily a:

* domain entity
* boundary object
* lifecycle owner
* orchestrator
* data carrier
* value object
* rule or decision object

### A.2 Public API naming

Public API names must be semantically complete, stable, and independent of internal file layout.

### A.3 Concrete implementation naming

If multiple implementations exist for one role, differentiate them by:

* mechanism
* backend
* storage medium
* strategy
* scope

Do not use `Impl` or similar placeholder names when a real qualifier exists.

When an abstract base class is the primary public abstraction, the abstract base class normally owns the shortest stable role name, and concrete subclasses should add qualifying specificity.

---

## Appendix B — Summary of ignored Google material

For this project, ignore Google subsections whose primary purpose is compatibility with older runtimes, old internal code patterns, or transitional import/typing history.

Specifically:

* ignore subsection 2.2.4.1 exemptions
* ignore compatibility-era Python-baseline discussion where it does not affect the chosen modern baseline
* ignore deprecated grouping or historical import pattern discussion
* ignore old string-typing compatibility discussion beyond the modern `str` / `bytes` rule

---

## Final rule of interpretation

When Google’s guide and this extension both apply, interpret them in this order:

1. explicit project architectural rule in this extension
2. Google’s rule as the default baseline
3. tool-specific implementation chosen by the project

The intent is not to reject Google’s guide. The intent is to keep its strong Pythonic baseline while adding the architectural strictness expected from a greenfield, strongly typed, boundary-oriented codebase.
