# Answer Engineering

Answer Engineering is a Python library for steering language model generation with explicit, local rules.

It is designed for situations where the *path* to an answer matters, not just the final output — for example when models must follow engineering practices, clinical protocols, safety procedures, or organizational standards.

Instead of retraining the model or post-processing the output, Answer Engineering intervenes during generation and redirects specific steps in real time. The result is behavior that is inspectable, reproducible, and policy-constrained.

---

## Quickstart

The fastest way to understand the value of Answer Engineering is to run the notebook:

- [Quickstart notebook](notebooks/quickstart.ipynb)

### The problem

Language models are very good at producing output — sometimes *too* good.

For example, in vibe-coding tasks, models often start implementing new code immediately because they are trained to generate solutions. Human engineers, however, frequently pause to check whether an existing library or component can be reused instead.

This mismatch leads to unnecessary complexity, duplicated logic, and code that violates team conventions.

### What the notebook does

The notebook demonstrates how a single rule can redirect the generation path.

The rule says:

> Replace **"Implement"** with **"Consider reusing an existing"** and continue from there

This intervention happens locally during generation — at the moment the target phrase appears — not after the answer is finished.

### What you will see

The notebook runs the same prompt twice:

1. Baseline generation
   - The model immediately writes new code from scratch.

2. Generation with Answer Engineering
   - The rule intercepts the reasoning step.
   - The model pauses to evaluate reuse options.
   - The final answer uses an existing component instead of reimplementing one.

The saved output cell shows the divergence clearly: the baseline implements, while the guided run reuses.

This is the core idea of Answer Engineering:

*Change the trajectory, and the outcome changes naturally.*

---

## Why this project exists

After running the Quickstart, the key observation becomes hard to ignore: generation can be corrected locally, and the model will continue naturally from the corrected state.

Large language models do not maintain hidden commitments to earlier tokens. They simply continue from the visible text prefix. This means that when a trajectory step is edited — for example, redirecting "Implement" toward "Consider reusing an existing component" — the model proceeds as if that step had always been written that way.

In other words, trajectory correction is not a hack. It is a property of how autoregressive generation works.

Once this is understood, allowing generation to proceed without correction in protocol-sensitive settings starts to look like an unnecessary risk. If a step can be repaired immediately, the downstream reasoning — and the final outcome — can change in a predictable way.

This matters beyond individual steps.

Small trajectory corrections accumulate. A corrected assumption leads to a different branch of reasoning. A different branch of reasoning leads to a different decision. And a different decision often determines whether the system behaves safely, efficiently, or correctly.

Answer Engineering exists to make this capability explicit and reliable.

It provides a runtime layer that can:

- detect when a trajectory enters a risky or non-compliant path
- apply deterministic local edits at that moment
- continue generation from the corrected state
- record what changed and why

The result is not just cleaner intermediate steps, but more dependable final answers — because the reasoning path itself stayed within the required boundaries.

This repository includes both the runtime implementation and a reproducible evaluation pipeline that demonstrates this effect in a controlled benchmark.

For the full research description of the system, see:

- [`docs/paper/lavrenko2026_answer_engineering.pdf`](docs/paper/lavrenko2026_answer_engineering.pdf)
- [`docs/paper/main.tex`](docs/paper/main.tex)
- [`docs/paper/generated/paper-metrics.tex`](docs/paper/generated/paper-metrics.tex)

---

## Repository structure

This repository contains two related layers.

- [`answer_engineering`](src/answer_engineering/) — the runtime library and rule system.
- [`ae_paper_reproduction`](src/ae_paper_reproduction/) — the notebook, telemetry, reporting, and paper-reproduction layer.

For the full documentation index, start with [`docs/README.md`](docs/README.md).

### `answer_engineering`

The runtime library.

It provides:

- rule parsing and compilation
- deterministic trajectory intervention
- observable runtime behavior
- telemetry and inspection tools

This is the component you use to integrate Answer Engineering into applications.

### `ae_paper_reproduction`

The research and evaluation layer.

It provides:

- notebooks used in the paper
- telemetry aggregation and reporting
- reproducibility workflows
- metric generation for the manuscript

You typically do not need this layer to use the runtime, but it is included so that all reported results can be reproduced exactly.

---

## What works today

The current implementation supports rule-guided generation through a narrow public runtime API:

- [`GenerationRuntime`](src/answer_engineering/inference/answering.py)
- [`GenerationRequest`](src/answer_engineering/inference/contracts.py)
- [`GenerationPolicy`](src/answer_engineering/inference/contracts.py)
- [`GenerationResult`](src/answer_engineering/inference/contracts.py)
- [`CompiledRules`](src/answer_engineering/rules/__init__.py)

Current code-faithful documentation:

- [Current capabilities](docs/current/capabilities.md)
- [Current architecture](docs/current/architecture.md)
- [Current runtime model](docs/current/runtime-model.md)
- [Current codebase reality](docs/current/codebase-reality.md)
- [Telemetry schema](docs/current/telemetry-schema.md)

## What this project is not

This repository is not currently a general-purpose agent framework, a production LLM serving platform, a broad prompt-engineering toolkit, or a generic safety moderation library.

Its core concern is controlled generation under explicit local rules.

## Core runtime model

The canonical public call is [`GenerationRuntime.generate(request, policy)`](src/answer_engineering/inference/answering.py).

The current execution path is described in [Runtime model](docs/current/runtime-model.md) and [Runtime entry points](docs/dev/runtime-entrypoints.md). At a high level:

```text
GenerationRuntime.generate(...)
StreamSession.run()
GreedyDecoder.decode()
ExecutionSession.apply_step(...)
PlanRunner
GenerationResult
```

When rules are present, the runtime monitors the evolving answer, evaluates compiled rule plans, applies deterministic text edits, records telemetry, and continues generation from the edited state. This is not just post-processing: intervention happens during generation.

## Rule system

Rules are authored in a compact Markdown-based domain-specific language and compiled into executable plans. The exact syntax is documented in [Rule language reference](docs/rules/language-reference.md), and practical authoring guidance is in [Writing rules](docs/users/writing-rules.md).

Rule families:

- **`Replace`** — normalize protocol-critical terminology by replacing matched text with approved alternatives.
- **`After`** — insert approved text after an anchor once the relevant concept has appeared.
- **`Avoid`** — detect risky trajectories using prefix/postfix guards and redirect generation through fallback or probed continuations.
- **`Force`** — enforce a required statement within a scope.

A minimal rule looks like this:

```ae-rules
## Replace (once): sensorineural hearing loss
With:
- sudden sensorineural hearing loss
```

For the full grammar, modifiers, guard operators, scope syntax, options, and template expansion rules, see [Rule language reference](docs/rules/language-reference.md).

## Minimal shape

```python
from answer_engineering import GenerationRuntime, GenerationRequest, GenerationPolicy

runtime = GenerationRuntime(MODEL_ID)
answer = runtime.generate(
    GenerationRequest(question=QUESTION),
    GenerationPolicy(
        rules=RULES,
        system_prompt=SYSTEM_PROMPT,
    ),
)
```

## Minimal story

Load a model, ask a question, and apply a ruleset during generation.

The ruleset defines local trajectory edits that are enforced while the answer is being produced. The resulting output reflects those enforced constraints and can be inspected together with the associated runtime telemetry.

## Installation

For local development:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev,hf]"
```

Then validate the repository:

```bash
./scripts/check
```

Contribution and validation details are in [CONTRIBUTING.md](CONTRIBUTING.md). CI is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Reproducing the paper

The main reproduction entry point is:

- [notebooks/reproduce.ipynb](notebooks/reproduce.ipynb)

Reproducibility documentation:

- [Reproducibility guide](docs/current/reproducibility.md)
- [Paper artifacts](docs/current/paper-artifacts.md)
- [Generated paper metrics](docs/paper/generated/paper-metrics.tex)
- [Paper source](docs/paper/main.tex)
- [Rendered paper PDF](docs/paper/lavrenko2026_answer_engineering.pdf)

The reproduction layer emits structured artifacts such as evaluation summaries, telemetry summaries, paper tables, manifests, and generated TeX metrics. The current artifact flow is described in [Paper artifacts](docs/current/paper-artifacts.md).

## Repository layout

- [`src/answer_engineering/`](src/answer_engineering/) — runtime library, rules, inference, engine, config, telemetry, and infrastructure.
- [`src/ae_paper_reproduction/`](src/ae_paper_reproduction/) — evaluation, reporting, telemetry export, and paper-reproduction workflows.
- [`notebooks/`](notebooks/) — notebook entry points, including [quickstart](notebooks/quickstart.ipynb) and [reproduction](notebooks/reproduce.ipynb).
- [`docs/`](docs/README.md) — reader-facing documentation.
- [`docs/current/`](docs/current/) — code-faithful current architecture and behavior.
- [`docs/dev/`](docs/dev/) — developer entry points, extension seams, and golden snapshots.
- [`docs/rules/`](docs/rules/) — rule language reference.
- [`docs/users/`](docs/users/) — practical rule-authoring guidance.
- [`docs/vision/`](docs/vision/) — long-term system and trajectory-control vision.
- [`docs/gaps/`](docs/gaps/) — known gaps, roadmap, and convention-compliance work.
- [`conventions/`](conventions/) — coding and architectural conventions.
- [`tests/`](tests/) — tests, architecture checks, regression tests, and golden snapshots.

## Documentation map

Start here based on what you need:

- New user: [Writing rules](docs/users/writing-rules.md)
- Rule author: [Rule language reference](docs/rules/language-reference.md)
- Runtime integrator: [Runtime entry points](docs/dev/runtime-entrypoints.md)
- Extension author: [Extension points](docs/dev/extension-points.md)
- Reproduction user: [Reproducibility guide](docs/current/reproducibility.md)
- Paper reader: [Rendered paper PDF](docs/paper/lavrenko2026_answer_engineering.pdf) and [paper source](docs/paper/main.tex)
- Architecture reader: [Current architecture](docs/current/architecture.md), [Runtime model](docs/current/runtime-model.md), and [Codebase reality](docs/current/codebase-reality.md)
- Telemetry consumer: [Telemetry schema](docs/current/telemetry-schema.md)
- Maintainer: [Golden snapshots](docs/dev/golden-snapshots.md), [CONTRIBUTING.md](CONTRIBUTING.md), and [conventions](conventions/)
- Roadmap reader: [System vision](docs/vision/system-vision.md), [Trajectory control vision](docs/vision/trajectory-control.md), and [Functionality roadmap](docs/gaps/functionality-roadmap.md)

## Current status

This is an initial public implementation and research artifact. The core runtime is already meaningful, tested, and documented, but the repository is not architecturally finished.

The most accurate current-state summary is in [Current codebase reality](docs/current/codebase-reality.md). In short: the runtime package has a relatively narrow public API and stronger subsystem boundaries, while the reproduction and paper layer remains more shaped by active research workflows.

Backward compatibility is not guaranteed.

## Expected future development

Future work is expected in both architecture and capabilities.

Planned architectural directions include clearer runtime/reproduction boundaries, stronger extension seams, improved ownership of scoring and candidate-selection components, and continued convergence between documentation, tests, and implementation. See [Current architecture](docs/current/architecture.md), [Codebase reality](docs/current/codebase-reality.md), and [Extension points](docs/dev/extension-points.md).

Planned capability directions include causal trajectory repair, alternative trajectory tracking, branch-aware scoring, uncertainty signaling, partial-history editing, and richer multi-rule protocol control. See [Trajectory control vision](docs/vision/trajectory-control.md) and [Functionality roadmap](docs/gaps/functionality-roadmap.md).

The long-term goal is not merely “more rules”. The target is a runtime layer that can identify where a protocol violation appeared, what earlier commitment caused it, which repairs are valid, whether multiple trajectories remain plausible, and how uncertainty should be surfaced.

## Development validation

Before committing changes, run:

```bash
./scripts/check
```

This repository uses formatting, linting, type checking, convention checks, tests, and package-build validation. Details are in [CONTRIBUTING.md](CONTRIBUTING.md), [Golden snapshots](docs/dev/golden-snapshots.md), and the convention documents under [`conventions/`](conventions/).

## Citation

If you use this repository as a research artifact, cite the paper:

```text
Victor Lavrenko and Anastasiia Molodnitskaia.
Answer Engineering: Local Trajectory Editing for Protocol-Constrained Decision Making in Large Language Models.
2026.
```

Paper files:

- [Rendered PDF](docs/paper/lavrenko2026_answer_engineering.pdf)
- [TeX source](docs/paper/main.tex)
- [Bibliography](docs/paper/references.bib)
- [Generated metrics](docs/paper/generated/paper-metrics.tex)

## License

MIT. See [LICENSE](LICENSE).
