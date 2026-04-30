# Extension Points

This page is for developers integrating Answer Engineering into custom stacks.

The canonical extension imports are:

```python
from answer_engineering.extensions import (
    CandidateProvider,
    FullPlanCompiler,
    MarkdownRulesParser,
    RuntimeEventSink,
    Scorer,
)
```

These names are re-exported from `src/answer_engineering/extensions.py`.

## What each extension seam does

- **`MarkdownRulesParser`**
  - Parses markdown domain-specific language text into `RulesetAST`.
  - Use this seam if you need custom parsing behavior or alternate rule ingestion.

- **`FullPlanCompiler`**
  - Compiles parsed abstract-syntax-tree rules into executable `PlanIR`.
  - Use this seam if you need different compilation policy while keeping runtime execution.

- **`CandidateProvider`**
  - Generates candidate patch proposals during proposal generation.
  - Use this seam to inject domain-specific candidate strategies.

- **`Scorer`**
  - Scores normalized proposals before deterministic selection/conflict resolution.
  - Use this seam for custom ranking logic.

- **`RuntimeEventSink`**
  - Receives runtime events emitted by the engine execution flow.
  - Use this seam to plug in custom telemetry/logging/testing observers.

## Practical integration notes

- These seams are the supported parse → compile → runtime collaboration points.
- Prefer extending these contracts instead of reaching into orchestration internals.
- If you only need standard behavior, use the default `CompiledRules` + `GenerationRuntime` path and do not replace components.
