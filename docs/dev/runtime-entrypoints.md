# Runtime Entry Points

This page describes the main code-facing entry points for generation.

## Primary public generation flow

The canonical generation call is:

- [`GenerationRuntime.generate(...)`](../../src/answer_engineering/inference/answering.py)

The main public contracts are:

- [`GenerationRuntime`](../../src/answer_engineering/inference/answering.py)
- [`GenerationRequest`](../../src/answer_engineering/inference/contracts.py)
- [`GenerationPolicy`](../../src/answer_engineering/inference/contracts.py)
- [`GenerationResult`](../../src/answer_engineering/inference/contracts.py)

## Typical execution shape

1. Construct a `GenerationRuntime`
2. Materialize the runtime if needed
3. Build a `GenerationRequest`
4. Build a `GenerationPolicy`
5. Call `runtime.generate(request, policy)`
6. Inspect the returned `GenerationResult`

A concrete example is in the [quickstart notebook](../../notebooks/quickstart.ipynb).

## Rules in the public flow

In the quickstart, rules are supplied through `GenerationPolicy.rules` as markdown text.

For syntax and rule authoring details, see:

- [Writing rules](../users/writing-rules.md)
- [Rule language reference](../rules/language-reference.md)

## Related documentation

- [Architecture](../current/architecture.md)
- [Runtime model](../current/runtime-model.md)
- [Extension points](extension-points.md)
