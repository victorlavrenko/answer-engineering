"""Public runtime API for Answer Engineering generation.

Import from this package when you want to run model generation with optional
Answer Engineering rules. The public surface intentionally stays small: a
runtime loads the model, a request describes the question, a policy describes
how generation should proceed, compiled rules optionally intervene, and the
result returns the final answer with runtime metadata.

Typical use:
    ```python
    from answer_engineering import (
        CompiledRules,
        GenerationPolicy,
        GenerationRequest,
        GenerationRuntime,
    )
    compiled_rules = CompiledRules.from_text(rules_text)
    runtime = GenerationRuntime("Qwen/Qwen2.5-7B-Instruct")
    runtime.materialize()

    request = GenerationRequest(question="What is the likely diagnosis?")
    policy = GenerationPolicy(
        max_new_tokens=256,
        rules=compiled_rules,
    )
    result = runtime.generate(request, policy)
    print(result.text)
    ```

What Answer Engineering adds:
    The runtime still performs model-backed generation, but it can consult a
    compiled ruleset while the answer is being produced. Rules can redirect,
    replace, or avoid local continuations. This makes the system useful for
    protocol adherence experiments, safety-oriented wording constraints,
    domain-specific answer formats, and paper reproduction tasks where a
    baseline run is compared with rule-enabled generation.

What users should expect:
    Generation is deterministic only to the extent configured by the runtime,
    policy, model, tokenizer, and hardware. Rules are not a second model; they
    are explicit interventions applied around the model's continuation. The
    result object contains the answer text and may also expose telemetry useful
    for debugging or reporting.

Rule-language context:
    Rules are compiled before generation. In reproduction notebooks the rules
    are usually extracted from authored markdown; in standalone experiments they
    may be built through the rule compiler. Common rules describe text to match,
    replacement text, and matching options such as case behavior or scoped
    blocks. A failed compile should be treated as an invalid experiment setup,
    not as a runtime generation event.

Developer notes:
    This package root is the supported boundary for examples and notebooks. Keep
    it smaller than the internal architecture. Runtime orchestration, probing,
    scoring, selection, patching, and telemetry internals should remain behind
    the public generation objects unless a new user-facing capability is
    intentionally introduced.

Todo:
    As the runtime evolves toward richer branch-aware repair and reporting,
    preserve the simple request-policy-runtime-result shape for users who only
    need to run a controlled generation experiment.

"""

from __future__ import annotations

from answer_engineering.inference.answering import GenerationRuntime
from answer_engineering.inference.contracts import (
    GenerationPolicy,
    GenerationRequest,
    GenerationResult,
)
from answer_engineering.rules.compile.compiled_rules import CompiledRules

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CompiledRules",
    "GenerationPolicy",
    "GenerationRequest",
    "GenerationResult",
    "GenerationRuntime",
]
