from __future__ import annotations

from answer_engineering import CompiledRules, GenerationPolicy


def test_generation_policy_compiles_string_rules_at_boundary() -> None:
    policy = GenerationPolicy(rules="## Replace (once): x\n\nWith:\n\n- y\n")

    assert policy.compiled_rules is not None
    assert len(policy.compiled_rules.plan.rules) == 1


def test_generation_policy_preserves_compiled_rules_instance() -> None:
    compiled = CompiledRules("## Replace (once): x\n\nWith:\n\n- y\n")
    policy = GenerationPolicy(rules=compiled)

    assert policy.compiled_rules is compiled
