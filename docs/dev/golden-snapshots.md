# Golden Snapshots

Golden snapshots are deterministic reference artifacts used to detect unintentional behavior drift.

## When to regenerate

Regenerate snapshots only when the change is intentional, for example:

- parser output shape changed by design
- default prompt text changed by design
- text pattern behavior changed by design

Do **not** regenerate snapshots just to make failing tests pass.

## Regeneration command

```bash
python tests/regenerate_goldens.py
```

## Validation after regeneration

Always run:

```bash
./scripts/check
```

before committing regenerated snapshot files.

## What the snapshot tests protect

- [`tests/test_rules_ast_golden_snapshot.py`](../../tests/test_rules_ast_golden_snapshot.py) — parser output drift detection
- [`tests/test_prompt_golden.py`](../../tests/test_prompt_golden.py) — accidental edits to the default system prompt
- [`tests/test_text_patterns_golden.py`](../../tests/test_text_patterns_golden.py) — text pattern behavior regression checks
