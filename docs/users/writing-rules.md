# Writing Rules

This guide is for ruleset authors using Answer Engineering without changing Python internals.

## What you author

You author markdown rules.
At runtime, rules are parsed, compiled, and applied during generation.

For exact syntax and normalization details, see `../rules/language-reference.md`.

## Rule families at a glance

- **Replace**: replace a matched phrase with approved alternatives (`With:`).
- **After**: add approved text after an anchor (`Add:`).
- **Avoid**: detect risky trajectories and rewrite with guardrails (`Prefix`/`Connector`/`Postfix` + `Fallback`).
- **Force**: enforce a scope-wide statement (`Add:` candidates).

## Authoring workflow

1. Start with one clear intent per rule.
2. Keep anchors specific enough to avoid accidental matches.
3. Use `Scope:` to constrain where a rule can act.
4. For Avoid rules, include explicit `Fallback:` text.
5. Test with both trigger and non-trigger examples.

## Practical defaults

If omitted:

- `Replace` and `After` default to `fire: once`.
- `Avoid` defaults to `fire: repeat`.
- `Force` defaults to `fire: once`.
- Scope defaults to whole-document behavior.
- Guard defaults include:
  - Replace/After `Prefix` default operator: `any`
  - Avoid `Prefix` default operator: `all`
  - Avoid `Postfix` default operator: `all`

## Example

<!-- ae-example -->
```ae-rules
## Replace (once): sensorineural hearing loss

Prefix (any):
* sudden
* acute

With:
* sudden sensorineural hearing loss
* SSNHL

Scope:
* 800 chars
```

If two rules propose overlapping edits, runtime selection keeps a deterministic winner for that overlap.
