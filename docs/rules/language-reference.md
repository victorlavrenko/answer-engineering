# Rule Language Reference

This is the reference for the Markdown rules domain-specific language that is parsed by `MarkdownRulesParser` and compiled by `FullPlanCompiler`.

## Rule header

```text
## <Kind> (<modifier1>, <modifier2>, ...): <target>
```

- `Kind` is one of: `Replace`, `After`, `Avoid`, `Force`.
- Header modifiers are comma-separated.
- `target` is required text after `:`.

### Common fire modifiers

- `once`
- `repeat`

### Match modifiers (matching constructs only)

Canonical: `case-sensitive`, `case-insensitive`, `word`.

Aliases:

- case-sensitive: `respect-case`, `respect case`, `casefold-false`, `casefold=false`
- case-insensitive: `ignore-case`, `ignore case`, `casefold`, `casefold-true`, `casefold=true`
- word: `whole-word`, `whole word`

`case` is intentionally rejected as ambiguous.

Precedence: `item override > section override > rule override > engine default`.

Engine defaults are owned by `MatchDefaults` and are currently case-insensitive with `word=False`.

Authored/inherited syntax-level override state is represented by `MatchOptionsAST`, and resolved runtime/compiled behavior is represented by `ResolvedMatchOptions` via `resolve_match_options(...)`.

### Avoid edit-target modifiers

- `postfix`
- `prefix clause`
- `matched prefix clause`
- `clause containing anchor to scope end`
- `clause_containing_anchor_to_scope_end`
- `everything` / `all`
- `last sentence`
- `N sentences` / `N last sentences`
- `last clause`
- `N clauses` / `N last clauses`

## Supported sections

Section names are case-insensitive.
Bullets must use `*` or `-`.

### Replace

- `With:` (replacement candidates)
- optional `Prefix` guards: `Prefix`, `Prefix (any|all|none|incomplete)`
- optional `Prompt` guards: `Prompt (...)`
- optional `Scope:`

### After

- `Add:` (inserted candidates)
- optional `Prefix` guards
- optional `Prompt` guards
- optional `Options:` (after-rule options, including closing-parenthesis behavior)
- optional `Scope:`

### Avoid

- optional `Prefix` guards (`Prefix` defaults to `all` when operator omitted)
- optional `Prompt` guards
- optional `Connector:`
- optional `Postfix` guards (`Postfix` defaults to `all` when operator omitted)
- optional `Fallback:`
- optional `Options:` (probe/scoring config)
- optional `Scope:`

### Force

- `Add:`
- optional `Scope:`

## Guard operators

Supported guard operators are:

- `any`
- `all`
- `none`
- `incomplete`

`partial` and `missing` are accepted as synonyms for `incomplete`.

## Item-level match modifiers

Matching sections (`Prefix`, `Postfix`, `Prompt`) accept item overrides:

```text
Prefix (any, case-insensitive):
- API
- (case-sensitive) Flask
- (word) API
- (case-sensitive, word) Flask
```

## Strict modifier validation

The parser is fail-closed: unsupported modifiers are language errors and are never silently ignored.

- Unsupported modifiers in headers/sections/items raise `RulesSyntaxError`.
- Contradictory case modifiers in one list raise `RulesSyntaxError`.
- Matching modifiers are rejected in emission-only sections (`With`, `Add`, `Fallback`, etc.).

TODO: `ResolvedMatchOptions` is the runtime extension point for future richer match modes (for example, regex mode selection) if/when the domain-specific language adds them.

Note: avoid anchor extraction preserves per-term resolved match options for marker-derived anchor phrase lookup.

## Scope syntax

`Scope:` bullets are normalized as follows:

- `all`, `from beginning`, `from the beginning`, `from the start` → whole document
- text containing `clause` (optionally with leading number) → tail clauses
- text containing `sentence` (optionally with leading number) → tail sentences
- text containing `char` with a number → tail chars
- otherwise defaults to whole document

## Options syntax

`Options:` bullets use `key: value` and parse numeric values only.

### Avoid options

Normalized option keys:

- `probe_num_beams` via aliases: `num_beams`, `beams`, `beam_width`, `width`, `k`, `trajectories`, `max trajectories`, `max_trajectories`
- `probe_max_new_tokens` via aliases: `max_new_tokens`, `probe_max_new_tokens`, `tokens`, `token_count`
- `min_prob_ratio_to_best` via aliases: `min probability ratio`, `min_prob_ratio_to_best`
- `skip` via alias: `skip`

### After options

`Options:` supports parenthesis-wait control through keys such as:

- `regime` / `fire regime` / `parenthesis regime`
- `wait for closing` / `wait for closing parenthesis`

## Template expansion

Bullets can include template variants using unescaped pipes:

- `|` = dimension 1
- `||` = dimension 2
- `|||` = dimension 3

Rules:

- same dimension markers zip together,
- different dimensions cross-multiply,
- mixed separator lengths in one bullet are invalid,
- escaped pipes (`\|`) are literals.

## Minimal example

```ae-rules
## Avoid (postfix, repeat): conductive

Prefix (all):
* weber
* rinne

Connector:
* this suggests

Postfix (any):
* conductive

Fallback:
* these findings require further evaluation.
```
