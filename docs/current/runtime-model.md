# Current Runtime Model

This document describes the **current runtime behavior** of one generation call. It is based on the code path implemented today.

---

## 1. Public execution entrypoint

The current entrypoint is:

`GenerationRuntime.generate(request, policy)`

Current behavior of this method:

1. materialize tokenizer / model resources if needed
2. force eval mode
3. create a `StreamSession`
4. run one streaming generation session
5. return a `GenerationResult`

`GenerationRuntime` also exposes lower-level services used internally by the session and decode paths:

- `text_codec()`
- `execution_device()`
- `forward(...)`
- `generate_tokens(...)`

---

## 2. Per-call session setup

`StreamSession.run()` is the current top-level per-call coordinator.

Current responsibilities:

- build chat messages from the system prompt and question
- build prompt input ids
- optionally print the pre-existing partial assistant answer
- execute the greedy decode loop
- attach runtime telemetry and elapsed time
- package the final `GenerationResult`

The request currently supports:

- `question`
- `partial_answer`

The policy currently controls, among other things:

- compiled or raw rules
- system prompt
- `max_new_tokens`
- `stop_on_eos`
- verbosity

Rule text is compiled eagerly inside `GenerationPolicy` so downstream code sees one canonical `compiled_rules` view.

---

## 3. Decode model

The current decode path is a **greedy token-by-token decode loop**.

`GreedyDecoder.decode()` currently:

1. prepares prompt ids
2. runs an initial forward pass
3. initializes mutable `StreamingDecodeState`
4. optionally initializes rule execution if rules are present
5. iterates for at most `policy.max_new_tokens`
6. picks the next token with `argmax`
7. appends the token to visible assistant text
8. optionally executes Answer Engineering rule logic
9. stops on EOS if `policy.stop_on_eos` is enabled
10. packages final text, ids, and telemetry

Current decode style:

- greedy
- incremental
- streaming-capable
- optionally rule-intervened

This is not currently a branching decode runtime.

---

## 4. Behavior when rules are absent

If `policy.compiled_rules` is `None`, the runtime currently behaves as a plain greedy decoding loop with no Answer Engineering intervention.

In that case:

- no `ExecutionSession` is created
- no rule events are recorded
- telemetry remains the empty runtime snapshot (with runtime duration later
  attached)

---

## 5. Behavior when rules are present

If rules are present, the decode loop currently creates:

- a `PlanRunner`
- an `ExecutionSession`
- a recording telemetry sink

On each generated token, the decoder can invoke:

`execution_session.apply_step(...)`

That step:

- builds a `StepSnapshot`
- runs the compiled plan through `PlanRunner`
- receives a `Decision`
- checks whether the visible assistant text changed
- marks the decode state for rebuild when needed

So the current runtime model is:

**generate one token -> optionally run rule logic -> rebuild if edited -> continue**

---

## 6. Current edit and rebuild behavior

When a rule decision changes the visible assistant text, the decode state is
marked with `needs_rebuild = True`.

Current rebuild path:

1. retokenize the current visible assistant text
2. reconstruct token-to-character alignment
3. rebuild the full prefix ids = prompt ids + generated assistant ids
4. prefill the model again to recover:
   - `past_key_values`
   - next logits
5. clear the rebuild flag and continue decoding

This means the current system supports local text edits, but continuation after an edit is currently implemented by rebuilding the visible assistant prefix and cached decode state rather than by preserving a more fine-grained causal continuation structure.

Important nuance:

- at the document-text level, replace edits are local edits
- at the decode-state level, the runtime still rebuilds from the edited
  visible prefix after a change

That is the current implemented model.

---

## 7. Streaming behavior

The runtime currently supports console streaming controlled by policy verbosity.

Current behavior:

- `verbosity >= 1` enables stream output
- `verbosity >= 2` enables debug output

When an edit changes already-printed text, the current streaming behavior can rerender from the start so the visible console output matches the repaired text.

When an edit does not require rewriting already-printed text, the runtime can skip full rerender and continue after logging the rebuild reason.

---

## 8. Current telemetry flow

When rules are enabled, the runtime currently records engine events through a telemetry sink during decode.

At the end of the run:

1. recorded events are aggregated
2. a `RuntimeTelemetrySnapshot` is built
3. runtime duration is attached
4. the snapshot is placed into `GenerationResult.ae_telemetry`

So public results always return a telemetry snapshot, but meaningful runtime intervention telemetry is only populated when rule execution actually occurred.

---

## 9. Current stop conditions

The current decode loop stops when either:

- `policy.max_new_tokens` is exhausted
- EOS is generated and `policy.stop_on_eos` is true

The loop is therefore budget-bounded even when rules are enabled.

---

## Implementation sequencing effects

Some runtime mechanics reflect the order in which trajectory-control and intervention experiments were implemented.

The current execution path prioritizes reproducible intervention behavior and reliable stepwise edits. As a result, edit-triggered rebuild mechanics are explicit and some runtime seams remain coupled to proposal-layer coordination.

These mechanics are functional but may not represent the final scalable execution model.

---

## 10. Current runtime character

Today the runtime is best described as:

- a model-backed greedy decode loop
- with optional deterministic rule intervention
- with runtime telemetry
- with edit-triggered prefix/cache rebuild

It is already more than pure post-processing, but it is not yet a full branch-aware or causal-repair runtime.
