# Instruction Duplication as an Inference-Time Control Primitive

This directory is the paper-specific entry point for the Instruction Duplication preprint within the Answer Engineering repository.

## Paper and frozen source

- LaTeX source: [`main.tex`](main.tex)
- Rendered PDF: [`paper.pdf`](paper.pdf)
- Frozen source tag: `instruction-duplication-arxiv-v1`
- Frozen experiment artifacts: [GitHub Release `instruction-duplication-arxiv-v1`](https://github.com/victorlavrenko/answer-engineering/releases/tag/instruction-duplication-arxiv-v1)

The reusable implementation is in [`src/instruction_duplication/`](../../../src/instruction_duplication/). It is the Instruction Duplication 3.0.13 source snapshot imported from `victorlavrenko/instruction-duplication` commit `cbb55629e53802056ac7554ccd5b9a11f39beb7f`. Generated responses are intentionally kept out of ordinary Git history.

## Release assets

The tagged release distributes exactly three files:

| Asset | Purpose | SHA-256 |
|---|---|---|
| `instruction-duplication-paper-run.tar.gz` | Frozen primary 7-model x 300-question x 8-condition run, deterministic judgments, human audit, robustness and measurement-validation artifacts | `b4094a39981c427406aa925d0a3b8d9bfcaf1dadeecf4acacfa73903e86b7eec` |
| `instruction-placement-results.jsonl.gz` | Complete persisted 4,000-response Answer Engineering placement reproduction | `c004b8b83691b5fdc4aada06481259c5e64237ad6fe80a82c8291965c8c37897` |
| `SHA256SUMS` | Integrity manifest for the two data assets | - |

After downloading all three assets into one directory, verify them with:

```bash
python scripts/instruction_duplication/verify_release_assets.py /path/to/release-assets
```

The verifier checks the hashes, the 16,800-cell primary export, the 300 primary questions, all 4,000 AE responses, 2,000 unique AE questions, the frozen endpoint rules, aggregate scores, and paired transitions.

## Primary experiment: deterministic reanalysis

Extract the primary archive:

```bash
tar -xzf instruction-duplication-paper-run.tar.gz
WORKSPACE="$PWD/instruction-duplication-paper-run/paper-run"
```

The release intentionally omits the derived SQLite index. Reconstruct it from the frozen all-cell JSONL without contacting any model provider:

```bash
python scripts/instruction_duplication/restore_workspace_db.py \
  --workspace "$WORKSPACE" \
  --include-judgments
```

For a stronger check, reproduce every deterministic judgment from the frozen model outputs rather than importing the stored judgments:

```bash
python scripts/instruction_duplication/verify_rejudging_equivalence.py \
  --workspace "$WORKSPACE"
```

For the paper-facing robustness analysis, use:

```bash
python scripts/instruction_duplication/robustness_analysis.py \
  --workspace "$WORKSPACE" \
  --human-dir "$PWD/instruction-duplication-paper-run/human-validation" \
  --out /tmp/instruction-duplication-analysis.json \
  --tables-dir /tmp/instruction-duplication-tables \
  --verify-paper-claims
```

These paths operate only on frozen responses and deterministic analysis; they do not regenerate hosted-model answers.

## Answer Engineering placement experiment

Interactive reproduction and inspection:

- repository notebook: [`notebooks/instruction-placement-reproduction.ipynb`](../../../notebooks/instruction-placement-reproduction.ipynb)
- [open in Google Colab](https://colab.research.google.com/github/victorlavrenko/answer-engineering/blob/main/notebooks/instruction-placement-reproduction.ipynb)

The committed notebook preserves the executed aggregate outputs. The release asset `instruction-placement-results.jsonl.gz` preserves the independently persisted case-level reproduction used to audit those numbers.

Frozen operational endpoints:

- **SSNHL target:** the lowercased response contains `steroid`. Dose, route, urgency, and appropriateness are not separately scored.
- **Conductive contrast:** the response contains `conductive` and contains neither `senso` nor `ssnhl`. This tests diagnostic branch preservation only; it does not test steroid absence or conductive-treatment correctness.

| Scope | System-only AE | AE + trailing duplicate | Paired changes |
|---|---:|---:|---:|
| SSNHL target | 842/1000 (84.2%) | 971/1000 (97.1%) | 154 improved / 25 degraded |
| Conductive branch preservation | 786/1000 (78.6%) | 738/1000 (73.8%) | 153 improved / 201 degraded |

The AE artifact embeds the exact question text in each result row, so a second question-only release asset is unnecessary.

## Reproduction boundary

Paper-facing analysis is reproducible from the frozen outputs without provider credentials. Exact regeneration of hosted-model responses is a separate path and can differ because provider-side bitwise determinism and serving infrastructure are not fully exposed.
