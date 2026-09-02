# Instruction Duplication as an Inference-Time Control Primitive

This directory is the paper-specific entry point for the Instruction Duplication preprint within the Answer Engineering repository.

## Paper

- LaTeX source: [`main.tex`](main.tex)
- Frozen source version for the preprint: tag `instruction-duplication-arxiv-v1`
- Frozen experiment artifacts: [GitHub Release `instruction-duplication-arxiv-v1`](https://github.com/victorlavrenko/answer-engineering/releases/tag/instruction-duplication-arxiv-v1)

## Primary instruction-placement experiment

The implementation is shipped in [`src/instruction_duplication/`](../../../src/instruction_duplication/). The imported source snapshot is version 3.0.13 from `victorlavrenko/instruction-duplication` commit `cbb55629e53802056ac7554ccd5b9a11f39beb7f`; provenance and the retained experiment documentation are under [`reproduction/`](reproduction/).

Large generated outputs are intentionally not committed to Git. The paper release is the immutable distribution point for the frozen 16,800-cell run, including the selected questions, generations, deterministic judgments, analysis products, robustness outputs, and human challenge-audit material.

## Answer Engineering placement experiment

The executable and inspectable entry point is:

- repository notebook: [`notebooks/instruction-placement-reproduction.ipynb`](../../../notebooks/instruction-placement-reproduction.ipynb)
- [open the notebook in Google Colab](https://colab.research.google.com/github/victorlavrenko/answer-engineering/blob/main/notebooks/instruction-placement-reproduction.ipynb)

The committed notebook preserves the executed aggregate outputs. The paper release additionally carries the persisted case-level reproduction used to audit the placement result.

Frozen operational endpoints:

- **SSNHL target:** the lowercased response contains `steroid`. Dose, route, urgency, and appropriateness are not separately scored by this endpoint.
- **Conductive contrast:** the response contains `conductive` and contains neither `senso` nor `ssnhl`. This is diagnostic branch preservation only; it does not test steroid absence or conductive-treatment correctness.

Reported placement reproduction:

| Scope | System-only AE | AE + trailing duplicate |
|---|---:|---:|
| SSNHL target | 842/1000 (84.2%) | 971/1000 (97.1%) |
| Conductive branch preservation | 786/1000 (78.6%) | 738/1000 (73.8%) |

## Release assets

The intended release contains:

- `instruction-duplication-paper-run.tar.gz` — frozen primary experiment and audit artifacts.
- `instruction-placement-results.jsonl.gz` — complete persisted 4,000-response AE placement reproduction.
- `SHA256SUMS` — SHA-256 integrity hashes for the release assets.

Do not substitute partial notebook downloads for the complete 4,000-response case-level artifact.

## Verification principle

Paper-facing analysis should be reproducible from frozen outputs without regenerating hosted-model responses. Full model regeneration is a separate, more expensive path and can differ at the infrastructure level because provider-side bitwise determinism is not guaranteed.
