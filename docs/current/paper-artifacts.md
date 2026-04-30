# Paper Artifacts

The reproduction reporting pipeline generates paper-facing artifacts from group run outputs.

## Where artifacts are produced

- Group outputs are written under `reports/runs/run-<group_run_id>/`.
- Paper metrics macros are written to `docs/paper/generated`.

## Core paper/report artifacts

From the group artifact bundle:

- `paper_metrics.json`
  - group-level metadata describing the generated paper metrics artifact path.
- `paper-metrics.tex`

The generated `.tex` artifact is consumed by `docs/paper/main.tex`.

## Architectural maturity note

The current artifact flow is functional and authoritative for paper outputs.

Artifact generation still reflects the present reporting pipeline shape rather than a finalized platform reporting model.

## How this fits the pipeline

1. Subrun results are evaluated and aggregated into group telemetry.
2. Group reporting derives publication metrics from telemetry.
3. Artifact materialization writes JSON report artifacts and `paper-metrics.tex`.

For execution flow details, see `reproducibility.md`.
