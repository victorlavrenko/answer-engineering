# Reproducibility Guide

This document provides practical instructions for reproducing the experiments and inspecting generated artifacts in the **Answer Engineering** codebase.

------------------------------------------------------------------------

# Entry Points

## Full reproduction notebook

Start here to reproduce the paper results:

-   [notebooks/reproduce.ipynb](../../notebooks/reproduce.ipynb)

This notebook:

1.  installs dependencies
2.  downloads the dataset
3.  loads the model
4.  executes evaluation subruns
5.  generates telemetry
6.  regenerates paper metrics

Recommended runtime:

- GPU: Google Colab G4 runtime (tested on NVIDIA RTX PRO 6000 Blackwell, ~96 GB VRAM)
- Minimum supported GPU: 16 GB VRAM
- Environment: Google Colab or local GPU workstation

------------------------------------------------------------------------

## Quickstart notebook (mechanism exploration)

Use this notebook to experiment with trajectory editing:

-   [notebooks/quickstart.ipynb](../../notebooks/quickstart.ipynb)

------------------------------------------------------------------------

# Core Reproduction API

Public reproduction API:

-   [src/ae_paper_reproduction/api.py](../../src/ae_paper_reproduction/api.py)

Top-level execution service:

-   [src/ae_paper_reproduction/runner/session/reproduction_session.py](../../src/ae_paper_reproduction/runner/session/reproduction_session.py)

Summary and artifact assembly:

-   [src/ae_paper_reproduction/runner/session/summary.py](../../src/ae_paper_reproduction/runner/session/summary.py)

Artifact writer:

-   [src/ae_paper_reproduction/telemetry/artifacts.py](../../src/ae_paper_reproduction/telemetry/artifacts.py)

------------------------------------------------------------------------

# Output Locations

Evaluation produces outputs in two destinations.

## Telemetry and evaluation artifacts

Written locally under:

    reports/runs/run-<group_run_id>/

Each subrun is written under:

    reports/runs/run-<group_run_id>/subrun-<subrun_id>/

Typical contents:

-   evaluation summaries
-   telemetry statistics
-   comparison reports
-   configuration manifests

These directories are created automatically during execution.

They are not committed to Git.

------------------------------------------------------------------------

## Generated paper metrics

All paper numbers are generated automatically into:

-   [docs/paper/generated/paper-metrics.tex](../../docs/paper/generated/paper-metrics.tex)

This file is included directly by the manuscript:

    \input{generated/paper-metrics.tex}

This mechanism guarantees that reported metrics are generated programmatically.

------------------------------------------------------------------------

# Configuration

Default evaluation configuration:

-   [config/](../../config)

Rulesets used in experiments:

-   [rules/](../../rules)

------------------------------------------------------------------------

# Optional Telemetry Publication

Telemetry artifacts can be published automatically to a Hugging Face dataset.

Configuration variables:

    PUSH_TELEMETRY_TO_HF = True
    HF_TELEMETRY_DATASET_ID = "<YOUR_USERNAME>/<YOUR_DATASET>"

If the dataset does not exist, it is created automatically.

Example public dataset:

https://huggingface.co/datasets/lavrenko/answer-engineering

------------------------------------------------------------------------

# Deterministic Reproduction

Reproduction is deterministic under the following conditions:

-   fixed random seeds
-   identical configuration
-   identical model version
-   identical dataset version

All parameters required for reproducibility are recorded in the run manifest generated for each evaluation.
