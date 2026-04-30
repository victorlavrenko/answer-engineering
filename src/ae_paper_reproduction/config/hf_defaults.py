"""Hugging Face connector defaults for reproduction workflows.

Purpose:
    Keep repository-wide defaults for dataset, model, and artifact access in one
    small configuration module.

Architectural role:
    Reproduction configuration boundary between notebooks, runner code, and
    infrastructure connectors.

Inputs (architectural provenance):
    Values are authored as constants and may be referenced by notebook-facing
    execution helpers.

Outputs (downstream usage):
    Provides stable default names, paths, or identifiers consumed by Hugging
    Face loading and publishing code.

Invariants/constraints:
    Defaults should be boring, explicit, and easy to override at the workflow
    edge rather than hidden inside connector implementations.

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HuggingFaceDefaults:
    """Default connector settings for Hugging Face access.

    Purpose:
        Hold the environment-variable name and canonical repository metadata
        used by the dataset/model integration helpers.

    Architectural role:
        Infrastructure-configuration value object shared by remote-loading code
        and notebook/runtime environments.

    Inputs (architectural provenance):
        Consulted by Hugging Face adapters when resolving auth tokens, dataset
        repo type, and Colab-specific environment detection.

    Outputs (downstream usage):
        Produces stable connector constants consumed by remote dataset/model
        setup.

    Invariants/constraints:
        Field values are static process defaults rather than mutable session
        state.

    """

    token_env_name: str = "HF_TOKEN"
    repo_type_dataset: str = "dataset"
    colab_module_name: str = "google.colab"


__all__ = ["HuggingFaceDefaults"]
