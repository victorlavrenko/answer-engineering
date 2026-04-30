"""Remote publication connectors for reproduction artifacts.

Purpose:
    Define the publication adapter contracts and Hugging Face–backed connector
    implementations used to publish reproduction artifacts.

Architectural role:
    Remote-publication boundary for reproduction reporting outputs.

Architectural direction:
    Keep backend-specific mechanics behind connector contracts so publication
    workflows depend on explicit boundaries rather than backend-shaped
    assumptions.

Why this matters:
    The current implementation is practical and Hugging Face–centered, but it
    should not become the conceptual center of reproduction publishing.

What better would look like:
    Publication workflows remain stable while backend adapters evolve or new
    artifact publication targets are introduced.

How improvement can be recognized:
    - Clearer separation between publication workflow policy and backend APIs
    - Fewer Hugging Face assumptions leaking into higher-level reporting code
    - Lower cost to add new publication connectors

Open constraint:
    The adapter boundary should remain open to future artifact publication
    requirements.

"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from huggingface_hub import CommitOperationAdd, HfApi, login

from ae_paper_reproduction.config.hf_defaults import HuggingFaceDefaults

try:
    from google.colab import userdata as _colab_userdata_module
except ImportError:
    _colab_userdata_module = None


class ArtifactPublisher(Protocol):
    """Protocol for publishing reproduction artifacts to a remote dataset.

    Purpose:
        Define the minimal repository-creation, commit, and file-upload
        operations needed by code that pushes artifacts to Hugging Face.

    Architectural role:
        Remote-publication boundary in the reproduction adapter layer.

    Inputs (architectural provenance):
        Implemented by concrete remote publishers such as
        `HuggingFaceArtifactPublisher`.

    Outputs (downstream usage):
        Consumed by code that exports telemetry, reports, or other generated
        artifacts.

    Invariants/constraints:
        Implementations must treat `dataset_id` and `token` as referring to the
        same authenticated target repository.

    """

    def ensure_dataset_repo(
        self, *, dataset_id: str, private: bool, token: str
    ) -> None:
        """Ensure that a target dataset repository exists.

        Purpose:
            Define the repository-provisioning operation required before
            artifact commits or uploads.

        Architectural role:
            Protocol member on the remote artifact-publication boundary.

        Inputs (architectural provenance):
            Receives repository identity, authentication, and operation-
            specific data from reproduction export workflows.

        Outputs (downstream usage):
            Leaves remote artifact state updated or ready for later publication
            steps; callers consume completion rather than backend-specific
            return values.

        Invariants/constraints:
            Implementations must apply the operation to the authenticated
            dataset target represented by the supplied dataset id and token.

        """
        raise NotImplementedError

    def commit(
        self,
        *,
        dataset_id: str,
        operations: Iterable[CommitOperationAdd],
        message: str,
        token: str,
    ) -> None:
        """Create one remote commit containing artifact operations.

        Purpose:
            Define the batch-publication operation for prepared artifact
            changes.

        Architectural role:
            Protocol member on the remote artifact-publication boundary.

        Inputs (architectural provenance):
            Receives repository identity, authentication, and operation-
            specific data from reproduction export workflows.

        Outputs (downstream usage):
            Leaves remote artifact state updated or ready for later publication
            steps; callers consume completion rather than backend-specific
            return values.

        Invariants/constraints:
            Implementations must apply the operation to the authenticated
            dataset target represented by the supplied dataset id and token.

        """
        raise NotImplementedError

    def upload_file(
        self,
        *,
        path_or_fileobj: str | bytes | Path,
        path_in_repo: str,
        dataset_id: str,
        token: str,
    ) -> None:
        """Upload one artifact file to the target repository.

        Purpose:
            Define the single-file publication operation used when callers do
            not build an explicit commit batch.

        Architectural role:
            Protocol member on the remote artifact-publication boundary.

        Inputs (architectural provenance):
            Receives repository identity, authentication, and operation-
            specific data from reproduction export workflows.

        Outputs (downstream usage):
            Leaves remote artifact state updated or ready for later publication
            steps; callers consume completion rather than backend-specific
            return values.

        Invariants/constraints:
            Implementations must apply the operation to the authenticated
            dataset target represented by the supplied dataset id and token.

        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class HuggingFaceArtifactPublisher(ArtifactPublisher):
    """Concrete remote publisher backed by `huggingface_hub`.

    Purpose:
        Adapt the generic artifact-publishing protocol to Hugging Face dataset
        repositories using the configured defaults.

    Architectural role:
        Hugging Face–specific adapter in the remote-publication layer.

    Inputs (architectural provenance):
        Constructed with an `HfApi` client and Hugging Face defaults.

    Outputs (downstream usage):
        Provides repository and upload operations consumed by reproduction
        export code.

    Invariants/constraints:
        All remote operations must use the dataset repo type declared in the
        stored defaults.

    """

    api: HfApi
    defaults: HuggingFaceDefaults

    def ensure_dataset_repo(
        self, *, dataset_id: str, private: bool, token: str
    ) -> None:
        """Ensure that the target dataset repository exists.

        Purpose:
            Create the destination Hugging Face dataset repository when
            necessary and otherwise leave the existing repository untouched.

        Architectural role:
            Repository-provisioning operation on the Hugging Face publisher
            adapter.

        Inputs (architectural provenance):
            Consumes repository identity, privacy setting, and an authentication
            token from reproduction export code.

        Outputs (downstream usage):
            Leaves the remote repository ready for later commit or upload
            operations.

        Invariants/constraints:
            The operation must target the dataset repo type declared by the
            stored defaults.

        """
        self.api.create_repo(
            repo_id=dataset_id,
            repo_type=self.defaults.repo_type_dataset,
            private=private,
            token=token,
            exist_ok=True,
        )

    def commit(
        self,
        *,
        dataset_id: str,
        operations: Iterable[CommitOperationAdd],
        message: str,
        token: str,
    ) -> None:
        """Create one remote commit containing a batch of artifact operations.

        Purpose:
            Push a prepared set of add operations to the target Hugging Face
            dataset repository under one commit message.

        Architectural role:
            Batch-publication operation on the Hugging Face publisher adapter.

        Inputs (architectural provenance):
            Consumes prepared commit operations, a dataset id, a commit message,
            and an auth token from export code.

        Outputs (downstream usage):
            Persists artifact changes to the remote repository.

        Invariants/constraints:
            All operations in the batch must be committed to the same target
            dataset repository.

        """
        self.api.create_commit(
            dataset_id,
            operations,
            commit_message=message,
            repo_type=self.defaults.repo_type_dataset,
            token=token,
        )

    def upload_file(
        self,
        *,
        path_or_fileobj: str | bytes | Path,
        path_in_repo: str,
        dataset_id: str,
        token: str,
    ) -> None:
        """Upload one file object or path to the target repository.

        Purpose:
            Publish a single file into the configured Hugging Face dataset
            repository without building an explicit commit operation list.

        Architectural role:
            Single-file upload operation on the Hugging Face publisher adapter.

        Inputs (architectural provenance):
            Consumes a file source, destination path, dataset id, and auth token
            from export code.

        Outputs (downstream usage):
            Persists the uploaded file to the remote repository.

        Invariants/constraints:
            Uploads must target the dataset repo type declared by the stored
            defaults.

        """
        self.api.upload_file(
            path_or_fileobj=path_or_fileobj,
            path_in_repo=path_in_repo,
            repo_id=dataset_id,
            repo_type=self.defaults.repo_type_dataset,
            token=token,
        )


class ColabUserdataStore(Protocol):
    """Protocol for the subset of Colab userdata access used by auth resolution.

    Purpose:
        Abstract the `get()` operation used to retrieve secrets from Colab
        userdata without binding callers to the concrete Colab module object.

    Architectural role:
        Tiny capability protocol at the Colab-auth representation boundary.

    Inputs (architectural provenance):
        Implemented by the Colab userdata object when running inside Google
        Colab.

    Outputs (downstream usage):
        Consumed by `HuggingFaceAuthResolver` when searching for a token.

    Invariants/constraints:
        Implementations should return either a string token or `None`.

    """

    def get(self, key: str) -> str | None:
        """Return the secret value for a key when available.

        Purpose:
            Provide a narrow adapter over Colab userdata for credentials used by
            remote connectors.

        Architectural role:
            Infrastructure boundary between notebook runtime secrets and
            repository code that needs authentication tokens.

        Inputs (architectural provenance):
            Receives a secret key name from connector or reporting
            configuration.

        Outputs (downstream usage):
            Returns the secret string for remote operations, or `None` when the
            value is unavailable.

        Invariants/constraints:
            The method should not log secret values or convert absence into a
            misleading placeholder token.

        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class HuggingFaceAuthResolver:
    """Authentication helper for obtaining Hugging Face tokens in notebook or.

    Purpose:
        Resolve an access token from Colab userdata or environment variables and
        support explicit relogin when needed.

    Architectural role:
        Auth-resolution adapter in the remote-publication layer.

    Inputs (architectural provenance):
        Constructed with Hugging Face defaults describing the expected token
        environment name and Colab module hints.

    Outputs (downstream usage):
        Provides resolved tokens consumed by remote publisher code.

    Invariants/constraints:
        Token lookup order must remain deterministic: Colab userdata first, then
        process environment.

    """

    defaults: HuggingFaceDefaults

    def require_token(self, env_name: str | None = None) -> str:
        """Return the required Hugging Face token or fail clearly.

        Purpose:
            Resolve the token from Colab userdata or environment variables and
            raise a runtime error when no usable token is available.

        Architectural role:
            Primary token-resolution method on the auth helper.

        Inputs (architectural provenance):
            Consumes an optional environment-variable name override from callers
            that need a non-default token key.

        Outputs (downstream usage):
            Returns the resolved token string consumed by publisher operations.

        Invariants/constraints:
            The returned token must come from either the requested env name or
            the configured default token name.

        """
        token_key = env_name or self.defaults.token_env_name
        token = self._token_from_colab_userdata(token_key) or os.getenv(
            token_key
        )
        if token:
            return token
        raise RuntimeError(
            "Missing Hugging Face token. "
            f"Set {token_key} and re-run with "
            "PUSH_TELEMETRY_TO_HF=True."
        )

    def relogin(self, *, token: str) -> None:
        """Log in to Hugging Face again with the supplied token.

        Purpose:
            Refresh the local Hugging Face client authentication state
            explicitly instead of relying on previously cached credentials.

        Architectural role:
            Authentication side-effect method on the auth helper.

        Inputs (architectural provenance):
            Consumes a token already resolved by caller code.

        Outputs (downstream usage):
            Updates local Hugging Face login state for later remote operations.

        Invariants/constraints:
            Relogin should not silently fall back to unrelated credential
            sources.

        """
        login(token=token, add_to_git_credential=False, skip_if_logged_in=False)

    def _token_from_colab_userdata(self, token_key: str) -> str | None:
        """Try to read the Hugging Face token from Colab userdata.

        Purpose:
            Inspect the active Colab runtime or imported fallback module for a
            userdata object that can supply the requested secret key.

        Architectural role:
            Colab-specific secret lookup helper inside the auth resolver.

        Inputs (architectural provenance):
            Consumes the token-key name chosen by `require_token()`.

        Outputs (downstream usage):
            Returns the token string when Colab userdata lookup succeeds,
            otherwise `None`.

        Invariants/constraints:
            Failures in Colab-specific lookup must be contained and return
            `None` rather than aborting auth resolution.

        """
        if self.defaults.colab_module_name not in sys.modules:
            return None
        colab_module = sys.modules.get(self.defaults.colab_module_name)
        if colab_module is not None:
            colab_userdata_attr = getattr(colab_module, "userdata", None)
            if colab_userdata_attr is not None:
                colab_userdata = cast(ColabUserdataStore, colab_userdata_attr)
                try:
                    return colab_userdata.get(token_key)
                except (AttributeError, KeyError, TypeError, ValueError):
                    return None

        if _colab_userdata_module is None:
            return None
        colab_userdata = cast(ColabUserdataStore, _colab_userdata_module)
        try:
            return colab_userdata.get(token_key)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None


__all__ = [
    "ArtifactPublisher",
    "HuggingFaceArtifactPublisher",
    "HuggingFaceAuthResolver",
]
