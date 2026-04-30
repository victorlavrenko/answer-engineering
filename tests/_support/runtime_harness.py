from __future__ import annotations

from answer_engineering.config.patch_score_policy import PatchScorePolicy
from answer_engineering.inference.decode.session_orchestration import (
    ExecutionSession,
)
from answer_engineering.inference.model_types import (
    TokenGenerationRuntime,
)


def configure_runtime_scoring(
    runtime: ExecutionSession,
    *,
    generation_runtime: TokenGenerationRuntime | None,
    require_model_scoring: bool,
    patch_score_policy: PatchScorePolicy | None = None,
) -> None:
    runner = runtime.runner
    runner.runtime = generation_runtime
    runner.require_model_scoring = require_model_scoring
    runner.patch_score_policy = patch_score_policy
