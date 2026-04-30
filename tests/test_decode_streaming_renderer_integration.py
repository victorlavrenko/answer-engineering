from __future__ import annotations

from typing import cast

import torch
from _pytest.capture import CaptureFixture

from answer_engineering.inference.contracts import (
    GenerationPolicy,
    GenerationRequest,
)
from answer_engineering.inference.decode.decode_loop import GreedyDecoder
from answer_engineering.inference.model_types import ChatGenerationRuntime
from tests.core._scoring_stubs import GenerationRuntimeStub


def test_verbosity_one_streams_without_debug(
    capsys: CaptureFixture[str],
) -> None:
    runtime = GenerationRuntimeStub.loaded_runtime()
    request = GenerationRequest(question="Q", partial_answer="")
    policy = GenerationPolicy(max_new_tokens=2, stop_on_eos=False, verbosity=1)

    _ = GreedyDecoder(
        runtime=cast(ChatGenerationRuntime, runtime),
        input_ids=torch.tensor([[1]], dtype=torch.long),
        request=request,
        policy=policy,
    ).decode()

    out = capsys.readouterr().out
    assert out == "xx\n"
    assert "[AE]" not in out


def test_verbosity_two_flushes_visible_text_before_debug(
    capsys: CaptureFixture[str],
) -> None:
    runtime = GenerationRuntimeStub.loaded_runtime()
    request = GenerationRequest(question="Q", partial_answer="seed")
    policy = GenerationPolicy(
        rules="## Replace: x\n\nWith:\n\n- z",
        max_new_tokens=1,
        stop_on_eos=False,
        verbosity=2,
    )

    _ = GreedyDecoder(
        runtime=cast(ChatGenerationRuntime, runtime),
        input_ids=torch.tensor([[1]], dtype=torch.long),
        request=request,
        policy=policy,
    ).decode()

    out = capsys.readouterr().out
    assert "[AE]" in out
    assert out.startswith("seed")
    assert "[AE]" in out.splitlines()[1]
    assert out.endswith("z\n")
