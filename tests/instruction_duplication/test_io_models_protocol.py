from __future__ import annotations

import json
from pathlib import Path

import pytest

from instruction_duplication.io_utils import canonical_json, write_json, write_jsonl
from instruction_duplication.models import MODEL_BY_ID, Model, select_models
from instruction_duplication.protocol import PROTOCOL, PROTOCOL_HASH, render_messages


def test_strict_json_converts_nonfinite_to_null(tmp_path: Path):
    assert canonical_json({"x": float("nan")}) == '{"x":null}'
    path = tmp_path / "x.json"
    write_json(path, {"x": float("inf")})
    assert json.loads(path.read_text()) == {"x": None}


def test_jsonl_stream_write(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, ({"i": index} for index in range(3)))
    assert path.read_text().count("\n") == 3


def test_question_choices_are_immutable(question):
    with pytest.raises(TypeError):
        question.choices["A"] = "changed"


def test_model_validation_and_order():
    selected = select_models(["qwen3-30b-a3b-instruct-2507", "gemma-3-12b"])
    assert [model.id for model in selected] == [
        "qwen3-30b-a3b-instruct-2507",
        "gemma-3-12b",
    ]
    with pytest.raises(ValueError):
        select_models(["gemma-3-12b", "gemma-3-12b"])
    row = MODEL_BY_ID["gemma-3-12b"].to_dict()
    assert row["preferred_backend"] == "huggingface"
    row["max_concurrency"] = 0
    with pytest.raises(ValueError):
        Model.from_dict(row)
    row = MODEL_BY_ID["gemma-3-12b"].to_dict()
    row["preferred_backend"] = "invalid"
    with pytest.raises(ValueError):
        Model.from_dict(row)


def test_qwen3_235b_prefers_openrouter_away_from_deepinfra():
    model = MODEL_BY_ID["qwen3-235b-a22b-instruct-2507"]
    assert model.preferred_backend == "openrouter"
    assert model.openrouter_providers == ("alibaba", "parasail", "novita", "deepinfra")


def test_protocol_is_single_authority_and_uses_plain_question_text(question):
    messages = render_messages(
        question.stem + " & value < 3",
        question.choices,
        "system_before_after",
    )
    assert messages[0]["content"].count(PROTOCOL) == 1
    assert messages[1]["content"].count(PROTOCOL) == 2
    assert question.stem + " & value < 3" in messages[1]["content"]
    assert "Answer choices\nA. " in messages[1]["content"]
    assert "<question>" not in messages[1]["content"]


def test_smoke_profile_matches_model_request_limits():
    import json
    from importlib.resources import files

    from instruction_duplication.models import (
        MODELS,
        SMOKE_PROFILE_ID,
        SMOKE_PROFILE_SCHEMA_VERSION,
    )

    profile = json.loads(
        files("instruction_duplication")
        .joinpath("smoke-profile-2026-08-05.json")
        .read_text(encoding="utf-8")
    )
    assert profile["schema_version"] == SMOKE_PROFILE_SCHEMA_VERSION
    assert profile["profile_id"] == SMOKE_PROFILE_ID
    provenance = profile["provenance"]
    assert provenance["active_protocol_hash"] == PROTOCOL_HASH
    assert provenance["compatibility"] == "provisional_cross_protocol"
    assert provenance["source_protocol_hash"] is None
    assert provenance["source_protocol_identity"] == "not_recorded_by_legacy_workspace"
    assert len(provenance["source_archive_sha256"]) == 64
    for model in MODELS:
        row = profile["models"][model.id]
        assert row["output_tokens"]["max"] == model.smoke_observed_max_output_tokens
        assert row["request_output_tokens"] == model.request_output_tokens
        assert model.request_output_tokens >= 1.25 * model.smoke_observed_max_output_tokens

    mistral = next(model for model in MODELS if model.id == "mistral-large-3-2512")
    assert mistral.request_output_tokens == 2304
    supplemental = profile["supplemental_ceiling_evidence"]
    assert supplemental["mistral-large-3-2512"]["output_tokens_max"] == 1663
    qwen = next(model for model in MODELS if model.id == "qwen3-235b-a22b-instruct-2507")
    assert qwen.output_usd_per_million == pytest.approx(0.62)


def test_protocol_is_simple_and_requests_eight_named_sections():
    assert PROTOCOL.startswith(
        "Use the following eight headings, in order, to answer the question."
    )
    assert "Complete every section." in PROTOCOL
    assert "Do not select an answer before the Provisional answer section." in PROTOCOL
    assert "Plain text" not in PROTOCOL
    assert "Markdown" not in PROTOCOL
    assert "acceptable" not in PROTOCOL
    assert "code fence" not in PROTOCOL.lower()
    assert "markup" not in PROTOCOL.lower()
    assert "punctuation" not in PROTOCOL.lower()
    assert "1. Facts" in PROTOCOL
    assert "4. Best alternative" in PROTOCOL
    assert "6. What would change the answer" in PROTOCOL
    assert "7. Reconsideration" in PROTOCOL
    assert "8. Final answer" in PROTOCOL
    assert "<facts>" not in PROTOCOL
    assert 'option="' not in PROTOCOL
    assert PROTOCOL.rstrip().endswith("State the selected option and its answer text.")


@pytest.mark.parametrize(
    "condition_id",
    ("after", "system_after", "before_after", "system_before_after"),
)
def test_after_protocol_is_not_an_assistant_prefill(question, condition_id):
    user_message = render_messages(question.stem, question.choices, condition_id)[1]["content"]
    assert user_message.endswith("State the selected option and its answer text.")
    assert not user_message.rstrip().endswith("8. Final answer")
