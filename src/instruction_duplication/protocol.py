"""Experimental conditions and the single authoritative protocol text.

Paper mapping: implements the intervention and the 2x2x2 placement design described
in ``Experiment -> Models, Questions, and Factorial Conditions`` and the eight-role
instruction described in ``Experiment -> Procedural Instruction``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

from .io_utils import sha256_json
from .json_types import JsonObject


class ChatMessage(TypedDict):
    """One OpenAI-compatible chat message."""

    role: str
    content: str


SECTION_GUIDANCE: dict[str, str] = {
    "facts": (
        "Discuss every important fact in the question, including relevant timing, laterality, "
        "negation, measurements, and qualifiers."
    ),
    "implications": (
        "Explain what the facts support or argue against and how they distinguish the answer "
        "choices."
    ),
    "provisional_answer": "Select the best answer and explain why.",
    "second_best": (
        "Select a different answer as the best alternative and explain why it is less suitable."
    ),
    "decisive_fact": (
        "Identify the fact that best distinguishes the provisional answer from the best "
        "alternative."
    ),
    "answer_changing_change": (
        "Describe the smallest change to the question that would make the best alternative the "
        "best answer, and explain why."
    ),
    "rereasoning": (
        "Reconsider the provisional answer using the original facts. State whether you retain "
        "or revise it, and explain why."
    ),
    "final_answer": "State the selected option and its answer text.",
}

ROLE_TITLES: dict[str, str] = {
    "facts": "Facts",
    "implications": "Implications",
    "provisional_answer": "Provisional answer",
    "second_best": "Best alternative",
    "decisive_fact": "Decisive distinction",
    "answer_changing_change": "What would change the answer",
    "rereasoning": "Reconsideration",
    "final_answer": "Final answer",
}

PROTOCOL = "\n".join(
    (
        "Use the following eight headings, in order, to answer the question. Complete every "
        "section. Do not select an answer before the Provisional answer section.",
        "",
        *(
            f"{index}. {ROLE_TITLES[tag]} — {SECTION_GUIDANCE[tag]}"
            for index, tag in enumerate(SECTION_GUIDANCE, start=1)
        ),
    )
)

NEUTRAL_SYSTEM = (
    "Answer the multiple-choice question accurately and follow the user's instructions."
)
BASELINE_REQUEST = "Reason through the question and give your final answer."


@dataclass(frozen=True, slots=True)
class Condition:
    """One factorial instruction-placement condition."""

    id: str
    system: bool
    before: bool
    after: bool

    @property
    def copies(self) -> int:
        """Return the number of protocol copies in this condition."""
        return int(self.system) + int(self.before) + int(self.after)

    def to_dict(self) -> JsonObject:
        """Serialize the condition."""
        return {
            "id": self.id,
            "system": self.system,
            "before": self.before,
            "after": self.after,
        }


CONDITIONS: tuple[Condition, ...] = (
    Condition("zero", False, False, False),
    Condition("system", True, False, False),
    Condition("before", False, True, False),
    Condition("after", False, False, True),
    Condition("system_before", True, True, False),
    Condition("system_after", True, False, True),
    Condition("before_after", False, True, True),
    Condition("system_before_after", True, True, True),
)
CONDITION_BY_ID = {condition.id: condition for condition in CONDITIONS}
PROTOCOL_HASH = sha256_json({"protocol": PROTOCOL, "conditions": [c.to_dict() for c in CONDITIONS]})


def format_question(stem: str, choices: Mapping[str, str]) -> str:
    """Render a multiple-choice question in a deterministic plain-text wrapper."""
    lines = ["Question", stem.strip(), "", "Answer choices"]
    for label, text in choices.items():
        lines.append(f"{label}. {text.strip()}")
    return "\n".join(lines)


def render_messages(
    stem: str,
    choices: Mapping[str, str],
    condition_id: str,
) -> list[ChatMessage]:
    """Render chat messages for one experiment cell."""
    try:
        condition = CONDITION_BY_ID[condition_id]
    except KeyError as exc:
        raise ValueError(f"unknown condition: {condition_id}") from exc

    system = NEUTRAL_SYSTEM + ("\n\n" + PROTOCOL if condition.system else "")
    user_parts: list[str] = []
    if condition.before:
        user_parts.append(PROTOCOL)
    user_parts.append(format_question(stem, choices))
    if condition.after:
        user_parts.append(PROTOCOL)
    if condition.copies == 0:
        user_parts.append(BASELINE_REQUEST)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
