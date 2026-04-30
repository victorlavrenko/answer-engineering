# ruff: noqa: E501

from __future__ import annotations

import builtins
from dataclasses import dataclass

from answer_engineering.infra.console.reactive_visible_printer import (
    ReactiveVisiblePrinter,
)
from answer_engineering.infra.console.visible_layout import VisibleTextLayouter


@dataclass(slots=True)
class StdoutCapturingEmitter:
    parts: list[str]

    def emit(self, text: str) -> None:
        self.parts.append(text)
        builtins.print(text, end="", flush=True)


VISIBLE_TEXTS = [
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. The tuning fork testing suggests a conductive",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. The tuning fork testing suggests a conductive hearing",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. The tuning fork testing suggests that the hearing loss is conductive",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. However, the tuning fork testing suggests a conductive",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. Based on the tuning fork testing, there is a conductive",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. However, the tuning fork testing suggests that the hearing loss is conductive",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. \n\nTuning fork testing suggests that the patient has sensorineural",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. \n\nThe tuning fork testing suggests a conductive hearing",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. \n\nThe tuning fork testing suggests a sensorineural",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. \n\nThe tuning fork testing suggests that the patient has sensorineural",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. \n\nBased on the tuning fork testing, the patient has a conductive",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. \n\nBased on the tuning fork testing results, the patient has a conductive",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. \n\nBased on the tuning fork testing, there is a conductive",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. \n\nIn this case, the tuning fork testing suggests a conductive",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. \n\nIn this scenario, the tuning fork testing suggests a conductive",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. \n\nIn this case, the tuning fork test suggests a conductive",
]


def test_diagnostic_console_printer_replay_like_generation() -> None:
    emitter = StdoutCapturingEmitter(parts=[])
    printer = ReactiveVisiblePrinter(
        emitter=emitter,
        layouter=VisibleTextLayouter(wrap_width=80),
        retractable_tail_chars=50,
    )
    printer.reset()

    current = ""
    printer.observe_visible_text(current)

    for target in VISIBLE_TEXTS:
        current = _feed_transition_like_generation(
            printer=printer,
            current=current,
            target=target,
        )

    printer.observe_visible_text(current, is_final=True)
    diagnostic_output = "".join(emitter.parts)
    assert "\nvious external" not in diagnostic_output
    assert "\nology." not in diagnostic_output
    assert "\ngy." not in diagnostic_output
    assert "\ny." not in diagnostic_output


def _feed_transition_like_generation(
    *,
    printer: ReactiveVisiblePrinter,
    current: str,
    target: str,
) -> str:
    common_prefix_len = _common_prefix_len(current, target)

    if common_prefix_len < len(current):
        current = current[:common_prefix_len]
        printer.observe_visible_text(current)

    while len(current) < len(target):
        next_len = _next_generation_len(target, len(current))
        current = target[:next_len]
        printer.observe_visible_text(current)

    return current


def _next_generation_len(text: str, current_len: int) -> int:
    for index in range(current_len + 1, len(text) + 1):
        if text[index - 1].isspace():
            return index
    return len(text)


def _common_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit
