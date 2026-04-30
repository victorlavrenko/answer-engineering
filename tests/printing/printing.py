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
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external or middle ear pathology. In this case, the tuning fork test suggests a conductive",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external ear pathology. In this case, the tuning fork test suggests a conductive",
    "The patient presents with sudden onset hearing loss in the right ear, which is a concerning symptom. The otoscopic examination is normal, which rules out any obvious external ear pathology. In this case, the tuning fork test doesn't suggests a conductive",
    "The patient presents with sudden onset hearing loss.",
    "The patient presents with sudden onset hearing loss. Now what",
]


def _run_case(*, retractable_tail_chars: int) -> str:
    emitter = StdoutCapturingEmitter(parts=[])
    printer = ReactiveVisiblePrinter(
        emitter=emitter,
        layouter=VisibleTextLayouter(wrap_width=80),
        retractable_tail_chars=retractable_tail_chars,
    )
    printer.reset()

    current = ""
    printer.observe_visible_text(current)

    for target in VISIBLE_TEXTS:
        while len(current) < len(target):
            next_len = min(len(target), len(current) + 7)
            current = target[:next_len]
            printer.observe_visible_text(current)

        if current != target:
            current = target
            printer.observe_visible_text(current)

    printer.observe_visible_text(current, is_final=True)
    return "".join(emitter.parts)


if __name__ == "__main__":
    _run_case(retractable_tail_chars=80)
