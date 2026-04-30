import builtins
from dataclasses import dataclass

from answer_engineering.infra.console.reactive_visible_printer import (
    ReactiveVisiblePrinter,
)
from answer_engineering.infra.console.visible_layout import VisibleTextLayouter

# main function

SOURCE = (
    "The patient presents with sudden onset hearing loss in the right "
    "ear, which is a concerning symptom. The otoscopic examination is "
    "normal, which rules out any obvious external or middle ear pathology. "
    "The tuning fork testing suggests that the hearing loss is conduct"
)


@dataclass(slots=True)
class StdoutCapturingEmitter:
    parts: list[str]

    def emit(self, text: str) -> None:
        self.parts.append(text)
        builtins.print(text, end="", flush=True)


def main():
    words = SOURCE.split(" ")

    emitter = StdoutCapturingEmitter(parts=[])
    printer = ReactiveVisiblePrinter(
        emitter=emitter,
        layouter=VisibleTextLayouter(wrap_width=80),
        retractable_tail_chars=80,
    )
    printer.reset()

    current = ""
    for word in words:
        current += word + " "
        printer.observe_visible_text(current)

    printer.flush()


if __name__ == "__main__":
    main()
