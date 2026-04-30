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


def main():
    print("Starting...")

    emitter = StdoutCapturingEmitter(parts=[])
    printer = ReactiveVisiblePrinter(
        emitter=emitter,
        layouter=VisibleTextLayouter(wrap_width=80),
        retractable_tail_chars=50,
    )
    printer.reset()

    with open("../tmp/o.o.o") as file:
        for line in file:
            printer.observe_visible_text(line.rstrip("\n").replace("\\n", "\n"))

    printer.flush()


if __name__ == "__main__":
    main()
