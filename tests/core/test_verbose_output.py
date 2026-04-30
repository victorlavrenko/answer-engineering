from __future__ import annotations

from dataclasses import dataclass

from _pytest.capture import CaptureFixture

from answer_engineering.engine.telemetry.events.event_sink import (
    ConsoleRuntimeEventSink,
    DebugEventEmitter,
)
from answer_engineering.infra.console.reactive_visible_printer import (
    ReactiveVisiblePrinter,
)
from answer_engineering.infra.console.visible_layout import VisibleTextLayouter


@dataclass
class _CapturingEmitter:
    chunks: list[str]

    def emit(self, text: str) -> None:
        self.chunks.append(text)


def test_verbose_output_starts_on_new_line_with_visible_renderer() -> None:
    chunks: list[str] = []
    printer = ReactiveVisiblePrinter(
        emitter=_CapturingEmitter(chunks),
        layouter=VisibleTextLayouter(wrap_width=80),
        retractable_tail_chars=0,
    )
    printer.reset()
    printer.observe_visible_text("loss")
    sink = ConsoleRuntimeEventSink.with_debug_line_emitter(printer)
    emitter = DebugEventEmitter(event_sink=sink)

    emitter.emit("rule debug")

    out = "".join(chunks)
    assert "\n[AE] rule debug" in out
    assert "loss[AE]" not in out


def test_verbose_output_without_visible_renderer_still_prints_debug(
    capsys: CaptureFixture[str],
) -> None:
    emitter = DebugEventEmitter(event_sink=ConsoleRuntimeEventSink())

    emitter.emit("rule debug")

    out = capsys.readouterr().out
    assert "[AE] rule debug" in out
