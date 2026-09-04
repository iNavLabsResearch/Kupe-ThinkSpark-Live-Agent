"""Two-column live TUI: RIGHT = ThinkSpark per-frame flags (the 80 ms fountain, runs even
on silence); LEFT = the agent's STT / LLM / TTS / actions. Drop-in for the plain UI."""
from __future__ import annotations

import statistics as stats
import time
from collections import deque

FLAG_STYLE = {
    "LISTEN": "grey42", "HOLD": "steel_blue1", "INCOMPLETE": "khaki1",
    "TURN_END": "bold spring_green2", "BARGE_SOFT": "bold orange1",
    "BARGE_HARD": "bold red1", "CONTINUE": "grey58",
    "PREFETCH_LLM": "bold magenta", "COMMIT_LLM": "bold cyan",
    "CANCEL_LLM": "bold yellow", "SILENCE_BREAK": "blue",
}
STAGE = {
    "boot": ("· ", "grey58"), "ready": ("· ", "bold green"),
    "stt": ("STT   ", "grey62"), "stt-final": ("STT ✓ ", "white"),
    "tts": ("TTS ◂ ", "bold cyan"),
    "PREFETCH": ("LLM ⟳ ", "bold magenta"), "COMMIT": ("LLM ✓ ", "bold cyan"),
    "CANCEL": ("LLM ✗ ", "bold yellow"),
    "TURN_END": ("TURN ▸", "bold spring_green2"),
    "STT_END": ("TURN ▸", "bold spring_green2"),
    "BARGE_HARD": ("BARGE!", "bold red1"), "BARGE_SOFT": ("DUCK  ", "bold orange1"),
    "SPOKEN": ("BACKCH", "blue"), "SILENCE_BREAK": ("REOPEN", "blue"),
    "error": ("ERROR ", "bold red"),
}
QUIET = {"LISTEN", "CONTINUE", "HOLD"}


class SplitUI:
    def __init__(self, show_raw: bool = True, active_fn=None):
        from rich.console import Console
        from rich.layout import Layout
        from rich.live import Live

        # optional callable -> True while the model is hearing REAL (faked-user) audio,
        # used to colour those frames vs the silence fountain.
        self._active_fn = active_fn
        self.c = Console()
        rows = max(8, self.c.size.height - 4)
        self.left: deque[str] = deque(maxlen=rows)
        self.right: deque[str] = deque(maxlen=rows)
        self.raw_counts: dict[str, int] = {}
        self.act_counts: dict[str, int] = {}
        self.frames = 0
        self._layout = Layout()
        self._layout.split_row(Layout(name="left", ratio=1), Layout(name="right", ratio=1))
        self.live = Live(self._render(), console=self.c, refresh_per_second=10, screen=True)
        self.live.start()

    def _panel(self, lines, title, border):
        from rich.panel import Panel
        from rich.text import Text
        body = Text.from_markup("\n".join(lines)) if lines else Text("")
        return Panel(body, title=title, border_style=border, padding=(0, 1))

    def _render(self):
        self._layout["left"].update(self._panel(self.left, "agent · STT / LLM / TTS", "cyan"))
        self._layout["right"].update(
            self._panel(self.right, "ThinkSpark · every 80 ms (LISTEN=silence)", "magenta"))
        return self._layout

    def _push(self):
        try:
            self.live.update(self._render())
        except Exception:
            pass

    def frame(self, flag: str, ms: float, raw: bool) -> None:
        ts = time.strftime("%H:%M:%S")
        if raw:
            self.frames += 1
            self.raw_counts[flag] = self.raw_counts.get(flag, 0) + 1
            st = FLAG_STYLE.get(flag, "white")
            active = bool(self._active_fn and self._active_fn())
            if active:
                # the model is hearing the faked Soniox user right now — highlight it
                self.right.append(
                    f"[black on yellow] {ts} ▶ HEARING [/] [bold {st}]{flag:<13}[/] "
                    f"[grey58]{ms:5.1f}ms[/]")
            else:
                self.right.append(
                    f"[grey42]{ts}[/] [{st}]{flag:<13}[/] [grey35]{ms:5.1f}ms[/]  [grey30]· silence[/]")
            self._push()
        else:
            self.act_counts[flag] = self.act_counts.get(flag, 0) + 1
            if flag not in QUIET:
                st = FLAG_STYLE.get(flag, "white")
                self.left.append(f"[grey42]{ts}[/] [bold]TS ▸[/] [{st}]{flag}[/]")
                self._push()

    def log(self, kind: str, detail: str = "", style: str | None = None) -> None:
        from rich.markup import escape
        tag, st = STAGE.get(kind, (f"{kind[:6]:<6}", style or "white"))
        self.left.append(f"[grey42]{time.strftime('%H:%M:%S')}[/] [{st}]{tag}[/] "
                         f"{escape(str(detail))}")
        self._push()

    def summary(self, frames: int, decode_ms: list[float]) -> None:
        try:
            self.live.stop()
        except Exception:
            pass
        if decode_ms:
            p50 = stats.median(decode_ms)
            p95 = sorted(decode_ms)[int(len(decode_ms) * 0.95) - 1] if len(decode_ms) > 1 else p50
            self.c.print(f"[bold]{frames} frames[/] · decode p50 {p50:.1f} · p95 {p95:.1f}ms (budget 80)")
        for flag, n in sorted(self.raw_counts.items(), key=lambda kv: -kv[1]):
            self.c.print(f"  [{FLAG_STYLE.get(flag,'white')}]{flag:<14}[/] {n}")
