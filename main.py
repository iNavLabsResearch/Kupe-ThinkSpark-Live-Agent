#!/usr/bin/env python
"""Kupe ThinkSpark Live Agent — terminal voice agent driven by ThinkSpark decisions.

    pip install -r requirements.txt
    export SUPABASE_DB_URL=postgres://...        # or the three *_API_KEY vars
    python main.py

Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics as stats
import time

from rich.console import Console

from agent.config import load_keys

FLAG_STYLE = {
    "LISTEN": "grey42",
    "HOLD": "steel_blue1",
    "INCOMPLETE": "khaki1",
    "TURN_END": "bold spring_green2",
    "BARGE_SOFT": "bold orange1",
    "BARGE_HARD": "bold red1",
    "CONTINUE": "grey58",
    "PREFETCH_LLM": "bold magenta",
    "COMMIT_LLM": "bold cyan",
    "CANCEL_LLM": "bold yellow",
    "SILENCE_BREAK": "blue",
}

KIND_STYLE = {
    "boot": "grey58", "ready": "bold green", "stt": "grey58",
    "stt-final": "white", "tts": "cyan",
}


class UI:
    """Flat scrolling log — every decision and every action, in order."""

    def __init__(self, show_raw: bool = False):
        self.c = Console()
        self.show_raw = show_raw
        self.counts: dict[str, int] = {}
        self.suppressed = 0

    def _ts(self) -> str:
        return time.strftime("%H:%M:%S")

    def frame(self, flag: str, ms: float, raw: bool) -> None:
        self.counts[flag] = self.counts.get(flag, 0) + 1
        if raw:
            if self.show_raw:
                self.c.print(f"[dim]{self._ts()}  raw    {flag:<14} {ms:6.1f} ms[/dim]")
            return
        if flag in ("LISTEN", "CONTINUE"):
            return  # steady state, not worth a line
        style = FLAG_STYLE.get(flag, "white")
        self.c.print(f"{self._ts()}  [bold]FLAG[/bold]   "
                     f"[{style}]{flag:<14}[/{style}] {ms:6.1f} ms")

    def log(self, kind: str, detail: str, style: str | None = None) -> None:
        style = style or KIND_STYLE.get(kind, "white")
        self.c.print(f"{self._ts()}  [{style}]{kind:<14}[/{style}] {detail}")

    def summary(self, frames: int, decode_ms: list[float]) -> None:
        self.c.print()
        if decode_ms:
            p50 = stats.median(decode_ms)
            p95 = sorted(decode_ms)[int(len(decode_ms) * 0.95) - 1] if len(decode_ms) > 1 else p50
            self.c.print(f"[bold]{frames} frames[/bold]  "
                         f"decode p50 {p50:.1f} ms · p95 {p95:.1f} ms · "
                         f"budget 80 ms/frame")
        for flag, n in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            style = FLAG_STYLE.get(flag, "white")
            self.c.print(f"  [{style}]{flag:<16}[/{style}] {n}")


async def _run(args) -> None:
    from agent.live import LiveAgent

    ui = UI(show_raw=args.raw)
    keys = load_keys()
    agent = LiveAgent(keys, ui, device=args.device, window=args.window,
                      denoise=not args.no_denoise)
    try:
        await agent.run()
    except KeyboardInterrupt:
        pass
    finally:
        ui.summary(agent.frames, agent.decode_ms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--window", type=int, default=3,
                    help="smoothing window in frames (3 = 240 ms, matches the eval collar)")
    ap.add_argument("--raw", action="store_true", help="also print pre-smoothing flags")
    ap.add_argument("--no-denoise", action="store_true", help="disable RNNoise")
    args = ap.parse_args()

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
