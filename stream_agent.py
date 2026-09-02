#!/usr/bin/env python
"""stream_agent.py — terminal FULL-DUPLEX voice agent, ThinkSpark is the floor controller.

Moshi / PersonaPlex-style behaviour, but your own cascade writes the words:

    mic  ->  ThinkSpark (VAD + endpoint + barge + back-channel referee, 80 ms frames)
         ->  AssemblyAI STT (parallel, not a gate)
         ->  Krutrim LLM  ->  Soniox TTS  ->  your speakers

It listens while it speaks, barges when you cut in, holds when you pause mid-thought,
back-channels while you think, and speculatively starts the LLM before you finish so the
reply lands instantly. ThinkSpark only decides WHEN — your Indic LLM decides WHAT.

    export ASSEMBLYAI_API_KEY=...  KRUTRIM_API_KEY=...  SONIOX_API_KEY=...  HF_TOKEN=...
    python stream_agent.py

Same engine as `python main.py`; this is a focused full-duplex console with a live
floor-state readout. Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics as stats
import time

from rich.console import Console

from agent.config import load_keys

FLAG_STYLE = {
    "LISTEN": "grey42", "HOLD": "steel_blue1", "INCOMPLETE": "khaki1",
    "TURN_END": "bold spring_green2", "BARGE_SOFT": "bold orange1",
    "BARGE_HARD": "bold red1", "CONTINUE": "grey58",
    "PREFETCH_LLM": "bold magenta", "COMMIT_LLM": "bold cyan",
    "CANCEL_LLM": "bold yellow", "SILENCE_BREAK": "blue",
}
KIND_STYLE = {
    "boot": "grey58", "ready": "bold green", "stt": "grey58", "stt-final": "white",
    "tts": "cyan", "TURN_END": "bold spring_green2", "PREFETCH": "bold magenta",
    "COMMIT": "bold cyan", "CANCEL": "bold yellow", "BARGE_HARD": "bold red1",
    "BARGE_SOFT": "bold orange1", "SPOKEN": "blue", "SILENCE_BREAK": "blue",
    "STT_END": "bold spring_green2", "error": "bold red",
}
# flags that are steady state — not worth a line each
QUIET = {"LISTEN", "CONTINUE", "HOLD"}

PERSONA_REFEREE = ("You are a warm, attentive Indic voice agent on a live call. Decide "
                   "when to listen, hold, interrupt, or back-channel — never the words.")
PERSONA_LLM = ("You are a concise, warm voice assistant on a phone call. Reply in ONE or "
               "two short spoken sentences. Match the caller's language "
               "(English, Hindi, or Gujarati). No markdown, no emojis.")


class UI:
    """Colored full-duplex console: flags on the left, actions + floor state inline."""

    def __init__(self, show_raw: bool = False):
        self.c = Console()
        self.show_raw = show_raw
        self.counts: dict[str, int] = {}

    def _ts(self) -> str:
        return time.strftime("%H:%M:%S")

    def frame(self, flag: str, ms: float, raw: bool) -> None:
        if raw:
            self.counts[flag] = self.counts.get(flag, 0) + 1
            if self.show_raw and flag not in QUIET:
                self.c.print(f"[dim]{self._ts()}  raw   {flag:<13} {ms:5.1f}ms[/dim]")
            return
        if flag in QUIET:
            return
        style = FLAG_STYLE.get(flag, "white")
        self.c.print(f"{self._ts()}  [bold]FLAG[/] [{style}]{flag:<13}[/] {ms:5.1f}ms")

    def log(self, kind: str, detail: str = "", style: str | None = None) -> None:
        style = style or KIND_STYLE.get(kind, "white")
        self.c.print(f"{self._ts()}  [{style}]{kind:<13}[/] {detail}")

    def summary(self, frames: int, decode_ms: list[float]) -> None:
        self.c.print()
        if decode_ms:
            p50 = stats.median(decode_ms)
            p95 = sorted(decode_ms)[int(len(decode_ms) * 0.95) - 1] if len(decode_ms) > 1 else p50
            self.c.print(f"[bold]{frames} frames[/]  decode p50 {p50:.1f} · p95 {p95:.1f}ms "
                         f"(budget 80)")
        for flag, n in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            self.c.print(f"  [{FLAG_STYLE.get(flag,'white')}]{flag:<14}[/] {n}")


async def _run(args) -> None:
    from agent.live import LiveAgent

    ui = UI(show_raw=args.raw)
    keys = load_keys()
    agent = LiveAgent(keys, ui, device=args.device, window=args.window,
                      denoise=not args.no_denoise)

    # personas: referee floor-control prompt + LLM answer prompt
    try:
        agent.floor.referee._referee.system_prompt = args.persona_referee
    except Exception:
        pass
    try:
        agent.floor.llm.system = args.persona_llm
    except Exception:
        pass

    ui.log("ready", "full-duplex — speak; it listens while it talks. Ctrl+C to stop.",
           style="bold green")
    try:
        await agent.run()
    except KeyboardInterrupt:
        pass
    finally:
        ui.summary(agent.frames, agent.decode_ms)


def main() -> None:
    ap = argparse.ArgumentParser(description="ThinkSpark full-duplex terminal voice agent")
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--window", type=int, default=3,
                    help="smoothing window in frames (3 = 240 ms, the eval collar)")
    ap.add_argument("--raw", action="store_true", help="also print pre-smoothing flags")
    ap.add_argument("--no-denoise", action="store_true", help="disable RNNoise")
    ap.add_argument("--persona-referee", default=PERSONA_REFEREE,
                    help="floor-controller system prompt (ThinkSpark)")
    ap.add_argument("--persona-llm", default=PERSONA_LLM,
                    help="answer system prompt (LLM)")
    args = ap.parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
