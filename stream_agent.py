#!/usr/bin/env python
"""stream_agent.py — terminal FULL-DUPLEX voice agent, ThinkSpark is the floor controller.

Moshi / PersonaPlex-style behaviour, your own cascade writes the words:

    MIC  ->  ThinkSpark (VAD + endpoint + barge + back-channel referee, 80 ms frames)
         ->  AssemblyAI STT (parallel, not a gate)
         ->  Krutrim LLM  ->  Soniox TTS  ->  your speakers

It listens while it speaks, barges when you cut in, holds when you pause mid-thought,
back-channels while you think, and speculatively starts the LLM before you finish so the
reply lands instantly. ThinkSpark decides WHEN; your Indic LLM decides WHAT.

    export ASSEMBLYAI_API_KEY=...  KRUTRIM_API_KEY=...  SONIOX_API_KEY=...  HF_TOKEN=...
    python stream_agent.py                # auto device (cuda > mps > cpu)
    python stream_agent.py --device mps   # Apple GPU (Metal) on a Mac
    python stream_agent.py --raw          # also show every 80 ms MIC->TS frame

SMART FLAG MANAGEMENT (agent/smoothing.py) — every ThinkSpark flag goes through it:
  * 3-frame majority vote (240 ms, the eval collar) — one noisy frame never acts.
  * event latch + per-flag cooldown — PREFETCH_LLM / TURN_END / BARGE fire ONCE per real
    event and stay quiet on the flag-flicker that follows, until the decision truly
    changes. (Raw stream_infer.py has NO smoothing — that is why it showed 53 in a row.)
  * policy guards on top — no 2nd LLM while one is speculating, no commit while a turn is
    already running, barge only while TTS is actually speaking.

Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics as stats
import time

from rich.console import Console

from agent.config import load_keys

# raw per-frame flag colours (the MIC -> ThinkSpark firehose)
FLAG_STYLE = {
    "LISTEN": "grey42", "HOLD": "steel_blue1", "INCOMPLETE": "khaki1",
    "TURN_END": "bold spring_green2", "BARGE_SOFT": "bold orange1",
    "BARGE_HARD": "bold red1", "CONTINUE": "grey58",
    "PREFETCH_LLM": "bold magenta", "COMMIT_LLM": "bold cyan",
    "CANCEL_LLM": "bold yellow", "SILENCE_BREAK": "blue",
}
QUIET = {"LISTEN", "CONTINUE", "HOLD"}       # steady states — not worth a line each

# action / event kind  ->  (stage tag shown, colour)
STAGE = {
    "boot": ("· BOOT ", "grey58"), "ready": ("· READY", "bold green"),
    "stt": ("STT   ", "grey58"), "stt-final": ("STT ✓ ", "white"),
    "tts": ("TTS ◂ ", "cyan"),
    "PREFETCH": ("LLM ⟳ ", "bold magenta"), "COMMIT": ("LLM ✓ ", "bold cyan"),
    "CANCEL": ("LLM ✗ ", "bold yellow"),
    "TURN_END": ("TURN ▸", "bold spring_green2"),
    "STT_END": ("TURN ▸", "bold spring_green2"),
    "BARGE_HARD": ("BARGE!", "bold red1"), "BARGE_SOFT": ("DUCK  ", "bold orange1"),
    "SPOKEN": ("BACKCH", "blue"), "SILENCE_BREAK": ("REOPEN", "blue"),
    "error": ("ERROR ", "bold red"),
}

PERSONA_REFEREE = ("You are a warm, attentive Indic voice agent on a live call. Decide "
                   "when to listen, hold, interrupt, or back-channel — never the words.")
PERSONA_LLM = ("You are a concise, warm voice assistant on a phone call. Reply in ONE or "
               "two short spoken sentences. Match the caller's language "
               "(English, Hindi, or Gujarati). No markdown, no emojis.")


class UI:
    """Colored full-duplex pipeline view: MIC -> TS -> STT -> LLM -> TTS, in order."""

    def __init__(self, show_raw: bool = False):
        self.c = Console()
        self.show_raw = show_raw
        self.raw_counts: dict[str, int] = {}     # raw per-frame flags
        self.act_counts: dict[str, int] = {}     # smoothed decisions that acted
        self.mic_frames = 0

    def _ts(self) -> str:
        return time.strftime("%H:%M:%S")

    def frame(self, flag: str, ms: float, raw: bool) -> None:
        if raw:
            self.mic_frames += 1
            self.raw_counts[flag] = self.raw_counts.get(flag, 0) + 1
            if self.show_raw:
                st = FLAG_STYLE.get(flag, "white")
                self.c.print(f"[grey30]{self._ts()}[/] [grey30]MIC▸TS[/] "
                             f"[{st}]{flag:<13}[/] [grey30]{ms:5.1f}ms[/]")
            return
        # a SMOOTHED decision survived the vote+latch+cooldown — this is what ACTS
        self.act_counts[flag] = self.act_counts.get(flag, 0) + 1
        if flag in QUIET:
            return
        st = FLAG_STYLE.get(flag, "white")
        self.c.print(f"{self._ts()}  [bold]TS ▸[/]  [{st}]{flag:<13}[/] {ms:5.1f}ms  "
                     f"[grey42](smoothed decision)[/]")

    def log(self, kind: str, detail: str = "", style: str | None = None) -> None:
        tag, st = STAGE.get(kind, (f"{kind:<6}", style or "white"))
        self.c.print(f"{self._ts()}  [{st}]{tag}[/] {detail}")

    def summary(self, frames: int, decode_ms: list[float]) -> None:
        self.c.print()
        if decode_ms:
            p50 = stats.median(decode_ms)
            p95 = sorted(decode_ms)[int(len(decode_ms) * 0.95) - 1] if len(decode_ms) > 1 else p50
            self.c.print(f"[bold]{frames} MIC frames[/] · decode p50 {p50:.1f} · "
                         f"p95 {p95:.1f}ms (80 ms budget)")
        self.c.print("[bold]raw flags (MIC▸TS, per 80 ms):[/]")
        for flag, n in sorted(self.raw_counts.items(), key=lambda kv: -kv[1]):
            self.c.print(f"  [{FLAG_STYLE.get(flag,'white')}]{flag:<14}[/] {n}")
        if self.act_counts:
            self.c.print("[bold]smoothed decisions (what actually acted):[/]")
            for flag, n in sorted(self.act_counts.items(), key=lambda kv: -kv[1]):
                self.c.print(f"  [{FLAG_STYLE.get(flag,'white')}]{flag:<14}[/] {n}")


async def _run(args) -> None:
    from agent.live import LiveAgent

    ui = UI(show_raw=args.raw)
    keys = load_keys()
    agent = LiveAgent(keys, ui, device=args.device, window=args.window,
                      denoise=not args.no_denoise)

    try:
        agent.floor.referee._referee.system_prompt = args.persona_referee
    except Exception:
        pass
    try:
        agent.floor.llm.system = args.persona_llm
    except Exception:
        pass

    dev = getattr(agent.referee, "device", "?")
    ui.c.print(f"\n[bold]FLOW[/]  MIC(80ms) ▸ [bold]ThinkSpark[/]({dev}) ▸ smoothing ▸ "
               f"policy ▸ STT‖LLM‖TTS ▸ speakers")
    ui.c.print("[grey42]tags: MIC▸TS raw frame · TS▸ smoothed decision · STT · "
               "LLM⟳prefetch/✓commit/✗cancel · TURN▸ · BARGE!/DUCK · BACKCH/REOPEN · TTS◂[/]")
    ui.log("ready", "full-duplex — speak; it listens while it talks. Ctrl+C to stop.")
    try:
        await agent.run()
    except KeyboardInterrupt:
        pass
    finally:
        ui.summary(agent.frames, agent.decode_ms)


def main() -> None:
    ap = argparse.ArgumentParser(description="ThinkSpark full-duplex terminal voice agent")
    ap.add_argument("--device", default="auto", help="cuda | mps (Apple GPU) | cpu | auto")
    ap.add_argument("--window", type=int, default=3,
                    help="smoothing window in frames (3 = 240 ms, the eval collar)")
    ap.add_argument("--raw", action="store_true",
                    help="also print every 80 ms MIC▸TS frame (the full firehose)")
    ap.add_argument("--no-denoise", action="store_true", help="disable RNNoise")
    ap.add_argument("--persona-referee", default=PERSONA_REFEREE,
                    help="floor-controller system prompt (ThinkSpark)")
    ap.add_argument("--persona-llm", default=PERSONA_LLM, help="answer system prompt (LLM)")
    args = ap.parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
