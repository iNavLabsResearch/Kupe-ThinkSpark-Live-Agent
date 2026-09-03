#!/usr/bin/env python
"""stream_agent_radio.py — the FULL agent, but the 'caller' is a live RADIO stream.

Identical stack to stream_agent.py (ThinkSpark floor control + AssemblyAI STT + Krutrim
LLM + Soniox TTS), except the audio comes from a radio URL via ffmpeg instead of your
microphone. This lets you test the whole agent on clean, in-domain-ish broadcast speech
without a mic. The agent's TTS plays out your speakers so you HEAR the replies, and the
terminal shows the whole flow — radio▸ThinkSpark flags, STT, LLM, TTS.

    export ASSEMBLYAI_API_KEY=...  KRUTRIM_API_KEY=...  SONIOX_API_KEY=...  HF_TOKEN=...
    python stream_agent_radio.py --raw                 # NPR live, full agent
    python stream_agent_radio.py --url "<hls/mp3>" --device mps

Needs ffmpeg on PATH. Ctrl+C to stop.
"""
from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import asyncio
import shutil
import subprocess

import numpy as np

from agent.config import load_keys
from agent.live import LiveAgent
from stream_agent import PERSONA_LLM, PERSONA_REFEREE, UI

DEFAULT_URL = "https://npr-ice.streamguys1.com/live.mp3"
SR = 24_000
FRAME = SR * 80 // 1000            # 1920 samples / 80 ms


class RadioLiveAgent(LiveAgent):
    """LiveAgent with the microphone replaced by an ffmpeg radio reader."""

    def __init__(self, *a, url: str = DEFAULT_URL, **kw):
        super().__init__(*a, **kw)
        self._url = url

    def _mic_thread(self) -> None:  # overrides the sounddevice mic
        if not shutil.which("ffmpeg"):
            self.ui.log("error", "ffmpeg not found — apt-get install -y ffmpeg")
            return
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-reconnect", "1", "-reconnect_streamed", "1",
               "-reconnect_delay_max", "5", "-user_agent", "Mozilla/5.0",
               "-i", self._url, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
        self.ui.log("boot", f"radio ◂ {self._url}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=FRAME * 4)
        nbytes = FRAME * 4
        got = 0
        try:
            while not self._shutdown.is_set():
                buf = proc.stdout.read(nbytes)
                if not buf or len(buf) < nbytes:
                    break
                got += 1
                frame = np.frombuffer(buf, dtype=np.float32).copy()
                self.floor.push_audio(frame, SR)   # live stream -> real-time paced
        finally:
            if got == 0:
                self.ui.log("error", "radio delivered 0 frames — bad/blocked URL "
                            "(try --url with a reachable stream)")
            proc.kill()


async def _run(args) -> None:
    ui = UI(show_raw=args.raw)
    keys = load_keys()
    agent = RadioLiveAgent(keys, ui, device=args.device, window=args.window,
                           denoise=not args.no_denoise, url=args.url)

    # radio is a monologue: ThinkSpark TURN_END fires a lot, so keep the STT endpoint on
    # (dual trigger) for clean turns unless --pure-thinkspark is set.
    agent.floor._use_stt_endpoint = not args.pure_thinkspark
    try:
        agent.floor.referee._referee.system_prompt = args.persona_referee
    except Exception:
        pass
    try:
        agent.floor.llm.system = args.persona_llm
    except Exception:
        pass

    dev = getattr(agent.referee, "device", "?")
    ui.c.print(f"\n[bold]FLOW[/]  RADIO(80ms) ▸ [bold]ThinkSpark[/]({dev}) ▸ smoothing ▸ "
               f"policy ▸ STT‖LLM‖TTS ▸ speakers")
    ui.c.print("[grey42]tags: MIC▸TS = radio▸ThinkSpark frame · TS▸ decision · STT · "
               "LLM⟳/✓/✗ · TURN▸ · BARGE!/DUCK · BACKCH/REOPEN · TTS◂ (plays on speakers)[/]")
    endp = "STT+ThinkSpark" if agent.floor._use_stt_endpoint else "ThinkSpark only"
    ui.log("ready", f"radio agent live — endpoint: {endp}. Ctrl+C to stop.")
    try:
        await agent.run()
    except KeyboardInterrupt:
        pass
    finally:
        ui.summary(agent.frames, agent.decode_ms)


def main() -> None:
    ap = argparse.ArgumentParser(description="ThinkSpark full agent driven by a radio stream")
    ap.add_argument("--url", default=DEFAULT_URL, help="radio HLS/MP3 URL (default: NPR live)")
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--window", type=int, default=3, help="smoothing window (3 = 240 ms)")
    ap.add_argument("--raw", action="store_true", help="print every 80 ms radio▸TS frame")
    ap.add_argument("--no-denoise", action="store_true", help="disable RNNoise")
    ap.add_argument("--pure-thinkspark", action="store_true",
                    help="commit turns ONLY on ThinkSpark TURN_END (ignore STT endpoint)")
    ap.add_argument("--persona-referee", default=PERSONA_REFEREE)
    ap.add_argument("--persona-llm", default=PERSONA_LLM)
    args = ap.parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
