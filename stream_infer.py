#!/usr/bin/env python
"""ThinkSpark inference ONLY — pull a live radio stream, print per-frame decisions.

No STT/LLM/TTS, no floor-control actions. Just the model watching live audio at 80 ms
frames and printing what it decides, in colour.

    export HF_TOKEN=...                 # if the checkpoint repo is private
    python stream_infer.py             # BBC World Service live
    python stream_infer.py --url <hls-or-audio-url> --device cuda

Needs ffmpeg on PATH (decodes any HLS/stream to raw PCM).
"""
from __future__ import annotations

import argparse
import shutil
import statistics as stats
import subprocess
import time
from collections import Counter

import numpy as np
from rich.console import Console

from agent import config

# BBC World Service live HLS (non-UK). Override with --url.
DEFAULT_URL = ("https://a.files.bbci.co.uk/media/live/manifesto/audio/simulcast/hls/"
               "nonuk/sbr_high/ak/bbc_world_service.m3u8")

SR = 24_000
FRAME = SR * 80 // 1000            # 1920 samples / 80 ms

STYLE = {
    "LISTEN": "grey42", "HOLD": "steel_blue1", "INCOMPLETE": "khaki1",
    "TURN_END": "bold spring_green2", "BARGE_SOFT": "bold orange1",
    "BARGE_HARD": "bold red1", "CONTINUE": "grey58",
    "PREFETCH_LLM": "bold magenta", "COMMIT_LLM": "bold cyan",
    "CANCEL_LLM": "bold yellow", "SILENCE_BREAK": "blue",
}
STEADY = {"LISTEN", "CONTINUE"}


def ffmpeg_frames(url: str, c: Console):
    if not shutil.which("ffmpeg"):
        c.print("[bold red]ffmpeg not found[/] — install it: apt-get install -y ffmpeg")
        raise SystemExit(1)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", url,
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SR), "-"]
    c.print(f"[grey58]ffmpeg <- {url}[/]")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=FRAME * 4)
    nbytes = FRAME * 4
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                break
            yield np.frombuffer(buf, dtype=np.float32).copy()
    finally:
        proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--agent-state", default="IDLE",
                    help="IDLE|LLM_GEN|TTS_SPEAKING|TTS_DONE fed to the model each frame")
    args = ap.parse_args()

    c = Console()
    from kupe import ThinkSpark

    c.print(f"[grey58]loading[/] {config.TS_REPO} :: {config.TS_SUBFOLDER}")
    ts = ThinkSpark(config.TS_REPO, device=args.device, subfolder=config.TS_SUBFOLDER)
    c.print(f"[bold green]ThinkSpark ready[/] on [bold]{ts.device}[/]  "
            f"dtype={ts._referee._dtype}  agent_state={args.agent_state}")
    c.print("[grey42]Ctrl+C to stop — flag per 80 ms frame[/]\n")

    counts: Counter = Counter()
    lat: list[float] = []
    n = 0
    t_start = time.time()
    try:
        for d in ts.stream(ffmpeg_frames(args.url, c), sample_rate=SR,
                           agent_state=args.agent_state):
            n += 1
            counts[d.flag] += 1
            lat.append(d.latency_ms)
            style = STYLE.get(d.flag, "white")
            ts_s = time.strftime("%H:%M:%S")
            if d.flag in STEADY:
                c.print(f"[grey30]{ts_s}[/] [{style}]{d.flag:<13}[/] "
                        f"[grey30]{d.latency_ms:5.1f}ms[/]")
            else:
                c.print(f"[grey58]{ts_s}[/] [bold]FLAG[/] [{style}]{d.flag:<13}[/] "
                        f"{d.latency_ms:5.1f}ms  [grey42]enc {ts.last_encode_ms:.1f}ms[/]")
            if n % 125 == 0:   # ~every 10 s
                p50 = stats.median(lat[-500:])
                p95 = sorted(lat[-500:])[int(len(lat[-500:]) * 0.95) - 1]
                c.print(f"[dim]— {n} frames · {n*0.08:.0f}s · decode p50 {p50:.1f} "
                        f"p95 {p95:.1f}ms (budget 80) —[/]")
    except KeyboardInterrupt:
        pass

    c.print()
    if lat:
        p50 = stats.median(lat)
        p95 = sorted(lat)[int(len(lat) * 0.95) - 1] if len(lat) > 1 else p50
        c.print(f"[bold]{n} frames[/] · {time.time()-t_start:.0f}s wall · "
                f"decode p50 {p50:.1f} · p95 {p95:.1f}ms (budget 80)")
    for flag, k in counts.most_common():
        c.print(f"  [{STYLE.get(flag,'white')}]{flag:<14}[/] {k}")


if __name__ == "__main__":
    main()
