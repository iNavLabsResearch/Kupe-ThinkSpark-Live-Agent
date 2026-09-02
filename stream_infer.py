#!/usr/bin/env python
"""ThinkSpark inference ONLY — live stream + realtime AssemblyAI STT, flags only.

The "new mode": STT runs IN PARALLEL (not as a gate). Its partials are fed to the
referee as context; the model watches the live audio at 80 ms frames and prints the
control flag it would emit. NO LLM, NO TTS, no floor actions — just the referee.

The agent's persona is a quiet LISTENER (it only decides when to listen / hold /
interject), so decisions reflect a human listener on the call.

    export HF_TOKEN=...                 # if the checkpoint is private
    export ASSEMBLYAI_API_KEY=...       # realtime STT (omit or --no-stt to skip)
    python stream_infer.py             # BBC World Service live

Needs ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import queue
import shutil
import statistics as stats
import subprocess
import threading
import time
from collections import Counter

import numpy as np
from rich.console import Console

from agent import config

# Live English talk radio — reliable, not geo/rate-blocked. Override with --url.
# (BBC World Service public HLS is UK-geo/rate-limited and often 404/429 from cloud boxes.)
DEFAULT_URL = "https://npr-ice.streamguys1.com/live.mp3"

LISTENER_PROMPT = ("You are a calm human listener on a live call. You only decide WHEN "
                   "to listen, hold, back-channel, or interject — you never speak the "
                   "answer. Languages: English, Hindi, Gujarati.")

SR = 24_000
STT_SR = 16_000
FRAME = SR * 80 // 1000            # 1920 samples / 80 ms

STYLE = {
    "LISTEN": "grey42", "HOLD": "steel_blue1", "INCOMPLETE": "khaki1",
    "TURN_END": "bold spring_green2", "BARGE_SOFT": "bold orange1",
    "BARGE_HARD": "bold red1", "CONTINUE": "grey58",
    "PREFETCH_LLM": "bold magenta", "COMMIT_LLM": "bold cyan",
    "CANCEL_LLM": "bold yellow", "SILENCE_BREAK": "blue",
}
STEADY = {"LISTEN", "CONTINUE"}


class Shared:
    partial = ""
    stt_status = "off"


def f32_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    n = max(1, int(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x), n, endpoint=False),
                     np.arange(len(x)), x).astype(np.float32)


def ffmpeg_reader(url: str, c: Console, q_model: queue.Queue, q_stt: queue.Queue | None,
                  stop: threading.Event):
    if not shutil.which("ffmpeg"):
        c.print("[bold red]ffmpeg not found[/] — apt-get install -y ffmpeg")
        q_model.put(None)
        if q_stt is not None:
            q_stt.put(None)
        return
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning",
           "-reconnect", "1", "-reconnect_streamed", "1",
           "-reconnect_delay_max", "5", "-rw_timeout", "15000000",
           "-user_agent", "Mozilla/5.0", "-i", url,
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SR), "-"]
    c.print(f"[grey58]ffmpeg <- {url}[/]")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=FRAME * 4)

    # surface ffmpeg's own errors on its stderr (so a dead/blocked URL is visible)
    def _pump_err():
        for line in iter(proc.stderr.readline, b""):
            s = line.decode("utf-8", "replace").rstrip()
            if s:
                c.print(f"[yellow]ffmpeg: {s}[/]")
    threading.Thread(target=_pump_err, daemon=True).start()

    nbytes = FRAME * 4
    got = 0
    try:
        while not stop.is_set():
            buf = proc.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                break
            got += 1
            frame = np.frombuffer(buf, dtype=np.float32).copy()
            for q in (q_model, q_stt):
                if q is None:
                    continue
                try:
                    q.put_nowait(frame)
                except queue.Full:
                    pass
    finally:
        if got == 0:
            c.print("[bold red]ffmpeg delivered 0 frames[/] — URL likely dead/blocked "
                    "from this box. Try --url with a reachable stream (see above errors).")
        proc.kill()
        q_model.put(None)
        if q_stt is not None:
            q_stt.put(None)


def stt_thread(key: str, ts, sh: Shared, c: Console, q_stt: queue.Queue,
               stop: threading.Event):
    from agent.providers import AssemblyAISTT

    async def run():
        stt = AssemblyAISTT(key, sample_rate=STT_SR)
        try:
            await stt.connect()
            sh.stt_status = "live"
            c.print("[bold green]AssemblyAI STT connected[/]")
        except Exception as e:
            sh.stt_status = f"connect failed: {e}"
            c.print(f"[bold red]STT connect failed[/]: {e}")
            return
        loop = asyncio.get_running_loop()

        async def sender():
            while not stop.is_set():
                frame = await loop.run_in_executor(None, q_stt.get)
                if frame is None:
                    return
                pcm = f32_to_pcm16(resample(frame, SR, STT_SR))
                try:
                    await stt.send_audio(pcm)
                except Exception:
                    return

        async def receiver():
            try:
                async for text, is_final in stt.transcripts():
                    if stop.is_set():
                        return
                    sh.partial = text
                    ts.set_context(agent_text="", stt_partial=text)
                    tag = "STT-FINAL" if is_final else "stt"
                    col = "white" if is_final else "grey58"
                    c.print(f"[{col}]{time.strftime('%H:%M:%S')}  {tag:<9} {text}[/]")
            except Exception as e:
                c.print(f"[yellow]stt recv ended: {e}[/]")

        try:
            await asyncio.gather(sender(), receiver())
        finally:
            await stt.close()

    asyncio.run(run())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--agent-state", default="IDLE")
    ap.add_argument("--no-stt", action="store_true", help="skip AssemblyAI STT")
    args = ap.parse_args()

    c = Console()
    from kupe import ThinkSpark

    c.print(f"[grey58]loading[/] {config.TS_REPO} :: {config.TS_SUBFOLDER}")
    ts = ThinkSpark(config.TS_REPO, device=args.device, subfolder=config.TS_SUBFOLDER)
    # listener persona (works on old + new SDK)
    try:
        ts._referee.system_prompt = LISTENER_PROMPT
    except Exception:
        pass
    dt = getattr(getattr(ts, "_referee", None), "_dtype", "?")
    c.print(f"[bold green]ThinkSpark ready[/] on [bold]{ts.device}[/]  dtype={dt}  "
            f"agent_state={args.agent_state}")

    key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    use_stt = (not args.no_stt) and bool(key)
    if not use_stt and not args.no_stt:
        c.print("[yellow]ASSEMBLYAI_API_KEY not set — running flags-only, no STT[/]")

    sh = Shared()
    stop = threading.Event()
    q_model: queue.Queue = queue.Queue(maxsize=200)
    q_stt: queue.Queue | None = queue.Queue(maxsize=200) if use_stt else None

    threading.Thread(target=ffmpeg_reader,
                     args=(args.url, c, q_model, q_stt, stop), daemon=True).start()
    if use_stt:
        sh.stt_status = "connecting"
        threading.Thread(target=stt_thread,
                         args=(key, ts, sh, c, q_stt, stop), daemon=True).start()

    def frames():
        while True:
            f = q_model.get()
            if f is None:
                return
            yield f

    c.print("[grey42]Ctrl+C to stop — flag per 80 ms frame (dec/enc/tot ms)[/]\n")
    counts: Counter = Counter()
    lat: list[float] = []
    n = 0
    t0 = time.time()
    try:
        for d in ts.stream(frames(), sample_rate=SR, agent_state=args.agent_state):
            n += 1
            counts[d.flag] += 1
            lat.append(d.latency_ms)
            style = STYLE.get(d.flag, "white")
            enc = getattr(ts, "last_encode_ms", 0.0)
            tot = d.latency_ms + enc
            tsr = time.strftime("%H:%M:%S")
            part = (sh.partial[-48:]) if sh.partial else ""
            if d.flag in STEADY:
                c.print(f"[grey30]{tsr} {d.flag:<13} dec {d.latency_ms:5.1f} "
                        f"enc {enc:4.1f} tot {tot:5.1f}ms[/]  [grey30]{part}[/]")
            else:
                c.print(f"[grey58]{tsr}[/] [bold]FLAG[/] [{style}]{d.flag:<13}[/] "
                        f"[bold]dec {d.latency_ms:5.1f}[/] enc {enc:4.1f} tot {tot:5.1f}ms"
                        f"  [italic grey58]{part}[/]")
            if n % 125 == 0:
                w = lat[-500:]
                p95 = sorted(w)[int(len(w) * 0.95) - 1]
                c.print(f"[dim]— {n} frames · {n*0.08:.0f}s · dec p50 {stats.median(w):.1f} "
                        f"p95 {p95:.1f}ms (budget 80) · stt {sh.stt_status} —[/]")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()

    c.print()
    if lat:
        p95 = sorted(lat)[int(len(lat) * 0.95) - 1] if len(lat) > 1 else lat[0]
        c.print(f"[bold]{n} frames[/] · {time.time()-t0:.0f}s wall · "
                f"dec p50 {stats.median(lat):.1f} · p95 {p95:.1f}ms (budget 80)")
    for flag, k in counts.most_common():
        c.print(f"  [{STYLE.get(flag,'white')}]{flag:<14}[/] {k}")


if __name__ == "__main__":
    main()
