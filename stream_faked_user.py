#!/usr/bin/env python
"""stream_faked_user.py — feed the agent a SONIOX-VOICED copy of your speech.

The model only understands Soniox-TTS audio (its training distribution), not your raw mic.
So we fake the user:

  PART 1 (faker, separate): your mic ▸ Soniox STT#1 ▸ (final transcript)
                                     ▸ Soniox TTS#1 (a TRAINING voice) ▸ in-distribution audio
  PART 2 (the agent):       that audio ▸ ThinkSpark + Soniox STT#2 + LLM + Soniox TTS ▸ reply

- STT is Soniox in BOTH stages (config.STT_PROVIDER).
- The model gets ONLY the final transcript as text (partials never churn it).
- A continuous SILENCE feeder keeps ThinkSpark processing every 80 ms even when nobody
  speaks — the "infinite fountain" in the right pane.
- Split screen: RIGHT = ThinkSpark flags, LEFT = STT/LLM/TTS.

    export ASSEMBLYAI_API_KEY=... KRUTRIM_API_KEY=... SONIOX_API_KEY=... HF_TOKEN=...
    python stream_faked_user.py --device mps

Ctrl+C to stop.
"""
from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import asyncio
import queue

import numpy as np

from agent import config
from agent.config import load_keys
from agent.live import LiveAgent
from agent.providers import SonioxTTS, make_stt
from agent.split_ui import SplitUI
from stream_agent import PERSONA_LLM, PERSONA_REFEREE, UI

SR = 24_000            # Soniox out + model in
STT1_SR = 16_000       # faker STT input
FRAME = SR * 80 // 1000
FAKE_VOICE = "Emma"    # a voice from the training data-gen catalog (soniox_default_voices)


async def _fake_user(agent: LiveAgent, keys, args, ui) -> None:
    """PART 1 — your mic ▸ Soniox STT#1 ▸ Soniox TTS#1 ▸ (queue) ▸ agent."""
    import sounddevice as sd

    stt1 = make_stt(config.STT_PROVIDER, keys, sample_rate=STT1_SR)     # SEPARATE
    tts1 = SonioxTTS(keys.tts, voice=args.voice, language="en", sample_rate=SR)

    q: queue.Queue = queue.Queue(maxsize=400)
    feed_q: asyncio.Queue = asyncio.Queue()          # frames to stream into the agent
    loop = asyncio.get_running_loop()

    def _cb(indata, frames_, time_info, status):
        try:
            q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    mic = sd.RawInputStream(samplerate=STT1_SR, channels=1, dtype="int16",
                            blocksize=1600, callback=_cb)
    mic.start()

    try:
        await stt1.connect()
    except Exception as e:
        ui.log("error", f"faker STT connect failed: {e}")
        return
    ui.log("boot", f"FAKED USER active — you ▸ Soniox STT ▸ Soniox({args.voice}) ▸ agent")

    async def _feeder():
        """Push a frame every 80 ms — a queued synth frame, else silence. Keeps
        ThinkSpark's fountain running continuously (even on dead air), and marks when the
        frame is REAL faked-user audio so the UI can colour it."""
        silence = np.zeros(FRAME, dtype=np.float32)
        next_t = loop.time()
        last_real = 0.0
        while not agent._shutdown.is_set():
            try:
                frame = feed_q.get_nowait()
                last_real = loop.time()
                agent.floor._user_audio_active = True
            except asyncio.QueueEmpty:
                frame = silence
                if loop.time() - last_real > 0.2:
                    agent.floor._user_audio_active = False
            agent.floor.push_audio(frame, SR)
            next_t += 0.08
            # never try to "catch up" a big event-loop stall by flooding the queue
            if next_t < loop.time() - 0.24:
                next_t = loop.time()
            await asyncio.sleep(max(0.0, next_t - loop.time()))

    async def _sender():
        while not agent._shutdown.is_set():
            pcm = await loop.run_in_executor(None, q.get)
            if pcm is None:
                return
            if agent.floor._echo_active():           # echo guard: don't hear the agent
                pcm = b"\x00\x00" * (len(pcm) // 2)
            try:
                await stt1.send_audio(pcm)
            except Exception:
                return

    async def _synth(text: str) -> None:
        ui.log("boot", f"→ Soniox({args.voice}) synth: {text!r}")
        buf = bytearray()
        try:
            async for pcm in tts1.stream(text):
                buf += pcm
        except Exception as e:
            ui.log("error", f"faker TTS: {e}")
            return
        if not buf:
            return
        audio = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0
        for i in range(0, len(audio), FRAME):        # queue; the feeder paces at 80 ms
            frame = audio[i:i + FRAME]
            if len(frame) < FRAME:
                frame = np.pad(frame, (0, FRAME - len(frame)))
            await feed_q.put(frame)

    async def _receiver():
        while not agent._shutdown.is_set():
            try:
                async for text, is_final in stt1.transcripts():
                    if agent._shutdown.is_set():
                        return
                    text = (text or "").strip()
                    if not text:
                        continue
                    if not is_final:
                        ui.log("stt", f"(you) {text}")
                        continue
                    ui.log("stt-final", f"(you) {text}")
                    await _synth(text)
            except Exception as e:
                if agent._shutdown.is_set():
                    return
                ui.log("error", f"faker STT recv: {e}")
                await asyncio.sleep(0.5)
                try:
                    await stt1.close(); await stt1.connect()
                except Exception:
                    return

    try:
        await asyncio.gather(_feeder(), _sender(), _receiver())
    finally:
        try:
            mic.stop(); mic.close()
        except Exception:
            pass
        try:
            await stt1.close()
        except Exception:
            pass


async def _run(args) -> None:
    keys = load_keys()
    _holder = {}   # late-bound so active_fn can reach agent.floor created below
    ui = UI(show_raw=args.raw) if args.plain else SplitUI(
        active_fn=lambda: getattr(_holder.get("floor"), "_user_audio_active", False))
    agent = LiveAgent(keys, ui, device=args.device, window=args.window,
                      denoise=not args.no_denoise)
    _holder["floor"] = agent.floor
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
    await agent.floor.start()
    ui.log("ready", f"you ▸ Soniox STT ▸ Soniox({args.voice}) ▸ ThinkSpark({dev}) ▸ "
                    f"STT#2‖LLM‖TTS. speak — Ctrl+C to stop.")

    tasks = [
        asyncio.create_task(agent.floor.run_until_closed()),
        asyncio.create_task(agent._playback_loop()),
        asyncio.create_task(_fake_user(agent, keys, args, ui)),
    ]
    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        agent._shutdown.set()
        try:
            agent.floor.playback_q.put_nowait(None)
        except Exception:
            pass
        await agent.floor.close()
        agent._close_out()
        ui.summary(agent.frames, agent.decode_ms)


def main() -> None:
    ap = argparse.ArgumentParser(description="Feed the agent a Soniox-voiced copy of your speech")
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--voice", default=FAKE_VOICE, help="Soniox training voice for the faked user")
    ap.add_argument("--window", type=int, default=3, help="smoothing window (3 = 240 ms)")
    ap.add_argument("--raw", action="store_true", help="(plain mode) print every 80 ms frame")
    ap.add_argument("--plain", action="store_true", help="scrolling log instead of the split TUI")
    ap.add_argument("--no-denoise", action="store_true", help="disable RNNoise")
    ap.add_argument("--pure-thinkspark", action="store_true",
                    help="commit only on ThinkSpark TURN_END (ignore STT endpoint)")
    ap.add_argument("--persona-referee", default=PERSONA_REFEREE)
    ap.add_argument("--persona-llm", default=PERSONA_LLM)
    args = ap.parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
