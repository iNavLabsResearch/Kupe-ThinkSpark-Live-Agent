"""One browser connection = one WebSession.

Same pipeline as the terminal agent, but audio arrives over a websocket instead of the
microphone and TTS goes back down the same socket instead of to the speakers. The
ThinkSpark model is loaded once by the server and shared; everything else (STT socket,
LLM history, denoiser, smoother, policy) is per-session.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time

import numpy as np

from agent import config
from agent.denoise import Denoiser
from agent.policy import AgentState, Policy
from agent.providers import KrutrimLLM, SonioxSTT, SonioxTTS
from agent.smoothing import FlagSmoother

MIC_RATE = 24_000
STT_RATE = 16_000


def _f32_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x
    n = int(len(x) * dst / src)
    return np.interp(np.linspace(0, len(x), n, endpoint=False),
                     np.arange(len(x)), x).astype(np.float32)


class _WsUI:
    """Mirrors the terminal UI's interface, but emits JSON to the browser."""

    def __init__(self, ws, loop):
        self.ws = ws
        self.loop = loop

    def _send(self, payload: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            self.ws.send_text(json.dumps(payload)), self.loop
        )

    def log(self, kind: str, detail: str = "", style: str = "") -> None:
        self._send({"type": "log", "kind": kind, "detail": detail})

    def frame(self, flag: str, latency_ms: float, raw: bool = False) -> None:
        self._send({"type": "flag", "flag": flag,
                    "latency_ms": round(latency_ms, 2), "raw": raw})


class WebSession:
    def __init__(self, ws, referee, keys, window: int = 3, denoise: bool = True):
        self.ws = ws
        self.referee = referee          # shared, loaded once by the server
        self.ui = _WsUI(ws, asyncio.get_event_loop())

        self.stt = SonioxSTT(keys.stt, sample_rate=STT_RATE)
        self.llm = KrutrimLLM(keys.llm)
        self.tts = SonioxTTS(keys.tts)
        self.denoiser = Denoiser(sample_rate=MIC_RATE, enabled=denoise)
        self.smoother = FlagSmoother(window=window)
        self.policy = Policy(self)

        self._audio_q: queue.Queue = queue.Queue()
        self._stt_q: asyncio.Queue = asyncio.Queue()
        self._speaking = False
        self._volume = 1.0
        self._stop_playback = threading.Event()
        self._closed = False
        self.frames = 0
        self.decode_ms: list[float] = []

    # -- audio out (down the socket) ------------------------------------ #
    def duck(self) -> None:
        self._volume = 0.35

    def stop_speaking(self) -> None:
        self._stop_playback.set()
        self._speaking = False
        self._volume = 1.0

    async def speak(self, text: str, filler: bool = False) -> None:
        if not text.strip() or self._closed:
            return
        self.policy.state = AgentState.TTS_SPEAKING
        self._speaking = True
        self._stop_playback.clear()
        self._volume = 1.0
        self.referee.set_context(agent_text=text)
        self.ui.log("tts", f"speaking: {text!r}")
        await self.ws.send_text(json.dumps({"type": "tts_start", "text": text}))

        try:
            async for pcm in self.tts.stream(text):
                if self._stop_playback.is_set() or self._closed:
                    break
                if self._volume != 1.0:
                    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
                    pcm = (a * self._volume).astype(np.int16).tobytes()
                await self.ws.send_bytes(pcm)
        except Exception as e:
            self.ui.log("error", f"tts failed: {e}")

        await self.ws.send_text(json.dumps({"type": "tts_end"}))
        self._speaking = False
        self.policy.state = AgentState.TTS_DONE if not filler else AgentState.IDLE
        self.referee.set_context(agent_text="")

    # -- run ------------------------------------------------------------- #
    async def run(self) -> None:
        await self.stt.connect()
        self.ui.log("boot", f"ready · {self.referee.device} · {config.LLM_MODEL} · "
                            f"{config.TTS_MODEL}/{config.TTS_VOICE}")

        await asyncio.gather(
            self._recv_loop(),
            self._referee_loop(),
            self._stt_send_loop(),
            self._stt_recv_loop(),
        )

    async def _recv_loop(self) -> None:
        """Browser -> server: PCM16 @ 24 kHz, or a JSON control message."""
        while not self._closed:
            msg = await self.ws.receive()
            if msg.get("type") == "websocket.disconnect":
                self._closed = True
                return
            if (data := msg.get("bytes")) is not None:
                pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self._audio_q.put(pcm)
            elif (txt := msg.get("text")) is not None:
                try:
                    if json.loads(txt).get("type") == "reset":
                        self.stt.reset_turn()
                        self.smoother.reset()
                        self.policy.state = AgentState.IDLE
                except Exception:
                    pass

    def _chunks(self):
        while not self._closed:
            try:
                chunk = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            chunk = self.denoiser(chunk)
            try:
                self._stt_q.put_nowait(_f32_to_pcm16(_resample(chunk, MIC_RATE, STT_RATE)))
            except asyncio.QueueFull:
                pass
            yield chunk

    async def _referee_loop(self) -> None:
        loop = asyncio.get_running_loop()
        gen = self.referee.stream(source=self._chunks(), sample_rate=MIC_RATE)

        def _next():
            try:
                return next(gen)
            except StopIteration:
                return None

        while not self._closed:
            decision = await loop.run_in_executor(None, _next)
            if decision is None:
                return

            self.frames += 1
            self.decode_ms.append(decision.latency_ms)
            self.ui.frame(decision.flag, decision.latency_ms, raw=True)

            smoothed = self.smoother.push(decision.flag)
            if smoothed is None:
                continue
            self.ui.frame(smoothed, decision.latency_ms, raw=False)

            action = await self.policy.handle(smoothed)
            if action:
                await self.ws.send_text(json.dumps(
                    {"type": "action", "kind": action.kind, "detail": action.detail}))

    async def _stt_send_loop(self) -> None:
        while not self._closed:
            pcm = await self._stt_q.get()
            await self.stt.send_audio(pcm)

    async def _stt_recv_loop(self) -> None:
        async for text, is_final in self.stt.transcripts():
            if self._closed:
                return
            await self.ws.send_text(json.dumps(
                {"type": "stt", "text": text, "final": is_final}))
            self.referee.set_context(agent_text="", stt_partial=self.stt.partial)

            if is_final and "<end>" in text:
                action = await self.policy.commit_from_stt()
                if action:
                    await self.ws.send_text(json.dumps(
                        {"type": "action", "kind": action.kind, "detail": action.detail}))

    async def close(self) -> None:
        self._closed = True
        self._stop_playback.set()
        try:
            await self.stt.close()
        except Exception:
            pass
