"""The live agent: mic -> ThinkSpark + STT -> policy -> LLM -> TTS -> speakers.

Everything is realtime and concurrent:

  * one thread owns the microphone and fans each 80 ms frame out to two consumers
  * ThinkSpark decides on every frame (local, ~3-45 ms depending on device)
  * Soniox transcribes the same audio in parallel over a websocket
  * the policy turns smoothed flags into LLM/TTS actions
  * TTS playback runs on its own stream and can be cut mid-chunk by BARGE_HARD
"""

from __future__ import annotations

import asyncio
import io
import queue
import threading
import time
import wave

import numpy as np

from agent import config
from agent.denoise import Denoiser
from agent.policy import AgentState, Policy
from agent.providers import KrutrimLLM, SarvamTTS, SonioxSTT
from agent.smoothing import FlagSmoother

MIC_RATE = 24_000        # ThinkSpark's native rate
STT_RATE = 16_000        # Soniox
FRAME_MS = 80
FRAME_SAMPLES = MIC_RATE * FRAME_MS // 1000


def _f32_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    n = int(len(x) * dst / src)
    return np.interp(np.linspace(0, len(x), n, endpoint=False),
                     np.arange(len(x)), x).astype(np.float32)


class LiveAgent:
    def __init__(self, keys, ui, device: str = "auto", window: int = 3,
                 denoise: bool = True):
        from kupe import ThinkSpark

        self.ui = ui
        ui.log("boot", f"loading ThinkSpark on device={device} ...")
        self.referee = ThinkSpark(device=device)
        ui.log("boot", f"ThinkSpark ready on {self.referee.device}")

        self.stt = SonioxSTT(keys.stt, sample_rate=STT_RATE)
        self.llm = KrutrimLLM(keys.llm)
        self.tts = SarvamTTS(keys.tts)
        self.denoiser = Denoiser(sample_rate=MIC_RATE, enabled=denoise)
        ui.log("boot", "RNNoise active" if self.denoiser.available
               else "RNNoise unavailable — passthrough (pip install pyrnnoise)",
               style="green" if self.denoiser.available else "yellow")
        self.smoother = FlagSmoother(window=window)
        self.policy = Policy(self)

        self._mic_q: queue.Queue = queue.Queue()
        self._stt_q: asyncio.Queue = asyncio.Queue()
        self._play_stream = None
        self._speaking = False
        self._volume = 1.0
        self._stop_playback = threading.Event()
        self.frames = 0
        self.decode_ms: list[float] = []

    # ------------------------------------------------------------------ #
    # audio out
    # ------------------------------------------------------------------ #
    def duck(self) -> None:
        self._volume = 0.25

    def stop_speaking(self) -> None:
        self._stop_playback.set()
        self._speaking = False
        self._volume = 1.0

    async def speak(self, text: str, filler: bool = False) -> None:
        if not text.strip():
            return
        self.policy.state = AgentState.TTS_SPEAKING
        self._speaking = True
        self._stop_playback.clear()
        self._volume = 1.0
        self.referee.set_context(agent_text=text)
        self.ui.log("tts", f"speaking: {text!r}", style="cyan")

        wav = await self.tts.synth(text)
        if wav:
            await asyncio.get_running_loop().run_in_executor(None, self._play, wav)

        self._speaking = False
        self.policy.state = AgentState.TTS_DONE if not filler else AgentState.IDLE
        self.referee.set_context(agent_text="")

    def _play(self, wav_bytes: bytes) -> None:
        import sounddevice as sd

        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            rate = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        audio = pcm.astype(np.float32) / 32768.0

        # play in 40 ms blocks so a barge-in cuts within one block
        block = rate * 40 // 1000
        with sd.OutputStream(samplerate=rate, channels=1, dtype="float32") as out:
            for i in range(0, len(audio), block):
                if self._stop_playback.is_set():
                    self.ui.log("tts", "playback cut", style="bold red")
                    return
                out.write(audio[i:i + block] * self._volume)

    # ------------------------------------------------------------------ #
    # audio in
    # ------------------------------------------------------------------ #
    def _mic_thread(self) -> None:
        import sounddevice as sd

        def _cb(indata, frames_, time_info, status):
            self._mic_q.put(indata[:, 0].copy())

        with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="float32",
                            blocksize=FRAME_SAMPLES * 2, callback=_cb):
            while not self._shutdown.is_set():
                time.sleep(0.05)

    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        self._shutdown = threading.Event()
        threading.Thread(target=self._mic_thread, daemon=True).start()

        await self.stt.connect()
        self.ui.log("boot", "Soniox STT connected", style="green")
        self.ui.log("boot", f"LLM {config.LLM_MODEL} · TTS {config.TTS_MODEL}/{config.TTS_VOICE}",
                    style="green")
        self.ui.log("ready", "listening — Ctrl+C to stop", style="bold green")

        await asyncio.gather(
            self._referee_loop(),
            self._stt_send_loop(),
            self._stt_recv_loop(),
        )

    async def _referee_loop(self) -> None:
        """Local ThinkSpark decisions on every 80 ms frame."""
        loop = asyncio.get_running_loop()
        gen = self.referee.stream(source=self._chunks(), sample_rate=MIC_RATE)

        def _next():
            try:
                return next(gen)
            except StopIteration:
                return None

        while True:
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
                self.ui.log(action.kind, action.detail, style="bold yellow")

    def _chunks(self):
        """Mic frames: denoise once, then fan out to ThinkSpark and STT."""
        while True:
            chunk = self.denoiser(self._mic_q.get())
            pcm = _f32_to_pcm16(_resample(chunk, MIC_RATE, STT_RATE))
            try:
                self._stt_q.put_nowait(pcm)
            except asyncio.QueueFull:
                pass
            yield chunk

    async def _stt_send_loop(self) -> None:
        while True:
            pcm = await self._stt_q.get()
            await self.stt.send_audio(pcm)

    async def _stt_recv_loop(self) -> None:
        async for text, is_final in self.stt.transcripts():
            kind = "stt-final" if is_final else "stt"
            self.ui.log(kind, text, style="white" if is_final else "grey58")
            self.referee.set_context(agent_text="", stt_partial=self.stt.partial)
