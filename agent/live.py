"""The live agent: mic -> FloorAgent (ThinkSpark + STT + Policy) -> speakers."""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np

from agent import config
from agent.config import load_thinkspark
from agent.orchestrator import FRAME_SAMPLES, MIC_RATE, FloorAgent


class LiveAgent:
    def __init__(self, keys, ui, device: str = "auto", window: int = 3,
                 denoise: bool = True):
        self.ui = ui
        ui.log("boot", f"loading ThinkSpark on device={device} ...")
        self.referee = load_thinkspark(device)
        ui.log("boot", f"ThinkSpark ready on {self.referee.device}")
        self.floor = FloorAgent(
            self.referee, keys, ui=ui, window=window, denoise=denoise,
        )
        ui.log(
            "boot",
            "RNNoise active" if self.floor.denoiser.available
            else "RNNoise unavailable — passthrough (pip install pyrnnoise)",
            style="green" if self.floor.denoiser.available else "yellow",
        )
        self._shutdown = threading.Event()
        self._out = None

        # aliases the terminal UI reads at Ctrl+C
        self.frames = 0
        self.decode_ms: list[float] = []

    def _mic_thread(self) -> None:
        import sounddevice as sd

        def _cb(indata, frames_, time_info, status):
            self.floor.push_audio(indata[:, 0].copy(), MIC_RATE)

        with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="float32",
                            blocksize=FRAME_SAMPLES * 2, callback=_cb):
            while not self._shutdown.is_set():
                time.sleep(0.05)

    def _play_pcm(self, pcm: np.ndarray) -> None:
        import sounddevice as sd

        if self._out is None:
            self._out = sd.OutputStream(
                samplerate=config.TTS_SAMPLE_RATE, channels=1, dtype="float32",
            )
            self._out.start()
        audio = pcm.astype(np.float32) / 32768.0
        block = config.TTS_SAMPLE_RATE * 20 // 1000
        for i in range(0, len(audio), block):
            if self.floor._stop_playback.is_set():
                self.ui.log("tts", "playback cut", style="bold red")
                return
            self._out.write(audio[i:i + block])

    def _close_out(self) -> None:
        if self._out is not None:
            try:
                self._out.stop()
                self._out.close()
            except Exception:
                pass
            self._out = None

    async def _playback_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._shutdown.is_set():
            pcm = await loop.run_in_executor(None, self.floor.playback_q.get)
            if pcm is None:
                return
            if isinstance(pcm, np.ndarray):
                await loop.run_in_executor(None, self._play_pcm, pcm)
            self.frames = self.floor.frames
            self.decode_ms = self.floor.decode_ms

    async def run(self) -> None:
        await self.floor.start()
        self.ui.log("ready", "listening — Ctrl+C to stop", style="bold green")
        threading.Thread(target=self._mic_thread, daemon=True).start()
        try:
            await asyncio.gather(
                self.floor.run_until_closed(),
                self._playback_loop(),
            )
        finally:
            self._shutdown.set()
            try:
                self.floor.playback_q.put_nowait(None)
            except Exception:
                pass
            await self.floor.close()
            self._close_out()
            self.frames = self.floor.frames
            self.decode_ms = self.floor.decode_ms
