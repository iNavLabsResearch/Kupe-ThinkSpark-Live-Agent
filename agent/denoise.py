"""RNNoise denoising, applied once at the microphone.

Placed at the head of the pipeline so a single pass cleans the audio for *both*
consumers — ThinkSpark's frame decisions and Soniox's transcription. Denoising twice
would double the CPU cost for no benefit.

Why it matters for ThinkSpark specifically: the model decides on 80 ms of audio at a
time using energy and f0 alongside the Mimi tokens. Steady background noise raises the
noise floor and pushes borderline frames toward false SILENCE_BREAK and false barge-in.
Cleaning the input is cheaper than compensating for it downstream.

RNNoise runs natively at 48 kHz on 10 ms frames; pyrnnoise handles the framing. Our mic
runs at 24 kHz, so we resample in and back out. Degrades to a passthrough if pyrnnoise
is not installed — the agent still runs, just noisier.
"""

from __future__ import annotations

import numpy as np

RNNOISE_RATE = 48_000


class Denoiser:
    """Stateful RNNoise wrapper. One instance per audio stream, never shared."""

    def __init__(self, sample_rate: int = 24_000, enabled: bool = True):
        self.sample_rate = sample_rate
        self.available = False
        self._rn = None

        if not enabled:
            return
        try:
            from pyrnnoise import RNNoise

            self._rn = RNNoise(sample_rate=RNNOISE_RATE)
            self.available = True
        except Exception:
            # missing package, or a build without the native lib — passthrough
            self.available = False

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Denoise one float32 mono frame in [-1, 1]. Returns same length, same rate."""
        if not self.available:
            return frame

        try:
            up = _resample(frame, self.sample_rate, RNNOISE_RATE)

            # pyrnnoise wants int16-scaled input
            peak = float(np.max(np.abs(up))) if up.size else 0.0
            if peak > 1.0:
                up = up / peak
            up_i16 = (up * 32767.0).astype(np.float32)

            out = [chunk for _, chunk in self._rn.process_chunk(up_i16)]
            if not out:
                return frame
            clean = np.concatenate(out).astype(np.float32) / 32768.0

            down = _resample(clean, RNNOISE_RATE, self.sample_rate)

            # RNNoise emits in whole 10 ms frames, so length can drift by a few samples
            if len(down) < len(frame):
                down = np.pad(down, (0, len(frame) - len(down)))
            return down[: len(frame)]
        except Exception:
            # never let denoising take down the audio path
            self.available = False
            return frame


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x
    n = int(len(x) * dst / src)
    return np.interp(
        np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x
    ).astype(np.float32)
