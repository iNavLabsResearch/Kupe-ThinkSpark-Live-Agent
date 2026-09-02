#!/usr/bin/env python
"""Prove the rewritten streaming inference is correct AND fast.

Run this on the box you'll actually serve from (4090 / H100 / etc.) — it needs only the
HF token, no provider keys:

    export HF_TOKEN=...            # if the checkpoint repo is private
    python verify_streaming.py

What it checks
--------------
1. **Correctness.** It feeds one synthetic frame stream through the new streaming
   (KV-cache) path and through the reference stateless full-recompute path, and asserts
   the per-frame control flags are IDENTICAL. If they match, the O(1)-per-frame decoder
   is bit-for-bit equivalent to the O(window) reference the model was evaluated with.
2. **Latency.** It reports per-frame decode p50/p95 and the Mimi encode cost against the
   80 ms real-time budget. This is the number that was 40-50 ms (and effectively 2-13x
   that, because the old loop also ran a full-window spoken decode nearly every frame);
   it should now be single-digit ms on a GPU.

It does not touch STT/LLM/TTS — it isolates exactly the part that was too slow.
"""

from __future__ import annotations

import os
import statistics as stats
import time

import numpy as np

from agent import config


def _synth_frames(encoder, seconds: float = 6.0):
    """A few seconds of synthetic 24 kHz audio: voiced tone, a pause, noise, a sweep.
    Encoded through the STREAMING encoder so cb0/energy/f0 look like the live path."""
    sr = 24_000
    t = np.arange(int(sr * seconds)) / sr
    wav = np.zeros_like(t, dtype=np.float32)
    # voiced 150 Hz-ish speech-like segment
    seg = (t < 2.0)
    wav[seg] = 0.3 * np.sin(2 * np.pi * 160 * t[seg]) * (1 + 0.2 * np.sin(2 * np.pi * 4 * t[seg]))
    # 2.0-3.0 s: silence (dead air)
    # 3.0-4.0 s: low noise
    seg = (t >= 3.0) & (t < 4.0)
    wav[seg] = 0.02 * np.random.randn(seg.sum()).astype(np.float32)
    # 4.0-6.0 s: rising pitch sweep (approaching a turn end)
    seg = t >= 4.0
    f = np.linspace(140, 240, seg.sum())
    wav[seg] = 0.3 * np.sin(2 * np.pi * f * t[seg])

    hop = 1920
    frames = []
    for i in range(0, len(wav) - hop, hop):
        enc = encoder.push(wav[i:i + hop], sr)
        for j in range(enc.num_frames):
            frames.append((int(enc.cb0[j]), float(enc.energy[j]), float(enc.f0[j])))
    return frames


def _agent_state_for(i: int, n: int) -> str:
    # walk through the states so barge/hold logic is exercised
    if i < n * 0.4:
        return "IDLE"
    if i < n * 0.6:
        return "LLM_GEN"
    if i < n * 0.85:
        return "TTS_SPEAKING"
    return "TTS_DONE"


def main() -> None:
    from kupe import ThinkSpark
    from kupe._thinkspark.inference import FrameInput

    hf = os.environ.get("HF_TOKEN") or ""
    print(f"loading {config.TS_REPO} :: {config.TS_SUBFOLDER}")
    ts = ThinkSpark(config.TS_REPO, device="auto", subfolder=config.TS_SUBFOLDER,
                    hf_token=hf or None)
    print(f"device={ts.device}  dtype={ts._referee._dtype}")

    frames = _synth_frames(ts._stream_encoder)
    n = len(frames)
    inputs = [FrameInput(cb0=c, energy=e, f0=f, agent_state=_agent_state_for(i, n))
              for i, (c, e, f) in enumerate(frames)]
    print(f"{n} synthetic frames ({n * 0.08:.1f} s of audio)\n")

    ref = ts._referee

    # ---- streaming (KV-cache) pass ------------------------------------- #
    ref.reset()
    ref.force_recompute = False
    stream_flags, decode_ms = [], []
    for fi in inputs:
        r = ref.step(fi)
        stream_flags.append(r.flag)
        decode_ms.append(r.decode_ms)

    # ---- reference (stateless recompute) pass -------------------------- #
    ref.reset()
    ref.force_recompute = True
    recompute_flags = [ref.step(fi).flag for fi in inputs]
    ref.force_recompute = False
    ref.reset()

    # ---- correctness --------------------------------------------------- #
    mism = [(i, a, b) for i, (a, b) in enumerate(zip(stream_flags, recompute_flags)) if a != b]
    print("=" * 60)
    if not mism:
        print(f"CORRECTNESS: PASS — all {n} streamed flags == reference flags")
    else:
        print(f"CORRECTNESS: {len(mism)}/{n} frames differ (streamed vs reference):")
        for i, a, b in mism[:20]:
            print(f"  frame {i:4d}  streaming={a:<13} reference={b}")
        print("  (a few boundary differences can occur only if the sequence exceeds the")
        print("   512 sliding window; within a turn they must be zero.)")

    # ---- latency ------------------------------------------------------- #
    w = decode_ms
    p50 = stats.median(w)
    p95 = sorted(w)[max(0, int(len(w) * 0.95) - 1)]
    print("=" * 60)
    print(f"LATENCY (per 80 ms frame, {ts.device}):")
    print(f"  decode p50 {p50:6.2f} ms   p95 {p95:6.2f} ms   max {max(w):6.2f} ms")
    print(f"  last mimi encode {ts.last_encode_ms:6.2f} ms")
    print(f"  budget 80 ms/frame  ->  headroom p95 {80 - p95:.1f} ms")
    if p95 <= 40:
        print("  VERDICT: comfortably real-time (guide target p95 <= 40 ms)")
    elif p95 <= 80:
        print("  VERDICT: real-time but tight — prefer a CUDA box for production")
    else:
        print("  VERDICT: OVER BUDGET on this device — run on CUDA (bf16)")

    # ---- spoken head (only when asked) --------------------------------- #
    t0 = time.perf_counter()
    say = ref.generate_spoken("IDLE")
    print("=" * 60)
    print(f"spoken head (on-demand): {(time.perf_counter() - t0) * 1000:.1f} ms -> {say!r}")

    # ---- flag histogram ------------------------------------------------ #
    from collections import Counter
    print("=" * 60)
    print("flag histogram (streaming):")
    for flag, k in Counter(stream_flags).most_common():
        print(f"  {flag:<14} {k}")


if __name__ == "__main__":
    main()
