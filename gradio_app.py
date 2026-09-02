#!/usr/bin/env python
"""Gradio voice UI for Colab / Kaggle. No ngrok — Gradio share handles the public URL.

    python gradio_app.py
    ./start.sh --gradio
"""

from __future__ import annotations

import asyncio
import os
import sys

import numpy as np

from agent import config
from agent.config import load_keys
from agent.providers import KrutrimLLM, SonioxSTT, SonioxTTS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MIC_RATE = 24_000
STT_RATE = 16_000

_llm: KrutrimLLM | None = None
_keys = None


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x
    n = int(len(x) * dst / src)
    return np.interp(np.linspace(0, len(x), n, endpoint=False),
                     np.arange(len(x)), x).astype(np.float32)


def _to_mono_f32(sr: int, data) -> tuple[int, np.ndarray]:
    x = np.asarray(data)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 1.5:
            x = x / 32768.0
    return sr, np.clip(x, -1.0, 1.0)


def _pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _boot():
    global _llm, _keys
    _keys = load_keys()
    _llm = KrutrimLLM(_keys.llm)
    print("==> gradio agent ready  (STT + LLM + TTS). No ngrok.")


async def _turn(sr: int, wav: np.ndarray) -> tuple[str, str, tuple[int, np.ndarray] | None]:
    stt = SonioxSTT(_keys.stt, sample_rate=STT_RATE)
    tts = SonioxTTS(_keys.tts)
    wav16 = _resample(wav, sr, STT_RATE)
    user = await stt.transcribe_clip(_pcm16(wav16))
    if not user:
        return "", "(no speech detected)", None
    reply_parts: list[str] = []
    async for delta in _llm.stream(user):
        reply_parts.append(delta)
    reply = "".join(reply_parts).strip()
    chunks: list[np.ndarray] = []
    async for pcm in tts.stream(reply or "Sorry, I didn't catch that."):
        chunks.append(np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0)
    out = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    return user, reply, (config.TTS_SAMPLE_RATE, out)


def respond(audio, history):
    history = history or []
    if audio is None:
        return history, None, "record something first"
    sr, data = audio
    sr, wav = _to_mono_f32(int(sr), data)
    if wav.size < sr * 0.2:
        return history, None, "too short — hold the mic a bit longer"
    try:
        user, reply, out = asyncio.run(_turn(sr, wav))
    except Exception as e:
        return history, None, f"error: {e}"
    if user:
        if history and isinstance(history[0], dict):
            history = history + [
                {"role": "user", "content": user},
                {"role": "assistant", "content": reply},
            ]
        else:
            history = history + [[user, reply]]
    return history, out, "ok"


def _in_colab_or_kaggle() -> bool:
    if os.path.isdir("/kaggle"):
        return True
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True
    return "google.colab" in sys.modules


def main() -> None:
    import gradio as gr

    _boot()
    with gr.Blocks(title="Kupe ThinkSpark") as demo:
        gr.Markdown(
            "## Kupe ThinkSpark\n"
            "Hold the mic, speak, release. The agent transcribes, replies, and talks back."
        )
        try:
            chat = gr.Chatbot(label="Conversation", height=360, type="tuples")
        except TypeError:
            chat = gr.Chatbot(label="Conversation", height=360)
        with gr.Row():
            mic = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Microphone",
            )
            speaker = gr.Audio(type="numpy", autoplay=True, label="Agent")
        status = gr.Textbox(label="Status", interactive=False)
        try:
            mic.stop_recording(respond, inputs=[mic, chat], outputs=[chat, speaker, status])
        except Exception:
            mic.change(respond, inputs=[mic, chat], outputs=[chat, speaker, status])

    share = _in_colab_or_kaggle()
    print("==> launching Gradio  share=" + str(share) + "  (not ngrok)")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        share=share or True,  # public link via Gradio, never ngrok
        inline=_in_colab_or_kaggle(),
    )


if __name__ == "__main__":
    main()
