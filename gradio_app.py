#!/usr/bin/env python
"""Realtime bidirectional voice UI (WebRTC) via Gradio + FastRTC.

    python gradio_app.py
    ./start.sh --gradio

Click the orb, talk, the agent talks back on the same stream. No ngrok.
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

STT_RATE = 16_000
TTS_RATE = config.TTS_SAMPLE_RATE

_llm: KrutrimLLM | None = None
_keys = None

CSS = """
.gradio-container { max-width: 720px !important; margin: 0 auto; }
footer { display: none !important; }
#orb button, #orb .wrap, #orb { border-radius: 999px !important; }
#flag textarea, #flag input { font-size: 28px !important; font-weight: 700;
  text-align: center; letter-spacing: 0.08em; }
#meta textarea { font-family: ui-monospace, monospace; font-size: 13px; }
"""


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x
    n = max(1, int(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x), n, endpoint=False),
                     np.arange(len(x)), x).astype(np.float32)


def _to_mono_f32(sr: int, data) -> tuple[int, np.ndarray]:
    x = np.asarray(data)
    if x.size == 0:
        return sr, x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=0 if x.shape[0] <= 8 else -1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / max(1, np.iinfo(x.dtype).max)
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
    print("==> realtime voice agent ready  (WebRTC send-receive). No ngrok.")


async def _turn(sr: int, wav: np.ndarray) -> tuple[str, str, np.ndarray]:
    stt = SonioxSTT(_keys.stt, sample_rate=STT_RATE)
    tts = SonioxTTS(_keys.tts)
    wav16 = _resample(wav, sr, STT_RATE)
    if wav16.size < STT_RATE * 0.15:
        return "", "", np.zeros(TTS_RATE // 10, dtype=np.float32)
    user = await stt.transcribe_clip(_pcm16(wav16))
    if not user:
        return "", "", np.zeros(TTS_RATE // 10, dtype=np.float32)
    parts: list[str] = []
    async for delta in _llm.stream(user):
        parts.append(delta)
    reply = "".join(parts).strip() or "Sorry, I didn't catch that."
    chunks: list[np.ndarray] = []
    async for pcm in tts.stream(reply):
        chunks.append(np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0)
    out = np.concatenate(chunks) if chunks else np.zeros(TTS_RATE // 10, dtype=np.float32)
    return user, reply, out


def _in_colab_or_kaggle() -> bool:
    if os.path.isdir("/kaggle"):
        return True
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True
    return "google.colab" in sys.modules


def _rtc_config():
    token = os.environ.get("HF_TOKEN") or ""
    try:
        from fastrtc import get_hf_turn_credentials
        if token:
            os.environ.setdefault("HF_TOKEN", token)
        return get_hf_turn_credentials()
    except Exception as e:
        print(f"==> TURN creds skipped ({e}) — WebRTC may fail behind NAT")
        return None


def main() -> None:
    import gradio as gr
    from fastrtc import AdditionalOutputs, ReplyOnPause, WebRTC

    _boot()

    def on_pause(audio):
        if not audio:
            yield AdditionalOutputs("", "no audio", "LISTEN")
            return
        sr, data = audio
        sr, wav = _to_mono_f32(int(sr), data)
        try:
            user, reply, out = asyncio.run(_turn(sr, wav))
        except Exception as e:
            yield AdditionalOutputs("", f"error: {e}", "ERROR")
            return
        flag = "TURN_END" if user else "LISTEN"
        audio_out = out.reshape(1, -1) if getattr(out, "ndim", 1) == 1 else out
        yield (TTS_RATE, audio_out)
        yield AdditionalOutputs(user, reply, flag)

    rtc = _rtc_config()
    with gr.Blocks(title="Kupe ThinkSpark", css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Kupe ThinkSpark\n"
            "Click the **orb**, grant the mic, **talk**. "
            "Audio is bidirectional WebRTC — you speak, the agent speaks back on the same stream."
        )
        flag = gr.Textbox(value="—", label="ThinkSpark", elem_id="flag", interactive=False)
        transcript = gr.Textbox(value="…", label="Live transcript", elem_id="meta",
                                lines=3, interactive=False)
        webrtc = WebRTC(
            label="Talk",
            modality="audio",
            mode="send-receive",
            rtc_configuration=rtc,
            elem_id="orb",
        )
        stream_fn = ReplyOnPause(on_pause)
        try:
            stream_fn = ReplyOnPause(
                on_pause, input_sample_rate=24000, output_sample_rate=TTS_RATE
            )
        except TypeError:
            stream_fn = ReplyOnPause(on_pause)
        webrtc.stream(
            stream_fn,
            inputs=[webrtc],
            outputs=[webrtc],
            time_limit=600,
            concurrency_limit=1,
        )
        webrtc.on_additional_outputs(
            lambda *a: (
                (a[-1] if a else "—") or "—",
                ((str(a[-3] or "") if len(a) >= 3 else "") + "\n"
                 + (str(a[-2] or "") if len(a) >= 2 else "")).strip() or "…",
            ),
            outputs=[flag, transcript],
        )

    print("==> launching Gradio WebRTC UI  (share=True, not ngrok)")
    launch_kw = dict(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        share=True,
    )
    try:
        demo.launch(inline=_in_colab_or_kaggle(), ssr_mode=False, **launch_kw)
    except TypeError:
        demo.launch(**launch_kw)


if __name__ == "__main__":
    main()
