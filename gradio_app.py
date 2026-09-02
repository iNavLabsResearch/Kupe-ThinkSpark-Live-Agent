#!/usr/bin/env python
"""Realtime bidirectional voice UI (WebRTC) via Gradio + FastRTC.

ThinkSpark is the floor controller — no Silero, no ReplyOnPause.
Every 80 ms frame is scored; the table shows flag / STT / context / output / latency.

    python gradio_app.py
    ./start.sh --gradio
"""

from __future__ import annotations

import os
import sys

import numpy as np

from agent.config import load_keys, load_thinkspark
from agent.orchestrator import FloorAgent, MIC_RATE

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TTS_RATE = 24_000
TABLE_HEADERS = ["time", "flag", "spoken", "stt", "context", "output", "ms"]

_keys = None
_referee = None
_active: FloorAgent | None = None

CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto; }
footer { display: none !important; }
#orb button, #orb .wrap, #orb { border-radius: 999px !important; }
#flag textarea, #flag input { font-size: 28px !important; font-weight: 700;
  text-align: center; letter-spacing: 0.08em; }
#meta textarea, #stats textarea, #ctx textarea {
  font-family: ui-monospace, monospace; font-size: 13px; }
"""


def _in_colab_or_kaggle() -> bool:
    if os.path.isdir("/kaggle"):
        return True
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True
    return "google.colab" in sys.modules


def _hf_token() -> str:
    if _keys and _keys.hf:
        return _keys.hf
    try:
        from agent import keys as _k
        return getattr(_k, "HF_TOKEN", "") or ""
    except ImportError:
        return os.environ.get("HF_TOKEN", "") or ""


def _normalize_ice(creds: dict) -> dict:
    servers = creds.get("iceServers") or creds.get("ice_servers") or []
    extra = {k: v for k, v in creds.items() if k not in ("iceServers", "ice_servers")}
    return {"iceServers": servers, **extra}


def _rtc_config(ttl: int = 600):
    """ICE servers for Colab/Kaggle NAT. Empty dict = Connection failed."""
    import time

    import httpx

    token = _hf_token().strip()
    if token:
        os.environ["HF_TOKEN"] = token

    errors = []
    if token:
        try:
            from fastrtc import get_cloudflare_turn_credentials
            creds = _normalize_ice(
                get_cloudflare_turn_credentials(hf_token=token, ttl=ttl)
            )
            n = len(creds["iceServers"])
            if n:
                print(f"==> TURN ready  ({n} iceServers, ttl={ttl})")
                return creds
            errors.append("cloudflare helper returned no iceServers")
        except Exception as e:
            errors.append(f"fastrtc helper: {e}")

        for i in range(3):
            try:
                r = httpx.get(
                    "https://turn.fastrtc.org/credentials",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"ttl": str(ttl)},
                    timeout=25.0,
                )
                r.raise_for_status()
                creds = _normalize_ice(r.json())
                n = len(creds["iceServers"])
                if n:
                    print(f"==> TURN ready via httpx  ({n} iceServers, ttl={ttl})")
                    return creds
            except Exception as e:
                errors.append(f"httpx {i+1}: {e}")
                time.sleep(1.2)

    print("==> TURN failed: " + " | ".join(errors[:4]))
    return {
        "iceServers": [
            {"urls": ["stun:stun.cloudflare.com:3478"]},
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    }


def _boot():
    global _keys, _referee
    _keys = load_keys()
    print("==> loading ThinkSpark (no Silero VAD) ...")
    _referee = load_thinkspark("auto")
    print(f"==> ThinkSpark ready on {_referee.device}")
    print("==> realtime voice agent ready  (WebRTC pipe + ThinkSpark referee). No ngrok.")


def _snapshot():
    a = _active
    empty = [["", "", "", "", "", "", ""]]
    if a is None:
        return "—", "0 frames · waiting for Record", "…", "IDLE", empty
    flag, stats, stt, ctx, rows = a.snapshot()
    return flag, stats, stt, ctx, rows or empty


def _make_handler():
    from fastrtc import AsyncStreamHandler

    class ThinkSparkHandler(AsyncStreamHandler):
        """Mic frames in, TTS PCM out. ThinkSpark referees; no Silero VAD."""

        def __init__(self):
            try:
                super().__init__(
                    expected_layout="mono",
                    input_sample_rate=MIC_RATE,
                    output_sample_rate=TTS_RATE,
                )
            except TypeError:
                super().__init__(
                    input_sample_rate=MIC_RATE,
                    output_sample_rate=TTS_RATE,
                )
            self.agent: FloorAgent | None = None

        def copy(self):
            return ThinkSparkHandler()

        async def start_up(self):
            global _active
            self.agent = FloorAgent(_referee, _keys)
            await self.agent.start()
            _active = self.agent

        async def shutdown(self):
            global _active
            if self.agent is not None:
                if _active is self.agent:
                    _active = None
                await self.agent.close()
                self.agent = None

        async def receive(self, frame: tuple[int, np.ndarray]) -> None:
            if self.agent is None:
                return
            sr, data = frame
            self.agent.push_audio(data, int(sr) if sr else MIC_RATE)

        async def emit(self):
            if self.agent is None:
                return None
            try:
                pcm = self.agent.playback_q.get_nowait()
            except Exception:
                return None
            if pcm is None:
                return None
            if isinstance(pcm, np.ndarray):
                a = np.ascontiguousarray(pcm, dtype=np.int16).reshape(1, -1)
                return (TTS_RATE, a)
            return None

    return ThinkSparkHandler()


def main() -> None:
    import gradio as gr
    from fastrtc import WebRTC

    _boot()
    handler = _make_handler()

    def ice():
        return _rtc_config(ttl=600)

    with gr.Blocks(title="Kupe ThinkSpark") as demo:
        gr.Markdown(
            "# Kupe ThinkSpark\n"
            "Click **Record**, grant the mic, talk. ThinkSpark scores every "
            "**80 ms** frame (no Silero VAD). The table is the live referee log."
        )
        flag = gr.Textbox(value="—", label="ThinkSpark", elem_id="flag", interactive=False)
        stats = gr.Textbox(value="0 frames", label="Budget", elem_id="stats", interactive=False)
        with gr.Row():
            stt_box = gr.Textbox(
                value="…", label="STT (processing)", elem_id="meta",
                lines=2, interactive=False,
            )
            ctx_box = gr.Textbox(
                value="IDLE", label="Context (agent state + text)", elem_id="ctx",
                lines=2, interactive=False,
            )
        table = gr.Dataframe(
            headers=TABLE_HEADERS,
            value=[["", "", "", "", "", "", ""]],
            datatype=["str"] * 7,
            label="ThinkSpark generations (newest first)",
            interactive=False,
        )
        rtc_kw = dict(
            label="Talk",
            modality="audio",
            mode="send-receive",
            elem_id="orb",
        )
        server_ice = _rtc_config(ttl=360_000)
        try:
            webrtc = WebRTC(
                rtc_configuration=ice,
                server_rtc_configuration=server_ice,
                **rtc_kw,
            )
        except TypeError:
            webrtc = WebRTC(rtc_configuration=server_ice, **rtc_kw)
        webrtc.stream(
            handler,
            inputs=[webrtc],
            outputs=[webrtc],
            time_limit=600,
            concurrency_limit=1,
        )
        outs = [flag, stats, stt_box, ctx_box, table]
        try:
            tick = gr.Timer(0.1)
            tick.tick(_snapshot, outputs=outs)
        except (AttributeError, TypeError):
            demo.load(_snapshot, None, outs, every=0.1)

    print("==> launching Gradio WebRTC UI  (share=True, not ngrok)")
    launch_kw = dict(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        share=True,
        css=CSS,
        theme=gr.themes.Soft(),
    )
    try:
        demo.launch(inline=_in_colab_or_kaggle(), ssr_mode=False, **launch_kw)
    except TypeError:
        launch_kw.pop("css", None)
        launch_kw.pop("theme", None)
        try:
            demo.launch(**launch_kw)
        except TypeError:
            demo.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    main()
