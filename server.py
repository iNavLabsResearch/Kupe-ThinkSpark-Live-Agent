#!/usr/bin/env python
"""FastAPI websocket server — connect any frontend to the live agent.

    python server.py                 # binds 0.0.0.0, prints the LAN URL to paste
    python server.py --port 8080 --device cuda

Protocol (one websocket, both directions):

    client -> server   binary   PCM16 mono @ 24 kHz, any chunk size
    client -> server   text     {"type":"reset"}
    server -> client   text     {"type":"flag","flag":...,"latency_ms":...,"raw":bool}
                                {"type":"stt","text":...,"final":bool}
                                {"type":"action","kind":...,"detail":...}
                                {"type":"state","state":...}
                                {"type":"tts_start"} / {"type":"tts_end"}
    server -> client   binary   PCM16 mono @ 24 kHz TTS audio to play

The model loads once at startup and is shared by every connection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket

import numpy as np

from agent import config
from agent.config import load_keys

MIC_RATE = 24_000


def lan_ip() -> str:
    """Best-effort LAN address. No traffic is actually sent."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def build_app(device: str, window: int, denoise: bool):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware

    from agent.session import WebSession

    app = FastAPI(title="Kupe ThinkSpark Live Agent")
    # deliberately wide open — this is a local dev/demo server
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    state = {"referee": None}

    @app.on_event("startup")
    async def _load():
        from kupe import ThinkSpark

        print(f"loading ThinkSpark on device={device} ...")
        state["referee"] = ThinkSpark(device=device)
        print(f"ThinkSpark ready on {state['referee'].device}")
        _print_banner()

    @app.get("/health")
    async def health():
        r = state["referee"]
        return {"ok": r is not None,
                "device": getattr(r, "device", None),
                "llm": config.LLM_MODEL,
                "tts": f"{config.TTS_MODEL}/{config.TTS_VOICE}"}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        session = WebSession(ws, state["referee"], load_keys(),
                             window=window, denoise=denoise)
        try:
            await session.run()
        except WebSocketDisconnect:
            pass
        finally:
            await session.close()

    def _print_banner():
        ip = lan_ip()
        url = f"ws://{ip}:{PORT}/ws"
        line = "=" * 62
        print(f"\n{line}\n  Kupe ThinkSpark Live Agent — ready\n{line}")
        print(f"  Paste this into the UI:\n\n      {url}\n")
        print(f"  local:   ws://127.0.0.1:{PORT}/ws")
        print(f"  health:  http://{ip}:{PORT}/health")
        print(f"  web UI:  cd web && npm install && npm run dev")
        print(f"{line}\n")

    return app


PORT = 8000


def main() -> None:
    global PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--no-denoise", action="store_true")
    args = ap.parse_args()
    PORT = args.port

    import uvicorn

    app = build_app(args.device, args.window, not args.no_denoise)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
