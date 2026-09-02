#!/usr/bin/env python
"""FastAPI websocket server. Binds 0.0.0.0 and, if NGROK_AUTHTOKEN is set, opens ngrok.

    python server.py                 # loads .env, serves :8000, prints the UI URL
    python server.py --no-ngrok      # local / direct-IP only

Protocol (one websocket, both directions):

    client -> server   binary   PCM16 mono @ 24 kHz, any chunk size
    client -> server   text     {"type":"reset"}
    server -> client   text     JSON events, plus binary PCM16 TTS
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from contextlib import asynccontextmanager

from agent import config
from agent.config import load_keys
from agent import expose

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MIC_RATE = 24_000
PORT = 8000
WEB_APP = Path(__file__).parent / "web" / "app.html"


def build_app(device: str, window: int, denoise: bool, use_ngrok: bool):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse

    from agent.session import WebSession

    state = {"referee": None}

    @asynccontextmanager
    async def lifespan(app):
        print(f"loading ThinkSpark on device={device} ...")
        state["referee"] = config.load_thinkspark(device)
        print(f"ThinkSpark ready on {state['referee'].device}")
        public_https = expose.open_ngrok(PORT) if use_ngrok else None
        _print_banner(public_https)
        try:
            yield
        finally:
            expose.close_ngrok()

    app = FastAPI(title="Kupe ThinkSpark Live Agent", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def index():
        return FileResponse(WEB_APP)

    @app.get("/health")
    async def health():
        r = state["referee"]
        return {
            "ok": r is not None,
            "device": getattr(r, "device", None),
            "llm": config.LLM_MODEL,
            "tts": f"{config.TTS_MODEL}/{config.TTS_VOICE}",
        }

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        print(f"ws handshake  origin={ws.headers.get('origin')}  "
              f"ua={(ws.headers.get('user-agent') or '')[:60]}")
        await ws.accept()
        print("ws accepted")
        session = WebSession(
            ws, state["referee"], load_keys(), window=window, denoise=denoise
        )
        try:
            await session.run()
        except WebSocketDisconnect:
            print("ws disconnected")
        finally:
            await session.close()

    def _print_banner(public_https: str | None) -> None:
        line = "=" * 66
        print(f"\n{line}\n  Kupe ThinkSpark Live Agent — ready\n{line}")
        if public_https:
            print("  Open this URL in the browser (click ngrok 'Visit Site' once):\n")
            print(f"      {public_https}\n")
            print("  Then hit Connect — same origin, WebSockets work.")
            print("  Do NOT paste wss:// into localhost:5173 (ngrok blocks that).\n")
            print(f"  ws:      {expose.to_ws(public_https)}")
            print(f"  health:  {public_https}/health")
        else:
            print("  UI:  http://127.0.0.1:{}/".format(PORT))
            print(f"  ws:  ws://127.0.0.1:{PORT}/ws")
        print(f"{line}\n")

    return app


def main() -> None:
    global PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--no-ngrok", action="store_true", help="do not open an ngrok tunnel")
    args = ap.parse_args()
    PORT = args.port

    use_ngrok = not args.no_ngrok

    import uvicorn

    app = build_app(args.device, args.window, not args.no_denoise, use_ngrok)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
