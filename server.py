#!/usr/bin/env python
"""FastAPI websocket server. Binds 0.0.0.0 and opens a Cloudflare quick tunnel.

    python server.py                 # loads .env, serves :8000, prints the UI URL
    python server.py --no-tunnel     # local / direct-IP only
"""

from __future__ import annotations

import argparse
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

PORT = 8000
WEB_APP = Path(__file__).parent / "web" / "app.html"


def build_app(device: str, window: int, denoise: bool, use_tunnel: bool):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, Response

    from agent.session import WebSession

    state = {"referee": None}

    @asynccontextmanager
    async def lifespan(app):
        print(f"loading ThinkSpark on device={device} ...")
        state["referee"] = config.load_thinkspark(device)
        print(f"ThinkSpark ready on {state['referee'].device}")
        public_https = expose.open_tunnel(PORT) if use_tunnel else None
        _print_banner(public_https)
        try:
            yield
        finally:
            expose.close_tunnel()

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

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

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
        print(f"ws handshake  origin={ws.headers.get('origin')}")
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
            print("  Open this in the browser, then hit Connect:\n")
            print(f"      {public_https}\n")
            print(f"  ws:      {expose.to_ws(public_https)}")
            print(f"  health:  {public_https}/health")
        else:
            print(f"  UI:  http://127.0.0.1:{PORT}/")
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
    ap.add_argument("--no-tunnel", action="store_true", help="do not open a public tunnel")
    ap.add_argument("--no-ngrok", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    PORT = args.port

    import uvicorn

    app = build_app(
        args.device, args.window, not args.no_denoise,
        use_tunnel=not (args.no_tunnel or args.no_ngrok),
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        proxy_headers=True,
        forwarded_allow_ips="*",
        ws="websockets",
    )


if __name__ == "__main__":
    main()
