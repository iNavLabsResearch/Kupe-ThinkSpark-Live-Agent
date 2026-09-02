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
import socket
from contextlib import asynccontextmanager

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


def public_ip(timeout: float = 2.0) -> str | None:
    """The machine's public address, so a remote GPU box prints a URL you can actually
    connect to. Asks an external echo service for our own IP — nothing else is sent.
    Returns None (and the banner falls back) if there is no egress."""
    import urllib.request

    for url in ("https://api.ipify.org", "https://ifconfig.me/ip",
                "https://icanhazip.com"):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                ip = r.read().decode().strip()
                if ip and len(ip) < 46:
                    return ip
        except Exception:
            continue
    return None


def runpod_proxy_url(port: int) -> str | None:
    """RunPod publishes each declared HTTP port at a proxy hostname. If we are on a pod,
    that URL is the only thing reachable from a browser without an SSH tunnel."""
    import os

    pod_id = os.environ.get("RUNPOD_POD_ID")
    return f"wss://{pod_id}-{port}.proxy.runpod.net/ws" if pod_id else None


def in_container() -> bool:
    """True inside Docker. The container's own IP (172.x) is NOT reachable from a
    browser on the host, so the banner must not advertise it."""
    import os

    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup") as f:
            return any(x in f.read() for x in ("docker", "containerd", "kubepods"))
    except Exception:
        return False


def build_app(device: str, window: int, denoise: bool):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware

    from agent.session import WebSession

    state = {"referee": None}

    @asynccontextmanager
    async def lifespan(app):
        print(f"loading ThinkSpark on device={device} ...")
        state["referee"] = config.load_thinkspark(device)
        print(f"ThinkSpark ready on {state['referee'].device}")
        _print_banner()
        yield

    app = FastAPI(title="Kupe ThinkSpark Live Agent", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

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
        line = "=" * 66
        print(f"\n{line}\n  Kupe ThinkSpark Live Agent — ready\n{line}")

        pub = public_ip()
        runpod = runpod_proxy_url(PORT)
        containerised = in_container()

        print("  Paste ONE of these into the UI — whichever your host actually routes.\n")
        if pub:
            print(f"  direct IP (Vast / open port / nginx on {PORT}):")
            print(f"      ws://{pub}:{PORT}/ws")
            print(f"  nginx on port 80 (./expose.sh):")
            print(f"      ws://{pub}/ws")
        print(f"  same machine:")
        print(f"      ws://127.0.0.1:{PORT}/ws")
        if not containerised:
            print(f"  LAN:")
            print(f"      ws://{lan_ip()}:{PORT}/ws")
        print()
        print("  Tunnel is optional. Only if the provider blocks inbound ports:")
        print(f"      ssh ... -N -L {PORT}:localhost:{PORT}   then  ws://127.0.0.1:{PORT}/ws")
        print("      ./tunnel.sh quick                         then  wss://<cloudflare>/ws")
        if runpod:
            print(f"  RunPod HTTP proxy (often no WebSocket upgrade):")
            print(f"      {runpod}")
        print()
        if pub:
            print(f"  health:  http://{pub}:{PORT}/health")
            print(f"           (fails = port not published — open it or use a tunnel)")
        else:
            print(f"  health:  http://localhost:{PORT}/health")
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
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
