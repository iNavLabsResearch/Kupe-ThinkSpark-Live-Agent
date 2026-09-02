"""Optional ngrok tunnel so Colab / Kaggle / RunPod can reach /ws without inbound ports."""

from __future__ import annotations

import os
from typing import Optional

_tunnel = None


def to_ws(https_url: str) -> str:
    u = https_url.rstrip("/")
    if u.startswith("https://"):
        return "wss://" + u[len("https://"):] + "/ws"
    if u.startswith("http://"):
        return "ws://" + u[len("http://"):] + "/ws"
    return u + "/ws"


def open_ngrok(port: int) -> Optional[str]:
    """Open an HTTPS tunnel to localhost:`port`. Returns the public https URL, or None if unset.

    Requires NGROK_AUTHTOKEN. Optional NGROK_DOMAIN for a reserved hostname.
    """
    global _tunnel
    token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if not token:
        return None

    try:
        from pyngrok import ngrok
    except ImportError as e:
        raise SystemExit("pyngrok missing — pip install pyngrok") from e

    ngrok.set_auth_token(token)
    try:
        ngrok.kill()
    except Exception:
        pass

    kwargs = {}
    domain = os.environ.get("NGROK_DOMAIN", "").strip()
    if domain:
        kwargs["hostname"] = domain

    _tunnel = ngrok.connect(addr=port, proto="http", **kwargs)
    url = _tunnel.public_url
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url


def close_ngrok() -> None:
    global _tunnel
    try:
        from pyngrok import ngrok
        ngrok.kill()
    except Exception:
        pass
    _tunnel = None
