"""ngrok tunnel so Colab / Kaggle / RunPod can reach /ws without inbound ports."""

from __future__ import annotations

import os
from typing import Optional

# Fallback if .env / the environment has no token (Colab, Kaggle, fresh pods).
_DEFAULT_NGROK_AUTHTOKEN = "3Imhi3otkOTZqNt0Ln1FuXbq69a_7buSE2HNJSsJnzM5mVWRb"

_tunnel = None


def auth_token() -> str:
    return os.environ.get("NGROK_AUTHTOKEN", "").strip() or _DEFAULT_NGROK_AUTHTOKEN


def to_ws(https_url: str) -> str:
    u = https_url.rstrip("/")
    if u.startswith("https://"):
        return "wss://" + u[len("https://"):] + "/ws"
    if u.startswith("http://"):
        return "ws://" + u[len("http://"):] + "/ws"
    return u + "/ws"


def open_ngrok(port: int) -> str:
    """Open an HTTPS tunnel to localhost:`port`. Always returns a public https URL."""
    global _tunnel
    token = auth_token()

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

    try:
        _tunnel = ngrok.connect(addr=port, proto="http", **kwargs)
    except Exception as e:
        raise SystemExit(f"ngrok failed to start: {e}") from e

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
