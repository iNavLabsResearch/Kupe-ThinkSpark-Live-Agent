"""Public HTTPS URL for Colab / Kaggle / RunPod.

Ngrok's free plan intercepts WebSocket upgrades (health works, /ws does not).
Cloudflare quick tunnels do upgrade WebSockets, so that is the default.
"""

from __future__ import annotations

import os
import platform
import re
import stat
import subprocess
import threading
import time
import urllib.request

_cf_proc: subprocess.Popen | None = None
_cf_bin = "/tmp/kupe-cloudflared"


def to_ws(https_url: str) -> str:
    u = https_url.rstrip("/")
    if u.startswith("https://"):
        return "wss://" + u[len("https://"):] + "/ws"
    if u.startswith("http://"):
        return "ws://" + u[len("http://"):] + "/ws"
    return u + "/ws"


def _ensure_cloudflared() -> str:
    for name in ("cloudflared", _cf_bin):
        if name != _cf_bin and _which(name):
            return name
        if name == _cf_bin and os.path.isfile(_cf_bin) and os.access(_cf_bin, os.X_OK):
            return _cf_bin

    machine = platform.machine().lower()
    arch = "arm64" if ("arm" in machine or "aarch" in machine) else "amd64"
    url = (
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        f"cloudflared-linux-{arch}"
    )
    print(f"==> downloading cloudflared ({arch})")
    urllib.request.urlretrieve(url, _cf_bin)
    os.chmod(_cf_bin, os.stat(_cf_bin).st_mode | stat.S_IEXEC)
    return _cf_bin


def _which(cmd: str) -> str | None:
    from shutil import which
    return which(cmd)


def open_tunnel(port: int) -> str:
    """HTTPS URL that can carry WebSockets. Cloudflare first; ngrok is HTTP-only on free."""
    try:
        url = _open_cloudflare(port)
        print(f"==> cloudflare tunnel  {url}")
        return url
    except Exception as e:
        print(f"==> cloudflare tunnel failed ({e}) — WebSockets will not work on free ngrok")
        raise SystemExit(
            "Need a WebSocket-capable tunnel. Install cloudflared or allow GitHub downloads.\n"
            f"  last error: {e}"
        ) from e


def _open_cloudflare(port: int) -> str:
    global _cf_proc
    bin_path = _ensure_cloudflared()
    _cf_proc = subprocess.Popen(
        [bin_path, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    found: list[str] = []
    done = threading.Event()

    def _read():
        assert _cf_proc and _cf_proc.stdout
        for line in _cf_proc.stdout:
            line = line.strip()
            if line:
                print(f"    cloudflared: {line}")
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line, re.I)
            if m and not found:
                found.append(m.group(0))
                done.set()

    threading.Thread(target=_read, daemon=True).start()
    if not done.wait(timeout=45):
        close_tunnel()
        raise TimeoutError("cloudflared did not print a trycloudflare.com URL in 45s")
    time.sleep(0.4)
    return found[0]


def close_tunnel() -> None:
    global _cf_proc
    if _cf_proc and _cf_proc.poll() is None:
        _cf_proc.terminate()
        try:
            _cf_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _cf_proc.kill()
    _cf_proc = None


# Back-compat names used by older server.py copies on the pod
open_ngrok = open_tunnel
close_ngrok = close_tunnel
