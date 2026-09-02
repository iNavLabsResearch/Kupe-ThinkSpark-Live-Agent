"""One browser connection = one WebSession.

Audio arrives over a websocket; ThinkSpark + STT + Policy run in FloorAgent;
TTS PCM goes back down the same socket.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np

from agent.orchestrator import FloorAgent, MIC_RATE
from agent.policy import AgentState


class _WsUI:
    """Mirrors the terminal UI's interface, but emits JSON to the browser."""

    def __init__(self, ws, loop):
        self.ws = ws
        self.loop = loop

    def _send(self, payload: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            self.ws.send_text(json.dumps(payload)), self.loop
        )

    def log(self, kind: str, detail: str = "", style: str = "") -> None:
        self._send({"type": "log", "kind": kind, "detail": detail})
        if kind in ("stt", "stt-final") and detail:
            self._send({"type": "stt", "text": detail, "final": kind == "stt-final"})

    def frame(self, flag: str, latency_ms: float, raw: bool = False) -> None:
        self._send({"type": "flag", "flag": flag,
                    "latency_ms": round(latency_ms, 2), "raw": raw})


class WebSession:
    def __init__(self, ws, referee, keys, window: int = 3, denoise: bool = True):
        self.ws = ws
        loop = asyncio.get_event_loop()
        self.ui = _WsUI(ws, loop)
        self.floor = FloorAgent(referee, keys, ui=self.ui, window=window, denoise=denoise)
        self._closed = False

    async def run(self) -> None:
        await self.floor.start()
        await asyncio.gather(
            self._recv_loop(),
            self._playback_loop(),
            self.floor.run_until_closed(),
        )

    async def _recv_loop(self) -> None:
        while not self._closed:
            msg = await self.ws.receive()
            if msg.get("type") == "websocket.disconnect":
                await self.close()
                return
            if (data := msg.get("bytes")) is not None:
                pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self.floor.push_audio(pcm, MIC_RATE)
            elif (txt := msg.get("text")) is not None:
                try:
                    if json.loads(txt).get("type") == "reset":
                        self.floor.reset_user_stt()
                        self.floor.smoother.reset()
                        self.floor.policy.state = AgentState.IDLE
                        self.floor.agent_text = ""
                        self.floor._refresh_context()
                except Exception:
                    pass

    async def _playback_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._closed:
            pcm = await loop.run_in_executor(None, self.floor.playback_q.get)
            if pcm is None or self._closed:
                return
            if isinstance(pcm, np.ndarray):
                await self.ws.send_bytes(np.ascontiguousarray(pcm).tobytes())
            else:
                await self.ws.send_bytes(pcm)

    async def close(self) -> None:
        self._closed = True
        try:
            self.floor.playback_q.put_nowait(None)
        except Exception:
            pass
        await self.floor.close()
