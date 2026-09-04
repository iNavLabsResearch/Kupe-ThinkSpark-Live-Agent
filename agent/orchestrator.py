"""ThinkSpark floor-control loop shared by Gradio, websocket, and the terminal.

Audio is framed at 80 ms. ThinkSpark is the only VAD/endpoint/barge referee.
AssemblyAI streaming STT runs in parallel (not as a gate). LLM/TTS fire only
when Policy says so. TTS audio is never fed to STT (echo mute + hangover).
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from agent import config
from agent.denoise import Denoiser
from agent.policy import Action, AgentState, Policy
from agent.providers import KrutrimLLM, SonioxTTS, make_stt
from agent.smoothing import FlagSmoother

MIC_RATE = 24_000
STT_RATE = 16_000
FRAME_MS = 80
FRAME_SAMPLES = MIC_RATE * FRAME_MS // 1000
TABLE_MAX = 80
TTS_RATE = config.TTS_SAMPLE_RATE

# Placeholder / back-channel strings that must never be treated as the user's turn.
_JUNK_STT = frozenset({
    "please wait", "please wait.", "pls wait",
    "i see", "i see.", "i see...",
    "mm-hmm", "mm hmm", "mhm", "uh huh", "uh-huh", "uh-huh.",
})


def _norm_utt(text: str) -> str:
    return " ".join((text or "").lower().strip().strip(".!?,;:").split())


def _is_junk_stt(text: str) -> bool:
    n = _norm_utt(text)
    return (not n) or n in _JUNK_STT or n.startswith("please wait")


def _f32_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x
    n = max(1, int(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x), n, endpoint=False),
                     np.arange(len(x)), x).astype(np.float32)


def _mono_f32(data) -> np.ndarray:
    x = np.asarray(data)
    if x.size == 0:
        return x.astype(np.float32)
    if x.ndim == 2:
        x = x[0] if x.shape[0] <= 8 else x.mean(axis=-1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / max(1, np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float32)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 1.5:
            x = x / 32768.0
    return np.clip(x, -1.0, 1.0)


@dataclass
class FrameRow:
    time: str
    flag: str
    spoken: str
    stt: str
    context: str
    output: str
    ms: float

    def as_list(self) -> list[str]:
        return [
            self.time, self.flag, self.spoken, self.stt,
            self.context, self.output, f"{self.ms:.2f}",
        ]


class PrintUI:
    def log(self, kind: str, detail: str = "", style: str = "") -> None:
        print(f"[{kind}] {detail}", flush=True)

    def frame(self, flag: str, latency_ms: float, raw: bool = False) -> None:
        pass


class FloorAgent:
    """One conversation: 80 ms frames -> ThinkSpark + STT -> Policy -> TTS queue."""

    def __init__(self, referee, keys, ui=None, window: int = 3, denoise: bool = True):
        self.referee = referee
        self.ui = ui or PrintUI()
        self.stt = make_stt(config.STT_PROVIDER, keys, sample_rate=STT_RATE)
        self.llm = KrutrimLLM(keys.llm)
        self.tts = SonioxTTS(keys.tts)
        self.denoiser = Denoiser(sample_rate=MIC_RATE, enabled=denoise)
        self.smoother = FlagSmoother(window=window)
        self.policy = Policy(self)

        self.agent_text = ""
        self.last_output = ""
        self.frames = 0
        self.decode_ms: list[float] = []
        self.rows: deque[FrameRow] = deque(maxlen=TABLE_MAX)

        self._in_buf = np.zeros(0, dtype=np.float32)
        # SMALL model-audio queue: if the referee falls behind (e.g. MPS), drop the OLDEST
        # frames and stay near real-time instead of building a multi-second backlog. 200
        # frames = 16 s of lag; 8 frames caps it at ~0.6 s. STT keeps its own full queue.
        self._audio_q: queue.Queue = queue.Queue(maxsize=8)
        # set true by an external feeder while REAL (non-silence) audio is being pushed,
        # so the UI can colour the frames the model is actually "hearing".
        self._user_audio_active = False
        self._stt_q: queue.Queue = queue.Queue(maxsize=200)
        self.playback_q: queue.Queue = queue.Queue(maxsize=400)

        self._speaking = False
        self._volume = 1.0
        self._stop_playback = threading.Event()
        self._closed = False
        self._turn_busy = False
        # True: also commit a turn on AssemblyAI end_of_turn (safety fallback).
        # False: ThinkSpark TURN_END is the ONLY endpoint (pure floor-controller mode).
        self._use_stt_endpoint = True
        self._tasks: list[asyncio.Task] = []
        self._dirty = True
        self._echo_until = 0.0
        self._play_until = 0.0
        self._last_tts_text = ""
        self._user_partial = ""
        self._user_final = ""
        self._stt_status = "connecting"

    # -- context -------------------------------------------------------- #
    def _refresh_context(self) -> None:
        # Feed the model ONLY the finalized transcript, never live partials: partials
        # churn every few hundred ms and the model should see stable, complete text (you
        # asked for this). The next audio frames after a final carry the committed text.
        self.referee.set_context(
            agent_text=self.agent_text,
            stt_partial=self._user_final,
        )

    def _context_str(self) -> str:
        state = self.policy.state.value
        text = (self.agent_text or "").strip()
        if text:
            return f"{state} | {text[:80]}"
        return state

    def _stt_str(self) -> str:
        return (self._user_partial or self._user_final or "").strip()

    def _looks_like_echo(self, text: str) -> bool:
        n = _norm_utt(text)
        if not n:
            return True
        if n in _JUNK_STT or n.startswith("please wait"):
            return True
        last = _norm_utt(self._last_tts_text)
        if last and (n == last or n in last or last in n):
            return True
        return False

    # -- audio out ------------------------------------------------------ #
    def duck(self) -> None:
        self._volume = 0.35

    def stop_speaking(self) -> None:
        self._stop_playback.set()
        self._speaking = False
        self._play_until = 0.0
        self._echo_until = time.time() + 0.25
        self._volume = 1.0
        while True:
            try:
                self.playback_q.get_nowait()
            except queue.Empty:
                break

    def _enqueue_pcm(self, pcm: bytes) -> None:
        a = np.frombuffer(pcm, dtype=np.int16)
        if self._volume != 1.0:
            a = (a.astype(np.float32) * self._volume).astype(np.int16)
        now = time.time()
        start = max(now, self._play_until)
        self._play_until = start + (len(a) / float(TTS_RATE))
        try:
            self.playback_q.put_nowait(a)
        except queue.Full:
            try:
                self.playback_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.playback_q.put_nowait(a)
            except queue.Full:
                pass

    async def speak(self, text: str, filler: bool = False) -> None:
        if not text.strip() or self._closed:
            return
        self.policy.state = AgentState.TTS_SPEAKING
        self._speaking = True
        self._stop_playback.clear()
        self._volume = 1.0
        self.agent_text = text
        self._last_tts_text = text
        self._refresh_context()
        self.ui.log("tts", f"speaking: {text!r}")
        self.last_output = f"TTS: {text}"
        self._dirty = True
        try:
            async for pcm in self.tts.stream(text):
                if self._stop_playback.is_set() or self._closed:
                    self.ui.log("tts", "playback cut")
                    break
                self._enqueue_pcm(pcm)
        except Exception as e:
            self.ui.log("error", f"tts failed: {e}")
        self._speaking = False
        # TTS bytes arrive faster than realtime — keep the floor until speakers finish.
        wait = max(0.0, self._play_until - time.time())
        if wait and not self._stop_playback.is_set() and not self._closed:
            await asyncio.sleep(wait)
        self._echo_until = time.time() + 0.40
        self.agent_text = ""
        self.policy.state = AgentState.TTS_DONE if not filler else AgentState.IDLE
        self._refresh_context()
        self._dirty = True

    # -- audio in ------------------------------------------------------- #
    def push_audio(self, pcm, sample_rate: int) -> None:
        if self._closed:
            return
        x = _mono_f32(pcm)
        x = _resample(x, int(sample_rate), MIC_RATE)
        if x.size == 0:
            return
        self._in_buf = np.concatenate([self._in_buf, x]) if self._in_buf.size else x
        while len(self._in_buf) >= FRAME_SAMPLES:
            frame = self._in_buf[:FRAME_SAMPLES]
            self._in_buf = self._in_buf[FRAME_SAMPLES:]
            # RNNoise ONCE at the mic, before the fan-out (README: one pass cleans audio
            # for both ThinkSpark and STT). Runs on the mic thread — never gated by the
            # model, so denoise + STT stay real-time even if the referee falls behind.
            frame = self.denoiser(frame)
            # feed STT DIRECTLY here (parallel path). Previously STT audio was produced
            # inside _step_chunk, i.e. gated by the model's per-frame speed — on a slow
            # device the referee loop lagged and STARVED the STT stream (30 s of speech
            # arriving as "Look."). STT is a parallel pass, never a model-gated one.
            self._feed_stt(frame)
            try:
                self._audio_q.put_nowait(frame)
            except queue.Full:
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._audio_q.put_nowait(frame)
                except queue.Full:
                    pass

    def _feed_stt(self, frame: np.ndarray) -> None:
        """Enqueue one (already-denoised) 80 ms frame to the STT sender, echo-muted."""
        pcm = _f32_to_pcm16(_resample(frame, MIC_RATE, STT_RATE))
        if self._echo_active():
            pcm = b"\x00\x00" * (len(pcm) // 2)
        try:
            self._stt_q.put_nowait(pcm)
        except queue.Full:
            pass

    def _echo_active(self) -> bool:
        """Mute STT while audio is actually coming out of the speakers, not just while
        the TTS websocket is open. Generation finishes in ~200 ms; playback lasts seconds."""
        now = time.time()
        return (
            self._speaking
            or self.policy.state is AgentState.TTS_SPEAKING
            or now < self._play_until
            or now < self._echo_until
        )

    def _accept_stt(self, text: str, is_final: bool) -> bool:
        """Keep ThinkSpark / the table on the user's words, never the agent's."""
        text = (text or "").strip()
        if not text or self._echo_active() or self._looks_like_echo(text):
            self.stt.partial = self._user_partial
            self.stt.final = self._user_final
            return False
        self._user_partial = text
        self.stt.partial = text
        if is_final:
            self._user_final = text
            self.stt.final = text
        self._refresh_context()
        self._stt_status = "live"
        self._dirty = True
        return True

    def _step_chunk(self, chunk: np.ndarray) -> list:
        # chunk is already denoised + already fed to STT in push_audio (parallel path);
        # here we only run the referee. So a slow model can never starve STT.
        self._refresh_context()
        state = self.policy.state.value
        if self._echo_active() and state == AgentState.IDLE.value:
            state = AgentState.TTS_SPEAKING.value
        return list(self.referee.stream(
            source=[chunk],
            sample_rate=MIC_RATE,
            agent_state=state,
        ))

    def _record(self, flag: str, spoken: str, ms: float, output: str = "") -> FrameRow:
        row = FrameRow(
            time=time.strftime("%H:%M:%S"),
            flag=flag,
            spoken=spoken or "",
            stt=self._stt_str(),
            context=self._context_str(),
            output=output,
            ms=float(ms),
        )
        self.rows.appendleft(row)
        self._dirty = True
        return row

    # -- decisions ------------------------------------------------------ #
    async def _on_decision(self, decision) -> None:
        """One raw 80 ms flag in. Smooth it, and act only on the smoothed decision.

        The raw flag is decided cheaply every frame by the streaming referee. We never
        act on a bare 80 ms flag — a single frame flips on noise. The smoother's sliding
        window + event-latch collapses the stream into one stable decision per real
        event (Section 10 collar), and only THAT reaches the policy.
        """
        self.frames += 1
        self.decode_ms.append(decision.latency_ms)
        raw = decision.flag
        self.ui.frame(raw, decision.latency_ms, raw=True)
        self._record(raw, "", decision.latency_ms)

        smoothed = self.smoother.push(raw)
        if smoothed is None:
            return
        self.ui.frame(smoothed, decision.latency_ms, raw=False)
        asyncio.create_task(self._policy_task(smoothed))

    async def gen_spoken(self) -> str:
        """Decode a spoken back-channel OFF the event loop, only when the policy asks.

        This is the spoken head — deliberately NOT run every frame (that was the old
        latency bug). Never runs while the agent is, or is about to be, audible: that
        would echo the agent's own voice back into STT.
        """
        if self._echo_active() or self._closed or self._speaking:
            return ""
        # old kupe SDKs have no spoken head — degrade silently (LLM re-open still works)
        if not hasattr(self.referee, "generate_spoken"):
            return ""
        loop = asyncio.get_running_loop()
        state = self.policy.state.value
        try:
            text = await loop.run_in_executor(None, self.referee.generate_spoken, state)
        except Exception as e:
            self.ui.log("error", f"spoken: {e}")
            return ""
        return (text or "").strip()

    async def _policy_task(self, flag: str) -> None:
        urgent = flag in {"BARGE_HARD", "BARGE_SOFT", "CANCEL_LLM"}
        idle = flag in {"LISTEN", "HOLD", "CONTINUE", "INCOMPLETE", "SILENCE_BREAK"}
        # A turn-committing flag (TURN_END / PREFETCH / COMMIT) must not stack on an
        # in-flight turn; idle/urgent flags are always allowed through.
        if not urgent and not idle and self._turn_busy:
            return
        if not urgent and not idle:
            self._turn_busy = True
        try:
            action = await self.policy.handle(flag)
        except Exception as e:
            self.ui.log("error", f"policy {flag}: {e}")
            return
        finally:
            if not urgent and not idle:
                self._turn_busy = False
        if action:
            self._note_action(action)

    def _note_action(self, action: Action) -> None:
        self.last_output = f"{action.kind}: {action.detail}"
        self.ui.log(action.kind, action.detail)
        self._record(action.kind, "", 0.0, output=self.last_output)

    async def _referee_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._closed:
            chunk = await loop.run_in_executor(None, self._audio_q.get)
            if chunk is None:
                return
            try:
                decisions = await loop.run_in_executor(None, self._step_chunk, chunk)
            except Exception as e:
                self.ui.log("error", f"ThinkSpark: {e}")
                continue
            for d in decisions:
                await self._on_decision(d)

    async def _stt_send_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._closed:
            pcm = await loop.run_in_executor(None, self._stt_q.get)
            if pcm is None:
                return
            try:
                await self.stt.send_audio(pcm)
            except Exception as e:
                self.ui.log("error", f"stt send: {e}")

    async def _stt_recv_loop(self) -> None:
        while not self._closed:
            try:
                async for text, is_final in self.stt.transcripts():
                    if self._closed:
                        return
                    if not self._accept_stt(text, is_final):
                        continue
                    kind = "stt-final" if is_final else "stt"
                    self.ui.log(kind, text)
                    # AssemblyAI's own end_of_turn is a SEPARATE endpoint from ThinkSpark.
                    # When _use_stt_endpoint is False, we ignore it entirely — ThinkSpark's
                    # TURN_END is the ONLY thing that commits a turn (the floor controller
                    # is the endpoint, per the guide). STT partials/finals are still used
                    # as context + transcript, just never as a turn trigger.
                    if self._use_stt_endpoint and is_final and not _is_junk_stt(text):
                        if self._turn_busy:
                            continue
                        self._turn_busy = True
                        try:
                            action = await self.policy.commit_from_stt()
                        finally:
                            self._turn_busy = False
                        if action:
                            self._note_action(action)
                if self._closed:
                    return
            except Exception as e:
                if self._closed:
                    return
                self._stt_status = f"error: {e}"
                self.ui.log("error", f"stt recv: {e}")
            await asyncio.sleep(0.8)
            if self._closed:
                return
            try:
                await self.stt.close()
                await self.stt.connect()
                self._stt_status = "reconnected"
                self.ui.log("boot", "AssemblyAI STT reconnected")
            except Exception as e:
                self._stt_status = f"reconnect failed: {e}"
                self.ui.log("error", f"stt reconnect: {e}")

    async def start(self) -> None:
        try:
            await self.stt.connect()
            self._stt_status = "live"
        except Exception as e:
            self._stt_status = f"connect failed: {e}"
            self.ui.log("error", f"stt connect: {e}")
            raise
        _stt = ("Soniox " + config.SONIOX_STT_MODEL if config.STT_PROVIDER == "soniox"
                else "AssemblyAI " + config.STT_MODEL)
        self.ui.log(
            "boot",
            f"ready · {getattr(self.referee, 'device', '?')} · STT {_stt} · "
            f"{config.LLM_MODEL} · {config.TTS_MODEL}/{config.TTS_VOICE}",
        )
        self._tasks = [
            asyncio.create_task(self._referee_loop()),
            asyncio.create_task(self._stt_send_loop()),
            asyncio.create_task(self._stt_recv_loop()),
        ]

    async def run_until_closed(self) -> None:
        if not self._tasks:
            await self.start()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def close(self) -> None:
        self._closed = True
        self._stop_playback.set()
        for q in (self._audio_q, self._stt_q, self.playback_q):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for t in self._tasks:
            t.cancel()
        try:
            await self.stt.close()
        except Exception:
            pass

    # -- UI snapshot ---------------------------------------------------- #
    def snapshot(self) -> tuple[str, str, str, str, list[list[str]]]:
        p50 = 0.0
        head = 80.0
        if self.decode_ms:
            w = sorted(self.decode_ms[-200:])
            p50 = w[len(w) // 2]
            head = 80.0 - (sum(w) / len(w))
        flag = self.rows[0].flag if self.rows else "—"
        stats = (
            f"{self.frames} frames · p50 {p50:.1f} ms · "
            f"headroom {head:.1f} ms (of 80 ms)"
        )
        return flag, stats, self._stt_display(), self._context_str(), [
            r.as_list() for r in self.rows
        ]

    def _stt_display(self) -> str:
        text = self._stt_str()
        if text:
            return text
        if self._stt_status in ("", "live", "reconnected"):
            return "…"
        return self._stt_status or "…"

    def reset_user_stt(self) -> None:
        self._user_partial = ""
        self._user_final = ""
        self.stt.reset_turn()
        # a committed turn is a floor boundary: clear the referee's rolling audio history
        # + KV cache + Mimi context so the next turn starts clean (and the cache stays
        # bounded well inside the backbone's sliding window).
        try:
            self.referee.reset()
        except Exception:
            pass
        self._refresh_context()
