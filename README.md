# Kupe ThinkSpark Live Agent

A terminal voice agent where **ThinkSpark controls the pipeline**. Soniox STT, Krutrim
LLM, Sarvam TTS — all streaming — with every start/stop decision made by
Kupe-ThinkSpark-Realtime-270M running locally on your machine.

```bash
./setup.sh              # venv + CUDA-matched torch + deps
./setup.sh --docker     # or run it containerised
```

### Docker

The container's own IP (`172.x`) is **not** reachable from your browser — you must
publish the port and connect via the host:

```bash
docker compose up --build          # port 8000 already published
```

or plain docker:

```bash
docker build -t kupe-thinkspark-agent .
docker run --rm --gpus all -p 8000:8000 --env-file .env \
  -v hf-cache:/cache/hf kupe-thinkspark-agent
```

Then paste `ws://localhost:8000/ws` into the UI — or `ws://<host-lan-ip>:8000/ws`
from another device. Without `-p 8000:8000` nothing outside the container can connect.

**Terminal:**
```bash
source .venv/bin/activate
python main.py
```

**Browser (any device on your LAN):**
```bash
python server.py            # prints the ws:// URL to paste
cd web && npm install && npm run dev
```

The server binds `0.0.0.0` and prints your LAN URL:

```
==============================================================
  Kupe ThinkSpark Live Agent — ready
==============================================================
  Paste this into the UI:

      ws://192.168.1.42:8000/ws

  local:   ws://127.0.0.1:8000/ws
  health:  http://192.168.1.42:8000/health
==============================================================
```

Open the web UI, paste that URL, hit Connect. Mic streams up, TTS streams back,
every flag and action renders live. No auth — local demo server.

Keys live in `.env` (gitignored) or `agent/keys.py`. Copy `.env.example` to start.

## The stack

| stage | provider | model | streaming |
|---|---|---|---|
| floor control | **ThinkSpark** (local) | Kupe-ThinkSpark-Realtime-270M | 80 ms frames |
| STT | Soniox | `stt-rt-v5` | websocket |
| LLM | Krutrim | `gemma-4-31b-it` | SSE deltas |
| TTS | Soniox | `tts-rt-v2` / voice **Mina** | websocket, chunk-by-chunk |

Audio in is denoised once with **RNNoise** before anything sees it.

Keys live in `agent/keys.py`, which is gitignored so it never reaches GitHub. Env vars of
the same name override it.

## How ThinkSpark drives the cascade

The whole point: ThinkSpark doesn't transcribe, think, or speak — it decides *when* the
other three are allowed to. That mapping lives in [`agent/policy.py`](agent/policy.py):

| flag | what the agent does |
|---|---|
| `LISTEN` | nothing — keep feeding STT |
| `HOLD` | **refuse to commit** — user paused mid-thought |
| `INCOMPLETE` | keep buffering, do not answer yet |
| `TURN_END` | commit STT → LLM → TTS |
| `PREFETCH_LLM` | start the LLM on the *partial* transcript, buffer the reply |
| `COMMIT_LLM` | speculation was right — play the buffered reply, zero LLM wait |
| `CANCEL_LLM` | speculation was wrong — abort the in-flight call |
| `BARGE_SOFT` | duck TTS volume, keep talking |
| `BARGE_HARD` | **stop TTS mid-word** |
| `CONTINUE` | agent may keep speaking |
| `SILENCE_BREAK` | dead air — play a short filler (rate-limited to 1 per 6 s) |

`HOLD` is the one that matters most in practice — it is what stops an agent cutting off
someone who is still thinking, which no endpointing timeout gets right.

### Agent state is fed back in

Barge-in only means something while the agent is talking. The agent state
(`IDLE` / `LLM_GEN` / `TTS_SPEAKING` / `TTS_DONE`) is pushed into the referee every
frame, along with what the agent is currently saying. Getting this wrong silently
degrades the model.

## Smoothing: why 240 ms, not 80 ms

Raw per-frame flags flip on noise — a bare mic session logs spurious `SILENCE_BREAK`
frames in pure silence. [`agent/smoothing.py`](agent/smoothing.py) applies a sliding
window before anything reaches the policy:

1. **Majority vote** over 3 frames (240 ms) — a flag must win its window to fire at all.
2. **Event latching** — once fired, a flag will not re-fire until the window moves to a
   different decision, so one real barge-in produces one `BARGE_HARD`, not fourteen.
3. **Urgent exception** — `BARGE_HARD` / `BARGE_SOFT` fire on a single vote, because
   waiting 240 ms to stop talking over someone *is* the failure you are preventing.

240 ms is not arbitrary: it is the ±3-frame collar the model was evaluated at
(ctrl macro-F1 **0.860** with the collar vs **0.770** without). You are smoothing at
exactly the tolerance the model is accurate to.

Tune with `--window`. Use `--raw` to print pre-smoothing flags alongside smoothed ones
and watch the noise get filtered in real time.

## Noise reduction

RNNoise runs **once, at the microphone**, before the fan-out — so a single pass cleans
the audio for both ThinkSpark and Soniox. Denoising per-consumer would double the CPU
cost for no gain.

This matters for ThinkSpark specifically: it decides on 80 ms of audio using energy and
f0 alongside the Mimi tokens, so a raised noise floor pushes borderline frames toward
false `SILENCE_BREAK` and false barge-in. Cleaning the input is cheaper than compensating
downstream — RNNoise and the 240 ms smoothing window attack the same problem from
opposite ends.

Disable with `--no-denoise` to A/B it. If `pyrnnoise` is not installed the agent still
runs, just noisier — it degrades to a passthrough and says so at boot.

## Turn commit: two triggers, not one

ThinkSpark decides *when* to speak — but it is not the only endpoint signal, and relying
on it alone will hang the conversation. Soniox emits `<end>` when it detects
end-of-utterance, and that is hard ground truth.

So a turn commits on **either**:

1. a smoothed `TURN_END` from ThinkSpark (fast path — fires before STT finalizes), or
2. Soniox `<end>` while the agent is idle (fallback — guarantees the turn completes)

Whichever lands first wins; the other is a no-op because committing resets the turn
state. This is the difference between an agent that sometimes never replies and one
that always does.

## Terminal output

```
22:41:02  boot           ThinkSpark ready on mps
22:41:03  boot           Soniox STT connected
22:41:03  ready          listening — Ctrl+C to stop
22:41:07  stt            mujhe apna balance
22:41:08  FLAG   PREFETCH_LLM     41.2 ms
22:41:08  PREFETCH       speculating on 'mujhe apna balance'
22:41:09  stt-final      mujhe apna balance chahiye
22:41:09  FLAG   TURN_END         38.7 ms
22:41:09  TURN_END       used speculative reply (0 ms LLM wait)
22:41:09  tts            speaking: 'Aapka balance chaar hazaar do sau rupaye hai.'
22:41:11  FLAG   BARGE_HARD       36.1 ms
22:41:11  BARGE_HARD     stopped TTS mid-utterance
22:41:11  tts            playback cut
```

Ctrl+C prints decode p50/p95 and a flag histogram.

## Latency

ThinkSpark decodes in ~3 ms on a datacenter GPU and ~25-50 ms on Apple Silicon, against
an 80 ms frame budget. The SDK tunes the backend per device: TF32 + cuDNN autotune +
bf16 weights on CUDA (3060/4060/3090/4090/5090, L4, H100, RTX 6000), capped thread count
on CPU, fp32 on MPS. A warmup pass runs at load so the first real frame is not the slow
one.

If p95 creeps over 80 ms on your machine, frames queue and the agent drifts behind live
audio. Watch the summary line; if it does, run on CUDA or raise `--window`.

## Layout

```
main.py               terminal UI + entrypoint
agent/config.py       endpoints, model ids, key loading
agent/keys.py         API keys (gitignored)
agent/denoise.py      RNNoise, applied once at the mic
agent/smoothing.py    sliding-window vote + event latching
agent/policy.py       flag -> action mapping (the interesting file)
agent/providers.py    Soniox STT / Krutrim LLM / Sarvam TTS clients
agent/live.py         mic fan-out, concurrent loops, playback
agent/session.py      one browser connection (websocket in, TTS out)
server.py             FastAPI websocket server + LAN URL banner
web/                  React UI — paste the URL, talk
.env.example          key template
```

## Reachability

`python server.py` binds `:8000`. If `NGROK_AUTHTOKEN` is set it also opens ngrok
and prints a `wss://….ngrok-free.app/ws` URL — that is what you paste into the UI.
Works on Colab, Kaggle, and RunPod (no inbound ports, no nginx).

Direct IP still works on Vast / a box that publishes 8000: `ws://<public-ip>:8000/ws`.
`--no-ngrok` skips the tunnel.

## Model weights

Loads the finetuned checkpoint directly:

```
anuj-inavlabs/kupe-thinkspark-audio-270m
  phase2/runs/20260902-103400/step5500
```

That folder carries its own `config.json` and tokenizer, so the backbone is built from
config and the weights come from `model.pt` — **the gated `google/gemma-3-270m` repo is
never fetched.** Earlier versions downloaded it only to overwrite every weight seconds
later, which also meant a 401 for anyone without Gemma access.

Pin a different checkpoint in `agent/config.py` (`TS_REPO` / `TS_SUBFOLDER`).

## Status

Written against the verified provider catalog and the shipped `kupe` SDK, but **not yet
run end-to-end against live audio and live provider keys**. Expect to shake out protocol
details on first run — particularly the Soniox STT token schema and Sarvam's response
field names. Run it, paste the traceback, and it gets fixed.
