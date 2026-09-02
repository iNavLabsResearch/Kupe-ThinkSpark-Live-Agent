# Kupe ThinkSpark Live Agent

A terminal voice agent where **ThinkSpark controls the pipeline**. Soniox STT, Krutrim
LLM, Sarvam TTS — all streaming — with every start/stop decision made by
Kupe-ThinkSpark-Realtime-270M running locally on your machine.

```bash
pip install -r requirements.txt
export SUPABASE_DB_URL='postgres://...'    # or the three *_API_KEY vars
python main.py
```

## The stack

| stage | provider | model | streaming |
|---|---|---|---|
| floor control | **ThinkSpark** (local) | Kupe-ThinkSpark-Realtime-270M | 80 ms frames |
| STT | Soniox | `stt-rt-v5` | websocket |
| LLM | Krutrim | `gemma-4-31b-it` | SSE deltas |
| TTS | Sarvam | `bulbul:v3` / voice `ritu` | chunked playback |

Keys load from the Kupe `provider_api_keys` table via `SUPABASE_DB_URL`, or from
`KRUTRIM_API_KEY` / `SONIOX_API_KEY` / `SARVAM_API_KEY`. Nothing is written to disk.

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

ThinkSpark decodes in ~3 ms on a datacenter GPU and ~30-50 ms on Apple Silicon. Either
way it fits inside the 80 ms frame budget — the remainder is your headroom, printed in
the summary. The model is never the bottleneck; STT and LLM round trips are.

## Layout

```
main.py               terminal UI + entrypoint
agent/config.py       provider ids, endpoints, key loading
agent/smoothing.py    sliding-window vote + event latching
agent/policy.py       flag -> action mapping (the interesting file)
agent/providers.py    Soniox STT / Krutrim LLM / Sarvam TTS clients
agent/live.py         mic fan-out, concurrent loops, playback
```

## Status

Written against the verified provider catalog and the shipped `kupe` SDK, but **not yet
run end-to-end against live audio and live provider keys**. Expect to shake out protocol
details on first run — particularly the Soniox STT token schema and Sarvam's response
field names. Run it, paste the traceback, and it gets fixed.
