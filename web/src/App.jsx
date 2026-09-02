import React, { useEffect, useRef, useState } from "react";

const FLAG_COLOR = {
  LISTEN: "#8b8b8b",
  HOLD: "#6fa8ff",
  INCOMPLETE: "#e6d47a",
  TURN_END: "#4ade80",
  BARGE_SOFT: "#fb923c",
  BARGE_HARD: "#f87171",
  CONTINUE: "#a3a3a3",
  PREFETCH_LLM: "#e879f9",
  COMMIT_LLM: "#22d3ee",
  CANCEL_LLM: "#facc15",
  SILENCE_BREAK: "#60a5fa",
};

const MIC_RATE = 24000;

export default function App() {
  const [url, setUrl] = useState(() => {
    const saved = localStorage.getItem("kupe_ws_url") || "";
    // RunPod's HTTPS proxy usually cannot upgrade WebSockets. Keep other wss://
    // (nginx TLS, Cloudflare, Vast direct IP, etc.).
    if (saved.includes("proxy.runpod.net")) {
      localStorage.removeItem("kupe_ws_url");
      return "ws://127.0.0.1:8000/ws";
    }
    return saved || "ws://127.0.0.1:8000/ws";
  });
  const [status, setStatus] = useState("idle");
  const [events, setEvents] = useState([]);
  const [transcript, setTranscript] = useState("");
  const [flag, setFlag] = useState(null);
  const [stats, setStats] = useState({ frames: 0, p50: 0 });

  const wsRef = useRef(null);
  const audioCtxRef = useRef(null);
  const streamRef = useRef(null);
  const playHeadRef = useRef(0);
  const latRef = useRef([]);

  const push = (e) =>
    setEvents((prev) => [...prev.slice(-160), { ...e, t: new Date().toLocaleTimeString() }]);

  async function connect() {
    setStatus("connecting");
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = async () => {
      localStorage.setItem("kupe_ws_url", url);
      setStatus("connected");
      await startMic(ws);
    };

    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) return playPcm(ev.data);
      const m = JSON.parse(ev.data);
      if (m.type === "flag") {
        if (m.raw) {
          latRef.current.push(m.latency_ms);
          if (latRef.current.length > 300) latRef.current.shift();
          const s = [...latRef.current].sort((a, b) => a - b);
          setStats((p) => ({
            frames: p.frames + 1,
            p50: s[Math.floor(s.length / 2)] || 0,
          }));
        } else {
          setFlag(m.flag);
          push({ kind: m.flag, detail: `${m.latency_ms} ms`, flag: true });
        }
      } else if (m.type === "stt") {
        setTranscript(m.text);
        if (m.final) push({ kind: "stt-final", detail: m.text });
      } else if (m.type === "action") {
        push({ kind: m.kind, detail: m.detail, action: true });
      } else if (m.type === "tts_start") {
        push({ kind: "tts", detail: m.text });
      } else if (m.type === "log") {
        push({ kind: m.kind, detail: m.detail });
      }
    };

    ws.onclose = () => { setStatus("closed"); stopMic(); };
    ws.onerror = () => setStatus("error");
  }

  async function startMic(ws) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false },
    });
    streamRef.current = stream;

    const ctx = new AudioContext({ sampleRate: MIC_RATE });
    audioCtxRef.current = ctx;
    const src = ctx.createMediaStreamSource(stream);

    // 2048 frames @ 24 kHz ≈ 85 ms, close to ThinkSpark's 80 ms frame
    const node = ctx.createScriptProcessor(2048, 1, 1);
    node.onaudioprocess = (e) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const f32 = e.inputBuffer.getChannelData(0);
      const i16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++)
        i16[i] = Math.max(-1, Math.min(1, f32[i])) * 32767;
      ws.send(i16.buffer);
    };
    src.connect(node);
    node.connect(ctx.destination);
  }

  function stopMic() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    audioCtxRef.current?.close();
    streamRef.current = null;
    audioCtxRef.current = null;
  }

  function playPcm(buf) {
    const ctx = audioCtxRef.current;
    if (!ctx) return;
    const i16 = new Int16Array(buf);
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;

    const b = ctx.createBuffer(1, f32.length, MIC_RATE);
    b.copyToChannel(f32, 0);
    const s = ctx.createBufferSource();
    s.buffer = b;
    s.connect(ctx.destination);
    // schedule back-to-back so streamed chunks play gaplessly
    const now = ctx.currentTime;
    const at = Math.max(now, playHeadRef.current);
    s.start(at);
    playHeadRef.current = at + b.duration;
  }

  function disconnect() {
    wsRef.current?.close();
    stopMic();
  }

  useEffect(() => () => disconnect(), []);

  return (
    <div className="app">
      <header>
        <h1>Kupe ThinkSpark <span>Live Agent</span></h1>
        <div className="conn">
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="ws://HOST:8000/ws"
            disabled={status === "connected"}
          />
          {status === "connected" ? (
            <button className="stop" onClick={disconnect}>Disconnect</button>
          ) : (
            <button onClick={connect}>Connect</button>
          )}
          <span className={`dot ${status}`} title={status} />
        </div>
      </header>

      <section className="hero">
        <div className="flag" style={{ color: FLAG_COLOR[flag] || "#555" }}>
          {flag || "—"}
        </div>
        <div className="meta">
          {stats.frames} frames · p50 {stats.p50.toFixed(1)} ms · budget 80 ms
        </div>
        <div className="transcript">{transcript || "…"}</div>
      </section>

      <section className="log">
        {events.map((e, i) => (
          <div key={i} className={`row ${e.action ? "action" : ""}`}>
            <span className="time">{e.t}</span>
            <span
              className="kind"
              style={e.flag ? { color: FLAG_COLOR[e.kind] || "#ccc" } : undefined}
            >
              {e.kind}
            </span>
            <span className="detail">{e.detail}</span>
          </div>
        ))}
      </section>
    </div>
  );
}
