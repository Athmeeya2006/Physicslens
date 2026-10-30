"use client";

import { useEffect, useRef, useState } from "react";
import { Tex } from "@/components/Equation";
import { VoiceNotes } from "@/components/VoiceNotes";
import { accessCode, API_BASE, useBackend, useHealth, formatUnit } from "@/lib/api";

const FPS = 25;
const MAX_WIDTH = 640; // SAM bills per frame, not per pixel; smaller frames just keep the upload light
const TRAIL = 25;

type Point = [number, number, number];

interface LiveParam { name: string; latex: string; unit: string; value: number; spread: number | null }
interface FitMsg {
  type: "fit";
  status: "accepted" | "ambiguous" | "no_fit";
  reason: string;
  samples?: number;
  window_s?: number;
  equation?: { model_id: string; title: string; latex: string };
  parameters?: LiveParam[];
  ghost?: Point[];
}
interface Notice { level: string; message: string }

function wsUrl(): string {
  return API_BASE.replace(/^http/, "ws") + "/ws/live";
}

const LIVE_FAMILIES: [string, string][] = [
  ["pendulum", "Swinging"], ["oscillator", "On a spring"], ["projectile", "Thrown or falling"], ["line", "Rolling or sliding"],
];

export default function LivePage() {
  const online = useBackend();
  const health = useHealth();
  // Live sessions cost money per minute: when the server doesn't offer them, say so before anyone tries.
  const judgesOnly = !!health && !health.can_go_live;
  const video = useRef<HTMLVideoElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const socket = useRef<WebSocket | null>(null);
  const running = useRef(false);
  const sentFrames = useRef(0);
  const track = useRef<Map<number, [number, number]>>(new Map());
  const ghost = useRef<Map<number, [number, number]>>(new Map());
  const objectSize = useRef<number>(30);

  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [status, setStatus] = useState("Point the camera at something moving, side-on, then press Start.");
  const [scene, setScene] = useState<string | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const [fit, setFit] = useState<FitMsg | null>(null);
  const [notices, setNotices] = useState<Notice[]>([]);
  const [lengthCm, setLengthCm] = useState("");
  const [sizeCm, setSizeCm] = useState("");
  const [live, setLive] = useState(false);
  const [regime, setRegime] = useState<string | null>(null);
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [cue, setCue] = useState<CueMsg | null>(null);
  const [targetDeg, setTargetDeg] = useState("30");
  const audio = useRef<AudioContext | null>(null);
  const [now, setNow] = useState(0);
  // Session clock: seconds since the first frame's capture time, the same clock the server's predictions use.
  const firstCapture = useRef<number | null>(null);

  useEffect(() => () => stop(), []);
  // Ticks one period apart, then a higher beep on each push: people anticipate a beat far better than they react to
  // a beep. Times are on the session clock (the frames' capture times), corrected for the device's audio latency.
  const scheduleCue = (c: CueMsg) => {
    const ctx = audio.current;
    if (!ctx || !c.pushes_s?.length || firstCapture.current === null) return;
    const sessionNow = performance.now() / 1000 - firstCapture.current;
    const latency = (ctx as AudioContext & { outputLatency?: number }).outputLatency || ctx.baseLatency || 0;
    const at = (t: number) => ctx.currentTime + (t - sessionNow) - latency;
    const beep = (t: number, hz: number, dur: number) => {
      if (at(t) <= ctx.currentTime) return;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.frequency.value = hz;
      gain.gain.setValueAtTime(0.0001, at(t));
      gain.gain.exponentialRampToValueAtTime(0.4, at(t) + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, at(t) + dur);
      osc.connect(gain).connect(ctx.destination);
      osc.start(at(t));
      osc.stop(at(t) + dur + 0.02);
    };
    const first = c.pushes_s[0];
    for (let k = 3; k >= 1; k--) beep(first - k * c.period_s, 660, 0.06);
    for (const p of c.pushes_s) beep(p, 1320, 0.12);
  };

  const control = (goal: "pump" | "stop" | "off") => {
    if (goal !== "off" && !audio.current) audio.current = new AudioContext();  // must start on a tap
    void audio.current?.resume();
    socket.current?.send(JSON.stringify({ type: "control", goal, target_deg: Number(targetDeg) || 30 }));
    if (goal === "off") setCue(null);
  };

  useEffect(() => {
    if (cue?.action !== "push") return;
    const id = setInterval(() => setNow(performance.now() / 1000 - (firstCapture.current ?? performance.now() / 1000)), 100);
    return () => clearInterval(id);
  }, [cue]);

  useEffect(() => {
    if (forecast?.status !== "pending") return;
    const id = setInterval(() => setNow(performance.now() / 1000 - (firstCapture.current ?? performance.now() / 1000)), 100);
    return () => clearInterval(id);
  }, [forecast?.status]);

  const stop = () => {
    running.current = false;
    setLive(false);
    socket.current?.close();
    socket.current = null;
    const stream = video.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
  };

  const onMessage = (ev: MessageEvent) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "scene") setScene(`${msg.object} (${msg.family}), seen by ${msg.model}`);
    if (msg.type === "track") {
      for (const [f, x, y] of msg.points as Point[]) track.current.set(f, [x, y]);
      if (msg.object_size_px) objectSize.current = msg.object_size_px;
    }
    if (msg.type === "fit") {
      setFit(msg as FitMsg);
      if (msg.ghost) ghost.current = new Map((msg.ghost as Point[]).map(([f, x, y]) => [f, [x, y]]));
    }
    if (msg.type === "regime_change") {
      // Old-regime predictions no longer apply; the next fit uses only the new motion.
      ghost.current.clear();
      setFit(null);
      setRegime(msg.message);
    }
    if (msg.type === "prediction") setForecast(msg as Forecast);
    if (msg.type === "cue") {
      setCue(msg.action === "off" ? null : (msg as CueMsg));
      if (msg.action === "push") scheduleCue(msg as CueMsg);
    }
    if (msg.type === "selected") setStatus("Locked on. Keep the camera still while it moves.");
    if (msg.type === "notice") setNotices((n) => [{ level: msg.level, message: msg.message }, ...n].slice(0, 4));
    if (msg.type === "choose_family") {
      // The model could not settle what kind of motion this is: the type buttons come back for the user to choose.
      setPicked(null);
      setNotices((n) => [{ level: "info", message: msg.message }, ...n].slice(0, 4));
    }
  };

  const start = async () => {
    if (!online) return;
    setNotices([]);
    setFit(null);
    setPicked(null);
    setForecast(null);
    firstCapture.current = null;
    track.current.clear();
    ghost.current.clear();
    sentFrames.current = 0;
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", frameRate: FPS }, audio: false });
    } catch {
      setStatus("Camera unavailable. On a phone this page must be opened over HTTPS.");
      return;
    }
    const v = video.current!;
    v.srcObject = stream;
    await v.play();
    const scale = Math.min(1, MAX_WIDTH / v.videoWidth);
    const w = Math.round(v.videoWidth * scale) & ~1;
    const h = Math.round(v.videoHeight * scale) & ~1;
    setSize({ w, h });

    const ws = new WebSocket(wsUrl());
    socket.current = ws;
    ws.onmessage = onMessage;
    ws.onclose = () => { running.current = false; setLive(false); setStatus("Session ended."); };
    ws.onopen = () => {
      ws.send(JSON.stringify({ fps: FPS, pendulum_length_m: lengthCm ? Number(lengthCm) / 100 : null,
                               object_size_m: sizeCm ? Number(sizeCm) / 100 : null, access_code: accessCode() }));
      running.current = true;
      setLive(true);
      setStatus("Streaming. Tap the moving object to lock on (or wait a moment for it to be found).");
      captureLoop(v, w, h);
    };
  };

  const captureLoop = async (v: HTMLVideoElement, w: number, h: number) => {
    const grab = document.createElement("canvas");
    grab.width = w;
    grab.height = h;
    const g = grab.getContext("2d")!;
    let next = performance.now();
    while (running.current && socket.current?.readyState === WebSocket.OPEN) {
      g.drawImage(v, 0, 0, w, h);
      // The capture time travels with the frame: phones rarely hold the nominal rate, and the server fits on
      // these times, not on arrival order. Header: "PLT1" then seconds as little-endian float64.
      const captured = performance.now() / 1000;
      if (firstCapture.current === null) firstCapture.current = captured;
      // Sequential awaits keep frames in capture order; the server numbers them in arrival order.
      const blob = await new Promise<Blob | null>((res) => grab.toBlob(res, "image/jpeg", 0.7));
      if (blob && socket.current?.readyState === WebSocket.OPEN && socket.current.bufferedAmount < 4_000_000) {
        const header = new ArrayBuffer(12);
        new Uint8Array(header).set([0x50, 0x4c, 0x54, 0x31]);
        new DataView(header).setFloat64(4, captured, true);
        socket.current.send(await new Blob([header, blob]).arrayBuffer());
        sentFrames.current += 1;
      }
      next += 1000 / FPS;
      await new Promise((r) => setTimeout(r, Math.max(0, next - performance.now())));
    }
  };

  const onTap = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const ws = socket.current;
    const c = overlay.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !c) return;
    // Convert the tap from CSS pixels to the frame pixels the server sees.
    const r = c.getBoundingClientRect();
    const x = ((e.clientX - r.left) * c.width) / r.width;
    const y = ((e.clientY - r.top) * c.height) / r.height;
    ws.send(JSON.stringify({ type: "select", x, y }));
    track.current.clear();
    ghost.current.clear();
    setFit(null);
  };

  useEffect(() => {
    let raf = 0;
    const draw = () => {
      const c = overlay.current;
      const ctx = c?.getContext("2d");
      if (c && ctx && size) {
        ctx.clearRect(0, 0, c.width, c.height);
        const now = sentFrames.current - 1;
        const tracked = [...track.current.keys()];
        const newest = tracked.length ? Math.max(...tracked) : -1;
        ctx.strokeStyle = "rgba(255,255,255,0.8)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let f = newest - TRAIL; f <= newest; f++) {
          const p = track.current.get(f);
          if (p) ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke();
        const gp = ghost.current.get(now);
        if (gp) {
          ctx.strokeStyle = "#22d3ee";
          ctx.fillStyle = "#22d3ee44";
          ctx.lineWidth = 3;
          ctx.setLineDash([6, 4]);
          ctx.beginPath();
          ctx.arc(gp[0], gp[1], objectSize.current / 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [size]);

  return (
    <main>
      <header className="page-head">
        <h1>Live analysis</h1>
        <p className="tagline">{online ? status : online === null ? "Looking for the analysis server..." : "Live analysis needs the analysis server, which isn't running right now (it runs during judging sessions). The recordings on the Analyze page are complete, real analyses."}</p>
      </header>

      <div className="live-grid">
        <div className="stage" style={{ aspectRatio: size ? `${size.w} / ${size.h}` : "4 / 3" }}>
          <video ref={video} playsInline muted />
          <canvas
            ref={overlay}
            width={size?.w ?? 640}
            height={size?.h ?? 480}
            className={live ? "tappable" : ""}
            onPointerDown={onTap}
          />
        </div>

        <div className="panel">
          {!live && (
            <div className="upload stack" style={{ gap: 10 }}>
              <p className="small">
                Works live for anything that <strong>swings</strong>, bounces <strong>on a spring</strong>, is
                <strong> thrown or falls</strong>, or <strong>rolls or slides</strong>. Spinning objects and double pendulums
                need a recorded clip (Analyze page).
              </p>
              <p className="small muted">Scale, optional: an A4 sheet in view is read automatically; otherwise give one of these.</p>
              <label className="field">
                Pendulum length, cm
                <input type="number" inputMode="decimal" min="1" step="0.1" value={lengthCm} onChange={(e) => setLengthCm(e.target.value)} />
              </label>
              <label className="field">
                Or the moving object&apos;s longest side, cm
                <input type="number" inputMode="decimal" min="0.5" step="0.1" value={sizeCm} onChange={(e) => setSizeCm(e.target.value)} />
              </label>
              {judgesOnly && (
                <p className="small muted">
                  Live mode is paused right now. Every recording on the Analyze page is a complete, real analysis.
                </p>
              )}
              <button onClick={start} disabled={!online || judgesOnly}>Start live analysis</button>
            </div>
          )}
          {live && <button onClick={stop}>Stop</button>}
          {live && !picked && !fit?.equation && (
            <div className="stack" style={{ gap: 6 }}>
              <span className="small muted">What is it? Tap to start fitting now; the agent&apos;s hypotheses refine it when ready.</span>
              <div className="chips">
                {LIVE_FAMILIES.map(([family, label]) => (
                  <button key={family} type="button" className="chip" onClick={() => {
                    socket.current?.send(JSON.stringify({ type: "family", family }));
                    setPicked(label);
                  }}>{label}</button>
                ))}
              </div>
            </div>
          )}
          {live && picked && <p className="small muted">You picked: {picked}</p>}
          {live && fit?.status === "accepted" && fit.equation?.model_id.startsWith("pendulum") && (
            <div className="stack control-box" style={{ gap: 6 }}>
              <span className="small muted">Control it by hand: the verified law says when to push, the phone beeps in rhythm.</span>
              <div className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                <label className="small">Target swing (degrees){" "}
                  <input type="number" inputMode="numeric" min="3" max="80" value={targetDeg} style={{ width: 70 }}
                    onChange={(e) => setTargetDeg(e.target.value)} />
                </label>
                <button type="button" onClick={() => control("pump")}>Pump it up</button>
                <button type="button" onClick={() => control("stop")}>Stop it</button>
                {cue && <button type="button" className="secondary" onClick={() => control("off")}>Control off</button>}
              </div>
            </div>
          )}
          {cue && <CueBanner c={cue} now={now} />}
          {live && (
            <VoiceNotes onNotes={(n) => {
              // A length said while filming goes straight into the running session.
              if (n.pendulum_length_m && socket.current?.readyState === WebSocket.OPEN) {
                socket.current.send(JSON.stringify({ type: "length", meters: n.pendulum_length_m }));
              }
            }} />
          )}
          {scene && <p className="muted small">Object: {scene}</p>}
          {regime && <div className="regime-banner">{regime}</div>}
          {forecast && <ForecastBanner f={forecast} now={now} />}
          {fit && (
            <>
              <h3 className={`status ${fit.status}`}>
                {fit.status === "accepted" ? "Verified" : fit.status === "ambiguous" ? "Converging" : "No clean equation yet"}
                {fit.window_s ? <span className="muted small"> · {fit.window_s.toFixed(1)} s of motion</span> : null}
              </h3>
              {fit.equation && (
                <div className="equation"><Tex tex={fit.equation.latex} display /></div>
              )}
              {fit.parameters?.map((p) => (
                <p key={p.name}>
                  <Tex tex={p.latex} /> = {p.value.toPrecision(4)} {formatUnit(p.unit)}
                  {p.spread !== null && <span className="muted"> (± {p.spread.toPrecision(2)} across recent fits)</span>}
                </p>
              ))}
              <p className="muted small">{fit.reason}</p>
            </>
          )}
          {notices.map((n, i) => (
            <p key={i} className={n.level === "error" ? "error" : "muted small"}>{n.message}</p>
          ))}
        </div>
      </div>
    </main>
  );
}

type Forecast = {
  id: number; status: "pending" | "hit" | "miss" | "unmeasured"; target_s: number; window_s: number; horizon_s: number;
  made_s: number; error_s?: number;
};

// The prediction is shown (and stored on the server) before the event; the verdict comes from the measured track.
function ForecastBanner({ f, now }: { f: Forecast; now: number }) {
  if (f.status === "pending") {
    const left = Math.max(0, f.target_s - now);
    return (
      <div className="forecast-banner">
        <strong>Prediction {f.id}:</strong> it passes its rest point moving the same way in{" "}
        <span className="countdown">{left.toFixed(1)} s</span> (± {f.window_s.toFixed(2)} s), predicted {f.horizon_s.toFixed(0)} s ahead.
      </div>
    );
  }
  if (f.status === "unmeasured") {
    return <div className="forecast-banner">Prediction {f.id}: tracking lost the object around the predicted time, so it can&apos;t be judged.</div>;
  }
  const err = f.error_s ?? 0;
  return (
    <div className={`forecast-banner ${f.status}`}>
      <strong>Prediction {f.id}: {f.status === "hit" ? "hit" : "missed"}.</strong> It happened {Math.abs(err).toFixed(2)} s{" "}
      {err > 0 ? "later" : "earlier"} than predicted {f.horizon_s.toFixed(0)} s in advance; the stated window was ± {f.window_s.toFixed(2)} s.
    </div>
  );
}

type CueMsg = {
  action: "push" | "wait" | "done"; goal: "pump" | "stop"; period_s: number; direction: "left" | "right";
  amplitude_deg: number; target_deg: number; lead_s: number; reason: string; pushes_s?: number[];
};

function CueBanner({ c, now }: { c: CueMsg; now: number }) {
  const next = c.pushes_s?.find((p) => p > now);
  return (
    <div className={`forecast-banner cue ${c.action}`}>
      {c.action === "push" && next !== undefined ? (
        <strong>Push {c.direction.toUpperCase()} on the beep, in <span className="countdown">{(next - now).toFixed(1)} s</span></strong>
      ) : c.action === "done" ? <strong>Stopped.</strong> : c.action === "wait" ? <strong>At the target: hold, don&apos;t push.</strong>
        : <strong>Measuring the swing after your push...</strong>}
      <div className="small">
        Swing now {c.amplitude_deg.toFixed(0)}°, target {c.target_deg.toFixed(0)}°. {c.reason}
        {c.lead_s > 0.02 ? ` Your pushes land about ${c.lead_s.toFixed(2)} s after the beat, so the beats come that much early.` : ""}
      </div>
    </div>
  );
}
