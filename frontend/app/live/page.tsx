"use client";

import { useEffect, useRef, useState } from "react";
import { Tex } from "@/components/Equation";
import { API_BASE, HAS_BACKEND } from "@/lib/api";

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
  equation?: { title: string; latex: string };
  parameters?: LiveParam[];
  ghost?: Point[];
}
interface Notice { level: string; message: string }

function wsUrl(): string {
  return API_BASE.replace(/^http/, "ws") + "/ws/live";
}

export default function LivePage() {
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
  const [fit, setFit] = useState<FitMsg | null>(null);
  const [notices, setNotices] = useState<Notice[]>([]);
  const [lengthCm, setLengthCm] = useState("");
  const [live, setLive] = useState(false);
  const [regime, setRegime] = useState<string | null>(null);

  useEffect(() => () => stop(), []);

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
    if (msg.type === "selected") setStatus("Locked on. Keep the camera still while it moves.");
    if (msg.type === "notice") setNotices((n) => [{ level: msg.level, message: msg.message }, ...n].slice(0, 4));
  };

  const start = async () => {
    if (!HAS_BACKEND) return;
    setNotices([]);
    setFit(null);
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
      const code = new URLSearchParams(window.location.search).get("code");
      ws.send(JSON.stringify({ fps: FPS, pendulum_length_m: lengthCm ? Number(lengthCm) / 100 : null, access_code: code }));
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
        <p className="tagline">{HAS_BACKEND ? status : "Live analysis needs the analysis server, which isn't connected to this preview. The recordings on the Analyze page are complete, real analyses."}</p>
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
            <div className="upload">
              <label>
                Pendulum length in cm (optional, if no A4 sheet is in view)
                <input type="number" inputMode="decimal" min="1" step="0.1" value={lengthCm} onChange={(e) => setLengthCm(e.target.value)} />
              </label>
              <button onClick={start} disabled={!HAS_BACKEND}>Start live analysis</button>
            </div>
          )}
          {live && <button onClick={stop}>Stop</button>}
          {scene && <p className="muted small">Object: {scene}</p>}
          {regime && <div className="regime-banner">{regime}</div>}
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
                  <Tex tex={p.latex} /> = {p.value.toPrecision(4)} {p.unit}
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
