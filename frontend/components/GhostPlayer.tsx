"use client";

import { useEffect, useRef, useState } from "react";
import type { TrackPoint } from "@/lib/api";

// The page's data hues, lifted for contrast on the dark video stage: model blue first, then two alternates.
const GHOST_COLORS = ["#5aa9ff", "#f5a524", "#c4a1ff"];
const TRAIL_FRAMES = 20;
const WHATIF_COLOR = "#ff6fae";
const MUJOCO_COLOR = "#46d98a";
const VEL_COLOR = "#ffd84d"; // velocity (yellow: green is the MuJoCo replay)
const FORCE_COLOR = "#ff5a5a"; // net force: the fitted law's acceleration

// Velocity and acceleration of the model's own simulated path, by central differences (clip pixels per frame).
// Taken from the verified law, not the noisy measurement: on an orbit the force arrow points at the Sun.
function vectorsOf(m: Map<number, [number, number]>) {
  const out = new Map<number, { v: [number, number]; a: [number, number] }>();
  for (const [f, p] of m) {
    const a0 = m.get(f - 1), a2 = m.get(f + 1);
    if (!a0 || !a2) continue;
    out.set(f, { v: [(a2[0] - a0[0]) / 2, (a2[1] - a0[1]) / 2], a: [a2[0] - 2 * p[0] + a0[0], a2[1] - 2 * p[1] + a0[1]] });
  }
  return out;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
}

interface Props {
  src: string;
  width: number;
  height: number;
  fps: number;
  real: TrackPoint[];
  secondReal?: TrackPoint[];
  objectSizePx?: number;
  whatIf?: { label: string; points: TrackPoint[] } | null;
  mujoco?: TrackPoint[];
  ghosts: { model_id: string; points: TrackPoint[] }[];
}

function byFrame(points: TrackPoint[]): Map<number, [number, number]> {
  return new Map(points.map(([f, x, y]) => [f, [x, y]]));
}

export function GhostPlayer({ src, width, height, fps, real, secondReal, objectSizePx, ghosts, whatIf, mujoco }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [showReal, setShowReal] = useState(true);
  const [showVectors, setShowVectors] = useState(true);

  useEffect(() => {
    const v = video.current;
    const c = canvas.current;
    if (!v || !c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const realMaps = [byFrame(real), ...(secondReal?.length ? [byFrame(secondReal)] : [])];
    const ghostMaps = ghosts.map((g) => byFrame(g.points));
    const whatIfMap = whatIf ? byFrame(whatIf.points) : null;
    const mujocoMap = mujoco?.length ? byFrame(mujoco) : null;
    const vectors = ghostMaps.length ? vectorsOf(ghostMaps[0]) : new Map();
    // One scale per arrow kind for the whole clip: the median size maps to a fixed screen length, so arrows compare.
    const vMed = median([...vectors.values()].map((x) => Math.hypot(...x.v)));
    const aMed = median([...vectors.values()].map((x) => Math.hypot(...x.a)));
    // Canvas works in clip pixels (the same space as the track) and CSS scales it with the video, so marks
    // are sized in screen pixels here: legible on a phone, and never big enough to hide the object.
    const draw = (mediaTime: number) => {
      const frame = Math.round(mediaTime * fps);
      const k = width / Math.max(c.clientWidth, 1); // clip pixels per screen pixel
      const ringR = Math.min(Math.max((objectSizePx ?? 0) / 2 / k, 9), 22) * k;
      ctx.clearRect(0, 0, width, height);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      // Every mark gets a dark halo underneath so it reads on bright scenes as well as dark ones.
      const path = (m: Map<number, [number, number]>, from: number, to: number) => {
        ctx.beginPath();
        let started = false;
        for (let f = from; f <= to; f++) {
          const p = m.get(f);
          if (!p) continue;
          if (started) ctx.lineTo(p[0], p[1]);
          else ctx.moveTo(p[0], p[1]);
          started = true;
        }
      };
      const haloStroke = (color: string, w: number) => {
        ctx.strokeStyle = "rgba(0,0,0,0.55)";
        ctx.lineWidth = (w + 2.5) * k;
        ctx.stroke();
        ctx.strokeStyle = color;
        ctx.lineWidth = w * k;
        ctx.stroke();
      };
      const ring = (p: [number, number], color: string, dashed: boolean) => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], ringR, 0, Math.PI * 2);
        ctx.setLineDash(dashed ? [5 * k, 4 * k] : []);
        haloStroke(color, 2.5);
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(p[0], p[1], 2.5 * k, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      };

      if (showReal) {
        for (const realMap of realMaps) {
          path(realMap, frame - TRAIL_FRAMES, frame);
          haloStroke("rgba(255,255,255,0.9)", 2);
        }
      }
      const arrow = (from: [number, number], d: [number, number], scale: number, color: string) => {
        const len = Math.hypot(d[0], d[1]) * scale;
        if (!(len > 3 * k)) return;
        const capped = Math.min(len, 110 * k) / len;
        const to: [number, number] = [from[0] + d[0] * scale * capped, from[1] + d[1] * scale * capped];
        const ang = Math.atan2(to[1] - from[1], to[0] - from[0]);
        ctx.beginPath();
        ctx.moveTo(from[0], from[1]);
        ctx.lineTo(to[0], to[1]);
        ctx.moveTo(to[0] - 9 * k * Math.cos(ang - 0.45), to[1] - 9 * k * Math.sin(ang - 0.45));
        ctx.lineTo(to[0], to[1]);
        ctx.lineTo(to[0] - 9 * k * Math.cos(ang + 0.45), to[1] - 9 * k * Math.sin(ang + 0.45));
        haloStroke(color, 2.5);
      };
      ghostMaps.forEach((m, i) => {
        const color = GHOST_COLORS[i % GHOST_COLORS.length];
        path(m, frame - TRAIL_FRAMES, frame);
        haloStroke(color, 2);
        const p = m.get(frame);
        if (i === 0 && p && showVectors) {
          // Where the verified law says the object goes over the next second, then its force and velocity.
          for (let f = frame + 2; f < frame + Math.max(2, Math.round(fps)); f += 2) {
            const q = m.get(f);
            if (!q) continue;
            ctx.beginPath();
            ctx.arc(q[0], q[1], 1.8 * k, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
          }
          const vec = vectors.get(frame);
          if (vec) {
            arrow(p, vec.v, vMed > 0 ? (60 * k) / vMed : 0, VEL_COLOR);
            arrow(p, vec.a, aMed > 0 ? (60 * k) / aMed : 0, FORCE_COLOR);
          }
        }
        if (p) ring(p, color, i > 0);
      });
      const wp = whatIfMap?.get(frame);
      if (whatIfMap) {
        path(whatIfMap, frame - TRAIL_FRAMES, frame);
        haloStroke(WHATIF_COLOR, 2);
      }
      if (wp) ring(wp, WHATIF_COLOR, false);
      const mp = mujocoMap?.get(frame);
      if (mp) {
        // MuJoCo's replay of the exported model: a small solid dot that should sit inside the model's ring.
        ctx.beginPath();
        ctx.arc(mp[0], mp[1], 4 * k, 0, Math.PI * 2);
        ctx.fillStyle = MUJOCO_COLOR;
        ctx.fill();
        ctx.strokeStyle = "rgba(0,0,0,0.6)";
        ctx.lineWidth = 1.5 * k;
        ctx.stroke();
      }
    };

    const now = () => draw(v.currentTime);
    const resize = new ResizeObserver(now); // mark sizes are in screen pixels, so a resized player redraws
    resize.observe(c);
    v.addEventListener("seeked", now);
    now();
    let stop: () => void;
    if ("requestVideoFrameCallback" in v) {
      // Called once per presented video frame with that frame's own timestamp: the overlay can't drift a
      // frame from the picture, and nothing redraws while the video is paused.
      let handle = 0;
      const onFrame = (_: number, meta: VideoFrameCallbackMetadata) => {
        draw(meta.mediaTime);
        handle = v.requestVideoFrameCallback(onFrame);
      };
      handle = v.requestVideoFrameCallback(onFrame);
      stop = () => v.cancelVideoFrameCallback(handle);
    } else {
      let raf = 0;
      const loop = () => {
        now();
        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);
      stop = () => cancelAnimationFrame(raf);
    }
    return () => {
      stop();
      resize.disconnect();
      v.removeEventListener("seeked", now);
    };
  }, [real, secondReal, ghosts, fps, width, height, showReal, showVectors, objectSizePx, whatIf, mujoco]);

  return (
    <div className="player">
      <div className="stage" style={{ aspectRatio: `${width} / ${height}` }}>
        <video ref={video} src={src} controls playsInline muted loop />
        <canvas ref={canvas} width={width} height={height} />
      </div>
      <div className="legend">
        {ghosts.map((g, i) => (
          <span key={g.model_id} className="key">
            <span className="swatch ring" style={{ borderColor: GHOST_COLORS[i % GHOST_COLORS.length], borderStyle: i > 0 ? "dashed" : "solid" }} />
            {i === 0 ? "Model ghost" : "Alternative"}: {g.model_id}
          </span>
        ))}
        {mujoco?.length ? (
          <span className="key"><span className="swatch dot" style={{ background: MUJOCO_COLOR }} />MuJoCo replay of the export</span>
        ) : null}
        {whatIf && (
          <span className="key"><span className="swatch ring" style={{ borderColor: WHATIF_COLOR }} />What if: {whatIf.label}</span>
        )}
        {ghosts.length > 0 && showVectors && (
          <>
            <span className="key"><span className="swatch dot" style={{ background: FORCE_COLOR }} />Net force (the law&apos;s acceleration)</span>
            <span className="key"><span className="swatch dot" style={{ background: VEL_COLOR }} />Velocity</span>
          </>
        )}
        <label>
          <input type="checkbox" checked={showReal} onChange={(e) => setShowReal(e.target.checked)} /> Measured trail
        </label>
        {ghosts.length > 0 && (
          <label>
            <input type="checkbox" checked={showVectors} onChange={(e) => setShowVectors(e.target.checked)} /> Vectors and prediction
          </label>
        )}
      </div>
    </div>
  );
}
