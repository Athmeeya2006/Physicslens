"use client";

import { useEffect, useRef, useState } from "react";
import type { TrackPoint } from "@/lib/api";

const GHOST_COLORS = ["#22d3ee", "#f472b6", "#facc15"];
const TRAIL_FRAMES = 20;
const WHATIF_COLOR = "#e879f9";

interface Props {
  src: string;
  width: number;
  height: number;
  fps: number;
  real: TrackPoint[];
  objectSizePx?: number;
  whatIf?: { label: string; points: TrackPoint[] } | null;
  ghosts: { model_id: string; points: TrackPoint[] }[];
}

function byFrame(points: TrackPoint[]): Map<number, [number, number]> {
  return new Map(points.map(([f, x, y]) => [f, [x, y]]));
}

export function GhostPlayer({ src, width, height, fps, real, objectSizePx, ghosts, whatIf }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [showReal, setShowReal] = useState(true);

  useEffect(() => {
    const v = video.current;
    const c = canvas.current;
    if (!v || !c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const realMap = byFrame(real);
    const ghostMaps = ghosts.map((g) => byFrame(g.points));
    const whatIfMap = whatIf ? byFrame(whatIf.points) : null;
    // Canvas works in clip pixels (the same space as the track), CSS scales it with the video.
    // Ghost drawn at the tracked object's own size, so it visibly covers (or misses) the real thing.
    const radius = objectSizePx ? objectSizePx / 2 : Math.max(6, Math.min(width, height) / 40);
    let raf = 0;

    const draw = () => {
      const frame = Math.round(v.currentTime * fps);
      ctx.clearRect(0, 0, width, height);
      if (showReal) {
        ctx.strokeStyle = "rgba(255,255,255,0.7)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let f = frame - TRAIL_FRAMES; f <= frame; f++) {
          const p = realMap.get(f);
          if (p) ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke();
      }
      ghostMaps.forEach((m, i) => {
        const p = m.get(frame);
        if (!p) return;
        ctx.strokeStyle = GHOST_COLORS[i % GHOST_COLORS.length];
        ctx.fillStyle = GHOST_COLORS[i % GHOST_COLORS.length] + "55";
        const stroke = Math.max(3, width / 300);
        ctx.lineWidth = stroke;
        ctx.setLineDash([stroke * 2, stroke * 1.5]);
        ctx.beginPath();
        ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
      });
      const wp = whatIfMap?.get(frame);
      if (wp) {
        ctx.strokeStyle = WHATIF_COLOR;
        ctx.fillStyle = WHATIF_COLOR + "44";
        ctx.lineWidth = Math.max(3, width / 300);
        ctx.beginPath();
        ctx.arc(wp[0], wp[1], radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [real, ghosts, fps, width, height, showReal, objectSizePx, whatIf]);

  return (
    <div className="player">
      <div className="stage" style={{ aspectRatio: `${width} / ${height}` }}>
        <video ref={video} src={src} controls playsInline muted loop />
        <canvas ref={canvas} width={width} height={height} />
      </div>
      <div className="legend">
        {ghosts.map((g, i) => (
          <span key={g.model_id}>
            <i style={{ borderColor: GHOST_COLORS[i % GHOST_COLORS.length] }} /> ghost: {g.model_id}
          </span>
        ))}
        {whatIf && (
          <span>
            <i style={{ borderColor: WHATIF_COLOR, borderStyle: "solid" }} /> what if: {whatIf.label}
          </span>
        )}
        <label>
          <input type="checkbox" checked={showReal} onChange={(e) => setShowReal(e.target.checked)} /> measured trail
        </label>
      </div>
    </div>
  );
}
