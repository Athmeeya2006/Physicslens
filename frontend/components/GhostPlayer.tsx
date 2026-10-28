"use client";

import { useEffect, useRef, useState } from "react";
import type { TrackPoint } from "@/lib/api";

// The page's data hues, lifted for contrast on the dark video stage: model blue first, then two alternates.
const GHOST_COLORS = ["#5aa9ff", "#f5a524", "#c4a1ff"];
const TRAIL_FRAMES = 20;
const WHATIF_COLOR = "#ff6fae";
const MUJOCO_COLOR = "#46d98a";

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
      ghostMaps.forEach((m, i) => {
        const color = GHOST_COLORS[i % GHOST_COLORS.length];
        path(m, frame - TRAIL_FRAMES, frame);
        haloStroke(color, 2);
        const p = m.get(frame);
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
  }, [real, secondReal, ghosts, fps, width, height, showReal, objectSizePx, whatIf, mujoco]);

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
        <label>
          <input type="checkbox" checked={showReal} onChange={(e) => setShowReal(e.target.checked)} /> Measured trail
        </label>
      </div>
    </div>
  );
}
