"use client";

import { isLevel, LEVEL_WARN_DEG, type Level, useLevel } from "@/lib/level";

// A bubble level for filming: the measured limits assume a still, level phone held side-on.
export function LevelIndicator({ onLock, locked }: { onLock?: (l: Level | null) => void; locked?: Level | null }) {
  const { level, state, request } = useLevel();
  if (state === "unsupported") return null;
  if (state === "needs-permission" || state === "denied") {
    return (
      <div className="level">
        <button type="button" className="secondary" onClick={request}>Show a level while filming</button>
        {state === "denied" && <span className="small">Motion access was refused; the level needs it.</span>}
      </div>
    );
  }
  if (!level) return null;
  const ok = isLevel(level);
  const words = level.moving ? "Hold still" : Math.abs(level.roll) > LEVEL_WARN_DEG
    ? `Tilted ${Math.abs(level.roll).toFixed(1)}°: turn ${level.roll > 0 ? "left" : "right"}`
    : Math.abs(level.pitch) > 15 ? `Pointed ${level.pitch > 0 ? "up" : "down"} ${Math.abs(level.pitch).toFixed(0)}°: face it side-on`
      : `Level (${Math.abs(level.roll).toFixed(1)}°)`;
  return (
    <div className={`level ${ok ? "level-ok" : "level-warn"}`} role="status" aria-live="polite">
      <svg viewBox="-50 -12 100 24" width="100" height="24" aria-hidden="true">
        <line x1="-48" y1="0" x2="48" y2="0" className="level-ref" />
        <g transform={`rotate(${Math.max(-30, Math.min(30, -level.roll))})`}><line x1="-40" y1="0" x2="40" y2="0" className="level-bar" /></g>
      </svg>
      <span>{words}</span>
      {onLock && (
        <button type="button" className="secondary" onClick={() => onLock(locked ? null : level)}>
          {locked ? `Tilt saved (${Math.abs(locked.roll).toFixed(1)}°): clear` : "Save this tilt with the clip"}
        </button>
      )}
    </div>
  );
}
