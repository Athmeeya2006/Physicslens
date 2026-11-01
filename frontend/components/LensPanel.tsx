"use client";

import { useEffect, useState } from "react";
import { calibrateLens, type LensCalibration } from "@/lib/api";

const KEY = "physicslens-lens";

// The calibration lives in this browser only (it describes this phone's camera), and is sent with each upload.
export function loadLens(): LensCalibration | null {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "null") as LensCalibration | null;
  } catch {
    return null;
  }
}

function saveLens(lens: LensCalibration | null): void {
  try {
    if (lens) localStorage.setItem(KEY, JSON.stringify(lens));
    else localStorage.removeItem(KEY);
  } catch {
    /* storage blocked: the calibration still applies to uploads from this page */
  }
}

export function LensPanel({ onChange }: { onChange: (lens: LensCalibration | null) => void }) {
  const [lens, setLens] = useState<LensCalibration | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const saved = loadLens();
    setLens(saved);
    onChange(saved);
  }, [onChange]);

  const set = (l: LensCalibration | null) => {
    setLens(l);
    saveLens(l);
    onChange(l);
  };

  const run = async () => {
    if (!video) return;
    setBusy(true);
    setError(null);
    try {
      set((await calibrateLens(video)).lens);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <details className="lens-panel">
      <summary>Lens calibration, optional: {lens ? `this phone is calibrated (${lens.width}x${lens.height})` : "remove this phone's lens distortion"}</summary>
      {lens ? (
        <div className="stack" style={{ gap: 8 }}>
          <p className="small">
            Focal length {((lens.fx + lens.fy) / 2).toFixed(0)} px, distortion k1 = {lens.dist[0].toFixed(3)}, from {lens.views} views
            ({lens.rms_px.toFixed(2)} px error). Tracked points are undistorted before fitting, for clips of the same shape.
          </p>
          <button type="button" className="secondary" onClick={() => set(null)}>Forget this calibration</button>
        </div>
      ) : (
        <div className="stack" style={{ gap: 8 }}>
          <ol className="small">
            <li><a href="/checkerboard.svg" target="_blank" rel="noopener">Print the board</a> at 100% scale and tape it flat.</li>
            <li>Film it for 10 to 20 seconds with the same camera setting you use for clips, moving slowly so the board
              visits every corner of the frame at a few angles.</li>
            <li>Upload that video here. It is used once to measure the lens and then deleted.</li>
          </ol>
          <input type="file" accept="video/*" aria-label="Video of the printed board" onChange={(e) => setVideo(e.target.files?.[0] ?? null)} />
          <button type="button" className="secondary" disabled={!video || busy} onClick={run}>
            {busy ? "Measuring the lens" : "Calibrate"}
          </button>
          {error && <p className="error" role="alert">{error}</p>}
        </div>
      )}
    </details>
  );
}
