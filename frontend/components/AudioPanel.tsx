"use client";

import type { AudioComparison, AudioResult } from "@/lib/api";
import { formatUnit } from "@/lib/api";
import { Tex } from "./Equation";

// Decimals from the larger uncertainty of the pair, so both sensors are shown to the same precision.
function decimals(std: number): number {
  if (!(std > 0) || !Number.isFinite(std)) return 3;
  return Math.min(6, Math.max(0, 1 - Math.floor(Math.log10(std))));
}

function Comparison({ c }: { c: AudioComparison }) {
  const d = decimals(Math.max(c.camera.std, c.microphone.std));
  const v = (x: { value: number; std: number }) => `${x.value.toFixed(d)} ± ${x.std.toFixed(d)}${c.unit ? ` ${c.unit}` : ""}`;
  return (
    <p className="num">
      {c.label}: camera says {v(c.camera)}, microphone says {v(c.microphone)}.{" "}
      {c.agree && c.fused
        ? <span className="verdict accepted">Agree, combined {v(c.fused)}</span>
        : <span className="verdict rejected">Disagree by {c.z.toFixed(1)} sigma</span>}
    </p>
  );
}

export function AudioPanel({ audio, cameraLawTitle }: { audio: AudioResult; cameraLawTitle?: string }) {
  if (!audio.available) {
    return (
      <div className="panel">
        <h3>Audio</h3>
        <p className="muted small">{audio.note}</p>
      </div>
    );
  }
  const r = audio.restitution;
  const sd = audio.spin_down;
  const onsets = audio.onsets ?? [];
  return (
    <div className="panel">
      <h3>Audio: the microphone as a second sensor</h3>
      {(audio.comparisons ?? []).map((c) => <Comparison key={c.quantity} c={c} />)}
      {r && (
        <p className="num" style={{ marginTop: 8 }}>
          From flight times between impact sounds: <Tex tex="e" /> = {r.e_mean.toFixed(3)} ± {r.e_std.toFixed(3)}{" "}
          <span className="muted small">({r.impacts.map((i) => i.e.toFixed(3)).join(", ")}; no scale or g needed)</span>
        </p>
      )}
      {sd?.law && (
        <p style={{ marginTop: 8 }}>
          The tone falls from {sd.f_start_hz.toFixed(1)} to {sd.f_end_hz.toFixed(1)} Hz: <strong>{sd.law.title}</strong>
          {sd.camera_law && cameraLawTitle && (
            <>, {sd.laws_agree ? "the same law the camera found" : <>while the camera found <strong>{cameraLawTitle}</strong></>}</>
          )}.{" "}
          <span className="num">
            {(sd.rotor_params.length ? sd.rotor_params : sd.acoustic_params).map((p) => (
              <span key={p.name}>{p.name} = {p.value.toPrecision(4)} ± {p.std.toPrecision(2)} {formatUnit(p.unit)} </span>
            ))}
          </span>
          {!sd.rotor_params.length && <span className="muted small">(of the tone; give the blade count for the rotor&apos;s own rates)</span>}
        </p>
      )}
      {sd && !sd.law && <p className="muted small" style={{ marginTop: 8 }}>Spin-down from sound: {sd.reason}</p>}
      {audio.tone && (
        <p className="num" style={{ marginTop: 8 }}>
          Strongest steady tone: {audio.tone.freq_hz.toFixed(2)} ± {audio.tone.halfwidth_hz.toFixed(2)} Hz
          {audio.tone.rotor_rate_hz != null && `, ${audio.tone.rotor_rate_hz.toFixed(2)} turns/s for the given blade count`}
        </p>
      )}
      {onsets.length > 0 && (
        <p className="muted small" style={{ marginTop: 8 }}>
          {audio.onset_count} impact sound{audio.onset_count === 1 ? "" : "s"} at{" "}
          {onsets.map((t) => t.toFixed(3)).join(", ")} s
          {audio.onset_sigma_s != null && ` (timing resolution ${(audio.onset_sigma_s * 1000).toFixed(1)} ms)`}
        </p>
      )}
      {(audio.notes ?? []).map((n, i) => <p key={i} className="muted small" style={{ marginTop: 4 }}>{n}</p>)}
    </div>
  );
}
