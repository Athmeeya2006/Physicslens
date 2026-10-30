"use client";

import { useEffect, useRef, useState } from "react";

// Physics audit at scale: the share of clips the audit flags, per source, with a 95% interval. The real-footage
// control is the reference (neutral grey, hollow diamond, its own row above a divider); generated clips carry the
// model hue (#0b63ce, validated on the #f5f6f3 surface). Identity never rests on colour alone.

interface Group {
  n: number; judged: number; flagged: number; consistent: number; cant_judge: number; error: number;
  rate: number | null; lo: number; hi: number;
}
interface Source extends Group { key: string; label: string }
interface Example { id: string; generator: string; caption: string; human_pc: number; headline: string; url: string }
export interface StudySummary {
  clips: number;
  sources: Source[];
  by_human_score: (Group & { pc: number })[];
  examples: Example[];
  model_alone: Record<string, number> | null;
}

const PAD = { left: 150, right: 56, top: 26, bottom: 30 };
const ROW = 30;
const pct = (v: number) => `${Math.round(v * 100)}%`;

function IntervalChart({ rows, caption }: { rows: { label: string; g: Group; reference?: boolean }[]; caption: string }) {
  const box = useRef<HTMLElement>(null);
  const [W, setW] = useState(720);
  const [hover, setHover] = useState<number | null>(null);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(300, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const narrow = W < 520;
  const left = narrow ? 112 : PAD.left;
  const gap = rows.some((r) => r.reference) ? 12 : 0;
  const H = PAD.top + rows.length * ROW + gap + PAD.bottom;
  const x = (v: number) => left + v * (W - left - PAD.right);
  const y = (i: number) => PAD.top + i * ROW + ROW / 2 + (rows[i].reference ? 0 : gap);
  return (
    <figure ref={box} className="study-chart">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" onPointerLeave={() => setHover(null)}
           aria-label={`${caption}: ${rows.map((r) => `${r.label} ${r.g.rate === null ? "not judged" : pct(r.g.rate)}`).join(", ")}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className="axis" x1={x(t)} x2={x(t)} y1={PAD.top - 6} y2={H - PAD.bottom + 4} opacity={t === 0 ? 0.9 : 0.35} />
            <text className="tick" x={x(t)} y={H - 10} textAnchor="middle">{pct(t)}</text>
          </g>
        ))}
        {gap > 0 && <line className="axis" x1={8} x2={W - 8} y1={PAD.top + ROW + gap / 2} y2={PAD.top + ROW + gap / 2} opacity={0.6} />}
        {rows.map((r, i) => {
          const cy = y(i);
          const judged = r.g.rate !== null;
          return (
            <g key={r.label} onPointerEnter={() => setHover(i)} className={r.reference ? "ref" : "gen"}>
              <rect x={0} y={cy - ROW / 2} width={W} height={ROW} fill="transparent" />
              <text className="row-label" x={left - 10} y={cy + 4} textAnchor="end">{r.label}</text>
              {judged ? (
                <>
                  <line className="ci" x1={x(r.g.lo)} x2={x(r.g.hi)} y1={cy} y2={cy} />
                  {r.reference
                    ? <rect className="mark" x={x(r.g.rate!) - 5} y={cy - 5} width={10} height={10} transform={`rotate(45 ${x(r.g.rate!)} ${cy})`} />
                    : <circle className="mark" cx={x(r.g.rate!)} cy={cy} r={5} />}
                  <text className="value" x={W - PAD.right + 8} y={cy + 4}>{pct(r.g.rate!)}</text>
                </>
              ) : <text className="value muted" x={x(0) + 6} y={cy + 4}>none judged yet</text>}
            </g>
          );
        })}
        {hover !== null && rows[hover].g.rate !== null && (() => {
          const r = rows[hover];
          const tipW = 230;
          const tx = Math.min(Math.max(x(r.g.rate!) - tipW / 2, 4), W - tipW - 4);
          const ty = y(hover) + (hover < rows.length - 2 ? 12 : -64);
          return (
            <g className="tip" transform={`translate(${tx},${ty})`} pointerEvents="none">
              <rect width={tipW} height={52} rx={3} />
              <text x={10} y={18}>{r.label}: {r.g.flagged} of {r.g.judged} judged flagged</text>
              <text x={10} y={34}>95% interval {pct(r.g.lo)} to {pct(r.g.hi)}</text>
              <text x={10} y={48}>{r.g.cant_judge} can&apos;t judge, {r.g.error} not trackable</text>
            </g>
          );
        })()}
      </svg>
      <figcaption className="forecast-caption">
        <span>
          <span className="key"><svg width="12" height="12" aria-hidden="true"><rect x="2.5" y="2.5" width="7" height="7" transform="rotate(45 6 6)" style={{ fill: "var(--ref)" }} /></svg>Real footage, the control</span>
          <span className="key"><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="4.5" style={{ fill: "var(--model)" }} /></svg>AI-generated</span>
          <span className="key"><svg width="18" height="8" aria-hidden="true"><line x1="0" x2="18" y1="4" y2="4" style={{ stroke: "var(--ref)" }} strokeWidth="2" /></svg>95% interval</span>
        </span>
        <span>{caption}</span>
      </figcaption>
    </figure>
  );
}

export function AuditStudy() {
  const [s, setS] = useState<StudySummary | null>(null);
  useEffect(() => {
    fetch("/audit_study.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then(setS).catch(() => setS(null));
  }, []);
  if (!s || !s.sources.length) return null;
  const real = s.sources.find((x) => x.key === "real");
  const gens = s.sources.filter((x) => x.key !== "real");
  return (
    <section className="panel stack" style={{ gap: 14 }}>
      <h2>Measured at scale: AI-generated video against real footage</h2>
      <p>
        The same audit, with the same settings, on {s.clips} public clips: AI-generated video from seven generators
        (VideoPhy-2, MIT licence) and real footage of the same kinds of motion (Physics-IQ, CC BY 4.0) as the control.
        Real objects can&apos;t break physics, so the control row is the audit&apos;s own false-accusation rate.
        Clips it can&apos;t judge (too short, poorly tracked, no dynamics check applies) are counted separately, never as
        a pass or a fail.
      </p>
      <IntervalChart caption="Share of judged clips flagged as breaking physics"
                     rows={[...(real ? [{ label: "Real footage", g: real, reference: true }] : []),
                            ...gens.map((g) => ({ label: g.label, g }))]} />
      {s.by_human_score.length > 0 && (
        <>
          <h3>Against people&apos;s own ratings</h3>
          <p className="small muted">
            VideoPhy-2 asked people to rate each generated clip&apos;s physical commonsense from 1 (breaks it) to 5
            (looks right). If the audit measures something real, its flags should fall as the human score rises.
          </p>
          <IntervalChart caption="Share flagged, by the human physical-commonsense score"
                         rows={s.by_human_score.map((g) => ({ label: `Human score ${g.pc}`, g }))} />
        </>
      )}
      {s.examples.length > 0 && (
        <details>
          <summary>Flagged AI-generated clips, with the reason</summary>
          <ul className="stack" style={{ gap: 8, paddingLeft: 18, margin: "8px 0 0" }}>
            {s.examples.map((e) => (
              <li key={e.id} className="small">
                <strong>{e.generator}</strong> (human score {e.human_pc}): <a href={e.url} rel="noopener">{e.caption}</a>
                <div className="muted">{e.headline}</div>
              </li>
            ))}
          </ul>
        </details>
      )}
      <details>
        <summary>Table view</summary>
        <div className="table-wrap">
          <table className="cards">
            <thead><tr><th>Source</th><th>Clips</th><th>Judged</th><th>Flagged</th><th>Consistent</th><th>Can&apos;t judge</th><th>Not trackable</th></tr></thead>
            <tbody>
              {s.sources.map((g) => (
                <tr key={g.key}>
                  <td data-label="Source">{g.label}</td><td data-label="Clips">{g.n}</td><td data-label="Judged">{g.judged}</td>
                  <td data-label="Flagged">{g.flagged}{g.rate !== null ? ` (${pct(g.rate)})` : ""}</td>
                  <td data-label="Consistent">{g.consistent}</td><td data-label="Can't judge">{g.cant_judge}</td>
                  <td data-label="Not trackable">{g.error}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <p className="small muted">
        Every clip, its source, licence and verdict: <a href="https://github.com/Athmeeya2006/Physicslens/tree/main/data/audit_study">data/audit_study</a>.
      </p>
    </section>
  );
}
