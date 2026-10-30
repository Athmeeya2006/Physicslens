"use client";

import { useEffect, useRef, useState } from "react";

// Physics audit at scale (scripts/audit_study.py report): AI-generated clips (VideoPhy-2) against real footage
// (Physics-IQ), every verdict replayed under the current rules. The real-footage control is the neutral reference
// (grey, hollow diamond, its own row); generated clips carry the model hue. Identity never rests on colour alone.

interface Share { k: number; n: number; rate: number | null; lo: number; hi: number }
interface SetStats {
  clips: number; verdicts: Record<string, number>; errors: Record<string, number>;
  judged: Share; flagged_among_judged: Share; tracked: Share; verified_law_among_tracked: Share;
  heldout_error_median: number | null; heldout_errors: number[]; model_alone: Record<string, number>;
}
interface Analysis {
  sets: Record<string, SetStats>;
  generators: Record<string, Record<string, number> & { flagged_among_judged: Share }>;
  forecast_error_auc_generated_vs_real: number | null;
  false_accusation?: { accused: number; judged_real: number; upper_95: number | null };
}
interface Example { id: string; generator: string; caption: string; human_pc: number; headline: string; url: string }
interface Summary { clips: number; examples: Example[]; analysis?: Analysis }

const NAMES: Record<string, string> = { cogvideo: "CogVideoX", cosmos: "Cosmos", hunyuan: "HunyuanVideo", ray2: "Ray2",
  sora: "Sora", videocrafter: "VideoCrafter", wan: "Wan" };
const pct = (v: number) => `${Math.round(v * 100)}%`;

function useWidth(min = 300): [React.RefObject<HTMLElement | null>, number] {
  const ref = useRef<HTMLElement>(null);
  const [w, setW] = useState(720);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(min, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, [min]);
  return [ref, w];
}

/** One row per source: the share with its 95% interval. */
function IntervalChart({ rows, caption, unit }: { rows: { label: string; s: Share; reference?: boolean }[]; caption: string; unit: string }) {
  const [box, W] = useWidth();
  const [hover, setHover] = useState<number | null>(null);
  const left = W < 520 ? 112 : 150;
  const right = 56;
  const ROW = 30;
  const gap = rows.some((r) => r.reference) ? 12 : 0;
  const H = 26 + rows.length * ROW + gap + 30;
  const x = (v: number) => left + v * (W - left - right);
  const y = (i: number) => 26 + i * ROW + ROW / 2 + (rows[i].reference ? 0 : gap);
  return (
    <figure ref={box as React.RefObject<HTMLElement>} className="study-chart">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" onPointerLeave={() => setHover(null)}
           aria-label={`${caption}: ${rows.map((r) => `${r.label} ${r.s.rate === null ? "none" : pct(r.s.rate)}`).join(", ")}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className="axis" x1={x(t)} x2={x(t)} y1={20} y2={H - 26} opacity={t === 0 ? 0.9 : 0.35} />
            <text className="tick" x={x(t)} y={H - 10} textAnchor="middle">{pct(t)}</text>
          </g>
        ))}
        {gap > 0 && <line className="axis" x1={8} x2={W - 8} y1={26 + ROW + gap / 2} y2={26 + ROW + gap / 2} opacity={0.6} />}
        {rows.map((r, i) => {
          const cy = y(i);
          return (
            <g key={r.label} onPointerEnter={() => setHover(i)} className={r.reference ? "ref" : "gen"}>
              <rect x={0} y={cy - ROW / 2} width={W} height={ROW} fill="transparent" />
              <text className="row-label" x={left - 10} y={cy + 4} textAnchor="end">{r.label}</text>
              {r.s.rate !== null && r.s.n > 0 ? (
                <>
                  <line className="ci" x1={x(r.s.lo)} x2={x(r.s.hi)} y1={cy} y2={cy} />
                  {r.reference
                    ? <rect className="mark" x={x(r.s.rate) - 5} y={cy - 5} width={10} height={10} transform={`rotate(45 ${x(r.s.rate)} ${cy})`} />
                    : <circle className="mark" cx={x(r.s.rate)} cy={cy} r={5} />}
                  <text className="value" x={W - right + 8} y={cy + 4}>{pct(r.s.rate)}</text>
                </>
              ) : <text className="value muted" x={x(0) + 6} y={cy + 4}>no clips yet</text>}
            </g>
          );
        })}
        {hover !== null && rows[hover].s.rate !== null && rows[hover].s.n > 0 && (() => {
          const r = rows[hover];
          const tipW = 240;
          const tx = Math.min(Math.max(x(r.s.rate!) - tipW / 2, 4), W - tipW - 4);
          const ty = y(hover) + (hover < rows.length - 2 ? 12 : -50);
          return (
            <g className="tip" transform={`translate(${tx},${ty})`} pointerEvents="none">
              <rect width={tipW} height={38} rx={3} />
              <text x={10} y={16}>{r.label}: {r.s.k} of {r.s.n} {unit}</text>
              <text x={10} y={31}>95% interval {pct(r.s.lo)} to {pct(r.s.hi)}</text>
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

/** Every clip's held-out forecast error as a dot, real against generated, on a log scale; the line is the median. */
function ErrorStrip({ real, gen }: { real: number[]; gen: number[] }) {
  const [box, W] = useWidth();
  const left = W < 520 ? 112 : 150;
  const right = 24;
  const lo = Math.log10(0.001);
  const hi = Math.log10(3);
  const x = (v: number) => left + ((Math.log10(Math.min(Math.max(v, 0.001), 3)) - lo) / (hi - lo)) * (W - left - right);
  const rows = [{ label: "Real footage", values: real, cls: "ref", y: 34 }, { label: "AI-generated", values: gen, cls: "gen", y: 78 }];
  const median = (v: number[]) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)];
  return (
    <figure ref={box as React.RefObject<HTMLElement>} className="study-chart">
      <svg width={W} height={124} viewBox={`0 0 ${W} 124`} role="img"
           aria-label={`Forecast error of the best law per clip: real median ${median(real)?.toFixed(2)}, AI median ${median(gen)?.toFixed(2)}`}>
        {[0.001, 0.01, 0.1, 1].map((t) => (
          <g key={t}>
            <line className="axis" x1={x(t)} x2={x(t)} y1={14} y2={100} opacity={0.35} />
            <text className="tick" x={x(t)} y={116} textAnchor="middle">{t >= 0.01 ? `${t * 100}%` : "0.1%"}</text>
          </g>
        ))}
        {rows.map((r) => (
          <g key={r.label} className={r.cls}>
            <text className="row-label" x={left - 10} y={r.y + 4} textAnchor="end">{r.label}</text>
            {r.values.map((v, i) => (
              <circle key={i} className="strip" cx={x(v)} cy={r.y + ((i * 7) % 15) - 7} r={3.2} />
            ))}
            {r.values.length > 0 && <line className="median" x1={x(median(r.values))} x2={x(median(r.values))} y1={r.y - 14} y2={r.y + 14} />}
          </g>
        ))}
      </svg>
      <figcaption className="forecast-caption">
        <span>Each dot is one clip: how far the best law&apos;s forecast misses the motion it never saw, as a share of the
          motion&apos;s spread (log scale). The bar is the median.</span>
      </figcaption>
    </figure>
  );
}

export function AuditStudy() {
  const [s, setS] = useState<Summary | null>(null);
  useEffect(() => {
    fetch("/audit_study.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then(setS).catch(() => setS(null));
  }, []);
  const a = s?.analysis;
  if (!a || !a.sets.real || !a.sets.generated) return null;
  const real = a.sets.real;
  const gen = a.sets.generated;
  const gens = Object.keys(a.generators).filter((g) => a.generators[g] && g !== "ray2").sort();
  const lawRows = [{ label: "Real footage", s: real.verified_law_among_tracked, reference: true },
                   { label: "AI-generated", s: gen.verified_law_among_tracked }];
  const flagRows = [{ label: "Real footage", s: real.flagged_among_judged, reference: true },
                    ...gens.map((g) => ({ label: NAMES[g] ?? g, s: a.generators[g].flagged_among_judged }))];
  const fa = a.false_accusation;
  const alone = (set: SetStats) => {
    const m = set.model_alone;
    const n = Object.values(m).reduce((x, y) => x + y, 0);
    return n ? `${m.violates_physics ?? 0} of ${n} called fake` : "not run";
  };
  return (
    <section className="panel stack" style={{ gap: 16 }}>
      <h2>Measured at scale: AI-generated video against real footage</h2>
      <p>
        The same pipeline, with the same settings, on {real.clips + gen.clips} public clips: AI-generated video from
        public benchmarks (VideoPhy-2, MIT licence) and real footage of the same kinds of motion (Physics-IQ, CC BY 4.0)
        as the control. Every verdict below is from the current rules, replayed on each clip&apos;s saved measurements.
      </p>
      <ul className="hero-stats">
        <li><span className="hero-number">{pct(real.verified_law_among_tracked.rate ?? 0)} vs {pct(gen.verified_law_among_tracked.rate ?? 0)}</span>
          <span className="hero-label">real vs AI motion with a law that forecasts what the model never saw</span></li>
        <li><span className="hero-number">{(real.heldout_error_median ?? 0).toFixed(2)} vs {(gen.heldout_error_median ?? 0).toFixed(2)}</span>
          <span className="hero-label">median forecast error of the best law, real vs AI (lower is more law-like)</span></li>
        {fa && <li><span className="hero-number">{fa.accused} of {fa.judged_real}</span>
          <span className="hero-label">real clips accused of breaking physics{fa.upper_95 !== null ? ` (at most ${pct(fa.upper_95)} at 95%)` : ""}</span></li>}
      </ul>
      <h3>Is there a law at all?</h3>
      <p className="small muted">
        Real objects obey physics, so a law fitted to the first part of the motion should forecast the rest. Among the
        clips SAM could follow, how often one did:
      </p>
      <IntervalChart rows={lawRows} unit="tracked clips" caption="Share of tracked clips with a verified law" />
      <ErrorStrip real={real.heldout_errors} gen={gen.heldout_errors} />
      {a.forecast_error_auc_generated_vs_real !== null && (
        <p className="small muted">
          Pick one real and one AI clip at random: the AI clip&apos;s motion is the harder to forecast{" "}
          {pct(a.forecast_error_auc_generated_vs_real)} of the time (50% would mean no difference).
        </p>
      )}
      <h3>Clear violations</h3>
      <p className="small muted">
        The audit accuses only an impossible event it can measure (an object teleporting, changing size with nothing to
        explain it, gaining energy in free flight) and says &ldquo;can&apos;t judge&rdquo; whenever the clip can&apos;t
        settle it. False accusations are worse than misses, so the real footage is the test of the rules.
      </p>
      <IntervalChart rows={flagRows} unit="judged clips flagged" caption="Share of judged clips flagged as breaking physics" />
      {s!.examples.length > 0 && (
        <div className="stack" style={{ gap: 8 }}>
          {s!.examples.map((e) => (
            <p key={e.id} className="small"><strong>{e.generator}:</strong> <a href={e.url} rel="noopener">{e.caption}</a>
              <span className="muted"> {e.headline}</span></p>
          ))}
        </div>
      )}
      <div className="table-wrap">
        <table className="cards">
          <thead><tr><th>Source</th><th>Clips</th><th>Tracked</th><th>Verified law</th><th>Judged</th><th>Flagged</th><th>Model alone, from 4 frames</th></tr></thead>
          <tbody>
            {[["Real footage", real], ["AI-generated", gen]].map(([label, x]) => {
              const st = x as SetStats;
              return (
                <tr key={label as string}>
                  <td data-label="Source">{label as string}</td><td data-label="Clips">{st.clips}</td>
                  <td data-label="Tracked">{st.tracked.k} of {st.tracked.n}</td>
                  <td data-label="Verified law">{st.verified_law_among_tracked.k} of {st.verified_law_among_tracked.n}</td>
                  <td data-label="Judged">{st.judged.k}</td><td data-label="Flagged">{st.flagged_among_judged.k}</td>
                  <td data-label="Model alone">{alone(st)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        Limits, stated plainly: clips run on the lighter profile (one hypothesis sample, no referee) to fit the study&apos;s
        budget; Ray2&apos;s links in the benchmark no longer serve video; many clips are short or leave the frame, so
        &ldquo;can&apos;t judge&rdquo; is common by design. Every clip, its source, licence and verdict:{" "}
        <a href="https://github.com/Athmeeya2006/Physicslens/tree/main/data/audit_study">data/audit_study</a>.
      </p>
    </section>
  );
}
