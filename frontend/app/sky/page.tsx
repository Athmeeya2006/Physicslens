"use client";

import { Clapperboard, Crosshair, Pause, Play, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Law, Tex } from "@/components/Equation";
import { Sky3D, type Sky3DHandle } from "@/components/Sky3D";
import { useBackend, useHealth } from "@/lib/api";
import {
  bundled, bundledDiscovery, changed, type Changes, decodeState, type Discovery, discoverOnServer, type Elements, encodeState, filmOnServer,
  HYPOTHETICAL, integrate, kicked, listScenarios, type ScenarioSummary, type Simulation, simulateOnServer, SUN_GM,
} from "@/lib/sky";

const NONE: Changes = { mass_scale: {}, disabled: [] };
const SPEEDS = [1, 10, 30, 100, 365];

function MiniChart({ title, unit, base, modified, pick }: {
  title: string; unit: string; base: Elements[]; modified: Elements[] | null; pick: (e: Elements) => number;
}) {
  const W = 260, H = 90, pad = 6;
  const all = [...base, ...(modified ?? [])];
  if (!all.length) return null;
  const ts = all.map((e) => e.t), vs = all.map(pick);
  const t0 = Math.min(...ts), t1 = Math.max(...ts), lo = Math.min(...vs), hi = Math.max(...vs);
  const x = (t: number) => pad + ((t - t0) / (t1 - t0 || 1)) * (W - 2 * pad);
  const y = (v: number) => H - pad - ((v - lo) / (hi - lo || 1)) * (H - 2 * pad);
  const line = (rows: Elements[]) => rows.map((e, i) => `${i ? "L" : "M"}${x(e.t).toFixed(1)},${y(pick(e)).toFixed(1)}`).join("");
  const last = (rows: Elements[]) => (rows.length ? pick(rows[rows.length - 1]) : NaN);
  return (
    <figure className="mini-chart">
      <figcaption><strong>{title}</strong> <span className="muted small">{unit}</span></figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`${title}: ends at ${last(base).toPrecision(5)}${modified ? `, changed run ${last(modified).toPrecision(5)}` : ""} ${unit}`}>
        <path d={line(base)} className="mini-base" />
        {modified && <path d={line(modified)} className="mini-modified" />}
      </svg>
      <span className="small muted">{last(base).toPrecision(6)}{modified ? ` → ${last(modified).toPrecision(6)}` : ""}</span>
    </figure>
  );
}

// The target's equation of motion written out term by term: the Sun's pull, each body's (with the indirect term),
// relativity when it is on. Beside each term its own formula and its size right now, as a share of the Sun's pull,
// from the positions on screen: change a mass, remove a body or add one, and its term changes with it.
const sym = (n: string) => (n === HYPOTHETICAL ? "X" : n.replace(/[^A-Za-z]/g, "").slice(0, 3));
const bodyTerm = (n: string) => {
  const s = String.raw`\mathrm{${sym(n)}}`;
  return String.raw`GM_{${s}}\left(\frac{\mathbf r_{${s}} - \mathbf r}{|\mathbf r_{${s}} - \mathbf r|^3} - \frac{\mathbf r_{${s}}}{r_{${s}}^3}\right)`;
};
const GR_TERM = String.raw`\frac{GM_\odot}{c^2 r^3}\left[\left(\frac{4GM_\odot}{r} - v^2\right)\mathbf r + 4(\mathbf r\cdot\mathbf v)\,\mathbf v\right]`;
const C_AU_PER_DAY = 173.144632674;

function Equations({ sim, init, changes, t, central, subject }: {
  sim: Simulation; init: NonNullable<Simulation["initial"]>; changes: Changes; t: number; central: string; subject: string;
}) {
  const sun = central === "Sun";
  const gmC = sun ? String.raw`GM_\odot` : String.raw`GM_{\mathrm{${sym(central)}}}`;
  const target = sim.bodies.find((b) => b.name === subject);
  if (!target) return null;
  const at = (b: { points: number[][] }, when: number) => {
    const pts = b.points;
    let k = pts.findIndex((p) => p[0] > when);
    if (k < 0) k = pts.length - 1;
    return pts[Math.max(0, k)];
  };
  const pos = (b: { points: number[][] }) => at(b, t).slice(1, 4);
  const r = pos(target);
  const norm = (v: number[]) => Math.hypot(v[0], v[1], v[2] ?? 0);
  const sunPull = init.central_gm / norm(r) ** 2;
  const terms = sim.bodies.filter((b) => b.name !== subject).map((b) => {
    const gm = b.name === HYPOTHETICAL ? SUN_GM * (changes.extra_body?.gm_suns ?? 0)
      : (init.bodies.find((x) => x.name === b.name)?.gm ?? 0) * (changes.mass_scale[b.name] ?? 1);
    const rk = pos(b);
    const d = [rk[0] - r[0], rk[1] - r[1], (rk[2] ?? 0) - (r[2] ?? 0)];
    const dn = norm(d), rkn = norm(rk);
    const a = [0, 1, 2].map((i) => gm * (d[i] / dn ** 3 - (rk[i] ?? 0) / rkn ** 3));
    return { name: b.name, gm, share: norm(a) / sunPull, distance: dn };
  }).filter((x) => x.gm > 0).sort((a, b) => b.share - a.share);
  const gr = changes.gr ?? init.gr;
  const cScale = changes.c_scale ?? init.c_scale;
  // Relativity's size needs the velocity: a central difference of the drawn track (AU and days only).
  let grShare: number | null = null;
  if (gr && sim.units.length === "AU" && sim.units.time === "day") {
    const p0 = at(target, t), p1 = at(target, t + init.frame_step_days);
    const dt = p1[0] - p0[0];
    if (dt > 0) {
      const v = [1, 2, 3].map((i) => (p1[i] - p0[i]) / dt);
      const rn = norm(r), v2 = v[0] ** 2 + v[1] ** 2 + v[2] ** 2, rv = r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
      const k = (init.central_gm * cScale) / (C_AU_PER_DAY ** 2 * rn ** 3);
      const a = [0, 1, 2].map((i) => k * ((4 * init.central_gm / rn - v2) * r[i] + 4 * rv * v[i]));
      grShare = norm(a) / sunPull;
    }
  }
  const latex = String.raw`\ddot{\mathbf r} = -\frac{${gmC}\,\mathbf r}{r^3}`
    + terms.slice(0, 6).map((x) => ` + ${bodyTerm(x.name)}`).join("")
    + (terms.length > 6 ? String.raw` + \dots` : "") + (gr ? String.raw` + \mathbf a_{\mathrm{GR}}` : "");
  const unit = sim.units.length;
  return (
    <section className="panel stack" style={{ gap: 10 }} aria-labelledby="eq-title">
      <h2 id="eq-title">Every force on {target.name}, now</h2>
      <p className="small muted">Newton&apos;s law for {target.name}, one term per body (click any body in the view to see its own); each body also pulls {sun ? "the Sun" : central},
        so its term subtracts that pull (the indirect term). Shares are of {sun ? "the Sun" : central}&apos;s pull at this
        moment: change a mass, switch a body off or add one, and its row changes.</p>
      <div className="law-scroll"><Tex tex={latex} display /></div>
      <div className="table-wrap">
        <table className="cards">
          <thead><tr><th>Pulling body</th><th>Its term</th><th>GM ({unit}³/{sim.units.time}²)</th><th>Distance ({unit})</th><th>Share of {sun ? "the Sun" : central}&apos;s pull</th></tr></thead>
          <tbody>
            <tr><td data-label="Pulling body">{central}</td><td data-label="Its term" className="law-cell"><Tex tex={String.raw`-\frac{${gmC}\,\mathbf r}{r^3}`} /></td>
              <td data-label="GM" className="num">{init.central_gm.toExponential(4)}</td>
              <td data-label="Distance" className="num">{norm(r).toFixed(3)}</td><td data-label="Share" className="num">1</td></tr>
            {terms.map((x) => (
              <tr key={x.name}>
                <td data-label="Pulling body">{x.name === HYPOTHETICAL ? "Hypothetical planet (X)" : x.name}
                  {(changes.mass_scale[x.name] ?? 1) !== 1 ? ` (mass ${changes.mass_scale[x.name]}×)` : ""}</td>
                <td data-label="Its term" className="law-cell"><Tex tex={bodyTerm(x.name)} /></td>
                <td data-label="GM" className="num">{x.gm.toExponential(3)}</td>
                <td data-label="Distance" className="num">{x.distance.toFixed(3)}</td>
                <td data-label="Share" className="num">{x.share.toExponential(2)}</td></tr>
            ))}
            {gr && (
              <tr><td data-label="Pulling body">Relativity (the Sun){cScale !== 1 ? `, ${cScale}× (fictional)` : ""}</td>
                <td data-label="Its term" className="law-cell"><Tex tex={String.raw`\mathbf a_{\mathrm{GR}} = ${cScale !== 1 ? `${cScale}\\,` : ""}${GR_TERM}`} /></td>
                <td data-label="GM" className="num" /><td data-label="Distance" className="num" />
                <td data-label="Share" className="num">{grShare === null ? "on" : grShare.toExponential(2)}</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// Which candidate laws were tried on the track and why each won or lost: the verification's own reasoning, by rule.
function Chosen({ d }: { d: Discovery }) {
  if (!d.ranking?.length) return null;
  const best = Math.min(...d.ranking.map((r) => r.bic));
  return (
    <details className="stack" open>
      <summary><strong>How the law was chosen</strong> <span className="small muted">every candidate, fitted on the first 75% and judged
        on the rest{d.n_eff ? `; BIC with ${Math.round(d.n_eff)} effective samples` : ""}</span></summary>
      <div className="table-wrap">
        <table className="cards">
          <thead><tr><th>Candidate law</th><th>Forecast error on the held-out quarter</th><th>BIC behind the best</th><th>Verdict and why</th></tr></thead>
          <tbody>
            {d.ranking.map((r) => (
              <tr key={r.title}>
                <td data-label="Candidate law">{r.title}</td>
                <td data-label="Held-out error" className="num">{Number.isFinite(r.heldout_nrmse) ? `${(r.heldout_nrmse * 100).toPrecision(2)}%` : "diverged"}</td>
                <td data-label="BIC behind" className="num">{Number.isFinite(r.bic) ? (r.bic - best).toFixed(1) : ""}</td>
                <td data-label="Verdict" className="small">{r.status ? <span className={`verdict ${r.status === "accepted" ? "accepted" : r.status === "tied" ? "tied" : "rejected"}`}>{r.status}</span> : null} {r.why}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">A law is accepted only if it forecasts the held-out quarter (and from four later starting
        points) and beats every rival by 10 BIC; within 10 rivals tie and the honest answer is that the data cannot tell
        them apart. A law that only adds terms to a simpler one is set aside when those terms don&apos;t improve it. In the lab the candidates come from a fixed list of gravity laws; on a filmed clip the reasoning model
        proposes them and the same rules decide.</p>
    </details>
  );
}

const KM_S_PER_AU_DAY = 149597870.7 / 86400;

// Osculating orbit from one exact state (heliocentric AU and AU/day): what the orbit would be if only the central body
// pulled from this moment. Tilt is measured from the target's orbital plane (the view's flat plane).
function orbitOf(state: number[], gm: number, normal: number[]) {
  const [x, y, z, vx, vy, vz] = state;
  const r = Math.hypot(x, y, z), v = Math.hypot(vx, vy, vz);
  const a = 1 / (2 / r - v * v / gm);
  const h = [y * vz - z * vy, z * vx - x * vz, x * vy - y * vx];
  const hn = Math.hypot(h[0], h[1], h[2]);
  const rv = x * vx + y * vy + z * vz;
  const ev = [0, 1, 2].map((i) => ((v * v - gm / r) * state[i] - rv * state[3 + i]) / gm);
  const e = Math.hypot(ev[0], ev[1], ev[2]);
  const tilt = (Math.acos(Math.max(-1, Math.min(1, (h[0] * normal[0] + h[1] * normal[1] + h[2] * normal[2]) / hn))) * 180) / Math.PI;
  const bound = a > 0 && e < 1;
  return { r, speed: v * KM_S_PER_AU_DAY, a, e, tilt, bound,
           peri: bound ? a * (1 - e) : (hn * hn / gm) / (1 + e), apo: bound ? a * (1 + e) : Infinity,
           period: bound ? 2 * Math.PI * Math.sqrt(a ** 3 / gm) : Infinity,
           escape: Math.sqrt(2 * gm / r) * KM_S_PER_AU_DAY };
}

function seen(points: number[][]) {
  const d = points.map((p) => Math.hypot(p[1], p[2], p[3]));
  return { near: Math.min(...d), far: Math.max(...d), d, t: points.map((p) => p[0]) };
}

function DistanceChart({ name, base, modified, unit }: { name: string; base: number[][]; modified: number[][] | null; unit: string }) {
  const W = 560, H = 150, L = 44, B = 22, T = 8, R = 8;
  const b = seen(base), m = modified ? seen(modified) : null;
  const t1 = Math.max(b.t[b.t.length - 1], m ? m.t[m.t.length - 1] : 0) || 1;
  const lo = Math.min(b.near, m?.near ?? Infinity) * 0.95, hi = Math.max(b.far, m && Number.isFinite(m.far) ? m.far : 0) * 1.05;
  const X = (t: number) => L + (t / t1) * (W - L - R), Y = (v: number) => T + (1 - (v - lo) / (hi - lo || 1)) * (H - T - B);
  const line = (s: ReturnType<typeof seen>) => s.t.map((t, i) => `${i ? "L" : "M"}${X(t).toFixed(1)},${Y(s.d[i]).toFixed(1)}`).join("");
  return (
    <figure className="mini-chart" style={{ maxWidth: 640, width: "100%" }}>
      <figcaption><strong>Distance from the centre over the run</strong> <span className="muted small">{unit}; years along the bottom</span></figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${name}: distance from ${b.near.toFixed(3)} to ${b.far.toFixed(3)} ${unit}${m ? `; with your changes ${m.near.toFixed(3)} to ${m.far.toFixed(3)}` : ""}`}>
        {[lo, (lo + hi) / 2, hi].map((v) => <text key={v} x={L - 4} y={Y(v) + 4} textAnchor="end" className="tick">{v.toPrecision(3)}</text>)}
        {[0, 0.5, 1].map((f) => <text key={f} x={X(f * t1)} y={H - 6} textAnchor="middle" className="tick">{(f * t1 / 365.25).toFixed(1)}</text>)}
        <path d={line(b)} className="mini-base" />
        {m && <path d={line(m)} className="mini-modified" />}
      </svg>
    </figure>
  );
}

const PUSH_DIRS = [["along its motion", "faster (+) or slower (−)"], ["outward", "away from (+) or toward (−) the centre"],
  ["out of the plane", "up (+) or down (−)"]] as const;

function BodyCard({ name, base, modified, init, changes, central, onKick, onClose }: {
  name: string; base: Simulation; modified: Simulation | null; init: NonNullable<Simulation["initial"]>; changes: Changes;
  central: string; onKick: (k: [number, number, number]) => void; onClose: (() => void) | null;
}) {
  const start = init.bodies.find((b) => b.name === name);
  const bPts = base.bodies.find((b) => b.name === name)?.points ?? [];
  const mPts = modified?.bodies.find((b) => b.name === name)?.points ?? null;
  const kick = changes.kicks?.[name] ?? [0, 0, 0];
  const pushed = kick.some((v) => v !== 0);
  if (!start || !bPts.length) return null;
  const normal = init.basis[2];
  const o0 = orbitOf(start.state, init.central_gm, normal);
  const o1 = pushed ? orbitOf(kicked(start.state, kick), init.central_gm, normal) : null;
  const unit = base.units.length, tu = base.units.time;
  const s0 = seen(bPts), s1 = mPts ? seen(mPts) : null;
  const scale = changes.mass_scale[name] ?? 1;
  const fmt = (v: number, d = 4) => (Number.isFinite(v) ? v.toPrecision(d) : "none (it escapes)");
  const rows: [string, (o: ReturnType<typeof orbitOf>) => string][] = [
    [`Size of the orbit, a (${unit})`, (o) => (o.bound ? fmt(o.a) : "unbound")],
    ["Eccentricity, e (0 is a circle)", (o) => fmt(o.e, 3)],
    [`Closest to ${central} (${unit})`, (o) => fmt(o.peri)],
    [`Farthest from ${central} (${unit})`, (o) => fmt(o.apo)],
    [`Period (${tu}s, Kepler's third law)`, (o) => (o.bound ? `${fmt(o.period, 5)}${tu === "day" ? ` (${(o.period / 365.25).toPrecision(3)} yr)` : ""}` : "none")],
    [`Tilt from ${base.target}'s orbital plane (degrees)`, (o) => o.tilt.toFixed(2)],
    ["Speed at the start (km/s)", (o) => o.speed.toFixed(3)],
  ];
  return (
    <section className="panel stack" style={{ gap: 12 }} aria-labelledby="body-title">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8 }}>
        <h2 id="body-title">{name === HYPOTHETICAL ? "Hypothetical planet (X)" : name}{name === base.target ? " (the body whose law is found)" : ""}</h2>
        {onClose && <button type="button" className="secondary" onClick={onClose}>Back to {base.target}</button>}
      </div>
      <p className="small muted">Click any body in the view, or its name, to see it here. Its orbit from its exact starting position and
        velocity{start.gm > 0 ? `; its pull, GM = ${(start.gm * scale).toExponential(3)} ${unit}³/${tu}²${scale !== 1 ? ` (${scale}× real)` : ""}, is in every other body's equation` : "; it is too light to pull on anything"}.</p>
      {!o1?.bound && o1 && <p className="badge bad">With this push {name} is unbound: faster than the {o1.escape.toFixed(1)} km/s escape speed here, it leaves {central} for good.</p>}
      <div className="table-wrap">
        <table className="cards">
          <thead><tr><th>Orbit</th><th>As written</th>{o1 && <th>After your push</th>}</tr></thead>
          <tbody>
            {rows.map(([label, f]) => (
              <tr key={label}><td data-label="Orbit">{label}</td><td data-label="As written" className="num">{f(o0)}</td>
                {o1 && <td data-label="After your push" className="num">{f(o1)}</td>}</tr>
            ))}
            <tr><td data-label="Orbit">Seen over the run: nearest to farthest ({unit})</td>
              <td data-label="As written" className="num">{s0.near.toPrecision(4)} to {s0.far.toPrecision(4)}</td>
              {o1 && <td data-label="After your push" className="num">{s1 ? `${s1.near.toPrecision(4)} to ${s1.far.toPrecision(4)}` : "computing"}</td>}</tr>
          </tbody>
        </table>
      </div>
      <p className="small muted">The rows above use the central body alone (Kepler); the last row is the full simulation, every pull
        included, so the two differ where the other bodies matter.</p>
      <DistanceChart name={name} base={bPts} modified={mPts} unit={unit} />
      <fieldset className="stack" style={{ gap: 8 }}>
        <legend>Push {name} (changes its starting velocity)</legend>
        {PUSH_DIRS.map(([dir, hint], i) => (
          <label key={dir} className="small sky-push">
            <span>{dir} <span className="muted">({hint})</span></span>
            <input type="range" min={-15000} max={15000} step={10} value={kick[i]}
                   aria-valuetext={`${kick[i]} metres per second ${dir}`}
                   onChange={(e) => { const k = [...kick] as [number, number, number]; k[i] = Number(e.target.value); onKick(k); }} />
            <input type="number" step={10} min={-20000} max={20000} value={kick[i]} aria-label={`${dir}, metres per second`}
                   onChange={(e) => { const k = [...kick] as [number, number, number]; k[i] = Math.max(-20000, Math.min(20000, Number(e.target.value) || 0)); onKick(k); }} />
            <span className="muted">m/s</span>
          </label>
        ))}
        <p className="small muted">For scale: {name} moves at {o0.speed.toFixed(1)} km/s; escaping {central} from here takes {o0.escape.toFixed(1)} km/s.
          A spacecraft&apos;s engine gives a few km/s; a planet has never been pushed like this, so any push is fictional.</p>
        {pushed && <button type="button" className="secondary" onClick={() => onKick([0, 0, 0])}>No push</button>}
      </fieldset>
    </section>
  );
}

const GREEK = new Set(["mu", "nu", "phi", "omega", "gamma", "beta", "alpha", "theta", "lambda", "sigma", "tau"]);

// Real questions each scenario answers, one click each.
const USES: { title: string; text: string; scenario: string; changes?: Partial<Changes>; link?: [string, string]; pick?: string }[] = [
  { title: "Will a comet come back on time?", scenario: "halley_with_jupiter_saturn", changes: { disabled: ["Jupiter"] },
    text: "Halley returns every 75 to 76 years, and the giant planets decide the exact date. Jupiter is switched off here: compare the period with the scenario as written.",
    link: ["/?run=gallery-halley", "Halley's law, found from a video"] },
  { title: "How does a near miss change an asteroid?", scenario: "apophis_with_planets",
    text: "Apophis passes 38,000 km from Earth in April 2029. The pull of every planet is in its equation; switch Earth off to see how much of the motion it explains.",
    link: ["/?run=gallery-apophis", "Apophis's law, found from a video"] },
  { title: "Is there a planet nobody has seen?", scenario: "uranus_neptune_hidden",
    text: "Le Verrier found Neptune in 1846 from the way Uranus drifted. Neptune pulls here but is not drawn: find the law, then read the residuals for its signature.", },
  { title: "Is Newton enough? Testing gravity", scenario: "mercury_gr_exaggerated",
    text: "Mercury's perihelion turns 43 arcseconds a century more than Newton allows; relativity explains it. Here it is made 20,000 times stronger, so a few years show it.", },
  { title: "Would the inner planets survive a heavier Jupiter?", scenario: "jupiter_x10",
    text: "A fictional Jupiter ten times heavier: how far the inner orbits are pushed, as a stress test for a planetary system's stability.", },
  { title: "Moons locked in rhythm", scenario: "galilean_resonance",
    text: "Io, Europa and Ganymede orbit in a 1:2:4 resonance that missions such as Juno and Europa Clipper plan around; Jupiter's flattening (J2) is in their equations.", },
  { title: "How much push gets you from Earth to Mars?", scenario: "real_inner_planets", pick: "Earth",
    changes: { kicks: { Earth: [2940, 0, 0] } },
    text: "A spacecraft leaving Earth's orbit needs about 2.9 km/s more along its motion to coast out to Mars's distance (a Hohmann transfer, the route most Mars missions fly). Here Earth itself gets that push: its farthest point moves out to Mars's orbit, and the Moon is left behind.", },
  { title: "How fast to leave the Sun for good?", scenario: "real_inner_planets", pick: "Earth",
    changes: { kicks: { Earth: [12400, 0, 0] } },
    text: "From Earth's distance the Sun's escape speed is about 42 km/s; Earth moves at about 30. Add 12.4 km/s along its motion and the orbit opens: no farthest point, no period. Try 12,000 to see it stay bound.", },
];

export default function SkyLab() {
  const online = useBackend();
  const health = useHealth();
  const [scenarios, setScenarios] = useState<ScenarioSummary[]>([]);
  const [name, setName] = useState<string>("");
  const [changes, setChanges] = useState<Changes>(NONE);
  const [base, setBase] = useState<Simulation | null>(null);
  const [modified, setModified] = useState<Simulation | null>(null);
  const [discovery, setDiscovery] = useState<Discovery | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(30);
  const [follow, setFollow] = useState<string | null>(null);
  const [time, setTime] = useState<[number, number]>([0, 0]);
  const [noise, setNoise] = useState(2);
  const [filmed, setFilmed] = useState<string | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const view = useRef<Sky3DHandle>(null);
  const request = useRef(0);

  // Start from the link (#name, or ?s= with every change), else the first scenario. Motion only if the visitor allows it.
  useEffect(() => {
    if (online === null) return;
    listScenarios(online).then((list) => {
      setScenarios(list);
      const token = new URLSearchParams(window.location.search).get("s");
      const state = token ? decodeState(token) : null;
      const hash = window.location.hash.slice(1);
      const start = state?.s ?? (list.some((s) => s.name === hash) ? hash : list[0]?.name ?? "");
      setName(start);
      if (state && state.s === start) setChanges(state.c);
    }).catch((e) => setError(`The scenarios could not be loaded: ${(e as Error).message}`));
    setPlaying(!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);
  }, [online]);

  // The scenario as written: the server's run when online, else the bundled one.
  useEffect(() => {
    if (!name) return;
    setBase(null);
    setDiscovery(null);
    const load = online ? simulateOnServer(name, NONE).catch(() => bundled(name)) : bundled(name);
    load.then((sim) => {
      if (!sim.initial) bundled(name).then((b) => setBase({ ...sim, initial: b.initial })).catch(() => setBase(sim));
      else setBase(sim);
    }).catch((e) => setError(`This scenario could not be loaded: ${(e as Error).message}`));
  }, [name, online]);

  // Every change: an instant preview in the browser, then (with a server) the authoritative run and the law, 400 ms
  // after the last change.
  useEffect(() => {
    if (!base) return;
    const params = new URLSearchParams(window.location.search);
    if (changed(changes)) params.set("s", encodeState(name, changes)); else params.delete("s");
    const q = params.toString();
    window.history.replaceState(null, "", `${window.location.pathname}${q ? `?${q}` : ""}#${name}`);
    if (!changed(changes)) {
      setModified(null);
      return;
    }
    const id = ++request.current;
    if (base.initial && (changes.years ?? base.initial.clip_days / 365.25) <= 50 && base.initial.bodies.length <= 10) {
      const bodies = integrate(base.initial, changes, base.target).map((b) => ({
        ...b, colour: base.bodies.find((x) => x.name === b.name)?.colour ?? "#cccccc",
      }));
      setModified({ ...base, bodies, elements: [], detectability: null, preview: true, source: "simulation, no pixels (in this browser)" });
    }
    if (!online) return;
    const timer = setTimeout(() => {
      setBusy("Simulating on the server");
      simulateOnServer(name, changes).then((sim) => {
        if (id !== request.current) return;
        setModified(sim);
        setBusy("Finding the law from the simulated track");
        return discoverOnServer(name, changes, noise).then((d) => id === request.current && setDiscovery(d));
      }).catch((e) => {
        if (id !== request.current) return;
        // A server older than pushes refuses them: the browser's run stands, and says so, instead of an error.
        if (Object.keys(changes.kicks ?? {}).length) setModified((m) => (m ? { ...m, source: "simulation, no pixels (in this browser; the server cannot run pushes yet)" } : m));
        else setError((e as Error).message);
      }).finally(() => id === request.current && setBusy(null));
    }, 400);
    return () => clearTimeout(timer);
  }, [base, changes, name, online, noise]);

  const discover = useCallback(() => {
    if (!name) return;
    if (!online) {
      // No server: the scenario as written has its law found in advance; a changed one needs the server.
      if (changed(changes)) return;
      bundledDiscovery(name).then(setDiscovery).catch((e) => setError(`The bundled law could not be loaded: ${(e as Error).message}`));
      return;
    }
    setBusy("Finding the law from the simulated track");
    discoverOnServer(name, changes, noise).then(setDiscovery).catch((e) => setError((e as Error).message)).finally(() => setBusy(null));
  }, [online, name, changes, noise]);

  const load = (scenario: string, preset?: Partial<Changes>, pick?: string) => {
    setChanges({ ...NONE, ...preset });
    setName(scenario);
    setPicked(pick ?? null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const sc = scenarios.find((s) => s.name === name);
  const others = useMemo(() => (base?.initial?.bodies ?? []).map((b) => b.name).filter((n) => n !== base?.target), [base]);
  const shown = modified ?? base;
  const subject = picked && (shown?.bodies ?? []).some((b) => b.name === picked) ? picked : (base?.target ?? null);
  const fictional = !!sc?.fictional || (changes.c_scale !== undefined && changes.c_scale !== 1) || changed(changes);
  const set = (patch: Partial<Changes>) => { setDiscovery(null); setChanges((c) => ({ ...c, ...patch })); };
  const label = `${sc?.title ?? "Solar system"}: ${(shown?.bodies ?? []).filter((b) => !b.hidden).map((b) => b.name).join(", ")} `
    + `around the ${sc?.name.startsWith("galilean") ? "planet" : "Sun"}${modified ? ", the changed run dashed over the original" : ""}.`;

  return (
    <main id="main" className="stack" style={{ gap: 20 }}>
      <header className="stack" style={{ gap: 6 }}>
        <h1>Solar System Lab</h1>
        <p className="lead">
          Change the solar system and watch both the orbits and the law change. Simulated by PhysicsLens&apos;s own N-body
          engine, checked against NASA/JPL; the law is found from a simulated track with tracker-like noise, not from video.
        </p>
      </header>
      {error && <p className="error" role="alert">{error}</p>}
      <section className="panel stack" style={{ gap: 12 }} aria-labelledby="uses-title">
        <h2 id="uses-title">Start from a real-world question</h2>
        <p className="small muted">The same equations astronomers and mission planners integrate, on real starting positions from JPL.
          Each button loads a scenario, with a change where the question needs one.</p>
        <div className="sky-uses">
          {USES.filter((u) => scenarios.some((s) => s.name === u.scenario)).map((u) => (
            <article key={u.title} className="sky-use">
              <h3>{u.title}</h3>
              <p className="small">{u.text}</p>
              <div className="row" style={{ gap: 10, flexWrap: "wrap", alignItems: "center" }}>
                <button type="button" className="secondary" onClick={() => load(u.scenario, u.changes, u.pick)}>Load it</button>
                {u.link && <a className="small" href={u.link[0]}>{u.link[1]}</a>}
              </div>
            </article>
          ))}
        </div>
      </section>
      <div className="sky-layout">
        <section className="sky-stage panel" aria-label="The simulation">
          <Sky3D handle={view} base={base?.bodies ?? null} modified={modified?.bodies ?? null} playing={playing}
                 daysPerSecond={speed} follow={follow} label={label} centre={sc?.name.startsWith("galilean") ? "Jupiter" : "Sun"}
                 onTime={(t, end) => setTime([t, end])} onPick={setPicked} picked={subject} />
          <p className="small muted">Drag to turn, scroll or pinch to zoom, right-drag to pan. Sizes are not to scale (at
            scale every planet would be smaller than a pixel); positions are. The orange and blue dots are an illustrative
            swarm of asteroids on Kepler orbits, as in NASA&apos;s charts.</p>
          <div className="sky-transport">
            <button type="button" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause" : "Play"}>
              {playing ? <Pause size={16} aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}
            </button>
            <label className="sky-scrub">
              <span className="visually-hidden">Time</span>
              <input type="range" min={0} max={time[1] || 1} step={Math.max(time[1] / 1000, 0.01)} value={time[0]}
                     onChange={(e) => { setPlaying(false); view.current?.setTime(Number(e.target.value)); }} />
            </label>
            <span className="small muted">{(time[0] / 365.25).toFixed(1)} / {(time[1] / 365.25).toFixed(1)} yr</span>
            <label className="small">
              <span className="visually-hidden">Speed</span>
              <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
                {SPEEDS.map((s) => <option key={s} value={s}>{s === 365 ? "1 year/s" : `${s} days/s`}</option>)}
              </select>
            </label>
            <label className="small">
              <Crosshair size={14} aria-hidden="true" /> <span className="visually-hidden">Follow</span>
              <select value={follow ?? ""} onChange={(e) => setFollow(e.target.value || null)}>
                <option value="">Sun-centred</option>
                {(shown?.bodies ?? []).map((b) => <option key={b.name} value={b.name}>Follow {b.name}</option>)}
              </select>
            </label>
            <button type="button" className="secondary" onClick={() => view.current?.fit()}>Fit</button>
          </div>
          <div className="row sky-pick" role="group" aria-label="See one body: its orbit, its equation, push it">
            <span className="small muted">See:</span>
            {(shown?.bodies ?? []).filter((b) => !b.hidden).map((b) => (
              <button key={b.name} type="button" className={`chip${b.name === subject ? " on" : ""}`} aria-pressed={b.name === subject}
                      onClick={() => setPicked(b.name)}>{b.name === HYPOTHETICAL ? "planet X" : b.name}</button>
            ))}
          </div>
          {shown?.detectability && (
            <p className={`badge ${shown.detectability.visible ? "ok" : "tie"}`}>
              A phone-video tracker {shown.detectability.visible ? "would" : "would not"} see this: {shown.detectability.message}
            </p>
          )}
          {modified?.preview && <p className="small muted">{modified.source.includes("cannot run pushes") ? "Computed in this browser (the analysis server cannot run pushes yet, so finding the law of a pushed system waits for it)." : `Preview computed in this browser${online ? "; the server's run replaces it when it arrives." : "."}`}</p>}
        </section>

        <aside className="sky-controls panel stack" style={{ gap: 14 }} aria-label="Change the system">
          <label className="field">
            Scenario
            <select value={name} onChange={(e) => { setChanges(NONE); setPicked(null); setName(e.target.value); }}>
              {scenarios.map((s) => <option key={s.name} value={s.name}>{s.title}</option>)}
            </select>
          </label>
          {sc && <p className="small">{sc.description}</p>}
          {fictional && <p className="badge bad">FICTIONAL: {changed(changes) ? "changed from the real system" : "not the real solar system"}</p>}
          <fieldset className="stack" style={{ gap: 8 }}>
            <legend>Bodies</legend>
            {others.map((n) => {
              const off = changes.disabled.includes(n);
              const scale = changes.mass_scale[n] ?? sc?.mass_scale[n] ?? 1;
              return (
                <div key={n} className="sky-body-row">
                  <label><input type="checkbox" checked={!off}
                                onChange={(e) => set({ disabled: e.target.checked ? changes.disabled.filter((x) => x !== n) : [...changes.disabled, n] })} /> {n}</label>
                  <label className="small">
                    <span className="visually-hidden">{n} mass</span>
                    <input type="range" min={-2} max={2} step={0.05} value={Math.log10(Math.max(scale, 0.01))} disabled={off}
                           aria-valuetext={`${scale.toPrecision(3)} times its real mass`}
                           onChange={(e) => set({ mass_scale: { ...changes.mass_scale, [n]: Number((10 ** Number(e.target.value)).toPrecision(3)) } })} />
                    <span>{scale.toPrecision(3)}×</span>
                  </label>
                </div>
              );
            })}
          </fieldset>
          <fieldset className="stack" style={{ gap: 8 }}>
            <legend>Relativity</legend>
            <label><input type="checkbox" checked={changes.gr ?? sc?.gr ?? false} onChange={(e) => set({ gr: e.target.checked })} /> The Sun&apos;s general relativity</label>
            <label className="small">
              Strength (1 is real; anything else is fictional)
              <input type="range" min={0} max={5} step={0.1} value={Math.log10(changes.c_scale ?? sc?.c_scale ?? 1)}
                     aria-valuetext={`${(changes.c_scale ?? sc?.c_scale ?? 1).toPrecision(3)} times real`}
                     onChange={(e) => set({ c_scale: Number((10 ** Number(e.target.value)).toPrecision(3)), gr: true })} />
              <span>{(changes.c_scale ?? sc?.c_scale ?? 1).toPrecision(3)}×</span>
            </label>
          </fieldset>
          <fieldset className="stack" style={{ gap: 8 }}>
            <legend>Add a hypothetical body</legend>
            <label><input type="checkbox" checked={!!changes.extra_body}
                          onChange={(e) => set({ extra_body: e.target.checked ? { gm_suns: 0.001, a_au: 3, e: 0.05, i_deg: 0, M_deg: 0 } : null })} /> A planet that isn&apos;t there</label>
            {changes.extra_body && (["gm_suns", "a_au", "e", "M_deg"] as const).map((k) => (
              <label key={k} className="field small">
                {{ gm_suns: "Mass (fraction of the Sun's)", a_au: "Distance (AU)", e: "Eccentricity", M_deg: "Phase (degrees)" }[k]}
                <input type="number" step="any" value={changes.extra_body![k]}
                       onChange={(e) => set({ extra_body: { ...changes.extra_body!, [k]: Number(e.target.value) } })} />
              </label>
            ))}
          </fieldset>
          <button type="button" className="secondary" onClick={() => setChanges(NONE)} disabled={!changed(changes)}>
            <RotateCcw size={14} aria-hidden="true" /> Back to the scenario as written
          </button>
        </aside>
      </div>

      {base?.initial && subject && (
        <BodyCard name={subject} base={base} modified={modified} init={base.initial} changes={changes}
                  central={sc?.name.startsWith("galilean") ? "Jupiter" : "the Sun"}
                  onKick={(k) => set({ kicks: { ...(changes.kicks ?? {}), [subject]: k } })}
                  onClose={subject !== base.target ? () => setPicked(null) : null} />
      )}

      <section className="panel stack" style={{ gap: 10 }} aria-labelledby="changed-title">
        <h2 id="changed-title">What changed: {base?.target ?? "the target"}&apos;s orbit</h2>
        {shown?.precession_arcsec_per_century != null && (
          <p>Its perihelion turns by <strong>{shown.precession_arcsec_per_century.toLocaleString(undefined, { maximumSignificantDigits: 4 })}</strong> arcseconds a century{modified && base?.precession_arcsec_per_century != null ? ` (as written: ${base.precession_arcsec_per_century.toLocaleString(undefined, { maximumSignificantDigits: 4 })})` : ""}.</p>
        )}
        <div className="mini-charts">
          {base && <MiniChart title="Semi-major axis" unit="AU" base={base.elements} modified={modified?.elements?.length ? modified.elements : null} pick={(e) => e.a} />}
          {base && <MiniChart title="Eccentricity" unit="" base={base.elements} modified={modified?.elements?.length ? modified.elements : null} pick={(e) => e.e} />}
          {base && <MiniChart title="Perihelion longitude" unit="degrees" base={base.elements} modified={modified?.elements?.length ? modified.elements : null} pick={(e) => e.peri_longitude_deg} />}
          {base && <MiniChart title="Period" unit="days" base={base.elements} modified={modified?.elements?.length ? modified.elements : null} pick={(e) => e.period_days} />}
        </div>
        <p className="small muted">Solid: the scenario as written. Dashed: with your changes.</p>
      </section>

      {base?.initial && <Equations sim={modified ?? base} init={base.initial} changes={changes} t={time[0]}
                                   central={sc?.name.startsWith("galilean") ? "Jupiter" : "Sun"} subject={subject ?? base.target} />}

      <section className="panel stack" style={{ gap: 10 }} aria-labelledby="law-title">
        <h2 id="law-title">The law, found from the motion</h2>
        <p className="small muted">Found from a simulated track with tracker-like noise, not from video: the same fitting and held-out verification as a recorded clip, from a fixed list of laws (free).</p>
        <div className="row" style={{ gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <label className="small">Tracker noise
            <select value={noise} onChange={(e) => setNoise(Number(e.target.value))}>
              {[2, 0.5, 0.1, 0.02].map((v) => <option key={v} value={v}>{v} px{v === 2 ? " (a phone video)" : ""}</option>)}
            </select>
          </label>
          <button type="button" onClick={discover} disabled={(!online && changed(changes)) || !!busy}>Find the law</button>
          {busy && <span className="small muted" role="status">{busy}</span>}
        </div>
        {!online && <p className="note">The analysis server is offline right now. For the scenario as written, the law was found in
          advance (at 2 px of noise) and shows here; with your changes the orbits still run in this browser, but finding
          their law needs the server.</p>}
        {discovery?.noise_px !== undefined && <p className="small muted">Found in advance for the scenario as written, at {discovery.noise_px} px of tracker noise.</p>}
        <div aria-live="polite" className="visually-hidden">{discovery?.law ? `Accepted law: ${discovery.law.title}` : ""}</div>
        {discovery && (
          <div className="stack" style={{ gap: 8 }}>
            <p><span className={`verdict ${discovery.status === "accepted" ? "accepted" : "rejected"}`}>{discovery.status}</span> {discovery.reason}</p>
            {discovery.law && <><h3>{discovery.law.title}</h3><Law tex={discovery.law.latex} /></>}
            {discovery.parameters && (
              <ul className="small">
                {discovery.parameters.map((p) => <li key={p.name}><Tex tex={GREEK.has(p.name) ? `\\${p.name}` : p.name} /> = {p.value.toPrecision(7)} ± {p.std.toPrecision(2)}</li>)}
                {discovery.bootstrap_reps ? <li className="muted">intervals from {discovery.bootstrap_reps} bootstrap resamples</li> : null}
              </ul>
            )}
            <Chosen d={discovery} />
            {discovery.perturbation && <p className="note">{discovery.perturbation}</p>}
            {discovery.residual_spectrum && (
              <p className="small">Residuals: {discovery.residual_spectrum.significant ? "structured" : "like noise"}
                {discovery.residual_spectrum.peaks[0] ? `; strongest peak ${discovery.residual_spectrum.peaks[0].period.toPrecision(4)} days (${discovery.residual_spectrum.peaks[0].component}, false-alarm probability ${discovery.residual_spectrum.peaks[0].false_alarm_probability.toExponential(1)})` : ""}
                {discovery.residual_spectrum.period_matches[0] ? `, near ${discovery.residual_spectrum.period_matches[0].body}'s ${discovery.residual_spectrum.period_matches[0].kind}` : ""}.</p>
            )}
            {discovery.equation_diff && (
              <div className="stack" style={{ gap: 6 }}>
                <h3>How the equation changed</h3>
                <p className="small">Before: {discovery.equation_diff.before.title}</p>
                <Law tex={discovery.equation_diff.before.latex} />
                <p className="small">After: {discovery.equation_diff.after.title}</p>
                <Law tex={discovery.equation_diff.after.latex} />
                <p>{discovery.equation_diff.summary}</p>
              </div>
            )}
          </div>
        )}
        {online && health?.can_upload && (
          <div className="row" style={{ gap: 10, alignItems: "center" }}>
            <button type="button" className="secondary" disabled={!!busy} onClick={() => {
              setBusy("Rendering the video and starting the analysis");
              filmOnServer(name, changes).then((r) => setFilmed(r.run_id)).catch((e) => setError((e as Error).message)).finally(() => setBusy(null));
            }}><Clapperboard size={14} aria-hidden="true" /> Film it: analyze the rendered video (SAM 3.1 and Muse Spark Open)</button>
            {filmed && <a href={`/?run=${filmed}`}>Open the analysis</a>}
          </div>
        )}
      </section>
    </main>
  );
}
