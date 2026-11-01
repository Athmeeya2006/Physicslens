"use client";

import { Clapperboard, Crosshair, Pause, Play, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Law, Tex } from "@/components/Equation";
import { Sky3D, type Sky3DHandle } from "@/components/Sky3D";
import { useBackend, useHealth } from "@/lib/api";
import {
  bundled, changed, type Changes, decodeState, type Discovery, discoverOnServer, type Elements, encodeState, filmOnServer,
  integrate, listScenarios, type ScenarioSummary, type Simulation, simulateOnServer,
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
// relativity when it is on. Beside each term its size right now, as a share of the Sun's pull, from the positions on
// screen: change a mass or remove a body and its term changes with it.
function Equations({ sim, init, changes, t }: { sim: Simulation; init: NonNullable<Simulation["initial"]>; changes: Changes; t: number }) {
  const target = sim.bodies.find((b) => b.target);
  if (!target) return null;
  const pos = (b: { points: number[][] }) => {
    const pts = b.points;
    let k = pts.findIndex((p) => p[0] > t);
    if (k < 0) k = pts.length - 1;
    return pts[Math.max(0, k)].slice(1, 4);
  };
  const r = pos(target);
  const norm = (v: number[]) => Math.hypot(v[0], v[1], v[2] ?? 0);
  const sunPull = init.central_gm / norm(r) ** 2;
  const terms = sim.bodies.filter((b) => !b.target).map((b) => {
    const gm0 = init.bodies.find((x) => x.name === b.name)?.gm ?? 0;
    const gm = gm0 * (changes.mass_scale[b.name] ?? 1);
    const rk = pos(b);
    const d = [rk[0] - r[0], rk[1] - r[1], (rk[2] ?? 0) - (r[2] ?? 0)];
    const dn = norm(d), rkn = norm(rk);
    const a = [0, 1, 2].map((i) => gm * (d[i] / dn ** 3 - (rk[i] ?? 0) / rkn ** 3));
    return { name: b.name, gm, share: norm(a) / sunPull, distance: dn };
  }).filter((x) => x.gm > 0).sort((a, b) => b.share - a.share);
  const gr = changes.gr ?? init.gr;
  const sym = (n: string) => n.replace(/[^A-Za-z]/g, "").slice(0, 3);
  const latex = String.raw`\ddot{\mathbf r} = -\frac{GM_\odot\,\mathbf r}{r^3}` + terms.slice(0, 6).map((x) =>
    String.raw` + GM_{\mathrm{${sym(x.name)}}}\left(\frac{\mathbf r_{\mathrm{${sym(x.name)}}} - \mathbf r}{|\mathbf r_{\mathrm{${sym(x.name)}}} - \mathbf r|^3} - \frac{\mathbf r_{\mathrm{${sym(x.name)}}}}{r_{\mathrm{${sym(x.name)}}}^3}\right)`).join("")
    + (terms.length > 6 ? String.raw` + \dots` : "") + (gr ? String.raw` + \mathbf a_{\mathrm{GR}}` : "");
  return (
    <section className="panel stack" style={{ gap: 10 }} aria-labelledby="eq-title">
      <h2 id="eq-title">Every force on {target.name}, now</h2>
      <p className="small muted">Newton&apos;s law for {target.name}, one term per body; each body also pulls the Sun, so its
        term subtracts that pull (the indirect term). Shares are of the Sun&apos;s pull at this moment.</p>
      <div className="law-scroll"><Tex tex={latex} display /></div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Pulling body</th><th>Distance (AU)</th><th>Pull, as a share of the Sun&apos;s</th></tr></thead>
          <tbody>
            <tr><td>Sun</td><td className="num">{norm(r).toFixed(3)}</td><td className="num">1</td></tr>
            {terms.map((x) => (
              <tr key={x.name}><td>{x.name}{(changes.mass_scale[x.name] ?? 1) !== 1 ? ` (mass ${changes.mass_scale[x.name]}×)` : ""}</td>
                <td className="num">{x.distance.toFixed(3)}</td><td className="num">{x.share.toExponential(2)}</td></tr>
            ))}
            {gr && <tr><td>Relativity (the Sun){(changes.c_scale ?? init.c_scale) !== 1 ? `, ${(changes.c_scale ?? init.c_scale)}× (fictional)` : ""}</td><td /><td className="num">on</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}

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
      }).catch((e) => id === request.current && setError((e as Error).message))
        .finally(() => id === request.current && setBusy(null));
    }, 400);
    return () => clearTimeout(timer);
  }, [base, changes, name, online, noise]);

  const discover = useCallback(() => {
    if (!online || !name) return;
    setBusy("Finding the law from the simulated track");
    discoverOnServer(name, changes, noise).then(setDiscovery).catch((e) => setError((e as Error).message)).finally(() => setBusy(null));
  }, [online, name, changes, noise]);

  const sc = scenarios.find((s) => s.name === name);
  const others = useMemo(() => (base?.initial?.bodies ?? []).map((b) => b.name).filter((n) => n !== base?.target), [base]);
  const shown = modified ?? base;
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
      <div className="sky-layout">
        <section className="sky-stage panel" aria-label="The simulation">
          <Sky3D handle={view} base={base?.bodies ?? null} modified={modified?.bodies ?? null} playing={playing}
                 daysPerSecond={speed} follow={follow} label={label} centre={sc?.name.startsWith("galilean") ? "Jupiter" : "Sun"}
                 onTime={(t, end) => setTime([t, end])} />
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
          {shown?.detectability && (
            <p className={`badge ${shown.detectability.visible ? "ok" : "tie"}`}>
              A phone-video tracker {shown.detectability.visible ? "would" : "would not"} see this: {shown.detectability.message}
            </p>
          )}
          {modified?.preview && <p className="small muted">Preview computed in this browser{online ? "; the server's run replaces it when it arrives." : "."}</p>}
        </section>

        <aside className="sky-controls panel stack" style={{ gap: 14 }} aria-label="Change the system">
          <label className="field">
            Scenario
            <select value={name} onChange={(e) => { setChanges(NONE); setName(e.target.value); }}>
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

      {base?.initial && <Equations sim={modified ?? base} init={base.initial} changes={changes} t={time[0]} />}

      <section className="panel stack" style={{ gap: 10 }} aria-labelledby="law-title">
        <h2 id="law-title">The law, found from the motion</h2>
        <p className="small muted">Found from a simulated track with tracker-like noise, not from video: the same fitting and held-out verification as a recorded clip, from a fixed list of laws (free).</p>
        <div className="row" style={{ gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <label className="small">Tracker noise
            <select value={noise} onChange={(e) => setNoise(Number(e.target.value))}>
              {[2, 0.5, 0.1, 0.02].map((v) => <option key={v} value={v}>{v} px{v === 2 ? " (a phone video)" : ""}</option>)}
            </select>
          </label>
          <button type="button" onClick={discover} disabled={!online || !!busy}>Find the law</button>
          {busy && <span className="small muted" role="status">{busy}</span>}
        </div>
        {!online && <p className="note">The analysis server is offline right now, so the law cannot be found here; the orbits above still run in this browser.</p>}
        <div aria-live="polite" className="visually-hidden">{discovery?.law ? `Accepted law: ${discovery.law.title}` : ""}</div>
        {discovery && (
          <div className="stack" style={{ gap: 8 }}>
            <p><span className={`verdict ${discovery.status === "accepted" ? "accepted" : "rejected"}`}>{discovery.status}</span> {discovery.reason}</p>
            {discovery.law && <><h3>{discovery.law.title}</h3><Law tex={discovery.law.latex} /></>}
            {discovery.parameters && (
              <ul className="small">
                {discovery.parameters.map((p) => <li key={p.name}><Tex tex={p.name} /> = {p.value.toPrecision(7)} ± {p.std.toPrecision(2)}</li>)}
                {discovery.bootstrap_reps ? <li className="muted">intervals from {discovery.bootstrap_reps} bootstrap resamples</li> : null}
              </ul>
            )}
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
