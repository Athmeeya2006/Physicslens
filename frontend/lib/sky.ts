// The Solar System Lab's data: scenarios and simulations from the server (/api/sky/*) when it is online, otherwise the
// precomputed bundles in /sky/*.json, with changes re-run in the browser by a small integrator. The server's result
// is authoritative and replaces a browser preview as soon as it arrives.

import { API_BASE, codeHeaders, json } from "./api";

export interface ScenarioSummary {
  name: string; title: string; description: string; fictional: boolean; target: string; start: string;
  clip_days: number; bodies: string[]; hidden: string[]; mass_scale: Record<string, number>; gr: boolean;
  c_scale: number; baseline: string | null; compare: string;
}

// points: [t (days), x, y, z (AU)], x and y in the target's orbital plane, z along its normal.
export interface SkyBody { name: string; colour: string; hidden: boolean; target: boolean; radius_km?: number; points: number[][] }
export interface Elements { t: number; a: number; e: number; peri_longitude_deg: number; period_days: number }
export interface Detectability { max_px: number; visible: boolean; message: string }

export interface Simulation {
  scenario: ScenarioSummary;
  units: { length: string; time: string };
  bodies: SkyBody[];
  target: string;
  elements: Elements[];
  precession_arcsec_per_century: number | null;
  energy_drift_per_century: number | null;
  warnings: string[];
  detectability: Detectability | null;
  source: string;
  cached?: boolean;
  preview?: boolean;  // computed in this browser, not by the server
  initial?: Initial;
}

export interface Initial {
  central_gm: number; start_jd: number; clip_days: number; frame_step_days: number; basis: number[][];
  gr: boolean; c_scale: number;
  bodies: { name: string; gm: number; massive: boolean; state: number[] }[];
}

export interface Changes {
  mass_scale: Record<string, number>;
  disabled: string[];
  hidden?: string[];
  gr?: boolean;
  c_scale?: number;
  years?: number;
  extra_body?: { gm_suns: number; a_au: number; e: number; i_deg: number; M_deg: number } | null;
  // A push to a body's starting velocity, m/s: [along its motion, outward from the centre, out of its orbit's plane].
  kicks?: Record<string, [number, number, number]>;
}

export interface Discovery {
  source: string; status: string; reason: string;
  law?: { title: string; latex: string };
  parameters?: { name: string; value: number; std: number; ci95: [number, number] }[];
  bootstrap_reps?: number;
  perturbation?: string;
  residual_spectrum?: { significant: boolean; peaks: { component: string; period: number; power: number; false_alarm_probability: number }[];
    period_matches: { body: string; kind: string; period: number; relative_error: number }[] };
  equation_diff?: { before: { title: string; latex: string }; after: { title: string; latex: string }; added_terms: string[];
    summary: string; heldout_before: number; heldout_after: number };
  detectability?: Detectability;
  // Every candidate law the rules judged: its forecast error, BIC, and why it won or lost.
  ranking?: { title: string; heldout_nrmse: number; bic: number; status?: string; why?: string }[];
  n_eff?: number | null;
  noise_px?: number;  // set on a law found in advance and bundled with the site
}

export const changed = (c: Changes) =>
  Object.keys(c.mass_scale).length > 0 || c.disabled.length > 0 || c.hidden !== undefined || c.gr !== undefined
  || c.c_scale !== undefined || c.years !== undefined || !!c.extra_body
  || Object.values(c.kicks ?? {}).some((k) => k.some((v) => v !== 0));

const body = (name: string, c: Changes) => JSON.stringify({
  scenario: name, ...c, extra_body: c.extra_body ?? undefined,
  kicks: c.kicks ? Object.fromEntries(Object.entries(c.kicks).filter(([, k]) => k.some((v) => v !== 0))) : undefined,
});

export async function listScenarios(online: boolean): Promise<ScenarioSummary[]> {
  if (online) {
    try {
      return await fetch(`${API_BASE}/api/sky/scenarios`).then((r) => json<ScenarioSummary[]>(r));
    } catch {
      // fall through to the bundled list: the server went away between calls
    }
  }
  return fetch("/sky/index.json").then((r) => json<ScenarioSummary[]>(r));
}

export const bundled = (name: string): Promise<Simulation> => fetch(`/sky/${name}.json`).then((r) => json<Simulation>(r));

/** The law found in advance for a scenario as written (2 px of tracker noise), for when no server is online. */
export const bundledDiscovery = (name: string): Promise<Discovery> =>
  fetch(`/sky/${name}.discovery.json`).then((r) => json<Discovery>(r));

export const simulateOnServer = (name: string, c: Changes): Promise<Simulation> =>
  fetch(`${API_BASE}/api/sky/simulate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: body(name, c) })
    .then((r) => json<Simulation>(r));

export const discoverOnServer = (name: string, c: Changes, noisePx: number): Promise<Discovery> =>
  fetch(`${API_BASE}/api/sky/discover`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...JSON.parse(body(name, c)), noise_px: noisePx }),
  }).then((r) => json<Discovery>(r));

export const filmOnServer = (name: string, c: Changes): Promise<{ run_id: string }> =>
  fetch(`${API_BASE}/api/sky/render`, { method: "POST", headers: { "Content-Type": "application/json", ...codeHeaders() }, body: body(name, c) })
    .then((r) => json<{ run_id: string }>(r));

// ---------- the in-browser integrator (previews only) ----------

const C_AU_PER_DAY = 173.144632674;
export const SUN_GM = 2.9591220828559115e-4;  // AU^3/day^2, as the server's catalogue
export const HYPOTHETICAL = "hypothetical body";
const M_S_TO_AU_DAY = 86400 / 149_597_870_700;

/** The state with its velocity pushed by dv (m/s) along the motion, outward and along the orbit's normal, exactly as
 * the server does it (sky/scenarios.kicked). */
export function kicked(state: number[], dv: [number, number, number]): number[] {
  const r = state.slice(0, 3), v = state.slice(3, 6);
  const len = (a: number[]) => Math.hypot(a[0], a[1], a[2]);
  const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const along = v.map((x) => x / len(v));
  const h = cross(r, v), normal = h.map((x) => x / len(h));
  let out = cross(along, normal);
  if (out[0] * r[0] + out[1] * r[1] + out[2] * r[2] < 0) out = out.map((x) => -x);
  return [...r, ...v.map((x, i) => x + (dv[0] * along[i] + dv[1] * out[i] + dv[2] * normal[i]) * M_S_TO_AU_DAY)];
}

/** Heliocentric position and velocity from elements in the reference plane (node and perihelion at zero), as the
 * server places a hypothetical body: Kepler's equation by Newton's method. */
export function stateFromElements(a: number, e: number, iDeg: number, mDeg: number, gm: number): number[] {
  const M = (mDeg * Math.PI) / 180, inc = (iDeg * Math.PI) / 180;
  let E = e < 0.8 ? M : Math.PI;
  for (let k = 0; k < 50; k++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
  const n = Math.sqrt(gm / a ** 3), q = Math.sqrt(1 - e * e), den = 1 - e * Math.cos(E);
  const x = a * (Math.cos(E) - e), y = a * q * Math.sin(E);
  const vx = (-a * n * Math.sin(E)) / den, vy = (a * n * q * Math.cos(E)) / den;
  return [x, y * Math.cos(inc), y * Math.sin(inc), vx, vy * Math.cos(inc), vy * Math.sin(inc)];
}

/** Heliocentric Newtonian N-body with the indirect term (and optional 1PN relativity), RK4 with a step sized to the
 * fastest orbit. Good to a fraction of a pixel for the lab's spans; the server's DOP853 run is the real answer. */
export function integrate(init: Initial, c: Changes, target: string, maxPoints = 1500): Simulation["bodies"] {
  const bodies = init.bodies.filter((b) => !c.disabled.includes(b.name)).map((b) => ({
    ...b, gm: b.massive ? b.gm * (c.mass_scale[b.name] ?? 1) : 0,
    state: c.kicks?.[b.name]?.some((v) => v !== 0) ? kicked(b.state, c.kicks[b.name]) : [...b.state],
  }));
  const x = c.extra_body;
  if (x && x.a_au > 0 && x.e >= 0 && x.e < 1 && x.gm_suns >= 0) {
    bodies.push({ name: HYPOTHETICAL, gm: SUN_GM * x.gm_suns, massive: true,
                  state: stateFromElements(x.a_au, x.e, x.i_deg, x.M_deg, SUN_GM) });
  }
  const n = bodies.length;
  const gm = bodies.map((b) => b.gm);
  const gr = c.gr ?? init.gr;
  const cScale = c.c_scale ?? init.c_scale;
  const days = (c.years ? c.years * 365.25 : init.clip_days);
  // Step: a fiftieth of the shortest period among the bodies (Kepler's third law from the starting distance).
  const shortest = Math.min(...bodies.map((b) => {
    const r = Math.hypot(b.state[0], b.state[1], b.state[2]);
    return 2 * Math.PI * Math.sqrt(r ** 3 / init.central_gm);
  }));
  const steps = Math.min(200000, Math.max(200, Math.ceil(days / (shortest / 60))));
  const h = days / steps;
  const acc = (y: number[]) => {
    const a = new Array(3 * n).fill(0);
    let ix = 0, iy = 0, iz = 0;  // the indirect term: the bodies' pull on the Sun
    for (let j = 0; j < n; j++) {
      if (!gm[j]) continue;
      const r3 = Math.hypot(y[3 * j], y[3 * j + 1], y[3 * j + 2]) ** 3;
      ix += gm[j] * y[3 * j] / r3; iy += gm[j] * y[3 * j + 1] / r3; iz += gm[j] * y[3 * j + 2] / r3;
    }
    for (let i = 0; i < n; i++) {
      const x = y[3 * i], yy = y[3 * i + 1], z = y[3 * i + 2];
      const r = Math.hypot(x, yy, z);
      let ax = -init.central_gm * x / r ** 3 - ix, ay = -init.central_gm * yy / r ** 3 - iy, az = -init.central_gm * z / r ** 3 - iz;
      for (let j = 0; j < n; j++) {
        if (j === i || !gm[j]) continue;
        const dx = y[3 * j] - x, dy = y[3 * j + 1] - yy, dz = y[3 * j + 2] - z;
        const d3 = Math.hypot(dx, dy, dz) ** 3;
        ax += gm[j] * dx / d3; ay += gm[j] * dy / d3; az += gm[j] * dz / d3;
      }
      if (gr) {
        const vx = y[3 * n + 3 * i], vy = y[3 * n + 3 * i + 1], vz = y[3 * n + 3 * i + 2];
        const k = init.central_gm * cScale / (C_AU_PER_DAY ** 2 * r ** 3);
        const v2 = vx * vx + vy * vy + vz * vz, rv = x * vx + yy * vy + z * vz;
        ax += k * ((4 * init.central_gm / r - v2) * x + 4 * rv * vx);
        ay += k * ((4 * init.central_gm / r - v2) * yy + 4 * rv * vy);
        az += k * ((4 * init.central_gm / r - v2) * z + 4 * rv * vz);
      }
      a[3 * i] = ax; a[3 * i + 1] = ay; a[3 * i + 2] = az;
    }
    return a;
  };
  const deriv = (y: number[]) => [...y.slice(3 * n), ...acc(y)];
  let y = [...bodies.flatMap((b) => b.state.slice(0, 3)), ...bodies.flatMap((b) => b.state.slice(3, 6))];
  const every = Math.max(1, Math.floor(steps / maxPoints));
  const [e1, e2, e3] = init.basis;
  const out = bodies.map(() => [] as number[][]);
  const dot = (p: number[], e: number[]) => p[0] * e[0] + p[1] * e[1] + p[2] * e[2];
  const record = (t: number) => bodies.forEach((_, i) => {
    const p = [y[3 * i], y[3 * i + 1], y[3 * i + 2]];
    out[i].push([t, dot(p, e1), dot(p, e2), dot(p, e3)]);
  });
  record(0);
  for (let s = 1; s <= steps; s++) {
    const k1 = deriv(y);
    const k2 = deriv(y.map((v, i) => v + 0.5 * h * k1[i]));
    const k3 = deriv(y.map((v, i) => v + 0.5 * h * k2[i]));
    const k4 = deriv(y.map((v, i) => v + h * k3[i]));
    y = y.map((v, i) => v + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    if (s % every === 0) record(s * h);
  }
  return bodies.map((b, i) => ({ name: b.name, colour: "", hidden: (c.hidden ?? []).includes(b.name), target: b.name === target, points: out[i] }));
}

// ---------- the lab's state in the URL ----------

export function encodeState(name: string, c: Changes): string {
  const raw = JSON.stringify({ s: name, c });
  return btoa(unescape(encodeURIComponent(raw))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function decodeState(token: string): { s: string; c: Changes } | null {
  try {
    const b64 = token.replace(/-/g, "+").replace(/_/g, "/");
    const parsed = JSON.parse(decodeURIComponent(escape(atob(b64)))) as { s: string; c: Changes };
    return typeof parsed.s === "string" && parsed.c && typeof parsed.c === "object" ? parsed : null;
  } catch {
    return null;  // a hand-edited or truncated link: start from the plain scenario instead
  }
}
