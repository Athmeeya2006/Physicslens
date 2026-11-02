// The Solar System Lab's data: scenarios and simulations from the server (/api/sky/*) when it is online, otherwise the
// precomputed bundles in /sky/*.json, with changes re-run in the browser by a small integrator. The server's result
// is authoritative and replaces a browser preview as soon as it arrives.

import { API_BASE, codeHeaders, json } from "./api";

export type { Changes, Detectability, Discovery, Elements, Initial, ScenarioSummary, Simulation, SkyBody } from "./skyTypes";
import type { Changes, Discovery, ScenarioSummary, Simulation } from "./skyTypes";

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

// ---------- the in-browser integrator (lib/skyIntegrate.ts, also run in a worker) ----------

export { HYPOTHETICAL, integrate, kicked, stateFromElements, SUN_GM } from "./skyIntegrate";

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
