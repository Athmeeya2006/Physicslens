import { useEffect, useState } from "react";

// Empty NEXT_PUBLIC_API_BASE builds the static site: the gallery is static files, and the analysis server is
// found at runtime from /backend.json (its address changes whenever the GPU notebook restarts its tunnel).
// Unset means local development against localhost.
const BUILD_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const STATIC_SITE = BUILD_BASE === "";
// Live bindings: importers see the values once connectBackend() has found a server.
export let API_BASE = BUILD_BASE;
export let HAS_BACKEND = !STATIC_SITE;

let connecting: Promise<boolean> | null = null;

export function connectBackend(): Promise<boolean> {
  if (!STATIC_SITE) return Promise.resolve(true);
  connecting ??= (async () => {
    try {
      const cfg = await fetch("/backend.json", { cache: "no-store" }).then((r) => r.json() as Promise<{ api_base?: string }>);
      const base = (cfg.api_base ?? "").replace(/\/$/, "");
      if (!base) return false;
      // Only switch the analysis features on if the server really answers; a stale tunnel means gallery only.
      const ok = await fetch(`${base}/api/health`, { signal: AbortSignal.timeout(6000) }).then((r) => r.ok);
      if (ok) {
        API_BASE = base;
        HAS_BACKEND = true;
      }
      return ok;
    } catch {
      return false;
    }
  })();
  return connecting;
}

/** null while looking for the analysis server, then whether it is there. */
export function useBackend(): boolean | null {
  const [online, setOnline] = useState<boolean | null>(STATIC_SITE ? null : true);
  useEffect(() => {
    let live = true;
    connectBackend().then((ok) => live && setOnline(ok));
    return () => {
      live = false;
    };
  }, []);
  return online;
}

// Gallery entries are precomputed real analyses, bundled as static files; they never need the backend.
export const isGallery = (id: string) => id.startsWith("gallery-");
const staticBase = (id: string) => `/gallery/${id}`;
const EXPORT_EXT = { json: "json", py: "py", mjcf: "xml" } as const;

export type TrackPoint = [frame: number, x: number, y: number];

export interface Parameter {
  name: string;
  latex: string;
  unit: string;
  value: number;
  std: number;
  ci95: [number, number];
  derived?: string;
  uncalibrated?: boolean;
}

export interface RankedFit {
  model_id: string;
  title: string;
  params: Record<string, number>;
  bic: number;
  heldout_nrmse: number;
  rationale: string | null;
  param_meta?: { name: string; latex: string; unit: string }[];
}

export interface TrailStep {
  step: "plan" | "study_turn" | "scene" | "tracking" | "hypotheses" | "verification" | "design" | "plausibility" | "track_check" | "collision" | "regime" | "chaos" | "agent";
  provider: string | null;
  model: string;
  text: string;
  status?: string;
  action?: string;
  reasoning_summary?: string;
  items?: { model_id: string; title: string; why: string; status?: "accepted" | "tied" | "rejected" }[];
}

export interface AnalysisResult {
  status: "accepted" | "ambiguous" | "no_fit";
  reason: string;
  family: string;
  families_tried: string[];
  scene: { moving_object: string; motion_family: string; reference_kind: string; observations: string };
  clip: { width: number; height: number; fps: number; frames: number; slow_motion?: number };
  calibration: { px_per_m: number; rel_std: number; source: string };
  agent: { action: string; message: string; reasoning_summary: string };
  ranking: RankedFit[];
  real_track_px: TrackPoint[];
  object_size_px?: number;
  ghosts?: { model_id: string; points: TrackPoint[] }[];
  sindy?: Record<string, string> | null;
  equation?: { model_id: string; title: string; latex: string };
  parameters?: Parameter[];
  llm_calls: { step: string; provider: string; model: string; cached: boolean }[];
  trail: TrailStep[];
  regime_change?: { t_estimate: number; t_safe: number; error_ratio: number };
  fit_state?: Record<string, unknown>;
  mujoco?: { rms_vs_fit_m: number | null; notes: string[]; points: TrackPoint[] };
  forecast?: {
    model_id: string; title: string; label: string; unit: string;
    t: number[]; measured: number[]; forecast: number[]; heldout_start: number; heldout_nrmse: number;
  };
  experiment_design?: {
    recommended: { label: string; separation_mm: number; separation_vs_noise: number };
    decisive: boolean; noise_mm: number; options_tested: number; models: string[];
  };
  scale_note?: string;
  scale_basis?: string;
  double?: {
    params: { g: number; mu: number; gamma: number }; L1: number; L2: number; accepted: boolean; reason: string;
    lyapunov_per_s: number; chaotic: boolean; predictability_s: number | null; latex: string;
    outer_track_px: TrackPoint[];
  };
  bounces?: {
    count: number; g: number; g_std: number; e_mean: number; e_std: number; accepted: boolean;
    forecast_nrmse: number; settled_at: number | null; latex: string;
    impacts: { t: number; e: number; e_std: number; tangential: number }[];
  };
}

export interface RunView {
  run_id: string;
  status: "queued" | "running" | "done" | "error";
  progress: string[];
  error: string | null;
  result: AnalysisResult | null;
}

export interface Health {
  status: string;
  app_env: string;
  mock_mode: boolean;
  audience: "dev" | "judge" | "public";
  can_upload: boolean;
  can_go_live: boolean;
  turnstile_site_key: string;
}

export interface GalleryItem {
  id: string;
  title: string;
  description: string;
  credit: string;
  status?: string;
  family?: string;
  equation_title?: string;
  object?: string;
}

const CODE_KEY = "physicslens-access-code";

// The judges' code arrives once as ?code=...; it then lives in this tab's session storage and leaves the URL,
// so it survives navigation without ending up in history, shared links or screenshots.
export function accessCode(): string | null {
  if (typeof window === "undefined") return null;
  const url = new URL(window.location.href);
  const fromUrl = url.searchParams.get("code");
  try {
    if (fromUrl) {
      sessionStorage.setItem(CODE_KEY, fromUrl);
      url.searchParams.delete("code");
      window.history.replaceState(null, "", url);
    }
    return fromUrl ?? sessionStorage.getItem(CODE_KEY);
  } catch {
    return fromUrl; // storage blocked (private mode): the code still works for this page
  }
}

function codeHeaders(): HeadersInit {
  const code = accessCode();
  return code ? { "X-Access-Code": code } : {};
}

export type Budget = Record<string, { spent_usd: number; hard_cap_usd: number; spent_today_usd: number; daily_cap_usd: number }>;

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // body was not JSON; keep the status line
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const getHealth = (): Promise<Health> =>
  connectBackend().then((ok) =>
    ok ? fetch(`${API_BASE}/api/health`, { headers: codeHeaders() }).then((r) => json<Health>(r)) : Promise.reject(new Error("offline")));
const staticGallery = () => fetch("/gallery/index.json").then((r) => json<GalleryItem[]>(r));
// The static site always serves its own gallery copy: it loads fast and never depends on the tunnel.
export const getGallery = (): Promise<GalleryItem[]> =>
  STATIC_SITE ? staticGallery() : fetch(`${API_BASE}/api/gallery`).then((r) => json<GalleryItem[]>(r)).catch(staticGallery);
export const getBudget = () => fetch(`${API_BASE}/api/budget`).then((r) => json<Budget>(r));
export const getRun = (id: string): Promise<RunView> =>
  isGallery(id)
    ? fetch(`${staticBase(id)}/result.json`).then((r) => json<AnalysisResult>(r))
        .then((result) => ({ run_id: id, status: "done" as const, progress: ["Done"], error: null, result }))
    : fetch(`${API_BASE}/api/runs/${id}`).then((r) => json<RunView>(r));
export const clipUrl = (id: string) => (isGallery(id) ? `${staticBase(id)}/clip.mp4` : `${API_BASE}/api/runs/${id}/clip.mp4`);
export const posterUrl = (id: string) => `${staticBase(id)}/poster.jpg`;
export const exportUrl = (id: string, kind: "json" | "py" | "mjcf" | "report") =>
  !isGallery(id) ? `${API_BASE}/api/runs/${id}/export/${kind}`
    : kind === "report" ? `${staticBase(id)}/physicslens-${id}-report.md` : `${staticBase(id)}/physicslens-${id}.${EXPORT_EXT[kind]}`;

export function askWhatIf(id: string, question: string) {
  return fetch(`${API_BASE}/api/runs/${id}/whatif`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...codeHeaders() },
    body: JSON.stringify({ question }),
  }).then((r) => json<import("@/components/WhatIf").WhatIfAnswer>(r));
}

export function makeShareVideo(id: string): Promise<string> {
  if (isGallery(id)) return Promise.resolve(`${staticBase(id)}/share.mp4`);
  return fetch(`${API_BASE}/api/runs/${id}/share`, { method: "POST" })
    .then((r) => json<{ url: string }>(r))
    .then((r) => `${API_BASE}${r.url}`);
}

export interface RunInputs {
  pendulumLengthM: number | null;
  objectSizeM: number | null;
  slowMotion: number | null;
  objectHint: string | null;
}

export function startRun(file: File, inputs: RunInputs, turnstileToken: string | null): Promise<RunView> {
  const { pendulumLengthM, objectSizeM, slowMotion, objectHint } = inputs;
  const form = new FormData();
  form.append("video", file);
  if (objectHint?.trim()) form.append("object_hint", objectHint.trim().slice(0, 60));
  if (pendulumLengthM) form.append("pendulum_length_m", String(pendulumLengthM));
  if (objectSizeM) form.append("object_size_m", String(objectSizeM));
  if (slowMotion && slowMotion > 1) form.append("slow_motion", String(slowMotion));
  if (turnstileToken) form.append("turnstile_token", turnstileToken);
  return fetch(`${API_BASE}/api/runs`, { method: "POST", body: form, headers: codeHeaders() }).then((r) => json<RunView>(r));
}

// The open-weight reasoning model: Muse Glimmer, served from our own GPU.
export function isOpenWeight(model: string | undefined): boolean {
  return !!model && model.startsWith("muse-glimmer");
}

// Units as people write them: m/s² rather than m/s^2, s⁻¹ rather than 1/s.
const SUP: Record<string, string> = { "-": "⁻", "1": "¹", "2": "²", "3": "³", "4": "⁴" };
export function formatUnit(unit: string | undefined | null): string {
  if (!unit) return "";
  return unit
    .replace(/^1\/([a-z]+)$/i, "$1^-1")
    .replace(/\^(-?\d)/g, (_, e: string) => [...e].map((c) => SUP[c] ?? c).join(""));
}

// ---------- goal-driven studies ----------

export interface PlannedRecording { label: string; how_to_film: string }
export interface StudyTurn {
  action: "record_more" | "finish"; next_recording: PlannedRecording | null; message: string;
  maintenance_note: string | null; next_check: string | null; model: string; overruled?: string;
}
export interface Combined {
  name: string; latex: string; unit: string; value: number; std: number; n: number; consistent: boolean;
  p_value: number | null; uncalibrated: boolean;
}
export interface Study {
  id: string; goal: string; status: "recording" | "finished";
  plan: { deliverable: "simulation_model" | "health_check" | "parameter_sheet"; quantities: string[];
          recordings: PlannedRecording[]; done_when: string; reasoning_summary: string };
  recordings: { run_id: string; label: string }[];
  combined: Combined[];
  health: { drift: boolean; summary: string } | null;
  turns: StudyTurn[];
  trail: TrailStep[];
}

export const createStudy = (goal: string) =>
  fetch(`${API_BASE}/api/studies`, { method: "POST", headers: { "Content-Type": "application/json", ...codeHeaders() },
    body: JSON.stringify({ goal }) }).then((r) => json<Study>(r));
export const getStudy = (id: string) =>
  STATIC_SITE && id.startsWith("gallery-") ? fetch(`/gallery/${id}/study.json`).then((r) => json<Study>(r))
    : fetch(`${API_BASE}/api/studies/${id}`).then((r) => json<Study>(r));
export const reviewStudy = (id: string) =>
  fetch(`${API_BASE}/api/studies/${id}/review`, { method: "POST", headers: codeHeaders() }).then((r) => json<Study>(r));
export function addStudyRun(id: string, file: File, label: string, inputs: RunInputs): Promise<{ study: Study; run: RunView }> {
  const form = new FormData();
  form.append("video", file);
  form.append("label", label.slice(0, 60));
  if (inputs.pendulumLengthM) form.append("pendulum_length_m", String(inputs.pendulumLengthM));
  if (inputs.objectSizeM) form.append("object_size_m", String(inputs.objectSizeM));
  if (inputs.slowMotion && inputs.slowMotion > 1) form.append("slow_motion", String(inputs.slowMotion));
  if (inputs.objectHint?.trim()) form.append("object_hint", inputs.objectHint.trim().slice(0, 60));
  return fetch(`${API_BASE}/api/studies/${id}/runs`, { method: "POST", body: form, headers: codeHeaders() })
    .then((r) => json<{ study: Study; run: RunView }>(r));
}
export const studyFileUrl = (id: string, kind: "report" | "mjcf") =>
  id.startsWith("gallery-") ? `/gallery/${id}/physicslens-${id}.${kind === "report" ? "md" : "xml"}` : `${API_BASE}/api/studies/${id}/${kind}`;
