// Empty NEXT_PUBLIC_API_BASE builds a backend-free site: the gallery is served as static files and the
// analysis features explain that the backend is offline. Unset means local development.
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
export const HAS_BACKEND = API_BASE !== "";

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
}

export interface TrailStep {
  step: "scene" | "tracking" | "hypotheses" | "verification" | "design" | "collision" | "regime" | "chaos" | "agent";
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
  clip: { width: number; height: number; fps: number; frames: number };
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

export function accessCode(): string | null {
  return typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("code");
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
  HAS_BACKEND
    ? fetch(`${API_BASE}/api/health`, { headers: codeHeaders() }).then((r) => json<Health>(r))
    : Promise.reject(new Error("offline"));
const staticGallery = () => fetch("/gallery/index.json").then((r) => json<GalleryItem[]>(r));
export const getGallery = (): Promise<GalleryItem[]> =>
  HAS_BACKEND ? fetch(`${API_BASE}/api/gallery`).then((r) => json<GalleryItem[]>(r)).catch(staticGallery) : staticGallery();
export const getBudget = () => fetch(`${API_BASE}/api/budget`).then((r) => json<Budget>(r));
export const getRun = (id: string): Promise<RunView> =>
  isGallery(id)
    ? fetch(`${staticBase(id)}/result.json`).then((r) => json<AnalysisResult>(r))
        .then((result) => ({ run_id: id, status: "done" as const, progress: ["Done"], error: null, result }))
    : fetch(`${API_BASE}/api/runs/${id}`).then((r) => json<RunView>(r));
export const clipUrl = (id: string) => (isGallery(id) ? `${staticBase(id)}/clip.mp4` : `${API_BASE}/api/runs/${id}/clip.mp4`);
export const posterUrl = (id: string) => `${staticBase(id)}/poster.jpg`;
export const exportUrl = (id: string, kind: "json" | "py" | "mjcf") =>
  isGallery(id) ? `${staticBase(id)}/physicslens-${id}.${EXPORT_EXT[kind]}` : `${API_BASE}/api/runs/${id}/export/${kind}`;

export function askWhatIf(id: string, question: string) {
  const code = typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("code");
  return fetch(`${API_BASE}/api/runs/${id}/whatif`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(code ? { "X-Access-Code": code } : {}) },
    body: JSON.stringify({ question }),
  }).then((r) => json<import("@/components/WhatIf").WhatIfAnswer>(r));
}

export function makeShareVideo(id: string): Promise<string> {
  if (isGallery(id)) return Promise.resolve(`${staticBase(id)}/share.mp4`);
  return fetch(`${API_BASE}/api/runs/${id}/share`, { method: "POST" })
    .then((r) => json<{ url: string }>(r))
    .then((r) => `${API_BASE}${r.url}`);
}

export function startRun(file: File, pendulumLengthM: number | null, turnstileToken: string | null): Promise<RunView> {
  const form = new FormData();
  form.append("video", file);
  if (pendulumLengthM) form.append("pendulum_length_m", String(pendulumLengthM));
  if (turnstileToken) form.append("turnstile_token", turnstileToken);
  return fetch(`${API_BASE}/api/runs`, { method: "POST", body: form, headers: codeHeaders() }).then((r) => json<RunView>(r));
}

// Open-weight reasoning models: Muse Spark Open on Baseten, or Muse Glimmer on our own GPU.
export function isOpenWeight(model: string | undefined): boolean {
  return !!model && (model.startsWith("meta-models/muse-spark-open") || model.startsWith("muse-glimmer"));
}
