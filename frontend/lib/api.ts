export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type TrackPoint = [frame: number, x: number, y: number];

export interface Parameter {
  name: string;
  latex: string;
  unit: string;
  value: number;
  std: number;
  ci95: [number, number];
  derived?: string;
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
  step: "scene" | "tracking" | "hypotheses" | "verification" | "collision" | "regime" | "agent";
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

export const getHealth = () => fetch(`${API_BASE}/api/health`).then((r) => json<Health>(r));
export const getBudget = () => fetch(`${API_BASE}/api/budget`).then((r) => json<Budget>(r));
export const getRun = (id: string) => fetch(`${API_BASE}/api/runs/${id}`).then((r) => json<RunView>(r));
export const clipUrl = (id: string) => `${API_BASE}/api/runs/${id}/clip.mp4`;
export const exportUrl = (id: string, kind: "json" | "py" | "mjcf") => `${API_BASE}/api/runs/${id}/export/${kind}`;

export function askWhatIf(id: string, question: string) {
  const code = typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("code");
  return fetch(`${API_BASE}/api/runs/${id}/whatif`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(code ? { "X-Access-Code": code } : {}) },
    body: JSON.stringify({ question }),
  }).then((r) => json<import("@/components/WhatIf").WhatIfAnswer>(r));
}

export function makeShareVideo(id: string): Promise<string> {
  return fetch(`${API_BASE}/api/runs/${id}/share`, { method: "POST" })
    .then((r) => json<{ url: string }>(r))
    .then((r) => `${API_BASE}${r.url}`);
}

export function startRun(file: File, pendulumLengthM: number | null, accessCode: string | null): Promise<RunView> {
  const form = new FormData();
  form.append("video", file);
  if (pendulumLengthM) form.append("pendulum_length_m", String(pendulumLengthM));
  const headers: HeadersInit = accessCode ? { "X-Access-Code": accessCode } : {};
  return fetch(`${API_BASE}/api/runs`, { method: "POST", body: form, headers }).then((r) => json<RunView>(r));
}
