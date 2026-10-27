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
  ghosts?: { model_id: string; points: TrackPoint[] }[];
  sindy?: Record<string, string> | null;
  equation?: { model_id: string; title: string; latex: string };
  parameters?: Parameter[];
  llm_calls: { step: string; provider: string; model: string; cached: boolean }[];
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

export function startRun(file: File, pendulumLengthM: number | null, accessCode: string | null): Promise<RunView> {
  const form = new FormData();
  form.append("video", file);
  if (pendulumLengthM) form.append("pendulum_length_m", String(pendulumLengthM));
  const headers: HeadersInit = accessCode ? { "X-Access-Code": accessCode } : {};
  return fetch(`${API_BASE}/api/runs`, { method: "POST", body: form, headers }).then((r) => json<RunView>(r));
}
