import { useEffect, useState } from "react";

// Empty NEXT_PUBLIC_API_BASE builds the static site: the gallery is static files, and the analysis server is
// found at runtime from /backend.json (its address changes whenever the GPU server restarts its tunnel).
// Unset means local development against localhost.
const BUILD_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const STATIC_SITE = BUILD_BASE === "";
// Live bindings: importers see the values once connectBackend() has found a server.
export let API_BASE = BUILD_BASE;
export let HAS_BACKEND = !STATIC_SITE;

let connecting: Promise<boolean> | null = null;
// When the next judging session starts (ISO time in UTC), if one is scheduled: shown while no server is running.
export let NEXT_SESSION_UTC: string | null = null;

// The server found on an earlier page of this visit, reused for a few minutes so moving between pages asks nothing
// of GitHub (its API allows 60 unauthenticated requests an hour per visitor).
const REMEMBER_KEY = "physicslens-server";
const REMEMBER_MS = 10 * 60 * 1000;

function remembered(): string {
  try {
    const r = JSON.parse(sessionStorage.getItem(REMEMBER_KEY) ?? "null") as { base?: string; at?: number } | null;
    return r?.base && r.at && Date.now() - r.at < REMEMBER_MS ? r.base : "";
  } catch {
    return "";
  }
}

function remember(base: string): void {
  try {
    sessionStorage.setItem(REMEMBER_KEY, JSON.stringify({ base, at: Date.now() }));
  } catch {
    /* storage blocked: the next page simply looks again */
  }
}

async function answers(base: string): Promise<boolean> {
  // Only switch the analysis features on if the server really answers; a stale tunnel means gallery only.
  return fetch(`${base}/api/health`, { signal: AbortSignal.timeout(6000) }).then((r) => r.ok).catch(() => false);
}

export function connectBackend(): Promise<boolean> {
  if (!STATIC_SITE) return Promise.resolve(true);
  connecting ??= (async () => {
    try {
      const cfg = await fetch("/backend.json", { cache: "no-store" })
        .then((r) => r.json() as Promise<{ api_base?: string; next_session_utc?: string; pointer_gist?: string }>);
      NEXT_SESSION_UTC = cfg.next_session_utc && !Number.isNaN(Date.parse(cfg.next_session_utc)) ? cfg.next_session_utc : null;
      // In order: the server this visit already found, the fixed address (set by hand), and only if neither answers
      // the address a self-restarting server (a scheduled Kaggle session) published to a public gist.
      const tried = new Set<string>();
      const attempt = async (raw: string): Promise<boolean> => {
        const base = raw.replace(/\/$/, "");
        if (!base || tried.has(base)) return false;
        tried.add(base);
        if (!(await answers(base))) return false;
        API_BASE = base;
        HAS_BACKEND = true;
        remember(base);
        return true;
      };
      if (await attempt(remembered())) return true;
      if (await attempt(cfg.api_base ?? "")) return true;
      if (cfg.pointer_gist) return attempt(await gistAddress(cfg.pointer_gist).catch(() => ""));
      return false;
    } catch {
      return false;
    }
  })();
  return connecting;
}

const POINTER_FILE = "physicslens-backend.json";

async function gistAddress(gistId: string): Promise<string> {
  // The API answers with the current content; when it refuses (its hourly limit per visitor), the raw file, which
  // has no such limit but may be up to five minutes old.
  const id = encodeURIComponent(gistId);
  let content = await fetch(`https://api.github.com/gists/${id}`, { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .then((g: { files?: Record<string, { content?: string }> } | null) => g?.files?.[POINTER_FILE]?.content)
    .catch(() => undefined);
  content ??= await fetch(`https://gist.githubusercontent.com/raw/${id}/${POINTER_FILE}`, { cache: "no-store" })
    .then((r) => (r.ok ? r.text() : undefined)).catch(() => undefined);
  if (!content) return "";
  const p = JSON.parse(content) as { api_base?: string; until_utc?: string };
  if (p.until_utc && Date.parse(p.until_utc) < Date.now()) return ""; // that session has ended
  return p.api_base ?? "";
}

/** What this visitor may do on the analysis server: null while asking, undefined when there is no server. */
export function useHealth(): Health | null | undefined {
  const [health, setHealth] = useState<Health | null | undefined>(null);
  useEffect(() => {
    let live = true;
    getHealth().then((h) => live && setHealth(h)).catch(() => live && setHealth(undefined));
    return () => {
      live = false;
    };
  }, []);
  return health;
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
  latex?: string;
  param_meta?: { name: string; latex: string; unit: string }[];
}

export interface TrailStep {
  step: "skeptic" | "scaling" | "locate" | "plan" | "study_turn" | "joint_fit" | "scene" | "tracking" | "hypotheses" | "verification" | "design" | "plausibility" | "track_check" | "collision" | "regime" | "chaos" | "audio" | "agent" | "setup_guide" | "controller" | "control_test";
  provider: string | null;
  model: string;
  text: string;
  status?: string;
  action?: string;
  reasoning_summary?: string;
  reasoning?: string;
  fallbacks?: string[];
  expectation?: string | null;
  label?: string;
  // latex: a law the agent wrote, typeset (it has no familiar name to go by)
  items?: { model_id: string; title: string; why: string; latex?: string | null; status?: "accepted" | "tied" | "rejected" }[];
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
  sindy_check?: { shown: boolean; reason: string; heldout_nrmse: number | null };
  full_clip_verdict?: { status: string; reason: string; winner: string | null } | null;
  units?: { length: string; time: string } | null;
  hypothesis_votes?: { samples: number; winner: string; proposed_by: number; families: string[] } | null;
  skeptic?: { tested: number; survived: boolean; accepted_before: string; concerns: string[];
             verdict_after?: { status: string; winner: string | null } } | { skipped: string } | null;
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
    mujoco_contact?: { dampratio: number; timeconst: number; e_mujoco: number; radius_m: number };
    impacts: { t: number; e: number; e_std: number; tangential: number }[];
  };
  audio?: AudioResult;
  audit?: PhysicsAudit;
  depth3d?: Depth3d;
}

export interface AudioValue { value: number; std: number }

export interface AudioComparison {
  quantity: string; label: string; unit: string; camera: AudioValue; microphone: AudioValue;
  z: number; agree: boolean; fused: AudioValue | null;
}

export interface AudioResult {
  available: boolean;
  note?: string;
  duration_s?: number;
  onsets?: number[];
  onset_count?: number;
  onset_sigma_s?: number;
  restitution?: { e_mean: number; e_std: number; impacts: { t: number; e: number; e_std: number }[] } | null;
  tone?: { freq_hz: number; halfwidth_hz: number; snr_db: number; rotor_rate_hz?: number } | null;
  spin_down?: {
    f_start_hz: number; f_end_hz: number; t_start: number; t_end: number; status: string; reason: string;
    law: { model_id: string; title: string; latex: string } | null; blade_count: number | null;
    acoustic_params: { name: string; value: number; std: number; unit: string; meaning: string }[];
    rotor_params: { name: string; latex: string; value: number; std: number; unit: string }[];
    camera_law?: string; laws_agree?: boolean;
  } | null;
  comparisons?: AudioComparison[];
  notes?: string[];
}

// The physics audit: each check measured on the track, with its size and its distance from what physics allows.
export interface AuditCheck {
  check: "gravity" | "energy" | "period" | "law" | "kinematics";
  title: string;
  status: "pass" | "violation" | "skipped";
  text: string;
  value: number | null;
  std: number | null;
  expected: number | null;
  unit: string;
  z: number | null;
}

export interface PhysicsAudit {
  verdict: "consistent" | "violates_physics" | "cant_judge";
  headline: string;
  reasons: string[];
  checks: AuditCheck[];
  violations: AuditCheck[];
  thresholds?: Record<string, number>;
}


// 3D from one phone (balls only): depth from the ball's apparent size. "skipped" carries only the reason.
export interface Depth3d {
  status: "accepted" | "ambiguous" | "no_fit" | "skipped";
  reason: string;
  focal_px?: number;
  diameter_m?: number;
  diameter_noise_px?: number;
  frames_used?: number;
  frames_interpolated?: number;
  frames_rejected?: { blurred: number; occluded: number };
  depth_range_m?: [number, number];
  depth_travel_vs_error?: number;
  // [frame, t_s, x_m, y_m, z_m, z_err_m, xy_err_m, interpolated]
  track?: [number, number, number, number, number, number, number, boolean][];
  fit_track?: [number, number, number][];
  equation?: { model_id: string; title: string; latex: string };
  parameters?: Parameter[];
  g?: Parameter;
  ranking?: { model_id: string; title: string; params: Record<string, number>; bic: number; heldout_nrmse: number;
              status: string; why: string }[];
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
  reasoning?: { provider: string; model: string } | null;
}

export interface GalleryItem {
  id: string;
  kind?: "study";
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
export const exportUrl = (id: string, kind: "json" | "py" | "mjcf" | "report" | "contact") =>
  !isGallery(id) ? `${API_BASE}/api/runs/${id}/export/${kind}`
    : kind === "report" ? `${staticBase(id)}/physicslens-${id}-report.md`
    : kind === "contact" ? `${staticBase(id)}/physicslens-${id}-contact.xml` : `${staticBase(id)}/physicslens-${id}.${EXPORT_EXT[kind]}`;

export function askWhatIf(id: string, question: string) {
  return fetch(`${API_BASE}/api/runs/${id}/whatif`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...codeHeaders() },
    body: JSON.stringify({ question }),
  }).then((r) => json<import("@/components/WhatIf").WhatIfAnswer>(r));
}

// Computed on request from the stored result (local math and MuJoCo, no model calls); the static site ships it.
export const getSimToReal = (id: string): Promise<import("@/components/SimToReal").SimToRealData> =>
  fetch(STATIC_SITE && isGallery(id) ? `${staticBase(id)}/sim2real.json` : `${API_BASE}/api/runs/${id}/sim2real`)
    .then((r) => json<import("@/components/SimToReal").SimToRealData>(r));

export interface SetupGuideView {
  status: "picture" | "text_only";
  instruction: string;
  caption: string;
  reason: string | null;
  model: string;
  planned_by?: { provider: string; model: string } | null;
  image_url?: string;
  frame_url?: string;
}

// Drawing a guide is a paid call, so it only ever starts from the "Show me" button; a stored one is free.
export function requestSetupGuide(id: string, stepRef: number): Promise<SetupGuideView> {
  if (isGallery(id)) {
    return fetch(`${staticBase(id)}/setup_guide.json`).then((r) => json<SetupGuideView>(r)).then((g) => ({
      ...g, image_url: `${staticBase(id)}/setup_guide.jpg`, frame_url: `${staticBase(id)}/setup_guide_frame.jpg`,
    }));
  }
  return fetch(`${API_BASE}/api/runs/${id}/setup-guide`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...codeHeaders() },
    body: JSON.stringify({ step_ref: stepRef }),
  }).then((r) => json<SetupGuideView>(r)).then((g) => ({
    ...g,
    image_url: g.image_url && `${API_BASE}${g.image_url}`,
    frame_url: g.frame_url && `${API_BASE}${g.frame_url}`,
  }));
}

// Muse Wonder writes a controller in the background; MuJoCo simulation decides whether it is used.
export function startControl(id: string, task: import("@/components/ControlPanel").ControlTask) {
  return fetch(`${API_BASE}/api/runs/${id}/control`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...codeHeaders() },
    body: JSON.stringify(task),
  }).then((r) => json<import("@/components/ControlPanel").ControlView>(r));
}
export const getControl = (id: string, taskId: string) =>
  fetch(`${API_BASE}/api/runs/${id}/control/${taskId}`, { headers: codeHeaders() })
    .then((r) => json<import("@/components/ControlPanel").ControlView>(r));

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
  setting?: number | null;
  depth3d?: boolean;
  focalPx?: number | null;
}

// path: "/api/audit" runs the same analysis as an audit (every result carries its physics audit).
export function startRun(file: File, inputs: RunInputs, turnstileToken: string | null, path = "/api/runs"): Promise<RunView> {
  const { pendulumLengthM, objectSizeM, slowMotion, objectHint } = inputs;
  const form = new FormData();
  form.append("video", file);
  if (objectHint?.trim()) form.append("object_hint", objectHint.trim().slice(0, 60));
  if (pendulumLengthM) form.append("pendulum_length_m", String(pendulumLengthM));
  if (objectSizeM) form.append("object_size_m", String(objectSizeM));
  if (slowMotion && slowMotion > 1) form.append("slow_motion", String(slowMotion));
  if (turnstileToken) form.append("turnstile_token", turnstileToken);
  if (inputs.depth3d) form.append("depth_3d", "true");
  if (inputs.focalPx) form.append("focal_px", String(inputs.focalPx));
  return fetch(`${API_BASE}${path}`, { method: "POST", body: form, headers: codeHeaders() }).then((r) => json<RunView>(r));
}

// The open-weight reasoning model: Muse Glimmer, self-hosted or on a hosted provider serving the same weights.
export function isOpenWeight(model: string | undefined): boolean {
  return !!model && /muse-glimmer|muse-spark-open/i.test(model);
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
export interface PlannedSweep {
  variable: string; unit: string; response: string; values: number[]; expected_law: string; why: string;
}
export interface Sweep {
  variable: string; unit: string; response: string; expected_law: string; why: string;
  status: "accepted" | "ambiguous" | "no_fit" | "too_few"; reason: string;
  points: { label: string; run_id: string; x: number; y: number; std: number }[];
  winner: { law: string; title: string; latex: string; params: number[]; loo_z: number[]; matches_expectation: boolean } | null;
  next: { x: number; y?: number; std?: number; low?: number; high?: number; separates?: string[] } | null;
  checks: { x: number; label: string; predicted: number; predicted_std: number; measured: number; measured_std: number;
            z: number; within_95: boolean }[];
}
export interface Study {
  id: string; goal: string; status: "recording" | "finished";
  plan: { deliverable: "simulation_model" | "health_check" | "parameter_sheet"; quantities: string[];
          recordings: PlannedRecording[]; done_when: string; reasoning_summary: string; sweep?: PlannedSweep | null };
  recordings: { run_id: string; label: string; x?: number }[];
  sweep?: Sweep | null;
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
  if (inputs.setting) form.append("setting", String(inputs.setting));
  return fetch(`${API_BASE}/api/studies/${id}/runs`, { method: "POST", body: form, headers: codeHeaders() })
    .then((r) => json<{ study: Study; run: RunView }>(r));
}
export const studyFileUrl = (id: string, kind: "report" | "mjcf") =>
  id.startsWith("gallery-") ? `/gallery/${id}/physicslens-${id}.${kind === "report" ? "md" : "xml"}` : `${API_BASE}/api/studies/${id}/${kind}`;

// A value rounded to the precision its uncertainty supports: two significant figures of the uncertainty.
const SUPERSCRIPT_DIGITS: Record<string, string> = {
  "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
};

/** Value and uncertainty as text with matching decimals; tiny or huge values share one power of ten (shown after
 *  the uncertainty, so "2.9598 ± 0.0010 × 10⁻⁴" reads as both numbers times 10⁻⁴). */
export function withUncertainty(value: number, std: number): [string, string] {
  const a = Math.abs(value);
  const e = a !== 0 && Number.isFinite(a) && (a < 1e-3 || a >= 1e5) ? Math.floor(Math.log10(a)) : 0;
  const k = 10 ** e;
  const tail = e ? ` × 10${[...String(e)].map((c) => SUPERSCRIPT_DIGITS[c] ?? c).join("")}` : "";
  if (!(std > 0) || !Number.isFinite(std)) return [e ? `${(value / k).toPrecision(4)}${tail}` : value.toPrecision(4), ""];
  const d = Math.min(8, Math.max(0, 1 - Math.floor(Math.log10(std / k))));
  return [(value / k).toFixed(d), `${(std / k).toFixed(d)}${tail}`];
}
