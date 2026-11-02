// The Solar System Lab's data shapes, shared by the page, the API helpers and the worker (which loads nothing else).

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
  // A body on a fixed straight path (a passing star): pulls, is not pulled.
  prescribed?: { name: string; gm: number; at_start: number[]; velocity: number[] }[];
  j2?: { gm: number; j2: number; radius: number; pole: number[] } | null;
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

