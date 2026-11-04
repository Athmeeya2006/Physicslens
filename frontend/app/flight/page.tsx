"use client";

import { useEffect, useState } from "react";
import { OnThisPage } from "@/components/OnThisPage";
import { ProofNav } from "@/components/ProofNav";

// Flying machines from their own logs: a real quadrotor's thrust and drag laws (scripts/drone_demo.py,
// data/drone/result.json) and Falcon 9's engine performance from launch webcasts (scripts/rocket_demo.py,
// data/rocket/result.json). Each section is shown only when its stored result exists; every number is read from it.

type Axes = Record<"x" | "y" | "z", number>;
interface Drone {
  data: { segments: number; test_segments: number; mass_kg: number };
  chosen: Record<"x" | "y" | "z", string>;
  heldout_rms_n: { physicslens: Axes; textbook: Axes; neurobem_bem_nn: Axes };
  heldout_total_rms_n: { physicslens: number; textbook: number; neurobem_bem_nn: number };
  thrust_coefficient_n_per_rad2_s2: number;
}
interface Rocket {
  summary: { launches_with_first_stage: number; with_accepted_law: number; isp_sl_median_s: number | null; isp_sl_iqr_s: [number, number] | null;
    isp_sl_launches: number; isp_vac_median_s: number | null; bucket_edge_error_median_s: number | null; bucket_launches: number;
    chosen_counts: Record<string, number>; published: { isp_sl_s: number; isp_vac_s: number; source: string } };
}

const LAW: Record<string, string> = { thrust: "thrust k Σω²", thrust_climb: "thrust k Σω² with a loss as it climbs",
  thrust_climb_inflow: "thrust with climb and inflow losses", none: "nothing sideways", rotor_drag: "rotor drag (rotor speed times velocity)",
  rotor_drag_quadratic: "rotor drag plus quadratic drag" };

function useJson<T>(url: string): T | null {
  const [d, setD] = useState<T | null>(null);
  useEffect(() => {
    fetch(url).then((r) => (r.ok ? r.json() : null)).then(setD).catch(() => setD(null));
  }, [url]);
  return d;
}

function DroneSection({ d }: { d: Drone }) {
  const e = d.heldout_rms_n, t = d.heldout_total_rms_n;
  const gap = (t.textbook - t.physicslens) / (t.textbook - t.neurobem_bem_nn);
  const rows: [string, Axes, number, boolean][] = [["Textbook law (what simulators use)", e.textbook, t.textbook, false],
    ["PhysicsLens: the vehicle's own laws (4 numbers)", e.physicslens, t.physicslens, true],
    ["NeuroBEM: blade-element theory plus a neural network", e.neurobem_bem_nn, t.neurobem_bem_nn, false]];
  return (
    <section className="panel stack" aria-labelledby="fl-drone">
      <h2 id="fl-drone">A real quadrotor: {(t.textbook / t.physicslens).toFixed(1)} times closer than the textbook law, from its flight logs</h2>
      <p>Drone simulators push along the body axis with thrust proportional to the squared rotor speeds and nothing sideways. From
        {" "}{d.data.segments} real flights of a {d.data.mass_kg} kg racing quadrotor (NeuroBEM, University of Zurich), the same loop that
        calibrates robot arms chose {LAW[d.chosen.z]} and {LAW[d.chosen.x]}. On the authors&apos; own held-out test flights, four readable
        numbers bring the force error from {t.textbook.toFixed(2)} N to <strong className="num">{t.physicslens.toFixed(2)} N</strong>, closing{" "}
        {(gap * 100).toFixed(0)}% of the gap to a research neural hybrid.</p>
      <div className="table-wrap">
        <table className="cards">
          <thead><tr><th>Model</th><th className="num">Sideways x</th><th className="num">Sideways y</th><th className="num">Thrust</th><th className="num">All</th></tr></thead>
          <tbody>
            {rows.map(([label, a, all, ours]) => (
              <tr key={label}>
                <td data-label="Model">{ours ? <strong>{label}</strong> : label}</td>
                <td data-label="Sideways x" className="num">{a.x.toFixed(2)} N</td>
                <td data-label="Sideways y" className="num">{a.y.toFixed(2)} N</td>
                <td data-label="Thrust" className="num">{a.z.toFixed(2)} N</td>
                <td data-label="All" className="num">{ours ? <strong>{all.toFixed(2)} N</strong> : `${all.toFixed(2)} N`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">Held-out RMS force error per sample. Thrust coefficient {d.thrust_coefficient_n_per_rad2_s2.toExponential(3)} N per (rad/s)².
        Data: NeuroBEM (Bauersfeld et al., RSS 2021), download.ifi.uzh.ch/rpg/NeuroBEM; results only are published here.</p>
    </section>
  );
}

function RocketSection({ r }: { r: Rocket }) {
  const s = r.summary;
  if (s.bucket_edge_error_median_s === null) return null;
  return (
    <section className="panel stack" aria-labelledby="fl-rocket">
      <h2 id="fl-rocket">Falcon 9 from its launch webcasts: the max-Q throttle-down found blind, within {s.bucket_edge_error_median_s.toFixed(1)} s</h2>
      <p>Each launch webcast shows the first stage&apos;s speed and altitude. Candidate laws, from the ideal rocket equation up, were fitted
        to {s.launches_with_first_stage} first stages with no hint of when the engines throttle. {s.with_accepted_law} have a verified law, and on{" "}
        {s.bucket_launches} of them that law contains the throttle-down for maximum aerodynamic pressure, found on its own: its start and end
        land a median <strong className="num">{s.bucket_edge_error_median_s.toFixed(1)} s</strong> from the times the telemetry lists.
        {s.isp_sl_median_s !== null && s.isp_vac_median_s !== null && <> The same fit gives the engines&apos; specific impulse:{" "}
          {s.isp_sl_median_s.toFixed(0)} s at sea level and {s.isp_vac_median_s.toFixed(0)} s in vacuum (median over {s.isp_sl_launches} launches),
          against the {s.published.isp_sl_s.toFixed(0)} s and {s.published.isp_vac_s.toFixed(0)} s SpaceX publishes.</>}</p>
      <p className="small muted">Speed and altitude read from SpaceX webcasts (shahar603/Telemetry-Data, the Unlicense). Each law is fitted on
        the first 75% of the burn and must forecast the rest; {s.with_accepted_law} of {s.launches_with_first_stage} first stages have a
        verified law. Published figure: {s.published.source}.</p>
    </section>
  );
}

export default function FlightPage() {
  const drone = useJson<Drone>("/astro/drone.json");
  const rocket = useJson<Rocket>("/astro/rocket.json");
  const items = [{ id: "fl-drone", label: "Quadrotor" },
    ...(rocket && rocket.summary.bucket_edge_error_median_s !== null ? [{ id: "fl-rocket", label: "Falcon 9" }] : [])];
  return (
    <main id="main">
      <ProofNav caseTitle="flying machines" />
      <header className="page-head">
        <h1>Flying machines, calibrated from their own logs</h1>
        <p className="tagline">
          The loop that corrects robot simulators needs only a record of motion and the inputs that drove it. A drone&apos;s flight
          log and a rocket&apos;s launch broadcast are both: from them come the vehicle&apos;s own laws, each forecasting flight it never
          saw and each checked against an independent answer.
        </p>
      </header>
      {(drone || rocket) && <OnThisPage items={items} />}
      {drone && <DroneSection d={drone} />}
      {rocket && <RocketSection r={rocket} />}
    </main>
  );
}
